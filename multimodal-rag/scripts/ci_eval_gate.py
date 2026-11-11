#!/usr/bin/env python3
"""CI gate for the committed faithfulness baseline (Feature E, 2026-11-10).

    python scripts/ci_eval_gate.py \\
        --baseline evaluations/eval_pipelines/baseline/faithfulness_baseline.json \\
        --results  <path-to-results.json> \\
        --min-score 0.80 --max-regression 0.02 --max-unsupported-growth 0

Compares a results file (produced by
evaluations/eval_pipelines/run_faithfulness_eval.py) against the committed
baseline, per case.  The gate is deterministic and offline: it reads two
JSON files and prints a verdict.  No network, no LLM, stdlib only.

The should_be_faithful flags come from the BASELINE, not the results file:
the baseline is the reviewed artifact, so the gate scores the faithful set
defined by the baseline's flags.  The results file's flags are validated
for presence and type but ignored for scoring -- a producer (or a runner
bug) cannot relabel cases to launder a bad result into a pass (F6).  Case
ids present in the results file but absent from the baseline are a hard
input error (exit 2), not a free pass.

Failure conditions (exit 1):
  * any case whose baseline score exceeds its results score by more than
    --max-regression;
  * the mean score over the faithful set (cases with
    should_be_faithful: true in the BASELINE) is below --min-score;
  * a baseline case is missing from the results file (a regression that
    cannot be verified is a gate failure);
  * the results file has no faithful cases at all (nothing to gate);
  * the total number of unsupported claims in the results file
    (sum of n_claims - n_supported_claims) exceeds the baseline's by more
    than --max-unsupported-growth.  The default is 0: any growth fails.
    This catches a NEW case that is wrong from birth, which a per-case
    regression check cannot see (F8).

A results document whose "method" is not "lexical" is rejected (exit 2):
the gate only evaluates the deterministic scorer, an LLM judge never
participates in CI.  A scorer_version mismatch is a warning, not a
failure.  Each case in both documents must carry non-negative integer
n_claims and n_supported_claims (with n_supported_claims <= n_claims).

Exit codes: 0 = pass, 1 = gate failure, 2 = usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["GateInputError", "evaluate", "load_results_doc", "main"]

_EPS = 1e-12


class GateInputError(ValueError):
    """A results/baseline file the gate cannot consume."""


def _as_score_map(doc: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cases = doc.get("cases")
    if not isinstance(cases, list) or not cases:
        raise GateInputError(f"{label}: missing non-empty 'cases' list")
    out: dict[str, dict[str, Any]] = {}
    for c in cases:
        if not isinstance(c, dict):
            raise GateInputError(f"{label}: each case must be a JSON object")
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            raise GateInputError(f"{label}: case missing string 'id'")
        if cid in out:
            raise GateInputError(f"{label}: duplicate case id {cid!r}")
        score = c.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise GateInputError(f"{label}: case {cid!r} missing numeric 'score'")
        faithful = c.get("should_be_faithful")
        if not isinstance(faithful, bool):
            raise GateInputError(f"{label}: case {cid!r} missing boolean 'should_be_faithful'")
        n_claims = c.get("n_claims")
        if isinstance(n_claims, bool) or not isinstance(n_claims, int) or n_claims < 0:
            raise GateInputError(f"{label}: case {cid!r} missing non-negative integer 'n_claims'")
        n_supported = c.get("n_supported_claims")
        if isinstance(n_supported, bool) or not isinstance(n_supported, int) or n_supported < 0:
            raise GateInputError(f"{label}: case {cid!r} missing non-negative integer 'n_supported_claims'")
        if n_supported > n_claims:
            raise GateInputError(f"{label}: case {cid!r} n_supported_claims {n_supported} > n_claims {n_claims}")
        out[cid] = {
            "score": float(score),
            "should_be_faithful": faithful,
            "n_claims": n_claims,
            "n_supported_claims": n_supported,
        }
    return out


def load_results_doc(path: str | Path) -> dict[str, Any]:
    """Read one results/baseline JSON document and validate its shape."""
    p = Path(path)
    if not p.is_file():
        raise GateInputError(f"file not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateInputError(f"{p}: invalid JSON ({exc.msg})") from None
    if not isinstance(doc, dict):
        raise GateInputError(f"{p}: top level must be a JSON object")
    _as_score_map(doc, str(p))
    return doc


def _unsupported_total(cases: dict[str, dict[str, Any]]) -> int:
    return sum(c["n_claims"] - c["n_supported_claims"] for c in cases.values())


def evaluate(
    baseline: dict[str, Any],
    results: dict[str, Any],
    min_score: float,
    max_regression: float,
    max_unsupported_growth: int = 0,
) -> dict[str, Any]:
    """Pure comparison of two result documents.

    The faithful set is defined by the BASELINE's should_be_faithful flags
    (the reviewed artifact); the results file's flags are ignored for
    scoring (F6).  Also fails when the results document has more
    unsupported claims than the baseline by more than
    max_unsupported_growth (F8; default 0 = zero-tolerance growth).

    Returns {"passed": bool, "lines": [str], "missing": [str],
    "new_cases": [str], "regressions": [(id, b, r, delta)],
    "faithful_mean": float | None, "unsupported_baseline": int,
    "unsupported_results": int, "fail_reasons": [str]}.
    """
    base = _as_score_map(baseline, "baseline")
    res = _as_score_map(results, "results")
    missing = sorted(set(base) - set(res))
    new_cases = sorted(set(res) - set(base))
    regressions: list[tuple[str, float, float, float]] = []
    for cid in sorted(set(base) & set(res)):
        b = base[cid]["score"]
        r = res[cid]["score"]
        if b - r > max_regression + _EPS:
            regressions.append((cid, b, r, b - r))
    # F6: faithful set from the BASELINE's flags (cases missing from
    # results are already failed via 'missing').
    faithful_scores = [res[cid]["score"] for cid, c in base.items() if c["should_be_faithful"] and cid in res]
    faithful_mean = (sum(faithful_scores) / len(faithful_scores)) if faithful_scores else None
    below_min = faithful_mean is not None and faithful_mean < min_score
    # F8: aggregate gate on unsupported claims.
    unsupported_baseline = _unsupported_total(base)
    unsupported_results = _unsupported_total(res)
    growth_ok = unsupported_results - unsupported_baseline <= max_unsupported_growth

    fail_reasons: list[str] = []
    if missing:
        fail_reasons.append("missing cases")
    if regressions:
        fail_reasons.append("regressions")
    if below_min:
        fail_reasons.append("below min-score")
    if faithful_mean is None:
        fail_reasons.append("no faithful cases")
    if not growth_ok:
        fail_reasons.append("unsupported-claim growth")
    passed = not fail_reasons

    lines: list[str] = []
    if missing:
        lines.append(f"  missing from results (present in baseline): {', '.join(missing)}")
    for cid, b, r, delta in regressions:
        lines.append(
            f"  regression: {cid} baseline={b:.4f} results={r:.4f} delta={delta:.4f} (max allowed {max_regression:.4f})"
        )
    if below_min:
        lines.append(f"  faithful-set mean {faithful_mean:.4f} below min-score {min_score:.4f}")
    if faithful_mean is None:
        lines.append("  no faithful cases in results file: nothing to gate")
    if not growth_ok:
        lines.append(
            f"  unsupported-claim growth: results {unsupported_results} vs baseline "
            f"{unsupported_baseline} (max growth {max_unsupported_growth})"
        )
    if new_cases:
        lines.append(f"  new cases (informational, not gated): {', '.join(new_cases)}")
    return {
        "passed": passed,
        "lines": lines,
        "missing": missing,
        "new_cases": new_cases,
        "regressions": regressions,
        "faithful_mean": faithful_mean,
        "unsupported_baseline": unsupported_baseline,
        "unsupported_results": unsupported_results,
        "fail_reasons": fail_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a faithfulness results file against the committed baseline.")
    parser.add_argument("--baseline", required=True, help="committed baseline results JSON")
    parser.add_argument("--results", required=True, help="fresh results JSON to gate")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.80,
        help="minimum mean score over the faithful set, as defined by the BASELINE's flags (default 0.80)",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.02,
        help="maximum allowed per-case score drop vs baseline (default 0.02)",
    )
    parser.add_argument(
        "--max-unsupported-growth",
        type=int,
        default=0,
        help="maximum allowed growth in total unsupported claims vs baseline (default 0: any growth fails)",
    )
    args = parser.parse_args(argv)

    try:
        baseline = load_results_doc(args.baseline)
        results = load_results_doc(args.results)
    except (OSError, GateInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if results.get("method") != "lexical":
        print(
            "error: results method is "
            f"{results.get('method')!r}; the gate only evaluates the deterministic "
            "'lexical' scorer (an LLM judge never runs in CI)",
            file=sys.stderr,
        )
        return 2
    # F6: the baseline is the reviewed artifact; case ids it does not know
    # about cannot be gated (they would just be scored on the results file's
    # own, unreviewed flags) -- hard input error, not a free pass.
    base_ids = {c["id"] for c in baseline["cases"]}
    unknown = sorted({c["id"] for c in results["cases"]} - base_ids)
    if unknown:
        print(
            "error: results contain case(s) absent from the baseline: "
            f"{', '.join(unknown)} -- refresh the baseline deliberately; "
            "the gate will not score unreviewed cases",
            file=sys.stderr,
        )
        return 2
    if baseline.get("scorer_version") != results.get("scorer_version"):
        print(
            "warning: scorer_version differs (baseline "
            f"{baseline.get('scorer_version')!r} vs results {results.get('scorer_version')!r}); "
            "refresh the baseline if the scorer rules changed"
        )

    outcome = evaluate(baseline, results, args.min_score, args.max_regression, args.max_unsupported_growth)
    verdict = "PASS" if outcome["passed"] else "FAIL"
    print(f"faithfulness gate: {verdict}")
    for line in outcome["lines"]:
        print(line)
    fm = outcome["faithful_mean"]
    print(f"faithful-set mean: {'n/a' if fm is None else f'{fm:.4f}'} (min-score {args.min_score:.4f})")
    if outcome["passed"]:
        print("verdict: PASS")
    else:
        print(f"verdict: FAIL ({', '.join(outcome['fail_reasons'])})")
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
