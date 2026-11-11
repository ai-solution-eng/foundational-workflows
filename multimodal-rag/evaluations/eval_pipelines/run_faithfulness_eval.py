#!/usr/bin/env python3
"""Run the golden faithfulness fixtures and write a results file (Feature E).

    python evaluations/eval_pipelines/run_faithfulness_eval.py \
        --fixtures evaluations/eval_pipelines/fixtures/faithfulness_golden.jsonl \
        --out /tmp/faithfulness_results.json

The scorer is deterministic and offline (stdlib only: no network, no LLM),
so two runs over the same fixtures produce byte-identical output.  The
committed baseline at evaluations/eval_pipelines/baseline/faithfulness_baseline.json
is exactly such a run; refresh it (and review the diff) whenever the scorer
or the fixtures change.  See documentation/FAITHFULNESS-EVAL.md.

Exit codes: 0 on success, 2 on usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # imported as a package module
    from .faithfulness import METHOD, SCORER_VERSION, score_faithfulness
except ImportError:  # executed directly as a script
    from faithfulness import (  # type: ignore[no-redef]
        METHOD,
        SCORER_VERSION,
        score_faithfulness,
    )

__all__ = ["FixtureError", "build_results", "load_fixtures", "main", "run_eval"]


class FixtureError(ValueError):
    """A fixtures file the runner cannot consume."""


def load_fixtures(path: str | Path) -> list[dict[str, Any]]:
    """Load golden fixtures: one JSON object per line (blank lines skipped).

    Required keys per case: id (unique non-empty str), contexts (list of
    str), answer (str).  Optional: category (non-empty str, default
    "uncategorized"), should_be_faithful (bool, default False), note (str,
    default "").  Raises FixtureError on any violation.
    """
    p = Path(path)
    if not p.is_file():
        raise FixtureError(f"fixtures file not found: {p}")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FixtureError(f"{p}:{lineno}: invalid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise FixtureError(f"{p}:{lineno}: each case must be a JSON object")
        cid = obj.get("id")
        if not isinstance(cid, str) or not cid:
            raise FixtureError(f"{p}:{lineno}: missing string 'id'")
        if cid in seen:
            raise FixtureError(f"{p}:{lineno}: duplicate case id {cid!r}")
        seen.add(cid)
        contexts = obj.get("contexts")
        if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
            raise FixtureError(f"{p}:{lineno}: 'contexts' must be a list of strings")
        answer = obj.get("answer")
        if not isinstance(answer, str):
            raise FixtureError(f"{p}:{lineno}: 'answer' must be a string")
        category = obj.get("category", "uncategorized")
        if not isinstance(category, str) or not category:
            raise FixtureError(f"{p}:{lineno}: 'category' must be a non-empty string")
        faithful = obj.get("should_be_faithful", False)
        if not isinstance(faithful, bool):
            raise FixtureError(f"{p}:{lineno}: 'should_be_faithful' must be a boolean")
        note = obj.get("note", "")
        if not isinstance(note, str):
            raise FixtureError(f"{p}:{lineno}: 'note' must be a string")
        cases.append(
            {
                "id": cid,
                "category": category,
                "should_be_faithful": faithful,
                "contexts": contexts,
                "answer": answer,
                "note": note,
            }
        )
    if not cases:
        raise FixtureError(f"{p}: no fixture cases found")
    return cases


def run_eval(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every fixture (fixture order preserved).  Each case becomes
    {"id", "category", "should_be_faithful", "score", "n_claims",
    "n_supported_claims"}."""
    cases: list[dict[str, Any]] = []
    for fx in fixtures:
        res = score_faithfulness(fx["answer"], fx["contexts"])
        claims = res["claims"]
        cases.append(
            {
                "id": fx["id"],
                "category": fx["category"],
                "should_be_faithful": fx["should_be_faithful"],
                "score": res["score"],
                "n_claims": len(claims),
                "n_supported_claims": sum(1 for c in claims if c["supported"]),
            }
        )
    return cases


def build_results(fixtures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the full results document: per-case rows plus aggregate
    statistics.  The output is deterministic (no timestamps)."""
    cat_scores: dict[str, list[float]] = {}
    for case in cases:
        cat_scores.setdefault(case["category"], []).append(case["score"])
    per_category = {
        cat: {"n_cases": len(scores), "mean_score": sum(scores) / len(scores)}
        for cat, scores in sorted(cat_scores.items())
    }
    scores = [c["score"] for c in cases]
    faithful = [c["score"] for c in cases if c["should_be_faithful"]]
    return {
        "method": METHOD,
        "scorer_version": SCORER_VERSION,
        "n_cases": len(cases),
        "cases": cases,
        "aggregate": {
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "faithful_set_mean": (sum(faithful) / len(faithful)) if faithful else None,
            "min_case_score": min(scores) if scores else None,
            "per_category": per_category,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden faithfulness fixtures (deterministic, offline).")
    parser.add_argument("--fixtures", required=True, help="path to the golden fixtures JSONL file")
    parser.add_argument("--out", required=True, help="path to write the results JSON")
    args = parser.parse_args(argv)
    try:
        fixtures = load_fixtures(args.fixtures)
    except (OSError, FixtureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cases = run_eval(fixtures)
    results = build_results(fixtures, cases)
    out = Path(args.out)
    try:
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
        return 2
    agg = results["aggregate"]
    fsm = agg["faithful_set_mean"]
    fsm_txt = "n/a" if fsm is None else f"{fsm:.4f}"
    print(
        f"faithfulness eval: {results['n_cases']} cases, "
        f"mean_score={agg['mean_score']:.4f}, faithful_set_mean={fsm_txt}, written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
