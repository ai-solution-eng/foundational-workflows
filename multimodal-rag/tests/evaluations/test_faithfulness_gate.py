"""Tests for the CI faithfulness gate (Feature E, 2026-11-10).

These tests make the gate part of the normal pytest suite: they run the
real runner and the real gate script as subprocesses against the committed
golden fixtures and baseline, and assert that the gate FAILS on a
deliberately degraded results file.  Everything is offline and
deterministic; the subprocesses are invoked with sys.executable so they
use the same interpreter as the test session.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "evaluations" / "eval_pipelines" / "run_faithfulness_eval.py"
GATE = REPO_ROOT / "scripts" / "ci_eval_gate.py"
FIXTURES = REPO_ROOT / "evaluations" / "eval_pipelines" / "fixtures" / "faithfulness_golden.jsonl"
BASELINE = REPO_ROOT / "evaluations" / "eval_pipelines" / "baseline" / "faithfulness_baseline.json"


def _run(script: Path, *args: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *[str(a) for a in args]],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )


def _run_runner(out_path: Path) -> subprocess.CompletedProcess:
    return _run(RUNNER, "--fixtures", FIXTURES, "--out", out_path)


def _run_gate(results: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run(GATE, "--baseline", BASELINE, "--results", results, *extra)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _first_faithful_full_case(doc: dict) -> dict:
    for case in doc["cases"]:
        if case["should_be_faithful"] and case["score"] == 1.0:
            return case
    raise AssertionError("no faithful case with score 1.0 in baseline")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def test_runner_succeeds_and_writes_valid_json(tmp_path):
    out = tmp_path / "results.json"
    proc = _run_runner(out)
    assert proc.returncode == 0, proc.stderr
    doc = _load(out)
    assert doc["method"] == "lexical"
    assert doc["n_cases"] >= 12
    assert len(doc["cases"]) == doc["n_cases"]
    assert "faithfulness eval:" in proc.stdout


def test_runner_is_deterministic_byte_for_byte(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run_runner(a).returncode == 0
    assert _run_runner(b).returncode == 0
    assert a.read_bytes() == b.read_bytes()


def test_fresh_results_match_committed_baseline(tmp_path):
    # Drift canary: if the scorer or the fixtures change without a
    # baseline refresh, this fails.
    out = tmp_path / "results.json"
    assert _run_runner(out).returncode == 0
    fresh, base = _load(out), _load(BASELINE)
    assert fresh["n_cases"] == base["n_cases"]
    by_fresh = {c["id"]: c for c in fresh["cases"]}
    for case in base["cases"]:
        assert by_fresh[case["id"]]["score"] == case["score"], case["id"]


# ---------------------------------------------------------------------------
# gate: pass path
# ---------------------------------------------------------------------------


def test_gate_passes_on_fresh_results_against_baseline(tmp_path):
    out = tmp_path / "results.json"
    assert _run_runner(out).returncode == 0
    proc = _run_gate(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "faithfulness gate: PASS" in proc.stdout
    assert "verdict: PASS" in proc.stdout


def test_gate_passes_when_regression_within_tolerance(tmp_path):
    # A per-case drop of exactly --max-regression is allowed (the gate
    # fails only on strictly larger drops).
    doc = _load(BASELINE)
    _first_faithful_full_case(doc)["score"] = 0.98
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "verdict: PASS" in proc.stdout


def test_gate_rejects_unknown_case_ids(tmp_path):
    # F6: the baseline is the reviewed artifact.  A case id the baseline
    # does not know about has no reviewed flag, so it is a hard input
    # error (exit 2), not an informational free pass.
    doc = _load(BASELINE)
    doc["cases"].append(
        {
            "id": "zz-new",
            "category": "hallucinated",
            "should_be_faithful": False,
            "score": 1.0,
            "n_claims": 1,
            "n_supported_claims": 0,
        }
    )
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "zz-new" in proc.stderr
    assert "absent from the baseline" in proc.stderr


def test_gate_warns_on_scorer_version_mismatch_but_still_passes(tmp_path):
    doc = _load(BASELINE)
    doc["scorer_version"] = 99
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning: scorer_version differs" in proc.stdout


# ---------------------------------------------------------------------------
# gate: fail paths
# ---------------------------------------------------------------------------


def test_gate_fails_on_deliberately_degraded_results(tmp_path):
    doc = _load(BASELINE)
    target = _first_faithful_full_case(doc)
    target["score"] = target["score"] - 0.10
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 1
    assert "faithfulness gate: FAIL" in proc.stdout
    assert "regression:" in proc.stdout
    assert target["id"] in proc.stdout
    assert "verdict: FAIL" in proc.stdout


def test_gate_fails_when_faithful_mean_below_min_score(tmp_path):
    doc = _load(BASELINE)
    for case in doc["cases"]:
        if case["should_be_faithful"]:
            case["score"] = case["score"] - 0.20
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 1
    assert "below min-score" in proc.stdout
    assert "verdict: FAIL" in proc.stdout


def test_gate_scores_faithful_set_on_baseline_flags(tmp_path):
    # F6: the results file's should_be_faithful flags are validated but
    # ignored for scoring.  Flipping the ps-* flags to true in the results
    # file must NOT change the faithful set: the gate scores the baseline's
    # 10 faithful cases, mean 0.9167, and still passes.
    doc = _load(BASELINE)
    for case in doc["cases"]:
        if case["id"].startswith("ps-"):
            case["should_be_faithful"] = True
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "verdict: PASS" in proc.stdout
    assert "faithful-set mean: 0.9167" in proc.stdout


def test_gate_fails_on_unsupported_claim_growth(tmp_path):
    # F8: one extra unsupported claim (a new wrong-from-birth shape)
    # exceeds the zero-tolerance default.
    doc = _load(BASELINE)
    for case in doc["cases"]:
        if case["id"] == "ps-001":
            case["n_claims"] += 1
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "unsupported-claim growth" in proc.stdout
    assert "verdict: FAIL" in proc.stdout


def test_gate_allows_unsupported_growth_within_tolerance(tmp_path):
    doc = _load(BASELINE)
    for case in doc["cases"]:
        if case["id"] == "ps-001":
            case["n_claims"] += 1
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out, "--max-unsupported-growth", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "verdict: PASS" in proc.stdout


def test_gate_fails_with_custom_stricter_min_score(tmp_path):
    doc = _load(BASELINE)
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out, "--min-score", "0.999")
    assert proc.returncode == 1
    assert "below min-score" in proc.stdout


def test_gate_fails_when_baseline_case_missing(tmp_path):
    doc = _load(BASELINE)
    removed = doc["cases"].pop()
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 1
    assert "missing from results" in proc.stdout
    assert removed["id"] in proc.stdout
    assert "verdict: FAIL" in proc.stdout


def test_gate_rejects_non_lexical_method(tmp_path):
    doc = _load(BASELINE)
    doc["method"] = "llm-judge"
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 2
    assert "lexical" in proc.stderr


# ---------------------------------------------------------------------------
# gate: input errors (exit 2)
# ---------------------------------------------------------------------------


def test_gate_missing_results_file_is_exit_2(tmp_path):
    proc = _run_gate(tmp_path / "does-not-exist.json")
    assert proc.returncode == 2
    assert "file not found" in proc.stderr


def test_gate_malformed_results_json_is_exit_2(tmp_path):
    out = tmp_path / "broken.json"
    out.write_text("{not json", encoding="utf-8")
    proc = _run_gate(out)
    assert proc.returncode == 2
    assert "invalid JSON" in proc.stderr


def test_gate_results_case_missing_fields_is_exit_2(tmp_path):
    doc = _load(BASELINE)
    del doc["cases"][0]["should_be_faithful"]
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 2
    assert "should_be_faithful" in proc.stderr


def test_gate_results_case_missing_claim_counts_is_exit_2(tmp_path):
    # F8 needs per-case claim counts in both documents.
    doc = _load(BASELINE)
    del doc["cases"][0]["n_claims"]
    out = tmp_path / "results.json"
    _write(out, doc)
    proc = _run_gate(out)
    assert proc.returncode == 2
    assert "n_claims" in proc.stderr


# ---------------------------------------------------------------------------
# runner: fixture validation (exit 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "breakage",
    ["invalid-json", "duplicate-id", "missing-answer"],
    ids=["invalid-json", "duplicate-id", "missing-answer"],
)
def test_runner_rejects_malformed_fixtures(tmp_path, breakage):
    rows = [line for line in FIXTURES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if breakage == "invalid-json":
        rows[0] = "{not valid json"
    elif breakage == "duplicate-id":
        first = json.loads(rows[0])
        rows.append(json.dumps(first))
    elif breakage == "missing-answer":
        obj = json.loads(rows[1])
        del obj["answer"]
        rows[1] = json.dumps(obj)
    fx = tmp_path / "fixtures.jsonl"
    fx.write_text("\n".join(rows) + "\n", encoding="utf-8")
    proc = _run(RUNNER, "--fixtures", fx, "--out", tmp_path / "out.json")
    assert proc.returncode == 2
    assert "error:" in proc.stderr


def test_runner_missing_fixtures_file_is_exit_2(tmp_path):
    proc = _run(RUNNER, "--fixtures", tmp_path / "nope.jsonl", "--out", tmp_path / "out.json")
    assert proc.returncode == 2
    assert "fixtures file not found" in proc.stderr
