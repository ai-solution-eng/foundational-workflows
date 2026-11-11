"""Unit tests for the deterministic lexical faithfulness scorer (Feature E).

The scorer is imported by file path rather than through the
evaluations.eval_pipelines package: that package's __init__ imports the
ViDoRe benchmark stack (torch, PIL, ...), which is not installed in the
server venv and is irrelevant to the offline CI gate.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = REPO_ROOT / "evaluations" / "eval_pipelines"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from faithfulness import (  # noqa: E402
    METHOD,
    SCORER_VERSION,
    extract_claims,
    judge_faithfulness,
    score_faithfulness,
)

# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


def test_extract_claims_single_sentence():
    assert extract_claims("The API supports streaming.") == ["The API supports streaming."]


def test_extract_claims_multi_sentence():
    claims = extract_claims("A is true. B is false! C is maybe?")
    assert claims == ["A is true.", "B is false!", "C is maybe?"]


def test_extract_claims_newlines_are_boundaries():
    claims = extract_claims("first claim\n- second claim\nthird claim")
    assert claims == ["first claim", "- second claim", "third claim"]


def test_extract_claims_empty_and_whitespace():
    assert extract_claims("") == []
    assert extract_claims("   \n  ") == []


def test_extract_claims_abbreviations_mis_split_documented():
    # Documented limitation: "e.g." splits on the abbreviation period.
    claims = extract_claims("Use e.g. this. And that.")
    assert len(claims) == 3


def test_extract_claims_rejects_non_string():
    with pytest.raises(TypeError):
        extract_claims(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# score_faithfulness
# ---------------------------------------------------------------------------


def test_score_fully_supported():
    res = score_faithfulness(
        "Aurora stores all documents in Qdrant vector collections.",
        ["The Aurora platform stores all documents in Qdrant vector collections."],
    )
    assert res["score"] == 1.0
    assert res["method"] == METHOD
    assert res["claims"][0]["supported"] is True
    assert res["claims"][0]["best_context"] == 0


def test_score_best_context_is_highest_overlap():
    res = score_faithfulness(
        "When the quota is exhausted, the platform returns HTTP 429.",
        [
            "Authentication uses short-lived JWT tokens issued by the identity service.",
            "Rate limiting returns HTTP 429 when the quota is exhausted.",
        ],
    )
    assert res["score"] == 1.0
    assert res["claims"][0]["best_context"] == 1


def test_score_partially_supported_fraction():
    res = score_faithfulness(
        "Deployments run in Kubernetes clusters. Each cluster has a dedicated GPU.",
        ["Deployments run in Kubernetes clusters in us-east-1."],
    )
    assert len(res["claims"]) == 2
    assert [c["supported"] for c in res["claims"]] == [True, False]
    assert res["score"] == pytest.approx(0.5)


def test_score_hallucination_is_zero():
    res = score_faithfulness(
        "The system achieves 99.999% uptime across 500 days.",
        ["P95 query latency is 120 milliseconds in production."],
    )
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False
    assert res["claims"][0]["best_context"] is None


def test_score_negation_flip_is_unsupported_despite_full_overlap():
    res = score_faithfulness(
        "The API supports streaming responses.",
        ["The API does not support streaming responses."],
    )
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False
    assert res["claims"][0]["best_context"] is None


def test_score_negation_marker_never_within_window():
    res = score_faithfulness(
        "The cache expires on its own after an hour.",
        ["The cache never expires on its own."],
    )
    assert res["score"] == 0.0


def test_score_negation_window_is_two_tokens():
    # "not" is three tokens before "require" here (index 3 vs 6), outside
    # the 2-token window: the claim is treated as un-negated and scores as
    # supported.  Content coverage is 4/5 = 0.8, so the window is what
    # decides this case.  (The window is a heuristic, not full negation
    # logic -- a farther negation can still be missed.)
    res = score_faithfulness(
        "The system does not in fact require network access.",
        ["The system requires network access."],
    )
    assert res["claims"][0]["supported"] is True


def test_score_prefix_matching_for_inflection():
    res = score_faithfulness("Each cluster has limits.", ["The clusters have limits."])
    # cluster~clusters prefix match plus limits: 2/2 >= 0.7
    assert res["claims"][0]["supported"] is True


def test_score_short_prefix_not_matched():
    # "jobs" vs "job": the shorter token has 3 chars, below the prefix
    # minimum, so only "green" matches -> 1/3 < 0.5 -> unsupported.
    res = score_faithfulness("The jobs are green and fast.", ["The job is green."])
    assert res["score"] == 0.0


def test_score_empty_contexts_never_support():
    res = score_faithfulness("Aurora supports 40 languages and 30 regions.", [])
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False
    assert res["claims"][0]["best_context"] is None


def test_score_empty_context_strings_are_skipped():
    res = score_faithfulness(
        "Aurora stores all documents in Qdrant vector collections.",
        ["", "   ", "The Aurora platform stores all documents in Qdrant vector collections."],
    )
    assert res["score"] == 1.0
    assert res["claims"][0]["best_context"] == 2


def test_score_empty_answer_is_vacuously_supported():
    res = score_faithfulness("   ", ["Some context text here."])
    assert res["score"] == 1.0
    assert res["claims"] == []
    assert "note" in res
    assert res["method"] == METHOD


def test_score_deterministic_across_calls():
    a = score_faithfulness(
        "The worker pool scales between 2 and 8 replicas. The pool runs on AWS in eu-west-1.",
        ["The worker pool scales between 2 and 8 replicas. Autoscaling triggers on CPU utilization."],
    )
    b = score_faithfulness(
        "The worker pool scales between 2 and 8 replicas. The pool runs on AWS in eu-west-1.",
        ["The worker pool scales between 2 and 8 replicas. Autoscaling triggers on CPU utilization."],
    )
    assert a == b
    # deep-equal even after a copy round-trip
    assert a == copy.deepcopy(b)


def test_score_result_schema():
    res = score_faithfulness("X is Y.", ["X is Y. Z."])
    assert set(res.keys()) == {"score", "claims", "method"}
    assert set(res["claims"][0].keys()) == {"claim", "supported", "best_context"}
    assert res["method"] == "lexical"
    assert isinstance(res["score"], float)
    assert 0.0 <= res["score"] <= 1.0


def test_score_rejects_bad_types():
    with pytest.raises(TypeError):
        score_faithfulness(42, ["ctx"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        score_faithfulness("answer", "not a list")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        score_faithfulness("answer", [1, 2])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# plausible-contradiction: context vocabulary reused around a wrong number
# ---------------------------------------------------------------------------


def test_score_threshold_is_seventy_percent_coverage():
    # 3 of 5 content tokens (0.6) no longer passes the 0.7 bar; this exact
    # shape scored as supported under the old 0.5 threshold.
    res = score_faithfulness(
        "The router forwards traffic north to the relay.",
        ["The router forwards traffic to the east zone."],
    )
    assert res["claims"][0]["supported"] is False
    assert res["score"] == 0.0


def test_score_plausible_contradiction_digit():
    # Lead's defect case: identical context vocabulary, wrong distance metric
    # and wrong dimensionality plus an invented store -> must not be faithful.
    res = score_faithfulness(
        "The collection uses euclidean distance and stores 4096-dimensional vectors in redis.",
        ["The qdrant collection uses cosine distance and stores 768-dimensional vectors."],
    )
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False


def test_score_plausible_contradiction_high_coverage_vetoed_by_number():
    # 8 of 9 content tokens match (0.889 coverage); only the dimension
    # number differs.  Overlap alone would call this faithful; the numeric
    # guard must veto it.
    res = score_faithfulness(
        "The qdrant collection uses cosine distance and stores 4096-dimensional vectors.",
        ["The qdrant collection uses cosine distance and stores 768-dimensional vectors."],
    )
    assert res["claims"][0]["supported"] is False
    assert res["claims"][0]["best_context"] is None
    assert res["score"] == 0.0


def test_score_plausible_contradiction_version():
    res = score_faithfulness(
        "Kubernetes version 1.29.0 is the minimum supported release.",
        ["Kubernetes version 1.19.0 is the minimum supported release."],
    )
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False


def test_score_plausible_contradiction_number_word():
    res = score_faithfulness(
        "The job retries five times before giving up.",
        ["The job retries three times before giving up."],
    )
    assert res["score"] == 0.0
    assert res["claims"][0]["supported"] is False


def test_score_matching_number_is_still_supported():
    # Positive control: a number that DOES appear in the best context is no
    # obstacle; the guard only vetoes contradictions.
    res = score_faithfulness(
        "The service processes 1000 requests per minute.",
        ["The service processes 1000 requests per minute, on average across all regions."],
    )
    assert res["score"] == 1.0
    assert res["claims"][0]["best_context"] == 0


def test_scorer_version_is_bumped():
    # v3: 0.7 content-coverage threshold + numeric contradiction guard
    # (v2) + multi-token negation patterns and the vacuous-claim note (v3)
    assert SCORER_VERSION == 3


@pytest.mark.parametrize(
    ("answer", "context"),
    [
        (
            "The exporter delivers webhooks when the queue is full.",
            "The exporter fails to deliver webhooks when the queue is full.",
        ),
        (
            "The exporter delivers webhooks when the queue is full.",
            "The exporter failed to deliver webhooks when the queue is full.",
        ),
        (
            "The cache evicts cold entries under memory pressure.",
            "The cache is unable to evict cold entries under memory pressure.",
        ),
        (
            "The exporter is available during deploys.",
            "The exporter is not available during deploys.",
        ),
        (
            "The proxy forwards multipart uploads.",
            "The proxy cannot forward multipart uploads.",
        ),
        (
            "The batch job starts before its dependencies finish.",
            "The batch job won't start before its dependencies finish.",
        ),
        (
            "The legacy API is available.",
            "The legacy API is no longer available.",
        ),
        (
            "The scheduler evicts the oldest tenant.",
            "The scheduler evicts the smallest tenant instead of the oldest one.",
        ),
        (
            "Requests are routed to the cold pool.",
            "Requests are routed to the warm pool rather than the cold pool.",
        ),
    ],
    ids=[
        "fails-to",
        "failed-to",
        "unable-to",
        "is-not",
        "cannot",
        "wont",
        "no-longer",
        "instead-of",
        "rather-than",
    ],
)
def test_negation_phrase_families(answer, context):
    # F5: full-overlap claims whose polarity differs only through a
    # failure form must be vetoed by the negation guard.
    res = score_faithfulness(answer, [context])
    assert res["claims"][0]["supported"] is False
    assert res["claims"][0]["best_context"] is None
    assert res["score"] == 0.0


def test_vacuous_claim_carries_note():
    # F7: an all-stopword claim is unmeasurable, not supported; it must
    # carry a note so readers can tell the two apart.
    res = score_faithfulness("The of and a.", ["Some grounded text here."])
    claim = res["claims"][0]
    assert claim["note"] == "no scorable claim; vacuously scored"
    assert claim["best_context"] is None
    assert res["score"] == 1.0
    # A content claim has no note.
    res2 = score_faithfulness("The cache expires.", ["The cache expires."])
    assert "note" not in res2["claims"][0]
    # Vacuous with no context at all: unsupported, still noted.
    res3 = score_faithfulness("The of and a.", [])
    assert res3["claims"][0]["supported"] is False
    assert res3["claims"][0]["note"] == "no scorable claim; vacuously scored"


def test_path_import_works_with_torch_absent():
    # The package __init__ imports the ViDoRe stack (torch), which is not
    # installed in the server venv / CI.  The documented path-based import
    # must work in a fresh interpreter and must never pull torch in.
    script = (
        "import importlib.util, json, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('faithfulness', "
        + repr(str(_EVAL_DIR / "faithfulness.py"))
        + "); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "score = mod.score_faithfulness('X is Y.', ['X is Y. Z.'])['score']; "
        "print(json.dumps({'ok': True, 'score': score, 'torch_loaded': 'torch' in sys.modules}))"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True
    assert out["score"] == 1.0
    assert out["torch_loaded"] is False, "path-based import must not load torch"


# ---------------------------------------------------------------------------
# judge_faithfulness (optional LLM path; never used in CI)
# ---------------------------------------------------------------------------


def test_judge_passes_through_score_and_reason():
    def fake_judge(answer, contexts):
        return {"score": 0.75, "reason": "most of it is grounded"}

    res = judge_faithfulness("an answer", ["a context"], fake_judge)
    assert res["score"] == 0.75
    assert res["method"] == "llm-judge"
    assert res["claims"] == []
    assert res["reason"] == "most of it is grounded"


def test_judge_receives_the_case_inputs():
    seen = {}

    def fake_judge(answer, contexts):
        seen["answer"] = answer
        seen["contexts"] = contexts
        return {"score": 1.0}

    judge_faithfulness("Q?", ["C1", "C2"], fake_judge)
    assert seen["answer"] == "Q?"
    assert seen["contexts"] == ["C1", "C2"]


def test_judge_rejects_out_of_range_scores():
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError):
            judge_faithfulness("a", ["c"], lambda ans, ctx: {"score": bad})


def test_judge_rejects_missing_or_non_numeric_score():
    with pytest.raises(TypeError):
        judge_faithfulness("a", ["c"], lambda ans, ctx: {"nope": 1})
    with pytest.raises(TypeError):
        judge_faithfulness("a", ["c"], lambda ans, ctx: {"score": "high"})
    with pytest.raises(TypeError):
        judge_faithfulness("a", ["c"], lambda ans, ctx: {"score": True})


def test_judge_rejects_out_of_range_raises_value_error():
    # wrong type vs wrong value: out-of-range is a value problem
    with pytest.raises(ValueError):
        judge_faithfulness("a", ["c"], lambda ans, ctx: {"score": 1.5})


def test_judge_rejects_non_dict_verdict_and_non_callable():
    with pytest.raises(TypeError):
        judge_faithfulness("a", ["c"], lambda ans, ctx: 0.9)
    with pytest.raises(TypeError):
        judge_faithfulness("a", ["c"], "not callable")  # type: ignore[arg-type]


def test_method_and_version_constants():
    assert METHOD == "lexical"
    assert isinstance(SCORER_VERSION, int) and SCORER_VERSION >= 1
