"""Offline tests for agentic iterative retrieval (feature C, opt-in).

The whole point of the feature is that the judge is INJECTABLE and FAILS OPEN:
these tests drive the loop with fake search/judge callables only -- no model
endpoint, no network, no cluster.  They pin, in order:

  * the loop (run_agentic_retrieval): sufficient-on-round-1, rewrite-then-new-doc,
    and EVERY fail-open path -- judge raises, returns None, returns garbage,
    times out, proposes the same rewrite forever; plus the round / search / gain
    bounds and the follow-up-search failure that must return what was collected.
  * the trace + config normalisation (clamping, unknown keys, JSON safety).
  * parse_judge_verdict / doc_identity / run_agentic_entries.
  * the rag_system hook: the non-agentic path is unchanged, the trace rides on
    the first result, and a missing/raising judge still returns results.
  * the MCP surface: _acore_retrieval / _arun_retrieval thread agentic=True, the
    media back-fill is skipped in agentic mode (one search, not two), and the
    tool's WIRE schema exposes both new arguments.
  * the REST surface: request validation, the per-dataset helper, the federated
    per-round merge, and both endpoints' body fields.

Run::

    pytest tests/test_agentic_retrieval.py -q
"""

import asyncio
import hashlib
import json
import os
import sys
from typing import Any, ClassVar, cast

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multimodal_rag import api_server, mcp_server
from multimodal_rag.agentic_retrieval import (
    DEFAULT_JUDGE_TIMEOUT_S,
    MAX_AGENTIC_ROUNDS,
    MAX_JUDGE_TIMEOUT_S,
    MIN_JUDGE_TIMEOUT_S,
    STOP_REASONS,
    AgenticConfig,
    AgenticTrace,
    build_chat_judge,
    build_judge_prompt,
    coerce_agentic_config,
    doc_identity,
    parse_judge_verdict,
    query_text,
    run_agentic_entries,
    run_agentic_retrieval,
)
from multimodal_rag.rag_system import EmbeddingModel, MultimodalRAG, MultiModalRAGSystem
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

DIM = 8


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _run(coro):
    """Run one coroutine to completion (house style for the offline suite)."""
    return asyncio.run(coro)


def _doc(text: str, source: str, page: int = 1, chunk: int = 0, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"text": text, "source": source, "page": page, "chunk_index": chunk}
    d.update(extra)
    return d


DOC_A = _doc("alpha doc about widgets", "/docs/a.md")
DOC_B = _doc("beta doc about sprockets", "/docs/b.md")
DOC_C = _doc("gamma doc about flanges", "/docs/c.md")


class _StubEmbedder:
    """Deterministic hash embedder -- no endpoint, no lexical signal."""

    allowable_modalities = ("text", "image", "video")
    model_name = "stub-embedder"
    base_url = "http://stub/v1"
    url_remote = "http://stub"
    api_key = ""
    embedding_dim = None
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None
    code_chunk_size = 1200
    code_chunk_overlap = 0
    code_text_splitter = None

    def __init__(self) -> None:
        self.query_calls: list[Any] = []
        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[method-assign, assignment]
        self.model.aembed_query = self._aembed_query  # type: ignore[method-assign, assignment]
        self.model.embed_query = self._embed_query  # type: ignore[method-assign, assignment]

    def remote(self) -> None:
        """No-op stand-in for the real model's remote-mode switch."""

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:DIM]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        return [self._vec(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    async def _aembed_query(self, query: Any) -> list[float]:
        self.query_calls.append(query)
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)

    def _embed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)


def _make_rag(**kwargs: Any) -> MultimodalRAG:
    return MultimodalRAG(embedder=cast("EmbeddingModel", _StubEmbedder()), preprocess=False, **kwargs)


class _StubChatModel:
    """Minimal chat-model surface the default judge uses."""

    model_name = "stub-chat"
    allowable_modalities = ("text",)
    __test__ = False

    def __init__(self, replies: Any = None) -> None:
        self.replies = replies if isinstance(replies, list) else [replies]
        self.calls: list[Any] = []

    def remote(self) -> None:
        """No-op."""

    async def llm_async_chat_function_call(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

        class _Message:
            content = reply

        class _Choice:
            message = _Message()

        class _Response:
            def __init__(self) -> None:
                self.choices = [_Choice()]

        return _Response()


class _SearchRecorder:
    """search_fn fake: records the query, returns the next canned result set."""

    def __init__(self, rounds: list[list[tuple[Any, float]]]) -> None:
        self.rounds = rounds
        self.queries: list[Any] = []

    async def __call__(self, query: Any) -> list[tuple[Any, float]]:
        self.queries.append(query)
        index = min(len(self.queries) - 1, len(self.rounds) - 1)
        return list(self.rounds[index])


class _JudgeRecorder:
    """judge_fn fake: records calls, returns the next canned verdict."""

    def __init__(self, verdicts: Any) -> None:
        self.verdicts = verdicts if isinstance(verdicts, list) and verdicts else [verdicts]
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, query: str, results: Any) -> Any:
        self.calls.append((query, len(list(results))))
        index = min(len(self.calls) - 1, len(self.verdicts) - 1)
        verdict = self.verdicts[index]
        if isinstance(verdict, BaseException):
            raise verdict
        if callable(verdict):
            return verdict()
        return verdict


# ---------------------------------------------------------------------------
# run_agentic_retrieval -- happy paths
# ---------------------------------------------------------------------------


def test_sufficient_on_the_first_round_stops_without_a_rewrite():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": True})
    results, trace = _run(run_agentic_retrieval("widgets", search, judge))

    assert [d for d, _ in results] == [DOC_A]
    assert trace.rounds == 1
    assert trace.queries == ["widgets"]
    assert trace.stop_reason == "sufficient"
    assert trace.new_doc_ids == []
    assert trace.judge_verdicts == [{"sufficient": True, "missing": "", "rewrite": ""}]
    assert trace.elapsed_s >= 0.0
    assert search.queries == ["widgets"]
    assert len(judge.calls) == 1


def test_rewrite_round_accumulates_the_new_documents():
    search = _SearchRecorder([[(DOC_A, 0.9)], [(DOC_A, 0.4), (DOC_B, 0.8)]])
    judge = _JudgeRecorder(
        [
            {"sufficient": False, "missing": "sprockets", "rewrite": "sprockets"},
            {"sufficient": True},
        ]
    )
    results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=3)))

    assert [d["source"] for d, _ in results] == ["/docs/a.md", "/docs/b.md"]
    assert trace.rounds == 2
    assert trace.queries == ["widgets", "sprockets"]
    assert trace.stop_reason == "sufficient"
    assert trace.new_doc_ids == ["/docs/b.md#p1#c0"]
    assert len(trace.judge_verdicts) == 2
    assert search.queries == ["widgets", "sprockets"]
    # the accumulated list keeps discovery order and the FIRST score seen per doc
    assert [score for _, score in results] == [0.9, 0.8]


def test_rewrite_replaces_only_the_text_of_a_multimodal_query():
    """The loop hands the rewrite STRING to search_fn; the caller folds it back."""
    seen: list[Any] = []

    async def search(query: Any) -> list[tuple[Any, float]]:
        seen.append(query)
        return [(DOC_A, 0.9)]

    judge = _JudgeRecorder([{"sufficient": False, "rewrite": "better words"}, {"sufficient": True}])
    query = {"text": "widgets", "image": "data:image/png;base64,AA=="}
    _results, trace = _run(run_agentic_retrieval(query, search, judge, AgenticConfig(max_rounds=3)))

    assert seen[0] is query, "round 1 receives the caller's query object verbatim"
    assert seen[1] == "better words", "rewrite rounds are issued as a plain string"
    assert trace.queries == ["widgets", "better words"]


def test_media_only_query_runs_one_plain_pass_without_a_judge():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": True})
    results, trace = _run(run_agentic_retrieval({"image": "data:image/png;base64,AA=="}, search, judge))

    assert len(results) == 1
    assert trace.rounds == 1
    assert trace.stop_reason == "no_query_text"
    assert trace.queries == [""]
    assert judge.calls == [], "nothing to judge -- no LLM call may be spent"


def test_sync_search_and_sync_judge_are_supported():
    def search(query: Any) -> list[tuple[Any, float]]:
        return [(DOC_A, 0.5)]

    def judge(query: str, results: Any) -> dict[str, Any]:
        return {"sufficient": True}

    results, trace = _run(run_agentic_retrieval("widgets", search, judge))
    assert len(results) == 1 and trace.stop_reason == "sufficient"


def test_odd_search_return_shapes_are_tolerated():
    for payload in (None, "nonsense", {"not": "results"}, [], [DOC_A], [(DOC_A, "bad-score")]):
        results, trace = _run(
            run_agentic_retrieval("q", lambda q, payload=payload: payload, lambda q, r: {"sufficient": True})
        )
        assert trace.stop_reason == "sufficient"
        if payload == [DOC_A]:
            assert results == [(DOC_A, 0.0)]


# ---------------------------------------------------------------------------
# run_agentic_retrieval -- fail-open paths (the feature's core promise)
# ---------------------------------------------------------------------------


def test_judge_that_raises_fails_open_to_collected_results():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder(RuntimeError("judge endpoint is down"))
    results, trace = _run(run_agentic_retrieval("widgets", search, judge))

    assert [d for d, _ in results] == [DOC_A], "the retrieved docs must survive a judge failure"
    assert trace.stop_reason == "judge_error"
    assert trace.judge_verdicts == [{"error": "judge_error"}]
    assert trace.rounds == 1


def test_judge_that_returns_none_fails_open():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    results, trace = _run(run_agentic_retrieval("widgets", search, _JudgeRecorder(None)))
    assert [d for d, _ in results] == [DOC_A]
    assert trace.stop_reason == "invalid_verdict"


@pytest.mark.parametrize(
    "garbage",
    [
        "not json at all",
        "",
        '{"sufficient": ',
        {"foo": "bar"},
        {},
        [],
        42,
        object(),
    ],
)
def test_judge_returning_garbage_fails_open(garbage):
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    results, trace = _run(run_agentic_retrieval("widgets", search, _JudgeRecorder(garbage)))
    assert [d for d, _ in results] == [DOC_A]
    assert trace.stop_reason == "invalid_verdict"


def test_judge_that_times_out_fails_open_quickly():
    search = _SearchRecorder([[(DOC_A, 0.9)]])

    async def slow_judge(query: str, results: Any) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {"sufficient": True}

    results, trace = _run(run_agentic_retrieval("widgets", search, slow_judge, AgenticConfig(judge_timeout_s=0.02)))
    assert [d for d, _ in results] == [DOC_A]
    assert trace.stop_reason == "judge_timeout"
    assert trace.elapsed_s < 2.0, "the timeout must not wait for the judge"


@pytest.mark.parametrize("zero_timeout", [0, 0.0, -1, None, "abc"])
def test_zero_or_invalid_timeout_still_times_out_instead_of_hanging(zero_timeout, monkeypatch):
    """F11 regression: judge_timeout_s=0 used to mean UNBOUNDED.

    A caller passing 0 (or a config file serialising a missing value as 0) must
    still get a BOUNDED wait: the sleeping judge is abandoned and the collected
    results come back with stop_reason 'judge_timeout'.  The default bound is
    shrunk here so the test proves the mapping (0 -> default, never unbounded)
    without waiting the production 10 s; the judge sleeps far longer either way.
    """
    import multimodal_rag.agentic_retrieval as agentic

    monkeypatch.setattr(agentic, "DEFAULT_JUDGE_TIMEOUT_S", 0.2)
    search = _SearchRecorder([[(DOC_A, 0.9)]])

    async def sleeping_judge(query: str, results: Any) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {"sufficient": True}

    results, trace = _run(
        run_agentic_retrieval(
            "widgets", search, sleeping_judge, AgenticConfig(judge_timeout_s=zero_timeout, max_rounds=3)
        )
    )
    assert [d for d, _ in results] == [DOC_A], "fail open with what was collected"
    assert trace.stop_reason == "judge_timeout"
    assert trace.elapsed_s < 5.0, "the wait must be the bounded default, not the judge's 30s sleep"


def test_repeated_rewrite_stops_the_loop():
    search = _SearchRecorder([[(DOC_A, 0.9)], [(DOC_A, 0.9), (DOC_B, 0.8)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "the same query every time"})
    results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=6)))

    assert len(results) == 2
    assert trace.stop_reason == "repeated_rewrite"
    assert trace.rounds == 2, "one rewrite is issued, the repeat stops it"
    assert trace.queries == ["widgets", "the same query every time"]


def test_judge_rewriting_back_to_the_original_query_stops():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "widgets"})
    _results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=6)))
    assert trace.stop_reason == "repeated_rewrite"
    assert trace.rounds == 1


def test_not_sufficient_without_a_rewrite_stops():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": False, "missing": "everything"})
    _results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=4)))
    assert trace.stop_reason == "no_rewrite"
    assert trace.rounds == 1


def test_follow_up_search_error_returns_the_results_already_collected():
    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, query: Any) -> list[tuple[Any, float]]:
            self.calls += 1
            if self.calls == 1:
                return [(DOC_A, 0.9)]
            raise RuntimeError("collection disappeared mid-loop")

    search = _Flaky()
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "second try"})
    results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=3)))

    assert [d for d, _ in results] == [DOC_A], "fail open: the first round's docs are returned"
    assert trace.stop_reason == "search_error"
    assert trace.rounds == 1


def test_first_search_error_propagates_like_the_non_agentic_path():
    async def boom(query: Any) -> list[tuple[Any, float]]:
        raise RuntimeError("dataset unreadable")

    with pytest.raises(RuntimeError, match="dataset unreadable"):
        _run(run_agentic_retrieval("widgets", boom, _JudgeRecorder({"sufficient": True})))


# ---------------------------------------------------------------------------
# run_agentic_retrieval -- bounds
# ---------------------------------------------------------------------------


def test_max_rounds_bounds_the_search_count():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "next"})
    _results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=2)))

    assert trace.rounds == 2
    assert len(search.queries) == 2
    assert trace.stop_reason == "max_rounds"
    assert len(judge.calls) == 1, "the judge is not asked when the loop cannot act on the answer"


def test_single_round_budget_never_calls_the_judge():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "next"})
    _results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=1)))
    assert trace.stop_reason == "max_rounds"
    assert judge.calls == []


def test_max_total_searches_bounds_the_loop_independently():
    search = _SearchRecorder([[(DOC_A, 0.9)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "next"})
    _results, trace = _run(
        run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=6, max_total_searches=2))
    )
    assert trace.rounds == 2
    assert trace.stop_reason == "max_searches"


def test_no_new_document_stops_the_loop():
    search = _SearchRecorder([[(DOC_A, 0.9)], [(DOC_A, 0.3)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "again"})
    results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=4)))

    assert len(results) == 1
    assert trace.stop_reason == "no_gain"
    assert trace.rounds == 2
    assert trace.new_doc_ids == []
    assert len(judge.calls) == 1


def test_min_round_gain_requires_more_than_the_configured_gain():
    search = _SearchRecorder([[(DOC_A, 0.9)], [(DOC_A, 0.4), (DOC_B, 0.8)]])
    judge = _JudgeRecorder({"sufficient": False, "rewrite": "again"})
    _results, trace = _run(
        run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=4, min_round_gain=1))
    )
    assert trace.stop_reason == "no_gain", "one new doc does not clear min_round_gain=1"
    assert trace.new_doc_ids == ["/docs/b.md#p1#c0"]


def test_loop_always_terminates_with_an_adversarial_judge():
    """A judge that never agrees and always offers a fresh rewrite is bounded."""

    counter = {"n": 0}

    async def judge(query: str, results: Any) -> dict[str, Any]:
        counter["n"] += 1
        return {"sufficient": False, "rewrite": f"rewrite-{counter['n']}"}

    search = _SearchRecorder(
        [
            [(DOC_A, 0.9)],
            [(DOC_A, 0.9), (DOC_B, 0.8)],
            [(DOC_A, 0.9), (DOC_B, 0.8), (DOC_C, 0.7)],
        ]
    )
    results, trace = _run(run_agentic_retrieval("widgets", search, judge, AgenticConfig(max_rounds=3)))
    assert len(results) == 3, "every round contributed a new doc"
    assert trace.rounds == 3
    assert trace.stop_reason == "max_rounds"
    assert trace.stop_reason in STOP_REASONS


# ---------------------------------------------------------------------------
# config / trace / helper units
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_contract():
    cfg = AgenticConfig()
    assert (cfg.max_rounds, cfg.max_total_searches, cfg.min_round_gain, cfg.judge_timeout_s) == (2, 4, 0, 10.0)


def test_config_normalisation_clamps_out_of_range_values():
    cfg = AgenticConfig(max_rounds=0, max_total_searches=10**6, min_round_gain=-5, judge_timeout_s=-1).normalized()
    assert cfg.max_rounds == 1
    assert cfg.max_total_searches == 16
    assert cfg.min_round_gain == 0
    assert cfg.judge_timeout_s == DEFAULT_JUDGE_TIMEOUT_S, "a negative timeout means the default, never 'no timeout'"
    assert AgenticConfig(max_rounds=999).normalized().max_rounds == MAX_AGENTIC_ROUNDS


def test_config_normalisation_tolerates_wrong_types():
    cfg = AgenticConfig(max_rounds="nope", max_total_searches=None, judge_timeout_s=float("nan")).normalized()
    assert cfg.max_rounds == 2
    assert cfg.max_total_searches == 4
    assert cfg.judge_timeout_s == 10.0


@pytest.mark.parametrize("raw", [0, 0.0, -1, -0.5, None, "abc", "", float("nan"), float("inf"), float("-inf")])
def test_judge_timeout_is_always_bounded(raw):
    """F11: a non-positive/missing/invalid timeout must never mean 'no timeout'."""
    cfg = AgenticConfig(judge_timeout_s=raw).normalized()
    assert cfg.judge_timeout_s == DEFAULT_JUDGE_TIMEOUT_S
    assert 0 < cfg.judge_timeout_s <= MAX_JUDGE_TIMEOUT_S
    # and the same through the dict form a config file / request body uses
    assert coerce_agentic_config({"judge_timeout_s": raw}).judge_timeout_s == DEFAULT_JUDGE_TIMEOUT_S


@pytest.mark.parametrize("raw", [0.001, 0.05, 0.0999])
def test_tiny_positive_judge_timeouts_are_raised_to_the_floor(raw):
    cfg = AgenticConfig(judge_timeout_s=raw).normalized()
    assert cfg.judge_timeout_s == MIN_JUDGE_TIMEOUT_S
    assert cfg.judge_timeout_s > 0


def test_valid_positive_judge_timeout_is_kept():
    assert AgenticConfig(judge_timeout_s=2.5).normalized().judge_timeout_s == 2.5
    assert AgenticConfig(judge_timeout_s=MAX_JUDGE_TIMEOUT_S).normalized().judge_timeout_s == MAX_JUDGE_TIMEOUT_S
    assert AgenticConfig(judge_timeout_s=10**6).normalized().judge_timeout_s == MAX_JUDGE_TIMEOUT_S


def test_coerce_agentic_config_accepts_dicts_and_instances():
    assert coerce_agentic_config(None).max_rounds == 2
    assert coerce_agentic_config({"max_rounds": 3}).max_rounds == 3
    assert coerce_agentic_config({"judge_fn": lambda q, r: None}).max_rounds == 2
    assert coerce_agentic_config({"bogus_key": 1}).max_rounds == 2, "unknown keys are ignored, not fatal"
    assert coerce_agentic_config(AgenticConfig(max_rounds=4)).max_rounds == 4
    assert coerce_agentic_config(7).max_rounds == 2, "unusable input degrades to the defaults"


def test_trace_as_dict_is_json_safe():
    trace = AgenticTrace(
        rounds=2,
        queries=["a", "b"],
        new_doc_ids=["x"],
        judge_verdicts=[{"sufficient": False, "missing": "", "rewrite": "b"}],
        stop_reason="sufficient",
        elapsed_s=0.123456789,
    )
    payload = trace.as_dict()
    assert set(payload) == {"rounds", "queries", "new_doc_ids", "judge_verdicts", "stop_reason", "elapsed_s"}
    assert json.loads(json.dumps(payload)) == payload
    assert payload["elapsed_s"] == 0.123457
    assert trace.stop_reason in STOP_REASONS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"sufficient": True, "missing": "x", "rewrite": " y "}, {"sufficient": True, "missing": "x", "rewrite": "y"}),
        ('{"sufficient": false, "rewrite": "z"}', {"sufficient": False, "missing": "", "rewrite": "z"}),
        (
            'here you go: {"sufficient": "yes", "rewrite": "q"} thanks',
            {"sufficient": True, "missing": "", "rewrite": "q"},
        ),
        ('{"sufficient": 0, "rewrite": null}', {"sufficient": False, "missing": "", "rewrite": ""}),
        ('{"sufficient": "no"}', {"sufficient": False, "missing": "", "rewrite": ""}),
    ],
)
def test_parse_judge_verdict_accepts_every_useful_shape(raw, expected):
    assert parse_judge_verdict(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "no json here", [], {}, 42, {"unrelated": 1}])
def test_parse_judge_verdict_returns_none_for_unusable_input(raw):
    assert parse_judge_verdict(raw) is None


def test_parse_judge_verdict_reads_an_sdk_style_response():
    class _Message:
        content = '{"sufficient": true}'

    class _Choice:
        message = _Message()

    class _Response:
        def __init__(self) -> None:
            self.choices = [_Choice()]

    assert parse_judge_verdict(_Response()) == {"sufficient": True, "missing": "", "rewrite": ""}


def test_doc_identity_is_stable_and_discriminating():
    assert doc_identity(DOC_A) == doc_identity(dict(DOC_A)), "same doc -> same id across rounds"
    assert doc_identity(DOC_A) != doc_identity(DOC_B)
    assert doc_identity({"text": "hello"}) == doc_identity({"text": "hello"})
    assert doc_identity({"text": "hello"}) != doc_identity({"text": "world"})
    assert doc_identity("a string") != doc_identity("another string")
    assert doc_identity({"_point_id": "abc"}) == "id:abc"
    # a formatted entry recurses into content and is dataset-qualified
    assert doc_identity({"dataset": "ds", "content": DOC_A}) == "ds/" + doc_identity(DOC_A)
    assert doc_identity({"content": DOC_A}) == doc_identity(DOC_A)
    # total on odd input
    assert isinstance(doc_identity(None), str)
    assert isinstance(doc_identity(object()), str)


def test_query_text_extracts_the_text_view():
    assert query_text("hello") == "hello"
    assert query_text({"text": "hello", "image": "data:"}) == "hello"
    assert query_text({"image": "data:"}) == ""
    assert query_text(None) == ""


def test_build_judge_prompt_is_bounded_and_mentions_the_question():
    docs = [_doc("x" * 5000, f"/docs/{i}.md") for i in range(20)]
    prompt = build_judge_prompt("what is it", [(d, 0.5) for d in docs])
    assert "what is it" in prompt
    assert prompt.count("[") >= 8, "at least the passage markers"
    assert len(prompt) < 8 * 5000, "passages are truncated and capped"


def test_build_judge_prompt_handles_no_passages():
    prompt = build_judge_prompt("q", [])
    assert "no passages retrieved" in prompt


def test_build_chat_judge_returns_none_without_a_model():
    assert build_chat_judge(None) is None
    assert build_chat_judge(object()) is None, "an object without the chat call is not a judge"


def test_build_chat_judge_calls_the_model_and_parses_the_reply():
    chat = _StubChatModel('{"sufficient": true}')
    judge = build_chat_judge(chat)
    assert judge is not None
    verdict = _run(judge("q", [(DOC_A, 0.9)]))
    assert verdict == {"sufficient": True, "missing": "", "rewrite": ""}
    assert len(chat.calls) == 1
    assert "retrieval judge" in chat.calls[0][0]["content"]


def test_run_agentic_entries_merges_formatted_entries():
    rounds = [
        [{"dataset": "alpha", "content": DOC_A, "score": 0.9, "score_kind": "cosine"}],
        [
            {"dataset": "alpha", "content": DOC_A, "score": 0.2, "score_kind": "cosine"},
            {"dataset": "beta", "content": DOC_A, "score": 0.8, "score_kind": "rrf"},
        ],
    ]
    calls = {"n": 0}

    async def search(query: Any) -> list[dict[str, Any]]:
        calls["n"] += 1
        return rounds[min(calls["n"] - 1, len(rounds) - 1)]

    judge = _JudgeRecorder([{"sufficient": False, "rewrite": "more"}, {"sufficient": True}])
    entries, trace = _run(run_agentic_entries("q", search, judge, AgenticConfig(max_rounds=3)))

    assert len(entries) == 2, "the same doc in two datasets is two entries"
    assert [(e["dataset"], e["score"]) for e in entries] == [("alpha", 0.9), ("beta", 0.8)]
    assert trace.rounds == 2
    assert trace.new_doc_ids == ["beta/" + doc_identity(DOC_A)]


def test_run_agentic_entries_rejects_a_missing_judge():
    with pytest.raises(TypeError):
        _run(run_agentic_entries("q", lambda q: [], None))


# ---------------------------------------------------------------------------
# rag_system hook (MultimodalRAG.aretrieve / MultiModalRAGSystem.aretrieve)
# ---------------------------------------------------------------------------


def test_non_agentic_path_is_unchanged():
    rag = _make_rag()
    docs = [dict(DOC_A), dict(DOC_B), dict(DOC_C)]
    plain = _run(rag.aretrieve("widgets", documents=list(docs), top_k=2))
    single = _run(rag._aretrieve_once("widgets", documents=list(docs), top_k=2))

    assert [(d["source"], round(s, 6)) for d, s in plain] == [(d["source"], round(s, 6)) for d, s in single]
    assert all("_agentic" not in d for d, _ in plain), "the default path must not grow metadata"
    sync_out = rag.retrieve("widgets", documents=list(docs), top_k=2)
    assert [(d["source"], round(s, 6)) for d, s in sync_out] == [(d["source"], round(s, 6)) for d, s in plain]


def test_agentic_hook_attaches_the_trace_and_accumulates_rounds():
    rag = _make_rag()
    docs = [dict(DOC_A), dict(DOC_B), dict(DOC_C)]
    judge = _JudgeRecorder(
        [{"sufficient": False, "missing": "sprockets", "rewrite": "sprockets"}, {"sufficient": True}]
    )
    out = _run(
        rag.aretrieve(
            "widgets",
            documents=list(docs),
            top_k=2,
            agentic=True,
            agentic_config={"judge_fn": judge, "max_rounds": 3},
        )
    )

    trace = out[0][0].get("_agentic")
    assert trace is not None, "the trace rides on the first result"
    assert trace["rounds"] == 2
    assert trace["stop_reason"] == "sufficient"
    assert trace["queries"] == ["widgets", "sprockets"]
    assert len(out) == 2 + len(trace["new_doc_ids"]), "each new doc id is one extra result"
    assert all("_agentic" not in d for d, _ in out[1:]), "attached exactly once"


def test_agentic_hook_degrades_when_no_judge_model_is_configured():
    rag = _make_rag()  # no vlm, no injected judge
    out = _run(rag.aretrieve("widgets", documents=[dict(DOC_A), dict(DOC_B)], top_k=2, agentic=True))

    assert len(out) == 2, "the plain single pass still returns results"
    assert out[0][0]["_agentic"]["stop_reason"] == "judge_unavailable"
    assert out[0][0]["_agentic"]["rounds"] == 1


def test_agentic_hook_returns_results_when_the_judge_raises():
    rag = _make_rag()
    judge = _JudgeRecorder(RuntimeError("model endpoint down"))
    out = _run(
        rag.aretrieve(
            "widgets", documents=[dict(DOC_A), dict(DOC_B)], top_k=2, agentic=True, agentic_config={"judge_fn": judge}
        )
    )
    assert len(out) == 2
    assert out[0][0]["_agentic"]["stop_reason"] == "judge_error"


def test_agentic_hook_does_not_mutate_the_callers_documents():
    rag = _make_rag()
    docs = [dict(DOC_A), dict(DOC_B)]
    judge = _JudgeRecorder({"sufficient": True})
    _run(rag.aretrieve("widgets", documents=docs, top_k=2, agentic=True, agentic_config={"judge_fn": judge}))
    assert all("_agentic" not in d for d in docs), "the caller's own dicts stay untouched"


def test_agentic_hook_rewrite_keeps_the_media_keys_of_a_multimodal_query():
    embedder = _StubEmbedder()
    rag = MultimodalRAG(embedder=embedder, preprocess=False)
    query = {"text": "widgets", "image": "data:image/png;base64,AA=="}
    judge = _JudgeRecorder([{"sufficient": False, "rewrite": "sprockets"}, {"sufficient": True}])
    _run(
        rag.aretrieve(
            query,
            documents=[dict(DOC_A), dict(DOC_B)],
            top_k=2,
            agentic=True,
            agentic_config={"judge_fn": judge, "max_rounds": 3},
        )
    )

    assert embedder.query_calls[0] is query, "round 1 embeds the caller's query object"
    rewritten = [q for q in embedder.query_calls[1:] if isinstance(q, dict)]
    assert rewritten, "the rewrite was re-embedded"
    assert rewritten[0]["text"] == "sprockets"
    assert rewritten[0]["image"] == "data:image/png;base64,AA==", "media keys survive the rewrite"


def test_agentic_hook_falls_back_when_the_loop_itself_breaks(monkeypatch):
    from multimodal_rag import rag_system

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a bug in the loop")

    monkeypatch.setattr(rag_system, "run_agentic_retrieval", _boom)
    rag = _make_rag()
    judge = _JudgeRecorder({"sufficient": True})
    out = _run(
        rag.aretrieve(
            "widgets", documents=[dict(DOC_A), dict(DOC_B)], top_k=2, agentic=True, agentic_config={"judge_fn": judge}
        )
    )

    assert len(out) == 2, "the safety net re-runs the plain single pass"
    assert out[0][0]["_agentic"]["stop_reason"] == "error_fallback"


def test_system_facade_uses_the_generation_model_as_the_judge():
    chat = _StubChatModel('{"sufficient": true}')
    system = MultiModalRAGSystem(llm=chat, embedder=_StubEmbedder(), preprocess=False)
    out = _run(system.aretrieve("widgets", documents=[dict(DOC_A), dict(DOC_B)], top_k=2, agentic=True))

    assert len(out) == 2
    assert out[0][0]["_agentic"]["stop_reason"] == "sufficient"
    assert len(chat.calls) == 1, "the facade's own chat model judged (no VLM is configured)"


def test_sync_retrieve_supports_agentic_with_a_sync_judge():
    rag = _make_rag()
    out = rag.retrieve(
        "widgets",
        documents=[dict(DOC_A), dict(DOC_B)],
        top_k=2,
        agentic=True,
        agentic_config={"judge_fn": lambda q, r: {"sufficient": True}},
    )
    assert out[0][0]["_agentic"]["stop_reason"] == "sufficient"


# ---------------------------------------------------------------------------
# MCP surface (_acore_retrieval / _arun_retrieval / the tool's wire schema)
# ---------------------------------------------------------------------------


class _FakeRag:
    """Smallest MultimodalRAG surface the MCP retrieval helpers touch."""

    def __init__(self, vlm: Any = None, docs: list[dict[str, Any]] | None = None) -> None:
        self.vlm = vlm
        self.reranker = None
        self._postprocessor = None
        self._docs = docs if docs is not None else [dict(DOC_A)]
        self.calls: list[dict[str, Any]] = []

    async def aretrieve(self, query: Any, **kwargs: Any) -> list[tuple[Any, float]]:
        self.calls.append({"query": query, **kwargs})
        return [(dict(d), 0.9) for d in self._docs]


class _StubVLM:
    """Marker: any non-None object is enough for the VLM/media heuristics."""


def test_acore_retrieval_keeps_the_non_agentic_request_identical():
    rag = _FakeRag()
    out = _run(mcp_server._acore_retrieval(rag, "ds", "q", "q", 5, False, 3, ["text"]))
    assert out is not None
    kwargs = {k: v for k, v in rag.calls[0].items() if k != "query"}
    assert kwargs == {
        "top_k": 5,
        "use_reranker": False,
        "reranker_top_k": 3,
        "query_vector": None,
        "need_media": False,
        "filters": None,
        "rrf": None,
        "agentic": False,
        "agentic_config": None,
    }


def test_acore_retrieval_threads_agentic_and_fetches_media_up_front():
    rag = _FakeRag()
    _run(mcp_server._acore_retrieval(rag, "ds", "q", "q", 5, False, 3, ["text"], agentic=True, agentic_max_rounds=3))

    assert len(rag.calls) == 1
    assert rag.calls[0]["agentic"] is True
    assert rag.calls[0]["agentic_config"] == {"max_rounds": 3}
    assert rag.calls[0]["need_media"] is True, "the accumulated multi-query set is fetched with media up front"


def test_acore_retrieval_skips_the_media_backfill_only_in_agentic_mode():
    media_doc = _doc("photo", "/docs/i.png", image="data:image/png;base64,AA==")

    plain = _FakeRag(vlm=_StubVLM(), docs=[media_doc])
    _run(mcp_server._acore_retrieval(plain, "ds", "q", "q", 5, False, 3, ["text"]))
    assert len(plain.calls) == 2, "historical behaviour: light search then a media back-fill re-fetch"

    agentic = _FakeRag(vlm=_StubVLM(), docs=[media_doc])
    _run(mcp_server._acore_retrieval(agentic, "ds", "q", "q", 5, False, 3, ["text"], agentic=True))
    assert len(agentic.calls) == 1, "agentic mode must not re-fetch with a single query"


def test_arun_retrieval_threads_agentic_to_the_rag():
    rag = _FakeRag()
    out = _run(
        mcp_server._arun_retrieval(
            rag, "ds", "q", "q", 5, False, 3, ["text"], None, None, agentic=True, agentic_max_rounds=2
        )
    )
    assert isinstance(out, str)
    assert rag.calls[0]["agentic"] is True
    assert rag.calls[0]["agentic_config"] == {"max_rounds": 2}


def test_clamp_agentic_rounds_bounds_the_tool_argument():
    assert mcp_server._clamp_agentic_rounds(1) == 1
    assert mcp_server._clamp_agentic_rounds(99) == MAX_AGENTIC_ROUNDS
    with pytest.raises(mcp_server.ToolError):
        mcp_server._clamp_agentic_rounds(0)
    with pytest.raises(mcp_server.ToolError):
        mcp_server._clamp_agentic_rounds("not-an-int")


def test_mcp_wire_schema_exposes_the_agentic_arguments():
    tools = _run(mcp_server.mcp.list_tools())
    tool = next(t for t in tools if t.name == "search_dataset")
    schema = tool.input_schema or {}
    props = schema.get("properties", {})

    assert props["agentic"]["default"] is False, "the feature is OPT-IN on the wire"
    assert props["agentic_max_rounds"]["default"] == 2
    assert "agentic" not in (schema.get("required") or []), "agentic must never be a required argument"
    assert "untrusted" in (tool.description or "").lower(), "the untrusted-content notice survives the docstring edit"


# ---------------------------------------------------------------------------
# REST surface (helpers + both endpoints)
# ---------------------------------------------------------------------------


class _StubDM:
    """Duck-typed DatasetManager for the REST agentic helpers."""

    def __init__(self, vlm: Any = "judge") -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.meta = {
            "alpha": {"name": "alpha", "document_count": 1, "has_password": False},
            "beta": {"name": "beta", "document_count": 1, "has_password": False},
        }
        self.rag = _FakeRag(vlm=_StubChatModel('{"sufficient": true}') if vlm == "judge" else vlm)

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        if name not in self.meta:
            raise FileNotFoundError(name)
        return dict(self.meta[name])

    def list_datasets(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.meta.values()]

    def has_password(self, name: str) -> bool:
        return False

    def _get_rag(self, name: str, check_embedder: bool = True) -> Any:
        return self.rag

    def search(
        self,
        name: str,
        query: Any,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
        filters: dict[str, Any] | None = None,
        rrf: Any = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append({"dataset": name, "query": query, "top_k": top_k})
        if query == "beta":
            return [{"content": dict(DOC_B), "score": 0.7, "score_kind": "cosine"}]
        return [{"content": dict(DOC_A), "score": 0.9, "score_kind": "cosine"}]


def test_agentic_request_params_defaults_and_validation():
    assert api_server._agentic_request_params({}) == (False, 2)
    assert api_server._agentic_request_params({"agentic": True, "agentic_max_rounds": 3}) == (True, 3)

    for body in (
        {"agentic": "false"},
        {"agentic": 1},
        {"agentic": True, "agentic_max_rounds": 0},
        {"agentic": True, "agentic_max_rounds": MAX_AGENTIC_ROUNDS + 1},
        {"agentic": True, "agentic_max_rounds": True},
        {"agentic": True, "agentic_max_rounds": "2"},
    ):
        with pytest.raises(Exception) as excinfo:
            api_server._agentic_request_params(body)
        assert getattr(excinfo.value, "status_code", None) == 400, body


def test_agentic_dataset_search_runs_the_loop_over_dm_search():
    dm = _StubDM()
    dm.rag = _FakeRag(vlm=_StubChatModel(['{"sufficient": false, "rewrite": "beta"}', '{"sufficient": true}']))
    results, trace = _run(
        api_server._agentic_dataset_search(
            dm, "alpha", "alpha", top_k=5, use_reranker=False, reranker_top_k=3, filters=None, rrf=None, max_rounds=3
        )
    )

    assert [r["content"]["source"] for r in results] == ["/docs/a.md", "/docs/b.md"]
    assert trace.rounds == 2
    assert trace.stop_reason == "sufficient"
    assert [c["query"] for c in dm.search_calls] == ["alpha", "beta"], "each round is one dm.search call"
    assert all(c["top_k"] == 5 for c in dm.search_calls)


def test_agentic_dataset_search_degrades_without_a_judge_model():
    dm = _StubDM(vlm=None)
    results, trace = _run(
        api_server._agentic_dataset_search(
            dm, "alpha", "alpha", top_k=5, use_reranker=False, reranker_top_k=3, filters=None, rrf=None, max_rounds=2
        )
    )
    assert len(results) == 1
    assert trace.stop_reason == "judge_unavailable"
    assert len(dm.search_calls) == 1


def test_agentic_dataset_search_falls_back_when_the_loop_breaks(monkeypatch):
    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("loop bug")

    monkeypatch.setattr(api_server, "run_agentic_entries", _boom)
    dm = _StubDM()
    results, trace = _run(
        api_server._agentic_dataset_search(
            dm, "alpha", "alpha", top_k=5, use_reranker=False, reranker_top_k=3, filters=None, rrf=None, max_rounds=2
        )
    )
    assert len(results) == 1
    assert trace.stop_reason == "error_fallback"


def test_federated_rest_search_non_agentic_output_is_unchanged():
    dm = _StubDM()
    out = _run(api_server._federated_rest_search(dm, ["alpha"], "alpha", top_k=2, is_unlocked=lambda n: False))

    assert set(out) == {"query", "filters", "results", "skipped", "errors"}
    assert len(dm.search_calls) == 1
    assert out["results"][0]["dataset"] == "alpha"


def test_federated_rest_search_agentic_merges_rounds_without_repeating_round_one():
    dm = _StubDM()
    dm.rag = _FakeRag(vlm=_StubChatModel(['{"sufficient": false, "rewrite": "beta"}', '{"sufficient": true}']))
    out = _run(
        api_server._federated_rest_search(
            dm, ["alpha"], "alpha", top_k=2, is_unlocked=lambda n: False, agentic=True, agentic_max_rounds=3
        )
    )

    assert out["agentic"]["rounds"] == 2
    assert out["agentic"]["stop_reason"] == "sufficient"
    assert [r["content"]["source"] for r in out["results"]] == ["/docs/a.md", "/docs/b.md"]
    assert [c["query"] for c in dm.search_calls] == ["alpha", "beta"], "round 1 must not be searched twice"


def test_federated_rest_search_agentic_dedupes_error_notes_across_rounds():
    class _BrokenDM(_StubDM):
        def search(
            self,
            name: str,
            query: Any,
            top_k: int = 10,
            use_reranker: bool = False,
            reranker_top_k: int = 3,
            filters: dict[str, Any] | None = None,
            rrf: Any = None,
        ) -> list[dict[str, Any]]:
            if name == "beta":
                raise RuntimeError("collection gone")
            return super().search(name, query, top_k, use_reranker, reranker_top_k, filters, rrf)

    dm = _BrokenDM()
    dm.rag = _FakeRag(vlm=_StubChatModel(['{"sufficient": false, "rewrite": "beta"}', '{"sufficient": true}']))
    out = _run(
        api_server._federated_rest_search(
            dm, ["alpha", "beta"], "alpha", is_unlocked=lambda n: False, agentic=True, agentic_max_rounds=3
        )
    )

    assert out["agentic"]["rounds"] == 2
    assert len(out["errors"]) == 1, "the same dataset failure is reported once, not once per round"
    assert out["errors"][0]["dataset"] == "beta"
    assert {r["dataset"] for r in out["results"]} == {"alpha"}
    assert [r["content"]["source"] for r in out["results"]] == ["/docs/a.md", "/docs/b.md"]


def test_federated_rest_search_agentic_degrades_without_a_judge_model():
    dm = _StubDM(vlm=None)
    out = _run(
        api_server._federated_rest_search(
            dm, ["alpha"], "alpha", is_unlocked=lambda n: False, agentic=True, agentic_max_rounds=2
        )
    )
    assert out["agentic"]["stop_reason"] == "judge_unavailable"
    assert len(dm.search_calls) == 1
    assert len(out["results"]) == 1


def test_dataset_search_endpoint_accepts_the_agentic_body_fields(monkeypatch):
    dm = _StubDM()

    async def _fake_manager() -> Any:
        return dm

    monkeypatch.setattr(api_server, "get_manager_async", _fake_manager)
    dm.rag = _FakeRag(vlm=_StubChatModel(['{"sufficient": false, "rewrite": "beta"}', '{"sufficient": true}']))

    out = _run(
        api_server.api_search_multimodal("alpha", None, {"query": "alpha", "agentic": True, "agentic_max_rounds": 3})
    )
    assert "agentic" in out
    assert out["agentic"]["rounds"] == 2
    assert out["agentic"]["stop_reason"] == "sufficient"
    assert len(out["results"]) == 2

    plain = _run(api_server.api_search_multimodal("alpha", None, {"query": "alpha"}))
    assert "agentic" not in plain, "the default body keeps the historical response shape"
    assert len(plain["results"]) == 1


def test_federated_search_endpoint_accepts_the_agentic_body_fields(monkeypatch):
    dm = _StubDM()

    async def _fake_manager() -> Any:
        return dm

    monkeypatch.setattr(api_server, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api_server, "_unlock_client_id", lambda request: "test-client")
    dm.rag = _FakeRag(vlm=_StubChatModel(['{"sufficient": false, "rewrite": "beta"}', '{"sufficient": true}']))

    out = _run(api_server.api_search_federated(None, {"datasets": ["alpha"], "q": "alpha", "agentic": True}))
    assert out["agentic"]["rounds"] == 2
    assert out["agentic"]["queries"] == ["alpha", "beta"]

    plain = _run(api_server.api_search_federated(None, {"datasets": ["alpha"], "q": "alpha"}))
    assert "agentic" not in plain
    assert set(plain) == {"query", "filters", "results", "skipped", "errors"}
