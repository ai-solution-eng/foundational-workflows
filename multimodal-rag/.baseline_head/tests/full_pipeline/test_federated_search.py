"""Offline tests for federated multi-dataset search (roadmap feature 8).

Covers the fan-out / merge logic (deliberately module-level functions, NOT
inside the MCP tool registry, so the runtime is not needed):

  * ``resolve_federated_targets`` — list / single-name / ``"all"`` expansion,
    password-protected datasets skipped WITH a note (never a ``password``
    argument — the v3.0.0 decision), unknown names reported as errors.
  * ``dedup_federated_results`` / ``merge_federated_results`` — the
    dataset-qualified twin-identity key ``(dataset, source, page,
    chunk_index, time-window)``: twins collapse within a dataset, the same
    identity in two datasets survives as two labelled hits.
  * ``mcp_server._afederated_search`` — the concurrent fan-out behind the
    MCP ``search_datasets`` tool: per-dataset ``top_k``, dataset labels,
    per-dataset scores, one dataset erroring → note (others still return),
    filters reaching every dataset, and the single merged rerank pass.
  * ``api_server._federated_rest_search`` — the ``sync_pool`` fan-out behind
    ``POST /api/search``, with ``dm.search`` stubbed and the ``filters``
    kwarg asserted per dataset.
  * End-to-end over two REAL embedded local Qdrant collections (stub
    embedder) — production retrieval path on both datasets, merged.

No model endpoint required — the embedder is stubbed.

Run::

    python tests/full_pipeline/test_federated_search.py    # standalone
    pytest tests/full_pipeline/test_federated_search.py    # under pytest
"""

import asyncio
import hashlib
import os
import sys
import uuid
from typing import Any

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from multimodal_rag.api_server import _federated_rest_search
from multimodal_rag.mcp_server import _afederated_search
from multimodal_rag.rag_system import (
    MultimodalRAG,
    dedup_federated_results,
    federated_identity_key,
    merge_federated_results,
    resolve_federated_targets,
)
from multimodal_rag.vector_store import QdrantVectorStore

DIM = 8


# ---------------------------------------------------------------------------
# Stubs (house pattern: embedded local Qdrant + stub embedder, no models)
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Deterministic hash "embedding" — distinct docs get distinct scores."""

    model_name = "stub"
    base_url = "http://stub/v1"

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:DIM]]

    async def aembed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)

    def embed_query(self, query: Any) -> list[float]:
        return self._vec(query if isinstance(query, str) else "")


class _DatasetRig:
    """One dataset: embedded local Qdrant collection + real MultimodalRAG.

    ``rag.aretrieve`` is wrapped with a recorder so tests can assert what the
    federated fan-out actually asked each dataset for (per-dataset top_k,
    use_reranker=False, the filters dict, ...).
    """

    def __init__(self, name: str, docs: list[dict[str, Any]], has_password: bool = False) -> None:
        self.name = name
        self.meta: dict[str, Any] = {"name": name, "document_count": len(docs), "has_password": has_password}
        self.client = QdrantClient(":memory:")
        self.coll = f"fed_{name}"
        try:
            self.client.delete_collection(self.coll)
        except Exception:
            pass
        self.client.create_collection(self.coll, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
        self.store = QdrantVectorStore(self.client, self.coll, embedding=_StubEmbedder())
        self.rag = MultimodalRAG(embedder=_StubEmbedder(), vector_store=self.store, preprocess=False)
        self.calls: list[dict[str, Any]] = []

        original = self.rag.aretrieve

        async def _spy(query: Any, **kwargs: Any) -> list[tuple[Any, float]]:
            self.calls.append({"query": query, **kwargs})
            return await original(query, **kwargs)

        self.rag.aretrieve = _spy  # type: ignore[method-assign]
        self._upsert(docs)

    def _upsert(self, docs: list[dict[str, Any]]) -> None:
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=_StubEmbedder._vec(d["text"]),
                payload={"page_content": d["text"], "metadata": d.get("metadata", {"source": d["source"]})},
            )
            for d in docs
        ]
        if points:
            self.client.upsert(self.coll, points=points, wait=True)


class _StubDM:
    """Duck-typed DatasetManager over a set of rigs (house pattern: the
    smallest surface the federated code touches — ``get_dataset``,
    ``list_datasets``, ``has_password``, ``_get_rag`` — plus a stub
    ``search`` for the REST face, mirroring its real signature)."""

    def __init__(self, rigs: list[_DatasetRig]) -> None:
        self._rigs = {r.name: r for r in rigs}
        self.search_calls: list[dict[str, Any]] = []

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        if name not in self._rigs:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        return dict(self._rigs[name].meta)

    def list_datasets(self) -> list[dict[str, Any]]:
        return [dict(r.meta) for r in self._rigs.values()]

    def has_password(self, name: str) -> bool:
        return bool(self._rigs[name].meta.get("has_password"))

    def _get_rag(self, name: str, check_embedder: bool = True) -> MultimodalRAG:
        return self._rigs[name].rag

    def search(
        self,
        dataset_name: str,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {
                "dataset": dataset_name,
                "query": query,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "reranker_top_k": reranker_top_k,
                "filters": filters,
            }
        )
        return [
            {
                "content": {"text": f"{dataset_name} doc {i}", "source": f"/{dataset_name}/f{i}.md"},
                "score": 0.5 + i / 10,
            }
            for i in range(min(top_k, 2))
        ]


def _doc(text: str, source: str, **meta: Any) -> dict[str, Any]:
    return {"text": text, "source": source, "metadata": {"source": source, **meta}}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# resolve_federated_targets — "all" expansion, locked datasets, unknown names
# ---------------------------------------------------------------------------


def test_resolve_targets_list_single_and_dedup():
    rigs = [_DatasetRig("alpha", []), _DatasetRig("beta", [])]
    dm = _StubDM(rigs)

    targets, skipped, errors = resolve_federated_targets(dm, ["alpha", "beta", "alpha"], is_unlocked=lambda n: False)
    assert targets == ["alpha", "beta"], "duplicates dropped, order preserved"
    assert skipped == [] and errors == []

    # A bare string is a single dataset name
    targets, _, _ = resolve_federated_targets(dm, "beta", is_unlocked=lambda n: False)
    assert targets == ["beta"]


def test_resolve_targets_unknown_name_is_an_error_note():
    dm = _StubDM([_DatasetRig("alpha", [])])
    targets, skipped, errors = resolve_federated_targets(dm, ["alpha", "ghost"], is_unlocked=lambda n: False)
    assert targets == ["alpha"]
    assert skipped == []
    assert len(errors) == 1 and errors[0]["dataset"] == "ghost" and "not found" in errors[0]["error"]


def test_all_expansion_skips_password_protected_dataset_with_note():
    """'all' expands to datasets readable WITHOUT a password; a locked one
    is skipped with a note (federated search accepts no password argument)."""
    rigs = [
        _DatasetRig("open1", [_doc("alpha text", "/open1/a.md")]),
        _DatasetRig("open2", [_doc("beta text", "/open2/b.md")]),
        _DatasetRig("secret", [_doc("secret text", "/secret/s.md")], has_password=True),
    ]
    dm = _StubDM(rigs)

    targets, skipped, errors = resolve_federated_targets(dm, "all", is_unlocked=lambda n: False)
    assert targets == ["open1", "open2"]
    assert errors == []
    assert [s["dataset"] for s in skipped] == ["secret"]
    assert "password" in skipped[0]["reason"].lower()

    # Case-insensitive, and a locked dataset that IS unlocked is included.
    targets, skipped, _ = resolve_federated_targets(dm, " ALL ", is_unlocked=lambda n: n == "secret")
    assert targets == ["open1", "open2", "secret"]
    assert skipped == []


def test_all_expansion_uses_the_real_mcp_unlock_cache():
    """'unlocked' = an unexpired entry in the in-process MCP unlock cache for
    the calling client (_cache_unlock → _is_unlocked)."""
    from multimodal_rag import mcp_server as mcp

    dm = _StubDM(
        [
            _DatasetRig("open", []),
            _DatasetRig("locked", [], has_password=True),
        ]
    )
    try:
        targets, skipped, _ = mcp._resolve_federated_targets(dm, "all")
        assert targets == ["open"] and [s["dataset"] for s in skipped] == ["locked"]

        mcp._cache_unlock("locked", "pw")  # unlocks it for the "default" client
        targets, skipped, _ = mcp._resolve_federated_targets(dm, "all")
        assert targets == ["open", "locked"] and skipped == []

        # Once the cached unlock is gone (expired/revoked), the dataset is
        # skipped again — and an explicitly named locked dataset is skipped
        # with a note, never an error.
        with mcp._unlocked_lock:
            mcp._unlocked.clear()
        targets, skipped, _ = mcp._resolve_federated_targets(dm, ["locked"])
        assert targets == [] and [s["dataset"] for s in skipped] == ["locked"]
    finally:
        with mcp._unlocked_lock:
            mcp._unlocked.clear()


def test_resolve_targets_rejects_bad_shapes():
    dm = _StubDM([_DatasetRig("alpha", [])])
    try:
        resolve_federated_targets(dm, 42)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        assert "datasets" in str(exc)
    else:
        raise AssertionError("non-list/non-str datasets must raise")


# ---------------------------------------------------------------------------
# Dataset-qualified dedup / merge
# ---------------------------------------------------------------------------


def test_federated_identity_key_is_dataset_qualified():
    doc = {"source": "/a/f.md", "page": 1, "chunk_index": 0, "timestamp_start": 0.0, "timestamp_end": 2.0}
    assert federated_identity_key("alpha", doc) == ("alpha", "/a/f.md", 1, 0, (0.0, 2.0))
    assert federated_identity_key("beta", doc) != federated_identity_key("alpha", doc)
    assert federated_identity_key("alpha", {"text": "no source"}) is None
    assert federated_identity_key("alpha", "bare string") is None


def test_dedup_keeps_same_identity_across_datasets():
    """The same (source, page, chunk) in TWO datasets must survive as two
    labelled hits — only the dataset qualifier distinguishes them."""
    doc_a = {"text": "shared runbook", "source": "/data/datasets/x/files/rb.md", "page": 1, "chunk_index": 0}
    doc_b = dict(doc_a)
    kept = dedup_federated_results([("alpha", doc_a, 0.9), ("beta", doc_b, 0.8)])
    assert sorted(ds for ds, _, _ in kept) == ["alpha", "beta"]


def test_dedup_collapses_twins_within_a_dataset():
    parent = {"text": "real text", "source": "/a/f.md", "page": 1, "chunk_index": 0}
    twin = dict(parent, text="real text", _twin=True)
    kept = dedup_federated_results([("alpha", parent, 0.9), ("alpha", twin, 0.95)])
    assert len(kept) == 1
    assert kept[0][0] == "alpha" and kept[0][1] is parent, "the non-twin parent is kept"

    # Identical TEXT within one dataset collapses to the best score...
    dup = {"text": "real text", "source": "/a/g.md", "page": 2, "chunk_index": 0}
    kept = dedup_federated_results([("alpha", parent, 0.9), ("alpha", dup, 0.99)])
    assert len(kept) == 1 and kept[0][1] is dup and kept[0][2] == 0.99

    # ...but the same text in another dataset is a different artifact.
    kept = dedup_federated_results([("alpha", dict(parent), 0.9), ("beta", dict(parent), 0.5)])
    assert len(kept) == 2


def test_merge_sorts_by_score_and_preserves_dataset_order_on_ties():
    entries = [("alpha", {"text": "a"}, 0.5), ("beta", {"text": "b"}, 0.9), ("alpha", {"text": "c"}, 0.9)]
    merged = merge_federated_results(entries, dedup=False)
    assert [round(m[2], 2) for m in merged] == [0.9, 0.9, 0.5]
    assert [m[0] for m in merged][:2] == ["beta", "alpha"], "stable sort keeps the caller's dataset order on ties"


def test_single_dataset_dedup_twins_still_collapses():
    """Regression guard: the refactor of _dedup_twins onto the shared
    identity dedup must not change the single-dataset behaviour."""
    parent = {"text": "real text", "source": "/a/f.md", "page": 1, "chunk_index": 0}
    twin = dict(parent, _twin=True)
    kept = MultimodalRAG._dedup_twins([(parent, 0.9), (twin, 0.95)])
    assert len(kept) == 1 and kept[0][0] is parent


# ---------------------------------------------------------------------------
# MCP fan-out core (_afederated_search) — the engine behind search_datasets
# ---------------------------------------------------------------------------


def test_federated_merge_labels_results_and_applies_per_dataset_top_k():
    rigs = [
        _DatasetRig(
            "alpha",
            [
                _doc("alpha one", "/alpha/1.md", page=1, chunk_index=0),
                _doc("alpha two", "/alpha/2.md", page=1, chunk_index=1),
                _doc("alpha three", "/alpha/3.md", page=2, chunk_index=0),
            ],
        ),
        _DatasetRig(
            "beta",
            [
                _doc("beta one", "/beta/1.md", page=1, chunk_index=0),
                _doc("beta two", "/beta/2.md", page=1, chunk_index=1),
            ],
        ),
    ]
    dm = _StubDM(rigs)

    payload = _run(_afederated_search(dm, ["alpha", "beta"], "alpha one", top_k=2))

    # per-dataset top_k (each dataset retrieved at most 2 candidates)
    assert len(rigs[0].calls) == 1 and rigs[0].calls[0]["top_k"] == 2
    assert len(rigs[1].calls) == 1 and rigs[1].calls[0]["top_k"] == 2
    # per-dataset retrieval must NOT rerank (one merged pass handles that)
    assert rigs[0].calls[0]["use_reranker"] is False

    results = payload["results"]
    assert len(results) == 4, "2 candidates per dataset, both datasets contribute"
    assert {r["dataset"] for r in results} == {"alpha", "beta"}
    assert sum(1 for r in results if r["dataset"] == "alpha") == 2
    assert sum(1 for r in results if r["dataset"] == "beta") == 2

    # merged pool sorted by score, every hit keeps its own score
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(r["score"], float) for r in results)

    # per-dataset headers in the formatted context
    assert "### Dataset: alpha" in payload["context"]
    assert "### Dataset: beta" in payload["context"]
    assert payload["datasets_searched"] == ["alpha", "beta"]
    assert payload["skipped"] == [] and payload["errors"] == []


def test_federated_all_expansion_reports_skipped_dataset_in_payload():
    rigs = [
        _DatasetRig("open", [_doc("open text", "/open/a.md")]),
        _DatasetRig("secret", [_doc("secret text", "/secret/s.md")], has_password=True),
    ]
    dm = _StubDM(rigs)

    payload = _run(_afederated_search(dm, "all", "open text", is_unlocked=lambda n: False))
    assert payload["datasets_searched"] == ["open"]
    assert [s["dataset"] for s in payload["skipped"]] == ["secret"]
    assert all(r["dataset"] != "secret" for r in payload["results"])
    assert "secret" in payload["context"], "the skip note must be visible in the text output"

    # Same call with the dataset unlocked → it is searched
    payload = _run(_afederated_search(dm, "all", "open text", is_unlocked=lambda n: n == "secret"))
    assert payload["datasets_searched"] == ["open", "secret"]
    assert {r["dataset"] for r in payload["results"]} == {"open", "secret"}
    assert payload["skipped"] == []


def test_federated_cross_dataset_dedup_through_the_pipeline():
    """Same source/page/chunk ingested into two datasets: the intra-dataset
    twin collapses, the other dataset's copy survives, labelled."""
    shared = {"source": "/data/datasets/x/files/rb.md"}
    rigs = [
        _DatasetRig(
            "alpha",
            [
                _doc("runbook text", **shared, page=1, chunk_index=0),
                _doc("runbook text", **shared, page=1, chunk_index=0, _twin=True),
            ],
        ),
        _DatasetRig("beta", [_doc("runbook text", **shared, page=1, chunk_index=0)]),
    ]
    dm = _StubDM(rigs)

    payload = _run(_afederated_search(dm, ["alpha", "beta"], "runbook text", top_k=5))
    assert len(payload["results"]) == 2, "alpha's twin collapsed; beta's copy kept"
    assert sorted(r["dataset"] for r in payload["results"]) == ["alpha", "beta"]
    alpha_hit = next(r for r in payload["results"] if r["dataset"] == "alpha")
    assert "_twin" not in alpha_hit.get("text", "")


def test_federated_one_dataset_failing_is_a_note_not_a_failure():
    rigs = [
        _DatasetRig("alpha", [_doc("alpha text", "/alpha/a.md")]),
        _DatasetRig("beta", [_doc("beta text", "/beta/b.md")]),
        _DatasetRig("boom", [_doc("boom text", "/boom/c.md")]),
    ]
    dm = _StubDM(rigs)

    async def _explode(query: Any, **kwargs: Any) -> list[tuple[Any, float]]:
        raise RuntimeError("qdrant exploded")

    rigs[2].rag.aretrieve = _explode  # type: ignore[method-assign]

    payload = _run(_afederated_search(dm, ["alpha", "boom", "beta"], "text", top_k=3))

    assert len(payload["errors"]) == 1
    assert payload["errors"][0]["dataset"] == "boom"
    assert "qdrant exploded" in payload["errors"][0]["error"]
    assert "Dataset 'boom' failed" in payload["context"]

    datasets_of_results = {r["dataset"] for r in payload["results"]}
    assert datasets_of_results == {"alpha", "beta"}, "the healthy datasets still contribute"
    assert all(r["dataset"] != "boom" for r in payload["results"])
    # the failing dataset's collection was never read by the healthy ones
    assert len(rigs[0].calls) == 1 and len(rigs[1].calls) == 1

    # A nonexistent dataset named explicitly is an error note too
    payload = _run(_afederated_search(dm, ["alpha", "ghost"], "text", top_k=3))
    assert payload["errors"][0]["dataset"] == "ghost"
    assert {r["dataset"] for r in payload["results"]} == {"alpha"}


def test_federated_filters_reach_every_dataset():
    filters = {"file_types": ["log"], "severities": ["ERROR"]}
    rigs = [
        _DatasetRig("alpha", [_doc("ERROR alpha log", "/alpha/a.log", severities=["ERROR"], file_type="log")]),
        _DatasetRig("beta", [_doc("ERROR beta log", "/beta/b.log", severities=["ERROR"], file_type="log")]),
    ]
    dm = _StubDM(rigs)

    payload = _run(_afederated_search(dm, ["alpha", "beta"], "ERROR", top_k=3, filters=filters))
    for rig in rigs:
        assert rig.calls, f"{rig.name} was searched"
        assert rig.calls[0]["filters"] == filters, "the filter dict must reach every dataset"
    assert len(payload["results"]) == 2
    assert {r["dataset"] for r in payload["results"]} == {"alpha", "beta"}


def test_federated_merged_rerank_single_pass():
    """One rerank pass over the MERGED pool (not per dataset), truncating to
    reranker_top_k, keeping the dataset labels and the embedding scores."""
    rigs = [
        _DatasetRig(
            "alpha",
            [
                _doc("alpha one", "/alpha/1.md", page=1, chunk_index=0),
                _doc("alpha two", "/alpha/2.md", page=1, chunk_index=1),
            ],
        ),
        _DatasetRig("beta", [_doc("beta one", "/beta/1.md", page=1, chunk_index=0)]),
    ]
    dm = _StubDM(rigs)

    rerank_batches: list[list[Any]] = []

    class _StubRerankModel:
        async def arerank(self, query: str, documents: list[Any]) -> list[list[dict[str, Any]]]:
            rerank_batches.append(list(documents))
            # relevance = reverse order, deterministic and distinct
            n = len(documents)
            return [[{"index": n - 1 - i, "relevance_score": (n - 1 - i) / 10} for i in range(n)]]

    class _StubReranker:
        model = _StubRerankModel()

    rigs[0].rag.reranker = _StubReranker()  # type: ignore[assignment]

    payload = _run(_afederated_search(dm, ["alpha", "beta"], "query", top_k=2, use_reranker=True, reranker_top_k=2))

    assert len(rerank_batches) == 1, "exactly one rerank call, over the merged pool"
    assert len(rerank_batches[0]) == 3, "all candidates from both datasets in one pass"
    # per-dataset retrieval did not rerank
    assert all(call["use_reranker"] is False for rig in rigs for call in rig.calls)

    results = payload["results"]
    assert len(results) == 2, "truncated to reranker_top_k"
    assert all("reranker_score" in r for r in results)
    assert [r["reranker_score"] for r in results] == sorted((r["reranker_score"] for r in results), reverse=True)
    # labels survived the sort/truncate, and the embedding score breakdown too
    assert {r["dataset"] for r in results} <= {"alpha", "beta"}
    assert all("embedding_score" in r for r in results)
    assert not any("_federated_dataset" in str(r) for r in results), "private label key must not leak"


def test_federated_no_targets_and_no_hits():
    dm = _StubDM([_DatasetRig("secret", [], has_password=True)])

    payload = _run(_afederated_search(dm, "all", "q", is_unlocked=lambda n: False))
    assert payload["results"] == [] and payload["datasets_searched"] == []
    assert payload["context"], "a text note explains that nothing was searchable"

    # every dataset empty → "no results" note, no crash
    dm2 = _StubDM([_DatasetRig("empty", [])])
    payload = _run(_afederated_search(dm2, ["empty"], "q"))
    assert payload["results"] == []
    assert payload["datasets_searched"] == ["empty"]


def test_federated_media_query_dict_shared_across_datasets():
    """image/video/audio build ONE multimodal query dict used by every dataset."""
    rigs = [_DatasetRig("alpha", []), _DatasetRig("beta", [])]
    dm = _StubDM(rigs)

    _run(_afederated_search(dm, ["alpha", "beta"], "find this", image="data:image/png;base64,AA=="))
    for rig in rigs:
        q = rig.calls[0]["query"]
        assert isinstance(q, dict)
        assert q["text"] == "find this" and q["image"] == "data:image/png;base64,AA=="


# ---------------------------------------------------------------------------
# REST face (_federated_rest_search behind POST /api/search)
# ---------------------------------------------------------------------------


def test_rest_federated_search_merges_labels_skips_and_propagates_filters():
    rigs = [
        _DatasetRig("alpha", []),
        _DatasetRig("beta", []),
        _DatasetRig("locked", [], has_password=True),
    ]
    dm = _StubDM(rigs)
    filters = {"file_types": ["log"]}

    out = _run(
        _federated_rest_search(
            dm,
            ["alpha", "beta", "locked"],
            "q text",
            top_k=3,
            filters=filters,
            is_unlocked=lambda n: False,
        )
    )

    # per-dataset dm.search calls, offloaded and gathered concurrently
    assert [c["dataset"] for c in dm.search_calls] == ["alpha", "beta"]
    assert all(c["top_k"] == 3 for c in dm.search_calls)
    assert all(c["filters"] == filters for c in dm.search_calls), "filters kwarg arrived on every dataset search"
    assert all(c["use_reranker"] is False for c in dm.search_calls)

    assert out["query"] == "q text" and out["filters"] == filters
    assert [s["dataset"] for s in out["skipped"]] == ["locked"]
    assert out["errors"] == []
    assert {r["dataset"] for r in out["results"]} == {"alpha", "beta"}
    assert all(r["score"] == round(r["score"], 4) for r in out["results"])
    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True), "merged pool is score-ordered"


def test_rest_federated_search_errors_are_per_dataset():
    rigs = [_DatasetRig("alpha", []), _DatasetRig("beta", [])]
    dm = _StubDM(rigs)

    def _search(name, query, top_k=10, use_reranker=False, reranker_top_k=3, filters=None):
        if name == "beta":
            raise RuntimeError("collection gone")
        return [{"content": {"text": "alpha doc", "source": "/alpha/a.md"}, "score": 0.8}]

    dm.search = _search  # type: ignore[method-assign]

    out = _run(_federated_rest_search(dm, ["alpha", "beta"], "q", is_unlocked=lambda n: False))
    assert len(out["errors"]) == 1 and out["errors"][0]["dataset"] == "beta"
    assert "collection gone" in out["errors"][0]["error"]
    assert [r["dataset"] for r in out["results"]] == ["alpha"]

    # unknown dataset → error note, not an exception
    out = _run(_federated_rest_search(dm, ["alpha", "ghost"], "q", is_unlocked=lambda n: False))
    assert out["errors"][0]["dataset"] == "ghost"
    assert [r["dataset"] for r in out["results"]] == ["alpha"]

    # malformed datasets argument → 400
    try:
        _run(_federated_rest_search(dm, 7, "q"))  # type: ignore[arg-type]
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    else:
        raise AssertionError("malformed datasets must raise HTTPException(400)")


def test_rest_federated_search_single_merged_rerank():
    rigs = [_DatasetRig("alpha", []), _DatasetRig("beta", [])]
    dm = _StubDM(rigs)

    rerank_docs: list[list[Any]] = []

    class _StubRerankModel:
        async def arerank(self, query: str, documents: list[Any]) -> list[list[dict[str, Any]]]:
            rerank_docs.append(list(documents))
            # same shape MultiModalReranker.arerank returns: one score list per query
            return [[{"index": i, "relevance_score": 0.42} for i in range(len(documents))]]

    class _StubReranker:
        model = _StubRerankModel()

    rig_rag = type(
        "_Rag",
        (),
        {
            "reranker": _StubReranker(),
            "rank": property(lambda self: _StubReranker().model),
            "_extract_doc": staticmethod(lambda d: d),
        },
    )()
    dm._get_rag = lambda name, check_embedder=True: rig_rag  # type: ignore[method-assign]

    out = _run(
        _federated_rest_search(
            dm,
            ["alpha", "beta"],
            "q",
            top_k=2,
            use_reranker=True,
            reranker_top_k=3,
            is_unlocked=lambda n: False,
        )
    )

    assert len(rerank_docs) == 1 and len(rerank_docs[0]) == 4, "one pass over the merged (2+2) pool"
    assert len(out["results"]) == 3
    assert all(r["reranker_score"] == 0.42 for r in out["results"])
    assert all("embedding_score" in r for r in out["results"]), "per-dataset score breakdown kept"
    assert {r["dataset"] for r in out["results"]} == {"alpha", "beta"}


def test_rest_federated_search_no_targets():
    dm = _StubDM([_DatasetRig("locked", [], has_password=True)])
    out = _run(_federated_rest_search(dm, "all", "q", is_unlocked=lambda n: False))
    assert out == {"query": "q", "filters": None, "results": [], "skipped": out["skipped"], "errors": []}
    assert [s["dataset"] for s in out["skipped"]] == ["locked"]


# ---------------------------------------------------------------------------
# End-to-end: two real embedded-local-Qdrant datasets through the real
# retrieval path, merged by the federated core
# ---------------------------------------------------------------------------


def test_end_to_end_two_qdrant_datasets_merged():
    rigs = [
        _DatasetRig(
            "alpha",
            [
                _doc("the deployment runbook for the edge cluster", "/alpha/runbook.md", page=1, chunk_index=0),
                _doc("unrelated alpha note about gardening", "/alpha/garden.md", page=1, chunk_index=0),
            ],
        ),
        _DatasetRig(
            "beta",
            [
                _doc("runbook addendum: rollback procedure", "/beta/rollback.md", page=1, chunk_index=0),
                _doc("beta meeting minutes", "/beta/minutes.md", page=1, chunk_index=0),
            ],
        ),
    ]
    dm = _StubDM(rigs)

    payload = _run(_afederated_search(dm, ["alpha", "beta"], "deployment runbook", top_k=2))

    assert payload["errors"] == [] and payload["skipped"] == []
    assert len(payload["results"]) == 4
    assert {r["dataset"] for r in payload["results"]} == {"alpha", "beta"}
    # every hit went through the real retrieval path (payload metadata intact)
    for r in payload["results"]:
        assert r["text"], "retrieved text surfaced"
        assert r["source"]
    # the runbook docs from both datasets rank in the merged pool
    texts = [r["text"] for r in payload["results"]]
    assert any("runbook" in t for t in texts)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
