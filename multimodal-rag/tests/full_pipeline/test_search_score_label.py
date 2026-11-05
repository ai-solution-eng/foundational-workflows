"""Offline tests for the search score-label contract.

The web UI's Search tab used to render a reranked hit as
"#1 . score: 0.4191 (rrf rank fusion)" even though 0.4191 was the RERANKER
relevance: DatasetManager.search copied the reranker score into `score` but
left the private `_score_kind` lane stamp inside `content`, and the
template fell back to it.  These tests pin the fixed contract (mirroring the
MCP surface, which already did it correctly):

  * every entry says WHAT its `score` is (`score_kind`: reranker | rrf |
    cosine);
  * the private lane stamps (`_score_kind` / `_retrieval_score`) never
    leak into the public `content` payload;
  * a reranked hit keeps its pre-rerank score visible as `retrieval_score`
    alongside `reranker_score`;
  * the federated REST path (`POST /api/search`) applies the same labels
    without losing the lane-aware merge.

Offline: embedded local Qdrant + stub embedder + stub reranker, no model
endpoints, no network.

Run::

    timeout 300 .venv/bin/python -m pytest tests/full_pipeline/test_search_score_label.py -q
"""

import asyncio
import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient

from multimodal_rag.api_server import _federated_rest_search
from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

DIM = 8
COLL = "score_label_test"

# Deliberately unlike an RRF sum (the bug report's number): the displayed
# score must be THIS value, and must be labelled "reranker".
RERANK_TOP = 0.4191


@contextmanager
def _env(**overrides: str):
    """Temporarily set env vars (standalone-safe counterpart of monkeypatch)."""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _StubEmbedder:
    """Deterministic hash "embedding" wrapped like the real embedder model.

    The hash carries NO lexical signal (sha256 of the text), so the BM25 lane
    is what makes the hybrid collection actually fuse — the same regime as
    test_hybrid_search.py.
    """

    allowable_modalities: ClassVar[tuple[str, ...]] = ("text", "image", "video")
    model_name = "stub-embedder"
    base_url = "http://stub/v1"
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None

    def __init__(self) -> None:
        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[assignment]
        self.model.aembed_query = self._aembed_query  # type: ignore[assignment]
        self.model.embed_query = self._embed_query  # type: ignore[assignment]

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:DIM]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        return [self._vec(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    async def _aembed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)

    def _embed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)


class _StubRerankModel:
    """Deterministic relevance: the first candidate scores RERANK_TOP."""

    async def arerank(self, query: Any, documents: list[Any]) -> list[list[dict[str, Any]]]:
        return [[{"index": i, "relevance_score": round(RERANK_TOP - 0.1 * i, 4)} for i in range(len(documents))]]


class _StubReranker:
    model = _StubRerankModel()


class _Rig:
    """Hybrid-capable store + real MultimodalRAG on embedded local Qdrant."""

    def __init__(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.emb = _StubEmbedder()
        self.client = QdrantClient(":memory:")
        self.files_dir = Path(self.tmp.name) / "ds" / "files"
        self.store = MultimodalRAG._build_qdrant_vector_store(  # type: ignore[assignment]
            embedding=self.emb.model,
            client=self.client,
            collection_name=COLL,
            bm25_stats_path=str(self.files_dir / bm25_lane.BM25_STATS_FILENAME),
        )
        self.rag = MultimodalRAG(embedder=self.emb, vector_store=self.store)  # type: ignore[arg-type]

    def ingest(self, docs: list[dict[str, Any]]) -> None:
        asyncio.run(self.rag.aadd_to_vector_store(docs, deduplicate=False))

    def close(self) -> None:
        self.tmp.cleanup()


def _dm(rig: _Rig) -> DatasetManager:
    """DatasetManager shell over the rig — no meta files, no live Qdrant.

    Only the two collaborators `search` touches are wired: `_get_rag` (the
    rig's real RAG) and `_effective_rrf` (no dataset defaults).
    """
    dm = DatasetManager.__new__(DatasetManager)
    dm._get_rag = lambda dataset_name, check_embedder=True: rig.rag  # type: ignore[method-assign]
    dm._effective_rrf = lambda dataset_name, rrf=None: rrf  # type: ignore[method-assign]
    dm.datasets_path = Path(rig.tmp.name)  # type: ignore[attr-defined]
    return dm


DOCS = [
    {"text": "WebSocketHandler negotiates the handshake upgrade", "source": "/tmp/x/ws.py"},
    {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
]


# ---------------------------------------------------------------------------
# Single-dataset DatasetManager.search contract
# ---------------------------------------------------------------------------


def test_reranked_hybrid_hit_is_labelled_reranker() -> None:
    """The reported bug: a reranked hybrid hit showed the RERANKER relevance
    labelled as RRF rank fusion.  It must be labelled "reranker" and keep the
    pre-rerank RRF score visible as `retrieval_score`."""
    with _env(RAG_HYBRID_SEARCH="1"):
        rig = _Rig()
        try:
            rig.ingest(DOCS)
            rig.rag.reranker = _StubReranker()  # type: ignore[assignment]
            dm = _dm(rig)

            # The same query without the rerank pass: its scores ARE the
            # pre-rerank RRF values the reranked entry must report.
            plain = dm.search("ds", "WebSocketHandler", top_k=2)
            assert plain and all(e["score_kind"] == "rrf" for e in plain)
            pre_rerank = {e["content"]["text"]: e["score"] for e in plain}

            results = dm.search("ds", "WebSocketHandler", top_k=2, use_reranker=True, reranker_top_k=2)
            assert results, "reranked search still returns hits"
            first = results[0]

            # The displayed score IS the reranker relevance — and says so.
            assert first["score"] == RERANK_TOP
            assert first["score_kind"] == "reranker"
            assert first["reranker_score"] == RERANK_TOP

            # The pre-rerank score stays visible: RRF rank arithmetic, carried
            # as retrieval_score rather than posing as an embedding cosine.
            assert "retrieval_score" in first
            assert first["retrieval_score"] == pre_rerank[first["content"]["text"]]
            # embedding_score, when the hybrid lane carries one, is the TRUE
            # recomputed dense cosine — never a stand-in for the RRF sum.
            if "embedding_score" in first:
                assert first["embedding_score"] != first["retrieval_score"]

            # Private lane stamps never travel in the public content payload.
            content = first["content"]
            assert "_score_kind" not in content
            assert "_retrieval_score" not in content
            assert "_reranker_score" not in content
            assert "_embedding_score" not in content
        finally:
            rig.close()


def test_hybrid_hit_without_rerank_is_labelled_rrf() -> None:
    with _env(RAG_HYBRID_SEARCH="1"):
        rig = _Rig()
        try:
            rig.ingest(DOCS)
            dm = _dm(rig)
            results = dm.search("ds", "WebSocketHandler", top_k=2)
            assert results
            for entry in results:
                assert entry["score_kind"] == "rrf"
                assert "_score_kind" not in entry["content"]
                # No rerank ran, so there is no pre-rerank/reranker breakdown.
                assert "retrieval_score" not in entry
                assert "reranker_score" not in entry
        finally:
            rig.close()


def test_dense_only_hit_is_labelled_cosine() -> None:
    with _env(RAG_HYBRID_SEARCH="0"):
        rig = _Rig()
        try:
            rig.ingest(DOCS)
            dm = _dm(rig)
            results = dm.search("ds", "WebSocketHandler", top_k=2)
            assert results
            for entry in results:
                assert entry["score_kind"] == "cosine"
                assert "_score_kind" not in entry["content"]
        finally:
            rig.close()


# ---------------------------------------------------------------------------
# Federated REST path (POST /api/search)
# ---------------------------------------------------------------------------


class _FederatedDM:
    """Duck-typed manager over ONE rig whose `search` is the REAL
    `DatasetManager.search` — so the federated assertions pin the whole
    contract, not a restatement of it."""

    def __init__(self, rig: _Rig) -> None:
        self._rig = rig
        self._dm = _dm(rig)

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        if name != "ds":
            raise FileNotFoundError(name)
        return {"name": "ds", "document_count": 2, "has_password": False}

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self.get_dataset("ds")]

    def has_password(self, name: str) -> bool:
        return False

    def _get_rag(self, name: str, check_embedder: bool = True) -> Any:
        return self._rig.rag

    def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return DatasetManager.search(self._dm, *args, **kwargs)


def test_federated_rest_labels_reranked_hit() -> None:
    with _env(RAG_HYBRID_SEARCH="1"):
        rig = _Rig()
        try:
            rig.ingest(DOCS)
            rig.rag.reranker = _StubReranker()  # type: ignore[assignment]
            dm = _FederatedDM(rig)
            out = asyncio.run(
                _federated_rest_search(
                    dm,
                    ["ds"],
                    "WebSocketHandler",
                    top_k=2,
                    use_reranker=True,
                    reranker_top_k=2,
                    is_unlocked=lambda name: False,
                )
            )
            assert out["results"]
            top = out["results"][0]
            assert top["score_kind"] == "reranker"
            assert top["score"] == top["reranker_score"] == RERANK_TOP
            for entry in out["results"]:
                assert "reranker_score" in entry
                assert "retrieval_score" in entry, "pre-rerank RRF score must survive the merge"
                assert "_score_kind" not in entry["content"]
                assert "_retrieval_score" not in entry["content"]
        finally:
            rig.close()


class _CannedDM:
    """Manager whose per-dataset search returns canned entries shaped exactly
    like the real `DatasetManager.search` now emits them: `score_kind` as
    an entry field, no private stamps inside `content`."""

    def __init__(self, pools: dict[str, list[dict[str, Any]]]) -> None:
        self._pools = pools

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        if name not in self._pools:
            raise FileNotFoundError(name)
        return {"name": name, "document_count": len(self._pools[name]), "has_password": False}

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self.get_dataset(name) for name in self._pools]

    def has_password(self, name: str) -> bool:
        return False

    def search(
        self,
        dataset_name: str,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return list(self._pools[dataset_name])


def test_federated_rest_keeps_lane_aware_merge_with_stripped_stamp() -> None:
    """`dm.search` no longer leaks `_score_kind` in content, so the
    federated fan-out must carry the lane label itself or the lane-aware merge
    silently degrades to a raw score sort (dense cosines burying RRF sums)."""
    dense = [
        {"content": {"text": "d1", "source": "/dense/1.md"}, "score": 0.95, "score_kind": "cosine"},
        {"content": {"text": "d2", "source": "/dense/2.md"}, "score": 0.90, "score_kind": "cosine"},
    ]
    hybrid = [
        {"content": {"text": "h1", "source": "/hybrid/1.md"}, "score": 0.60, "score_kind": "rrf"},
        {"content": {"text": "h2", "source": "/hybrid/2.md"}, "score": 0.50, "score_kind": "rrf"},
    ]
    dm = _CannedDM({"dense_ds": dense, "hybrid_ds": hybrid})
    out = asyncio.run(
        _federated_rest_search(dm, ["dense_ds", "hybrid_ds"], "q", top_k=2, is_unlocked=lambda name: False)
    )
    results = out["results"]
    assert len(results) == 4
    # The two lane tops interleave; a raw score sort would put d1,d2 first.
    assert {r["content"]["text"] for r in results[:2]} == {"d1", "h1"}
    labels = {r["content"]["text"]: r["score_kind"] for r in results}
    assert labels == {"d1": "cosine", "d2": "cosine", "h1": "rrf", "h2": "rrf"}
    # The private lane stamp stays internal; scores are unchanged.
    assert all("_score_kind" not in r["content"] for r in results)
    assert all("_retrieval_score" not in r["content"] for r in results)
    assert {r["content"]["text"]: r["score"] for r in results} == {
        "d1": 0.95,
        "d2": 0.9,
        "h1": 0.6,
        "h2": 0.5,
    }
