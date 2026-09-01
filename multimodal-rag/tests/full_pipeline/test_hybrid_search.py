"""Offline tests for hybrid dense + BM25 retrieval with RRF fusion (roadmap feature 2).

Covers:

  * ``utils/bm25.py`` math on a toy corpus (tokenizer-free, explicit tf/df
    maps): idf ordering, doc-length normalisation, tf saturation, unknown-term
    smoothing, query weights = idf, sparse-vector shape (sorted unique
    indices, positive values, deterministic).
  * ``.bm25_stats.json`` sidecar mechanics: locked record/forget, mtime-cache
    invalidation, reset.
  * Collection schema: new collections get the named ``dense`` vector + named
    ``bm25`` sparse vector; legacy unnamed-vector collections are detected as
    such (``vector_name=None``, never hybrid).
  * End-to-end on embedded local Qdrant (``:memory:``): ingest through
    ``MultimodalRAG.aadd_to_vector_store`` writes BM25 sparse vectors for
    real-text docs (skipping bare media placeholders, keeping caption text),
    maintains the df sidecar, and text retrieval runs an actual RRF fusion
    request — a lexically-unique term retrieves its document even though the
    stub dense embedder carries no lexical signal.
  * ``RAG_HYBRID_SEARCH=0`` forces dense-only (no sparse vectors at ingest).
  * Fusion-unsupported backends degrade gracefully to dense-only once;
    unrelated errors propagate.

Local mode (qdrant-client 1.19) DOES implement prefetch + Fusion.RRF — these
tests exercise the real fusion path.  The graceful-degradation test simulates
a backend that does not.

No model endpoint required — the embedder is stubbed.

Run::

    python tests/full_pipeline/test_hybrid_search.py    # standalone
    pytest tests/full_pipeline/test_hybrid_search.py    # under pytest
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    SparseVector,
    VectorParams,
)

from multimodal_rag.dataset_manager import DATASET_SCHEMA_VERSION, DatasetManager
from multimodal_rag.rag_system import MultimodalRAG, _bm25_indexable_text
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils import metrics
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings
from multimodal_rag.vector_store import QdrantVectorStore, _is_fusion_unsupported_error

COLL = "hybrid_search_test"
DIM = 8

IMG = "data:image/jpeg;base64,AA=="


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class _StubEmbedderModel:
    """Minimal stand-in for EmbeddingModel with a deterministic stub embedder.

    The hash "embedding" carries NO lexical signal (sha256 of the text) —
    dense scores between unrelated texts are uncorrelated noise, which is
    exactly the regime where the BM25 lane must carry the retrieval.
    """

    allowable_modalities: tuple[str, ...] = ("text", "image", "video")
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
    def _hash_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:DIM]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        out = []
        for d in docs:
            text = d if isinstance(d, str) else (d.get("text") or "")
            out.append(self._hash_vector(text))
        return out

    async def _aembed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._hash_vector(text)

    def _embed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._hash_vector(text)


class _HybridRig:
    """A bm25-capable store + rag on embedded local Qdrant with a real sidecar file.

    The sidecar lives at ``<tmp>/ds/files/.bm25_stats.json`` — the same
    dataset-dir layout DatasetManager computes — so the DatasetManager shell
    below resolves the identical path.
    """

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.emb = _StubEmbedderModel()
        self.client = QdrantClient(":memory:")
        self.files_dir = Path(self.tmp.name) / "ds" / "files"
        self.stats_path = self.files_dir / bm25_lane.BM25_STATS_FILENAME
        self.store: QdrantVectorStore = MultimodalRAG._build_qdrant_vector_store(  # type: ignore[assignment]
            embedding=self.emb.model,
            client=self.client,
            collection_name=COLL,
            bm25_stats_path=str(self.stats_path),
        )
        self.rag = MultimodalRAG(embedder=self.emb, vector_store=self.store)  # type: ignore[arg-type]

    def close(self) -> None:
        self.tmp.cleanup()


def _ingest(rig: _HybridRig, docs: list[dict[str, Any]]) -> list[str]:
    return asyncio.run(rig.rag.aadd_to_vector_store(docs, deduplicate=False))


def _sparse_of(client: QdrantClient, coll: str) -> dict[str, Any]:
    """Stored points with their named vectors: {point_id: vector_dict}."""
    records = client.retrieve(collection_name=coll, ids=_all_ids(client, coll), with_payload=True, with_vectors=True)
    return {str(r.id): r.vector for r in records}


def _all_ids(client: QdrantClient, coll: str) -> list[Any]:
    ids: list[Any] = []
    offset: Any = None
    while True:
        pts, offset = client.scroll(coll, limit=100, offset=offset, with_payload=False, with_vectors=False)
        ids.extend(p.id for p in pts)
        if offset is None:
            break
    return ids


def _texts_of(rig: _HybridRig) -> dict[str, str]:
    return {
        str(r.id): (r.payload or {}).get("page_content", "")
        for r in rig.client.retrieve(
            collection_name=COLL, ids=_all_ids(rig.client, COLL), with_payload=True, with_vectors=False
        )
    }


def _dm_for_rig(rig: _HybridRig) -> DatasetManager:
    """DatasetManager shell wired to the rig (house pattern from the other tests).

    ``datasets_path`` points at the rig's tmp root so
    ``_bm25_stats_path("ds")`` resolves to the rig's own sidecar.
    """
    dm = DatasetManager.__new__(DatasetManager)
    dm._get_rag = lambda ds, check_embedder=True: rig.rag  # type: ignore[method-assign]
    dm.datasets_path = Path(rig.tmp.name)  # type: ignore[attr-defined]
    return dm


# ---------------------------------------------------------------------------
# bm25.py math on a toy corpus (tokenizer-free)
# ---------------------------------------------------------------------------

# Toy corpus stats: 4 docs, avgdl 10, with a common/mid/rare term split.
TOY = {"n_docs": 4, "total_len": 40, "df": {"common": 4, "mid": 2, "rare": 1}}


def test_idf_orders_by_rarity():
    common, mid, rare = (bm25_lane.idf(TOY, t) for t in ("common", "mid", "rare"))
    assert 0 < common < mid < rare
    # Unknown term (df=0) gets the maximum idf the smoothed formula yields —
    # a brand-new identifier must stay searchable.
    assert rare < bm25_lane.idf(TOY, "brand_new_term")


def test_doc_weights_rank_rare_terms_higher():
    w = bm25_lane.bm25_doc_weights({"common": 1, "rare": 1}, TOY)
    assert w["rare"] > w["common"] > 0.0


def test_doc_weights_saturate_with_tf():
    w1 = bm25_lane.bm25_doc_weights({"t": 1}, TOY)["t"]
    w2 = bm25_lane.bm25_doc_weights({"t": 2}, TOY)["t"]
    w4 = bm25_lane.bm25_doc_weights({"t": 4}, TOY)["t"]
    assert w1 < w2 < w4, "more occurrences must score higher"
    assert (w2 - w1) > (w4 - w2), "…but with diminishing returns (tf saturation)"


def test_doc_weights_normalise_by_doc_length():
    # Same term, same tf, longer document (extra filler terms) → lower weight.
    short = bm25_lane.bm25_doc_weights({"t": 1}, TOY)["t"]
    long_doc = bm25_lane.bm25_doc_weights({"t": 1, "filler_a": 5, "filler_b": 5}, TOY)["t"]
    assert long_doc < short, "longer documents must be down-weighted for the same tf"


def test_unknown_term_gets_positive_weight():
    w = bm25_lane.bm25_doc_weights({"never_seen": 1}, TOY)
    assert w["never_seen"] > 0


def test_query_weights_are_idf_only():
    tf = {"rare": 3, "common": 1}
    qw = bm25_lane.bm25_query_weights(tf, TOY)
    assert qw == {"rare": bm25_lane.idf(TOY, "rare"), "common": bm25_lane.idf(TOY, "common")}


def test_to_sparse_vector_shape_and_determinism():
    v1 = bm25_lane.to_sparse_vector({"alpha": 1.5, "beta": 0.5, "gamma": 2.0})
    v2 = bm25_lane.to_sparse_vector({"gamma": 2.0, "alpha": 1.5, "beta": 0.5})
    assert isinstance(v1, SparseVector)
    assert v1.indices == sorted(v1.indices) and len(set(v1.indices)) == len(v1.indices)
    assert len(v1.indices) == len(v1.values) == 3
    assert all(v > 0 for v in v1.values)
    assert v1.indices == v2.indices and v1.values == v2.values, "term→dim hashing must be order-independent"


def test_build_query_sparse_vector_empty_is_none():
    assert bm25_lane.build_query_sparse_vector("", TOY) is None
    assert bm25_lane.build_query_sparse_vector("   ", TOY) is None
    vec = bm25_lane.build_query_sparse_vector("rare", TOY)
    assert isinstance(vec, SparseVector) and len(vec.indices) == 1


def test_term_counts_lowercase_and_aggregate():
    tf = bm25_lane.term_counts("Hello hello WORLD world")
    assert tf == {"hello": 2, "world": 2}


def test_env_knobs():
    with _env(RAG_HYBRID_SEARCH="0"):
        assert bm25_lane.hybrid_search_enabled() is False
    with _env(RAG_HYBRID_SEARCH="true"):
        assert bm25_lane.hybrid_search_enabled() is True
    assert bm25_lane.hybrid_search_enabled() is True  # default on
    with _env(RAG_BM25_K1="2.0", RAG_BM25_B="0.4"):
        assert bm25_lane.bm25_k1() == 2.0 and bm25_lane.bm25_b() == 0.4
    assert bm25_lane.bm25_k1() == 1.5 and bm25_lane.bm25_b() == 0.75  # defaults
    with _env(RAG_BM25_K1="garbage"):
        assert bm25_lane.bm25_k1() == 1.5, "unparseable values fall back to the default"


def test_bm25_indexable_text_rules():
    assert _bm25_indexable_text("[Image: photo.jpg]") == ""
    assert _bm25_indexable_text("[Video: v.mp4] [0s – 32s]") == ""
    # Caption text IS indexable — it is the caption twin's only content.
    assert "red car" in _bm25_indexable_text("[Image: photo.jpg]\n[Image description]: a red car")
    # Real extracted text always is.
    assert "checksum" in _bm25_indexable_text("def compute_checksum(): ...\n[Image description]: diagram")


# ---------------------------------------------------------------------------
# .bm25_stats.json sidecar mechanics
# ---------------------------------------------------------------------------


def test_stats_record_and_forget_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"alpha": 2, "beta": 1}, {"alpha": 1}])
        stats = bm25_lane.load_stats(path)
        assert stats["n_docs"] == 2
        assert stats["total_len"] == 4
        assert stats["df"] == {"alpha": 2, "beta": 1}

        bm25_lane.forget_documents(path, [{"alpha": 2, "beta": 1}])
        stats = bm25_lane.load_stats(path)
        assert stats["n_docs"] == 1 and stats["df"] == {"alpha": 1}

        # Forgetting terms the stats never counted must not go negative or
        # invent entries (legacy points, lost sidecar).
        bm25_lane.forget_documents(path, [{"ghost_term": 3}])
        stats = bm25_lane.load_stats(path)
        assert stats["df"] == {"alpha": 1} and stats["n_docs"] == 0


def test_stats_cache_invalidated_on_external_write():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".bm25_stats.json"
        bm25_lane.record_documents(path, [{"alpha": 1}])
        assert bm25_lane.load_stats(path)["n_docs"] == 1
        # Another pod writes under its lock — bump the mtime explicitly so
        # the change is visible even within the same clock tick.
        path.write_text(json.dumps({"n_docs": 7, "total_len": 7, "df": {}}), encoding="utf-8")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
        assert bm25_lane.load_stats(path)["n_docs"] == 7, "mtime change must invalidate the read cache"


def test_reset_stats_removes_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".bm25_stats.json"
        bm25_lane.record_documents(path, [{"alpha": 1}])
        bm25_lane.reset_stats(path)
        assert not path.exists()
        assert bm25_lane.load_stats(path)["n_docs"] == 0


# ---------------------------------------------------------------------------
# Collection schema
# ---------------------------------------------------------------------------


def test_new_collection_gets_hybrid_schema():
    rig = _HybridRig()
    try:
        params = rig.client.get_collection(COLL).config.params
        assert isinstance(params.vectors, dict) and list(params.vectors) == [bm25_lane.DENSE_VECTOR_NAME]
        assert bm25_lane.BM25_VECTOR_NAME in (params.sparse_vectors or {})
        assert rig.store.vector_name == bm25_lane.DENSE_VECTOR_NAME
        assert rig.store.supports_hybrid() is True
    finally:
        rig.close()


def test_store_reopens_on_existing_hybrid_collection():
    """A restart must re-detect the named-vector schema (not assume legacy)."""
    rig = _HybridRig()
    try:
        reopened = MultimodalRAG._build_qdrant_vector_store(
            embedding=rig.emb.model,
            client=rig.client,
            collection_name=COLL,
            bm25_stats_path=str(rig.stats_path),
        )
        assert reopened.vector_name == bm25_lane.DENSE_VECTOR_NAME
        assert reopened.supports_hybrid() is True
    finally:
        rig.close()


def test_legacy_unnamed_vector_collection_is_not_hybrid():
    client = QdrantClient(":memory:")
    try:
        client.create_collection("legacy_coll", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
        client.upsert(
            "legacy_coll",
            [
                PointStruct(
                    id=uuid.uuid4().hex, vector=[0.1] * DIM, payload={"page_content": "plain text", "metadata": {}}
                )
            ],
            wait=True,
        )
        vs = QdrantVectorStore(client, "legacy_coll", embedding=_StubEmbedderModel().model)
        assert vs.vector_name is None
        assert vs.supports_hybrid() is False, "unnamed default vector cannot host a second lane"

        results = asyncio.run(vs.asimilarity_search_with_relevance_scores("plain", 5))
        assert len(results) == 1
        assert results[0][0].page_content == "plain text"

        # The built request stays the flat dense form even when hybrid is forced on.
        req, is_hybrid = vs._query_requests([([0.1] * DIM, 5, True, None, "plain")], hybrid=True)[0]
        assert is_hybrid is False and req.prefetch is None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Ingest: sparse vectors + df stats
# ---------------------------------------------------------------------------


def test_ingest_writes_sparse_vectors_and_df_stats():
    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "def compute_checksum(payload): return sha256(payload)", "source": "/tmp/x/checksum.py"},
                {"text": "The quarterly report shows steady revenue growth.", "source": "/tmp/x/report.pdf"},
                # Bare media placeholder → no lexical lane for this point…
                {"text": "[Image: photo.jpg]", "image": IMG, "source": "/tmp/x/photo.jpg"},
                # …but caption text IS indexed (caption twin's only content).
                {
                    "text": "[Image: cat.jpg]\n[Image description]: a red cat on a sofa",
                    "image": IMG,
                    "source": "/tmp/x/cat.jpg",
                },
            ],
        )

        vectors = _sparse_of(rig.client, COLL)
        texts = _texts_of(rig)
        assert len(vectors) == 5, "4 base docs + 1 caption twin"

        indexed, bare = [], []
        for pid, vec in vectors.items():
            if "photo.jpg" in texts[pid]:
                bare.append(pid)
                assert "bm25" not in vec, "bare media placeholder must not get a lexical lane"
            else:
                indexed.append(pid)
                assert "bm25" in vec, f"real-text point {texts[pid]!r} must carry a sparse vector"
                sparse = vec["bm25"]
                assert sparse.indices and all(v > 0 for v in sparse.values)
                assert sparse.indices == sorted(sparse.indices)
        assert len(indexed) == 4 and len(bare) == 1

        # df sidecar: one entry per indexable doc (placeholder excluded,
        # caption twin counted — it stores its own sparse vector).
        stats = bm25_lane.load_stats(rig.stats_path)
        assert stats["n_docs"] == 4
        assert stats["total_len"] > 0 and stats["df"], "df counts must be persisted"
        assert "cat" in stats["df"] or "red" in stats["df"], "caption terms must be counted"
    finally:
        rig.close()


def test_ingest_hybrid_disabled_by_env():
    rig = _HybridRig()
    try:
        with _env(RAG_HYBRID_SEARCH="0"):
            _ingest(rig, [{"text": "plain text with words", "source": "/tmp/x/a.txt"}])
        vectors = _sparse_of(rig.client, COLL)
        assert len(vectors) == 1 and "bm25" not in vectors[next(iter(vectors))]
        assert bm25_lane.load_stats(rig.stats_path)["n_docs"] == 0, "env-off ingest must not touch the df stats"
    finally:
        rig.close()


def test_forget_bm25_documents_on_delete():
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "unique_identifier alpha beta", "source": "/tmp/x/a.txt"}])
        stats = bm25_lane.load_stats(rig.stats_path)
        assert stats["n_docs"] == 1

        dm = _dm_for_rig(rig)
        pid = _all_ids(rig.client, COLL)[0]
        dm.forget_bm25_documents("ds", [pid])
        stats = bm25_lane.load_stats(rig.stats_path)
        assert stats["n_docs"] == 0 and stats["df"] == {}, "deleted points must release their df counts"

        # And it must be a silent no-op for legacy collections.
        dm2 = DatasetManager.__new__(DatasetManager)
        legacy_vs = QdrantVectorStore(QdrantClient(":memory:"), "legacy", embedding=None)
        rag = type("_RagStub", (), {"vector_store": legacy_vs})()
        dm2._get_rag = lambda ds, check_embedder=True: rag  # type: ignore[method-assign]
        dm2.datasets_path = Path(rig.tmp.name)  # type: ignore[attr-defined]
        dm2.forget_bm25_documents("ds", [uuid.uuid4().hex])  # must not raise
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Query: RRF fusion end-to-end
# ---------------------------------------------------------------------------


def test_fusion_request_shape():
    rig = _HybridRig()
    try:
        # _query_requests skips the sparse lane on empty df stats — seed them.
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        req, is_hybrid = rig.store._query_requests([([0.2] * DIM, 7, True, None, "checksum payload")], hybrid=True)[0]
        assert is_hybrid is True
        assert isinstance(req.query, FusionQuery) and req.query.fusion == Fusion.RRF
        assert req.limit == 7 and req.prefetch is not None and len(req.prefetch) == 2
        dense_pf, sparse_pf = req.prefetch
        assert dense_pf.using == bm25_lane.DENSE_VECTOR_NAME and dense_pf.limit == 7
        assert sparse_pf.using == bm25_lane.BM25_VECTOR_NAME
        assert isinstance(sparse_pf.query, SparseVector)
        # No query text (multimodal query) → flat dense request.
        req2, is_hybrid2 = rig.store._query_requests([([0.2] * DIM, 7, True, None, None)], hybrid=True)[0]
        assert is_hybrid2 is False and req2.using == bm25_lane.DENSE_VECTOR_NAME
        # Filters are pushed into BOTH prefetches so each lane searches only
        # the filtered subset.
        flt_q, flt_h = rig.store._query_requests([([0.2] * DIM, 5, True, {"file_types": ["log"]}, "err")], hybrid=True)[
            0
        ]
        assert flt_h is True
        assert all(pf.filter is not None for pf in flt_q.prefetch)
    finally:
        rig.close()


def test_hybrid_query_returns_fused_results():
    """The real fusion path: a lexically-unique term retrieves its document.

    Dense scores from the hash embedder are uncorrelated noise, so only the
    BM25 lane can rank the matching document — and with ≥3 docs the sparse
    hit's RRF bonus is strictly larger than any other document's best
    possible dense-only contribution, so it must come first.
    """
    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
                {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
                {"text": "Quarterly revenue figures for the fiscal year", "source": "/tmp/x/rev.txt"},
            ],
        )
        results = asyncio.run(rig.rag.aretrieve("WebSocketHandler", top_k=3))
        assert results, "hybrid query must return results"
        top_text = results[0][0]["text"]
        assert "WebSocketHandler" in top_text, f"lexical hit must win the fusion, got: {top_text!r}"
    finally:
        rig.close()


def test_hybrid_query_counts_metric():
    if not metrics.AVAILABLE:
        return  # prometheus_client absent — counter is a no-op
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])
        before = metrics._REGISTRY.get_sample_value("rag_search_hybrid_total", {"mode": "hybrid"}) or 0.0
        asyncio.run(rig.rag.aretrieve("some text", top_k=3))
        after = metrics._REGISTRY.get_sample_value("rag_search_hybrid_total", {"mode": "hybrid"}) or 0.0
        assert after > before, "hybrid query must be counted"
    finally:
        rig.close()


def test_hybrid_search_disabled_by_env_forces_dense_only():
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"}])
        with _env(RAG_HYBRID_SEARCH="0"):
            results = asyncio.run(rig.rag.aretrieve("WebSocketHandler", top_k=3))
        assert len(results) == 1, "dense-only fallback must still search fine"
        req, is_hybrid = rig.store._query_requests([([0.2] * DIM, 5, True, None, "q")], hybrid=False)[0]
        assert is_hybrid is False and req.prefetch is None
    finally:
        rig.close()


def test_unsupported_fusion_degrades_gracefully():
    """A backend without the fusion API: one loud fallback, then dense-only."""
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])
        shapes: list[bool] = []  # per query_batch_points call: fusion request?
        real = rig.client.query_batch_points

        def flaky(**kwargs):
            reqs = kwargs.get("requests") or []
            shapes.append(any(getattr(r, "prefetch", None) for r in reqs))
            if len(shapes) == 1:
                raise ValueError("Fusion RRF is not supported by this backend")
            return real(**kwargs)

        rig.client.query_batch_points = flaky  # type: ignore[method-assign]
        results = asyncio.run(rig.store.asimilarity_search_with_relevance_scores("some text", 5))
        assert len(results) == 1, "first (unsupported) attempt must fall back and still return results"
        assert rig.store._fusion_supported is False
        assert shapes == [True, False], "fusion attempt must be retried dense-only within the same search"

        # Subsequent searches skip the fusion attempt entirely.
        results = asyncio.run(rig.store.asimilarity_search_with_relevance_scores("some text", 5))
        assert len(results) == 1
        assert shapes == [True, False, False], "after the fallback the store must not retry fusion per query"
    finally:
        rig.close()


def test_unrelated_query_errors_propagate():
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])

        def broken(**kwargs):
            raise ConnectionError("connection refused")

        rig.client.query_batch_points = broken  # type: ignore[method-assign]
        try:
            asyncio.run(rig.store.asimilarity_search_with_relevance_scores("some text", 5))
        except ConnectionError:
            pass
        else:
            raise AssertionError("unrelated failures must propagate, not be swallowed as 'unsupported'")
    finally:
        rig.close()


def test_fusion_unsupported_error_classifier():
    assert _is_fusion_unsupported_error(ValueError("Fusion RRF is not supported"))
    assert _is_fusion_unsupported_error(ValueError("Unknown prefetch type"))
    assert _is_fusion_unsupported_error(NotImplementedError("local mode"))
    assert not _is_fusion_unsupported_error(ConnectionError("connection refused"))
    assert not _is_fusion_unsupported_error(ValueError("Point id is not a valid UUID"))


# ---------------------------------------------------------------------------
# schema_version guard nudge
# ---------------------------------------------------------------------------


def test_schema_nudge_warns_for_legacy_meta():
    rig = _HybridRig()
    try:
        dm = _dm_for_rig(rig)

        class _Capture(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.records: list[str] = []

            def emit(self, record: Any) -> None:
                self.records.append(record.getMessage())

        handler = _Capture()
        from multimodal_rag.utils.logging_utils import logging as rag_logging

        logger_obj = rag_logging.getLogger("multimodal_rag.dataset_manager")
        logger_obj.addHandler(handler)
        try:
            # v1 dataset (no schema_version) → warned, but never raised.
            dm._write_meta("legacy_ds", {"name": "legacy_ds", "document_count": 3})
            dm._nudge_schema_upgrade("legacy_ds")
            assert any("hybrid dense+BM25" in msg for msg in handler.records)
            # Current-schema dataset → silent.
            handler.records.clear()
            dm._write_meta("fresh_ds", {"name": "fresh_ds", "schema_version": DATASET_SCHEMA_VERSION})
            dm._nudge_schema_upgrade("fresh_ds")
            assert handler.records == []
            # Missing meta entirely → silent.
            dm._nudge_schema_upgrade("ghost_ds")
            assert handler.records == []
        finally:
            logger_obj.removeHandler(handler)
    finally:
        rig.close()


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
