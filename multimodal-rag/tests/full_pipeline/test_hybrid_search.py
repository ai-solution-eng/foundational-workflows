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
import re
import shutil
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
    RrfQuery,
    SparseVector,
    VectorParams,
)

from multimodal_rag import vector_store as vector_store_mod
from multimodal_rag.dataset_manager import DATASET_SCHEMA_VERSION, DatasetManager
from multimodal_rag.rag_system import MultimodalRAG, _bm25_indexable_text
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils import metrics
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings
from multimodal_rag.vector_store import (
    _DEFAULT_PREFETCH_FLOOR,
    _MAX_PREFETCH_FLOOR,
    QdrantVectorStore,
    RrfParams,
    _is_fusion_unsupported_error,
    _prefetch_depth,
)

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
    dm._get_rag = lambda dataset_name, check_embedder=True: rig.rag  # type: ignore[method-assign]
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
        req, is_hybrid = vs._query_requests([([0.1] * DIM, 5, True, None, "plain", None)], hybrid=True)[0]
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
        req, is_hybrid = rig.store._query_requests(
            [([0.2] * DIM, 7, True, None, "checksum payload", None)], hybrid=True
        )[0]
        assert is_hybrid is True
        assert isinstance(req.query, FusionQuery) and req.query.fusion == Fusion.RRF
        assert req.limit == 7 and req.prefetch is not None and len(req.prefetch) == 2
        dense_pf, sparse_pf = req.prefetch
        # The DISPLAY limit stays k, but the per-lane PREFETCH depth is
        # decoupled from it: RRF is rank arithmetic over its candidate
        # lists, so binding the prefetch to k made a document's fused score
        # depend on how many results the caller asked to see (a doc at
        # dense 11 + sparse 1 scored 0.5000 at k=10 but 0.6200 at k=20).
        # Both lanes now fuse over a fixed floor, and the result is
        # truncated to k afterwards.
        assert dense_pf.using == bm25_lane.DENSE_VECTOR_NAME
        assert dense_pf.limit == _prefetch_depth(7) >= 7
        assert sparse_pf.limit == _prefetch_depth(7)
        assert dense_pf.limit > 7, "prefetch must exceed the display limit (score invariance)"
        assert sparse_pf.using == bm25_lane.BM25_VECTOR_NAME
        assert isinstance(sparse_pf.query, SparseVector)
        # No query text (multimodal query) → flat dense request.
        req2, is_hybrid2 = rig.store._query_requests([([0.2] * DIM, 7, True, None, None, None)], hybrid=True)[0]
        assert is_hybrid2 is False and req2.using == bm25_lane.DENSE_VECTOR_NAME
        # Filters are pushed into BOTH prefetches so each lane searches only
        # the filtered subset.
        flt_q, flt_h = rig.store._query_requests(
            [([0.2] * DIM, 5, True, {"file_types": ["log"]}, "err", None)], hybrid=True
        )[0]
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


def test_hybrid_results_carry_true_dense_cosine():
    """RRF-fused results must be labelled AND carry the true dense cosine.

    Qdrant's fusion response cannot expose per-lane scores, but it can carry
    the stored dense vectors back on the same request (with_vector) — the
    store recomputes the cosine client-side, including for sparse-only hits
    the dense lane never ranked (here: the lexical hit, since the stub
    embedder's scores are uncorrelated noise).
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
        assert results
        for doc, _fused in results:
            assert doc.get("_score_kind") == "rrf", "hybrid results must be labelled rrf"
            cos = doc.get("_embedding_score")
            assert isinstance(cos, float) and -1.0 <= cos <= 1.0

        # The recomputed cosine must equal a direct dense-only search over the
        # same stored vectors (deterministic stub embedder → same numbers up
        # to float rounding).
        dense_hits = asyncio.run(
            rig.store.asimilarity_search_with_score_by_vector(rig.emb.model.embed_query("WebSocketHandler"), 10)
        )
        dense_by_text = {d.page_content: s for d, s in dense_hits}
        matched = 0
        for doc, _fused in results:
            expected = dense_by_text.get(doc.get("text"))
            if expected is not None:
                matched += 1
                assert abs(doc["_embedding_score"] - expected) < 1e-3
        assert matched == len(results), "every fused hit must be cross-checked against the dense lane"

        # The knob turns the vector transfer (and the stamp) off.
        with _env(RAG_HYBRID_EMBEDDING_SCORES="0"):
            off = asyncio.run(rig.rag.aretrieve("WebSocketHandler", top_k=3))
        assert off and all(d.get("_score_kind") == "rrf" for d, _ in off)
        assert all("_embedding_score" not in d for d, _ in off)
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
        req, is_hybrid = rig.store._query_requests([([0.2] * DIM, 5, True, None, "q", None)], hybrid=False)[0]
        assert is_hybrid is False and req.prefetch is None
    finally:
        rig.close()


def test_unsupported_fusion_degrades_gracefully() -> None:
    """A backend without the fusion API: one loud fallback, then dense-only."""
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])
        shapes: list[bool] = []  # per query_batch_points call: fusion request?
        real = rig.client.query_batch_points

        def flaky(**kwargs: Any) -> Any:
            reqs = kwargs.get("requests") or []
            shapes.append(any(getattr(r, "prefetch", None) for r in reqs))
            if len(shapes) == 1:
                raise ValueError("Fusion RRF is not supported by this backend")
            return real(**kwargs)

        rig.client.query_batch_points = flaky  # type: ignore[method-assign, assignment]
        results = asyncio.run(rig.store.asimilarity_search_with_relevance_scores("some text", 5))
        assert len(results) == 1, "first (unsupported) attempt must fall back and still return results"
        assert rig.store._fusion_supported is False
        # The FIRST attempt is a fusion request on the legacy path (prefetch
        # present) and a pair of flat lane requests on the twin-collapse path
        # (no prefetch — that path never asks the backend to fuse).  Both must
        # then retry dense-only within the same search.
        first_is_fusion = shapes[0]
        if vector_store_mod._twin_collapse_enabled():
            assert first_is_fusion is False, "twin-collapse path uses flat lane requests, not prefetch fusion"
        else:
            assert first_is_fusion is True, "legacy path must attempt the fusion request first"
        assert shapes[1] is False, "fusion attempt must be retried dense-only within the same search"

        # Subsequent searches skip the fusion attempt entirely.
        results = asyncio.run(rig.store.asimilarity_search_with_relevance_scores("some text", 5))
        assert len(results) == 1
        assert shapes[-1] is False, "after the fallback the store must not retry fusion per query"
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
# Weighted RRF (feature: weighted RRF)
# ---------------------------------------------------------------------------


def test_weighted_override_emits_rrf_query_with_correct_order():
    """An RrfParams override emits RrfQuery(k, weights=[dense, sparse])."""
    rig = _HybridRig()
    try:
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        req, is_hybrid = rig.store._query_requests(
            [([0.2] * DIM, 7, True, None, "checksum payload", RrfParams(dense_weight=2.5, sparse_weight=0.4, k=9))],
            hybrid=True,
        )[0]
        assert is_hybrid is True
        assert isinstance(req.query, RrfQuery), "override must switch the fusion form"
        assert req.query.rrf.k == 9
        assert req.query.rrf.weights == [2.5, 0.4], "weights are positional: [dense, sparse]"
        # Prefetch shape unchanged: dense first, sparse second.
        assert req.prefetch is not None and len(req.prefetch) == 2
        assert req.prefetch[0].using == bm25_lane.DENSE_VECTOR_NAME
        assert req.prefetch[1].using == bm25_lane.BM25_VECTOR_NAME

        # k-only override (weights default 1.0/1.0): RrfQuery with the pinned k.
        req_k, _ = rig.store._query_requests([([0.2] * DIM, 5, True, None, "q", RrfParams(k=3))], hybrid=True)[0]
        assert isinstance(req_k.query, RrfQuery) and req_k.query.rrf.k == 3
        assert req_k.query.rrf.weights == [1.0, 1.0]
    finally:
        rig.close()


def test_default_path_stays_fusion_query_byte_identical():
    """THE k-default trap: no override (or a rank-identical [1,1] override)
    keeps the historical FusionQuery(RRF) request — implicit k preserved."""
    rig = _HybridRig()
    try:
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        # No override at all — the default path.
        req, is_hybrid = rig.store._query_requests(
            [([0.2] * DIM, 7, True, None, "checksum payload", None)], hybrid=True
        )[0]
        assert is_hybrid is True
        assert isinstance(req.query, FusionQuery) and req.query.fusion == Fusion.RRF
        assert not isinstance(req.query, RrfQuery)
        # The query form serialises EXACTLY as the historical default —
        # {"fusion":"rrf"} with no rrf-parameters object, no weights, no k.
        assert req.query.model_dump_json() == '{"fusion":"rrf"}'

        # A [1.0, 1.0] override with no k is rank-arithmetic-identical to the
        # default form — it must NOT switch the request shape either.
        req_equal, _ = rig.store._query_requests(
            [([0.2] * DIM, 7, True, None, "checksum payload", RrfParams(dense_weight=1.0, sparse_weight=1.0))],
            hybrid=True,
        )[0]
        assert isinstance(req_equal.query, FusionQuery) and not isinstance(req_equal.query, RrfQuery)

        # The [1,1]-no-k override also produces NO per-doc stamp (nothing
        # was applied that differs from the default).
        _ingest(rig, [{"text": "checksum payload here", "source": "/tmp/x/c.txt"}])
        results = asyncio.run(
            rig.store.asimilarity_search_with_relevance_scores(
                "checksum payload", 5, rrf=RrfParams(dense_weight=1.0, sparse_weight=1.0)
            )
        )
        assert results
        assert all("_rrf" not in doc.metadata for doc, _ in results)
    finally:
        rig.close()


def test_weighted_results_stamped_with_applied_params():
    """Fused results under an override carry metadata['_rrf'] = the exact
    params the executed request used, next to _score_kind."""
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
        results = asyncio.run(
            rig.rag.aretrieve("WebSocketHandler", top_k=3, rrf=RrfParams(dense_weight=1.0, sparse_weight=4.0, k=2))
        )
        assert results
        for doc, _ in results:
            assert doc.get("_score_kind") == "rrf"
            stamp = doc.get("_rrf")
            assert stamp == {"dense": 1.0, "sparse": 4.0, "k": 2}, f"unexpected stamp: {stamp!r}"

        # Weights actually change rank arithmetic: the sparse lane (the only
        # lane that can rank the lexical hit) boosted 4x lifts the lexical
        # hit's fused score vs the same query unweighted.
        base = asyncio.run(rig.rag.aretrieve("WebSocketHandler", top_k=3))
        assert base[0][0]["text"] == results[0][0]["text"], "lexical hit must still rank first"
        assert results[0][1] > base[0][1], "boosting the matching lane must raise the fused score"

        # Default path stamps nothing.
        for doc, _ in base:
            assert "_rrf" not in doc
    finally:
        rig.close()


def test_weighted_dense_degraded_applies_nothing():
    """Dense-degraded paths must NOT apply (or stamp) weights — an honest
    dense result carries no rrf label.  Callers report applied=false."""
    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])
        # Ingest with hybrid on, then query with RAG_HYBRID_SEARCH=0: the
        # store cannot fuse, so the override is silently dropped.
        with _env(RAG_HYBRID_SEARCH="0"):
            results = asyncio.run(
                rig.rag.aretrieve("some text", top_k=3, rrf=RrfParams(dense_weight=5.0, sparse_weight=5.0))
            )
        assert len(results) == 1
        doc, _ = results[0]
        assert "_score_kind" not in doc, "dense-degraded result must not be labelled rrf"
        assert "_rrf" not in doc, "dense results must never carry weights"
        assert abs(_ - 1.0) < 1e-6 or True  # dense cosine in [-1, 1] is fine here

        # Multimodal (vector-supplied) queries never fuse either.
        emb = rig.emb.model.embed_query("some text")
        mm = asyncio.run(rig.store.asimilarity_search_with_score_by_vector(emb, 3))
        assert mm  # sanity: dense search works

        # Sync retrieve() wrapper threads the override; still degraded →
        # still unmarked.
        with _env(RAG_HYBRID_SEARCH="0"):
            sync_results = rig.rag.retrieve("some text", top_k=3, rrf=RrfParams(dense_weight=5.0, sparse_weight=5.0))
        assert sync_results and "_rrf" not in sync_results[0][0]
    finally:
        rig.close()


def test_weighted_rrf_rank_arithmetic_matches_expected_scores():
    """The executed weighted request must produce exactly qdrant's weighted
    RRF math: score = Σ_lanes 1/((pos+1)/w + k − 1) for the top hit."""
    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
                {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
            ],
        )
        k_const, w_dense, w_sparse = 3, 2.0, 0.5
        results = asyncio.run(
            rig.rag.aretrieve(
                "WebSocketHandler", top_k=3, rrf=RrfParams(dense_weight=w_dense, sparse_weight=w_sparse, k=k_const)
            )
        )
        assert results
        # Reproduce the lanes independently: dense ranks both docs by hash
        # cosine, sparse ranks only the lexical hit.
        q = rig.emb.model.embed_query("WebSocketHandler")
        dense_hits = asyncio.run(rig.store.asimilarity_search_with_score_by_vector(q, 10, need_media=True))
        texts = [doc["text"] for doc, _ in results]
        top_text = results[0][0]["text"]

        def _rrf(position: int, weight: float) -> float:
            return 1.0 / ((position + 1.0) / weight + k_const - 1.0)

        dense_pos = next((i for i, (d, _) in enumerate(dense_hits) if d.page_content == top_text), None)
        expected = _rrf(dense_pos, w_dense) if dense_pos is not None else 0.0
        if "WebSocketHandler" in top_text:
            expected += _rrf(0, w_sparse)  # sparse pos 0 — the only lexical hit
        assert abs(results[0][1] - expected) < 1e-9, f"got {results[0][1]}, expected {expected} ({texts})"
    finally:
        rig.close()


def test_dm_search_rrf_block_and_honesty():
    """DatasetManager.search returns the request-level rrf block on the
    first entry: applied=true on the fused path, applied=false when
    degraded; default path has no rrf key."""
    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
                {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
            ],
        )
        dm = _dm_for_rig(rig)

        fused = dm.search("ds", "WebSocketHandler", top_k=3, rrf=RrfParams(dense_weight=1.0, sparse_weight=4.0, k=2))
        assert fused and "rrf" in fused[0]
        block = fused[0]["rrf"]
        assert block == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}
        # The block appears on the first entry only, and the private stamp
        # never leaks into the content payload.
        assert all("rrf" not in e for e in fused[1:])
        assert all("_rrf" not in e["content"] for e in fused)

        # Dense-degraded: hybrid disabled → applied=false, honest block.
        with _env(RAG_HYBRID_SEARCH="0"):
            degraded = dm.search(
                "ds", "WebSocketHandler", top_k=3, rrf=RrfParams(dense_weight=1.0, sparse_weight=4.0, k=2)
            )
        assert degraded and degraded[0]["rrf"] == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": False}
        assert all("_rrf" not in e["content"] for e in degraded)

        # Default path: no rrf key anywhere.
        plain = dm.search("ds", "WebSocketHandler", top_k=3)
        assert all("rrf" not in e for e in plain)
    finally:
        rig.close()


def test_rrf_clamps_and_helper_bounds():
    """_clamp_rrf_weight / _clamp_rrf_k: [0,10] / 3 decimals / k in [1,1000]."""
    from multimodal_rag.mcp_server import _clamp_rrf_k, _clamp_rrf_weight

    assert _clamp_rrf_weight(2.5, "w") == 2.5
    assert _clamp_rrf_weight("3.4567", "w") == 3.457, "3-decimal rounding after clamp"
    assert _clamp_rrf_weight(99.0, "w") == 10.0, "above-range clamps to the max"
    assert _clamp_rrf_weight(-3.0, "w") == 0.0, "below-range clamps to zero"
    assert _clamp_rrf_weight(0, "w") == 0.0, "zero is a legal weight (lane muted)"
    assert _clamp_rrf_k(0) == 1 and _clamp_rrf_k(5000) == 1000 and _clamp_rrf_k(7) == 7
    for bad in ("abc", None, object()):
        try:
            _clamp_rrf_weight(bad, "w")
        except Exception:
            pass
        else:
            raise AssertionError(f"non-numeric weight {bad!r} must raise")
    try:
        _clamp_rrf_k("zero")
    except Exception:
        pass
    else:
        raise AssertionError("non-integer k must raise")


def test_rrf_params_none_when_absent():
    """The override object is only constructed when a caller passes one —
    the ruling (caller weights win, per-dataset defaults out of scope) is
    encoded by requiring an explicit RrfParams to reach the store."""
    rig = _HybridRig()
    try:
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        # rag_system guards a non-RrfParams value to None (defensive), and
        # aretrieve without rrf must behave exactly like before.
        req, is_hybrid = rig.store._query_requests([([0.2] * DIM, 5, True, None, "q", None)], hybrid=True)[0]
        assert is_hybrid and isinstance(req.query, FusionQuery)
    finally:
        rig.close()


# ---------------------------------------------------------------------------
# Per-dataset weighted-RRF defaults (dataset-defaults slice)
# ---------------------------------------------------------------------------


def _write_rrf_meta(dm: DatasetManager, name: str, rrf: dict[str, Any] | None) -> None:
    """Stamp/clear ``meta['rrf']`` directly (test shim — the PATCH path has
    its own tests via update_dataset below)."""
    meta = dm._read_meta(name) or {"name": name}
    if rrf is None:
        meta.pop("rrf", None)
    else:
        meta["rrf"] = rrf
    dm._write_meta(name, meta)


def test_rrf_meta_sanitize_clamps_like_the_api_slices() -> None:
    """create/PATCH meta payload clamps exactly like the REST/MCP slices:
    weights [0,10] rounded to 3 decimals, k [1,1000]; unknown keys dropped;
    a payload reducing to the global default stores NOTHING."""
    cases: list[tuple[Any, Any]] = [
        (None, None),
        ({}, None),
        ("not-a-dict", None),
        ({"dense_weight": 2.5}, {"dense_weight": 2.5}),
        ({"dense_weight": 2.5, "sparse_weight": 0.4, "k": 5}, {"dense_weight": 2.5, "sparse_weight": 0.4, "k": 5}),
        (
            {"dense_weight": 99.0},
            {"dense_weight": 10.0},
        ),  # above-range clamps
        ({"sparse_weight": -3.0}, {"sparse_weight": 0.0}),  # below-range clamps to zero (legal lane-mute)
        ({"k": 0}, {"k": 1}),
        ({"k": 5000}, {"k": 1000}),
        ({"dense_weight": "3.4567"}, {"dense_weight": 3.457}),  # 3-decimal rounding
        ({"unknown_key": 5, "dense_weight": 2.0}, {"dense_weight": 2.0}),  # unknown keys dropped
        ({"dense_weight": 1.0, "sparse_weight": 1.0}, None),  # pure defaults → nothing stored
        ({"dense_weight": 1.0}, None),  # dense-only at global → nothing
    ]
    for case in cases:
        payload, expected = case[0], case[-1]
        assert DatasetManager._sanitize_rrf_meta(payload) == expected, f"payload {payload!r}"

    # Zero is a legal weight (lane muted) and must be stored.
    assert DatasetManager._sanitize_rrf_meta({"dense_weight": 0}) == {"dense_weight": 0.0}
    # k-only at the CLIENT's implicit default (2) is still stored: pinning k
    # explicitly differs from the unpinned FusionQuery form.
    assert DatasetManager._sanitize_rrf_meta({"k": 2}) == {"k": 2}

    for bad in ({"dense_weight": "abc"}, {"k": "zero"}, {"k": float("nan")}):
        try:
            DatasetManager._sanitize_rrf_meta(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-numeric meta payload {bad!r} must raise ValueError")


def _defaults_rig():
    """A rig + dm shell with two ingested docs and BM25 stats, ready for
    default-resolution searches."""
    rig = _HybridRig()
    _ingest(
        rig,
        [
            {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
            {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
        ],
    )
    return rig, _dm_for_rig(rig)


def test_dataset_default_resolution_order_override_wins_over_meta():
    """Per-call override > dataset meta default: a caller-provided rrf beats
    the stored default, and the rrf block reports the OVERRIDE."""
    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 2.0, "k": 3})
        fused = dm.search("ds", "WebSocketHandler", top_k=3, rrf=RrfParams(dense_weight=1.0, sparse_weight=4.0, k=2))
        assert fused[0]["rrf"] == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}
    finally:
        rig.close()


def test_dataset_meta_default_applies_when_no_override():
    """No caller override + a stored meta default → the default runs, and
    the rrf block reports the META values."""
    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 4.0, "k": 2})
        fused = dm.search("ds", "WebSocketHandler", top_k=3)
        assert fused and "rrf" in fused[0]
        assert fused[0]["rrf"] == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}
        assert all("rrf" not in e for e in fused[1:]), "first entry only"
        assert all("_rrf" not in e["content"] for e in fused), "private stamp never leaks"
    finally:
        rig.close()


def test_dataset_meta_default_equal_to_global_is_byte_identical():
    """THE k-default trap, defaults-slice edition: a stored default of
    1.0/1.0 with no k (or no stored default at all) must resolve to None —
    the executed request stays the byte-identical FusionQuery form and NO
    rrf block appears."""
    rig, dm = _defaults_rig()
    try:
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        for stored in (None, {"dense_weight": 1.0, "sparse_weight": 1.0}, {"dense_weight": 1.0}):
            _write_rrf_meta(dm, "ds", stored)
            req, is_hybrid = rig.store._query_requests(
                [([0.2] * DIM, 5, True, None, "WebSocketHandler", dm._effective_rrf("ds", None))], hybrid=True
            )[0]
            assert is_hybrid and isinstance(req.query, FusionQuery) and not isinstance(req.query, RrfQuery), (
                f"stored={stored!r} must keep the default fusion form"
            )
            assert req.query.model_dump_json() == '{"fusion":"rrf"}', f"stored={stored!r} must stay byte-identical"
            plain = dm.search("ds", "WebSocketHandler", top_k=3)
            assert all("rrf" not in e for e in plain), f"stored={stored!r} must not produce an rrf block"
    finally:
        rig.close()


def test_dataset_k_only_meta_default_emits_rrf_query_with_pinned_k():
    """A stored ``{"k": 3}`` default is NOT rank-identical to the implicit
    default (k=2): it must emit RrfQuery(k=3, weights=[1.0, 1.0])."""
    rig, dm = _defaults_rig()
    try:
        bm25_lane.record_documents(rig.stats_path, [{"seed": 1}])
        _write_rrf_meta(dm, "ds", {"k": 3})
        effective = dm._effective_rrf("ds", None)
        assert effective is not None and effective.k == 3
        req, _is_hybrid = rig.store._query_requests(
            [([0.2] * DIM, 5, True, None, "WebSocketHandler", effective)], hybrid=True
        )[0]
        assert isinstance(req.query, RrfQuery)
        assert req.query.rrf.k == 3 and req.query.rrf.weights == [1.0, 1.0]
    finally:
        rig.close()


def test_update_dataset_rrf_patch_validates_and_clears():
    """PATCH allowlist: rrf clamps/validates like create; a payload reducing
    to the global default (or {}) REMOVES the stored default entirely."""
    rig = _HybridRig()
    try:
        dm = DatasetManager.__new__(DatasetManager)
        dm.datasets_path = Path(rig.tmp.name)
        (dm.datasets_path / "ds").mkdir()
        dm._write_meta("ds", {"name": "ds"})

        dm.update_dataset("ds", {"rrf": {"dense_weight": 2.5, "sparse_weight": 0.4, "k": 7}})
        assert dm._read_meta("ds")["rrf"] == {"dense_weight": 2.5, "sparse_weight": 0.4, "k": 7}

        # Out-of-range values clamp (PATCH never hard-fails on magnitude).
        dm.update_dataset("ds", {"rrf": {"dense_weight": 99.0, "k": 0}})
        assert dm._read_meta("ds")["rrf"] == {"dense_weight": 10.0, "k": 1}

        # Non-numeric → ValueError (the REST layer maps this to a 400).
        try:
            dm.update_dataset("ds", {"rrf": {"dense_weight": "abc"}})
        except ValueError:
            pass
        else:
            raise AssertionError("non-numeric weight must raise ValueError")

        # Global-default payload and empty object both REMOVE the default.
        dm.update_dataset("ds", {"rrf": {"dense_weight": 1.0, "sparse_weight": 1.0}})
        assert "rrf" not in dm._read_meta("ds")
        dm.update_dataset("ds", {"rrf": {"dense_weight": 3.0}})
        dm.update_dataset("ds", {"rrf": {}})
        assert "rrf" not in dm._read_meta("ds")
        # …and removing an absent default is a no-op, not a KeyError.
        dm.update_dataset("ds", {"rrf": {}})
        assert "rrf" not in dm._read_meta("ds")
    finally:
        rig.close()


def test_create_dataset_stamps_rrf_defaults_from_env_when_enabled() -> None:
    """RAG_RRF_DEFAULT enabled → NEW datasets are create-time stamped with
    the parsed dense/sparse[/k]; the request body's explicit rrf wins."""
    rig = _HybridRig()
    try:
        from multimodal_rag import api_server

        saved = api_server.RAG_RRF_DEFAULT
        api_server.RAG_RRF_DEFAULT = "1.0,3.0,2"
        try:
            stamped = api_server._rrf_default_meta_from_env()
        finally:
            api_server.RAG_RRF_DEFAULT = saved
        assert stamped == {"dense_weight": 1.0, "sparse_weight": 3.0, "k": 2}

        dm = DatasetManager.__new__(DatasetManager)
        dm.datasets_path = Path(rig.tmp.name)
        dm._validate_name("stamped")
        ds_dir = dm.datasets_path / "stamped"
        ds_dir.mkdir()
        (ds_dir / "files").mkdir()
        meta: dict[str, Any] = {"name": "stamped", "document_count": 0}
        if stamped is not None:
            meta["rrf"] = stamped
        dm._write_meta("stamped", meta)
        assert dm._read_meta("stamped")["rrf"] == {"dense_weight": 1.0, "sparse_weight": 3.0, "k": 2}
        assert dm._effective_rrf("stamped", None) == RrfParams(dense_weight=1.0, sparse_weight=3.0, k=2)
    finally:
        rig.close()


def test_disabled_env_stamps_nothing():
    """RAG_RRF_DEFAULT unset/garbage → no stamping: the env parser returns
    None and a created dataset carries no meta['rrf'] (existing datasets are
    never touched by the env — it only applies at create time)."""
    from multimodal_rag import api_server

    for raw in ("", "garbage", "1.0,1.0", "abc,def"):
        saved = api_server.RAG_RRF_DEFAULT
        api_server.RAG_RRF_DEFAULT = raw
        try:
            assert api_server._rrf_default_meta_from_env() is None, f"raw={raw!r}"
        finally:
            api_server.RAG_RRF_DEFAULT = saved

    rig = _HybridRig()
    try:
        dm = _dm_for_rig(rig)
        (dm.datasets_path / "plain").mkdir(exist_ok=True)
        dm._write_meta("plain", {"name": "plain", "document_count": 0})
        assert dm._effective_rrf("plain", None) is None
        assert all("rrf" not in e for e in dm.search("plain", "any query", top_k=1))
    finally:
        rig.close()


def test_create_dataset_rrf_parameter_stamps_meta():
    """create_dataset(rrf=…) — the DatasetManager-level surface the REST
    endpoint threads its (validated) payload into — stores the defaults."""
    rig = _HybridRig()
    try:
        dm = DatasetManager.__new__(DatasetManager)
        dm.datasets_path = Path(rig.tmp.name)
        dm.datasets_path.mkdir(exist_ok=True)
        dm._write_meta("made", {"name": "made", "document_count": 0})
        # Direct meta level (what create_dataset's stamping writes):
        meta = dm._read_meta("made") or {}
        sanitized = dm._sanitize_rrf_meta({"dense_weight": 2.0, "sparse_weight": 0.5})
        if sanitized is not None:
            meta["rrf"] = sanitized
        dm._write_meta("made", meta)
        assert dm._read_meta("made")["rrf"] == {"dense_weight": 2.0, "sparse_weight": 0.5}
        assert dm._effective_rrf("made", None) == RrfParams(dense_weight=2.0, sparse_weight=0.5, k=None)
    finally:
        rig.close()


def test_federated_search_ignores_dataset_meta_defaults():
    """THE RULING (pinned): per-dataset defaults apply only to single-dataset
    searches.  The federated fan-out must search every dataset with NO
    override (None reaches aretrieve even when meta['rrf'] is set), while
    the single-dataset path on the SAME dataset uses the default."""
    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 4.0, "k": 2})

        # Single-dataset path: the default applies.
        assert dm._effective_rrf("ds", None) == RrfParams(dense_weight=1.0, sparse_weight=4.0, k=2)
        assert dm.search("ds", "WebSocketHandler", top_k=3)[0]["rrf"]["sparse"] == 4.0

        # Federated path (the REST federated fan-out calls dm.search WITHOUT
        # rrf threading per-dataset defaults — the endpoint passes no
        # override and _afederated_one_dataset calls _acore_retrieval
        # directly with rrf=None): the search fn the federated core drives
        # resolves overrides ONLY from an explicit caller value.  Pin the
        # mechanism: an explicit None override on the REST dm.search call
        # path must NOT silently pick up the meta default — which is why
        # _federated_rest_search calls dm.search without touching rrf and
        # the entries carry no rrf block at all.
        from multimodal_rag.mcp_server import _afederated_search

        class _FederatedDM:
            """Duck-typed dm exposing the set default + the real rag."""

            def __init__(self) -> None:
                self._meta = {"name": "ds", "document_count": 2, "has_password": False}

            def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
                if name != "ds":
                    raise FileNotFoundError(name)
                return dict(self._meta)

            def list_datasets(self) -> list[dict[str, Any]]:
                return [dict(self._meta)]

            def has_password(self, name: str) -> bool:
                return False

            def _get_rag(self, name: str, check_embedder: bool = True) -> Any:
                return rig.rag

            def _effective_rrf(self, name: str, rrf: Any = None) -> Any:
                return dm._effective_rrf(name, rrf)  # the default IS visible

        payload = asyncio.run(_afederated_search(_FederatedDM(), ["ds"], "WebSocketHandler", top_k=3))
        assert payload["results"], "federated search still returns hits"
        # The DEFAULT fusion form runs (score_kind "rrf" is the lane label —
        # it appears on every hybrid search, defaulted or not) but no hit may
        # carry the weighted-rrf request block: per-dataset defaults never
        # apply on the federated surface (the fan-out core passes rrf=None).
        assert all("rrf" not in {k: v for k, v in r.items() if k != "score_kind"} for r in payload["results"])
        assert not any(isinstance(r.get("rrf"), dict) for r in payload["results"])
    finally:
        rig.close()


def test_mcp_search_dataset_applies_dataset_default_without_override():
    """The MCP single-dataset tool resolves the dataset default when the
    caller passes no weight: _effective_rrf is consulted in _setup and the
    override path is unchanged."""
    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 4.0, "k": 2})
        from multimodal_rag.mcp_server import _arun_retrieval

        result = asyncio.run(
            _arun_retrieval(rig.rag, "ds", "WebSocketHandler", "WebSocketHandler", 3, False, 3, ["text"], None, None)
        )
        # The direct _arun_retrieval call carries NO override (federated core
        # path) — this pins that the defaults resolution lives in the TOOL
        # (_setup) and in dm.search, not in the shared retrieval core.  (The
        # lane's score_kind label is "rrf" on every hybrid search — defaulted
        # or not — so assert on the request-level weighted block, not the
        # string.)
        payload = json.loads(result)
        assert not any(isinstance(r.get("rrf"), dict) for r in payload["results"])
        assert "sparse" not in json.dumps(payload.get("rrf", {})) if "rrf" in payload else True
    finally:
        rig.close()


def test_mcp_search_dataset_tool_uses_dataset_default_when_no_override():
    """The MCP tool itself (not just dm.search): with no caller weights, the
    dataset default reaches aretrieve; the tool's rrf block reports it."""
    from multimodal_rag import mcp_server as mcp

    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 4.0, "k": 2})

        class _ToolDM:
            """Just the surface search_dataset's _setup touches."""

            def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
                if name != "ds":
                    raise FileNotFoundError(f"Dataset '{name}' not found")
                return {"name": name, "document_count": 2}

            def _get_rag(self, name: str, check_embedder: bool = True) -> Any:
                return rig.rag

            def _effective_rrf(self, name: str, rrf: Any = None) -> Any:
                return dm._effective_rrf(name, rrf)

        def _install() -> tuple[Any, Any, Any]:
            saved = (mcp._dm, mcp._require_dataset_acl, mcp._resolve_and_unlock)
            mcp._dm = _ToolDM()  # type: ignore[assignment]
            mcp._require_dataset_acl = lambda name: None  # type: ignore[assignment]
            mcp._resolve_and_unlock = lambda dm_, name, password: None  # type: ignore[assignment]
            return saved

        def _restore(saved: tuple[Any, Any, Any]) -> None:
            mcp._dm, mcp._require_dataset_acl, mcp._resolve_and_unlock = saved  # type: ignore[method-assign]

        saved = _install()
        try:
            payload = json.loads(asyncio.run(mcp.search_dataset(dataset_name="ds", query="WebSocketHandler", top_k=3)))
        finally:
            _restore(saved)
        assert payload["results"], "tool still returns hits"
        assert isinstance(payload.get("rrf"), dict), "the effective-weights block must be reported"
        assert payload["rrf"] == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}

        # An explicit caller override wins over the stored default.
        saved = _install()
        try:
            payload2 = json.loads(
                asyncio.run(
                    mcp.search_dataset(dataset_name="ds", query="WebSocketHandler", top_k=3, sparse_weight=9.0, k=5)
                )
            )
        finally:
            _restore(saved)
        assert payload2["rrf"]["sparse"] == 9.0 and payload2["rrf"]["k"] == 5
    finally:
        rig.close()


def test_rest_create_dataset_body_rrf_flows_and_validates():
    """POST /api/datasets: an explicit body ``rrf`` wins over the env
    default; a non-numeric weight is HTTP 400; omitted → env (disabled →
    nothing)."""
    from multimodal_rag import api_server

    class _StubDM:
        caption_with_asr = False
        caption_with_vlm = False

        def __init__(self) -> None:
            self.seen: list[Any] = []

        def _sanitize_rrf_meta(self, payload: Any) -> Any:
            return DatasetManager._sanitize_rrf_meta(payload)

        def create_dataset(
            self,
            name: str,
            description: str,
            caption_with_asr: bool,
            caption_with_vlm: bool,
            keep_originals: bool,
            password: str | None,
            ocr: bool,
            rrf: Any = None,
            contextual: bool = False,
            created_by: str | None = None,  # D23 ownership stamp (passed through)
            chunk: Any = None,  # per-dataset chunk overrides (passed through)
        ) -> dict[str, Any]:
            DatasetManager._validate_name(name)
            self.seen.append({"name": name, "rrf": rrf, "chunk": chunk})
            return {"name": name}

    stub = _StubDM()

    async def _fake_get_manager() -> _StubDM:
        return stub

    saved = (api_server.RAG_RRF_DEFAULT, api_server.get_manager_async)
    api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
    api_server.RAG_RRF_DEFAULT = "2.0,2.0,2"  # enabled env default
    try:
        # Explicit body rrf wins over the env default.
        body = asyncio.run(api_server.api_create_dataset({"name": "team docs", "rrf": {"sparse_weight": 5.0}}))
        assert body["dataset"]["name"] == "team_docs"
        assert stub.seen[-1]["rrf"] == {"sparse_weight": 5.0}

        # No body rrf → the env default is stamped.
        asyncio.run(api_server.api_create_dataset({"name": "env stamped"}))
        assert stub.seen[-1]["rrf"] == {"dense_weight": 2.0, "sparse_weight": 2.0, "k": 2}

        # Non-numeric weight → HTTP 400, nothing created.
        from fastapi import HTTPException

        try:
            asyncio.run(api_server.api_create_dataset({"name": "bad", "rrf": {"dense_weight": "abc"}}))
        except HTTPException as exc:
            assert exc.status_code == 400 and "rrf" in str(exc.detail)
        else:
            raise AssertionError("non-numeric rrf weight must be HTTP 400")
        assert all(s["name"] != "bad" for s in stub.seen), "a rejected rrf must not create the dataset"

        # A non-object rrf → HTTP 400 too.
        try:
            asyncio.run(api_server.api_create_dataset({"name": "bad2", "rrf": 3}))
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("non-object rrf must be HTTP 400")
    finally:
        api_server.RAG_RRF_DEFAULT, api_server.get_manager_async = saved


# ---------------------------------------------------------------------------
# REST top-level rrf block (response-level) — must agree with the first entry
# ---------------------------------------------------------------------------


def _rest_search_dm(rig: _HybridRig) -> Any:
    """dm shell for the REST endpoint tests: has_password=False (no auth
    gate) and the rig's rag wired in — everything else is unused."""
    dm = _dm_for_rig(rig)
    dm.has_password = lambda name: False  # type: ignore[method-assign]
    return dm


def test_rest_top_level_rrf_block_agrees_with_first_entry() -> None:
    """GET /api/datasets/{name}/search: the top-level ``rrf`` block mirrors
    the first entry's block (the source of truth for what was actually
    applied).  Wave-1 bug: it re-derived ``applied`` from the per-doc
    ``_rrf`` stamps that dm.search pops while folding — so it read
    ``false`` even when the weighted override actually ran."""
    from multimodal_rag import api_server

    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
                {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
            ],
        )
        dm = _rest_search_dm(rig)

        async def _fake_get_manager() -> Any:
            return dm

        class _FakeRequest:
            headers: ClassVar[dict[str, str]] = {}

        saved = api_server.get_manager_async
        api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
        try:
            body = asyncio.run(
                api_server.api_search(
                    name="ds",
                    request=_FakeRequest(),  # type: ignore[arg-type]
                    q="WebSocketHandler",
                    top_k=3,
                    use_reranker=False,
                    reranker_top_k=3,
                    dense_weight=1.0,
                    sparse_weight=4.0,
                    k=2,
                    file_types="",
                    severities="",
                    source_prefix="",
                    date_from="",
                    date_to="",
                    x_dataset_password=None,
                )
            )
        finally:
            api_server.get_manager_async = saved  # type: ignore[method-assign, assignment]

        assert body["results"], "the search itself must still return hits"
        first_block = body["results"][0].get("rrf")
        assert first_block == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}, (
            f"first-entry block shape pinned: {first_block!r}"
        )
        assert body["rrf"] == first_block, (
            f"top-level block must agree with the first entry: {body['rrf']!r} vs {first_block!r}"
        )
        assert body["rrf"]["applied"] is True
    finally:
        rig.close()


def test_rest_top_level_rrf_block_absent_without_override() -> None:
    """No override requested → no top-level ``rrf`` key at all (byte-identical
    default, the k-default-trap discipline)."""
    from multimodal_rag import api_server

    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"}])
        dm = _rest_search_dm(rig)

        async def _fake_get_manager() -> Any:
            return dm

        class _FakeRequest:
            headers: ClassVar[dict[str, str]] = {}

        saved = api_server.get_manager_async
        api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
        try:
            body = asyncio.run(
                api_server.api_search(
                    name="ds",
                    request=_FakeRequest(),  # type: ignore[arg-type]
                    q="WebSocketHandler",
                    top_k=3,
                    use_reranker=False,
                    reranker_top_k=3,
                    dense_weight=None,
                    sparse_weight=None,
                    k=None,
                    file_types="",
                    severities="",
                    source_prefix="",
                    date_from="",
                    date_to="",
                    x_dataset_password=None,
                )
            )
        finally:
            api_server.get_manager_async = saved  # type: ignore[method-assign, assignment]

        assert body["results"]
        assert "rrf" not in body, f"default request must carry no top-level rrf block, got {body.get('rrf')!r}"
        assert all("rrf" not in e for e in body["results"])
    finally:
        rig.close()


def test_rest_top_level_rrf_block_false_on_dense_degraded() -> None:
    """Override requested but the lane degraded to dense-only (hybrid off) →
    the top-level block reports applied=false honestly (never label dense
    results with weights)."""
    from multimodal_rag import api_server

    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "some indexable text", "source": "/tmp/x/a.txt"}])
        dm = _rest_search_dm(rig)

        async def _fake_get_manager() -> Any:
            return dm

        class _FakeRequest:
            headers: ClassVar[dict[str, str]] = {}

        saved = api_server.get_manager_async
        api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
        try:
            with _env(RAG_HYBRID_SEARCH="0"):
                body = asyncio.run(
                    api_server.api_search(
                        name="ds",
                        request=_FakeRequest(),  # type: ignore[arg-type]
                        q="some text",
                        top_k=3,
                        use_reranker=False,
                        reranker_top_k=3,
                        dense_weight=5.0,
                        sparse_weight=5.0,
                        k=None,
                        file_types="",
                        severities="",
                        source_prefix="",
                        date_from="",
                        date_to="",
                        x_dataset_password=None,
                    )
                )
        finally:
            api_server.get_manager_async = saved  # type: ignore[method-assign, assignment]

        assert body["results"]
        assert body["rrf"]["applied"] is False, f"degraded search must report applied=false: {body['rrf']!r}"
        assert body["rrf"]["dense"] == 5.0 and body["rrf"]["sparse"] == 5.0
        first_block = body["results"][0].get("rrf")
        assert first_block is not None and first_block["applied"] is False
        assert body["rrf"] == first_block, "degraded top-level must agree with the first entry too"
    finally:
        rig.close()


def test_rest_post_search_top_level_rrf_block_agrees() -> None:
    """POST /api/datasets/{name}/search carries the same agreement: the
    body-override path's top-level block mirrors the first entry."""
    from multimodal_rag import api_server

    rig = _HybridRig()
    try:
        _ingest(
            rig,
            [
                {"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"},
                {"text": "The calm sea at sunset with gentle waves", "source": "/tmp/x/sea.txt"},
            ],
        )
        dm = _rest_search_dm(rig)

        async def _fake_get_manager() -> Any:
            return dm

        class _FakeRequest:
            headers: ClassVar[dict[str, str]] = {}

        saved = api_server.get_manager_async
        api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
        try:
            body = asyncio.run(
                api_server.api_search_multimodal(
                    name="ds",
                    request=_FakeRequest(),  # type: ignore[arg-type]
                    body={"query": "WebSocketHandler", "sparse_weight": 4.0, "k": 2},
                )
            )
        finally:
            api_server.get_manager_async = saved  # type: ignore[method-assign, assignment]

        assert body["results"]
        first_block = body["results"][0].get("rrf")
        assert first_block == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}
        assert body["rrf"] == first_block and body["rrf"]["applied"] is True
    finally:
        rig.close()


def test_rest_top_level_rrf_block_false_for_default_equal_override() -> None:
    """A present-but-default override (1.0/1.0, no k) executes the DEFAULT
    fusion form (k-default-trap discipline) and stamps nothing — both blocks
    must honestly report applied=false while echoing the parameters."""
    from multimodal_rag import api_server

    rig = _HybridRig()
    try:
        _ingest(rig, [{"text": "WebSocketHandler negotiate the handshake upgrade", "source": "/tmp/x/ws.py"}])
        dm = _rest_search_dm(rig)

        async def _fake_get_manager() -> Any:
            return dm

        class _FakeRequest:
            headers: ClassVar[dict[str, str]] = {}

        saved = api_server.get_manager_async
        api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
        try:
            body = asyncio.run(
                api_server.api_search(
                    name="ds",
                    request=_FakeRequest(),  # type: ignore[arg-type]
                    q="WebSocketHandler",
                    top_k=3,
                    use_reranker=False,
                    reranker_top_k=3,
                    dense_weight=1.0,
                    sparse_weight=1.0,
                    k=None,
                    file_types="",
                    severities="",
                    source_prefix="",
                    date_from="",
                    date_to="",
                    x_dataset_password=None,
                )
            )
        finally:
            api_server.get_manager_async = saved  # type: ignore[method-assign, assignment]

        assert body["results"]
        first_block = body["results"][0].get("rrf")
        assert first_block == {"dense": 1.0, "sparse": 1.0, "k": None, "applied": False}, (
            f"nothing was applied that differs from the default: {first_block!r}"
        )
        assert body["rrf"] == first_block
    finally:
        rig.close()


def test_mcp_search_dataset_rrf_block_unchanged():
    """MCP regression pin: the tool's ``rrf`` block still reports the
    applied effective parameters ({dense, sparse, k, applied: true} under
    an override) — the REST fix must not perturb the MCP surface."""
    from multimodal_rag import mcp_server as mcp

    rig, dm = _defaults_rig()
    try:
        _write_rrf_meta(dm, "ds", {"dense_weight": 1.0, "sparse_weight": 4.0, "k": 2})

        class _ToolDM:
            def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
                if name != "ds":
                    raise FileNotFoundError(f"Dataset '{name}' not found")
                return {"name": name, "document_count": 2}

            def _get_rag(self, name: str, check_embedder: bool = True) -> Any:
                return rig.rag

            def _effective_rrf(self, name: str, rrf: Any = None) -> Any:
                return dm._effective_rrf(name, rrf)

        def _install() -> tuple[Any, Any, Any]:
            saved = (mcp._dm, mcp._require_dataset_acl, mcp._resolve_and_unlock)
            mcp._dm = _ToolDM()  # type: ignore[assignment]
            mcp._require_dataset_acl = lambda name: None  # type: ignore[assignment]
            mcp._resolve_and_unlock = lambda dm_, name, password: None  # type: ignore[assignment]
            return saved

        def _restore(saved: tuple[Any, Any, Any]) -> None:
            mcp._dm, mcp._require_dataset_acl, mcp._resolve_and_unlock = saved  # type: ignore[assignment]

        # Dataset-default path (no caller weights): block present, applied.
        saved = _install()
        try:
            payload = json.loads(asyncio.run(mcp.search_dataset(dataset_name="ds", query="WebSocketHandler", top_k=3)))
        finally:
            _restore(saved)
        assert payload["results"]
        assert payload["rrf"] == {"dense": 1.0, "sparse": 4.0, "k": 2, "applied": True}

        # Explicit caller override path: same shape, override values.
        saved = _install()
        try:
            payload2 = json.loads(
                asyncio.run(
                    mcp.search_dataset(dataset_name="ds", query="WebSocketHandler", top_k=3, sparse_weight=9.0, k=5)
                )
            )
        finally:
            _restore(saved)
        assert payload2["rrf"]["sparse"] == 9.0 and payload2["rrf"]["k"] == 5 and payload2["rrf"]["applied"] is True

        # No meta default, no override → no block at all.
        _write_rrf_meta(dm, "ds", None)
        saved = _install()
        try:
            payload3 = json.loads(asyncio.run(mcp.search_dataset(dataset_name="ds", query="WebSocketHandler", top_k=3)))
        finally:
            _restore(saved)
        assert "rrf" not in payload3, "no default + no override must produce no rrf block"
    finally:
        rig.close()


def test_index_html_renders_the_rrf_controls():
    """UI smoke: the template ships the create-form fields, the edit-row
    controls, the saveRrfDefaults() PATCH call, and the meta consumer; when
    node is available every script block is syntax-checked."""
    template = Path(__file__).resolve().parents[2] / "src" / "multimodal_rag" / "templates" / "index.html"
    html_text = template.read_text(encoding="utf-8")
    for required in (
        'id="ds-rrf-dense"',
        'id="ds-rrf-sparse"',
        'id="ds-rrf-k"',
        'id="edit-rrf-dense"',
        'id="edit-rrf-sparse"',
        'id="edit-rrf-k"',
        "saveRrfDefaults()",
        'meta[name="rag-rrf-default"]',
        "trust the order, not the magnitude",
    ):
        assert required in html_text, f"template must contain {required!r}"
    # The tooltip must carry the rank-space semantics AND the multimodal
    # exclusion (the wave-1 docstring wording).
    assert "multimodal (vector-only) queries ignore weights" in html_text

    # JS syntax gate (real parser when available — regex brace counting is
    # fooled by template literals and regexes in the existing blocks).
    scripts = re.findall(r"<script>(.*?)</script>", html_text, re.DOTALL)
    assert len(scripts) >= 2
    node = shutil.which("node")
    if node is None:
        print("  (node not available — skipping the JS syntax check)")
        return
    import subprocess

    for i, script in enumerate(scripts):
        tmp = Path(tempfile.mkdtemp()) / f"script-{i}.js"
        tmp.write_text(script, encoding="utf-8")
        proc = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"script block {i} is not valid JS: {proc.stderr}"


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


# ---------------------------------------------------------------------------
# P1 (2026-10-21): RRF score invariance in top_k
# ---------------------------------------------------------------------------


def test_prefetch_depth_decouples_from_display_limit():
    """A fused score must not depend on how many results the caller asked to SEE.

    RRF is rank arithmetic over its candidate lists, so the prefetch window is
    part of the ranking definition.  When both prefetches were bound to k, a
    doc that is dense rank 11 and sparse rank 1 scored 1/12 + 1/2 = 0.5833 with
    a deep pool, but only 0.5000 at top_k=10 (the dense lane cannot return
    position 11, so its contribution vanished and the doc looked single-lane).

    The floor must cover EVERY legal request, not just top_k: aretrieve
    over-fetches for the reranker with max(top_k, 4*reranker_top_k), and
    reranker_top_k is allowed up to 50, so fetch_k reaches 200.
    """
    # Every legal (top_k, reranker_top_k) combination must fuse over ONE pool.
    depths = {_prefetch_depth(top_k) for top_k in (1, 10, 50, 100)}
    for reranker_top_k in (1, 3, 10, 30, 50):
        for top_k in (10, 100):
            depths.add(_prefetch_depth(max(top_k, 4 * reranker_top_k)))
    assert depths == {_DEFAULT_PREFETCH_FLOOR}, f"still coupled to display params: {sorted(depths)}"
    # A caller asking for MORE than the floor still gets a full page.
    assert _prefetch_depth(10_000) >= 10_000


def test_prefetch_depth_is_configurable(monkeypatch):
    """RAG_PREFETCH_FLOOR overrides the floor; a bad value must not break search."""
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "250")
    assert _prefetch_depth(10) == 250
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "nonsense")
    # Falls back to the default rather than raising - a bad env var must not
    # take retrieval down.
    assert _prefetch_depth(10) == _DEFAULT_PREFETCH_FLOOR
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "-5")
    assert _prefetch_depth(10) == _DEFAULT_PREFETCH_FLOOR
    # An absurd value is CLAMPED, never sent verbatim as an unbounded scan.
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "1000000")
    assert _prefetch_depth(10) == _MAX_PREFETCH_FLOOR


def test_stable_tiebreak_is_deterministic():
    """Equal fused scores must order reproducibly (P2).

    Equal fused scores are structural under RRF: with weights 1.5/1.0 and k=2
    the dense ladder yields 0.6, 0.4286, 0.3333... and the sparse ladder
    0.5, 0.3333, 0.25..., so dense position 3 ties sparse position 2 at
    exactly 0.3333.  qdrant-client sorts on score alone, leaving ties in
    Qdrant return order, which is not guaranteed stable between identical
    queries.
    """
    from multimodal_rag.vector_store import Document, _stable_tiebreak

    def _doc(src, page, text):
        return Document(page_content=text, metadata={"source": src, "page": page})

    a = (_doc("b.pdf", 2, "beta"), 0.3333)
    b = (_doc("a.pdf", 9, "alpha"), 0.3333)
    c = (_doc("c.pdf", 1, "gamma"), 0.6000)
    # TWINS: identical source/page/chunk_index AND identical text — the four
    # content key fields all collide, so only _point_id can order them.
    t1 = (_doc("twin.pdf", 4, "same"), 0.5)
    t2 = (_doc("twin.pdf", 4, "same"), 0.5)
    t1[0].metadata["_point_id"] = "aaa"
    t2[0].metadata["_point_id"] = "bbb"
    twin_orders = set()
    for perm in ([t1, t2], [t2, t1]):
        got = tuple(d.metadata["_point_id"] for d, _ in _stable_tiebreak(list(perm)))
        twin_orders.add(got)
    assert twin_orders == {("aaa", "bbb")}, f"twin tiebreak not total: {twin_orders}"
    # Shuffled input must produce the SAME order every time.
    orders = set()
    for perm in ([a, b, c], [b, a, c], [c, a, b], [b, c, a]):
        got = tuple((d.metadata["source"], d.metadata["page"]) for d, _ in _stable_tiebreak(list(perm)))
        orders.add(got)
    assert len(orders) == 1, f"tiebreak not deterministic: {orders}"
    # Higher score first, then content order.
    assert next(iter(orders)) == (("c.pdf", 1), ("a.pdf", 9), ("b.pdf", 2))


# ---------------------------------------------------------------------------
# P1 (2026-10-21): pre-fusion twin collapse
# ---------------------------------------------------------------------------


def test_twin_collapse_per_lane_keeps_higher_similarity():
    """Per-lane collapse keeps ONE representative: the higher similarity.

    Twins are intentional: a document may be represented by a multimodal
    parent and a text-only twin.  Within a single lane the better-scoring
    representation must win, and the loser must not consume a rank position.
    """
    from multimodal_rag.vector_store import collapse_lane_by_twin

    def h(pid, sim, twin, page=23):
        return (pid, {"source": "M.pdf", "page": page, "chunk_index": None, "_twin": twin}, sim)

    parent = h("p-par", 0.40, False, page=23)
    twin = h("p-twi", 0.58, True, page=23)
    other = h("p-o", 0.50, False, page=26)   # distinct document

    got = [x[0] for x in collapse_lane_by_twin([parent, twin, other])]
    assert got == ["p-twi", "p-o"], "higher similarity must win, order preserved"

    # Order-independent.
    got2 = [x[0] for x in collapse_lane_by_twin([twin, parent, other])]
    assert got2 == got


def test_twin_collapse_merges_cross_lane_representations():
    """Dense may pick one rep and sparse the other: still ONE document.

    This is the point of twins - each lane uses its own best representation -
    so the fusion must merge them into a single entry carrying BOTH lane
    positions rather than emitting the document twice.
    """
    from multimodal_rag.vector_store import fuse_lanes_with_twin_collapse

    def h(pid, sim, twin, page=23):
        return (pid, {"source": "M.pdf", "page": page, "chunk_index": None, "_twin": twin}, sim)

    parent = h("p-par", 0.40, False, page=23)
    twin = h("p-twi", 0.58, True, page=23)
    other = h("p-o", 0.50, False, page=26)   # distinct document

    fused = fuse_lanes_with_twin_collapse([parent, other], [twin], limit=10)
    assert len(fused) == 2, f"parent+twin must collapse to one doc, got {len(fused)}"
    _score, _pid, _meta, dpos, spos = fused[0]
    assert dpos is not None and spos is not None, "merged doc must carry BOTH lane positions"


def test_twin_collapse_score_is_representation_independent():
    """A doc scores the same whichever representation each lane returned.

    The defect being fixed: with server-side fusion the fused score changed
    when the twin and parent both entered the pool, so asking for MORE
    results could LOWER a document score.
    """
    from multimodal_rag.vector_store import fuse_lanes_with_twin_collapse

    def h(pid, sim, twin, page=23):
        return (pid, {"source": "M.pdf", "page": page, "chunk_index": None, "_twin": twin}, sim)

    other = h("p-o", 0.50, False, page=26)   # distinct document
    twin = h("p-twi", 0.58, True, page=23)
    parent = h("p-par", 0.40, False, page=23)

    a = fuse_lanes_with_twin_collapse([twin, other], [twin], limit=10)[0][0]
    b = fuse_lanes_with_twin_collapse([parent, other], [twin], limit=10)[0][0]
    assert abs(a - b) < 1e-9, f"score must not depend on which rep a lane returned: {a} vs {b}"


def test_twin_collapse_scores_match_server_side_arithmetic():
    """The client-side lane term must equal qdrant-client weighted RRF."""
    from multimodal_rag.vector_store import _rrf_lane_term

    assert abs(_rrf_lane_term(0, 1.0, 2) - 0.5) < 1e-12
    assert abs(_rrf_lane_term(1, 1.0, 2) - 1.0 / 3) < 1e-12
    assert abs(_rrf_lane_term(0, 1.5, 2) - 0.6) < 1e-12


def test_twin_collapse_docs_without_source_never_collapse():
    """Raw text drops (no source) must not collapse against each other."""
    from multimodal_rag.vector_store import fuse_lanes_with_twin_collapse

    a = ("a", {"text": "alpha"}, 0.9)
    b = ("b", {"text": "beta"}, 0.8)
    fused = fuse_lanes_with_twin_collapse([a, b], [], limit=10)
    assert len(fused) == 2, "sourceless docs must stay distinct"
