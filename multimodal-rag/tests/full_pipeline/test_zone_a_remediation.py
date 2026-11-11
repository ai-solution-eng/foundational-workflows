"""Regression tests for the 2026-11-10 audit remediation of the data core (zone A/W1).

Every test here fails on the pre-remediation tree and passes after it:

  * P1-1 - the twin-collapse hybrid path stamped ``_score_kind='rrf'`` on
    DENSE-ONLY results (a hybrid-intent query with no usable BM25 stats runs a
    single flat dense request whose scores are raw cosine).
  * P1-2 - the 200-deep fusion window carried the full payload and dense
    vectors on BOTH lanes; it now projects only what fusion reads and re-reads
    the <= k survivors once (payload + vector) in a single retrieve.
  * P2-1 - dropping a tier-1 original rewrote the whole .hashes.json per file
    and its removal was resurrected by the batch's deferred flush.
  * P2-2 - the bundled tokenizer was re-parsed per structured file.
  * P2-3 - a malformed EMBED_BATCH_* env value crashed the batcher at import.
  * P2-4 - PATCH {'chunk': {}} could not clear a stored chunk override.
  * P3-1 - the query-vector LRU touched an O(n) list under the shared lock.
  * P3-2 - the prefetch floor was re-parsed (and re-warned) per query.

Offline: embedded local Qdrant is not needed - the vector-store tests drive a
fake qdrant client, and the DatasetManager tests use the suite's stub embedder.
"""

import json
import logging
import os
import sys
from collections import OrderedDict
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client.models import PayloadSelectorInclude, SparseVector

from multimodal_rag import dataset_manager as dm_mod
from multimodal_rag import embed_batcher as eb_mod
from multimodal_rag import vector_store as vs_mod
from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils import token_text_splitter as tts
from multimodal_rag.vector_store import QdrantVectorStore

# ---------------------------------------------------------------------------
# Fake qdrant client (no server, no local storage)
# ---------------------------------------------------------------------------


class _Point:
    def __init__(self, pid: str, payload: dict[str, Any], score: float = 0.0, vector: Any = None) -> None:
        self.id = pid
        self.payload = payload
        self.score = score
        self.vector = vector


class _Resp:
    def __init__(self, points: list[_Point]) -> None:
        self.points = points


class _FakeClient:
    """Records the requests it is given and replays canned responses."""

    def __init__(self, responses: list[_Resp], retrieved: dict[str, _Point] | None = None) -> None:
        self.responses = responses
        self.retrieved = retrieved or {}
        self.query_requests: list[Any] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.fail_retrieve = False

    def query_batch_points(self, *, collection_name: str, requests: list[Any]) -> list[_Resp]:
        self.query_requests.extend(requests)
        return self.responses

    def retrieve(self, *, collection_name: str, ids: list[Any], with_payload: Any, with_vectors: Any) -> list[_Point]:
        self.retrieve_calls.append({"ids": list(ids), "with_payload": with_payload, "with_vectors": with_vectors})
        if self.fail_retrieve:
            raise RuntimeError("retrieve exploded")
        return [self.retrieved[str(i)] for i in ids if str(i) in self.retrieved]


def _bare_store(client: Any) -> QdrantVectorStore:
    """A QdrantVectorStore with only the attributes the query path touches."""
    store = object.__new__(QdrantVectorStore)
    store.vector_name = bm25_lane.DENSE_VECTOR_NAME
    store._dense_vector_name = bm25_lane.DENSE_VECTOR_NAME
    store._dense_name_probed = True
    store.collection_name = "zone-a"
    store._client = client
    store._fusion_supported = None
    store.bm25_stats_path = None
    return store


def _window_meta(pid: str, source: str) -> dict[str, Any]:
    """Only the keys the window projection asks for."""
    return {"source": source, "page": 1, "chunk_index": 0}


# ---------------------------------------------------------------------------
# P1-1: the lane stamp belongs to the query that actually fused
# ---------------------------------------------------------------------------


def test_laned_dense_only_results_are_not_stamped_rrf(monkeypatch) -> None:
    """No usable BM25 stats => a single dense request => raw cosine scores.

    The laned path used to stamp every result 'rrf' regardless, which made the
    federated merge and the reranker treat a cosine as rank arithmetic.
    """
    points = [
        _Point("p1", {"page_content": "alpha", "metadata": {"source": "/a"}}, 0.91),
        _Point("p2", {"page_content": "beta", "metadata": {"source": "/b"}}, 0.42),
    ]
    client = _FakeClient([_Resp(points)])
    store = _bare_store(client)
    monkeypatch.setattr(store, "supports_hybrid", lambda: True)
    monkeypatch.setattr(bm25_lane, "hybrid_search_enabled", lambda: True)
    # Empty/absent stats: the store cannot build a sparse lane for this query.
    monkeypatch.setattr(store, "_bm25_stats", lambda: None)

    out = store.similarity_search_with_score_by_vector_batch([([0.1] * 4, 2, False, None, "alpha", None)])

    assert len(client.query_requests) == 1, "dense-only query must stay ONE request"
    assert client.query_requests[0].with_vector is False
    docs = [doc for doc, _score in out[0]]
    assert [d.page_content for d in docs] == ["alpha", "beta"]
    for doc in docs:
        assert "_score_kind" not in doc.metadata, "a dense-only result must not claim the rrf lane"
        assert "_rrf" not in doc.metadata


def test_laned_fused_results_are_stamped_and_survivors_are_refetched(monkeypatch) -> None:
    """A query that DOES fuse keeps the stamp, and its survivors are re-read.

    P1-2: the window carries no vectors and a projected payload, so the full
    payload and the dense cosine must come back from the survivor retrieve.
    """
    dense = _Resp(
        [
            _Point("a", {"page_content": "A", "metadata": _window_meta("a", "/a")}, 0.90),
            _Point("b", {"page_content": "B", "metadata": _window_meta("b", "/b")}, 0.80),
        ]
    )
    sparse = _Resp([_Point("b", {"page_content": "B", "metadata": _window_meta("b", "/b")}, 7.0)])
    retrieved = {
        "a": _Point(
            "a",
            {"page_content": "A", "metadata": {"source": "/a", "file_type": "text"}},
            vector={"dense": [0.1, 0.2, 0.3, 0.4]},
        ),
        "b": _Point(
            "b",
            {"page_content": "B", "metadata": {"source": "/b", "file_type": "text"}},
            vector={"dense": [0.4, 0.3, 0.2, 0.1]},
        ),
    }
    client = _FakeClient([dense, sparse], retrieved)
    store = _bare_store(client)
    monkeypatch.setattr(store, "supports_hybrid", lambda: True)
    monkeypatch.setattr(bm25_lane, "hybrid_search_enabled", lambda: True)
    monkeypatch.setattr(store, "_bm25_stats", lambda: {"n_docs": 1})
    monkeypatch.setattr(
        bm25_lane, "build_query_sparse_vector", lambda text, stats: SparseVector(indices=[1], values=[1.0])
    )

    out = store.similarity_search_with_score_by_vector_batch([([0.1] * 4, 2, False, None, "alpha", None)])

    # Two lane requests, both projected + vector-free (the P1-2 transport fix).
    assert len(client.query_requests) == 2
    for req in client.query_requests:
        assert req.with_vector is False, "the 200-deep window must not carry vectors"
        assert isinstance(req.with_payload, PayloadSelectorInclude)
        assert "metadata.source" in req.with_payload.include
        assert req.limit == vs_mod._prefetch_depth(2) > 2, "the floor is unchanged"

    # ONE retrieve for the survivors only, full payload + vector.
    assert len(client.retrieve_calls) == 1
    assert sorted(client.retrieve_calls[0]["ids"]) == ["a", "b"]
    assert client.retrieve_calls[0]["with_payload"] is True
    assert client.retrieve_calls[0]["with_vectors"] is True

    docs = [doc for doc, _score in out[0]]
    assert [d.page_content for d in docs] == ["B", "A"], "fused order: B has both lanes"
    for doc in docs:
        assert doc.metadata.get("file_type") == "text", "full payload must be restored"
        assert isinstance(doc.metadata.get("_embedding_score"), float)
        assert doc.metadata.get("_score_kind") == "rrf"


def test_survivor_refetch_failure_keeps_the_ranked_results(monkeypatch) -> None:
    """A retrieve failure must degrade the payload, never lose the search."""
    dense = _Resp([_Point("a", {"page_content": "A", "metadata": _window_meta("a", "/a")}, 0.9)])
    sparse = _Resp([_Point("a", {"page_content": "A", "metadata": _window_meta("a", "/a")}, 3.0)])
    client = _FakeClient([dense, sparse])
    client.fail_retrieve = True
    store = _bare_store(client)
    monkeypatch.setattr(store, "supports_hybrid", lambda: True)
    monkeypatch.setattr(bm25_lane, "hybrid_search_enabled", lambda: True)
    monkeypatch.setattr(store, "_bm25_stats", lambda: {"n_docs": 1})
    monkeypatch.setattr(
        bm25_lane, "build_query_sparse_vector", lambda text, stats: SparseVector(indices=[1], values=[1.0])
    )

    out = store.similarity_search_with_score_by_vector_batch([([0.1] * 4, 1, False, None, "q", None)])

    assert len(out[0]) == 1, "the ranked hit survives a failed re-fetch"
    doc, score = out[0][0]
    assert doc.page_content == "A" and score > 0


# ---------------------------------------------------------------------------
# P3-2: the prefetch floor is resolved once per distinct raw value
# ---------------------------------------------------------------------------


def test_prefetch_floor_warns_once_and_re_resolves(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "not-a-number")
    vs_mod._PREFETCH_FLOOR_CACHE.clear()
    vs_mod._PREFETCH_FLOOR_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=vs_mod.logger.name):
        for _ in range(5):
            assert vs_mod._prefetch_floor() == vs_mod._DEFAULT_PREFETCH_FLOOR
    warnings = [r for r in caplog.records if "RAG_PREFETCH_FLOOR" in r.getMessage()]
    assert len(warnings) == 1, "a bad value must warn once, not once per query"

    # A ConfigMap change still takes effect on the next query.
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "250")
    assert vs_mod._prefetch_floor() == 250
    assert vs_mod._prefetch_depth(10) == 250
    monkeypatch.setenv("RAG_PREFETCH_FLOOR", "999999")
    assert vs_mod._prefetch_floor() == vs_mod._MAX_PREFETCH_FLOOR
    vs_mod._PREFETCH_FLOOR_CACHE.clear()
    vs_mod._PREFETCH_FLOOR_WARNED.clear()


# ---------------------------------------------------------------------------
# P2-1: hash-index removals ride the deferred dirty set (no resurrection)
# ---------------------------------------------------------------------------


def test_hash_index_delete_is_deferred_and_survives_the_flush(tmp_path) -> None:
    """The batch's deferred store must not resurrect a deleted entry."""
    hashes = tmp_path / ".hashes.json"
    dm_mod._write_hash_index(hashes, {"hash-store": "/data/files/uuid_a.png"})
    dm_mod._hash_index_dirty.clear()
    try:
        # Simulate _store_file(defer_write=True) for the file we then drop.
        dm_mod._hash_index_dirty.setdefault(hashes, {})["hash-store"] = "/data/files/uuid_a.png"
        assert dm_mod._stage_hash_index_delete(hashes, "/data/files/uuid_a.png") is True
        # Nothing is written until the flush: the on-disk entry is still there.
        assert "hash-store" in json.loads(hashes.read_text())
        dm_mod._flush_hash_index_writes()
        assert "hash-store" not in json.loads(hashes.read_text())
    finally:
        dm_mod._hash_index_dirty.clear()


def test_hash_index_delete_skips_the_write_when_nothing_matches(tmp_path) -> None:
    hashes = tmp_path / ".hashes.json"
    dm_mod._write_hash_index(hashes, {"h1": "/data/files/one.png"})
    before = hashes.stat().st_mtime_ns
    dm_mod._hash_index_dirty.clear()
    assert dm_mod._stage_hash_index_delete(hashes, "/data/files/other.png") is False
    assert not dm_mod._hash_index_dirty, "no match must not stage a write"
    assert hashes.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# P2-2: the tokenizer object is loaded once per path
# ---------------------------------------------------------------------------


def test_tokenizer_is_parsed_once_per_path(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeTok:
        def encode(self, text: str) -> Any:
            return type("E", (), {"ids": list(range(len(text.split())))})()

    class _FakeTokenizers:
        class Tokenizer:
            @staticmethod
            def from_file(path: str) -> Any:
                calls.append(path)
                return _FakeTok()

    monkeypatch.setitem(sys.modules, "tokenizers", _FakeTokenizers)
    tts._TOKENIZER_OBJ_CACHE.clear()
    first = tts._load_tokenizer("/tmp/tok.json")
    second = tts._load_tokenizer("/tmp/tok.json")
    assert first is second, "the parsed Tokenizer must be shared"
    assert calls == ["/tmp/tok.json"], "the tokenizer.json must be parsed once"
    tts._TOKENIZER_OBJ_CACHE.clear()


# ---------------------------------------------------------------------------
# P2-3 / P3-1: embed-batcher env parsing and the LRU
# ---------------------------------------------------------------------------


def test_embed_batcher_env_parsing_is_defensive(monkeypatch, caplog) -> None:
    monkeypatch.setenv("EMBED_BATCH_CACHE_MAX", "4k")
    with caplog.at_level(logging.WARNING):
        assert eb_mod._env_int("EMBED_BATCH_CACHE_MAX", 4096) == 4096
    assert any("EMBED_BATCH_CACHE_MAX" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("EMBED_BATCH_FLUSH_TIMEOUT", "soon")
    assert eb_mod._env_float("EMBED_BATCH_FLUSH_TIMEOUT", 15.0) == 15.0
    monkeypatch.setenv("EMBED_BATCH_FLUSH_TIMEOUT", "7.5")
    assert eb_mod._env_float("EMBED_BATCH_FLUSH_TIMEOUT", 15.0) == 7.5


def test_query_vector_cache_evicts_least_recently_used(monkeypatch) -> None:
    monkeypatch.setattr(eb_mod, "_EMBED_BATCH_CACHE_MAX", 2)
    # P3-1: the LRU is an OrderedDict so the touch is O(1) instead of a list
    # scan under the lock embed() also takes.  The eviction ORDER assertions
    # below pin the behaviour; this pins the structure that makes it cheap.
    assert isinstance(eb_mod._query_vec_cache, OrderedDict)
    eb_mod._query_vec_cache.clear()
    eb_mod._cache_put("k1", [1.0, 0.0])
    eb_mod._cache_put("k2", [0.0, 1.0])
    assert eb_mod._cache_get("k1") is not None  # touch k1 -> k2 is now oldest
    eb_mod._cache_put("k3", [1.0, 1.0])
    assert eb_mod._cache_get("k2") is None, "the least recently used entry is evicted"
    assert eb_mod._cache_get("k1") is not None and eb_mod._cache_get("k3") is not None
    assert list(eb_mod._query_vec_cache.keys()) == ["k1", "k3"]
    eb_mod._query_vec_cache.clear()


# ---------------------------------------------------------------------------
# P2-4: an empty chunk payload clears the stored override
# ---------------------------------------------------------------------------


def test_patch_chunk_empty_object_clears_the_override(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("chunky", chunk={"chunk_size": 512})
    meta = dm._read_meta("chunky")
    assert meta is not None and meta["chunk"] == {"chunk_size": 512}

    dm.update_dataset("chunky", {"chunk": {}})
    meta = dm._read_meta("chunky")
    assert meta is not None and "chunk" not in meta, 'PATCH {"chunk": {}} must clear the block'


def test_patch_chunk_null_clears_only_the_named_key(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("chunky2", chunk={"chunk_size": 512, "code_chunk_size": 8192})
    dm.update_dataset("chunky2", {"chunk": {"chunk_size": None}})
    meta = dm._read_meta("chunky2")
    assert meta is not None and meta["chunk"] == {"code_chunk_size": 8192}


def test_patch_chunk_partial_update_keeps_the_rest(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("chunky3", chunk={"chunk_size": 512})
    dm.update_dataset("chunky3", {"chunk": {"chunk_overlap": 64}})
    meta = dm._read_meta("chunky3")
    assert meta is not None and meta["chunk"] == {"chunk_size": 512, "chunk_overlap": 64}
