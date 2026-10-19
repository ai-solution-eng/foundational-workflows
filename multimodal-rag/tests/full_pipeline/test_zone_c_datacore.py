"""Zone C remediation tests (audit 2026-10-16) - data core.

Covers the fixes landed in this zone, one section per finding:

  * P1-3 lane-aware federated merge (merge_federated_results /
    lane_rank_scores / federated_score_kind): RRF rank arithmetic and dense
    cosine are NOT comparable, so the merge ranks within each lane and
    orders by lane-local rank.  Includes the MCP re-sort (which used to
    undo the merge by re-sorting on the raw score).
  * P1-4 _truncate_audio_file returns the TRUNCATED path on success (it
    used to fall through to the original, making the 60s ASR cap a no-op
    for every caller) and the ORIGINAL on every failure branch.
  * P1-6 ingest/dedup resolve the dense lane through the same probe-backed
    resolver search uses (_write_vector_name), instead of the build-time
    vs.vector_name attribute a failed construction probe freezes at None.
  * P1-7 the BM25 df decrement counts the SAME placeholder-stripped text
    ingest counted (bm25.indexable_term_counts), so df/total_len stop
    drifting down on every delete.
  * P2 the Qdrant reference guard checks the tier preprocessed_* /
    original_* keys as well as metadata.source.
  * P2 deferred .ingested_hashes.json writes (dirty set + one flush), with
    unflushed hashes still visible to _is_ingested.
  * P2 _stored_sources streams pages instead of materialising up to 1M
    rows; _delete_original_file batches its set_payload.
  * P2 utils/media_hash.py: stat-keyed (mtime_ns + size) digest caching.

Offline: embedded local Qdrant + stub embedder, no model endpoints.
Tests needing ffmpeg/ffprobe skip cleanly when the binaries are absent.

Run::

    timeout 120 .venv/bin/python -m pytest tests/full_pipeline/test_zone_c_datacore.py -q
"""

import hashlib
import inspect
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.rag_system import (
    MultimodalRAG,
    _write_vector_name,
    federated_score_kind,
    lane_rank_scores,
    merge_federated_results,
)
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils.media_hash import (
    StatKeyedFileHasher,
    digest_cache_stats,
    file_digest,
)

DIM = 8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _stub_embedder_model() -> Any:
    """A deterministic stub embedder model for building real local stores."""
    from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

    class _Stub:
        allowable_modalities: ClassVar[tuple[str, ...]] = ("text",)
        model_name = "stub"
        base_url = "http://stub/v1"
        mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
        chunk_size = 2048
        chunk_overlap = 0
        text_splitter = None

        def __init__(self) -> None:
            self.model = MultiModalEmbeddings(self)

        def embed_query(self, q: Any) -> list[float]:
            return [0.1] * DIM

        async def aembed_query(self, q: Any) -> list[float]:
            return [0.1] * DIM

        async def aembed_documents(self, docs: Any) -> list[list[float]]:
            return [[0.1] * DIM for _ in docs]

    return _Stub()


def _bare_manager(tmp_path: Path) -> Any:
    """A DatasetManager without model wiring, rooted at tmp_path."""
    import threading
    from collections import OrderedDict

    from multimodal_rag.dataset_manager import DatasetManager

    dm = DatasetManager.__new__(DatasetManager)
    dm.datasets_path = tmp_path
    dm.embedder = _stub_embedder_model()
    # OrderedDict, not dict: _get_rag's LRU path calls move_to_end().
    dm._rag_cache = OrderedDict()
    dm._rag_cache_lock = threading.Lock()
    dm._rag_cache_max = 4
    dm._embedder_verified = set()
    dm._embedder_dim_cache = {}
    return dm


def _local_store(tmp_path: Path, coll: str, points: list[dict[str, Any]], ds: str = "ds") -> Any:
    """A local Qdrant collection seeded with *points* + a bound RAG."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from multimodal_rag.vector_store import QdrantVectorStore

    dm = _bare_manager(tmp_path)
    client = QdrantClient(":memory:")
    client.create_collection(coll, vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
    if points:
        client.upsert(
            coll,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": [0.1] * DIM},
                    payload={"page_content": "x", "metadata": meta},
                )
                for meta in points
            ],
        )
    store = QdrantVectorStore(client=client, collection_name=coll, embedding=dm.embedder.model)
    rag = MultimodalRAG(embedder=dm.embedder, vector_store=store)
    dm._rag_cache[ds] = rag
    return dm, rag, client


# ---------------------------------------------------------------------------
# P1-3 - lane-aware federated merge
# ---------------------------------------------------------------------------


def _rrf_doc(text: str) -> dict[str, Any]:
    """A doc as the hybrid lane delivers it: fused score + the lane stamp."""
    return {"text": text, "_score_kind": "rrf"}


def _dense_doc(text: str) -> dict[str, Any]:
    """A doc as a dense-only dataset delivers it (no stamp)."""
    return {"text": text}


def test_score_kind_detects_the_hybrid_lane():
    assert federated_score_kind(_rrf_doc("x")) == "rrf"
    assert federated_score_kind(_dense_doc("x")) == "cosine"
    assert federated_score_kind("bare string") == "cosine"
    assert federated_score_kind({"text": "x", "_score_kind": "cosine"}) == "cosine"


def test_lane_rank_scores_are_monotone_within_each_lane():
    lanes = [("a", 0.1), ("b", 0.9), ("a", 0.5)]
    ranks = lane_rank_scores(lanes)
    assert ranks[2] > ranks[0]
    assert ranks[1] == max(ranks)
    assert len(ranks) == 3


def test_single_lane_pool_keeps_the_historical_score_order():
    # Regression guard: with one lane the merge is identical to the old
    # score-descending sort (the common federated case).
    entries = [
        ("alpha", _dense_doc("a"), 0.5),
        ("beta", _dense_doc("b"), 0.9),
        ("alpha", _dense_doc("c"), 0.9),
    ]
    merged = merge_federated_results(entries, dedup=False)
    assert [round(m[2], 2) for m in merged] == [0.9, 0.9, 0.5]
    assert [m[0] for m in merged][:2] == ["beta", "alpha"], "stable ties keep dataset order"


def test_mixed_lane_pool_does_not_bury_the_hybrid_hits():
    # The P1-3 defect: a hybrid hit's RRF sum is bounded (~2/(k+1) for two
    # lanes) while a dense cosine can be ~1.0, so a raw score sort put EVERY
    # hybrid hit behind EVERY dense hit.
    entries = [
        ("dense_ds", _dense_doc("d1"), 0.95),
        ("dense_ds", _dense_doc("d2"), 0.90),
        ("hybrid_ds", _rrf_doc("h1"), 0.60),
        ("hybrid_ds", _rrf_doc("h2"), 0.50),
    ]
    merged = merge_federated_results(entries, dedup=False)
    order = [doc["text"] for _ds, doc, _s in merged]
    assert set(order[:2]) == {"d1", "h1"}, f"lane tops must interleave, got {order}"
    assert order.index("h1") < order.index("h2"), "hybrid hits keep their own order"
    assert order.index("d1") < order.index("d2"), "dense hits keep their own order"


def test_mixed_lane_merge_is_monotone_in_lane_local_rank():
    entries = [
        ("d", _dense_doc("d1"), 0.9),
        ("d", _dense_doc("d2"), 0.8),
        ("d", _dense_doc("d3"), 0.7),
        ("h", _rrf_doc("h1"), 0.3),
        ("h", _rrf_doc("h2"), 0.2),
        ("h", _rrf_doc("h3"), 0.1),
    ]
    merged = merge_federated_results(entries, dedup=False)
    order = [doc["text"] for _ds, doc, _s in merged]
    assert order[:2] == ["d1", "h1"]
    assert order[2:4] == ["d2", "h2"]
    assert order[4:6] == ["d3", "h3"]


def test_merge_preserves_the_original_scores():
    # Only the ORDER changes - the scores carried out are the real ones.
    entries = [
        ("h", _rrf_doc("h1"), 0.42),
        ("d", _dense_doc("d1"), 0.87),
    ]
    merged = merge_federated_results(entries, dedup=False)
    by_text = {doc["text"]: score for _ds, doc, score in merged}
    assert by_text["h1"] == 0.42
    assert by_text["d1"] == 0.87


def test_merge_still_dedups_before_ranking():
    parent = {"text": "real text", "source": "/a/f.md", "page": 1, "chunk_index": 0}
    twin = dict(parent, _twin=True)
    merged = merge_federated_results([("alpha", parent, 0.9), ("alpha", twin, 0.95)])
    assert len(merged) == 1 and merged[0][1] is parent


def test_merge_empty_pool_is_empty():
    assert merge_federated_results([]) == []


def test_mcp_federated_resort_keeps_lane_order():
    # The MCP formatter re-groups the pool per dataset and re-sorts it; that
    # re-sort must be lane-aware too, or it silently undoes the merge.
    from multimodal_rag.mcp_server import _merge_federated_sections

    dense_payload = {
        "context": "dense ctx",
        "results": [
            {"score": 0.95, "score_kind": "cosine", "text": "d1"},
            {"score": 0.90, "score_kind": "cosine", "text": "d2"},
        ],
    }
    hybrid_payload = {
        "context": "hybrid ctx",
        "results": [
            {"score": 0.60, "score_kind": "rrf", "text": "h1"},
            {"score": 0.50, "score_kind": "rrf", "text": "h2"},
        ],
    }
    out = _merge_federated_sections(
        [("dense_ds", dense_payload), ("hybrid_ds", hybrid_payload)],
        ["dense_ds", "hybrid_ds"],
        [],
        [],
    )
    order = [r["text"] for r in out["results"]]
    assert set(order[:2]) == {"d1", "h1"}, f"lane tops must interleave, got {order}"
    assert order.index("h1") < order.index("h2")
    assert order.index("d1") < order.index("d2")


def test_mcp_resort_single_lane_matches_score_order():
    from multimodal_rag.mcp_server import _merge_federated_sections

    payload = {
        "context": "ctx",
        "results": [
            {"score": 0.2, "score_kind": "cosine", "text": "low"},
            {"score": 0.8, "score_kind": "cosine", "text": "high"},
        ],
    }
    out = _merge_federated_sections([("ds", payload)], ["ds"], [], [])
    assert [r["text"] for r in out["results"]] == ["high", "low"]


# ---------------------------------------------------------------------------
# P1-4 - audio truncation returns the truncated path
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not installed")


def _make_tone(path: Path, seconds: float) -> None:
    """A real (tiny) audio file of the given duration."""
    import subprocess as sp

    sp.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "libmp3lame",
            "-y",
            str(path),
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )


@needs_ffmpeg
def test_truncate_audio_returns_the_truncated_path(tmp_path: Path):
    # P1-4: the success path must return the NEW sibling, not the original.
    from multimodal_rag.dataset_manager import _truncate_audio_file

    src = tmp_path / "long.mp3"
    _make_tone(src, 5.0)
    result = _truncate_audio_file(src, max_seconds=1.0)
    assert result != src, "a successful truncation must return the sibling"
    assert result.name.endswith("_truncated.mp3"), result.name
    assert result.exists() and result.stat().st_size > 0


@needs_ffmpeg
def test_truncate_audio_short_file_returns_original(tmp_path: Path):
    # Negative case: a file already within the cap is untouched.
    from multimodal_rag.dataset_manager import _truncate_audio_file

    src = tmp_path / "short.mp3"
    _make_tone(src, 1.0)
    result = _truncate_audio_file(src, max_seconds=60.0)
    assert result == src
    assert not (tmp_path / "short_truncated.mp3").exists()


@needs_ffmpeg
def test_truncate_audio_is_idempotent(tmp_path: Path):
    # A second call reuses the valid sibling instead of re-encoding.
    from multimodal_rag.dataset_manager import _truncate_audio_file

    src = tmp_path / "again.mp3"
    _make_tone(src, 5.0)
    first = _truncate_audio_file(src, max_seconds=1.0)
    stamp = first.stat().st_mtime_ns
    second = _truncate_audio_file(src, max_seconds=1.0)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "valid sibling must not be re-encoded"


def test_truncate_audio_probe_failure_returns_original(tmp_path: Path):
    # Negative case: an unreadable file keeps the ORIGINAL path (fail-open,
    # never a dangling sibling reference).
    from multimodal_rag.dataset_manager import _truncate_audio_file

    bogus = tmp_path / "notaudio.mp3"
    bogus.write_bytes(b"this is not audio")
    assert _truncate_audio_file(bogus, max_seconds=1.0) == bogus


def test_repair_audio_branch_reports_the_truncated_artifact():
    # The repair pass keys on *_preprocessed siblings but truncation writes a
    # *_truncated sibling; the audio branch must compare against the artifact
    # it actually produces, or it can never report a repair (P1-4 follow-up).
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.repair_preprocessed_media)
    assert "artifact = result if result != tier1 else preproc" in src
    assert "result == artifact" in src


# ---------------------------------------------------------------------------
# P1-6 - probe-backed dense-lane name on the write paths
# ---------------------------------------------------------------------------


class _FakeStore:
    """Stand-in exposing the resolver surface _write_vector_name uses."""

    def __init__(self, resolved: Any = "dense", configured: Any = "dense", probe_raises: bool = False) -> None:
        self.vector_name = configured
        self._resolved = resolved
        self._probe_raises = probe_raises
        self.probe_calls = 0

    def supports_hybrid(self) -> bool:
        self.probe_calls += 1
        if self._probe_raises:
            raise RuntimeError("qdrant unreachable")
        return True

    def _dense_using(self) -> Any:
        return self._resolved


def test_write_vector_name_prefers_the_probe_backed_resolver():
    # P1-6: the write path must use the SAME answer search uses.
    store = _FakeStore(resolved="dense", configured=None)
    assert _write_vector_name(store) == "dense"
    assert store.probe_calls == 1, "the probe must run before resolving"


def test_write_vector_name_returns_none_for_confirmed_unnamed():
    # A probe-confirmed unnamed collection targets the server default.
    store = _FakeStore(resolved=None, configured="stale-name")
    assert _write_vector_name(store) is None


def test_write_vector_name_survives_a_failing_probe():
    # Negative case: a probe failure must never break a write.
    store = _FakeStore(resolved="dense", configured="dense", probe_raises=True)
    assert _write_vector_name(store) == "dense"


def test_write_vector_name_falls_back_for_stores_without_the_resolver():
    class _Bare:
        vector_name = "legacy"

    assert _write_vector_name(_Bare()) == "legacy"


def test_ingest_and_dedup_no_longer_read_the_raw_attribute():
    # Regression guard: both write paths route through the shared resolver.
    import multimodal_rag.rag_system as rs

    src = inspect.getsource(rs.MultimodalRAG)
    assert "vector_name = _write_vector_name(vs)" in src
    assert "vector_name = vs.vector_name" not in src, "the raw attribute read must be gone"


def test_write_vector_name_used_for_a_real_named_collection(tmp_path: Path):
    # End-to-end over a real store: the resolved name is the named lane.
    from qdrant_client import QdrantClient

    from multimodal_rag.vector_store import QdrantVectorStore

    stub = _stub_embedder_model()
    # House pattern: bypass the model wrapper's async plumbing (there is no
    # endpoint in tests) by patching the embed methods on the wrapper itself.
    stub.model.aembed_query = stub.aembed_query  # type: ignore[assignment]
    stub.model.embed_query = stub.embed_query  # type: ignore[assignment]
    client = QdrantClient(":memory:")
    store = MultimodalRAG._build_qdrant_vector_store(
        embedding=stub.model,
        client=client,
        collection_name="zonec",
    )
    assert isinstance(store, QdrantVectorStore)
    assert _write_vector_name(store) == bm25_lane.DENSE_VECTOR_NAME


# ---------------------------------------------------------------------------
# P1-7 - BM25 df decrement counts the same text as ingest
# ---------------------------------------------------------------------------


def test_indexable_text_strips_media_placeholders():
    assert bm25_lane.indexable_text("[Image: photo.jpg]") == ""
    assert bm25_lane.indexable_text("[Video: v.mp4] [0s - 32s]") == ""
    assert "red car" in bm25_lane.indexable_text("[Image: p.jpg]\n[Image description]: a red car")


def test_indexable_term_counts_drops_placeholder_terms():
    # The whole P1-7 defect in one assertion: a bare-placeholder chunk must
    # contribute NO terms, so the decrement cannot subtract terms the
    # increment never added.
    assert bm25_lane.indexable_term_counts("[Image: photo.jpg]") == {}
    # A caption line is KEPT (it is the searchable content of a caption twin),
    # so its own words are counted - by BOTH sides.  What must never be
    # counted is the placeholder's filename tokens.
    counts = bm25_lane.indexable_term_counts("[Image: p.jpg]\n[Image description]: a red car")
    assert counts, "caption text is still indexed"
    assert "p" not in counts and "jpg" not in counts, "placeholder filename tokens must not be counted"
    assert "red" in counts and "car" in counts, "caption words are indexed"


def test_ingest_and_decrement_agree_on_the_same_text():
    # The invariant the fix establishes: one text, one counting rule.
    text = "[Image: scan.png]\n[Image description]: a red car on a road"
    assert bm25_lane.indexable_term_counts(text) == bm25_lane.term_counts(bm25_lane.indexable_text(text))


def test_rag_indexable_wrapper_delegates_to_the_shared_rule():
    # The rag_system name is kept for existing callers but must not drift.
    import multimodal_rag.rag_system as rs

    for sample in ("[Image: a.png]", "plain text here", "[Video: v.mp4] [0s - 1s]\nreal words"):
        assert rs._bm25_indexable_text(sample) == bm25_lane.indexable_text(sample)


def test_decrement_path_uses_the_shared_counter():
    # Regression guard on the call site itself.
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager._forget_bm25_texts)
    assert "indexable_term_counts" in src
    assert "bm25_lane.term_counts(t) for t in texts" not in src


def test_df_round_trip_is_stable_over_a_delete(tmp_path: Path):
    # End-to-end: ingest a doc, delete it, and the df map returns to empty
    # instead of drifting (the P1-7 symptom).
    stats_path = tmp_path / ".bm25_stats.json"
    text = "[Image: scan.png]\n[Image description]: a red car"

    bm25_lane.record_documents(stats_path, [bm25_lane.indexable_term_counts(text)])
    after_ingest = bm25_lane.load_stats(stats_path)
    assert after_ingest["n_docs"] == 1
    assert "scan" not in after_ingest["df"], "placeholder filename tokens must never be counted"

    bm25_lane.forget_documents(stats_path, [bm25_lane.indexable_term_counts(text)])
    after_delete = bm25_lane.load_stats(stats_path)
    assert after_delete["n_docs"] == 0
    assert after_delete["total_len"] == 0, "total_len must return to zero, not drift"
    assert after_delete["df"] == {}, "df must be empty after the round trip"


def test_raw_text_decrement_would_have_drifted(tmp_path: Path):
    # Documents the defect the fix removes (negative case for the OLD rule).
    stats_path = tmp_path / ".bm25_stats.json"
    caption = "[Image: scan.png]\n[Image description]: a red car"
    # A DIFFERENT document supplies real df for the word "image".
    bm25_lane.record_documents(stats_path, [bm25_lane.term_counts("image of a chart")])
    bm25_lane.record_documents(stats_path, [bm25_lane.indexable_term_counts(caption)])
    before = bm25_lane.load_stats(stats_path)["df"].get("image", 0)

    # The OLD decrement counted the raw text and would subtract "image" here.
    bm25_lane.forget_documents(stats_path, [bm25_lane.term_counts(caption)])
    after = bm25_lane.load_stats(stats_path)["df"].get("image", 0)
    assert after == before - 1, "demonstrates the cross-document subtraction the fix prevents"


# ---------------------------------------------------------------------------
# P2 - Qdrant reference guard mirrors the tier keys
# ---------------------------------------------------------------------------


def test_reference_guard_checks_tier_keys_on_qdrant(tmp_path: Path):
    # A point that references the artifact ONLY through preprocessed_image
    # must keep the file alive (the InMemory branch always did; Qdrant did
    # not - the P2 asymmetry).
    files_dir = tmp_path / "ds" / "files"
    files_dir.mkdir(parents=True)
    artifact = files_dir / "abc_preprocessed.jpg"
    artifact.write_bytes(b"jpegbytes")

    dm, rag, _client = _local_store(
        tmp_path,
        "refguard",
        [
            {
                "source": "/somewhere/else/original.jpg",
                "preprocessed_image": f"file://{artifact}",
            }
        ],
    )
    assert dm._file_referenced(rag, [str(artifact)]) is True
    assert dm._file_referenced(rag, [str(files_dir / "unrelated.jpg")]) is False


def test_reference_guard_matches_tier_key_given_as_file_url(tmp_path: Path):
    # The caller may pass the ref in either shape; both must match.
    files_dir = tmp_path / "ds" / "files"
    files_dir.mkdir(parents=True)
    artifact = files_dir / "def_original.jpg"
    artifact.write_bytes(b"x")

    dm, rag, _client = _local_store(
        tmp_path,
        "refguard3",
        [{"source": "/elsewhere.jpg", "original_image": f"file://{artifact}"}],
    )
    assert dm._file_referenced(rag, [str(artifact)]) is True
    assert dm._file_referenced(rag, [f"file://{artifact}"]) is True


def test_reference_guard_matches_source_as_well(tmp_path: Path):
    # Positive control: the original metadata.source check still works.
    files_dir = tmp_path / "ds" / "files"
    files_dir.mkdir(parents=True)
    original = files_dir / "abc_orig.jpg"
    original.write_bytes(b"x")

    dm, rag, _client = _local_store(tmp_path, "refguard2", [{"source": str(original)}])
    assert dm._file_referenced(rag, [str(original)]) is True


def test_reference_guard_is_conservative_on_error():
    # Negative case: any store failure keeps the file.
    from multimodal_rag.dataset_manager import DatasetManager

    class _Client:
        def scroll(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("qdrant down")

    class _Broken:
        def __init__(self) -> None:
            self._client = _Client()
            self.collection_name = "x"

    dm = DatasetManager.__new__(DatasetManager)
    rag = MultimodalRAG.__new__(MultimodalRAG)
    object.__setattr__(rag, "vector_store", _Broken())
    assert dm._file_referenced(rag, ["/data/datasets/x/files/a.jpg"]) is True


# ---------------------------------------------------------------------------
# P2 - deferred .ingested_hashes.json writes
# ---------------------------------------------------------------------------


def _isolated_dirty_state():
    """Scrub this process deferred-hash bookkeeping around a test."""
    from contextlib import contextmanager

    import multimodal_rag.dataset_manager as dmmod

    @contextmanager
    def _cm():
        with dmmod._ingested_hashes_dirty_lock:
            saved = dict(dmmod._ingested_hashes_dirty)
            dmmod._ingested_hashes_dirty.clear()
        try:
            yield dmmod
        finally:
            with dmmod._ingested_hashes_dirty_lock:
                dmmod._ingested_hashes_dirty.clear()
                dmmod._ingested_hashes_dirty.update(saved)

    return _cm()


def test_mark_ingested_defers_and_flush_persists(tmp_path: Path):
    # P2: N marks = ONE file write, and the hashes still land on disk.
    with _isolated_dirty_state() as dmmod:
        dm = _bare_manager(tmp_path)
        for i in range(25):
            dm._mark_ingested("ds", f"hash{i:03d}")
        path = dm._ingested_hashes_path("ds")
        assert not path.exists(), "nothing may be written before the flush"
        dmmod._flush_ingested_hash_writes()
        assert path.exists()
        on_disk = dmmod._load_ingested_hashes(path)
        assert len(on_disk) == 25


def test_is_ingested_sees_unflushed_hashes(tmp_path: Path):
    # Correctness guard: deferral must not change batch semantics.
    with _isolated_dirty_state():
        dm = _bare_manager(tmp_path)
        dm._mark_ingested("ds", "abc")
        assert dm._is_ingested("ds", "abc") is True, "pending hash must be visible"
        assert dm._is_ingested("ds", "zzz") is False


def test_flush_merges_rather_than_clobbers(tmp_path: Path):
    # Negative case: another pod's hashes must survive our flush.
    with _isolated_dirty_state() as dmmod:
        dm = _bare_manager(tmp_path)
        path = dm._ingested_hashes_path("ds")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('["other-pod-hash"]', encoding="utf-8")
        dmmod._ingested_hashes_cache.pop(path, None)

        dm._mark_ingested("ds", "mine")
        dmmod._flush_ingested_hash_writes()
        on_disk = dmmod._load_ingested_hashes(path)
        assert on_disk == {"other-pod-hash", "mine"}


def test_flush_is_a_noop_without_dirty_entries(tmp_path: Path):
    with _isolated_dirty_state() as dmmod:
        dm = _bare_manager(tmp_path)
        path = dm._ingested_hashes_path("ds")
        dmmod._flush_ingested_hash_writes()
        assert not path.exists()


def test_clear_ingested_hashes_drops_pending_entries(tmp_path: Path):
    # Recreate safety: a pending hash must not resurrect after a clear.
    with _isolated_dirty_state() as dmmod:
        dm = _bare_manager(tmp_path)
        dm._mark_ingested("ds", "pending")
        dm._clear_ingested_hashes("ds")
        dmmod._flush_ingested_hash_writes()
        assert dm._is_ingested("ds", "pending") is False
        assert not dm._ingested_hashes_path("ds").exists()


def test_flush_ingested_hashes_now_persists_one_dataset(tmp_path: Path):
    # The single-file ingest path must not leave its hash in memory.
    with _isolated_dirty_state() as dmmod:
        dm = _bare_manager(tmp_path)
        dm._mark_ingested("ds", "solo")
        dm._flush_ingested_hashes_now("ds")
        assert dmmod._load_ingested_hashes(dm._ingested_hashes_path("ds")) == {"solo"}


def test_batch_calls_the_deferred_flushes():
    # Regression guard: the batch finally must flush BOTH sidecars.
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager.add_files_batch)
    assert "_flush_hash_index_writes()" in src
    assert "_flush_ingested_hash_writes()" in src
    assert "finally:" in src


# ---------------------------------------------------------------------------
# P2 - perf: streaming source scan + batched set_payload
# ---------------------------------------------------------------------------


def test_stored_sources_returns_every_source(tmp_path: Path):
    # P2: the scan used to materialise up to 1M rows via scroll_documents,
    # which also silently truncated at that cap.  It must now stream pages
    # and return every source.
    dm, _rag, _client = _local_store(
        tmp_path,
        "sourcescan",
        [{"source": f"s3://b/p/obj{i}.txt"} for i in range(7)],
    )

    sources = dm._stored_sources("ds", ["s3://b/p/"])
    assert len(sources) == 7
    assert all(s.startswith("s3://b/p/obj") for s in sources)
    # A prefix outside the data returns nothing (negative case).
    assert dm._stored_sources("ds", ["s3://other/"]) == set()


def test_stored_sources_empty_prefixes_is_free(tmp_path: Path):
    dm = _bare_manager(tmp_path)
    assert dm._stored_sources("ds", []) == set()


def test_stored_sources_projects_only_source_keys():
    # Regression guard: the scan must stay on the projected selector and
    # must not go back to the list-building helper.  Assertions are made
    # against the EXECUTABLE lines only, so the explanatory comment above
    # the loop cannot satisfy (or defeat) them.
    import ast

    from multimodal_rag.dataset_manager import DatasetManager

    tree = ast.parse(inspect.getsource(DatasetManager._stored_sources).lstrip())
    code = "\n".join(
        line
        for line in inspect.getsource(DatasetManager._stored_sources).splitlines()
        if not line.strip().startswith("#")
    )
    assert "_source_scan_payload_selector()" in code
    assert "scroll_documents(" not in code, "the list-building helper must not be used"
    assert "limit=1_000_000" not in code.replace("limit=1024", ""), "the 1M cap must be gone"
    # The loop must page until the cursor is exhausted.
    assert "if offset is None" in code and "break" in code
    assert isinstance(tree, ast.Module)


def test_delete_original_file_batches_set_payload():
    # P2: one set_payload for N ids instead of N calls.
    from multimodal_rag.dataset_manager import DatasetManager

    src = inspect.getsource(DatasetManager._delete_original_file)
    assert "for doc_id in doc_ids" not in src, "the per-point loop must be gone"
    assert "points=clean_ids" in src


def test_delete_original_file_clears_the_key_in_one_call(tmp_path: Path):
    # Behavioural check: every id in one call, key nulled on all of them.
    orig = tmp_path / "ds" / "files" / "abc.jpg"
    orig.parent.mkdir(parents=True, exist_ok=True)
    orig.write_bytes(b"x")

    dm, _rag, client = _local_store(
        tmp_path,
        "delorig",
        [{"original_image": "file:///x.jpg"} for _ in range(3)],
    )
    ids = [str(pt.id) for pt in client.scroll("delorig", limit=10)[0]]
    assert len(ids) == 3

    dm._delete_original_file("ds", str(orig), ids, "original_image")

    assert not orig.exists()
    got = client.retrieve("delorig", ids=ids, with_payload=True)
    assert all((pt.payload or {}).get("metadata", {}).get("original_image") is None for pt in got)


# ---------------------------------------------------------------------------
# P2 - stat-keyed media hashing (utils/media_hash.py)
# ---------------------------------------------------------------------------


def test_media_hash_caches_by_stat(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    hasher = StatKeyedFileHasher()
    first = hasher.digest(f)
    second = hasher.digest(f)
    assert first == second
    assert hasher.files_hashed == 1, "an unchanged file must not be re-read"
    assert first == hashlib.sha256(b"hello").hexdigest()


def test_media_hash_misses_after_a_rewrite(tmp_path: Path):
    # The P2 defect: a path-keyed cache served a stale digest forever.
    f = tmp_path / "b.bin"
    f.write_bytes(b"first")
    hasher = StatKeyedFileHasher()
    before = hasher.digest(f)
    time.sleep(0.002)  # guarantee a distinct mtime_ns on coarse clocks
    f.write_bytes(b"second!!")  # same length, different content
    after = hasher.digest(f)
    assert before != after
    assert hasher.files_hashed == 2
    assert after == hashlib.sha256(b"second!!").hexdigest()


def test_media_hash_misses_on_a_same_size_rewrite(tmp_path: Path):
    # Negative case for size-only keying: equal size, different content.
    f = tmp_path / "c.bin"
    f.write_bytes(b"aaaa")
    hasher = StatKeyedFileHasher()
    a = hasher.digest(f)
    f.write_bytes(b"bbbb")
    b = hasher.digest(f)
    assert a != b, "a same-size rewrite must be a cache miss"


def test_media_hash_stat_failure_propagates(tmp_path: Path):
    hasher = StatKeyedFileHasher()
    with pytest.raises(OSError):
        hasher.digest(tmp_path / "missing.bin")


def test_media_hash_lru_is_bounded(tmp_path: Path):
    hasher = StatKeyedFileHasher(max_entries=3)
    for i in range(6):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(f"data{i}".encode())
        hasher.digest(p)
    assert len(hasher) <= 3


def test_media_hash_reuses_entries_like_an_lru(tmp_path: Path):
    hasher = StatKeyedFileHasher(max_entries=2)
    a, b, c = (tmp_path / n for n in ("a.bin", "b.bin", "c.bin"))
    for path, data in ((a, b"a"), (b, b"b")):
        path.write_bytes(data)
        hasher.digest(path)
    hasher.digest(a)  # touch a, so b becomes the eviction candidate
    c.write_bytes(b"c")
    hasher.digest(c)
    hashed_before = hasher.files_hashed
    hasher.digest(a)
    assert hasher.files_hashed == hashed_before, "a must still be cached"


def test_media_hash_clear_and_empty_cache(tmp_path: Path):
    hasher = StatKeyedFileHasher()
    assert len(hasher) == 0
    f = tmp_path / "g.bin"
    f.write_bytes(b"x")
    hasher.digest(f)
    assert len(hasher) == 1
    hasher.clear()
    assert len(hasher) == 0


def test_media_hash_module_level_helper_and_stats(tmp_path: Path):
    f = tmp_path / "d.bin"
    f.write_bytes(b"payload")
    assert file_digest(f) == hashlib.sha256(b"payload").hexdigest()
    stats = digest_cache_stats()
    assert stats["entries"] >= 1 and stats["files_hashed"] >= 1


def test_media_hash_spy_digest_is_used(tmp_path: Path):
    calls: list[int] = []

    class _Spy:
        def __init__(self) -> None:
            calls.append(1)
            self._inner = hashlib.sha256()

        def update(self, data: bytes) -> None:
            self._inner.update(data)

        def hexdigest(self) -> str:
            return self._inner.hexdigest()

    f = tmp_path / "e.bin"
    f.write_bytes(b"x")
    hasher = StatKeyedFileHasher(digest_fn=_Spy)
    hasher.digest(f)
    hasher.digest(f)
    assert len(calls) == 1, "the injected digest must be used once (then cached)"
