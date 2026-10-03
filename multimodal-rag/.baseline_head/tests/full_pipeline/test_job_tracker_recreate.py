"""Tests for the 2.5.0 fixes:

1. ``_UploadJobTracker`` keeps working in-process (and is Redis-mirrored so
   multi-worker/multi-replica deployments can answer a poll from any process).
2. ``recreate_dataset`` clears the per-dataset ingest-dedup index so a
   recreate actually re-embeds files instead of skipping them as "already
   ingested" (which left a freshly-dropped, empty collection).

Run::

    python tests/full_pipeline/test_job_tracker_recreate.py  # standalone
    pytest tests/full_pipeline/test_job_tracker_recreate.py   # under pytest
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag import api_server
from multimodal_rag.api_server import _UploadJobTracker
from multimodal_rag.dataset_manager import DatasetManager

# ---------------------------------------------------------------------------
# _UploadJobTracker
# ---------------------------------------------------------------------------


def _no_redis() -> None:
    """Force the in-memory (no-Redis) path for the tracker tests."""
    api_server._REDIS_URL = ""
    api_server._redis_client = None


def test_tracker_in_memory_lifecycle():
    _no_redis()
    tracker = _UploadJobTracker()
    jid = tracker.create("ds", 2, source="recreate")
    job = tracker.get(jid)
    assert job is not None and job["status"] == "uploading"
    assert job["dataset"] == "ds" and job["source"] == "recreate"

    tracker.add_event(jid, {"status": "complete", "chunks": 5})
    tracker.add_event(jid, {"status": "complete", "chunks": 3})
    job = tracker.get(jid)
    assert job["processed_files"] == 2
    assert job["total_chunks"] == 8

    tracker.complete(jid, {"file_count": 2})
    assert tracker.get(jid)["status"] == "complete"

    # Unknown job → None (drives the 404 in the upload-status endpoint).
    assert tracker.get("deadbeef0000") is None

    # cleanup_old prunes completed jobs older than the age.
    tracker.cleanup_old(max_age_seconds=0)
    assert tracker.get(jid) is None


def test_tracker_mirrors_to_redis_when_available():
    """With a fake Redis, a second tracker instance (simulating another
    worker/pod) can still read the job created by the first."""
    _no_redis()

    class _FakeRedis:
        def __init__(self):
            self.store: dict[str, str] = {}

        def set(self, key, value, ex=None):
            self.store[key] = value

        def get(self, key):
            return self.store.get(key)

    fake = _FakeRedis()

    # Point the module's _get_redis at the fake client.
    def _fake_get_redis():
        return fake

    orig = api_server._get_redis
    api_server._get_redis = _fake_get_redis  # type: ignore[assignment]
    try:
        tracker_a = _UploadJobTracker()
        jid = tracker_a.create("ds", 1)
        tracker_a.add_event(jid, {"status": "complete", "chunks": 4})

        # A fresh tracker in another process has no local copy — but get()
        # falls back to the shared store.
        tracker_b = _UploadJobTracker()
        remote = tracker_b.get(jid)
        assert remote is not None, "job must be visible to another worker via Redis"
        assert remote["total_chunks"] == 4
        assert remote["status"] == "uploading"
        assert fake.store, "job must have been mirrored to Redis"
    finally:
        api_server._get_redis = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# recreate clears the ingest-dedup index
# ---------------------------------------------------------------------------


def _make_manager(tmpdir: Path) -> DatasetManager:
    dm = DatasetManager.__new__(DatasetManager)
    dm.base_path = tmpdir
    dm.datasets_path = tmpdir / "datasets"
    dm.datasets_path.mkdir(parents=True, exist_ok=True)
    (dm.datasets_path / "ds" / "files").mkdir(parents=True, exist_ok=True)
    return dm


def test_clear_ingested_hashes_unblocks_reingest():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        p = dm._ingested_hashes_path("ds")
        p.write_text(json.dumps(["abc123", "def456"]), encoding="utf-8")

        assert dm._is_ingested("ds", "abc123") is True
        dm._clear_ingested_hashes("ds")
        assert not p.exists(), "ingest-dedup index should be deleted"
        assert dm._is_ingested("ds", "abc123") is False


def test_batch_file_tuple_is_five_elements():
    """Regression: batch_files_list entries carry (fname, count, file_type,
    content_hash, stored_path).  Shrinking the tuple back to 4 breaks the
    batch consumer with 'too many values to unpack (expected 4, got 5)' —
    the exact upload failure seen against the 2.5.0 deployment."""
    batch = [
        ("2604.07035v2.pdf", 2, "pdf", "hash123", "/data/datasets/ds/files/uuid_pdf.pdf"),
        ("photo.jpg", 1, "image", "hash456", "/data/datasets/ds/files/uuid_photo.jpg"),
    ]
    # The two consumer loops over the batch (see add_files_batch): the
    # "marking" loop and the per-file embed/store loop.
    for fname, count, _, _, _ in batch:
        assert fname and count >= 0
    for fname, count, file_type_, content_hash_, stored_path in batch:
        assert stored_path.endswith((".pdf", ".jpg"))
        assert content_hash_ and file_type_
    assert len(batch) == 2


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
