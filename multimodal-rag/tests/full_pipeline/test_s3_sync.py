"""Offline tests for S3 sync with pruning (roadmap feature 5).

Covers:

  * ``_sync_prefixes`` — prefix normalization and single-object rejection.
  * ``add_urls_batch(sync_dry_run=True)`` — the diff report (would-ingest /
    would-prune) without touching anything.
  * ``_prune_sources`` — real, against embedded local Qdrant: points whose
    ``metadata.source`` sits under a synced prefix but is absent from the
    listing are deleted; out-of-scope points survive; the document counter
    is decremented.
  * ``add_urls_batch(sync=True)`` — end-to-end with stubbed S3/ingest:
    URLs without stored points are force-re-ingested (heals pruned-and-
    reappeared files), and the prune result is attached.

No S3, no models — listing/download/ingest are stubbed; Qdrant runs in
embedded local mode.

Run::

    python tests/full_pipeline/test_s3_sync.py    # standalone
    pytest tests/full_pipeline/test_s3_sync.py    # under pytest
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.dataset_manager as dm_module
from multimodal_rag.dataset_manager import DatasetManager, _sync_prefixes
from multimodal_rag.vector_store import QdrantVectorStore

DIM = 8
COLL = "s3_sync_test"

PFX = "s3://bucket/pfx/"
EXPANDED = [
    "s3://bucket/pfx/a.pdf",
    "s3://bucket/pfx/b.log",
    "s3://bucket/pfx/new.txt",  # not yet in the dataset
]
GONE = "s3://bucket/pfx/gone.txt"  # stored, but no longer in the listing
OUTSIDE = "s3://bucket/other/keep.txt"  # stored, outside the synced prefix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Restore:
    """Save/restore module-global monkeypatches (standalone-runner friendly)."""

    def __init__(self) -> None:
        self._saved: list[tuple[Any, Any, Any]] = []

    def patch(self, obj: Any, name: str, value: Any) -> None:
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)


def _client():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(":memory:")
    try:
        client.delete_collection(COLL)
    except Exception:
        pass
    client.create_collection(COLL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    return client


def _upsert(client, source: str) -> str:
    from qdrant_client.models import PointStruct

    pid = uuid.uuid4().hex
    client.upsert(
        COLL,
        points=[PointStruct(id=pid, vector=[0.1] * DIM, payload={"page_content": "t", "metadata": {"source": source}})],
        wait=True,
    )
    return pid


def _dm_with(client) -> tuple[DatasetManager, dict[str, Any]]:
    dm = DatasetManager.__new__(DatasetManager)
    vs = QdrantVectorStore(client, COLL, embedding=None)
    rag = type("_RagStub", (), {"vector_store": vs})()
    dm._get_rag = lambda ds, check_embedder=True: rag  # type: ignore[method-assign]
    recorded: dict[str, Any] = {"decrements": [], "ingested": [], "force": None}
    dm._decrement_count = lambda ds, n: recorded["decrements"].append(n)  # type: ignore[method-assign]

    def _add_files_batch(ds_name, file_entries, progress_callback=None, batch_score=128.0, force_names=None):
        recorded["ingested"].append([name for _, name in file_entries])
        recorded["force"] = set(force_names or ())
        return {"status": "ok", "file_count": len(file_entries), "files": []}

    dm.add_files_batch = _add_files_batch  # type: ignore[method-assign]
    return dm, recorded


# ---------------------------------------------------------------------------
# _sync_prefixes
# ---------------------------------------------------------------------------


def test_sync_prefixes_normalization():
    assert _sync_prefixes(["s3://bucket/pfx", "s3://bucket/p2/?x=1"]) == [
        "s3://bucket/pfx/",
        "s3://bucket/p2/",
    ]


def test_sync_prefixes_rejects_single_object():
    try:
        _sync_prefixes(["s3://bucket/file.pdf"])
        raise AssertionError("expected ValueError for a single-object URL")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_dry_run_reports_diff_without_ingesting():
    rs = _Restore()
    with tempfile.TemporaryDirectory():
        try:
            client = _client()
            dm, recorded = _dm_with(client)
            _upsert(client, GONE)
            _upsert(client, OUTSIDE)
            rs.patch(dm_module, "_expand_urls", lambda urls: list(EXPANDED))
            rs.patch(dm, "_stored_sources", lambda ds, prefixes: {EXPANDED[0].rstrip("/"), GONE})

            plan = dm.add_urls_batch("ds", [PFX], sync_dry_run=True)
        finally:
            rs.restore()

    assert plan["status"] == "dry-run"
    assert plan["would_ingest"] == [EXPANDED[2], "s3://bucket/pfx/b.log"] or EXPANDED[2] in plan["would_ingest"]
    assert GONE in plan["would_prune"]
    assert OUTSIDE not in plan["would_prune"], "out-of-scope sources are never pruned"
    assert recorded["ingested"] == [], "dry run must not ingest"


# ---------------------------------------------------------------------------
# _prune_sources (real, local Qdrant)
# ---------------------------------------------------------------------------


def test_prune_sources_deletes_only_stale():
    client = _client()
    keep1, keep2, _gone, outside = (
        _upsert(client, EXPANDED[0]),
        _upsert(client, EXPANDED[1]),
        _upsert(client, GONE),
        _upsert(client, OUTSIDE),
    )
    dm, recorded = _dm_with(client)

    result = dm._prune_sources("ds", [PFX], expected={EXPANDED[0], EXPANDED[1], EXPANDED[2]})

    assert result["pruned_points"] == 1
    assert result["pruned_sources"] == [GONE]
    assert recorded["decrements"] == [1]
    remaining = {pid for pid, _ in dm.scroll_documents("ds", limit=100)}
    assert remaining == {keep1, keep2, outside}


def test_prune_sources_nothing_stale():
    client = _client()
    _upsert(client, EXPANDED[0])
    dm, recorded = _dm_with(client)
    result = dm._prune_sources("ds", [PFX], expected={EXPANDED[0], EXPANDED[1]})
    assert result["pruned_points"] == 0
    assert result["pruned_sources"] == []
    assert recorded["decrements"] == []


# ---------------------------------------------------------------------------
# sync end-to-end (stubbed S3 + ingest)
# ---------------------------------------------------------------------------


def test_sync_forces_new_urls_and_prunes():
    rs = _Restore()
    with tempfile.TemporaryDirectory() as td:
        try:
            client = _client()
            dm, recorded = _dm_with(client)
            _upsert(client, EXPANDED[0])
            _upsert(client, GONE)
            rs.patch(dm_module, "_expand_urls", lambda urls: list(EXPANDED))
            rs.patch(dm_module, "_download_url", lambda url: str(Path(td) / "dl"))

            # stored sources: only a.pdf has points (b.log was pruned before,
            # new.txt never seen) — both must be force-re-ingested
            rs.patch(dm, "_stored_sources", lambda ds, prefixes: {EXPANDED[0]})

            result = dm.add_urls_batch("ds", [PFX], sync=True)
        finally:
            rs.restore()

    assert recorded["ingested"] == [[Path(u).name for u in EXPANDED]]
    assert recorded["force"] == {Path(EXPANDED[1]).name, Path(EXPANDED[2]).name}
    assert result["sync"]["pruned_sources"] == [GONE]
    assert result["sync"]["pruned_points"] == 1


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
