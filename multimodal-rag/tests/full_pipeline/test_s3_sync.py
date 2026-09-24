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


def _fps(urls: list[str]) -> dict[str, dict[str, Any]]:
    """Fingerprint map in the shape `_list_s3_prefix_fingerprints` returns
    (deterministic per-URL etag/size for the tests)."""
    return {u: {"etag": f"etag-{i}", "size": 100 + i} for i, u in enumerate(urls)}


def _dm_with_state_dir(dm: DatasetManager, td: str) -> None:
    """Point the DatasetManager stub's dataset dir at *td* so the
    watched-state sidecar (files/.watched_state.json) has a real home."""
    base = Path(td)
    (base / "ds" / "files").mkdir(parents=True, exist_ok=True)
    dm.datasets_path = base  # type: ignore[attr-defined]


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
    dm._get_rag = lambda dataset_name, check_embedder=True: rag  # type: ignore[method-assign]
    recorded: dict[str, Any] = {"decrements": [], "ingested": [], "force": None}
    dm._decrement_count = lambda name, n, file_type=None: recorded["decrements"].append(n)  # type: ignore[method-assign]

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
    with tempfile.TemporaryDirectory() as td:
        try:
            client = _client()
            dm, recorded = _dm_with(client)
            _dm_with_state_dir(dm, td)
            _upsert(client, GONE)
            _upsert(client, OUTSIDE)
            rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: _fps(EXPANDED))
            rs.patch(dm, "_stored_sources", lambda ds, prefixes: {EXPANDED[0].rstrip("/"), GONE})

            plan = dm.add_urls_batch("ds", [PFX], sync_dry_run=True)
        finally:
            rs.restore()

    assert plan["status"] == "dry-run"
    assert plan["would_ingest"] == [EXPANDED[2], "s3://bucket/pfx/b.log"] or EXPANDED[2] in plan["would_ingest"]
    assert GONE in plan["would_prune"]
    assert OUTSIDE not in plan["would_prune"], "out-of-scope sources are never pruned"
    assert recorded["ingested"] == [], "dry run must not ingest"
    # Watched-state split (feature: watched sources): an empty state skips
    # nothing — every listed object is a would-download, none unchanged.
    assert plan["would_download"] == sorted({u.rstrip("/") for u in EXPANDED})
    assert plan["unchanged"] == []


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
            _dm_with_state_dir(dm, td)
            _upsert(client, EXPANDED[0])
            _upsert(client, GONE)
            rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: _fps(EXPANDED))
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


# ---------------------------------------------------------------------------
# Watched-source skip state (feature: watched sources)
# ---------------------------------------------------------------------------
# The state file is files/.watched_state.json next to .hashes.json —
# key = canonical s3://bucket/key, value = {etag, size} of the last
# successfully ingested version.  Sync consults it before downloading.

STATE_PATH = Path("ds/files/.watched_state.json")


def _read_state(td: str) -> dict[str, Any]:
    p = Path(td) / STATE_PATH
    if not p.exists():
        return {}
    import json as _json

    return _json.loads(p.read_text(encoding="utf-8"))


def _sync_env(
    td: str,
    fingerprints: dict[str, dict[str, Any]],
    stored: set[str] | None = None,
    batch_files: list[dict[str, Any]] | None = None,
    prune: dict[str, Any] | None = None,
):
    """Build a dm + _Restore wired for one watched-state sync test.

    Patches the fingerprint listing to return *fingerprints* and the batch
    result to report *batch_files* outcomes.  Returns (rs, dm, recorded).
    """
    rs = _Restore()
    client = _client()
    dm, recorded = _dm_with(client)
    _dm_with_state_dir(dm, td)
    rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: fingerprints)
    rs.patch(dm_module, "_download_url", lambda url: str(Path(td) / "dl"))
    rs.patch(dm, "_stored_sources", lambda ds, prefixes: set(stored or ()))
    if prune is not None:
        rs.patch(dm, "_prune_sources", lambda ds, prefixes, expected: prune)

    def _add_files_batch(ds_name, file_entries, progress_callback=None, batch_score=128.0, force_names=None):
        recorded["ingested"].append([name for _, name in file_entries])
        recorded["force"] = set(force_names or ())
        return {"status": "ok", "file_count": len(file_entries), "files": list(batch_files or [])}

    dm.add_files_batch = _add_files_batch  # type: ignore[method-assign]
    return rs, dm, recorded


def test_watched_state_unchanged_objects_are_skipped():
    """After a successful sync, the state file holds etag+size per object;
    the next sync with identical fingerprints downloads NOTHING."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED)
        rs, dm, recorded = _sync_env(
            td,
            fps,
            batch_files=[{"file": Path(u).name, "chunks": 3} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)
            state = _read_state(td)
            assert {k: {"etag": v["etag"], "size": v["size"]} for k, v in state.items()} == {
                u.rstrip("/"): fps[u] for u in EXPANDED
            }, "successful ingest records the fingerprints"

            # Second tick, identical listing: nothing re-downloaded, nothing ingested.
            recorded["ingested"].clear()
            result = dm.add_urls_batch("ds", [PFX], sync=True)
        finally:
            rs.restore()

    assert recorded["ingested"] == [[]], "unchanged objects are dropped before download"
    assert result["sync"]["pruned_points"] == 0, "unchanged objects are still in scope (never pruned)"
    assert result["sync"]["pruned_sources"] == []


def test_watched_state_etag_change_redownloads():
    """Content changed under the same key (new ETag): the object re-downloads
    and its state entry updates to the new fingerprint."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED)
        rs, dm, recorded = _sync_env(
            td,
            fps,
            batch_files=[{"file": Path(u).name, "chunks": 3} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)

            changed = dict(fps)
            changed[EXPANDED[1]] = {"etag": "etag-CHANGED", "size": 999}
            rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: changed)
            recorded["ingested"].clear()
            result = dm.add_urls_batch("ds", [PFX], sync=True)
            # Read the sidecar INSIDE the tempdir's lifetime (it is deleted
            # with the directory on exit).
            state = _read_state(td)
        finally:
            rs.restore()

    assert recorded["ingested"] == [[Path(EXPANDED[1]).name]], "only the changed object downloads"
    assert recorded["force"] == {Path(EXPANDED[1]).name}, "changed objects are force-re-ingested"
    assert state[EXPANDED[1].rstrip("/")]["etag"] == "etag-CHANGED"
    assert state[EXPANDED[0].rstrip("/")]["etag"] == fps[EXPANDED[0]]["etag"], "unchanged entries untouched"
    assert result["sync"]["pruned_points"] == 0


def test_watched_state_new_object_ingests_and_records():
    """A new object under the prefix ingests alongside skipped unchanged
    ones and joins the state afterwards."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED[:2])
        rs, dm, recorded = _sync_env(
            td,
            fps,
            batch_files=[{"file": Path(u).name, "chunks": 3} for u in EXPANDED[:2]],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)
            assert set(_read_state(td)) == {u.rstrip("/") for u in EXPANDED[:2]}

            with_new = dict(fps)
            with_new[EXPANDED[2]] = {"etag": "etag-new", "size": 777}
            rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: with_new)
            rs.patch(
                dm_module,
                "_download_url",
                lambda url: str(Path(td) / "dl"),
            )

            # The batch now reports all three outcomes (two dedup-skips + the new one).
            def _batch_with_skips(ds_name, file_entries, progress_callback=None, batch_score=128.0, force_names=None):
                recorded["ingested"].append([name for _, name in file_entries])
                return {
                    "status": "ok",
                    "file_count": len(file_entries),
                    "files": [{"file": name, "chunks": 0, "deduplicated": True} for _, name in file_entries[:2]]
                    + [{"file": Path(EXPANDED[2]).name, "chunks": 5}],
                }

            rs.patch(dm, "add_files_batch", _batch_with_skips)
            recorded["ingested"].clear()
            dm.add_urls_batch("ds", [PFX], sync=True)
            state = _read_state(td)
        finally:
            rs.restore()

    assert recorded["ingested"] == [[Path(EXPANDED[2]).name]], "only the new object moves"
    assert state[EXPANDED[2].rstrip("/")]["etag"] == "etag-new"
    assert len(state) == 3


def test_watched_state_failed_object_is_not_recorded():
    """An object whose ingest errored (or produced nothing) gets NO state
    entry — the next tick re-downloads it instead of silently skipping."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED)
        rs, dm, _recorded = _sync_env(
            td,
            fps,
            batch_files=[
                {"file": Path(EXPANDED[0]).name, "chunks": 3},
                {"file": Path(EXPANDED[1]).name, "chunks": 0, "error": "embedder exploded"},
                {"file": Path(EXPANDED[2]).name, "chunks": 0},
            ],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)
            state = _read_state(td)
        finally:
            rs.restore()

    assert set(state) == {EXPANDED[0].rstrip("/")}, "only the succeeded object is recorded"
    # And a failed/zero-chunk object would re-download next tick:
    with tempfile.TemporaryDirectory() as td2:
        rs2, dm2, _recorded2 = _sync_env(
            td2,
            fps,
            stored=set(),
            batch_files=[{"file": Path(u).name, "chunks": 1} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            plan = dm2.add_urls_batch("ds", [PFX], sync_dry_run=True)
        finally:
            rs2.restore()
    assert plan["unchanged"] == []
    assert plan["would_download"] == sorted({u.rstrip("/") for u in EXPANDED})


def test_watched_state_pruned_source_is_forgotten():
    """A source the prune deleted upstream loses its state entry, so a
    pruned-and-reappeared object re-ingests (the documented heal path)."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED[:2] + [GONE])
        rs, dm, _recorded = _sync_env(
            td,
            fps,
            batch_files=[{"file": Path(u).name, "chunks": 2} for u in EXPANDED[:2] + [GONE]],
            # Upstream deleted GONE between ticks:
            prune={"pruned_points": 1, "pruned_sources": [GONE.rstrip("/")], "scope": [PFX]},
        )
        try:
            # Seed the state as if a previous sync had ingested GONE.
            dm._watched_state_update("ds", fps, only_urls={u.rstrip("/") for u in EXPANDED[:2] + [GONE]})
            assert GONE.rstrip("/") in _read_state(td)

            dm.add_urls_batch("ds", [PFX], sync=True)
            state = _read_state(td)
        finally:
            rs.restore()

    assert GONE.rstrip("/") not in state, "pruned source's fingerprint is forgotten"
    assert set(state) == {u.rstrip("/") for u in EXPANDED[:2]}


def test_watched_state_lock_serialises_writers():
    """Concurrent state updates (two reconcilers/pods) both land — the
    fcntl lock guards the read→modify→write."""
    import threading

    with tempfile.TemporaryDirectory() as td:
        client = _client()
        dm, _recorded = _dm_with(client)
        _dm_with_state_dir(dm, td)
        fps_a = _fps(["s3://bucket/pfx/a.pdf"])
        fps_b = _fps(["s3://bucket/pfx/b.log"])

        errors: list[Exception] = []

        def _writer(which: str, fps: dict[str, dict[str, Any]]) -> None:
            try:
                for _ in range(25):
                    dm._watched_state_update("ds", fps, only_urls={next(iter(fps))})
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=_writer, args=("a", fps_a))
        t2 = threading.Thread(target=_writer, args=("b", fps_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

        state = _read_state(td)
        assert set(state) == {"s3://bucket/pfx/a.pdf", "s3://bucket/pfx/b.log"}, "both writers survive (no lost update)"


def test_watched_state_recreate_clears_it():
    """recreate_dataset drops the collection — the watched state must go too,
    or the reconciler would consider everything ingested in an empty dataset."""
    with tempfile.TemporaryDirectory() as td:
        client = _client()
        dm, _recorded = _dm_with(client)
        _dm_with_state_dir(dm, td)
        dm._watched_state_update("ds", _fps(EXPANDED), only_urls={u.rstrip("/") for u in EXPANDED})
        assert _read_state(td)

        # Standalone-runner friendly: call the reset hook recreate uses.
        dm._watched_state_clear("ds")
        assert _read_state(td) == {}, "recreate clears the skip state"


def test_watched_state_dry_run_distinguishes_download_vs_unchanged():
    """sync_dry_run reports would_download vs unchanged with a populated state."""
    with tempfile.TemporaryDirectory() as td:
        fps = _fps(EXPANDED)
        rs, dm, recorded = _sync_env(
            td,
            fps,
            batch_files=[{"file": Path(u).name, "chunks": 3} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)

            changed = dict(fps)
            changed[EXPANDED[0]] = {"etag": "etag-NEW", "size": 42}
            rs.patch(dm_module, "_list_s3_prefix_fingerprints", lambda url: changed)
            recorded["ingested"].clear()
            plan = dm.add_urls_batch("ds", [PFX], sync_dry_run=True)
        finally:
            rs.restore()

    assert plan["status"] == "dry-run"
    assert plan["unchanged"] == sorted({u.rstrip("/") for u in EXPANDED[1:]})
    assert plan["would_download"] == [EXPANDED[0].rstrip("/")]
    assert EXPANDED[0].rstrip("/") in plan["would_ingest"]
    assert recorded["ingested"] == [], "dry run must not ingest"


def test_list_s3_prefix_docstring_is_recursive():
    """Verified-bug fix: _list_s3_prefix IS recursive (paginates with Prefix
    only) — the docstring must say so, not claim immediate-level listing."""
    doc = dm_module._list_s3_prefix.__doc__ or ""
    assert "recursiv" in doc.lower()
    assert "immediate level is listed" not in doc


def test_watched_state_corrupt_file_degrades_to_full_download():
    """A corrupt .watched_state.json must never skip anything — it degrades
    to the pre-feature behaviour (re-download everything)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json", encoding="utf-8")
        rs, dm, _recorded = _sync_env(
            td,
            _fps(EXPANDED),
            stored=set(),
            batch_files=[{"file": Path(u).name, "chunks": 1} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            plan = dm.add_urls_batch("ds", [PFX], sync_dry_run=True)
        finally:
            rs.restore()
    assert plan["unchanged"] == []
    assert plan["would_download"] == sorted({u.rstrip("/") for u in EXPANDED})


def test_watched_state_never_applies_without_sync():
    """The skip state is a watched-source (sync) mechanism only: a plain
    (non-sync) URL ingest never consults it — re-running the same prefix
    ingest manually still downloads, exactly as before the feature."""
    with tempfile.TemporaryDirectory() as td:
        rs, dm, recorded = _sync_env(
            td,
            _fps(EXPANDED),
            batch_files=[{"file": Path(u).name, "chunks": 3} for u in EXPANDED],
            prune={"pruned_points": 0, "pruned_sources": [], "scope": [PFX]},
        )
        try:
            dm.add_urls_batch("ds", [PFX], sync=True)
            assert _read_state(td), "state populated by the sync"

            # Non-sync re-ingest of the SAME prefix, same fingerprints:
            recorded["ingested"].clear()
            rs.patch(dm_module, "_expand_urls", lambda urls: list(EXPANDED))
            dm.add_urls_batch("ds", [PFX])
        finally:
            rs.restore()

    assert recorded["ingested"] == [[Path(u).name for u in EXPANDED]], (
        "non-sync ingest never state-skips (the skip rides sync mode only)"
    )


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
