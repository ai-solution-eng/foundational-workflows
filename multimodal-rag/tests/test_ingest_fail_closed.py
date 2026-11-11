"""Feature D fail-closed property on the ingest path (cross-validation F4).

THE BUG THIS PINS
-----------------
Feature D's contract: enabling late interaction on a dataset with no token
embedder must FAIL the ingest, never silently store points without token
matrices.  On the BATCH path the failure was swallowed by the retry wrapper:
the first attempt upserted the main points and then raised
LateInteractionUnavailable from the sidecar write, the shared retry_call
retried ANY exception, the retry hit content-hash / vector dedup, stored
nothing and returned an empty id list — so the file was reported
"complete, 0 chunks" and the ingest SUCCEEDED.

THE CONTRACT AFTER THE FIX
--------------------------
* the batch consumer uses _ingest_retry_call, which re-raises
  LateInteractionUnavailable on the FIRST attempt (no retry) while still
  retrying transient I/O errors exactly as before;
* the file is reported FAILED (an error entry, never complete-with-0-chunks);
* the single-file path (no retry wrapper there) propagates the error;
* an archive member that fails closed fails the whole archive instead of being
  skipped into a 0-chunk success (ArchiveProcessor.process swallows member
  exceptions, so the handler records it and re-raises afterwards).

Offline: embedded local Qdrant, stub embedder, no network.

Run::

    pytest tests/test_ingest_fail_closed.py -q
"""

import base64
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multimodal_rag import dataset_manager as dm_mod
from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.late_interaction import LateInteractionUnavailable

#: Smallest valid PNG (1x1).  Kept inline so the test has no fixture dependency.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _raise_late(*args, **kwargs):
    raise LateInteractionUnavailable("late-enabled dataset has no token embedder")


@pytest.fixture()
def ingest_dm(base_path, embedder, monkeypatch):
    """Real offline DatasetManager with one dataset (embedded Qdrant)."""
    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("docs")
    return manager


# ===========================================================================
# The retry wrapper itself
# ===========================================================================


def test_ingest_retry_call_does_not_retry_a_late_interaction_failure():
    calls = []

    def _fail():
        calls.append(1)
        raise LateInteractionUnavailable("no token embedder")

    with pytest.raises(LateInteractionUnavailable):
        dm_mod._ingest_retry_call(_fail, max_attempts=3, base_delay=0.0, connection_delay=0.0)
    assert len(calls) == 1, "a deterministic configuration error must not be retried"


def test_ingest_retry_call_still_retries_a_transient_error():
    calls = []

    def _flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("connection reset")
        return "ok"

    assert dm_mod._ingest_retry_call(_flaky, max_attempts=3, base_delay=0.0, connection_delay=0.0) == "ok"
    assert len(calls) == 3, "transient I/O errors keep their retry behaviour"


def test_ingest_retry_call_reraises_the_last_transient_error():
    calls = []

    def _always_fail():
        calls.append(1)
        raise RuntimeError("still down")

    with pytest.raises(RuntimeError) as excinfo:
        dm_mod._ingest_retry_call(_always_fail, max_attempts=3, base_delay=0.0, connection_delay=0.0)
    assert "still down" in str(excinfo.value)
    assert len(calls) == 3


# ===========================================================================
# Batch path — the file is reported failed, not complete-with-0-chunks
# ===========================================================================


def test_batch_ingest_reports_a_late_failure_as_failed(ingest_dm, tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(PNG_1PX)
    rag = ingest_dm._get_rag("docs")

    calls = []

    def _fake_store(docs, dataset_lifetime=None):
        calls.append(list(docs))
        if len(calls) == 1:
            raise LateInteractionUnavailable("late-enabled dataset 'docs' has no token embedder")
        # What the OLD retry would have produced: dedup finds the points the
        # first attempt already stored, so nothing new is written.
        return []

    rag.add_to_vector_store = _fake_store
    progress = []
    result = ingest_dm.add_files_batch("docs", [(str(image), "tiny.png")], progress_callback=progress.append)

    assert result["status"] == "ok"
    entry = next(item for item in result["files"] if item["file"] == "tiny.png")
    assert entry.get("error"), "the file must be reported FAILED"
    assert "token embedder" in entry["error"]
    assert entry["chunks"] == 0
    assert len(calls) == 1, "the fail-closed error must not be retried into a silent success"
    assert not any(event.get("status") == "complete" for event in progress), (
        "a late failure must never be reported as complete (the old bug reported complete/0 chunks)"
    )


# ===========================================================================
# Single-document path — no retry wrapper, so it must propagate
# ===========================================================================


def test_single_file_ingest_propagates_the_late_failure(ingest_dm, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("a plain text document")
    rag = ingest_dm._get_rag("docs")
    rag.add_to_vector_store = _raise_late

    with pytest.raises(LateInteractionUnavailable):
        ingest_dm.add_file("docs", str(note))


def test_archive_ingest_fails_closed_instead_of_skipping_the_member(ingest_dm, tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("note.txt", "a plain text document")
    rag = ingest_dm._get_rag("docs")
    rag.add_to_vector_store = _raise_late

    # ArchiveProcessor.process swallows member exceptions; the handler records
    # the fail-closed one and re-raises it after process() returns.
    with pytest.raises(LateInteractionUnavailable):
        ingest_dm.add_file("docs", str(archive))


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  {name} ... OK")
        except Exception:
            failures += 1
            print(f"  {name} ... FAIL")
            traceback.print_exc()
    print(f"{'All tests passed!' if not failures else f'{failures} test(s) failed'}")
    sys.exit(1 if failures else 0)
