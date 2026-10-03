"""Deferred ``.bm25_stats.json`` writes for batch ingests (perf P1-1).

The audit finding: ``add_to_vector_store`` persisted the df stats at the end
of EVERY call, and dataset_manager calls it once per FILE — a 5000-file batch
meant 5000 full multi-MB JSON rewrites of the df map over NFS.  The fix
mirrors the ``.hashes.json`` deferral already in dataset_manager: mark dirty,
flush once per interval/at exit.

Covers:

  * **Deferral pays off**: N ``mark_dirty`` calls → at most one or two sidecar
    writes (the write function is monkeypatched to count), while the totals
    still reach the file exactly once each.
  * **Interval triggers**: the per-sidecar call count and the age threshold.
  * **Safety nets**: an atexit flush is registered; the exit/signal flush is
    non-blocking when the in-process lock is held (never deadlocks).
  * **Immediate mode**: ``record_documents`` (the unchanged public API) still
    writes every call, and ``RAG_BM25_FLUSH_CALLS=1`` restores per-call writes
    — non-batch callers behave exactly as before.
  * **Crash semantics**: a missing sidecar force-flushes on the first ingest
    (nothing re-derives the df map from stored points), a torn/absent file
    loads as empty rather than raising, and pending deltas are folded into the
    *weighting* view (``effective_stats``) so deferral does not cheapen the
    dense-quality of the batch it is batching.
  * **Recreate safety**: ``reset_stats`` drops pending deltas so a re-created
    collection cannot inherit the old df counts.

Offline (embedded local Qdrant, stub embedder); no model endpoint required.

Run::

    python tests/full_pipeline/test_bm25_deferral.py    # standalone
    pytest tests/full_pipeline/test_bm25_deferral.py    # under pytest
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient

from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.utils import bm25 as bm25_lane
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings
from multimodal_rag.vector_store import QdrantVectorStore

COLL = "bm25_deferral_test"
DIM = 8


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


@contextmanager
def _isolated_deferral_state():
    """Scrub this process's dirty bookkeeping around a test (leak-proof)."""
    with bm25_lane._dirty_lock:
        bm25_lane._dirty_deltas.clear()
        bm25_lane._dirty_calls.clear()
        bm25_lane._dirty_since.clear()
    try:
        yield
    finally:
        with bm25_lane._dirty_lock:
            bm25_lane._dirty_deltas.clear()
            bm25_lane._dirty_calls.clear()
            bm25_lane._dirty_since.clear()


@contextmanager
def _patch(target: Any, name: str, value: Any):
    """Set an attribute and restore it on exit (standalone-safe).

    Used instead of pytest's ``monkeypatch`` so the file runs identically
    under pytest and as a standalone script, and so a patch can never leak
    into the next test (the failure mode that silently double-counts writes).
    """
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield value
    finally:
        setattr(target, name, old)


class _CountingSave:
    """Wrap ``bm25.save_stats`` to count real sidecar writes."""

    def __init__(self) -> None:
        self.count = 0
        self._real = bm25_lane.save_stats

    def wrapper(self, stats_path: Path, stats: dict[str, Any]) -> None:
        self.count += 1
        self._real(stats_path, stats)


class _StubEmbedderModel:
    """Deterministic hash embedder (no lexical signal) — house test pattern."""

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


class _Rig:
    """A bm25-capable store + rag on embedded local Qdrant with a real sidecar."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.emb = _StubEmbedderModel()
        self.client = QdrantClient(":memory:")
        self.files_dir = Path(self.tmp.name) / "ds" / "files"
        self.stats_path = self.files_dir / bm25_lane.BM25_STATS_FILENAME
        self.store = MultimodalRAG._build_qdrant_vector_store(  # type: ignore[assignment]
            embedding=self.emb.model,
            client=self.client,
            collection_name=COLL,
            bm25_stats_path=str(self.stats_path),
        )
        self.rag = MultimodalRAG(embedder=self.emb, vector_store=self.store)  # type: ignore[arg-type]

    def ingest(self, n_files: int = 50) -> None:
        """One ``add_to_vector_store`` call per "file" (dataset_manager's loop)."""
        for i in range(n_files):
            asyncio.run(
                self.rag.aadd_to_vector_store(
                    [{"text": f"file_{i} term_{i} shared_term", "source": f"/tmp/x/f{i}.txt"}],
                    deduplicate=False,
                )
            )

    def close(self) -> None:
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# (a) deferral pays off: N calls → 1-2 writes
# ---------------------------------------------------------------------------


def test_fifty_record_calls_write_once_or_twice() -> None:
    """50 deferred marks on an EXISTING sidecar → 1 disk write.

    (Default RAG_BM25_FLUSH_CALLS=50 fires on the 50th mark, so a 50-file
    batch costs one sidecar rewrite instead of 50.)  The totals must still
    land exactly once each — no double-counting, no lost terms.
    """
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        # Seed the sidecar: the missing-file force-flush (covered separately)
        # would otherwise make the very first mark report due immediately.
        bm25_lane.record_documents(path, [{"seed": 1}])
        counter = _CountingSave()
        with _patch(bm25_lane, "save_stats", counter.wrapper):
            for i in range(50):
                due = bm25_lane.mark_dirty(path, [{"alpha": 1, f"t{i}": 1}])
                if i < 49:
                    assert not due, "under the count threshold nothing may be written"
                else:
                    assert due, "the 50th mark must report a flush due"
                assert counter.count == 0, "mark_dirty itself must never write"
            assert not bm25_lane.load_stats(path)["df"].get("alpha"), "deltas stay out of the file until flush"

            flushed = bm25_lane.flush_if_dirty(path)
            assert flushed == 1
            assert counter.count == 1, "50 marks → exactly one sidecar write"

        stats = bm25_lane.load_stats(path)
        assert stats["n_docs"] == 51, "every marked document must be counted exactly once"
        assert stats["df"]["alpha"] == 50 and stats["df"]["t0"] == 1 and stats["df"]["t49"] == 1

        # A second flush with nothing pending is a no-op (no spurious rewrite).
        assert bm25_lane.flush_if_dirty(path) == 0
        assert counter.count == 1


def test_batch_ingest_writes_far_fewer_than_one_per_file() -> None:
    """End-to-end: 50 add_to_vector_store calls cost a handful of writes.

    This is the audit scenario (dataset_manager calls it once per file).  The
    exact count is threshold-driven, so assert the bound that matters: a
    small constant, not one-per-file.  With the 50-default count trigger and
    a fresh (absent) sidecar the first file force-flushes, then files 2-50
    defer — i.e. 1 write, maybe 2 if the age trigger also fires.
    """
    with _isolated_deferral_state():
        rig = _Rig()
        try:
            counter = _CountingSave()
            with _patch(bm25_lane, "save_stats", counter.wrapper):
                rig.ingest(50)
            assert counter.count <= 2, f"50 files must cost ≤2 writes, got {counter.count}"
            assert counter.count >= 1, "at least the missing-sidecar force-flush must land"

            # Deliver the still-pending deltas explicitly (as the batch end /
            # exit hook would) and prove the totals are exact.
            bm25_lane.flush_if_dirty(rig.stats_path)
            stats = bm25_lane.load_stats(rig.stats_path)
            assert stats["n_docs"] == 50, "all 50 ingested docs must be in the sidecar"
            assert stats["df"].get("shared_term") == 50
            assert counter.count <= 3, f"post-flush total must stay bounded, got {counter.count}"
        finally:
            rig.close()


# ---------------------------------------------------------------------------
# (b) interval triggers
# ---------------------------------------------------------------------------


def test_count_trigger_env_immediate_and_disabled() -> None:
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME

        # RAG_BM25_FLUSH_CALLS=1 → every mark reports due (historical mode).
        with _env(RAG_BM25_FLUSH_CALLS="1", RAG_BM25_FLUSH_SECONDS="0"):
            assert bm25_lane.mark_dirty(path, [{"a": 1}]) is True
            bm25_lane.flush_if_dirty(path)
            assert bm25_lane.mark_dirty(path, [{"a": 1}]) is True
            bm25_lane.flush_if_dirty(path)
            assert bm25_lane.load_stats(path)["n_docs"] == 2

        # Count trigger disabled, age trigger far away → no flush reported.
        with _env(RAG_BM25_FLUSH_CALLS="0", RAG_BM25_FLUSH_SECONDS="3600"):
            for _ in range(10):
                assert bm25_lane.mark_dirty(path, [{"b": 1}]) is False
            # …but the age trigger can still fire it.
            with bm25_lane._dirty_lock:
                bm25_lane._dirty_since[path] = time.monotonic() - 3601.0
            assert bm25_lane.mark_dirty(path, [{"b": 1}]) is True
        bm25_lane.flush_if_dirty(path)

        # Both triggers disabled = explicit immediate mode ("defer forever" is
        # not expressible: every mark reports due).
        with _env(RAG_BM25_FLUSH_CALLS="0", RAG_BM25_FLUSH_SECONDS="0"):
            assert bm25_lane.mark_dirty(path, [{"c": 1}]) is True


def test_age_trigger_flushes_past_threshold() -> None:
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"seed": 1}])  # existing sidecar: age trigger only
        with _env(RAG_BM25_FLUSH_CALLS="1000", RAG_BM25_FLUSH_SECONDS="30"):
            assert bm25_lane.mark_dirty(path, [{"a": 1}]) is False
            # Simulate 31s of batch time without sleeping: age the marker.
            with bm25_lane._dirty_lock:
                bm25_lane._dirty_since[path] = time.monotonic() - 31.0
            assert bm25_lane.mark_dirty(path, [{"a": 1}]) is True, "an aged batch must trigger the flush"
        assert bm25_lane.flush_if_dirty(path) == 1
        assert bm25_lane.load_stats(path)["n_docs"] == 3


def test_flush_only_touches_named_sidecar() -> None:
    """A per-dataset flush must not write another dataset's pending deltas."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a" / bm25_lane.BM25_STATS_FILENAME
        b = Path(tmp) / "b" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(a, [{"seed": 1}])
        bm25_lane.mark_dirty(a, [{"a_term": 1}])
        bm25_lane.mark_dirty(b, [{"b_term": 1}])
        assert bm25_lane.flush_if_dirty(a) == 1
        assert bm25_lane.load_stats(a)["df"]["a_term"] == 1
        assert not b.exists(), "dataset b's deltas must still be pending"
        # The blanket (exit) flush takes care of the rest.
        assert bm25_lane.flush_if_dirty() == 1
        assert bm25_lane.load_stats(b)["n_docs"] == 1


def test_blanket_flush_drains_every_sidecar() -> None:
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / d / bm25_lane.BM25_STATS_FILENAME for d in ("a", "b", "c")]
        for i, p in enumerate(paths):
            bm25_lane.mark_dirty(p, [{f"term{i}": 1}])
        assert bm25_lane.flush_if_dirty() == 3
        for i, p in enumerate(paths):
            assert bm25_lane.load_stats(p)["n_docs"] == 1


# ---------------------------------------------------------------------------
# (c) exit / signal safety nets
# ---------------------------------------------------------------------------


def test_atexit_flush_registered_and_runs(tmp_path: Path | None = None) -> None:
    """The exit hook is registered AND actually drains deltas at interpreter exit.

    Proven out-of-process: an in-process test cannot exit the interpreter, so
    a child runs ``mark_dirty`` (no flush), exits normally, and the parent
    asserts the sidecar landed — that is exactly the atexit body.
    """

    tmp = tempfile.TemporaryDirectory() if tmp_path is None else None
    base = Path(tmp.name) if tmp is not None else Path(tmp_path)
    try:
        _run_atexit_check(base)
    finally:
        if tmp is not None:
            tmp.cleanup()


def _run_atexit_check(base: Path) -> None:
    import subprocess

    stats_path = base / "files" / bm25_lane.BM25_STATS_FILENAME
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # The sidecar must already exist, or the first mark force-flushes by
    # design (missing-map guard) and there would be nothing left for atexit.
    stats_path.write_text(json.dumps({"n_docs": 1, "total_len": 1, "df": {"seed": 1}}), encoding="utf-8")
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys;"
        f"sys.path.insert(0, {str(src)!r});"
        "from pathlib import Path;"
        "from multimodal_rag.utils import bm25;"
        f"p = Path({str(stats_path)!r});"
        "due = bm25.mark_dirty(p, [{'exit_term': 1}, {'exit_term': 2}]);"
        "assert due is False, 'deferral expected on an existing sidecar';"
        "assert bm25.load_stats(p)['df'].get('exit_term') is None, 'nothing may be written before exit';"
        "print('MARKED')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "MARKED" in proc.stdout
    stats = bm25_lane.load_stats(stats_path)
    assert stats["n_docs"] == 3 and stats["df"]["exit_term"] == 2, "atexit must flush deferred deltas at process exit"

    # And in-process the hook is registered (mark_dirty installs it once).
    assert bm25_lane._FLUSH_HOOKS_REGISTERED is True


def test_flush_if_dirty_is_idempotent_and_survives_exit() -> None:
    """The exit flush drains pending deltas and a second run is a no-op."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.mark_dirty(path, [{"alpha": 1}, {"alpha": 2}])
        assert not path.exists()
        bm25_lane.flush_if_dirty()  # the atexit body
        assert bm25_lane.load_stats(path)["n_docs"] == 2
        assert bm25_lane.flush_if_dirty() == 0


def test_signal_flush_never_blocks_on_held_lock() -> None:
    """A SIGTERM arriving while the main thread holds the dirty lock must not
    deadlock the shutdown (handlers run in the main thread)."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.mark_dirty(path, [{"alpha": 1}])

        held = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with bm25_lane._dirty_lock:
                held.set()
                release.wait(5.0)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        assert held.wait(5.0)
        try:
            # Must return promptly with 0 (skipped), never hang.
            started = time.monotonic()
            assert bm25_lane.flush_if_dirty(signal_safe=True) == 0
            assert time.monotonic() - started < 2.0, "signal flush must not block on the dirty lock"
        finally:
            release.set()
            t.join(5.0)
        # Deltas survived the skipped flush and land on the next real flush.
        assert bm25_lane.flush_if_dirty() == 1
        assert bm25_lane.load_stats(path)["n_docs"] == 1


def test_sigterm_handler_flushes_and_keeps_exit_status(tmp_path: Path | None = None) -> None:
    """SIGTERM flushes the deltas and the process still dies of SIGTERM.

    Out-of-process: the hook must both persist the pending deltas (Python's
    default SIGTERM disposition skips atexit, so this is the container-stop
    safety net) AND preserve the signal exit status so orchestrators see the
    real cause of death.
    """

    tmp = tempfile.TemporaryDirectory() if tmp_path is None else None
    base = Path(tmp.name) if tmp is not None else Path(tmp_path)
    try:
        _run_sigterm_check(base)
    finally:
        if tmp is not None:
            tmp.cleanup()


def _run_sigterm_check(base: Path) -> None:
    import subprocess

    stats_path = base / "files" / bm25_lane.BM25_STATS_FILENAME
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-existing sidecar so the mark defers (see the atexit test).
    stats_path.write_text(json.dumps({"n_docs": 1, "total_len": 1, "df": {}}), encoding="utf-8")
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import os, sys, time;"
        f"sys.path.insert(0, {str(src)!r});"
        "from pathlib import Path;"
        "from multimodal_rag.utils import bm25;"
        f"p = Path({str(stats_path)!r});"
        "due = bm25.mark_dirty(p, [{'sig_term': 1}]);"
        "assert due is False, 'deferral expected';"
        "print('READY', flush=True);"
        "time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "READY" in line, f"child never became ready ({line!r})"
        proc.terminate()  # SIGTERM
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - failure surface
            proc.kill()
            raise AssertionError("SIGTERM handler hung the process (deadlock)") from None
    finally:
        if proc.poll() is None:  # pragma: no cover - defensive cleanup
            proc.kill()
            proc.wait(timeout=10)
    assert stats_path.exists(), "the SIGTERM hook must flush deferred deltas"
    assert bm25_lane.load_stats(stats_path)["df"]["sig_term"] == 1
    # Default disposition was restored and re-raised → killed by SIGTERM (-15),
    # not a normal exit the orchestrator would misread.
    assert proc.returncode == -15, f"expected death by SIGTERM, got {proc.returncode}"


def test_sigterm_hook_registered_in_process() -> None:
    import signal

    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        prev = signal.getsignal(signal.SIGTERM)
        try:
            bm25_lane.mark_dirty(path, [{"alpha": 1}])
            installed = signal.getsignal(signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, prev)
            bm25_lane.flush_if_dirty(path)
        assert callable(installed), "mark_dirty must install a SIGTERM flush handler"
        assert bm25_lane.load_stats(path)["n_docs"] == 1


# ---------------------------------------------------------------------------
# (d) immediate mode / non-batch callers unchanged
# ---------------------------------------------------------------------------


def test_record_documents_stays_immediate() -> None:
    """The public immediate API still writes on every call, byte-for-byte."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        counter = _CountingSave()
        with _patch(bm25_lane, "save_stats", counter.wrapper):
            bm25_lane.record_documents(path, [{"alpha": 1}])
            assert counter.count == 1 and path.exists()
            bm25_lane.record_documents(path, [{"beta": 1}])
            assert counter.count == 2, "record_documents is the immediate API — one write per call"
            assert bm25_lane.load_stats(path)["n_docs"] == 2
            # It must not have enqueued deferred work.
            assert bm25_lane.flush_if_dirty(path) == 0

            # save_stats round-trips the shaped payload (used by every writer).
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert set(payload) == {"n_docs", "total_len", "df"}


def test_forget_documents_stays_immediate() -> None:
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"alpha": 2, "beta": 1}])
        counter = _CountingSave()
        with _patch(bm25_lane, "save_stats", counter.wrapper):
            bm25_lane.forget_documents(path, [{"alpha": 2, "beta": 1}])
            assert counter.count == 1, "the delete path must stay immediate"
            assert bm25_lane.load_stats(path)["n_docs"] == 0


def test_merge_doc_and_copy_stats_unchanged() -> None:
    """The in-memory primitives keep their shapes (callers depend on them)."""
    stats = bm25_lane._new_stats()
    bm25_lane.merge_doc(stats, {"a": 2, "b": 1})
    bm25_lane.merge_doc(stats, {"a": 1})
    assert stats == {"n_docs": 2, "total_len": 4, "df": {"a": 2, "b": 1}}
    cloned = bm25_lane.copy_stats(stats)
    cloned["df"]["a"] = 99
    assert stats["df"]["a"] == 2, "copy_stats must not alias the source df map"


# ---------------------------------------------------------------------------
# Crash semantics
# ---------------------------------------------------------------------------


def test_missing_sidecar_force_flushes_on_first_mark() -> None:
    """An absent stats file must never be left absent: first mark flushes.

    Nothing re-derives the df map from stored points (verified across the
    repo's bm25/df call sites), so the deferral must not gamble on the very
    first persist.
    """
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        assert bm25_lane.stats_fully_absent(path) is True
        assert bm25_lane.mark_dirty(path, [{"alpha": 1}]) is True, "missing sidecar must force a flush"
        bm25_lane.flush_if_dirty(path)
        assert path.exists()
        # Once it exists, deferral resumes (no per-call write any more).
        assert bm25_lane.stats_fully_absent(path) is False
        assert bm25_lane.mark_dirty(path, [{"beta": 1}]) is False


def test_missing_stats_file_degrades_gracefully() -> None:
    """Absent or corrupt sidecars load as empty and never raise (idf stays
    positive, so already-stored sparse vectors remain searchable)."""
    from multimodal_rag.rag_system import _bm25_ingest_context

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "nope" / bm25_lane.BM25_STATS_FILENAME
        assert bm25_lane.load_stats(missing)["n_docs"] == 0
        assert bm25_lane.idf(bm25_lane.load_stats(missing), "anything") > 0.0

        corrupt = Path(tmp) / bm25_lane.BM25_STATS_FILENAME
        corrupt.write_text("{not json", encoding="utf-8")
        stats = bm25_lane.load_stats(corrupt)
        assert stats["n_docs"] == 0 and stats["df"] == {}

        # A torn / vanished file must not break the query-side gate either:
        # no usable stats → the request stays flat dense.
        store = QdrantVectorStore(QdrantClient(":memory:"), "torn", embedding=None, bm25_stats_path=str(missing))
        assert store._bm25_stats() is None
        # …nor the ingest-side context (it stays usable, just empty).
        store2 = QdrantVectorStore(QdrantClient(":memory:"), "torn2", embedding=None, bm25_stats_path=str(missing))
        assert store2.supports_hybrid() is False  # plain collection → lane dark
        assert _bm25_ingest_context(store2) is None


def test_effective_stats_folds_pending_deltas() -> None:
    """Deferred batches weight against a df view that includes their own
    pending deltas (otherwise every file of a big batch would be weighted
    against a sidecar lagging behind it)."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"seed": 1}])
        bm25_lane.mark_dirty(path, [{"fresh": 1}, {"fresh": 2}])

        # On-disk view lags…
        assert "fresh" not in bm25_lane.load_stats(path)["df"]
        # …but the effective view sees both pending docs (df=2).
        eff = bm25_lane.effective_stats(path)
        assert eff["n_docs"] == 3 and eff["df"]["fresh"] == 2

        # It is an independent copy — never the cache, never the delta store.
        eff["df"]["fresh"] = 999
        assert bm25_lane.effective_stats(path)["df"]["fresh"] == 2
        assert bm25_lane.load_stats(path)["df"].get("fresh") is None

        # After the flush both views agree (nothing double-counted).
        bm25_lane.flush_if_dirty(path)
        final = bm25_lane.load_stats(path)
        assert final["n_docs"] == 3 and final["df"]["fresh"] == 2


def test_effective_stats_used_by_ingest_weighing() -> None:
    """The ingest context must fold pending deltas into its df snapshot."""
    from multimodal_rag.rag_system import _bm25_ingest_context

    with _isolated_deferral_state():
        rig = _Rig()
        try:
            rig.ingest(3)
            ctx = _bm25_ingest_context(rig.store)
            assert ctx is not None
            # 3 ingested docs; a flushed sidecar covers all of them, and any
            # still-pending deltas are folded in — never silently dropped.
            assert ctx["stats"]["n_docs"] >= 3
            assert "shared_term" in ctx["stats"]["df"]
        finally:
            rig.close()


def test_reset_stats_drops_pending_deltas() -> None:
    """A recreate must not resurrect old-collection df counts from the queue."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.mark_dirty(path, [{"stale": 1}])
        bm25_lane.reset_stats(path)
        assert bm25_lane.flush_if_dirty(path) == 0, "pending deltas must be discarded with the sidecar"
        assert not path.exists()


def test_stats_fully_absent_survives_stat_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files"

        def boom(self: Path) -> bool:
            raise OSError("nfs hiccup")

        with _patch(Path, "exists", boom):
            assert bm25_lane.stats_fully_absent(path / bm25_lane.BM25_STATS_FILENAME) is True, (
                "a stat failure must be treated as absent (force the write)"
            )
        assert bm25_lane.stats_fully_absent(path / bm25_lane.BM25_STATS_FILENAME) is True, (
            "a stat failure must be treated as absent (force the write)"
        )


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_failed_flush_does_not_double_count_later() -> None:
    """Deltas are dropped when a flush is attempted, so a transient error can
    never cause the same counts to be merged twice on the next flush."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"seed": 1}])
        bm25_lane.mark_dirty(path, [{"alpha": 1}])

        real = bm25_lane.save_stats
        calls = {"n": 0}

        def flaky(stats_path: Path, stats: dict[str, Any]) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("NFS write failed")
            real(stats_path, stats)

        with _patch(bm25_lane, "save_stats", flaky):
            assert bm25_lane.flush_if_dirty(path) == 0, "a failed flush reports zero sidecars"
            # The delta must be gone (dropped), not retried and double-counted.
            assert bm25_lane.flush_if_dirty(path) == 0
        assert bm25_lane.load_stats(path)["df"].get("alpha") is None

        # A subsequent mark/flush behaves normally.
        bm25_lane.mark_dirty(path, [{"beta": 1}])
        assert bm25_lane.flush_if_dirty(path) == 1
        assert bm25_lane.load_stats(path)["df"]["beta"] == 1


def test_concurrent_flushes_do_not_lose_deltas() -> None:
    """Two threads flushing the same sidecar: every delta lands exactly once."""
    with _isolated_deferral_state(), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "files" / bm25_lane.BM25_STATS_FILENAME
        bm25_lane.record_documents(path, [{"seed": 1}])
        for i in range(20):
            bm25_lane.mark_dirty(path, [{f"t{i}": 1}])

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                bm25_lane.flush_if_dirty(path)
            except BaseException as exc:  # pragma: no cover - failure surface
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30.0)
        assert not errors
        stats = bm25_lane.load_stats(path)
        assert stats["n_docs"] == 21, f"exactly-once merge expected, got {stats['n_docs']}"
        assert sum(1 for i in range(20) if stats["df"].get(f"t{i}") == 1) == 20


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
