"""Regression tests for the meta.json parsed-read cache (perf P1-2).

``DatasetManager._read_meta`` sits on the hottest search path — once per
search via ``_effective_rrf`` plus once per federated target — and used to
cost two NFS syscalls (``exists()`` + ``read_text()``) AND a full JSON parse
every call.  It is now stamp-checked (mtime_ns, size), matching
``clients_registry.is_public_dataset`` / the in-file ``_load_hash_index``
discipline.

What this file guards:

  * invalidation — a write through ``_write_meta`` is visible to the very next
    read, even when the stamp cannot prove it (same mtime_ns AND same size,
    i.e. the explicit-invalidation property);
  * hit behaviour — two reads of unchanged bytes parse the JSON exactly once;
  * no TTL — an out-of-band write (another pod on the shared RWX PVC) is
    picked up on the next read because the stamp is re-checked per call;
  * defensive copies — mutating a returned dict cannot poison the cache;
  * fail-closed — corrupt/missing meta returns ``None`` and never raises;
  * both caches clear on dataset delete.

No Qdrant, no models, no network: the manager is a ``__new__`` shell over a
temp ``datasets_path`` (the test_wave4_performance / test_preprocess_atomicity
house pattern).

Run::

    python tests/full_pipeline/test_meta_cache.py    # standalone
    pytest tests/full_pipeline/test_meta_cache.py    # under pytest
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.dataset_manager as dm_module
from multimodal_rag.dataset_manager import DatasetManager


def _make_manager(root: Path) -> DatasetManager:
    """A DatasetManager shell with only the meta layer wired up."""
    dm = DatasetManager.__new__(DatasetManager)
    dm.base_path = root  # type: ignore[attr-defined]
    dm.datasets_path = root / "datasets"  # type: ignore[attr-defined]
    dm.datasets_path.mkdir(parents=True, exist_ok=True)
    dm._rag_cache = {}  # type: ignore[attr-defined]
    dm._has_password_cache = {}  # type: ignore[attr-defined]
    dm._has_password_lock = threading.Lock()  # type: ignore[attr-defined]
    return dm


def _reset_cache() -> None:
    with dm_module._meta_cache_lock:
        dm_module._META_CACHE.clear()


class _CountingJson:
    """A proxy that counts ``json.loads`` calls, so one parse is provable."""

    def __init__(self) -> None:
        self.loads_calls = 0

    def loads(self, *args, **kwargs):
        self.loads_calls += 1
        return json.loads(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(json, item)


# ---------------------------------------------------------------------------
# (a) invalidation on write
# ---------------------------------------------------------------------------


def test_write_then_read_reflects_new_value():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        dm._write_meta("ds", {"name": "ds", "document_count": 1})
        assert dm._read_meta("ds")["document_count"] == 1

        dm._write_meta("ds", {"name": "ds", "document_count": 7})
        meta = dm._read_meta("ds")
        assert meta["document_count"] == 7, "a write must invalidate the parsed-meta cache"


def test_write_invalidates_even_when_stamp_is_identical():
    """Same-size rewrite with a forced-back mtime must still be visible.

    This is why the writers invalidate explicitly instead of trusting the
    stamp: a same-length value written within the filesystem's timestamp
    granularity produces an identical (mtime_ns, size) pair, and a
    stamp-only cache would serve the stale dict forever.
    """
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        p = dm._meta_path("ds")
        dm._write_meta("ds", {"name": "ds", "v": "aaaa"})
        stamp_ns = p.stat().st_mtime_ns
        assert dm._read_meta("ds")["v"] == "aaaa"  # populate the cache

        dm._write_meta("ds", {"name": "ds", "v": "bbbb"})  # same byte length
        os.utime(p, ns=(stamp_ns, stamp_ns))  # forge an identical stamp
        assert p.stat().st_mtime_ns == stamp_ns
        assert dm._read_meta("ds")["v"] == "bbbb", "explicit invalidation must not rely on the stamp"


# ---------------------------------------------------------------------------
# (b) one parse per unchanged meta
# ---------------------------------------------------------------------------


def test_unchanged_meta_parses_once(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        counter = _CountingJson()
        monkeypatch.setattr(dm_module, "json", counter)
        dm = _make_manager(Path(td))
        dm._write_meta("ds", {"name": "ds", "document_count": 3})

        first = dm._read_meta("ds")
        second = dm._read_meta("ds")
        third = dm._read_meta("ds")

        assert first == second == third == {"name": "ds", "document_count": 3}
        assert counter.loads_calls == 1, "unchanged meta must be parsed exactly once"


def test_out_of_band_write_is_seen_on_next_read():
    """No TTL: the per-call stamp check observes another writer immediately."""
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        dm._write_meta("ds", {"name": "ds", "document_count": 1})
        assert dm._read_meta("ds")["document_count"] == 1

        # A different pod writes the file directly (no _write_meta on ours).
        p = dm._meta_path("ds")
        p.write_text(json.dumps({"name": "ds", "document_count": 42, "note": "other pod"}))

        meta = dm._read_meta("ds")
        assert meta["document_count"] == 42 and meta["note"] == "other pod", (
            "the stamp must be re-checked per call — no TTL may hide another pod's write"
        )


# ---------------------------------------------------------------------------
# (c) defensive copies
# ---------------------------------------------------------------------------


def test_mutating_the_returned_dict_does_not_poison_the_cache():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        dm._write_meta("ds", {"name": "ds", "rrf": {"k": 2}, "document_count": 1})

        first = dm._read_meta("ds")
        first["document_count"] = 999  # caller mutates its copy ...
        first.pop("name", None)  # ... and pops a key (the set_password shape)
        first["rrf"]["k"] = 7  # ... including a nested structure

        second = dm._read_meta("ds")
        assert second == {"name": "ds", "rrf": {"k": 2}, "document_count": 1}

        # And a caller that mutates then writes (the read-modify-write shape)
        # must not have corrupted what a later reader sees.
        third = dm._read_meta("ds")
        third["document_count"] = 5
        dm._write_meta("ds", third)
        assert dm._read_meta("ds")["document_count"] == 5


def test_missing_meta_returns_none_and_is_cached():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        assert dm._read_meta("never-existed") is None
        assert dm._read_meta("never-existed") is None

        # Creating the dataset afterwards must be visible on the next read.
        dm._write_meta("never-existed", {"name": "never-existed"})
        assert dm._read_meta("never-existed") == {"name": "never-existed"}


# ---------------------------------------------------------------------------
# (d) corrupt meta — fail closed, never raise
# ---------------------------------------------------------------------------


def test_corrupt_meta_returns_none_and_does_not_raise():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        p = dm._meta_path("broken")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json at all")

        assert dm._read_meta("broken") is None, "corrupt meta must fail closed"
        assert dm._read_meta("broken") is None, "and stay None on the cached path"

        # A repaired file is picked up (the stamp changed).
        p.write_text(json.dumps({"name": "broken", "document_count": 2}))
        assert dm._read_meta("broken")["document_count"] == 2


def test_empty_and_non_dict_meta_do_not_raise():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        p = dm._meta_path("weird")
        p.parent.mkdir(parents=True, exist_ok=True)

        p.write_text("")  # zero-byte file (a truncated write)
        assert dm._read_meta("weird") is None

        p.write_text("[1, 2, 3]")  # valid JSON, wrong type
        meta = dm._read_meta("weird")
        assert meta == [1, 2, 3] or meta is None, "must not raise either way"


# ---------------------------------------------------------------------------
# Delete clears BOTH caches
# ---------------------------------------------------------------------------


def test_delete_dataset_clears_meta_and_password_caches():
    with tempfile.TemporaryDirectory() as td:
        _reset_cache()
        dm = _make_manager(Path(td))
        # Stub the Qdrant side: delete_dataset's collection drop is best-effort
        # and must not need a live RAG instance in this test.
        dm._get_rag = lambda *a, **k: type("_Rag", (), {"vector_store": None})()  # type: ignore[method-assign]
        dm._write_meta("ds", {"name": "ds", "password_hash": "salt$deadbeef"})
        assert dm._read_meta("ds") is not None  # populate the parsed-meta cache

        # Populate the has_password cache through the real (non-PBKDF2) path.
        assert dm.has_password("ds") is True
        assert "ds" in dm._has_password_cache

        dm.delete_dataset("ds")

        assert dm._read_meta("ds") is None, "a deleted dataset must not be served from cache"
        assert "ds" not in dm._has_password_cache, "the TTL cache must clear on delete too"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in fns:
        try:
            if fn.__code__.co_argcount:  # needs pytest's monkeypatch
                skipped += 1
                print(f"SKIP {fn.__name__} (pytest-only fixture)")
                continue
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    ran = len(fns) - skipped
    print(f"\n{ran - failed}/{ran} passed ({skipped} skipped — run under pytest for those)")
    sys.exit(1 if failed else 0)
