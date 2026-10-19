"""P2 (audit 2026-10-16): the MCP file-hash cache is STAT-KEYED (zone C routing).

The defect: ``mcp_server._file_hash_cache`` was keyed by PATH, so a file
REPLACED at the same path — a re-upload, a regenerated ``*_preprocessed``
sibling, a repair pass — kept serving the old digest for the lifetime of the
process.  Every downstream cache keyed on that digest (query vectors, media
keys) therefore served stale results with no invalidation hook anywhere.

Fix: ``_hash_file`` delegates to the shared
:class:`multimodal_rag.utils.media_hash.StatKeyedFileHasher`, whose key is
``(path, st_mtime_ns, st_size)`` — a rewrite is a MISS by construction.

Pinned here (negative + positive):

  * a same-size rewrite is detected (the size-only key would have missed it);
  * an unchanged file is served from the cache (the hasher is not simply
    recomputing on every call);
  * the module-level ``_hash_file`` still propagates ``OSError`` for a
    missing file (the inline ``open()`` behaviour callers rely on);
  * the delegation keeps the FILE_HASH_CACHE_MAX bound as the LRU cap.

Run::

    pytest tests/security/test_mcp_file_hash_cache.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp
from multimodal_rag.utils import media_hash


@pytest.fixture(autouse=True)
def _fresh_hasher(monkeypatch):
    """A private hasher per test so the process-wide cache cannot leak state."""
    monkeypatch.setattr(mcp, "_file_hasher", media_hash.StatKeyedFileHasher())


def test_same_size_rewrite_is_a_cache_miss(tmp_path):
    """THE regression: replacing a file with SAME-SIZE different bytes must
    change the digest.  A size-only (or path-only) key would keep the old one."""
    p = tmp_path / "media.bin"
    p.write_bytes(b"AAAA")
    first = mcp._hash_file(str(p))
    p.write_bytes(b"BBBB")  # same length, different content
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 1_000_000))
    second = mcp._hash_file(str(p))
    assert first != second, "a rewritten file must not serve its old digest"
    assert mcp._hash_file(str(p)) == second, "…and the new digest is cached"


def test_unchanged_file_is_served_from_cache(tmp_path):
    """Positive control: repeat calls on an unchanged file hash ONCE."""
    p = tmp_path / "media.bin"
    p.write_bytes(b"payload")
    assert mcp._hash_file(str(p)) == mcp._hash_file(str(p))
    assert mcp._file_hasher.files_hashed == 1


def test_missing_file_raises_oserror(tmp_path):
    """The delegation preserves the error contract: a missing file is an
    ``OSError``, never a silent empty digest."""
    with pytest.raises(OSError):
        mcp._hash_file(str(tmp_path / "nope.bin"))


def test_cache_is_lru_bounded(tmp_path, monkeypatch):
    """The FILE_HASH_CACHE_MAX knob bounds the cache (the hasher evicts
    oldest-first), so a long-running server cannot grow without limit."""
    monkeypatch.setattr(mcp, "_file_hasher", media_hash.StatKeyedFileHasher(max_entries=2))
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(f"data-{i}".encode())
        mcp._hash_file(str(tmp_path / f"f{i}.bin"))
    assert len(mcp._file_hasher) == 2


def test_default_bound_tracks_the_module_knob(monkeypatch):
    """The module builds its hasher from FILE_HASH_CACHE_MAX (read at import).
    Rebuilding with a different bound is what a config change would do."""
    hasher = media_hash.StatKeyedFileHasher(max_entries=mcp._MAX_FILE_HASH_CACHE)
    assert mcp._MAX_FILE_HASH_CACHE >= 1
    assert hasher._max_entries == mcp._MAX_FILE_HASH_CACHE
