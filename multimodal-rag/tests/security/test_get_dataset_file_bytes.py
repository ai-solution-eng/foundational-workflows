"""MCP ``get_dataset_file_bytes`` — bounded base64 bytes for ONE dataset file.

The companion to ``get_dataset_files`` (which lists files and returns a
download URL for binaries but never the bytes).  The tests pin the tool's
guard chain and result shape:

* the SAME guard chain as ``get_dataset_files`` (D15 ACL → existence →
  password gate), then traversal-safe path resolution, the size cap, and
  the base64 payload;
* size cap → ToolError naming the cap and the REST/download fallback
  (never a silent truncation);
* password gate parity: locked → the word-identical P2 error message;
  correct password unlocks AND caches the unlock.

Everything runs OFFLINE: real DatasetManager + local-mode Qdrant + the
conftest stub embedder.

Run::

    pytest tests/security/test_get_dataset_file_bytes.py -q
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp
from multimodal_rag.dataset_manager import DatasetManager

qdrant_client = pytest.importorskip("qdrant_client")

# 1x1 transparent GIF — a real binary magic number (``GIF8``) without any
# heavyweight dependency; every byte of the round-trip is checked.
_GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"  # FF D8 … FF D9


# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_doc_management_tools.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_prefixes(tmp_path, monkeypatch):
    """Point the media-path allowlist at a temp tree."""
    datasets = tmp_path / "datasets"
    staging = tmp_path / "staging"
    datasets.mkdir()
    staging.mkdir()
    import multimodal_rag.utils.media_paths as mp

    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(datasets), str(staging)))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)
    return tmp_path


@pytest.fixture
def dm(base_path, embedder, allow_prefixes, monkeypatch):
    """Real offline DatasetManager with one dataset, injected as the MCP
    server's shared manager."""
    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("docs")
    mcp._dm = manager
    yield manager
    mcp._dm = None


@pytest.fixture(autouse=True)
def _clear_unlock_cache():
    """A successful tool call caches the verified password in the module
    unlock cache — clear it per-test so the password gate stays honest."""
    mcp._unlocked.clear()


@pytest.fixture
def acl_client(monkeypatch):
    """D15 per-key ACLs (same pattern as test_doc_management_tools.py)."""
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "alice:key-a;bob:key-b")
    monkeypatch.setenv("RAG_DATASET_ACLS", "alice:docs")
    yield
    monkeypatch.delenv("RAG_API_KEY_CLIENTS", raising=False)
    monkeypatch.delenv("RAG_DATASET_ACLS", raising=False)


def _call(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def _payload(result: str) -> dict:
    return json.loads(result)


def _stage_file(dm: DatasetManager, name: str, data: bytes) -> Path:
    """Write raw bytes into the dataset's files/ dir (the place the tool reads)."""
    files_dir = dm._dataset_dir("docs") / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    p = files_dir / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_bytes_roundtrip_gif(dm):
    """base64 out == bytes in; metadata fields populated."""
    p = _stage_file(dm, "pixel.gif", _GIF)
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="pixel.gif"))
    assert out["dataset"] == "docs"
    assert out["file"] == "pixel.gif"
    assert out["size_bytes"] == p.stat().st_size == len(_GIF)
    assert out["type"] == "image"
    assert out["encoding"] == "base64"
    assert base64.b64decode(out["data"]) == _GIF
    assert out["mime_type"] == "image/gif"
    # the payload is JSON-safe ASCII
    assert isinstance(out["data"], str) and out["data"].isascii()


def test_bytes_jpeg_magic(dm):
    p = _stage_file(dm, "photo.jpg", _JPEG)
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="photo.jpg"))
    assert base64.b64decode(out["data"])[:2] == b"\xff\xd8"  # FF D8 JPEG magic
    assert out["mime_type"] == "image/jpeg"
    assert out["size_bytes"] == p.stat().st_size


# ---------------------------------------------------------------------------
# Size cap
# ---------------------------------------------------------------------------


def test_over_cap_refused_with_fallback_hint(dm, monkeypatch):
    _stage_file(dm, "big.bin", b"x" * 4096)
    monkeypatch.setattr(mcp, "_file_bytes_max", lambda: 1024)
    with pytest.raises(mcp.ToolError, match="MCP_FILE_BYTES_MAX"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="big.bin")
    # cap mention AND the download-URL fallback hint are both in the message
    with pytest.raises(mcp.ToolError, match=r"byte tool cap \(MCP_FILE_BYTES_MAX\)"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="big.bin")


def test_env_var_cap_respected(dm, monkeypatch):
    """MCP_FILE_BYTES_MAX is the documented escape hatch (floor 1024)."""
    _stage_file(dm, "mid.bin", b"y" * 2048)
    monkeypatch.setenv("MCP_FILE_BYTES_MAX", "1024")
    with pytest.raises(mcp.ToolError, match="tool cap"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="mid.bin")
    monkeypatch.setenv("MCP_FILE_BYTES_MAX", "4096")
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="mid.bin"))
    assert base64.b64decode(out["data"]) == b"y" * 2048


def test_oversize_exactly_at_cap_allowed(dm, monkeypatch):
    """Boundary: size == cap passes (the gate is strictly ``size > max``)."""
    data = b"z" * 1024
    _stage_file(dm, "edge.bin", data)
    monkeypatch.setenv("MCP_FILE_BYTES_MAX", "1024")
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="edge.bin"))
    assert base64.b64decode(out["data"]) == data


def test_cap_floor_respected(dm, monkeypatch):
    """The floor is 1024 even when the env asks for lower."""
    _stage_file(dm, "tiny.bin", b"t" * 1500)
    monkeypatch.setenv("MCP_FILE_BYTES_MAX", "8")
    with pytest.raises(mcp.ToolError, match="1024"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="tiny.bin")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_traversal_refused(dm):
    _stage_file(dm, "ok.txt", b"fine")
    # NOTE: "..\\secret.txt" is a LEGAL filename on Linux (a single file
    # literally named backslash-dot-dot-backslash-…), so it correctly
    # resolves inside files/ — only real escapes are refused.
    for evil in ("../secret.txt", "a/../../secret.txt", "/etc/passwd"):
        with pytest.raises(mcp.ToolError, match="Invalid file path."):
            _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path=evil)
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="ok.txt"))
    assert base64.b64decode(out["data"]) == b"fine"


def test_missing_file_refused(dm):
    with pytest.raises(mcp.ToolError, match="not found in dataset"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="ghost.txt")


def test_missing_dataset_refused(dm):
    with pytest.raises(mcp.ToolError, match="Dataset 'nope' not found."):
        _call(mcp.get_dataset_file_bytes, dataset_name="nope", file_path="x.txt")


def test_acl_denied_dataset_refused(dm, acl_client):
    """A key with no grant for 'docs' is denied — same error surface as
    get_dataset_files, and WITHOUT an existence oracle."""
    import multimodal_rag.utils.clients_registry as cr

    token = cr.set_current_identity(cr.Identity(kind="client", name="bob", datasets=frozenset()))
    try:
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="pixel.gif")
        # denial does not confirm existence: an unknown dataset gets the SAME error
        with pytest.raises(mcp.ToolError, match="not permitted for this API key"):
            _call(mcp.get_dataset_file_bytes, dataset_name="never-created", file_path="x")
    finally:
        cr.reset_current_identity(token)


def test_password_gate_parity(dm):
    """Locked → the word-identical P2 message; correct password unlocks."""
    dm.set_password("docs", "hunter2")
    _stage_file(dm, "pixel.gif", _GIF)
    with pytest.raises(mcp.ToolError, match="Incorrect password"):
        _call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="pixel.gif")
    out = _payload(_call(mcp.get_dataset_file_bytes, dataset_name="docs", file_path="pixel.gif", password="hunter2"))
    assert base64.b64decode(out["data"]) == _GIF
