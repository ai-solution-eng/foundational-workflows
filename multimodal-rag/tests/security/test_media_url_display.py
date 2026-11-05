"""Media URLs must be RENDERABLE by a chat client — never a base64 blob.

2026-11 incident (Open WebUI, no workspace and no file storage): the model
called the bytes tool to display a PDF figure.  Two things in this server
pushed it there — `get_dataset_files` returned a RELATIVE, unsigned
`download_url` (unusable for an `<img>` tag, which cannot send headers), and
the search hint told non-rendering clients to fetch base64 "and embed it from
your workspace", advice that only makes sense for an agentic harness.  The
model then tried to reproduce ~300 KB of base64 in its reply.

These tests pin the corrected contract:

* `get_dataset_files` returns an ABSOLUTE, signed `download_url` (plus a
  ready-to-paste markdown link for images) when `MEDIA_BASE_URL` is set, and
  falls back to the relative REST path when it is not;
* the uuid prefix storage adds is stripped from the URL;
* the search hint routes DISPLAY to the signed URL and forbids inlining
  base64, while still allowing bytes for a client that can persist them.

Offline: real DatasetManager + local-mode Qdrant + the conftest stub embedder.
"""

import asyncio
import base64
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp
from multimodal_rag.dataset_manager import DatasetManager

qdrant_client = pytest.importorskip("qdrant_client")

# 1x1 transparent GIF — a real binary magic number (GIF8), no heavy deps.
_GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
_BASE = "https://rag.example.com"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_prefixes(tmp_path, monkeypatch):
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
    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("docs")
    mcp._dm = manager
    yield manager
    mcp._dm = None


@pytest.fixture(autouse=True)
def _clear_unlock_cache():
    mcp._unlocked.clear()


@pytest.fixture
def signed(monkeypatch):
    """A configured deployment: a browser-reachable base URL and a token key."""
    monkeypatch.setattr(mcp, "MEDIA_BASE_URL", _BASE)
    monkeypatch.setattr(mcp, "_MEDIA_TOKEN_SECRET", "test-secret")
    return _BASE


@pytest.fixture
def unsigned(monkeypatch):
    """MEDIA_BASE_URL unset — the deployment that degrades to relative paths."""
    monkeypatch.setattr(mcp, "MEDIA_BASE_URL", "")
    monkeypatch.setattr(mcp, "_MEDIA_TOKEN_SECRET", "test-secret")


def _stage_file(dm: DatasetManager, name: str, data: bytes):
    files_dir = dm._dataset_dir("docs") / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    p = files_dir / name
    p.write_bytes(data)
    return p


def _files(**kwargs) -> dict:
    return json.loads(asyncio.run(mcp.get_dataset_files(**kwargs)))


# ---------------------------------------------------------------------------
# The URL helper
# ---------------------------------------------------------------------------


def test_public_media_url_is_none_without_a_base_url(unsigned):
    assert mcp._public_media_url("docs", "pixel.gif") is None


def test_public_media_url_is_absolute_signed_and_uuid_stripped(signed):
    url = mcp._public_media_url("docs", "0123456789abcdef0123456789abcdef_pixel.gif")
    assert url is not None
    assert url.startswith(_BASE + "/api/datasets/docs/files/pixel.gif?")
    assert "token=" in url
    # The random hex prefix is exactly what models garble — it must be gone.
    assert "0123456789abcdef" not in url


# ---------------------------------------------------------------------------
# get_dataset_files: the binary branch
# ---------------------------------------------------------------------------


def test_binary_returns_renderable_url_and_markdown(dm, signed):
    _stage_file(dm, "pixel.gif", _GIF)
    out = _files(dataset_name="docs", file_path="pixel.gif")

    assert out["type"] == "image"
    assert out["download_url"].startswith(_BASE + "/api/datasets/docs/files/pixel.gif?")
    assert "token=" in out["download_url"]
    assert out["expires_in_seconds"] > 0
    # A chat client can paste this straight into its reply.  The LABEL is
    # markdown-inert, so the dot is backslash-escaped — that is intended.
    assert out["markdown"].startswith("![")
    assert "pixel" in out["markdown"]
    assert f"]({_BASE}" in out["markdown"]
    assert "note" not in out, "a configured deployment must not fall back"


def test_markdown_label_cannot_splice_a_second_link(dm, signed):
    """P1-11 parity: a crafted upload filename must not close the image and
    open a link the model then emits verbatim."""
    hostile = "a](b)[c](d).gif"
    _stage_file(dm, hostile, _GIF)
    out = _files(dataset_name="docs", file_path=hostile)

    md = out["markdown"]
    assert md.startswith("![")
    # The live link syntax must not survive in the LABEL — only the one
    # destination we control may appear as `](...)`.
    assert md.count("](") == 1, md
    assert "](b)" not in md
    assert md.endswith(")")


def test_binary_falls_back_to_the_relative_path(dm, unsigned):
    _stage_file(dm, "pixel.gif", _GIF)
    out = _files(dataset_name="docs", file_path="pixel.gif")

    assert out["download_url"] == "/api/datasets/docs/files/pixel.gif"
    assert "MEDIA_BASE_URL" in out["note"]
    assert "markdown" not in out
    assert "expires_in_seconds" not in out


def test_bytes_mode_is_unchanged(dm, signed):
    """The bytes mode still works — it is the DISPLAY path that changed."""
    _stage_file(dm, "pixel.gif", _GIF)
    out = _files(dataset_name="docs", file_path="pixel.gif", return_bytes=True)
    assert base64.b64decode(out["data"]) == _GIF
    assert "download_url" not in out


# ---------------------------------------------------------------------------
# The search hint
# ---------------------------------------------------------------------------


def test_search_hint_forbids_inlining_base64(signed):
    doc = {
        "text": "a figure",
        "source": "paper.pdf",
        "image": "file:///data/datasets/docs/files/pixel.gif",
    }
    payload = mcp._format_retrieval_result("docs", [doc], [dict(doc)], [0.9], signed)
    context = payload["context"]

    assert "NEVER paste base64" in context
    assert "chat clients have no workspace" in context
    # The old advice (decode the base64 in your workspace and embed it) is
    # what produced a 300 KB blob in the reply — it must be gone.
    assert "decode the base64 payload to a local file" not in context
    # The renderable link itself is still emitted, verbatim.
    assert "![matched image" in context or "![" in context
