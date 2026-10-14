"""memory_status tool (2026-10): the report-only memory-write capability
probe for automated memory writers (the gateway distiller, harness shims).

Pinned semantics:
* answers ONLY about the CURRENT caller (identity via the D15 contextvar -
  no arguments, no cross-identity oracle);
* report-only: no unlock-cache write, no throttle consumption, no dataset
  mutation (the probe must be free to call before every distill);
* memory_available = a dataset resolves (binding / header / env fallback);
  writable = D15 ACL pass AND D26 write mode;
  unlocked = no password | saved selection password | unlock-cache hit;
  a DatasetManager construction failure leaves unlocked=None (unknown) and
  the writer attempts the real add_memory, which re-checks everything.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp


async def _status():
    r = await mcp.mcp.call_tool("memory_status", {})
    assert not r.is_error, r.content[0].text
    return json.loads(r.content[0].text)


def test_registered():
    names = asyncio.run(mcp.mcp.list_tools())
    assert "memory_status" in [t.name for t in names]


def test_unbound_identity_reports_unavailable():
    """No identity, no header, no env -> available:false with the refusal
    detail (the same message add_memory would raise)."""
    out = asyncio.run(_status())
    assert out["memory_available"] is False
    assert out["dataset"] is None
    assert "No memory dataset" in out["detail"]


def test_env_fallback_dataset_reports_available():
    """MEMORY_DATASET env (the bare single-user deployment shape) resolves."""
    os.environ["MEMORY_DATASET"] = "some-dataset"
    try:
        out = asyncio.run(_status())
        assert out["memory_available"] is True
        assert out["dataset"] == "some-dataset"
        # no ACL store configured -> admin-class caller -> writable
        assert out["writable"] is True
    finally:
        os.environ.pop("MEMORY_DATASET", None)


def test_report_only_no_side_effects(monkeypatch):
    """The probe must not write the unlock cache nor consume the throttle:
    _cache_unlock and _verify_password_throttled are never called on the
    memory_status path (the DM-unavailable branch also returns early)."""
    called = {"cache": 0, "throttle": 0}

    def boom(*a, **k):
        called["cache"] += 1
        raise AssertionError("probe must not cache unlocks")

    def boom2(*a, **k):
        called["throttle"] += 1
        raise AssertionError("probe must not consume the throttle")

    monkeypatch.setattr(mcp, "_cache_unlock", boom)
    monkeypatch.setattr(mcp, "_verify_password_throttled", boom2)
    os.environ["MEMORY_DATASET"] = "some-dataset"
    try:
        out = asyncio.run(_status())
        assert out["memory_available"] is True
    finally:
        os.environ.pop("MEMORY_DATASET", None)
    assert called == {"cache": 0, "throttle": 0}


def test_unlocked_unknown_when_dm_unavailable(monkeypatch):
    """DatasetManager construction needs the embedder env; a probe must NOT
    require model-serving config. dm failure -> unlocked None + detail."""

    def boom():
        raise RuntimeError("Embedder model is required")

    monkeypatch.setattr(mcp, "get_manager", boom)
    os.environ["MEMORY_DATASET"] = "some-dataset"
    try:
        out = asyncio.run(_status())
        assert out["memory_available"] is True
        assert out["unlocked"] is None
        assert "dataset manager unavailable" in out["detail"]
    finally:
        os.environ.pop("MEMORY_DATASET", None)
