"""P0 fix regression: self-selection requires PROOF (audit 2026-10-02).

The audit (documentation/AUDIT-2026-10-02.md, P0-1) demonstrated — by
execution — that ``verify_and_select`` demanded a password only when the
dataset HAS one: an unowned, password-less, non-public dataset could be
self-selected by any registry key, and the selection then widened
``dataset_allowed`` (the ACL check reads ``operator ACL ∪ selections``).
The fix makes proof mandatory at the shared choke point (REST and MCP both
funnel through :func:`access_store.verify_and_select`):

  * protected dataset  → the CORRECT password (unchanged);
  * public dataset     → no password needed (the feature, unchanged);
  * operator-ACL'd     → no proof needed (re-select / acl+pw sidecar, unchanged);
  * everything else    → SelectionDenied (the fix).

These tests pin the audit's exact repro (alice / ``bob-private``) plus every
previously-working path, at the same layer the audit exploited.

Run::

    pytest tests/security/test_selection_proof_gate.py -q
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr


@pytest.fixture(autouse=True)
def _store_env(tmp_path, monkeypatch):
    """Fresh store dir per test; the store ON (the P0 needs D16 active)."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setenv(acc.STORE_ENV, "1")
    monkeypatch.delenv(acc.DENY_ENV, raising=False)
    acc._mtime_cache.clear()
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()
    yield
    acc._mtime_cache.clear()
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()


def _client(name="alice", datasets=frozenset()):
    return cr.Identity(kind="client", name=name, datasets=frozenset(datasets))


def _mk_dataset(tmp_path, name, meta):
    """Create a dataset meta.json the way the manager would (the public-flag
    and ownership reads go through {DATA_PATH}/datasets/<name>/meta.json)."""
    d = Path(os.environ["DATA_PATH"]) / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def _no_pw(name):
    return False


def _check(name, supplied):
    return supplied == "correct-pw"


# ---------------------------------------------------------------------------
# The fix: no proof → no selection
# ---------------------------------------------------------------------------


def test_unowned_passwordless_private_dataset_refused(tmp_path):
    """The audit's exact repro: alice (ACL={alice-mem}) must NOT be able to
    select bob's password-less, non-public dataset."""
    _mk_dataset(tmp_path, "bob-private", {"created_by": "bob"})
    ident = _client("alice", {"alice-mem"})
    with pytest.raises(acc.SelectionDenied):
        acc.verify_and_select(
            ident, "bob-private", None, _no_pw, _check
        )
    # And the world did not widen.
    assert acc.selections_for(ident) == frozenset()
    assert acc.dataset_allowed(ident, "bob-private") is False


def test_refusal_leaves_no_store_entry(tmp_path):
    """A refused attempt must not persist anything (the old code selected
    FIRST and the ACL followed)."""
    _mk_dataset(tmp_path, "bob-private", {"created_by": "bob"})
    ident = _client("alice")
    with pytest.raises(acc.SelectionDenied):
        acc.verify_and_select(ident, "bob-private", None, _no_pw, _check)
    store = Path(os.environ["DATA_PATH"]) / "access" / "alice.json"
    assert not store.exists() or "bob-private" not in store.read_text()


# ---------------------------------------------------------------------------
# Previously-working paths preserved
# ---------------------------------------------------------------------------


def test_public_dataset_still_selectable_without_password(tmp_path):
    """Public = availability: password-free self-selection stays the feature."""
    _mk_dataset(tmp_path, "pub-ds", {"public": True})
    ident = _client("alice")
    entry = acc.verify_and_select(ident, "pub-ds", None, _no_pw, _check)
    assert entry["source"] == "self"
    assert acc.dataset_allowed(ident, "pub-ds") is True


def test_acl_granted_dataset_still_selectable(tmp_path):
    """An operator ACL grant needs no extra proof (re-select / no-op path)."""
    ident = _client("alice", {"acl-ds"})
    entry = acc.verify_and_select(ident, "acl-ds", None, _no_pw, _check)
    assert entry["source"] == "acl"
    assert acc.dataset_allowed(ident, "acl-ds") is True


def test_protected_dataset_with_correct_password_still_works(tmp_path):
    """The password path is unchanged — proof IS the correct password."""
    ident = _client("alice")
    entry = acc.verify_and_select(
        ident, "secret-ds", "correct-pw", lambda n: True, _check
    )
    assert entry["source"] == "self"
    assert acc.selection_password(ident, "secret-ds") == "correct-pw"


def test_wrong_password_still_valueerror(tmp_path):
    ident = _client("alice")
    with pytest.raises(ValueError, match="Incorrect password"):
        acc.verify_and_select(ident, "secret-ds", "wrong", lambda n: True, _check)


def test_missing_password_on_protected_still_valueerror(tmp_path):
    ident = _client("alice")
    with pytest.raises(ValueError, match="password protected"):
        acc.verify_and_select(ident, "secret-ds", None, lambda n: True, _check)


def test_denylist_still_refuses_even_with_password(tmp_path):
    """The operator floor wins over any proof (unchanged precedence)."""
    monkey_deny = "hr-data"
    os.environ[acc.DENY_ENV] = monkey_deny
    try:
        ident = _client("alice")
        with pytest.raises(acc.SelectionDenied):
            acc.verify_and_select(
                ident, "hr-data", "correct-pw", lambda n: True, _check
            )
    finally:
        os.environ.pop(acc.DENY_ENV, None)


def test_public_flag_with_password_hash_is_never_public(tmp_path):
    """Defense in depth from is_public_dataset: a hand-edited meta cannot
    publish a password-gated dataset — the password path must still gate."""
    _mk_dataset(tmp_path, "sneaky", {"public": True, "password_hash": "x"})
    ident = _client("alice")
    # has_password says True → the password branch runs (proof required).
    with pytest.raises(ValueError, match="password protected"):
        acc.verify_and_select(ident, "sneaky", None, lambda n: True, _check)


# ---------------------------------------------------------------------------
# Endpoint-level refusal mapping (cross-validation #1: SelectionDenied is a
# PermissionError — unhandled at the REST/MCP layer it surfaced as a 500 /
# opaque tool error instead of 403 / ToolError).
# ---------------------------------------------------------------------------


def test_rest_select_refusal_is_403_not_500(monkeypatch):
    """The audit repro through the REAL endpoint: a non-granted key selecting
    an unowned password-less non-public dataset gets 403 with the caller-safe
    message — never a 500."""

    from fastapi.testclient import TestClient

    import multimodal_rag.api_server as api
    import multimodal_rag.utils.clients_registry as cr

    class _FakeDM:
        def get_dataset(self, name, sync_count=False):
            return {"name": name, "document_count": 0}

        def has_password(self, name):
            return False

        def verify_password(self, name, pw):
            return False

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(acc.STORE_ENV, "1")
    api._UNLOCK_CACHE.clear()
    client = TestClient(api.app, raise_server_exceptions=False)
    r = client.post("/api/datasets/bob-private/select", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text[:200]}"
    assert "not available for self-selection" in r.json()["detail"]
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()


def test_mcp_select_refusal_is_toolerror_not_sdk_error(monkeypatch):
    """The MCP twin: a refused selection raises ToolError with the refusal
    message — not a bare SelectionDenied (the SDK wraps unknown exceptions as
    a generic 'Error executing tool' with no message)."""
    import asyncio

    import multimodal_rag.mcp_server as mcp

    class _DM:
        def get_dataset(self, name, sync_count=False):
            return {"name": name}

        def has_password(self, name):
            return False

        def verify_password(self, name, pw):
            return False

    monkeypatch.setattr(mcp, "get_manager", lambda: _DM())
    ident = cr.Identity(kind="client", name="alice", datasets=frozenset({"alice-mem"}))
    token = cr.set_current_identity(ident)
    try:
        with pytest.raises(mcp.ToolError, match="not available for self-selection"):
            asyncio.run(mcp.select_dataset(dataset_name="bob-private"))
    finally:
        cr.reset_current_identity(token)
