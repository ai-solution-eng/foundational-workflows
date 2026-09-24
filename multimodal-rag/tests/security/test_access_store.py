"""D16 self-service dataset selection + memory binding (access store).

The ratified model: any authenticated key SEES all dataset names (discovery),
SELECTS the ones it wants — public datasets freely, protected ones only with
the correct password, which is then saved per identity so both surfaces work
without passwords — and effective access is operator ACL ∪ selections (the
ACL is a floor).  OFF by default (``RAG_ACCESS_STORE``): every test here
exercises the knob explicitly, and the default-without-env behaviour is
pinned byte-identical.

Run::

    pytest tests/security/test_access_store.py -q
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr


@pytest.fixture(autouse=True)
def _store_env(tmp_path, monkeypatch):
    """Each test gets a fresh store dir under tmp; the store is OFF unless
    the test turns it on."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.delenv(acc.STORE_ENV, raising=False)
    monkeypatch.delenv(acc.DENY_ENV, raising=False)
    monkeypatch.delenv(acc.MEMORY_DEFAULT_ENV, raising=False)
    acc._mtime_cache.clear()
    yield
    acc._mtime_cache.clear()


def _client(name="alice", datasets=frozenset()):
    return cr.Identity(kind="client", name=name, datasets=frozenset(datasets))


def _admin():
    return cr.Identity(kind="admin", name=None, datasets=None)


# ---------------------------------------------------------------------------
# Store mechanics
# ---------------------------------------------------------------------------


def test_store_off_is_inert(monkeypatch):
    ident = _client("alice", {"a"})
    assert acc.store_enabled() is False
    assert acc.selections_for(ident) == frozenset()
    assert acc.selection_password(ident, "a") is None
    assert acc.dataset_allowed(ident, "a") is True   # via ACL
    assert acc.dataset_allowed(ident, "b") is False  # no store → no widening
    with pytest.raises(acc.SelectionDenied):
        acc.select_dataset(ident, "b")


def test_select_public_dataset_persists(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    entry = acc.select_dataset(ident, "public-ds")
    assert entry["source"] == "self"
    assert acc.selections_for(ident) == {"public-ds"}
    assert acc.dataset_allowed(ident, "public-ds") is True
    assert acc.selection_password(ident, "public-ds") is None  # no password saved
    # Persisted to the PVC file with 0600.
    p = Path(os.environ["DATA_PATH"]) / "access" / "alice.json"
    assert p.exists()
    doc = json.loads(p.read_text())
    assert doc["datasets"]["public-ds"]["dataset"] == "public-ds"


def test_select_saves_verified_password(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    acc.select_dataset(ident, "secret-ds", password="pw123")
    assert acc.selection_password(ident, "secret-ds") == "pw123"
    assert acc.dataset_allowed(ident, "secret-ds") is True


def test_select_union_acl_is_floor(monkeypatch):
    """ACL ∪ selections: selections widen, never narrow; ACL cannot be lost."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice", {"acl-ds"})
    acc.select_dataset(ident, "self-ds", password="pw")
    assert acc.dataset_allowed(ident, "acl-ds") is True
    assert acc.dataset_allowed(ident, "self-ds") is True
    assert acc.dataset_allowed(ident, "other") is False
    # Deselecting the self-selection removes ONLY that widening.
    assert acc.deselect_dataset(ident, "self-ds") is True
    assert acc.dataset_allowed(ident, "self-ds") is False
    assert acc.dataset_allowed(ident, "acl-ds") is True  # floor intact


def test_select_acl_granted_dataset_is_noop_without_entry(monkeypatch):
    """An ACL'd dataset offered WITHOUT a password needs no selection entry:
    the ACL is the stronger grant, and shadowing it would break operator
    revocation.  (Offering the password is the memory-sidecar case — see
    test_acl_granted_dataset_accepts_and_saves_password.)"""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice", {"acl-ds"})
    entry = acc.select_dataset(ident, "acl-ds")
    assert entry["source"] == "acl"
    assert acc.selections_for(ident) == set()  # nothing stored
    assert acc.selection_password(ident, "acl-ds") is None


def test_select_denies_denylisted_even_with_proof(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    monkeypatch.setenv(acc.DENY_ENV, "hr-data, payroll")
    ident = _client("alice")
    with pytest.raises(acc.SelectionDenied):
        acc.select_dataset(ident, "hr-data", password="correct")


def test_select_rejects_bad_identity_names(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = cr.Identity(kind="client", name="../evil", datasets=frozenset())
    with pytest.raises(ValueError):
        acc.select_dataset(ident, "ds")


def test_store_file_survives_concurrent_writers(monkeypatch, tmp_path):
    """Two in-process writers serialize (threading lock) — no lost update."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    import threading

    def worker(n):
        acc.select_dataset(ident, f"ds-{n}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(acc.selections_for(ident)) == 8


def test_corrupt_store_file_degrades_to_empty(monkeypatch, tmp_path):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    p = tmp_path / "access" / "alice.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    ident = _client("alice")
    assert acc.selections_for(ident) == frozenset()
    # And the store recovers: a select rewrites a valid file.
    acc.select_dataset(ident, "ds")
    assert json.loads(p.read_text())["datasets"]["ds"]["dataset"] == "ds"


# ---------------------------------------------------------------------------
# verify_and_select (the shared proof flow)
# ---------------------------------------------------------------------------


def test_verify_and_select_password_flow(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    entry = acc.verify_and_select(
        ident, "secret-ds", "correct",
        has_password=lambda n: True,
        verify_password=lambda n, pw: pw == "correct",
    )
    assert entry["password"] == "correct"
    with pytest.raises(ValueError, match="Incorrect password"):
        acc.verify_and_select(
            ident, "secret-ds", "WRONG",
            has_password=lambda n: True,
            verify_password=lambda n, pw: pw == "correct",
        )
    with pytest.raises(ValueError, match="password protected"):
        acc.verify_and_select(
            ident, "secret-ds", None,
            has_password=lambda n: True,
            verify_password=lambda n, pw: pw == "correct",
        )


# ---------------------------------------------------------------------------
# Memory binding
# ---------------------------------------------------------------------------


def test_memory_binding_roundtrip(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    assert acc.memory_dataset_for(ident, "env-fallback") == "env-fallback"  # unbound
    acc.select_dataset(ident, "mymem")
    acc.set_memory_dataset(ident, "mymem")
    assert acc.memory_dataset_for(ident, "env-fallback") == "mymem"
    acc.set_memory_dataset(ident, None)
    assert acc.memory_dataset_for(ident, "env-fallback") == "env-fallback"


def test_memory_binding_requires_accessibility(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    with pytest.raises(acc.SelectionDenied):
        acc.set_memory_dataset(ident, "not-mine")
    acc.select_dataset(ident, "mymem")
    acc.set_memory_dataset(ident, "mymem")
    # Losing the selection (deselect) fails the binding soft — falls back.
    acc.deselect_dataset(ident, "mymem")
    assert acc.memory_dataset_for(ident, "env-fallback") == "env-fallback"


def test_memory_default_env_used_as_last_fallback(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    monkeypatch.setenv(acc.MEMORY_DEFAULT_ENV, "deployment-mem")
    ident = _client("alice")
    # The env default is the CALLER's fallback argument in the real flow —
    # the module itself never reads it (resolution order lives in the
    # servers); pin that here so the env is wired in exactly one place.
    assert acc.memory_dataset_for(ident, "caller-provided") == "caller-provided"


# ---------------------------------------------------------------------------
# MCP integration: gate union + discovery listing + tool flow
# ---------------------------------------------------------------------------


def test_mcp_gate_union(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice", {"acl-ds"})
    token = cr.set_current_identity(ident)
    try:
        acc.select_dataset(ident, "self-ds", password="pw")
        assert mcp._identity_dataset_allowed("acl-ds")
        assert mcp._identity_dataset_allowed("self-ds")
        assert not mcp._identity_dataset_allowed("other")
        mcp._require_dataset_acl("self-ds")  # no raise
        with pytest.raises(Exception, match="not permitted"):
            mcp._require_dataset_acl("other")
    finally:
        cr.reset_current_identity(token)


def test_mcp_memory_resolution_uses_binding(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    token = cr.set_current_identity(ident)
    try:
        acc.select_dataset(ident, "bound-mem", password="mempw")
        acc.set_memory_dataset(ident, "bound-mem")
        assert mcp._resolve_memory_dataset(None) == "bound-mem"
        assert mcp._resolve_memory_password(None) == "mempw"
    finally:
        cr.reset_current_identity(token)


def test_mcp_memory_resolution_env_fallback_unchanged(monkeypatch):
    """Without the store on, resolution behaves exactly as before: the env
    dataset fallback, and the password coming only from the explicit arg or
    the request header (RAG_MEMORY_PASSWORD is an opencode-side var that
    arrives AS the header — the server never reads it from the environment)."""
    monkeypatch.setenv("MEMORY_DATASET", "env-ds")
    ident = _client("alice")
    token = cr.set_current_identity(ident)
    pw_token = mcp._memory_password_ctx.set("header-pw")
    try:
        assert mcp._resolve_memory_dataset(None) == "env-ds"
        assert mcp._resolve_memory_password(None) == "header-pw"
    finally:
        cr.reset_current_identity(token)
        mcp._memory_password_ctx.reset(pw_token)
        monkeypatch.delenv("MEMORY_DATASET")


def test_mcp_saved_selection_satisfies_unlock_check(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")
    token = cr.set_current_identity(ident)
    mcp._unlocked.clear()
    try:
        acc.select_dataset(ident, "protected-ds", password="saved-pw")
        # The MCP unlock-cache read falls through to the saved selection.
        assert mcp._is_unlocked("protected-ds") == "saved-pw"
    finally:
        cr.reset_current_identity(token)
        mcp._unlocked.clear()


def test_mcp_select_tool_end_to_end(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("alice")

    class _DM:
        def get_dataset(self, name, check_embedder=True):
            return {"name": name}

        def has_password(self, name):
            return name == "protected-ds"

        def verify_password(self, name, pw):
            return pw == "right"

    monkeypatch.setattr(mcp, "get_manager", lambda: _DM())
    token = cr.set_current_identity(ident)
    try:
        out = asyncio.run(mcp.select_dataset(dataset_name="protected-ds", password="right"))
        assert "selected" in out
        assert "password saved" in out
        assert acc.dataset_allowed(ident, "protected-ds")
        # Wrong password → refused, and nothing selected.
        with pytest.raises(Exception, match="Incorrect password"):
            asyncio.run(mcp.select_dataset(dataset_name="protected-ds", password="bad"))
        # Deselect round-trip.
        out = asyncio.run(mcp.deselect_dataset(dataset_name="protected-ds"))
        assert "deselected" in out
        assert not acc.dataset_allowed(ident, "protected-ds")
    finally:
        cr.reset_current_identity(token)
        mcp._unlocked.clear()


# ---------------------------------------------------------------------------
# REST integration (full stack: middleware → gate → handler)
# ---------------------------------------------------------------------------


class _FakeDM:
    def list_datasets(self):
        return [{"name": "public-ds", "document_count": 1}, {"name": "protected-ds", "document_count": 2}]

    def get_dataset(self, name, sync_count=False):
        if name == "missing":
            raise FileNotFoundError(f"Dataset '{name}' not found")
        return {"name": name, "document_count": 0}

    def has_password(self, name):
        return name == "protected-ds"

    def verify_password(self, name, pw):
        return pw == "right"


@pytest.fixture
def rest_client(monkeypatch):
    from fastapi.testclient import TestClient

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.delenv(cr.ACLS_ENV, raising=False)
    monkeypatch.setenv(acc.STORE_ENV, "1")
    api._UNLOCK_CACHE.clear()
    acc._mtime_cache.clear()
    yield TestClient(api.app)
    api._UNLOCK_CACHE.clear()
    acc._mtime_cache.clear()


def test_rest_listing_shows_only_effective_datasets(rest_client):
    """The 2026-09-24 ruling REVERSED discovery mode: the listing shows only
    the caller's EFFECTIVE datasets (operator grants ∪ self-selections) —
    access isolation, not name discovery.  With no grants/selections the
    list is empty (with the honest hidden count)."""
    r = rest_client.get("/api/datasets", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    assert r.json()["datasets"] == []
    assert r.json()["acl_hidden"] == 2
    # After a selection the dataset appears in the listing.
    rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    r = rest_client.get("/api/datasets", headers={"X-RAG-Api-Key": "alice-key"})
    assert [d["name"] for d in r.json()["datasets"]] == ["public-ds"]
    assert r.json()["acl_hidden"] == 1


def test_rest_listing_filtered_without_store(monkeypatch):
    """Store off → the D15 hiding is preserved byte-identically."""
    from fastapi.testclient import TestClient

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:public-ds")
    monkeypatch.delenv(acc.STORE_ENV, raising=False)
    client = TestClient(api.app)
    r = client.get("/api/datasets", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    assert [d["name"] for d in r.json()["datasets"]] == ["public-ds"]
    assert r.json()["acl_hidden"] == 1


def test_rest_select_public_no_body(rest_client):
    r = rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    assert r.json()["source"] == "self"
    ident = _client("alice")
    assert acc.dataset_allowed(ident, "public-ds")


def test_rest_select_protected_requires_password(rest_client):
    r = rest_client.post("/api/datasets/protected-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403  # missing password
    r = rest_client.post(
        "/api/datasets/protected-ds/select", json={"password": "wrong"}, headers={"X-RAG-Api-Key": "alice-key"}
    )
    assert r.status_code == 403
    r = rest_client.post(
        "/api/datasets/protected-ds/select", json={"password": "right"}, headers={"X-RAG-Api-Key": "alice-key"}
    )
    assert r.status_code == 200
    assert r.json()["source"] == "self"
    ident = _client("alice")
    assert acc.selection_password(ident, "protected-ds") == "right"


def test_rest_selection_grants_dataset_access(rest_client):
    """The payoff: after selecting, the REST dataset path is allowed —
    and the saved password satisfies the password gate."""
    rest_client.post(
        "/api/datasets/protected-ds/select", json={"password": "right"}, headers={"X-RAG-Api-Key": "alice-key"}
    )
    r = rest_client.get("/api/datasets/protected-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200  # union gate + saved-password fallback


def test_rest_unselected_protected_stays_gated(rest_client):
    """Without a selection, a protected dataset is refused at the ACL gate
    (403) AND hidden from the listing — access isolation, per the ruling."""
    r = rest_client.get("/api/datasets/protected-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403
    assert "not permitted" in r.json()["detail"]
    # But a public dataset is readable once selected.
    rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    r = rest_client.get("/api/datasets/public-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200


def test_rest_selections_endpoint_never_returns_passwords(rest_client):
    rest_client.post(
        "/api/datasets/protected-ds/select", json={"password": "right"}, headers={"X-RAG-Api-Key": "alice-key"}
    )
    r = rest_client.get("/api/access/selections", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    entry = body["selections"]["protected-ds"]
    assert entry["has_password"] is True  # boolean flag only…
    assert "right" not in r.text          # …never the password value itself
    assert set(entry.keys()) == {"selected_at", "source", "has_password"}


def test_rest_deselect_removes_access(rest_client):
    rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    r = rest_client.get("/api/datasets/public-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    rest_client.post("/api/datasets/public-ds/deselect", headers={"X-RAG-Api-Key": "alice-key"})
    r = rest_client.get("/api/datasets/public-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403


def test_rest_select_endpoint_reachable_for_ungranted_datasets(rest_client):
    """The select endpoint itself must NOT be pre-blocked by the ACL gate
    (it is how access is gained) — while every other path stays gated."""
    r = rest_client.post("/api/datasets/protected-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403  # from the PASSWORD proof, not the middleware
    assert "password" in r.json()["detail"].lower()
    r = rest_client.get("/api/datasets/protected-ds", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403  # still gated — no selection was made


def test_rest_memory_binding_endpoint(rest_client):
    rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "alice-key"})
    r = rest_client.post("/api/access/memory-dataset", json={"dataset": "public-ds"}, headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    assert r.json()["memory_dataset"] == "public-ds"
    # Binding to an inaccessible dataset is refused.
    r = rest_client.post("/api/access/memory-dataset", json={"dataset": "protected-ds"}, headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 409
    # Clear.
    r = rest_client.post("/api/access/memory-dataset", json={}, headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    assert r.json()["memory_dataset"] is None


def test_rest_selection_admin_key_refused(rest_client):
    r = rest_client.post("/api/datasets/public-ds/select", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# ACL-granted + password sidecar (source: "acl+pw") — the memory-binding case
# ---------------------------------------------------------------------------


def test_acl_granted_dataset_accepts_and_saves_password(monkeypatch):
    """The reported UX gap: a user whose memory dataset was ADMIN-GRANTED
    (ACL) could not record the dataset password anywhere — the ★ binding
    resolved the dataset but every memory call still demanded a password.
    Selecting an ACL-granted PROTECTED dataset with the correct password now
    saves a password-only sidecar entry (source: acl+pw)."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("andrew", {"andrew-memory"})
    entry = acc.select_dataset(ident, "andrew-memory", password="mempw")
    assert entry["source"] == "acl+pw"
    assert acc.selection_password(ident, "andrew-memory") == "mempw"
    assert acc.dataset_allowed(ident, "andrew-memory")  # ACL anyway — floor intact


def test_acl_granted_without_password_stays_unstored(monkeypatch):
    """No password offered on an ACL-granted dataset → no store entry (the
    minimal-store rule; the grant is the access)."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("andrew", {"acl-ds"})
    entry = acc.select_dataset(ident, "acl-ds")
    assert entry["source"] == "acl"
    assert acc.selections_for(ident) == set()


def test_acl_pw_sidecar_removed_by_deselect(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("andrew", {"andrew-memory"})
    acc.select_dataset(ident, "andrew-memory", password="mempw")
    assert acc.deselect_dataset(ident, "andrew-memory") is True
    assert acc.selection_password(ident, "andrew-memory") is None
    # The ACL grant itself survives the deselect (floor).
    assert acc.dataset_allowed(ident, "andrew-memory")


def test_verify_and_select_passes_password_for_acl_granted(monkeypatch):
    """The REST/MCP shared proof flow feeds the password through for
    ACL-granted datasets too (it used to be swallowed by the no-op branch)."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("andrew", {"andrew-memory"})
    entry = acc.verify_and_select(
        ident, "andrew-memory", "mempw",
        has_password=lambda n: True,
        verify_password=lambda n, pw: pw == "mempw",
    )
    assert entry["source"] == "acl+pw"
    assert acc.selection_password(ident, "andrew-memory") == "mempw"


def test_memory_binding_with_acl_grant_resolves_password(monkeypatch):
    """End-to-end shape of the user's exact case: memory dataset granted via
    the admin panel, password recorded once on the page, ★ bound — the MCP
    memory resolution finds BOTH the dataset and the password."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = _client("andrew", {"andrew-memory"})
    acc.select_dataset(ident, "andrew-memory", password="mempw")
    acc.set_memory_dataset(ident, "andrew-memory")
    assert acc.memory_dataset_for(ident, None) == "andrew-memory"
    assert acc.selection_password(ident, "andrew-memory") == "mempw"
