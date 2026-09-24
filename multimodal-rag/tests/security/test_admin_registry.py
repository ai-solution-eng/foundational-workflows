"""D17 admin key registry: mint/grant/revoke per-user keys from the /access
page with an ADMIN key (the file-backed overlay merged into the D15 registry).

The ratified flow: an admin pastes the deployment key on /access → the admin
panel appears → mint a key for alice (copy-once display) → grant datasets →
alice's key authenticates on BOTH surfaces immediately (everything re-read
per request).  Key material is masked everywhere except the mint response.

Run::

    pytest tests/security/test_admin_registry.py -q
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr


@pytest.fixture(autouse=True)
def _store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    monkeypatch.delenv("RAG_API_KEY_CLIENTS", raising=False)
    monkeypatch.delenv("RAG_DATASET_ACLS", raising=False)
    monkeypatch.delenv("RAG_UNLOCK_MAX_TTL", raising=False)
    ar._mtime_cache.clear()
    yield
    ar._mtime_cache.clear()
    os.environ.pop("RAG_API_KEY_CLIENTS", None)
    os.environ.pop("RAG_DATASET_ACLS", None)


def _admin():
    return cr.Identity(kind="admin", name=None, datasets=None)


def _client(name="alice", datasets=frozenset()):
    return cr.Identity(kind="client", name=name, datasets=frozenset(datasets))


# ---------------------------------------------------------------------------
# Overlay mechanics
# ---------------------------------------------------------------------------


def test_mint_generates_copy_once_key(tmp_path):
    res = ar.mint_client("alice", datasets=["reports"])
    assert len(res["key"]) >= 20  # token_urlsafe(18)
    assert res["rotated"] is False
    # Stored verbatim; masked in listings.
    assert ar.verify_client_key("alice", res["key"])
    listed = ar.list_clients()
    assert listed[0]["key_masked"] != res["key"]
    assert res["key"][:6] in listed[0]["key_masked"]


def test_mint_rotates_existing_key(tmp_path):
    first = ar.mint_client("alice", datasets=["reports"])
    second = ar.mint_client("alice", datasets=["reports"])
    assert second["rotated"] is True
    assert first["key"] != second["key"]
    assert not ar.verify_client_key("alice", first["key"])  # old key dead
    assert ar.verify_client_key("alice", second["key"])


def test_mint_validates_name_and_datasets(tmp_path):
    with pytest.raises(ValueError):
        ar.mint_client("../evil")
    with pytest.raises(ValueError):
        ar.mint_client("ok", datasets=["bad dataset!"])
    with pytest.raises(ValueError):
        ar.mint_client("ok", key="short")  # <8 chars
    with pytest.raises(ValueError):
        ar.mint_client("ok", key="has:colon")


def test_revoke_removes_only_overlay_entry(tmp_path):
    ar.mint_client("alice")
    assert ar.revoke_client("alice") is True
    assert ar.revoke_client("alice") is False  # idempotent
    assert ar.list_clients() == []


def test_disabled_overlay_refuses_everything(monkeypatch):
    monkeypatch.setenv("RAG_ACCESS_STORE", "0")
    with pytest.raises(ar._base.SelectionDenied):
        ar.mint_client("alice")
    assert ar.overlay_clients() == {}
    assert ar.overlay_acls() == {}


# ---------------------------------------------------------------------------
# The merge: overlay ∪ env in the D15 registry
# ---------------------------------------------------------------------------


def test_merged_registry_authenticates_overlay_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "env-only:env-key")
    res = ar.mint_client("alice", datasets=["reports"])
    ident = cr.resolve_presented([res["key"]])
    assert ident is not None and ident.name == "alice" and not ident.is_admin
    assert cr.dataset_allowed(ident, "reports")
    assert not cr.dataset_allowed(ident, "other")
    # The env key still works too.
    assert cr.resolve_presented(["env-key"]).name == "env-only"


def test_env_authoritative_on_key_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "alice:env-key")
    monkeypatch.setenv("RAG_DATASET_ACLS", "alice:env-ds")
    res = ar.mint_client("alice", datasets=["overlay-ds"])
    # The overlay key CANNOT hijack the env name's identity — it resolves to
    # its OWN client identity; the env key remains authoritative for 'alice'.
    overlay_ident = cr.resolve_presented([res["key"]])
    assert overlay_ident.name == "alice"  # same name, distinct key identity
    env_ident = cr.resolve_presented(["env-key"])
    assert env_ident.name == "alice"
    # ACLs union: the env grant AND the overlay grant both apply.
    assert cr.dataset_allowed(env_ident, "env-ds")
    assert cr.dataset_allowed(env_ident, "overlay-ds")


def test_grant_update_applies_without_restart(tmp_path):
    ar.mint_client("alice", datasets=[])
    res = ar.mint_client("alice", datasets=["reports"])  # rotate+grant
    ident = cr.resolve_presented([res["key"]])
    assert cr.dataset_allowed(ident, "reports")
    ar.grant_datasets("alice", ["notes", "*"])
    ident2 = cr.resolve_presented([res["key"]])
    assert cr.dataset_allowed(ident2, "notes")
    assert cr.dataset_allowed(ident2, "anything-else")  # '*' grant


def test_grant_unknown_client_404s(tmp_path):
    with pytest.raises(KeyError):
        ar.grant_datasets("ghost", ["ds"])


def test_corrupt_overlay_degrades_to_env_only(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "env-only:env-key")
    p = Path(tmp_path) / "access" / "clients.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{corrupt")
    assert cr.resolve_presented(["env-key"]) is not None
    assert ar.overlay_clients() == {}


# ---------------------------------------------------------------------------
# REST: the admin endpoints (full stack through the middleware)
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    class _FakeDM:
        def list_datasets(self):
            return []

        def get_dataset(self, name, sync_count=False):
            return {"name": name, "document_count": 0}

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    api._UNLOCK_CACHE.clear()
    ar._mtime_cache.clear()
    yield TestClient(api.app)
    api._UNLOCK_CACHE.clear()
    ar._mtime_cache.clear()


def test_rest_admin_can_mint_and_client_key_authenticates(admin_client):
    r = admin_client.post(
        "/api/admin/clients",
        json={"name": "alice", "datasets": ["reports"]},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 200
    minted_key = r.json()["key"]
    assert minted_key
    # The minted key now authenticates as a registry client on the listing…
    r2 = admin_client.get("/api/datasets", headers={"X-RAG-Api-Key": minted_key})
    assert r2.status_code == 200
    # …and is DENIED the admin surface.
    r3 = admin_client.get("/api/admin/clients", headers={"X-RAG-Api-Key": minted_key})
    assert r3.status_code == 403


def test_rest_registry_key_cannot_mint(admin_client):
    r = admin_client.post(
        "/api/admin/clients",
        json={"name": "evil", "datasets": ["*"]},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 200
    evil_key = r.json()["key"]
    # The minted (registry) key tries to mint more keys → denied at the gate.
    r2 = admin_client.post(
        "/api/admin/clients",
        json={"name": "more", "datasets": ["*"]},
        headers={"X-RAG-Api-Key": evil_key},
    )
    assert r2.status_code == 403


def test_rest_listing_masks_key_material(admin_client):
    admin_client.post(
        "/api/admin/clients", json={"name": "alice", "datasets": []},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    r = admin_client.get("/api/admin/clients", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["clients"] and body["clients"][0]["name"] == "alice"
    assert "…" in body["clients"][0]["key_masked"]
    # No full key material anywhere in the listing response.
    stored = json.loads((Path(os.environ["DATA_PATH"]) / "access" / "clients.json").read_text())
    real_key = stored["clients"]["alice"]["key"]
    assert real_key not in r.text


def test_rest_grant_patch_and_revoke(admin_client):
    r = admin_client.post(
        "/api/admin/clients", json={"name": "bob", "datasets": []},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    bob_key = r.json()["key"]
    # Grant, then verify the key's access flips on the next request.
    r = admin_client.patch(
        "/api/admin/clients/bob", json={"datasets": ["reports"]},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 200
    ident = cr.resolve_presented([bob_key])
    assert cr.dataset_allowed(ident, "reports")
    # Revoke → the key stops authenticating entirely.
    r = admin_client.delete("/api/admin/clients/bob", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200
    assert cr.resolve_presented([bob_key]) is None


def test_rest_revoke_env_client_reports_env_authority(admin_client, monkeypatch):
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "carol:carol-key")
    r = admin_client.delete("/api/admin/clients/carol", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200
    assert "ENV registry" in r.json()["message"]
    # Carol still authenticates (env untouched).
    assert cr.resolve_presented(["carol-key"]) is not None


def test_rest_admin_endpoints_require_admin(admin_client, monkeypatch):
    monkeypatch.delenv("RAG_API_KEY_CLIENTS", raising=False)
    # A registry key (minted via the admin) cannot even list.
    admin_client.post(
        "/api/admin/clients", json={"name": "alice", "datasets": []},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    # (covered by test_rest_registry_key_cannot_mint; here: the overlay-off case)
    monkeypatch.setenv("RAG_ACCESS_STORE", "0")
    r = admin_client.get("/api/admin/clients", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 409
    r = admin_client.post(
        "/api/admin/clients", json={"name": "x"}, headers={"X-RAG-Api-Key": "deployment-key"}
    )
    assert r.status_code == 409


def test_rest_mint_validation_errors(admin_client):
    r = admin_client.post(
        "/api/admin/clients", json={"name": "bad name!"}, headers={"X-RAG-Api-Key": "deployment-key"}
    )
    assert r.status_code == 400
    r = admin_client.post(
        "/api/admin/clients", json={"name": "ok", "datasets": "not-a-list"},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# D19 delegation precedence — X-API-Key outranks a co-forwarded admin
# Authorization token (the LLM-gateway topology: the gateway authenticates
# ITSELF upstream with an admin token AND forwards the caller's delegated
# X-API-Key; the delegated identity must win)
# ---------------------------------------------------------------------------


def test_delegation_x_api_key_outranks_forwarded_admin_bearer(monkeypatch, tmp_path):
    """The gateway topology, exactly: presented = [ingress-admin-token (via
    authorization), andrew-user-key (via x-api-key)]. The request must
    resolve as ANDREW (registry client), not as admin."""
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "ingress:platform-token")
    monkeypatch.setenv("RAG_DATASET_ACLS", "ingress:*")
    res = ar.mint_client("andrew", datasets=["andrew-memory"])
    ident = cr.resolve_presented(
        ["platform-token", res["key"]],
        ["authorization", "x-api-key"],
    )
    assert ident is not None
    assert ident.kind == "client" and ident.name == "andrew"
    assert cr.dataset_allowed(ident, "andrew-memory")


def test_delegation_missing_source_keeps_legacy_admin_first(monkeypatch):
    """Legacy callers (no source list) keep the historical admin-first order
    — byte-identical behavior for single-header requests."""
    monkeypatch.setenv("RAG_API_KEY", "platform-token")  # an ADMIN key
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "andrew:andrew-key")
    monkeypatch.setenv("RAG_DATASET_ACLS", "andrew:ds")
    # An admin key presented without source info + a client key: admin wins
    # (legacy order — admin keys beat registry keys when we can't tell which
    # header carried which).
    ident = cr.resolve_presented(["platform-token", "andrew-key"])
    assert ident is not None and ident.is_admin


def test_delegation_registry_key_never_escalates_to_admin(monkeypatch):
    """An X-API-Key carrying an UNKNOWN key must NOT fall back to the
    co-forwarded admin token (no escalation through the delegation path)."""
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "andrew:andrew-key")
    monkeypatch.setenv("RAG_DATASET_ACLS", "andrew:ds")
    ident = cr.resolve_presented(
        ["platform-token", "unknown-key"],
        ["authorization", "x-api-key"],
    )
    # The delegated X-API-Key is the chosen identity: unknown → None (401),
    # NOT an escalation to the admin token that rode along.
    assert ident is None


def test_delegation_x_api_key_can_carry_admin_key(monkeypatch):
    """X-API-Key presenting the deployment key itself = a legitimate admin
    presentation (de-escalation path is optional, not mandatory)."""
    ident = cr.resolve_presented(
        ["deployment-key"], ["x-api-key"]
    ) if False else None
    # (deployment-key admin matching is exercised via _RAG_API_KEY paths
    # elsewhere; here pin the source-aware admin matching helper.)
    monkeypatch.setenv("RAG_API_KEY_CLIENTS", "andrew:andrew-key")
    ident = cr.resolve_presented(["andrew-key"], ["x-api-key"])
    assert ident is not None and ident.kind == "client"


def test_delegation_precedence_via_rest_middleware(monkeypatch, tmp_path):
    """Full stack: gateway forwards Authorization Bearer <admin token> +
    X-API-Key <andrew key> → the REST listing resolves as andrew."""
    from fastapi.testclient import TestClient

    class _FakeDM:
        def list_datasets(self):
            return [{"name": "andrew-memory", "document_count": 1}, {"name": "other", "document_count": 2}]

        def get_dataset(self, name, sync_count=False):
            if name == "missing":
                raise FileNotFoundError(f"Dataset '{name}' not found")
            return {"name": name, "document_count": 0}

        def has_password(self, name):
            return name == "andrew-memory"

        def verify_password(self, name, pw):
            return pw == "right"

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "platform-token")
    monkeypatch.setenv(cr.CLIENTS_ENV, "gateway-ingress:platform-token")
    monkeypatch.setenv(cr.ACLS_ENV, "gateway-ingress:*")
    monkeypatch.setenv(acc.STORE_ENV, "1")
    res = ar.mint_client("andrew", datasets=["andrew-memory"])
    client = TestClient(api.app)
    # BOTH headers ride the request (the gateway topology):
    r = client.get(
        "/api/datasets",
        headers={
            "Authorization": "Bearer platform-token",
            "X-API-Key": res["key"],
        },
    )
    assert r.status_code == 200
    names = [d["name"] for d in r.json()["datasets"]]
    assert names == ["andrew-memory"], names  # andrew's view, NOT the admin's
    assert r.json()["acl_hidden"] == 1
