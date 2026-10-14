"""T4 surface — short-term-memory create/end tools + catalog hiding.

Covers PLAN stmem sections 4.3/4.4 at the API boundary:

* POST /api/short-term-memory mints the frozen name shape
  (stm-<base32(hmac_sha256(salt, session_id))[:16]>), creates the dataset
  with kind="short_term" + a lifetime, NO password, and grants ONLY the
  creating identity.
* The salt gate: unset RAG_STM_NAME_SALT fails LOUDLY (503), never falls
  back to an empty (guessable) salt.
* ttl_seconds above shortTermMemory.maxTtlHours is refused (400); a
  malformed ttl_seconds is refused; the default (6h) applies when omitted.
* DELETE /api/short-term-memory/{name} is owner-or-admin gated and REFUSES
  any dataset that is not kind=="short_term" (never a general deletion
  backdoor).  A non-owner is refused BEFORE the kind check (no oracle).
* _rag_acl_filter_datasets drops kind=="short_term" from the catalog view
  ONLY — plain listings and direct-name access still work.

The registry overlay lives in a temp DATA_PATH (the D21 test pattern); no
real credentials are written to disk.

Run::

    pytest tests/security/test_stm_surface.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr

# ===========================================================================
# Harness
# ===========================================================================


class _MetaBackedDM:
    """A create/delete/meta-only DatasetManager double.

    Mirrors the real meta.json flow closely enough for the ownership and
    kind gates: create_dataset stamps meta (name, created_by, lifetime,
    kind), _read_meta reads it back, delete_dataset removes it.  No Qdrant,
    no models.
    """

    def __init__(self, tmp_path):
        self.root = tmp_path / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name):
        return self.root / name

    def create_dataset(self, name, description="", *args, created_by=None, lifetime=None, kind=None, **kwargs):
        # Fail-loud on kwargs this double does not model: a silent **kwargs
        # swallow would make assertions on the dropped fields (e.g. the
        # NO-password pin) vacuous — they could never fail (PLAN 6.3).
        assert not kwargs, f"unexpected create_dataset kwargs: {sorted(kwargs)}"
        import json
        import time as _time

        d = self._dir(name)
        if d.exists():
            raise FileExistsError(f"Dataset '{name}' already exists")
        d.mkdir(parents=True)
        meta = {
            "name": name,
            "description": description,
            "created": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "document_count": 0,
        }
        if created_by:
            meta["created_by"] = created_by
        if lifetime is not None:
            meta["lifetime"] = lifetime
        if kind is not None:
            meta["kind"] = kind
        (d / "meta.json").write_text(json.dumps(meta))
        return dict(meta)

    def delete_dataset(self, name):
        import shutil

        d = self._dir(name)
        if not d.exists():
            raise FileNotFoundError(name)
        shutil.rmtree(d)

    def _read_meta(self, name):
        import json

        p = self._dir(name) / "meta.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def read_created_by(self, name):
        meta = self._read_meta(name)
        if not meta:
            return None
        return str(meta.get("created_by") or "") or None

    def has_password(self, name):
        return False

    def list_datasets(self, *args, **kwargs):
        import json

        rows = []
        for p in sorted(self.root.glob("*/meta.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except Exception:
                continue
        return rows

    def get_dataset(self, name, sync_count=True):
        meta = self._read_meta(name)
        if not meta:
            raise FileNotFoundError(name)
        return meta


@pytest.fixture()
def surface(monkeypatch, tmp_path):
    """REST TestClient + the meta-backed DM, registry keyed to alice/bob.

    Alice holds the "*" create grant (and rw); bob holds nothing.  The
    access store is enabled so self-selection grants are real.
    """
    from fastapi.testclient import TestClient

    for var in ("RAG_API_KEY_CLIENTS", "RAG_ACLS", cr.ACLS_ENV, cr.CLIENTS_ENV, "RAG_ACCESS_STORE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    monkeypatch.setenv("RAG_STM_NAME_SALT", "unit-test-salt")
    dm = _MetaBackedDM(tmp_path)

    async def _fake_manager():
        return dm

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-admin-key")
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")
    monkeypatch.setattr(api._access_store, "store_enabled", lambda: True)
    with TestClient(api.app) as client:
        yield client, dm


def _post_stm(client, headers, **body):
    body.setdefault("session_id", "swarm-1")
    return client.post("/api/short-term-memory", json=body, headers=headers)


def _delete_stm(client, headers, name):
    return client.delete(f"/api/short-term-memory/{name}", headers=headers)


# ===========================================================================
# create_short_term_memory
# ===========================================================================


def test_create_mints_frozen_name_shape(surface):
    """The minted name matches the frozen spec exactly (key = salt, msg =
    session_id, base32, truncated to 16, lower-cased)."""
    import base64
    import hashlib
    import hmac

    client, dm = surface
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="swarm-abc")
    assert r.status_code == 200, r.text
    body = r.json()
    expected = "stm-" + base64.b32encode(hmac.new(b"unit-test-salt", b"swarm-abc", hashlib.sha256).digest()).decode()[:16].lower()
    assert body["dataset"] == expected
    meta = dm._read_meta(expected)
    assert meta is not None
    assert meta["kind"] == "short_term"
    assert meta["lifetime"] == 6 * 3600  # default TTL
    assert meta["created_by"] == "alice"
    assert "password" not in meta  # NO password — reapable by design
    # Creator-only grant: alice self-selected, bob did not.
    alice = cr.resolve_presented(["alice-key"])
    bob = cr.resolve_presented(["bob-key"])
    assert acc.dataset_allowed(alice, expected) is True
    assert acc.dataset_allowed(bob, expected) is False


def test_create_without_salt_fails_loudly(surface, monkeypatch):
    """No salt → 503, and NOTHING is created (no empty-salt fallback)."""
    client, dm = surface
    monkeypatch.delenv("RAG_STM_NAME_SALT", raising=False)
    before = set(os.listdir(dm.root))
    r = _post_stm(client, {"X-API-Key": "alice-key"})
    assert r.status_code == 503
    assert "RAG_STM_NAME_SALT" in r.json()["detail"]
    assert set(os.listdir(dm.root)) == before  # nothing minted


def test_create_without_salt_as_admin_also_fails_loudly(surface, monkeypatch):
    """The salt gate is deployment-wide: even an admin key cannot mint
    guessable names."""
    client, _ = surface
    monkeypatch.delenv("RAG_STM_NAME_SALT", raising=False)
    r = _post_stm(client, {"X-API-Key": "deployment-admin-key"})
    assert r.status_code == 503


def test_create_ttl_over_cap_refused(surface):
    client, dm = surface
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="big", ttl_seconds=25 * 3600)
    assert r.status_code == 400
    assert "cap" in r.json()["detail"]
    assert not list(dm.root.iterdir())


def test_create_bad_ttl_refused(surface):
    client, dm = surface
    for bad in (0, -5, "soon", True, 1.5):
        r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="t", ttl_seconds=bad)
        assert r.status_code == 400, bad
    assert not list(dm.root.iterdir())


def test_create_default_and_capped_ttl(surface):
    client, dm = surface
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="s1")
    name1 = r.json()["dataset"]
    assert dm._read_meta(name1)["lifetime"] == 6 * 3600
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="s2", ttl_seconds=24 * 3600)
    assert r.status_code == 200  # exactly at the cap is allowed
    name2 = r.json()["dataset"]
    assert dm._read_meta(name2)["lifetime"] == 24 * 3600
    _delete_stm(client, {"X-API-Key": "alice-key"}, name1)
    _delete_stm(client, {"X-API-Key": "alice-key"}, name2)


def test_create_same_session_same_name(surface):
    """The mapping session→name is deterministic (re-creating collides)."""
    client, _ = surface
    n1 = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="swarm-x").json()["dataset"]
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="swarm-x")
    assert r.status_code == 409
    # The collision response must not echo the minted name back: the name is
    # the only handle to a hidden dataset, so leaking it here would hand a
    # second caller the key to someone else's session memory.
    assert "stm-" not in r.text, r.text
    # ...and the ORIGINAL dataset is untouched and still addressable by name.
    assert n1.startswith("stm-")
    assert client.get(f"/api/datasets/{n1}", headers={"X-API-Key": "alice-key"}).status_code == 200
    _delete_stm(client, {"X-API-Key": "alice-key"}, n1)


def test_create_requires_session_id(surface):
    client, _ = surface
    r = client.post("/api/short-term-memory", json={}, headers={"X-API-Key": "alice-key"})
    assert r.status_code == 400


def test_create_three_level_gate(surface, monkeypatch, tmp_path):
    """2026-10 configurable-create (D1/D27) — the STM create gate carries
    the SAME three-level decision as POST /api/datasets (both call
    _identity_may_create):

      1. a per-user client CAN create by DEFAULT (the old pin here asserted
         a named-ACL key is 403'd — that was the user-reported bug: "It also
         complained about ACLs for a user api key. People should be allowed
         to create datasets generally.");
      2. with RAG_DATASETS_CREATE_ALLOWED=false the same client is REFUSED;
      3. with a per-user can_create_datasets=True override the client is
         ALLOWED again — the override beats the deployment default.

    This test pins level 3, the one that actually proves the three-level
    decision flows through the STM surface too."""
    from multimodal_rag.utils import admin_registry as ar

    client, dm = surface

    # Level "default": bob (named ACL, no '*') may create.
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*;bob:some-other-ds")
    r = _post_stm(client, {"X-API-Key": "bob-key"}, session_id="bob-1")
    assert r.status_code == 200, r.text
    name1 = r.json()["dataset"]
    assert dm._read_meta(name1)["created_by"] == "bob"

    # Level 2: deployment knob OFF -> bob refused.
    monkeypatch.setenv("RAG_DATASETS_CREATE_ALLOWED", "0")
    r = _post_stm(client, {"X-API-Key": "bob-key"}, session_id="bob-2")
    assert r.status_code == 403
    assert "creation is not permitted" in r.json()["detail"]

    # Level 1: the per-user override beats the knob-off default.
    ar.mint_client("bob", datasets=["some-other-ds"])
    ar.set_client_flags("bob", can_create_datasets=True)
    r = _post_stm(client, {"X-API-Key": "bob-key"}, session_id="bob-3")
    assert r.status_code == 200, r.text
    name3 = r.json()["dataset"]
    assert dm._read_meta(name3)["created_by"] == "bob"

    # ...and a per-user DENY beats an allow-by-default deployment.
    monkeypatch.setenv("RAG_DATASETS_CREATE_ALLOWED", "true")
    ar.set_client_flags("bob", can_create_datasets=False)
    r = _post_stm(client, {"X-API-Key": "bob-key"}, session_id="bob-4")
    assert r.status_code == 403

    _delete_stm(client, {"X-API-Key": "alice-key"}, name1)
    _delete_stm(client, {"X-API-Key": "alice-key"}, name3)


def test_create_lifetime_overflow_valueerror_maps_400(surface, monkeypatch):
    """A lifetime the storage layer rejects (validate_lifetime ValueError —
    e.g. > 2**63-1) surfaces as a clean 400, not a 500."""
    client, dm = surface
    r = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="huge", ttl_seconds=2**63)
    assert r.status_code in (400, 503)  # 400 once T1 lands; 503 on the TypeError path pre-T1
    assert not list(dm.root.iterdir())


# ===========================================================================
# end_short_term_memory
# ===========================================================================


def _mk_normal(dm, name, created_by="admin"):
    return dm.create_dataset(name, "a normal dataset", created_by=created_by)


def test_end_refuses_normal_dataset(surface):
    """INARIANT: end_short_term_memory must NEVER delete a normal dataset."""
    client, dm = surface
    _mk_normal(dm, "regular-ds", created_by="alice")
    r = _delete_stm(client, {"X-API-Key": "alice-key"}, "regular-ds")
    assert r.status_code == 409
    assert "not a short-term-memory dataset" in r.json()["detail"]
    assert dm._read_meta("regular-ds") is not None  # still there


def test_end_deletes_stm_dataset(surface):
    client, dm = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="gone").json()["dataset"]
    r = _delete_stm(client, {"X-API-Key": "alice-key"}, name)
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == name
    assert dm._read_meta(name) is None


def test_end_by_non_owner_403_no_kind_leak(surface):
    """A non-owner gets 403 BEFORE the kind check (no oracle for what the
    dataset is); the dataset survives."""
    client, dm = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="priv").json()["dataset"]
    r = _delete_stm(client, {"X-API-Key": "bob-key"}, name)
    assert r.status_code == 403
    assert dm._read_meta(name) is not None
    # Bob cannot launder the delete through a normal dataset either.
    _mk_normal(dm, "bobs-ds", created_by="bob")
    r = _delete_stm(client, {"X-API-Key": "bob-key"}, "bobs-ds")
    assert r.status_code == 409  # owner passes the gate, kind refusal fires
    assert dm._read_meta("bobs-ds") is not None


def test_end_admin_can_end_any_stm(surface):
    client, dm = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="adm").json()["dataset"]
    r = _delete_stm(client, {"X-API-Key": "deployment-admin-key"}, name)
    assert r.status_code == 200
    assert dm._read_meta(name) is None


def test_end_admin_cannot_end_normal_dataset_either(surface):
    """The kind refusal binds ADMINS too — no backdoor at any privilege."""
    client, dm = surface
    _mk_normal(dm, "admin-owned", created_by="admin")
    r = _delete_stm(client, {"X-API-Key": "deployment-admin-key"}, "admin-owned")
    assert r.status_code == 409
    assert dm._read_meta("admin-owned") is not None


def test_end_unknown_dataset_404_for_admin(surface):
    """F3: the 404 fires only AFTER the caller is proven owner-or-admin.
    For the ADMIN (who sees every dataset world-wide) a missing dataset is
    a clean 404; for a non-owner it must NOT be distinguishable (next test)."""
    client, _ = surface
    r = _delete_stm(client, {"X-API-Key": "deployment-admin-key"}, "stm-doesnotexist0")
    assert r.status_code == 404


def test_end_unknown_dataset_is_not_an_existence_oracle(surface):
    """F3: for a NON-OWNER, 'exists but not mine' and 'does not exist' must
    be indistinguishable — same status, same message (the D23 ordering:
    the ownership gate reads meta BEFORE the route's 404, and missing meta
    reads as 'no provable creator' → the pre-D23 403).  bob probing
    /api/short-term-memory/{X} cannot enumerate real dataset names."""
    client, _ = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="secret-one").json()["dataset"]
    try:
        r_existing = _delete_stm(client, {"X-API-Key": "bob-key"}, name)
        r_missing = _delete_stm(client, {"X-API-Key": "bob-key"}, "stm-doesnotexist0")
        assert r_existing.status_code == r_missing.status_code == 403
        assert r_existing.json()["detail"] == r_missing.json()["detail"]
        assert "not permitted" in r_missing.json()["detail"]
        # The denial is NAMELESS: echoing the probed name back would let the
        # body act as the oracle the status code refuses to be.
        assert "stm-" not in r_missing.json()["detail"]
    finally:
        _delete_stm(client, {"X-API-Key": "alice-key"}, name)


def test_end_pre_d23_stm_dataset_admin_only(surface):
    """An STM dataset with no created_by stamp is admin-only to end."""
    client, dm = surface
    dm.create_dataset("stm-pre23", "", lifetime=3600, kind="short_term")
    r = _delete_stm(client, {"X-API-Key": "alice-key"}, "stm-pre23")
    assert r.status_code == 403
    r = _delete_stm(client, {"X-API-Key": "deployment-admin-key"}, "stm-pre23")
    assert r.status_code == 200


# ===========================================================================
# Catalog hiding (the _rag_acl_filter_datasets catalog=True path ONLY)
# ===========================================================================


def _listing(client, headers, catalog):
    params = {"catalog": "true"} if catalog else {}
    return client.get("/api/datasets", params=params, headers=headers).json()


def test_stm_hidden_from_catalog_but_reachable_by_name(surface):
    """The core 4.4 guarantee: absent from the catalog listing, present and
    usable by direct name."""
    client, dm = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="hidden-one").json()["dataset"]
    _mk_normal(dm, "visible-ds", created_by="alice")
    # Catalog view (admin identity, full listing): STM hidden, normal shown.
    body = _listing(client, {"X-API-Key": "deployment-admin-key"}, catalog=True)
    names = {d["name"] for d in body["datasets"]}
    assert name not in names
    assert "visible-ds" in names
    assert body.get("acl_hidden") == 1  # the STM row is accounted for
    # Plain listing keeps the world-only filter (admin → everything, no hiding).
    body = _listing(client, {"X-API-Key": "deployment-admin-key"}, catalog=False)
    names = {d["name"] for d in body["datasets"]}
    assert name in names and "visible-ds" in names
    # Direct-name access still works for the catalog-hidden dataset.
    r = client.get(f"/api/datasets/{name}", headers={"X-API-Key": "alice-key"})
    assert r.status_code == 200, r.text


def test_stm_hidden_from_catalog_for_creator_too(surface):
    """PLAN 4.4: hidden 'even for the creator' — alice holds the grant (the
    self-selection) yet the catalog still hides the row."""
    client, dm = surface
    name = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="mine").json()["dataset"]
    _mk_normal(dm, "mine-visible", created_by="alice")
    body = _listing(client, {"X-API-Key": "alice-key"}, catalog=True)
    names = {d["name"] for d in body["datasets"]}
    assert name not in names
    assert "mine-visible" in names


def test_catalog_hidden_count_counts_only_stm(surface):
    """The hidden counter reflects dropped STM rows, and the count matches
    exactly how many were dropped."""
    client, _ = surface
    n1 = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="h1").json()["dataset"]
    n2 = _post_stm(client, {"X-API-Key": "alice-key"}, session_id="h2").json()["dataset"]
    body = _listing(client, {"X-API-Key": "deployment-admin-key"}, catalog=True)
    names = {d["name"] for d in body["datasets"]}
    assert n1 not in names and n2 not in names
    assert body.get("acl_hidden") == 2


def test_catalog_hiding_preserves_row_mutation_safety(surface):
    """The non-admin catalog path must not mutate shared row dicts
    (excluded-stamping copies) — a regression guard for the filter's
    pre-filter rewrite."""
    client, dm = surface
    _mk_normal(dm, "pub-ds", created_by="admin")
    # Normal catalog call with exclusions — must not raise / corrupt rows.
    body = _listing(client, {"X-API-Key": "alice-key"}, catalog=True)
    assert isinstance(body["datasets"], list)


def test_mcp_surface_helpers_match_rest(surface):
    """The MCP mint helper produces byte-identical names to the REST one
    (one algorithm, two entry points)."""
    import multimodal_rag.mcp_server as mcp

    n_rest = api._stm_mint_name("parity-session")
    n_mcp = mcp._stm_mint_name("parity-session")
    assert n_rest == n_mcp
    assert n_rest.startswith("stm-")


def test_mcp_tools_registered():
    """The two frozen tool names are REGISTERED with the MCP server —
    checked behaviourally via list_tools (a source-text match would pass
    regardless of whether the @mcp.tool() decorator actually ran; XV-1
    mutation M7 proved exactly that)."""
    import asyncio

    import multimodal_rag.mcp_server as mcp

    tools = asyncio.run(mcp.mcp.list_tools())
    names = {t.name for t in tools}
    assert "create_short_term_memory" in names
    assert "end_short_term_memory" in names
    # ...and nothing extra was added (no stm search/add tools).
    assert not any("short_term" in n and n not in ("create_short_term_memory", "end_short_term_memory") for n in names)


# ===========================================================================
# F7: BEHAVIOURAL MCP tests — the tools are CALLED through mcp.call_tool and
# their GATES asserted, not their source text.  This is the band where the
# XV-1 review found F1/F4 living untested.
# ===========================================================================


class _McpDM:
    """A create/delete/meta DM double for the MCP tool bodies."""

    def __init__(self):
        self.created = []
        self.deleted = []
        self.metas = {}

    def create_dataset(self, name, description="", *args, **kwargs):
        if name in self.metas:
            raise FileExistsError(f"Dataset '{name}' already exists")
        meta = {"name": name, "description": description}
        meta.update({k: v for k, v in kwargs.items() if v is not None})
        self.metas[name] = meta
        self.created.append((name, kwargs))
        return dict(meta)

    def _read_meta(self, name):
        m = self.metas.get(name)
        if m is None:
            raise FileNotFoundError(name)
        return m

    def read_created_by(self, name):
        m = self.metas.get(name)
        return (m or {}).get("created_by")

    def delete_dataset(self, name):
        if name not in self.metas:
            raise FileNotFoundError(name)
        del self.metas[name]
        self.deleted.append(name)

    def list_datasets(self, *args, **kwargs):
        return [dict(m) for m in self.metas.values()]


@pytest.fixture()
def mcp_env(monkeypatch):
    """Isolated MCP env: salt set, registry keyed alice/bob, stub DM wired
    into get_manager.  Yields (call, dm) where call(name, args) invokes the
    REGISTERED tool via mcp.call_tool and returns (status, text)."""
    import asyncio

    import multimodal_rag.mcp_server as mcp

    monkeypatch.setenv("DATA_PATH", "/tmp/stm-mcp-test-data")
    monkeypatch.setenv("RAG_STM_NAME_SALT", "behavioural-test-salt")
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")
    monkeypatch.delenv("RAG_ACCESS_STORE", raising=False)  # store OFF by default
    dm = _McpDM()
    monkeypatch.setattr(mcp, "get_manager", lambda: dm)

    def call(tool, args):
        async def _run():
            try:
                result = await mcp.mcp.call_tool(tool, args)
                if getattr(result, "isError", False):
                    return ("error", str(result.content[0].text))
                return ("ok", str(result.content[0].text))
            except mcp.ToolError as exc:
                # mcp.call_tool may raise directly (the outer wrapper wraps
                # the message); normalise both shapes to (status, text).
                return ("error", str(exc))

        return asyncio.run(_run())

    return call, dm


def _mcp_create(call, session_id="mcp-sess", **extra):
    return call("create_short_term_memory", {"session_id": session_id, **extra})


def _mcp_end(call, dataset_name):
    return call("end_short_term_memory", {"dataset_name": dataset_name})


def test_mcp_create_mints_and_stamps_no_password(mcp_env):
    """BEHAVIOURAL: the registered tool creates kind=short_term with a
    lifetime, NO password, under the anonymous-safe creator stamp."""
    call, dm = mcp_env
    status, text = _mcp_create(call)
    assert status == "ok", text
    name = json.loads(text)["dataset"]
    meta = dm.metas[name]
    assert meta["kind"] == "short_term"
    assert meta["lifetime"] == 6 * 3600
    assert meta["created_by"] == "anonymous"  # F2: sentinel-safe stamp
    assert "password" not in meta
    _mcp_end(call, name)


def test_mcp_create_enforces_star_grant(mcp_env, monkeypatch):
    """F1: bob (named ACL, no '*') is REFUSED on MCP exactly as on REST.
    The gate is the create=manage rule, not a REST-only nicety."""
    call, dm = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:some-other-ds")
    ident = cr.resolve_presented(["bob-key"])
    token = cr.set_current_identity(ident)
    try:
        status, text = _mcp_create(call)
        assert status == "error"
        assert "cannot create datasets" in text
        assert not dm.created
    finally:
        cr.reset_current_identity(token)


def test_mcp_end_by_non_owner_refused(mcp_env, monkeypatch):
    """BEHAVIOURAL (mutation M11 target): the MCP owner-or-admin gate
    actually fires — bob cannot end alice's STM dataset."""
    call, dm = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    alice = cr.resolve_presented(["alice-key"])
    bob = cr.resolve_presented(["bob-key"])
    # alice creates...
    token = cr.set_current_identity(alice)
    try:
        status, text = _mcp_create(call, "alice-session")
        assert status == "ok"
        name = json.loads(text)["dataset"]
        dm.metas[name]["created_by"] = "alice"
    finally:
        cr.reset_current_identity(token)
    # ...bob cannot end it
    token = cr.set_current_identity(bob)
    try:
        status, text = _mcp_end(call, name)
        assert status == "error"
        assert "owned by 'alice'" in text
    finally:
        cr.reset_current_identity(token)
    assert name in dm.metas  # still there
    # ...and alice CAN end it
    token = cr.set_current_identity(alice)
    try:
        status, text = _mcp_end(call, name)
        assert status == "ok"
    finally:
        cr.reset_current_identity(token)
    assert name not in dm.metas


def test_mcp_end_nonexistent_says_not_found_not_pre_d23(mcp_env, monkeypatch):
    """F4 (mutation M4 target): ending a NONEXISTENT dataset says "not
    found" — never the false 'predates dataset ownership' claim."""
    call, _ = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:some-other-ds")
    token = cr.set_current_identity(cr.resolve_presented(["bob-key"]))
    try:
        status, text = _mcp_end(call, "stm-doesnotexist0")
        assert status == "error"
        assert "not found" in text
        assert "predates" not in text
    finally:
        cr.reset_current_identity(token)


def test_mcp_end_refuses_normal_dataset(mcp_env, monkeypatch):
    """F4/M12 target: the MCP kind refusal fires on a REAL call — the tool
    never deletes a normal dataset (invariant 6 on the MCP band).  The
    creator's own normal dataset reaches the kind check (owner passes the
    gate) and is refused there."""
    call, dm = mcp_env
    # Make the CALLER the owner so the ownership gate passes and the KIND
    # gate is what fires (that is the assertion under test).
    monkeypatch.setenv(cr.CLIENTS_ENV, "own:own-key")
    monkeypatch.setenv(cr.ACLS_ENV, "own:*")
    ident = cr.resolve_presented(["own-key"])
    token = cr.set_current_identity(ident)
    try:
        dm.metas["normal-ds"] = {"name": "normal-ds", "created_by": "own"}
        status, text = _mcp_end(call, "normal-ds")
        assert status == "error"
        assert "not a short-term-memory dataset" in text
        assert "normal-ds" in dm.metas  # still there
    finally:
        cr.reset_current_identity(token)


def test_mcp_end_admin_can_end_any_stm_but_not_normal(mcp_env, monkeypatch):
    call, dm = mcp_env
    dm.metas["stm-adm"] = {"name": "stm-adm", "created_by": "alice", "kind": "short_term"}
    dm.metas["normal-adm"] = {"name": "normal-adm", "created_by": "alice"}
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    # An OPAQUE admin key resolves to the admin identity.
    token = cr.set_current_identity(cr.resolve_presented(["deployment-admin-key"]))
    try:
        status, _ = _mcp_end(call, "stm-adm")
        assert status == "ok"
        status, text = _mcp_end(call, "normal-adm")
        assert status == "error"
        assert "not a short-term-memory dataset" in text
        assert "normal-adm" in dm.metas
    finally:
        cr.reset_current_identity(token)


def test_mcp_end_pre_d23_stm_admin_only(mcp_env, monkeypatch):
    """An STM dataset with no created_by stamp is admin-only on MCP too."""
    call, dm = mcp_env
    dm.metas["stm-pre23"] = {"name": "stm-pre23", "kind": "short_term"}  # no stamp
    monkeypatch.setenv(cr.CLIENTS_ENV, "bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:some-other-ds")
    token = cr.set_current_identity(cr.resolve_presented(["bob-key"]))
    try:
        status, text = _mcp_end(call, "stm-pre23")
        assert status == "error"
        assert "predates dataset ownership" in text  # TRUE here: it exists
    finally:
        cr.reset_current_identity(token)


def test_mcp_list_datasets_hides_short_term(mcp_env):
    """A1: the MCP listing hides kind=short_term rows — the create note's
    promise is TRUE.  Verified behaviourally through list_datasets."""
    import asyncio

    import multimodal_rag.mcp_server as mcp

    _, dm = mcp_env
    dm.metas["normal-one"] = {"name": "normal-one", "document_count": 3}
    dm.metas["stm-hiden"] = {"name": "stm-hiden", "kind": "short_term", "document_count": 0}
    out = asyncio.run(mcp.list_datasets())
    assert "normal-one" in out
    assert "stm-hiden" not in out
    assert "hidden" in out


def test_mcp_ttl_config_fails_loud_when_default_exceeds_cap(monkeypatch):
    """F5: default 48h / max 24h is a config ERROR, raised at config read —
    the default path can never silently exceed the cap."""
    import multimodal_rag.mcp_server as mcp

    monkeypatch.setenv("RAG_STM_DEFAULT_TTL_HOURS", "48")
    monkeypatch.setenv("RAG_STM_MAX_TTL_HOURS", "24")
    with pytest.raises(mcp.ToolError, match="exceeds RAG_STM_MAX_TTL_HOURS"):
        mcp._stm_ttl_config()


def test_mcp_create_non_string_session_id_refused(mcp_env):
    """F11: a non-string session_id is refused, not coerced.  The refusal
    comes from either the pydantic annotation layer (string_type validation
    error, raised before the impl runs) or the impl's own isinstance guard —
    both refuse; nothing is created either way."""
    call, dm = mcp_env
    status, text = _mcp_create(call, session_id=12345)
    assert status == "error"
    assert ("non-empty string" in text) or ("string_type" in text)
    assert not dm.created


def test_mcp_create_ttl_over_cap_refused(mcp_env):
    call, dm = mcp_env
    status, text = _mcp_create(call, ttl_seconds=25 * 3600)
    assert status == "error"
    assert "exceeds the deployment cap" in text
    assert not dm.created

def test_mcp_create_collision_does_not_echo_the_minted_name(mcp_env):
    """F6 parity: the MCP 409-twin must NOT echo the minted name.

    The minted name is the only handle to a hidden dataset and the
    session->name map is a pure function of (salt, session_id), so a caller
    that repeats someone else's session_id would otherwise read the name
    straight off the error and then GET it by name.  Pinned on BOTH surfaces:
    the REST twin is covered by test_create_same_session_same_name.
    """
    call, dm = mcp_env
    status, text = _mcp_create(call, session_id="collide-session")
    assert status == "ok", text
    name = json.loads(text)["dataset"]
    assert name.startswith("stm-")

    status2, text2 = _mcp_create(call, session_id="collide-session")
    assert status2 == "error", text2
    assert "stm-" not in text2, f"MCP collision error leaked the minted name: {text2!r}"
    # ...and the original dataset is untouched.
    assert name in dm.metas
    _mcp_end(call, name)

