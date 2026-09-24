"""Decision D15 (Wave-5 F6): multi-user API keys → dataset ACLs — OPT-IN matrix.

Pinned semantics:

* ``RAG_API_KEY_CLIENTS`` ("name:key;name:key", parsed like K8S_MCP_CLIENTS)
  mints per-key identities for BOTH the MCP and REST surfaces.
* ``RAG_DATASET_ACLS`` ("name:ds1,ds2;name2:*") binds each name to its
  datasets; a registry key matching NO ACL gets NO datasets (fail-closed).
* The plain deployment keys — RAG_API_KEY plus the MCP key set
  (MCP_API_KEYS / RAG_API_KEYS) — keep FULL (admin) access.
* DEFAULT (no registry configured): today's single-key behaviour,
  byte-identical — the middleware delegates, no identity is bound, the REST
  single-key check is unchanged.
* Per-key throttle buckets ride the D10 per-identity machinery: the resolved
  ``key:<name>`` client id scopes the unlock cache and the password-failure
  throttle per key.

Run::

    pytest tests/security/test_dataset_acls.py -q
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp
from multimodal_rag.utils import clients_registry as cr

# ===========================================================================
# 1 · Parsing (K8S_MCP_CLIENTS field rules)
# ===========================================================================


def test_parse_clients_valid_and_shapes():
    assert cr.parse_clients("alice:k1;bob:k2") == {"k1": "alice", "k2": "bob"}
    assert cr.parse_clients("") == {}
    assert cr.parse_clients("  ") == {}
    assert cr.parse_clients("  alice : k1 ;  bob:k2 ") == {"k1": "alice", "k2": "bob"}
    assert cr.parse_clients("solo:k") == {"k": "solo"}


@pytest.mark.parametrize(
    "raw",
    [
        "nokey",  # a single field
        "a:b:c",  # three fields (no exec-patterns in RAG's registry)
        ":k",  # empty name
        "a:",  # empty key
        "a: ; :b",  # both halves empty after strip
    ],
)
def test_parse_clients_malformed_fails_loud(raw):
    with pytest.raises(cr.ClientConfigError):
        cr.parse_clients(raw)


def test_parse_clients_duplicate_key_last_wins():
    assert cr.parse_clients("alice:k1;bob:k1") == {"k1": "bob"}


def test_parse_acls_valid_and_star():
    assert cr.parse_acls("alice:ds1,ds2;bob:*") == {
        "alice": frozenset({"ds1", "ds2"}),
        "bob": frozenset({"*"}),
    }
    assert cr.parse_acls("nobody:") == {"nobody": frozenset()}  # explicit empty = nothing
    assert cr.parse_acls("") == {}


@pytest.mark.parametrize("raw", ["nodataset", ":ds", "  :  "])
def test_parse_acls_malformed_fails_loud(raw):
    with pytest.raises(cr.ClientConfigError):
        cr.parse_acls(raw)


def test_import_fails_loud_on_malformed_config():
    """The K8S-MCP convention: a typo'd registry must never silently degrade
    to 'no ACLs enforced' — importing the module with a malformed value
    raises (the servers then fail their startup)."""
    code = (
        "import sys; sys.path.insert(0, {src!r});"
        "import os; os.environ['RAG_API_KEY_CLIENTS'] = 'broken-entry';"
        "import multimodal_rag.utils.clients_registry"
    ).format(src=os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "invalid client entry" in (proc.stderr + proc.stdout)


# ===========================================================================
# 2 · Identity resolution + ACL semantics
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def test_resolve_presented_admin_over_registry(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "admin-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    ident = cr.resolve_presented(["admin-key"])
    assert ident is not None and ident.is_admin and ident.datasets is None
    # registry key → client identity with its grant
    ident = cr.resolve_presented(["alice-key"])
    assert ident == cr.Identity(kind="client", name="alice", datasets=frozenset({"ds1"}))
    assert not ident.is_admin
    # unknown / absent → None
    assert cr.resolve_presented(["wrong"]) is None
    assert cr.resolve_presented([]) is None


def test_resolve_presented_mcp_keyset_is_admin(monkeypatch):
    monkeypatch.setenv("MCP_API_KEYS", "shared-mcp-key")
    monkeypatch.setenv("RAG_API_KEYS", "rag-mcp-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    for key in ("shared-mcp-key", "rag-mcp-key"):
        ident = cr.resolve_presented([key])
        assert ident is not None and ident.is_admin, key


def test_registry_key_with_no_acl_entry_is_fail_closed(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "orphan:no-key-given")
    monkeypatch.setenv(cr.ACLS_ENV, "someoneelse:ds1")
    ident = cr.resolve_presented(["no-key-given"])
    assert ident is not None and not ident.is_admin
    assert ident.datasets == frozenset()
    assert cr.dataset_allowed(ident, "ds1") is False
    assert cr.dataset_allowed(ident, "anything") is False


def test_dataset_allowed_matrix(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:k;bob:k2;carol:k3")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1,ds2;bob:*")
    alice = cr.resolve_presented(["k"])
    bob = cr.resolve_presented(["k2"])
    carol = cr.resolve_presented(["k3"])  # named in the registry, absent from ACLs
    assert cr.dataset_allowed(alice, "ds1") and cr.dataset_allowed(alice, "ds2")
    assert not cr.dataset_allowed(alice, "ds3")
    assert cr.dataset_allowed(bob, "any-dataset")  # '*' grant
    assert not cr.dataset_allowed(carol, "ds1"), "fail-closed: no ACL entry → no datasets"
    assert cr.dataset_allowed(None, "ds1"), "no identity = D15 inactive (default UX)"
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    assert cr.dataset_allowed(admin, "anything")


def test_registry_configured_reads_env_per_request(monkeypatch):
    assert cr.registry_configured() is False
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:k")
    assert cr.registry_configured() is True


def test_dataset_name_from_path():
    assert cr.dataset_name_from_path("/api/datasets") is None
    assert cr.dataset_name_from_path("/api/datasets/") is None
    assert cr.dataset_name_from_path("/api/datasets/reports") == "reports"
    assert cr.dataset_name_from_path("/api/datasets/reports/documents") == "reports"
    assert cr.dataset_name_from_path("/api/datasets/my%20ds/files") == "my ds"
    assert cr.dataset_name_from_path("/api/search") is None
    assert cr.dataset_name_from_path("/api/admin/health") is None


def test_filter_dataset_names():
    ident = cr.Identity(kind="client", name="alice", datasets=frozenset({"b"}))
    visible, hidden = cr.filter_dataset_names(ident, ["a", "b", "c"])
    assert visible == ["b"] and hidden == 2
    visible, hidden = cr.filter_dataset_names(None, ["a", "b"])
    assert visible == ["a", "b"] and hidden == 0


# ===========================================================================
# 3 · MCP middleware: identity resolution + byte-identical delegation
# ===========================================================================


def _scope(path="/mcp", headers=None, client=("10.9.0.1", 5000)):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "client": client,
        "query_string": b"",
    }


async def _drive(middleware, scope, app_body=None):
    """Run the middleware the way production wires it
    (RagClientAuth → MemoryHeader → app); return (messages, captured).

    The identity contextvar only lives inside the request task — app_body
    (when given) runs INSIDE the wrapped app so it observes the bound
    identity exactly as an MCP tool body would.
    """
    messages = []
    captured = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
        captured["identity"] = cr.current_identity()
        captured["cid"] = mcp._unlock_client_id()
        if app_body is not None:
            result = app_body()
            if asyncio.iscoroutine(result):
                result = await result
            captured["result"] = result
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware.app = mcp._MemoryHeaderMiddleware(app)
    await middleware(scope, receive, send)
    return messages, captured


def _mw():
    return mcp._RagClientAuthMiddleware(
        app=None, env_names=mcp.AUTH_ENV_NAMES, protected=lambda p: p.startswith("/mcp")
    )


def _bearer(key: str):
    return [(b"authorization", f"Bearer {key}".encode())]


def test_middleware_registry_unset_binds_anonymous_memory_identity(monkeypatch):
    """No registry (D20, ratified 2026-09-24): the deployment is fail-closed
    EXCEPT the backwards-compatible memory path — the middleware binds an
    anonymous client identity whose ONLY grant is the MEMORY_DATASET env
    (empty when unset).  With MCP API keys configured, the parent's key
    semantics still govern (401 on a wrong key)."""
    monkeypatch.delenv("MCP_API_KEYS", raising=False)
    monkeypatch.delenv("RAG_API_KEYS", raising=False)
    monkeypatch.delenv("MEMORY_DATASET", raising=False)
    messages, captured = asyncio.run(_drive(_mw(), _scope()))
    assert messages[0]["status"] == 200  # request proceeds (no gate refusal)
    ident = captured["identity"]
    assert ident is not None and ident.kind == "client"
    assert ident.name == "__anonymous__"
    assert ident.datasets == frozenset()  # no MEMORY_DATASET → no grants

    # With MEMORY_DATASET set: exactly that one grant.
    monkeypatch.setenv("MEMORY_DATASET", "andrew-memory")
    _, captured = asyncio.run(_drive(_mw(), _scope()))
    ident = captured["identity"]
    assert ident is not None and ident.datasets == frozenset({"andrew-memory"})

    # MCP API keys configured → the anon branch is skipped; the presented
    # key resolves instead (an MCP keyset key IS an admin key on this
    # surface — D19 source-aware resolution binds it).
    monkeypatch.setenv("RAG_API_KEYS", "k1")
    messages, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("k1"))))
    assert messages[0]["status"] == 200
    ident = captured["identity"]
    assert ident is not None and ident.is_admin
    messages, _ = asyncio.run(_drive(_mw(), _scope(headers=_bearer("wrong"))))
    assert messages[0]["status"] == 401
    assert messages[1]["body"] == (b'{"error": "unauthorized: missing or invalid API key"}')


def test_middleware_registry_set_resolves_client_identity(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", "admin-mcp-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports,notes;bob:*")
    messages, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key"))))
    assert messages[0]["status"] == 200
    ident = captured["identity"]
    assert ident is not None and ident.name == "alice"
    assert ident.datasets == frozenset({"reports", "notes"})


def test_middleware_registry_set_admin_key_full_access(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "rest-admin")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    for key in ("rest-admin", "alice-key"):
        monkeypatch.setenv("RAG_API_KEYS", key) if key == "rest-admin" else None
    # deployment REST key presented to the MCP surface: resolve first by registry?
    monkeypatch.setenv("MCP_API_KEYS", "mcp-admin")
    messages, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("mcp-admin"))))
    assert messages[0]["status"] == 200
    assert captured["identity"] is not None and captured["identity"].is_admin


def test_middleware_registry_set_wrong_key_401_same_body(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS", "admin-mcp-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    messages, _ = asyncio.run(_drive(_mw(), _scope(headers=_bearer("nope"))))
    assert messages[0]["status"] == 401
    assert messages[1]["body"] == b'{"error": "unauthorized: missing or invalid API key"}'


def test_middleware_registry_set_missing_key_401(monkeypatch):
    """Registry configured + no key on a protected path = 401 (fail-closed —
    'just omit the key' is not an ACL bypass)."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    messages, _ = asyncio.run(_drive(_mw(), _scope(headers=[])))
    assert messages[0]["status"] == 401


def test_middleware_health_paths_stay_open(monkeypatch):
    """The parent's protected-path gate wins: /healthz is never identity-
    checked even with the registry on."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.delenv("MCP_API_KEYS", raising=False)
    monkeypatch.delenv("RAG_API_KEYS", raising=False)
    messages, captured = asyncio.run(_drive(_mw(), _scope(path="/healthz")))
    assert messages[0]["status"] == 200
    assert captured["identity"] is None


# ===========================================================================
# 4 · Per-key throttle buckets ride the D10 machinery
# ===========================================================================


def test_per_key_client_id_via_middleware(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*;bob:*")
    _, a = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key"), client=("10.0.0.1", 5))))
    _, b = asyncio.run(_drive(_mw(), _scope(headers=_bearer("bob-key"), client=("10.0.0.1", 5))))
    assert a["cid"] == "key:alice"
    assert b["cid"] == "key:bob"


def test_admin_key_keeps_d10_identity(monkeypatch):
    """Admin keys are deployment keys — the D10 identity sources still apply
    (no key: scope is minted for them)."""
    monkeypatch.setenv("RAG_API_KEYS", "admin-mcp-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    _, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("admin-mcp-key"), client=("203.0.113.5", 9))))
    assert captured["cid"] == "203.0.113.5"


def test_throttle_buckets_are_isolated_per_key(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*;bob:*")
    _, a = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key"))))
    cid_a, cid_b = a["cid"], "key:bob"
    # max out alice's bucket (same env knobs the D10 throttle uses)
    for _ in range(mcp._MCP_PW_MAX_FAILURES):
        mcp._mcp_pw_record_failure(cid_a)
    with pytest.raises(mcp.ToolError, match="Too many password attempts"):
        mcp._mcp_pw_check_throttle(cid_a)
    # bob's bucket is untouched — per-key isolation via the D10 machinery
    assert mcp._mcp_pw_failure_count(cid_b) == 0
    mcp._mcp_pw_check_throttle(cid_b)
    mcp._mcp_pw_reset_failures(cid_a)


def test_registry_identity_outranks_shared_unlock_escape(monkeypatch):
    """RAG_MCP_SHARED_UNLOCK=1 is a single-user convenience; an explicit
    multi-user registry is the stronger, authenticated config — per-key
    scoping wins when both are set."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", "1")
    _, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key"))))
    assert captured["cid"] == "key:alice"


def test_registry_key_scopes_unlock_cache_like_d10():
    from multimodal_rag.mcp_server import _cache_unlock, _is_unlocked

    _cache_unlock("ds", "pw-a", ttl=60)
    token = cr.set_current_identity(cr.Identity(kind="client", name="bob", datasets=frozenset({"*"})))
    try:
        assert _is_unlocked("ds") is None, "one key's unlock must not unlock for another key"
        _cache_unlock("ds", "pw-b", ttl=60)
        assert _is_unlocked("ds") == "pw-b"
    finally:
        cr.reset_current_identity(token)
    assert _is_unlocked("ds") == "pw-a"


# ===========================================================================
# 5 · MCP tool-surface ACL enforcement
# ===========================================================================


@pytest.fixture
def acl_dm(base_path, embedder, monkeypatch):
    """Real offline DatasetManager with two datasets. Local-mode Qdrant
    allows ONE client per storage folder, so only 'reports' gets a real
    collection; 'notes' exists as metadata (the ACL tests never touch its
    points)."""
    from multimodal_rag.dataset_manager import DatasetManager

    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("reports")
    manager._write_meta("notes", {"name": "notes", "document_count": 0, "schema_version": 2})
    mcp._dm = manager
    yield manager
    mcp._dm = None


def test_require_dataset_acl_tool_error(acl_dm):
    token = cr.set_current_identity(cr.Identity(kind="client", name="bob", datasets=frozenset({"notes"})))
    try:
        with pytest.raises(mcp.ToolError, match="not permitted"):
            mcp._require_dataset_acl("reports")
        mcp._require_dataset_acl("notes")  # granted → silent
    finally:
        cr.reset_current_identity(token)
    mcp._require_dataset_acl("reports")  # no identity → no enforcement


def test_list_datasets_filtered_by_acl_with_note(acl_dm, monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    _, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key")), app_body=mcp.list_datasets))
    out = captured["result"]
    assert "reports" in out
    assert "notes" not in out
    assert "hidden by API-key dataset ACLs" in out


def test_list_datasets_admin_and_default_unfiltered(acl_dm, monkeypatch):
    # default (no registry): byte-identical full listing, no note
    out_default = asyncio.run(mcp.list_datasets())
    assert "reports" in out_default and "notes" in out_default
    assert "ACLs" not in out_default
    # admin identity: full listing, no note
    monkeypatch.setenv("RAG_API_KEYS", "admin-mcp-key")
    token = cr.set_current_identity(cr.Identity(kind="admin", name=None, datasets=None))
    try:
        out_admin = asyncio.run(mcp.list_datasets())
    finally:
        cr.reset_current_identity(token)
    assert out_admin == out_default


def test_federated_targets_filtered_by_acl(acl_dm, monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")

    def _resolve_all():
        return mcp._resolve_federated_targets(acl_dm, "all")

    def _resolve_explicit():
        return mcp._resolve_federated_targets(acl_dm, ["reports", "notes"])

    _, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key")), app_body=_resolve_all))
    targets, skipped, _errors = captured["result"]
    assert targets == ["reports"]
    assert any(s["dataset"] == "notes" and "ACL" in s["reason"] for s in skipped)
    _, captured = asyncio.run(_drive(_mw(), _scope(headers=_bearer("alice-key")), app_body=_resolve_explicit))
    targets, skipped, _errors = captured["result"]
    # explicit denied name → skipped note (never a hard fail, never widened)
    assert targets == ["reports"]
    assert {s["dataset"] for s in skipped} == {"notes"}


def test_search_and_files_and_info_denied_by_acl(acl_dm, monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:notes")
    for call in (
        lambda: mcp.search_dataset(dataset_name="reports", query="x"),
        lambda: mcp.get_dataset_files(dataset_name="reports"),
        lambda: mcp.get_dataset_info(dataset_name="reports"),
        lambda: mcp.unlock_dataset(dataset_name="reports", password="x"),
    ):
        with pytest.raises(mcp.ToolError, match="not permitted"):
            asyncio.run(_drive(_mw(), _scope(headers=_bearer("bob-key")), app_body=call))
    # granted dataset works end to end (existence + password gates pass)
    _, captured = asyncio.run(
        _drive(
            _mw(),
            _scope(headers=_bearer("bob-key")),
            app_body=lambda: mcp.get_dataset_info(dataset_name="notes"),
        )
    )
    assert json.loads(captured["result"])["name"] == "notes"


def test_memory_tools_denied_by_acl(acl_dm, monkeypatch):
    """The caller's memory dataset is a dataset — the same ACL applies."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:notes")
    for call in (
        lambda: mcp.add_memory(text="remember this", dataset_name="reports"),
        lambda: mcp.search_memory(query="x", dataset_name="reports"),
        lambda: mcp.delete_memory(memory_ids=["a"], dataset_name="reports"),
        lambda: mcp.list_memories(dataset_name="reports"),
    ):
        with pytest.raises(mcp.ToolError, match="not permitted"):
            asyncio.run(_drive(_mw(), _scope(headers=_bearer("bob-key")), app_body=call))


# ===========================================================================
# 6 · REST middleware + surfaces
# ===========================================================================


class _FakeRequest:
    def __init__(self, path, headers=None, method="GET", client=("10.9.0.1", 5000)):
        from starlette.datastructures import Headers

        self.url = type("U", (), {"path": path})()
        self.headers = Headers(raw=[(k.encode(), v.encode()) for k, v in (headers or {}).items()])
        self.method = method
        self.scope = {}
        self.client = type("C", (), {"host": client[0], "port": client[1]})()


def _hdrs(key=None):
    h = {}
    if key:
        h["x-rag-api-key"] = key
    return h


async def _call_mw(request, ran):
    async def call_next(req):
        ran.append(req)
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True})

    return await api._api_key_auth(request, call_next)


def test_rest_default_single_key_behaviour_byte_identical(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    ran = []
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/reports", _hdrs("deployment-key")), ran))
    assert resp.status_code == 200 and len(ran) == 1
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/reports", _hdrs("wrong")), ran))
    assert resp.status_code == 401
    assert json.loads(resp.body) == {"detail": "Missing or invalid API key"}
    assert len(ran) == 1


def test_rest_registry_key_allowed_and_denied_by_acl(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports,notes;bob:*")
    ran = []
    # granted dataset → 200, identity bound for the handlers (list filtering)
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/reports/documents", _hdrs("alice-key")), ran))
    assert resp.status_code == 200 and len(ran) == 1
    assert cr.current_identity() is None, "identity is scoped to the request, not leaked"
    # unknown key → 401 (never a silent allow)
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/reports", _hdrs("mallory-key")), ran))
    assert resp.status_code == 401
    # '*' grant → every dataset allowed
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/anything", _hdrs("bob-key")), ran))
    assert resp.status_code == 200
    # unit: the path-denial matrix itself
    alice = cr.resolve_presented(["alice-key"])
    bob = cr.resolve_presented(["bob-key"])
    assert api._rag_acl_path_denial("/api/datasets/reports", "GET", alice) is None
    assert "not permitted" in api._rag_acl_path_denial("/api/datasets/other", "GET", alice)
    assert api._rag_acl_path_denial("/api/datasets/other", "GET", bob) is None


def test_rest_registry_admin_keys(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv("MCP_API_KEYS", "mcp-admin")
    ran = []
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets", _hdrs("mcp-admin")), ran))
    assert resp.status_code == 200
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets", _hdrs("deployment-key")), ran))
    assert resp.status_code == 200


def test_rest_client_key_denied_admin_surface_and_create(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    ran = []
    resp = asyncio.run(_call_mw(_FakeRequest("/api/admin/health", _hdrs("alice-key")), ran))
    assert resp.status_code == 403
    assert "no admin access" in json.loads(resp.body)["detail"]
    # '*' grant: dataset creation allowed (everything is in scope anyway)
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets", _hdrs("alice-key"), method="POST"), ran))
    assert resp.status_code == 200


def test_rest_named_acl_key_cannot_create_datasets(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    ran = []
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets", _hdrs("alice-key"), method="POST"), ran))
    assert resp.status_code == 403
    assert "cannot create datasets" in json.loads(resp.body)["detail"]
    # GET list is fine (it is filtered, not denied)
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets", _hdrs("alice-key")), ran))
    assert resp.status_code == 200


def test_rest_registry_no_key_is_not_an_acl_bypass(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    ran = []
    resp = asyncio.run(_call_mw(_FakeRequest("/api/datasets/reports", {}), ran))
    assert resp.status_code == 401, "omitting the key must not reach the dataset"


def test_rest_acl_filter_datasets(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "x:k")  # registry must be ON for the gate
    datasets = [{"name": "a"}, {"name": "b"}]
    # registry off → identity, byte-identical
    assert api._rag_acl_filter_datasets(datasets) == (datasets, 0)
    # client identity → filtered + honest hidden count
    token = cr.set_current_identity(cr.Identity(kind="client", name="x", datasets=frozenset({"b"})))
    try:
        visible, hidden = api._rag_acl_filter_datasets(datasets)
        assert visible == [{"name": "b"}] and hidden == 1
    finally:
        cr.reset_current_identity(token)
    # admin → identity
    token = cr.set_current_identity(cr.Identity(kind="admin", name=None, datasets=None))
    try:
        assert api._rag_acl_filter_datasets(datasets) == (datasets, 0)
    finally:
        cr.reset_current_identity(token)


def test_rest_unlock_client_id_registry_priority(monkeypatch):
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    request = _FakeRequest("/api/datasets")
    # no identity → D10 behaviour (unchanged)
    assert api._unlock_client_id(request) == api._request_identity(request)
    token = cr.set_current_identity(cr.Identity(kind="client", name="alice", datasets=frozenset()))
    try:
        assert api._unlock_client_id(request) == "key:alice"
    finally:
        cr.reset_current_identity(token)


def test_rest_federated_search_filters_targets(monkeypatch):
    """POST /api/search fan-out honours the ACL via resolve_federated_targets."""
    from multimodal_rag.rag_system import resolve_federated_targets

    class _FakeDM:
        def list_datasets(self):
            return [{"name": "a"}, {"name": "b"}]

        def get_dataset(self, name, sync_count=False):
            if name not in ("a", "b"):
                raise FileNotFoundError(name)
            return {"name": name}

        def has_password(self, name):
            return False

    token = cr.set_current_identity(cr.Identity(kind="client", name="x", datasets=frozenset({"b"})))
    try:
        allowed = lambda name: cr.dataset_allowed(cr.current_identity(), name)
        targets, skipped, _ = resolve_federated_targets(_FakeDM(), "all", None, allowed)
        assert targets == ["b"]
        assert {s["dataset"] for s in skipped} == {"a"}
    finally:
        cr.reset_current_identity(token)


# ---------------------------------------------------------------------------
# 7 · Full-stack REST: TestClient → middleware → handler filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def list_client(monkeypatch):
    """The real FastAPI app with a stubbed manager (offline)."""
    from fastapi.testclient import TestClient

    class _FakeDM:
        def list_datasets(self):
            return [{"name": "reports"}, {"name": "notes"}, {"name": "other"}]

        def get_dataset(self, name, sync_count=False):
            if name not in ("reports", "notes", "other"):
                raise FileNotFoundError(f"Dataset '{name}' not found")
            return {"name": name, "document_count": 0}

        def has_password(self, name):
            return False

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    return TestClient(api.app)


def test_rest_full_stack_list_filtered_for_acl_key(list_client, monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;boss:boss-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports,notes;boss:*")
    # registry key → only the granted datasets, with an honest hidden count
    r = list_client.get("/api/datasets", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
    body = r.json()
    assert [d["name"] for d in body["datasets"]] == ["reports", "notes"]
    assert body["acl_hidden"] == 1
    # '*' grant → everything, no hidden key at all
    r = list_client.get("/api/datasets", headers={"X-RAG-Api-Key": "boss-key"})
    body = r.json()
    assert [d["name"] for d in body["datasets"]] == ["reports", "notes", "other"]
    assert "acl_hidden" not in body
    # admin deployment key → full listing, no acl keys, shape as before
    r = list_client.get("/api/datasets", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200
    body = r.json()
    assert [d["name"] for d in body["datasets"]] == ["reports", "notes", "other"]
    assert "acl_hidden" not in body


def test_rest_full_stack_denied_dataset_path(list_client, monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:reports")
    r = list_client.get("/api/datasets/notes", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403
    assert "not permitted" in r.json()["detail"]
    r = list_client.get("/api/datasets/reports", headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 200
