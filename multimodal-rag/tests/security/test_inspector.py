"""Decision D27 — the MCP Inspector: direct MCP tool calls as the session.

The homepage SPA's Inspector tab proxies two surfaces through the API
server — ``GET /api/inspector/tools`` (MCP ``tools/list``) and
``POST /api/inspector/call`` (MCP ``tools/call``) — to the configured MCP
endpoint, forwarding EXACTLY the credential envelope the request itself
authenticated with, re-expressed in the MCP middleware's accepted headers.
The invariants pinned here, all fail-closed:

* the forwarded credential is the session's own, never wider: a registry
  key rides ``X-API-Key`` (the same D19 "x-api-key" source the MCP side
  resolves), a Bearer key rides ``Authorization: Bearer``, the forwarded
  OIDC token rides ``X-Auth-Request-Access-Token``, the D22 SSO cookie JWT
  re-envelopes as ``Authorization: Bearer`` (a JWT can only ever resolve to
  a client identity on MCP — never admin);
* the D24 proxy-identity session forwards NOTHING (identity headers are
  names, not secrets) — the keyed sidecar's 401 surfaces verbatim;
* ``X-RAG-Api-Key`` is NEVER forwarded by name (the MCP middleware does not
  accept it — the re-envelope to ``X-API-Key`` is the whole point);
* tool-level failures (``isError`` results, JSON-RPC error objects) are 200
  ``ok: false`` — the transport round trip succeeded; transport failures
  (unreachable sidecar, rejected credential, non-JSON) are 502/504;
* the endpoints are never public: an unauthenticated caller gets the
  middleware's 401 before any proxying happens.

The MCP sidecar is a loopback stub HTTP server that records the exact
headers each JSON-RPC request presented — the assertion surface for every
forwarding rule.

Run::

    pytest tests/security/test_inspector.py -q
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr

# ===========================================================================
# The loopback MCP-sidecar stub: records headers, answers JSON-RPC
# ===========================================================================


class _StubMCPHandler(BaseHTTPRequestHandler):
    """A scripted MCP streamable-http endpoint.

    Class attributes (set per test):
      ``required_key``  — when set, requests without this ``X-API-Key``
                          (or matching Bearer) get a 401 like the keyed
                          sidecar middleware;
      ``rpc_response``  — the JSON-RPC response body for the CURRENT method
                          (default: an honest tools/list + tools/call pair);
      ``raw_status`` / ``raw_body`` — override both: answer with a raw
                          HTTP status/body instead of JSON-RPC.

    Instance state (``server.rpc_calls``): one dict per POST —
    ``{"method", "params", "headers"}``.
    """

    required_key: str | None = None
    rpc_response: dict | None = None
    raw_status: int | None = None
    raw_body: bytes = b""

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8"))
        except ValueError:
            payload = {}
        self.server.rpc_calls.append(
            {
                "method": payload.get("method"),
                "params": payload.get("params"),
                "headers": {k.lower(): v for k, v in self.headers.items()},
            }
        )
        if self.required_key is not None:
            presented = (
                self.headers.get("X-API-Key")
                or (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                or ""
            )
            if presented != self.required_key:
                self._respond(
                    401,
                    b'{"error": "unauthorized: missing or invalid API key"}',
                    "application/json",
                )
                return
        if self.raw_status is not None:
            self._respond(self.raw_status, self.raw_body, "text/plain")
            return
        response = self.rpc_response
        if response is None:
            method = payload.get("method")
            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "search_dataset",
                                "description": "Search ONE multimodal RAG dataset",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "dataset_name": {"type": "string"},
                                        "query": {"type": "string"},
                                    },
                                    "required": ["dataset_name", "query"],
                                },
                            },
                            {
                                "name": "add_memory",
                                "description": "Store a memory for later recall",
                                "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                            },
                        ]
                    },
                }
            elif method == "tools/call":
                response = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "stub tool ran"}],
                        "isError": False,
                    },
                }
            else:
                response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "Method not found"}}
        self._respond(200, json.dumps(response).encode("utf-8"), "application/json")

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the per-request stderr noise
        pass


@pytest.fixture
def mcp_stub():
    """One loopback stub MCP server per test; the URL rides
    RAG_INSPECTOR_MCP_URL (read per request — no restart discipline)."""
    server = HTTPServer(("127.0.0.1", 0), _StubMCPHandler)
    server.rpc_calls = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _StubMCPHandler.required_key = None
    _StubMCPHandler.rpc_response = None
    _StubMCPHandler.raw_status = None
    _StubMCPHandler.raw_body = b""
    url = f"http://127.0.0.1:{server.server_address[1]}/mcp"
    yield type("Stub", (), {"url": url, "calls": server.rpc_calls, "server": server})
    server.shutdown()
    server.server_close()


# ===========================================================================
# House hygiene: full env isolation (the D15/D17/D23/D25 fixture pattern)
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Every test starts with NO registry, NO keys, NO store, NO SSO —
    everything a test needs is set explicitly inside it."""
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
        "RAG_ACCESS_STORE",
        ar.SELF_MINT_ENV,
        "MEMORY_DATASET",
        "RAG_MEMORY_DATASET",
        "RAG_TRUST_PROXY_IDENTITY",
        "RAG_OIDC_SSO_ENABLED",
        "RAG_OIDC_SSO_CLIENT_ID",
        "RAG_OIDC_SSO_CLIENT_SECRET",
        "RAG_OIDC_SSO_REDIRECT_URI",
        "RAG_INSPECTOR_MCP_URL",
        "RAG_INSPECTOR_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    api._UNLOCK_CACHE.clear()
    yield
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    api._UNLOCK_CACHE.clear()


@pytest.fixture
def rest_client(monkeypatch, mcp_stub):
    """The real FastAPI app wired at the loopback stub (the D25 pattern),
    with a stubbed manager + no-op eager init (the Inspector touches
    neither)."""
    from fastapi.testclient import TestClient

    class _FakeDM:
        def list_datasets(self):
            return []

    async def _fake_manager():
        return _FakeDM()

    async def _noop_init():
        return None

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_eager_init", _noop_init)
    monkeypatch.setattr(api, "_RAG_API_KEY", "")
    monkeypatch.setenv("RAG_INSPECTOR_MCP_URL", mcp_stub.url)
    api._UNLOCK_CACHE.clear()
    yield TestClient(api.app)
    api._UNLOCK_CACHE.clear()


# ===========================================================================
# tools/list — catalog + credential chip
# ===========================================================================


def test_tools_endpoint_requires_auth(rest_client, monkeypatch):
    """A keyed deployment (registry configured), no credential → the auth
    middleware 401s BEFORE any proxying happens (never a public surface)."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    r = rest_client.get("/api/inspector/tools")
    assert r.status_code == 401


def test_tools_lists_catalog_and_labels_the_key(rest_client, mcp_stub, monkeypatch):
    """A registry-key session: the catalog passes through and the chip says
    ``api-key`` — the envelope a call will authenticate with."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    r = rest_client.get("/api/inspector/tools", headers={"X-RAG-Api-Key": "env-key-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint"] == mcp_stub.url
    assert body["session_credential"] == "api-key"
    names = [t["name"] for t in body["tools"]]
    assert names == ["search_dataset", "add_memory"]
    assert body["tools"][0]["inputSchema"]["required"] == ["dataset_name", "query"]


def test_tools_credential_none_when_session_has_no_key(rest_client, monkeypatch):
    """The D20 anonymous posture (nothing configured): the request passes
    the middleware bound to the anonymous identity, and the chip honestly
    reports that no credential will be forwarded."""
    r = rest_client.get("/api/inspector/tools")
    assert r.status_code == 200
    assert r.json()["session_credential"] == "none"


# ===========================================================================
# tools/call — the credential-forwarding matrix
# ===========================================================================


def test_call_forwards_registry_key_as_x_api_key(rest_client, mcp_stub, monkeypatch):
    """THE core contract: a pasted registry-key session authenticates the
    MCP call as the SAME identity — the key arrives as ``X-API-Key``, the
    JSON-RPC body is a plain tools/call, and ``X-RAG-Api-Key`` (a header
    the MCP middleware does not accept) never leaks through by name."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    r = rest_client.post(
        "/api/inspector/call",
        headers={"X-RAG-Api-Key": "env-key-1"},
        json={"name": "search_dataset", "arguments": {"dataset_name": "notes", "query": "runbook"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["tool"] == "search_dataset"
    assert body["duration_ms"] >= 0
    assert body["result"]["content"][0]["text"] == "stub tool ran"
    assert len(mcp_stub.calls) == 1
    call = mcp_stub.calls[0]
    assert call["method"] == "tools/call"
    assert call["params"] == {"name": "search_dataset", "arguments": {"dataset_name": "notes", "query": "runbook"}}
    assert call["headers"].get("x-api-key") == "env-key-1"
    assert "x-rag-api-key" not in call["headers"]


def test_call_forwards_bearer_key_as_authorization(rest_client, mcp_stub, monkeypatch):
    """A Bearer-presented registry key stays a Bearer credential (its D19
    source is 'authorization', not 'x-api-key')."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    r = rest_client.post(
        "/api/inspector/call",
        headers={"Authorization": "Bearer env-key-1"},
        json={"name": "add_memory", "arguments": {"text": "hello"}},
    )
    assert r.status_code == 200 and r.json()["ok"] is True
    call = mcp_stub.calls[0]
    assert call["headers"].get("authorization") == "Bearer env-key-1"
    assert "x-api-key" not in call["headers"]


def test_call_forwards_forwarded_token_envelope(rest_client, mcp_stub):
    """The auth-proxy's forwarded OIDC token rides the SAME header name to
    the sidecar (the MCP middleware accepts that envelope — D21).  With the
    registry unset the token fails to resolve to anything... on a keyed
    deployment.  Here NOTHING is keyed (D20 anonymous REST posture), so the
    envelope still forwards and the call round-trips open."""
    r = rest_client.post(
        "/api/inspector/call",
        headers={"X-Auth-Request-Access-Token": "fwd-jwt-value"},
        json={"name": "add_memory"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    call = mcp_stub.calls[0]
    assert call["headers"].get("x-auth-request-access-token") == "fwd-jwt-value"


def test_forward_headers_mapping_matrix():
    """Unit matrix for `_inspector_forward_headers` — every presented source
    maps to the MCP middleware's accepted envelope, in pair order, with
    duplicates collapsed and the proxy-identity session forwarding nothing."""
    from starlette.requests import Request

    def _request(headers: dict[str, str], client: str = "10.1.2.3") -> Request:
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": (client, 1234), "method": "POST", "path": "/"})

    # Paste key (either spelling) → X-API-Key.
    fwd, via = api._inspector_forward_headers(_request({"X-RAG-Api-Key": "k1"}))
    assert via == "api-key" and fwd == [("X-API-Key", "k1"), ("X-Forwarded-For", "10.1.2.3")]

    # Bearer → Bearer.
    fwd, via = api._inspector_forward_headers(_request({"Authorization": "Bearer k2"}))
    assert via == "bearer-token" and fwd == [("Authorization", "Bearer k2"), ("X-Forwarded-For", "10.1.2.3")]

    # Forwarded OIDC token → the same envelope name.
    fwd, via = api._inspector_forward_headers(_request({"X-Auth-Request-Access-Token": "jwt"}))
    assert via == "forwarded-jwt" and fwd == [("X-Auth-Request-Access-Token", "jwt"), ("X-Forwarded-For", "10.1.2.3")]

    # D22 SSO cookie → re-enveloped as a Bearer JWT (client-identity only).
    fwd, via = api._inspector_forward_headers(_request({"Cookie": "pcai-sso=jwt-token"}))
    cookie_pair = next((n, v) for n, v in fwd if n == "Authorization")
    assert via == "sso-jwt" and cookie_pair == ("Authorization", "Bearer jwt-token")

    # D24 proxy-identity only → NOTHING forwardable (a name is not a secret).
    fwd, via = api._inspector_forward_headers(_request({"X-Auth-Request-User": "andrew"}))
    assert via == "none" and fwd == [("X-Forwarded-For", "10.1.2.3")]

    # D19 delegation order preserved: X-API-Key outranks a co-forwarded
    # Bearer on the sidecar too (pair order kept).
    fwd, via = api._inspector_forward_headers(_request({"X-API-Key": "delegated", "Authorization": "Bearer transport"}))
    assert via == "api-key"
    assert fwd[:2] == [("X-API-Key", "delegated"), ("Authorization", "Bearer transport")]

    # Duplicate envelope collapsed (X-RAG-Api-Key + X-API-Key, same value).
    fwd, _ = api._inspector_forward_headers(_request({"X-RAG-Api-Key": "k1", "X-API-Key": "k1"}))
    assert [n for n, _ in fwd].count("X-API-Key") == 1


def test_call_tool_error_result_is_ok_false(rest_client, mcp_stub, monkeypatch):
    """A tool-level failure (FastMCP's CallToolResult isError=true) is a 200
    with ok:false — the round trip succeeded, the TOOL refused."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    _StubMCPHandler.rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": "Dataset 'x' is not permitted."}], "isError": True},
    }
    r = rest_client.post(
        "/api/inspector/call",
        headers={"X-RAG-Api-Key": "env-key-1"},
        json={"name": "search_dataset", "arguments": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] is None
    assert body["result"]["isError"] is True


def test_call_json_rpc_error_is_ok_false(rest_client, mcp_stub, monkeypatch):
    """A JSON-RPC protocol error passes through as ok:false + the error
    object (never morphed into a proxy 502)."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    _StubMCPHandler.rpc_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "Invalid arguments"},
    }
    r = rest_client.post(
        "/api/inspector/call",
        headers={"X-RAG-Api-Key": "env-key-1"},
        json={"name": "search_dataset", "arguments": {}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["message"] == "Invalid arguments"


def test_call_empty_arguments_sends_empty_object(rest_client, mcp_stub, monkeypatch):
    """`arguments` omitted → the proxy sends `{}` (a valid MCP tools/call),
    never a missing key."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    r = rest_client.post(
        "/api/inspector/call",
        headers={"X-RAG-Api-Key": "env-key-1"},
        json={"name": "list_datasets"},
    )
    assert r.status_code == 200 and r.json()["ok"] is True
    assert mcp_stub.calls[0]["params"] == {"name": "list_datasets", "arguments": {}}


def test_call_validates_body(rest_client, monkeypatch):
    """Shape validation happens BEFORE any proxying: a missing/garbled tool
    name or a non-object arguments → 400."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    for payload in (
        {"arguments": {}},
        {"name": ""},
        {"name": "bad name with spaces"},
        {"name": "../escape"},
        {"name": "ok_tool", "arguments": ["not", "an", "object"]},
    ):
        r = rest_client.post("/api/inspector/call", headers={"X-RAG-Api-Key": "env-key-1"}, json=payload)
        assert r.status_code == 400, payload


# ===========================================================================
# Transport failures — fail loudly, never silently succeed
# ===========================================================================


def test_keyed_sidecar_rejecting_the_credential_is_502(rest_client, mcp_stub, monkeypatch):
    """A deployment key that authenticates REST but is NOT an MCP keyset
    key: the sidecar 401s and the proxy surfaces it as 502 with the
    paste-an-key guidance — an honest fail-closed, never a widened call."""
    monkeypatch.setenv("RAG_API_KEY", "rest-only-deployment-key")
    monkeypatch.setattr(api, "_RAG_API_KEY", "rest-only-deployment-key")
    _StubMCPHandler.required_key = "some-other-mcp-key"
    r = rest_client.get("/api/inspector/tools", headers={"X-RAG-Api-Key": "rest-only-deployment-key"})
    assert r.status_code == 502
    assert "rejected this session's credential" in r.json()["detail"]


def test_unreachable_sidecar_is_502(rest_client, monkeypatch):
    """The sidecar down (connection refused) → 502 with the endpoint named,
    never a hang and never a fabricated empty catalog."""
    monkeypatch.setattr(api, "_RAG_API_KEY", "")
    monkeypatch.setenv("RAG_INSPECTOR_MCP_URL", "http://127.0.0.1:9/mcp")  # discard port, nothing listens
    r = rest_client.get("/api/inspector/tools")
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"]


def test_non_json_sidecar_is_502(rest_client, mcp_stub, monkeypatch):
    """A target that answers HTML/SSE where JSON was required (a mis-pointed
    RAG_INSPECTOR_MCP_URL) → a clear 502, not a confusing parse error."""
    monkeypatch.setattr(api, "_RAG_API_KEY", "")
    _StubMCPHandler.raw_status = 200
    _StubMCPHandler.raw_body = b"<html>not json</html>"
    r = rest_client.get("/api/inspector/tools")
    assert r.status_code == 502
    assert "non-JSON" in r.json()["detail"]


def test_timeout_env_read_per_request(rest_client, monkeypatch):
    """RAG_INSPECTOR_TIMEOUT_S is read per request; a garbage value falls
    back to the 300 s default (never a crash, never 0 = no timeout)."""
    monkeypatch.setattr(api, "_RAG_API_KEY", "")
    monkeypatch.setenv("RAG_INSPECTOR_TIMEOUT_S", "garbage")
    assert api._inspector_timeout_s() == 300.0
    monkeypatch.setenv("RAG_INSPECTOR_TIMEOUT_S", "0")
    assert api._inspector_timeout_s() == 300.0
    monkeypatch.setenv("RAG_INSPECTOR_TIMEOUT_S", "5")
    assert api._inspector_timeout_s() == 5.0


# ===========================================================================
# Open-sidecar topology (no keys anywhere — local dev)
# ===========================================================================


def test_open_sidecar_anonymous_session_round_trips(rest_client, mcp_stub):
    """The unconfigured deployment (D20 anonymous identity, MCP sidecar
    open): the Inspector works with NO credential forwarded — and the stub
    confirms the request arrived with neither X-API-Key nor Bearer."""
    r = rest_client.get("/api/inspector/tools")
    assert r.status_code == 200
    assert r.json()["session_credential"] == "none"
    call = mcp_stub.calls[0]
    assert "x-api-key" not in call["headers"]
    assert "authorization" not in call["headers"]
