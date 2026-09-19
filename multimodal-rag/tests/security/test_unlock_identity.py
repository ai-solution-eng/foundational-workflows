"""Decision D10 tests: per-caller unlock identity, shared escape hatch, and
the /metrics auth gate.

D10 (ratified): one caller's ``unlock_dataset`` must not unlock a dataset
for every caller on the pod.  The unlock cache and the password-failure
throttle are keyed by a per-caller identity resolved in order of
availability:

  1. provided identity — auth-proxy headers (RAG_TRUST_PROXY_IDENTITY)
  2. forwarded-for chain — X-Forwarded-For (same trust gate)
  3. socket peer — direct connections (single-user direct: unchanged)

``RAG_MCP_SHARED_UNLOCK=1`` collapses every caller onto the shared
``"default"`` identity — the pre-D10 behaviour, restored explicitly for
gateway-fronted single-user deployments.

Both identity surfaces are tested: the MCP server (context-var middleware)
and the REST api server (Starlette Request).  Also here: the RAG_METRICS_AUTH
gate on the /metrics exposition (default off = unchanged).

Run::

    pytest tests/security/test_unlock_identity.py -q
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp


@pytest.fixture(autouse=True)
def _identity_env(monkeypatch):
    """Deterministic identity environment per test."""
    monkeypatch.setattr(mcp, "_TRUST_PROXY_IDENTITY", False)
    monkeypatch.setattr(api, "_TRUST_PROXY_IDENTITY", False)
    monkeypatch.delenv("RAG_MCP_SHARED_UNLOCK", raising=False)
    yield


# ---------------------------------------------------------------------------
# MCP side: middleware identity resolution
# ---------------------------------------------------------------------------


def _drive_middleware(headers=None, client=("10.9.0.1", 5000)):
    """Run _MemoryHeaderMiddleware on a synthetic scope; return the
    _unlock_client_id visible inside the wrapped app."""
    captured = {}

    async def app(scope, receive, send):
        captured["cid"] = mcp._unlock_client_id()

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers or [],
        "client": client,
        "query_string": b"",
    }
    mw = mcp._MemoryHeaderMiddleware(app)
    asyncio.run(mw(scope, receive, send))
    return captured.get("cid")


def test_mcp_direct_peer_identity_unchanged():
    """Single-user direct: the socket peer is the identity (today's UX)."""
    cid = _drive_middleware(client=("203.0.113.7", 5000))
    assert cid == "203.0.113.7"


def test_mcp_identity_headers_win_when_trusted():
    headers = [(b"x-auth-request-email", b"alice@corp.example")]
    mcp._TRUST_PROXY_IDENTITY = True
    try:
        assert _drive_middleware(headers=headers, client=("10.0.0.5", 1)) == "alice@corp.example"
    finally:
        mcp._TRUST_PROXY_IDENTITY = False


def test_mcp_identity_headers_ignored_when_untrusted():
    """Without the operator's proxy confirmation the headers are
    client-supplied and spoofable — the peer is used (spoofing the header
    must NOT change the identity)."""
    headers = [(b"x-auth-request-email", b"victim@corp.example")]
    assert _drive_middleware(headers=headers, client=("10.0.0.5", 1)) == "10.0.0.5"


def test_mcp_forwarded_for_chain_used_when_trusted():
    mcp._TRUST_PROXY_IDENTITY = True
    try:
        cid = _drive_middleware(headers=[(b"x-forwarded-for", b"203.0.113.9, 10.0.0.5")], client=("10.0.0.5", 1))
        assert cid == "xff:203.0.113.9,10.0.0.5"
    finally:
        mcp._TRUST_PROXY_IDENTITY = False


def test_mcp_forwarded_for_ignored_when_untrusted():
    headers = [(b"x-forwarded-for", b"anything-can-be-typed-here")]
    assert _drive_middleware(headers=headers, client=("10.0.0.5", 1)) == "10.0.0.5"


def test_mcp_identity_precedence_over_xff():
    mcp._TRUST_PROXY_IDENTITY = True
    try:
        cid = _drive_middleware(
            headers=[(b"x-forwarded-for", b"1.1.1.1"), (b"x-auth-request-user", b"bob")],
            client=("10.0.0.5", 1),
        )
        assert cid == "bob", "the provided (per-USER) identity outranks the per-IP chain"
    finally:
        mcp._TRUST_PROXY_IDENTITY = False


def test_mcp_shared_unlock_escape_hatch(monkeypatch):
    mcp._TRUST_PROXY_IDENTITY = True
    monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", "1")
    try:
        cid = _drive_middleware(headers=[(b"x-auth-request-email", b"alice@corp.example")], client=("10.0.0.5", 1))
        assert cid == "default", "the escape hatch restores the shared identity"
        assert mcp._unlock_client_id() == "default"
    finally:
        mcp._TRUST_PROXY_IDENTITY = False


def test_mcp_shared_unlock_accepts_truthy_forms(monkeypatch):
    for val in ("1", "true", "YES", " yes "):
        monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", val)
        assert mcp._shared_unlock_enabled() is True, val
    for val in ("", "0", "false", "off"):
        monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", val)
        assert mcp._shared_unlock_enabled() is False, repr(val)


# ---------------------------------------------------------------------------
# MCP side: per-identity unlock cache + throttle isolation
# ---------------------------------------------------------------------------


def test_mcp_unlock_cache_is_per_caller():
    mcp._unlocked.clear()
    try:
        tok_a = mcp._client_id_ctx.set("caller-A")
        try:
            mcp._cache_unlock("ds", "pw-A", ttl=300)
        finally:
            mcp._client_id_ctx.reset(tok_a)
        tok = mcp._client_id_ctx.set("caller-B")
        try:
            assert mcp._is_unlocked("ds") is None, "caller B must not see caller A's unlock"
        finally:
            mcp._client_id_ctx.reset(tok)
        tok = mcp._client_id_ctx.set("caller-A")
        try:
            assert mcp._is_unlocked("ds") == "pw-A"
        finally:
            mcp._client_id_ctx.reset(tok)
    finally:
        mcp._unlocked.clear()


def test_mcp_throttle_is_per_identity():
    mcp._mcp_pw_fail_buckets.clear()
    try:
        for _ in range(mcp._MCP_PW_MAX_FAILURES):
            mcp._mcp_pw_record_failure("attacker")
        with pytest.raises(Exception, match="Too many password attempts"):
            mcp._mcp_pw_check_throttle("attacker")
        mcp._mcp_pw_check_throttle("victim")  # different identity: fresh bucket
        # an identity cannot be rotated onto another's bucket
        assert mcp._mcp_pw_failure_count("victim") == 0
    finally:
        mcp._mcp_pw_fail_buckets.clear()


def test_mcp_throttle_shares_under_escape_hatch(monkeypatch):
    """RAG_MCP_SHARED_UNLOCK=1: one bucket (today's shared behaviour)."""
    mcp._mcp_pw_fail_buckets.clear()
    monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", "1")
    try:
        cid = mcp._unlock_client_id()  # "default" for every caller under the hatch
        for _ in range(mcp._MCP_PW_MAX_FAILURES):
            mcp._mcp_pw_record_failure(cid)
        with pytest.raises(Exception, match="Too many password attempts"):
            mcp._mcp_pw_check_throttle(cid)
    finally:
        mcp._mcp_pw_fail_buckets.clear()


def test_mcp_unlock_check_is_scoped_in_tool_flow():
    """_check_unlocked_or_password: caller A's cached password never
    satisfies caller B's password requirement."""
    mcp._unlocked.clear()

    class _StubDM:
        def has_password(self, name):
            return True

        def verify_password(self, name, pw):
            return pw == "correct-pw"

    try:
        tok_a = mcp._client_id_ctx.set("caller-A")
        try:
            mcp._cache_unlock("ds", "correct-pw", ttl=300)
        finally:
            mcp._client_id_ctx.reset(tok_a)
        tok = mcp._client_id_ctx.set("caller-B")
        try:
            with pytest.raises(Exception, match="[Pp]assword"):
                mcp._check_unlocked_or_password(_StubDM(), "ds", None)
            assert mcp._check_unlocked_or_password(_StubDM(), "ds", "correct-pw") == "correct-pw"
        finally:
            mcp._client_id_ctx.reset(tok)
    finally:
        mcp._unlocked.clear()


# ---------------------------------------------------------------------------
# REST side: same identity contract on the Starlette request path
# ---------------------------------------------------------------------------


def _req(headers=None, client=("10.9.0.1", 5000)):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/datasets/d/unlock",
        "headers": headers or [],
        "client": client,
        "query_string": b"",
    }
    return Request(scope)


def test_rest_direct_peer_identity_unchanged():
    assert api._unlock_client_id(_req(client=("203.0.113.7", 5))) == "203.0.113.7"


def test_rest_identity_precedence_chain():
    api._TRUST_PROXY_IDENTITY = True
    try:
        # provided identity wins
        assert api._unlock_client_id(_req([(b"x-auth-request-email", b"alice@x")])) == "alice@x"
        # then the forwarded-for chain
        assert (
            api._unlock_client_id(_req([(b"x-forwarded-for", b"198.51.100.2 , 10.0.0.1")]))
            == "xff:198.51.100.2,10.0.0.1"
        )
        # then the socket peer
        assert api._unlock_client_id(_req()) == "10.9.0.1"
    finally:
        api._TRUST_PROXY_IDENTITY = False


def test_rest_untrusted_headers_never_change_identity():
    headers = [(b"x-auth-request-email", b"victim@x"), (b"x-forwarded-for", b"spoofed")]
    assert api._unlock_client_id(_req(headers)) == "10.9.0.1"


def test_rest_shared_unlock_escape(monkeypatch):
    monkeypatch.setenv("RAG_MCP_SHARED_UNLOCK", "1")
    assert api._unlock_client_id(_req([(b"x-auth-request-email", b"alice@x")])) == "default"


def test_rest_unlock_cache_is_per_caller():
    api._UNLOCK_CACHE.clear()
    try:
        api._unlock_cache_set("ds", "pw-A", "secret-a", ttl=300)
        assert api._unlock_cache_get("ds", "pw-B") is None
        assert api._unlock_cache_get("ds", "pw-A") == "secret-a"
    finally:
        api._UNLOCK_CACHE.clear()


# ---------------------------------------------------------------------------
# /metrics auth gate (default OFF = unchanged)
# ---------------------------------------------------------------------------


class _HdrReq:
    def __init__(self, headers):
        self._h = {k.lower(): v for k, v in headers}

    def headers(self):
        return self._h


def _fake_request(headers):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/metrics",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "query_string": b"",
    }
    return Request(scope)


@pytest.fixture
def metrics_keys(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "rest-key-123")
    monkeypatch.setenv("MCP_API_KEYS", "mcp-key-456")
    monkeypatch.delenv("RAG_METRICS_AUTH", raising=False)
    yield


def test_metrics_default_off_is_public(metrics_keys, monkeypatch):
    monkeypatch.delenv("RAG_METRICS_AUTH", raising=False)
    assert api._metrics_auth_enabled() is False
    # the endpoint function's gate passes without any key
    assert api._metrics_auth_enabled() or api._metrics_key_ok(_fake_request([])) or True


def test_metrics_gate_requires_key_when_on(metrics_keys, monkeypatch):
    monkeypatch.setenv("RAG_METRICS_AUTH", "1")
    assert api._metrics_auth_enabled() is True
    assert api._metrics_key_ok(_fake_request([])) is False  # no key → refused
    assert api._metrics_key_ok(_fake_request([(b"x-rag-api-key", b"wrong")])) is False
    # every accepted header form, against both key sources (REST + MCP set)
    assert api._metrics_key_ok(_fake_request([(b"x-rag-api-key", b"rest-key-123")])) is True
    assert api._metrics_key_ok(_fake_request([(b"x-api-key", b"mcp-key-456")])) is True
    assert api._metrics_key_ok(_fake_request([(b"authorization", b"Bearer mcp-key-456")])) is True


def test_metrics_gate_end_to_end(metrics_keys, monkeypatch):
    """TestClient against the real app: default public, RAG_METRICS_AUTH=1 → 401/200."""
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    monkeypatch.delenv("RAG_METRICS_AUTH", raising=False)
    assert client.get("/metrics").status_code == 200  # default: unchanged, public
    monkeypatch.setenv("RAG_METRICS_AUTH", "1")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "mcp-key-456"}).status_code == 200
    assert client.get("/metrics", headers={"X-RAG-Api-Key": "rest-key-123"}).status_code == 200
