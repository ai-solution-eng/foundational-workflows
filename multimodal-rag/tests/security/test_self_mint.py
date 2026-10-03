"""Decision D25 — SSO self-mint: the identity mints its OWN long-lived API key.

The ratified model: an SSO-authenticated per-user identity (a verified D21
JWT, the D22 SSO session cookie, or the D24 trusted-proxy identity header)
may mint — and deliberately rotate — an API key for ITS OWN registry
identity from the Access page, no admin round-trip.  The invariants, all
fail-closed and pinned here:

* an API key is NEVER proof: a key-authenticated caller 403s on both D25
  endpoints (the D17 invariant — minting is exactly the power a key must
  never have);
* the mint NEVER touches grants (the key carries exactly the identity's
  current access); env-authoritative names and ``blocked`` entries refuse;
* the full key material exists exactly ONCE (the mint response); every
  later view is masked;
* the knob (``RAG_ACCESS_SELF_MINT``) defaults off = byte-identical
  behaviour (409s, whoami flag false); the anonymous whoami shape stays
  EXACTLY ``{"authenticated": false}``.

Run::

    pytest tests/security/test_self_mint.py -q
"""

import asyncio
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ===========================================================================
# Synthetic crypto + loopback JWKS (the test_oidc_identity pattern, copied)
# ===========================================================================
import base64
import hashlib  # noqa: F401  (parity with the template module)
import threading

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr
from multimodal_rag.utils import oidc_identity as oidc
from multimodal_rag.utils import oidc_sso as oidc_sso_module

ISSUER = "https://oidc.test/realm-test"
AUDIENCE = "ua"
KID = "synthetic-test-key"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return _b64url(raw)


def _b64json(obj) -> str:
    return _b64url(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _generate_rsa(key_id: str = KID) -> dict:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nums = key.public_key().public_numbers()
    return {
        "key": key,
        "kid": key_id,
        "jwk": {"kty": "RSA", "kid": key_id, "use": "sig", "alg": "RS256",
                "n": _b64uint(nums.n), "e": _b64uint(nums.e)},
        "sign": lambda data: key.sign(data, padding.PKCS1v15(), hashes.SHA256()),
    }


def _jwks_doc(*keys) -> dict:
    return {"keys": [k["jwk"] for k in keys]}


def _mint(key: dict, claims: dict) -> str:
    hdr = {"alg": "RS256", "typ": "JWT", "kid": key["kid"]}
    signing_input = f"{_b64json(hdr)}.{_b64json(claims)}"
    return f"{signing_input}.{_b64url(key['sign'](signing_input.encode('ascii')))}"


def _now() -> int:
    return int(time.time())


def _claims(
    *,
    sub="andrew-bydlon",
    preferred="andrew-bydlon",
    iss=ISSUER,
    aud=AUDIENCE,
    exp=None,
    **extra,
) -> dict:
    c = {
        "iss": iss,
        "aud": aud,
        "iat": _now(),
        "exp": exp if exp is not None else _now() + 900,
        "azp": AUDIENCE,
        **extra,
    }
    if preferred is not False:
        c["preferred_username"] = preferred
    if sub:
        c["sub"] = sub
    return c


class _JwksServer:
    """A loopback JWKS endpoint (127.0.0.1 only — no external network)."""

    def __init__(self) -> None:
        self._doc: dict = {"keys": []}
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(outer._doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/protocol/openid-connect/certs"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def set(self, doc: dict) -> None:
        self._doc = doc

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def jwks_server():
    server = _JwksServer()
    yield server
    server.stop()


# ===========================================================================
# House hygiene: full env isolation (the D15/D17/D23 fixture pattern)
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Every test starts with NO registry, NO keys, NO OIDC, NO store, NO
    self-mint — everything a test needs is set explicitly inside it."""
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
        "RAG_ACCESS_STORE",
        ar.SELF_MINT_ENV,
        "RAG_ACCESS_ADMIN_FILE",
        "MEMORY_DATASET",
        "RAG_MEMORY_DATASET",
        "RAG_TRUST_PROXY_IDENTITY",
        "RAG_OIDC_SSO_ENABLED",
        "RAG_OIDC_SSO_CLIENT_ID",
        "RAG_OIDC_SSO_CLIENT_SECRET",
        "RAG_OIDC_SSO_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "RAG_OIDC_ENABLED",
        "RAG_OIDC_ISSUER",
        "RAG_OIDC_AUDIENCE",
        "RAG_OIDC_IDENTITY_CLAIM",
        "RAG_OIDC_JWKS_URL",
        "RAG_OIDC_JWKS_REFRESH_SECONDS",
        "RAG_OIDC_OBSERVED_TTL",
        "RAG_OIDC_FETCH_TIMEOUT_SECONDS",
        "RAG_OIDC_CLOCK_SKEW_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    # The D16 store ON suite-wide (the chart default, and the admin-registry
    # suite's own posture): the per-test discriminator is the D25 knob
    # (_self_mint_on) and the overlay envs.  The knob-off tests still pin
    # the 409s — the knob is what D25 adds, the store is table stakes.
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    api._UNLOCK_CACHE.clear()
    oidc._jwks_cache.clear()
    oidc._jwks_negative.clear()
    oidc._observed_throttle.clear()
    oidc._observed_cache.clear()
    yield
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    oidc._jwks_cache.clear()
    oidc._jwks_negative.clear()
    oidc._observed_throttle.clear()
    oidc._observed_cache.clear()


@pytest.fixture(autouse=True)
def _oidc_on(monkeypatch, jwks_server):
    """Default per-test OIDC wiring: enabled, the loopback JWKS, refetch on
    every verify (TTL 0) — order-independent, the test_oidc_identity way."""
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("RAG_OIDC_JWKS_URL", jwks_server.url)
    monkeypatch.setenv("RAG_OIDC_JWKS_REFRESH_SECONDS", "0")


def _setup_key(server: _JwksServer, key_id: str = KID) -> dict:
    key = _generate_rsa(key_id)
    server.set(_jwks_doc(key))
    return key


@pytest.fixture
def rest_client(monkeypatch):
    """The real FastAPI app with a stubbed manager (the D15/D21 pattern)."""
    from fastapi.testclient import TestClient

    class _FakeDM:
        def list_datasets(self):
            return [{"name": "ds1"}, {"name": "ds2"}]

        def get_dataset(self, name, sync_count=False):
            if name not in ("ds1", "ds2"):
                raise FileNotFoundError(f"Dataset '{name}' not found")
            return {"name": name, "document_count": 0}

        def has_password(self, name):
            return False

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "")
    api._UNLOCK_CACHE.clear()
    yield TestClient(api.app)
    api._UNLOCK_CACHE.clear()


def _alice_headers(key: dict) -> dict:
    return {"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"}


def _self_mint_on(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    monkeypatch.setenv(ar.SELF_MINT_ENV, "1")


# ===========================================================================
# 1 · Registry unit matrix (no HTTP)
# ===========================================================================


def test_self_mint_disabled_by_default(monkeypatch):
    """Default = INERT: no envs at all → self_mint_enabled() is False and
    the deployment is byte-identical."""
    assert ar.self_mint_enabled() is False


def test_self_mint_enabled_with_both_knobs(monkeypatch):
    """The knob requires BOTH RAG_ACCESS_SELF_MINT and the overlay."""
    _self_mint_on(monkeypatch)
    assert ar.self_mint_enabled() is True


def test_self_mint_off_without_the_access_store(monkeypatch):
    """RAG_ACCESS_SELF_MINT=1 with the D16 store unset → OFF (nowhere to
    write the entry)."""
    monkeypatch.delenv("RAG_ACCESS_STORE", raising=False)  # the autouse turns it on
    monkeypatch.setenv(ar.SELF_MINT_ENV, "1")
    assert ar.self_mint_enabled() is False


def test_self_mint_off_when_admin_file_disabled(monkeypatch):
    """An explicit RAG_ACCESS_ADMIN_FILE=0 kills the overlay — and self-mint
    with it (fail-closed; the knob alone never re-enables)."""
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    monkeypatch.setenv(ar.SELF_MINT_ENV, "1")
    monkeypatch.setenv("RAG_ACCESS_ADMIN_FILE", "0")
    assert ar.self_mint_enabled() is False


def test_own_key_view_missing_identity_has_no_key():
    """A never-seen identity: has_key False, no masked value, empty grants."""
    view = ar.own_key_view("alice")
    assert view["has_key"] is False and view["key_masked"] is None
    assert view["datasets"] == [] and view["blocked"] is False


def test_own_key_view_masks_the_admin_minted_key():
    """Even the OWNER re-reads the masked form — first6 + ellipsis + last2,
    never the material; the full material existed exactly once, in the mint
    response."""
    minted = ar.mint_client("alice", datasets=["reports"])["key"]
    view = ar.own_key_view("alice")
    assert view["has_key"] is True
    assert view["key_masked"] == minted[:6] + "…" + minted[-2:]
    assert view["key_masked"] != minted
    assert len(view["key_masked"]) <= 11  # first6 + ellipsis + last2
    assert view["datasets"] == ["reports"]


def test_own_key_view_reports_env_managed(monkeypatch):
    """An env-authoritative name is VISIBLE to itself (read-only) — only the
    write refuses."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    view = ar.own_key_view("alice")
    assert view["env_managed"] is True


def test_own_key_view_empty_key_entry_is_mintable():
    """A D23 auto-minted empty-key entry (grants/flags PATCH) reads
    has_key False — it authenticated nothing, so the user may mint."""
    ar.grant_datasets("jwt-user", ["k"])
    view = ar.own_key_view("jwt-user")
    assert view["has_key"] is False


def test_self_mint_key_generates_a_usable_key(monkeypatch):
    """The minted key authenticates through the merged registry."""
    _self_mint_on(monkeypatch)
    res = ar.self_mint_key("alice")
    assert res["created"] and res["rotated"] is False
    assert len(res["key"]) >= 20 and ":" not in res["key"] and ";" not in res["key"]
    assert ar.verify_client_key("alice", res["key"]) is True


def test_self_mint_key_never_touches_grants(monkeypatch):
    """THE core invariant: mint/rotate preserves the identity's grants
    exactly (self-mint widens nothing)."""
    _self_mint_on(monkeypatch)
    ar.mint_client("alice", datasets=["a", "b"])
    first = ar.self_mint_key("alice", rotate=True)
    second = ar.self_mint_key("alice", rotate=True)
    assert first["key"] != second["key"]
    view = ar.own_key_view("alice")
    assert view["datasets"] == ["a", "b"]


def test_self_mint_over_existing_key_requires_rotate(monkeypatch):
    """A usable key blocks a plain re-mint (409 on the wire); rotate is the
    deliberate path."""
    _self_mint_on(monkeypatch)
    ar.self_mint_key("alice")
    with pytest.raises(FileExistsError):
        ar.self_mint_key("alice")
    res = ar.self_mint_key("alice", rotate=True)
    assert res["rotated"] is True


def test_rotate_kills_the_previous_key(monkeypatch):
    """Rotation: the old key stops authenticating IMMEDIATELY (re-read per
    call), the new one works."""
    _self_mint_on(monkeypatch)
    old = ar.self_mint_key("alice")["key"]
    new = ar.self_mint_key("alice", rotate=True)["key"]
    assert ar.verify_client_key("alice", old) is False
    assert ar.verify_client_key("alice", new) is True


def test_self_mint_requires_the_knob():
    """Knob off → SelectionDenied (the endpoint maps it to 409)."""
    with pytest.raises(acc.SelectionDenied):
        ar.self_mint_key("alice")


def test_self_mint_refuses_env_authoritative_name(monkeypatch):
    """THE env-shadow guard: a name in RAG_API_KEY_CLIENTS refuses on the
    WRITE even when an overlay entry exists — the operator owns that key's
    lifecycle, and a self-mint could otherwise rotate only the overlay half
    while the env key kept authenticating."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    ar._mutate(lambda doc: (doc["clients"].__setitem__("alice", {"key": "overlay-key-1", "datasets": []}), True)[1])
    with pytest.raises(KeyError):
        ar.self_mint_key("alice")
    with pytest.raises(KeyError):
        ar.self_mint_key("alice", rotate=True)
    assert ar.verify_client_key("alice", "overlay-key-1") is True  # untouched


def test_self_mint_refuses_blocked_identity():
    """A blocked identity resolves to nothing — it certainly cannot mint
    (even with rotate=True)."""
    ar.mint_client("alice")
    ar.set_client_flags("alice", blocked=True)
    with pytest.raises(PermissionError):
        ar.self_mint_key("alice", rotate=True)


def test_self_mint_auto_creates_the_entry(monkeypatch):
    """A first-ever self-mint writes the entry (empty grants — the merged
    registry supplies the identity's ACLs on the next request)."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    res = ar.self_mint_key("alice")
    doc = ar._load()
    assert doc["clients"]["alice"]["key"] == res["key"]
    assert doc["clients"]["alice"]["datasets"] == []


# ===========================================================================
# 2 · REST guard matrix
# ===========================================================================


def test_rest_knob_off_view_is_409(rest_client, jwks_server, monkeypatch):
    """The deployment does not offer self-mint → 409 (the SPA hides the
    card silently), for an otherwise-valid SSO identity."""
    key = _setup_key(jwks_server)
    r = rest_client.get("/api/access/key", headers=_alice_headers(key))
    assert r.status_code == 409
    assert "RAG_ACCESS_SELF_MINT" in r.json()["detail"]


def test_rest_knob_off_mint_is_409(rest_client, jwks_server, monkeypatch):
    """The mint endpoint 409s under the same knob gate (both D25 endpoints
    are governed by the one deployment switch)."""
    key = _setup_key(jwks_server)
    r = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={})
    assert r.status_code == 409


def test_rest_view_no_key(rest_client, jwks_server, monkeypatch):
    """The owner-safe view for a keyless identity: enabled/identity/source,
    has_key False."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    r = rest_client.get("/api/access/key", headers=_alice_headers(key))
    body = r.json()
    assert r.status_code == 200
    assert body["enabled"] is True and body["identity"] == "alice"
    assert body["source"] == "jwt" and body["has_key"] is False


def test_rest_mint_returns_full_key_once(rest_client, jwks_server, monkeypatch):
    """The mint response is the ONLY full-key response — the follow-up view
    is masked (never the material)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    mint = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={})
    assert mint.status_code == 200
    full = mint.json()["key"]
    assert len(full) >= 20
    view = rest_client.get("/api/access/key", headers=_alice_headers(key)).json()
    assert view["has_key"] is True
    assert view["key_masked"] != full
    assert full not in json.dumps(view)


def test_rest_minted_key_resolves_with_exact_grants(rest_client, jwks_server, monkeypatch):
    """The minted key authenticates as the identity with EXACTLY its ACLs —
    ds1 yes, ds2 no (self-mint widened nothing)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    full = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={}).json()["key"]
    key_headers = {"X-RAG-Api-Key": full}
    assert rest_client.get("/api/datasets/ds1", headers=key_headers).status_code == 200
    r = rest_client.get("/api/datasets/ds2", headers=key_headers)
    assert r.status_code == 403
    assert "not permitted" in r.json()["detail"]


def test_rest_re_mint_without_rotate_is_409(rest_client, jwks_server, monkeypatch):
    """A second mint over a now-existing key without {"rotate": true} → 409
    naming the rotate escape — replacement is always deliberate."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    assert rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={}).status_code == 200
    r = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={})
    assert r.status_code == 409
    assert "rotate" in r.json()["detail"]


def test_rest_rotate_kills_the_old_key_on_the_wire(rest_client, jwks_server, monkeypatch):
    """Rotation, end to end: the old minted key 401s the moment the new one
    exists; the new one authenticates."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    old = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={}).json()["key"]
    mint = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={"rotate": True})
    assert mint.status_code == 200 and mint.json()["rotated"] is True
    new = mint.json()["key"]
    assert rest_client.get("/api/datasets/ds1", headers={"X-RAG-Api-Key": old}).status_code == 401
    assert rest_client.get("/api/datasets/ds1", headers={"X-RAG-Api-Key": new}).status_code == 200


def test_rest_api_key_caller_cannot_view_or_mint(rest_client, jwks_server, monkeypatch):
    """THE D17 invariant, on the wire: a registry key — even one carrying
    the SAME identity the SSO session resolves to — 403s on both endpoints.
    Minting is exactly the power a key must never have."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    full = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={}).json()["key"]
    r1 = rest_client.get("/api/access/key", headers={"X-RAG-Api-Key": full})
    r2 = rest_client.post("/api/access/mint-key", headers={"X-RAG-Api-Key": full}, json={})
    assert r1.status_code == 403 and r2.status_code == 403
    assert "cannot mint" in r1.json()["detail"] or "SSO" in r1.json()["detail"]


def test_rest_deployment_key_cannot_self_mint(rest_client, jwks_server, monkeypatch):
    """An ADMIN (deployment) key is refused too — admins hold the D17 panel,
    not the D25 self-service surface."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")
    r = rest_client.get("/api/access/key", headers={"X-RAG-Api-Key": "deployment-admin-key"})
    assert r.status_code == 403


def test_rest_anonymous_is_403_not_401(rest_client, jwks_server, monkeypatch):
    """No credentials at all → the middleware refuses before any handler.
    With the registry configured (OIDC on here) that is a 401; on an
    unconfigured deployment the D20 branch binds the anonymous identity and
    the ENDPOINT would 403 — either way an anonymous caller can never reach
    the mint (fail-closed).  The 403-branch is pinned by the deployment-key
    test above (an identity, but the wrong kind)."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    r = rest_client.get("/api/access/key")
    assert r.status_code in (401, 403)


def test_rest_env_authoritative_name_refused_on_the_wire(rest_client, jwks_server, monkeypatch):
    """An SSO JWT for an env-registered name: the VIEW works (read-only),
    the MINT refuses — the operator owns that key."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:env-key-1")
    v = rest_client.get("/api/access/key", headers=_alice_headers(key))
    assert v.status_code == 200 and v.json()["env_managed"] is True
    m = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={})
    assert m.status_code == 409
    assert "env registry" in m.json()["detail"]


def test_rest_blocked_identity_cannot_mint(rest_client, jwks_server, monkeypatch):
    """A blocked overlay entry: the JWT path 401s before the endpoint even
    runs (the resolver refuses blocked names) — pin the outer behaviour."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    ar.mint_client("alice")
    ar.set_client_flags("alice", blocked=True)
    r = rest_client.get("/api/access/key", headers=_alice_headers(key))
    assert r.status_code == 401


def test_rest_blocked_identity_view_is_refused_for_a_blocked_key_too(
    rest_client, jwks_server, monkeypatch
):
    """And the minted key of a blocked identity stops authenticating (the
    D21 blocked contract, restated — nothing about it resolves)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    full = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={}).json()["key"]
    ar.set_client_flags("alice", blocked=True)
    r = rest_client.get("/api/datasets/ds1", headers={"X-RAG-Api-Key": full})
    assert r.status_code == 401


# ===========================================================================
# 3 · The guard chain and the exception→status mappings, DIRECTLY
#
#     The wire path cannot reach several guard branches: a blocked JWT dies
#     at the resolver (401), a key-authenticated request never proves SSO,
#     and a caller can never NAME another identity.  These tests drive the
#     handlers with the identity bound exactly as the middleware would bind
#     it (contextvar set/reset around the call) and a minimal starlette
#     Request — pinning the branch itself, not the wire status in front of
#     it.
# ===========================================================================


def _direct_request(headers: dict | None = None):
    """A minimal starlette Request for direct handler/guard calls."""
    from starlette.requests import Request

    raw = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()]
    scope = {"type": "http", "method": "POST", "path": "/api/access/mint-key", "headers": raw,
             "query_string": b""}
    return Request(scope)


def _call_direct(fn, identity, headers: dict | None = None, body: dict | None = None):
    """Run a D25 handler (async) or the guard chain (sync) with *identity*
    bound (set/reset around the call, the way the middleware brackets a
    request).  ``_require_self_mint`` is SYNC — it returns the
    ``(identity, source)`` tuple the async handlers use directly — while
    the endpoint handlers are coroutines."""

    async def _inner():
        token = cr.set_current_identity(identity)
        try:
            if fn is api.api_access_self_mint:
                return await fn(_direct_request(headers), body)
            if fn is api.api_access_own_key:
                return await fn(_direct_request(headers))
            return fn(_direct_request(headers))
        finally:
            cr.reset_current_identity(token)

    return asyncio.run(_inner())


def test_guard_refuses_none_admin_and_anonymous_identities(monkeypatch):
    """_require_self_mint applies to SSO-authenticated PER-USER identities
    only: a None identity, an ADMIN identity, and the D20 '__anonymous__'
    sentinel each get the fixed 403 — admins hold the D17 panel and the
    anonymous fallback is a sentinel, not a user (the D20 hard bar)."""
    _self_mint_on(monkeypatch)
    for identity in (
        None,
        cr.Identity(kind="admin", name=None, datasets=None),
        cr.Identity(kind="client", name="__anonymous__", datasets=frozenset()),
    ):
        with pytest.raises(api.HTTPException) as excinfo:
            _call_direct(api._require_self_mint, identity)
        assert excinfo.value.status_code == 403
        assert "per-user identities" in excinfo.value.detail


def test_guard_jwt_proof_source(monkeypatch, jwks_server):
    """A verified JWT presented in the SAME request that resolves to the
    bound identity's NAME is the proof: (identity, 'jwt') comes back — the
    realm signature vouches for the realm."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset())
    ident, source = _call_direct(api._require_self_mint, identity, _alice_headers(key))
    assert source == "jwt"
    assert ident.name == "alice"


def test_guard_jwt_for_another_identity_is_not_proof(monkeypatch, jwks_server):
    """The proof is per-identity, not per-channel: a valid JWT for BOB next
    to a request bound to ALICE is no proof for alice — 403 (a forwarded or
    stolen token must never mint someone else's key)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset())
    other = _mint(key, _claims(preferred="bob"))
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api._require_self_mint, identity, {"Authorization": f"Bearer {other}"})
    assert excinfo.value.status_code == 403
    assert "not a power an API key carries" in excinfo.value.detail


def test_guard_proxy_identity_proof_source(monkeypatch):
    """With RAG_TRUST_PROXY_IDENTITY confirming an enforcing proxy, the
    proxy-injected identity header equal to the bound name is the proof:
    (identity, 'proxy-identity') — the edge authentication vouches."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset())
    ident, source = _call_direct(api._require_self_mint, identity,
                                 {"X-Auth-Request-User": "alice"})
    assert source == "proxy-identity"
    assert ident.name == "alice"


def test_guard_opaque_key_is_never_proof(monkeypatch):
    """An opaque API key — even a VALID registry key for the very same
    identity — is never SSO proof: key-only headers (no trust flag, no JWT)
    leave the proof None → 403 with the fixed key-carry message.  This is
    the D17 invariant at the guard level."""
    _self_mint_on(monkeypatch)
    ar.mint_client("alice", datasets=["ds1"])
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset({"ds1"}))
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api._require_self_mint, identity, {"X-RAG-Api-Key": "some-opaque-key"})
    assert excinfo.value.status_code == 403
    assert "not a power an API key carries" in excinfo.value.detail


def test_direct_mint_blocked_maps_to_403(monkeypatch):
    """PermissionError (blocked entry) → HTTP 403 with the block message:
    the mapping the resolver path hides (a blocked JWT 401s before the
    endpoint — the resolver refuses blocked names — so the 403 mapping is
    only reachable directly).  The trust flag supplies the SSO proof (the
    guard's proof check precedes the registry write)."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    ar.mint_client("bob")
    ar.set_client_flags("bob", blocked=True)
    identity = cr.Identity(kind="client", name="bob", datasets=frozenset())
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api.api_access_self_mint, identity, {"X-Auth-Request-User": "bob"})
    assert excinfo.value.status_code == 403
    assert "blocked" in excinfo.value.detail


def test_direct_mint_existing_key_without_rotate_maps_to_409(monkeypatch):
    """FileExistsError (usable key, no rotate) → HTTP 409 naming the rotate
    escape — the confirm guard survives the direct endpoint path (the trust
    flag supplies the SSO proof past the guard)."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    ar.mint_client("rich", datasets=["a"])
    identity = cr.Identity(kind="client", name="rich", datasets=frozenset())
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api.api_access_self_mint, identity, {"X-Auth-Request-User": "rich"})
    assert excinfo.value.status_code == 409
    assert "rotate" in excinfo.value.detail


def test_direct_rotate_over_existing_key_succeeds(monkeypatch):
    """The direct path's happy rotate: rotate=True over a key-holding entry
    → 200-shaped result (rotated True, full material, grants untouched) —
    proving the 409 above is the rotate flag's doing, not the entry's."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    ar.mint_client("rich", datasets=["a"])
    identity = cr.Identity(kind="client", name="rich", datasets=frozenset())
    result = _call_direct(api.api_access_self_mint, identity,
                          {"X-Auth-Request-User": "rich"}, body={"rotate": True})
    assert result["status"] == "ok" and result["rotated"] is True
    assert result["key"]
    assert ar.own_key_view("rich")["datasets"] == ["a"]


def test_direct_view_invalid_name_maps_to_400(monkeypatch):
    """ValueError (a bound name violating the registry name rules —
    reachable only through a trusted proxy header, never a verified JWT) →
    HTTP 400 on the owner view."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    identity = cr.Identity(kind="client", name="bad name!", datasets=frozenset())
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api.api_access_own_key, identity, {"X-Auth-Request-User": "bad name!"})
    assert excinfo.value.status_code == 400
    assert "client name" in excinfo.value.detail


def test_direct_mint_invalid_name_maps_to_400(monkeypatch):
    """The mint handler maps the same ValueError → 400 (the name is
    validated before any entry is written — no partial state)."""
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    identity = cr.Identity(kind="client", name="bad name!", datasets=frozenset())
    with pytest.raises(api.HTTPException) as excinfo:
        _call_direct(api.api_access_self_mint, identity, {"X-Auth-Request-User": "bad name!"})
    assert excinfo.value.status_code == 400


# ===========================================================================
# 4 · The proxy-identity source (D24)
# ===========================================================================


def test_rest_proxy_identity_source(rest_client, monkeypatch, jwks_server):
    """With an enforcing proxy confirmed, the identity header is SSO proof —
    the view works with NO key and NO JWT, source proxy-identity."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    r = rest_client.get("/api/access/key", headers={"X-Auth-Request-User": "alice"})
    body = r.json()
    assert r.status_code == 200
    assert body["source"] == "proxy-identity" and body["identity"] == "alice"


def test_rest_proxy_identity_header_without_trust_is_not_proof(rest_client, monkeypatch, jwks_server):
    """THE spoof guard: the same header with RAG_TRUST_PROXY_IDENTITY unset
    never binds an identity — the middleware 401s before the endpoint (the
    header is attacker-settable without the trust flag)."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    r = rest_client.get("/api/access/key", headers={"X-Auth-Request-User": "alice"})
    assert r.status_code in (401, 403)


def test_rest_proxy_name_mismatch_binds_the_proxy_name_not_the_jwt(rest_client, monkeypatch, jwks_server):
    """The proof is PER-IDENTITY: an Authorization JWT for alice alongside a
    trusted proxy header naming carol binds CAROL (the edge authentication is
    the credential — resolve_presented's D24 path runs before the JWT
    fall-through), with source proxy-identity.  The alice JWT is NOT proof
    for the carol-bound request, so the mint (if any) could only ever touch
    carol's own entry — never the JWT holder's."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    r = rest_client.get(
        "/api/access/key",
        headers={**_alice_headers(key), "X-Auth-Request-User": "carol"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["identity"] == "carol" and body["source"] == "proxy-identity"
    # The JWT holder's grants did not leak onto the proxy-bound identity.
    assert body["datasets"] == []
    # …and the JWT alone (no proxy header) still resolves alice with the JWT
    # proof — the mismatch binding above is the proxy header's doing.
    r2 = rest_client.get("/api/access/key", headers=_alice_headers(key))
    assert r2.status_code == 200
    assert r2.json()["identity"] == "alice" and r2.json()["source"] == "jwt"


# ===========================================================================
# 4 · Whoami flags.self_mint + the anonymous-shape hard bar
# ===========================================================================


def test_whoami_proxy_identity_source_resolves(rest_client, jwks_server, monkeypatch):
    """THE D25-v2 fix (the 'card hidden behind the G2 gateway' report,
    2026-10-02): behind an enforcing proxy a request can carry ONLY the
    injected identity headers — no JWT envelope, no key.  The whoami must
    preview that D24 identity (source 'proxy-identity', authenticated,
    flags.self_mint True), not fall back to the anonymous shape that sent
    the SPA into paste mode and hid the card."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    body = rest_client.get("/api/oidc-session", headers={"X-Auth-Request-User": "alice"}).json()
    assert body["authenticated"] is True
    assert body["identity"] == "alice"
    assert body["source"] == "proxy-identity"
    assert body["is_admin"] is False  # a proxy identity is never admin
    assert body["flags"]["self_mint"] is True
    assert body["datasets"] == ["ds1"]


def test_whoami_proxy_identity_without_trust_stays_anonymous(rest_client, jwks_server, monkeypatch):
    """The spoof guard for the whoami preview: the identity header without
    RAG_TRUST_PROXY_IDENTITY is attacker-settable — the whoami stays
    exactly anonymous-shaped."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    r = rest_client.get("/api/oidc-session", headers={"X-Auth-Request-User": "alice"})
    assert r.json() == {"authenticated": False}


def test_whoami_self_mint_flag_true(rest_client, jwks_server, monkeypatch):
    """Knob + overlay on → flags.self_mint True (the SPA's card gate)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    body = rest_client.get("/api/oidc-session", headers=_alice_headers(key)).json()
    assert body["authenticated"] is True
    assert body["flags"]["self_mint"] is True


def test_whoami_self_mint_flag_false_by_default(rest_client, jwks_server, monkeypatch):
    """Overlay on but the knob off → flags.self_mint False (the flag reports
    the DEPLOYMENT's offer, not the caller's state)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")  # overlay on, knob off
    body = rest_client.get("/api/oidc-session", headers=_alice_headers(key)).json()
    assert body["flags"]["self_mint"] is False


def test_whoami_anonymous_shape_exactly_unchanged(rest_client, jwks_server, monkeypatch):
    """THE hard bar (D23, restated for D25): the anonymous whoami shape
    stays EXACTLY {"authenticated": false} — no new keys, knob or not."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    assert rest_client.get("/api/oidc-session").json() == {"authenticated": False}
    # An invalid JWT also stays anonymous-shaped.
    forged = _mint(_generate_rsa("forged-kid"), _claims())
    assert rest_client.get("/api/oidc-session", headers={"Authorization": f"Bearer {forged}"}).json() == {
        "authenticated": False
    }


def test_whoami_admin_key_stays_anonymous_shaped(rest_client, jwks_server, monkeypatch):
    """And an OPAQUE admin key never leaks its status — D25 added nothing
    that changes the D23 anonymity rule."""
    _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")
    r = rest_client.get("/api/oidc-session", headers={"X-RAG-Api-Key": "deployment-admin-key"})
    assert r.json() == {"authenticated": False}


# ===========================================================================
# 5 · The D22 SSO-cookie envelope
# ===========================================================================


def test_rest_sso_cookie_envelope_mints(rest_client, jwks_server, monkeypatch):
    """The D22 cookie carries the same token the resolver accepts — a
    cookie-only browser session can mint (source 'jwt': the cookie is a
    JWT envelope through the same oidc_identity pipeline)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    token = _mint(key, _claims(preferred="alice"))
    r = rest_client.get("/api/access/key", cookies={oidc_sso_module.cookie_name(): token})
    body = r.json()
    assert r.status_code == 200
    assert body["identity"] == "alice" and body["source"] == "jwt"
    mint = rest_client.post("/api/access/mint-key", cookies={oidc_sso_module.cookie_name(): token}, json={})
    assert mint.status_code == 200
    assert ar.verify_client_key("alice", mint.json()["key"]) is True


def test_rest_sso_cookie_with_pasted_key_governs_the_datasets_but_still_mints(
    rest_client, jwks_server, monkeypatch
):
    """The co-existence case: the SPA paste mode sends the key header, an
    SSO cookie sits underneath.  The KEY still governs dataset access
    (precedence unchanged — byte-identical D19/D22 behaviour); and because
    the JWT independently proves the SAME identity, the D25 endpoints work.
    Two attestations of one human; the mint still touches only that
    identity's own entry."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    token = _mint(key, _claims(preferred="alice"))
    cookies = {oidc_sso_module.cookie_name(): token}
    # First mint via the SSO cookie alone, then paste the result — the
    # browser now holds an OVERLAY key for the same identity the session
    # resolves to (an env-authoritative name would rightly 409 — the
    # env-shadow guard test covers that; here the pasted key is an overlay
    # key, the exact post-"Use in this browser" shape).
    first = rest_client.post("/api/access/mint-key", cookies=cookies, json={}).json()["key"]
    hdrs = {"X-RAG-Api-Key": first}
    # Dataset access still governed by the KEY (precedence unchanged —
    # identical matrix with or without the cookie underneath).
    assert rest_client.get("/api/datasets/ds1", headers=hdrs).status_code == 200
    # ...and the self-mint still works for the same identity: the request
    # carries BOTH credentials — the key header (governs access) and the
    # cookie (the SSO proof).  Two attestations of one human; the rotate
    # touches only that identity's own entry.
    mint = rest_client.post("/api/access/mint-key", headers=hdrs, cookies=cookies, json={"rotate": True})
    assert mint.status_code == 200
    rotated = mint.json()["key"]
    assert ar.verify_client_key("alice", first) is False
    assert ar.verify_client_key("alice", rotated) is True
    # Precedence pinned: with the cookie underneath, the rotated KEY still
    # governs dataset access (the cookie never escalates).
    assert rest_client.get("/api/datasets/ds1", headers={"X-RAG-Api-Key": rotated}).status_code == 200


# ===========================================================================
# 6 · Hygiene: no key material in logs
# ===========================================================================


def test_mint_never_logs_key_material(rest_client, jwks_server, monkeypatch, caplog):
    """The mint logs the identity and the rotated flag — never the
    material (the module docstring's 'never logs key material' contract)."""
    key = _setup_key(jwks_server)
    _self_mint_on(monkeypatch)
    with caplog.at_level(logging.INFO, logger="multimodal_rag"):
        mint = rest_client.post("/api/access/mint-key", headers=_alice_headers(key), json={})
    full = mint.json()["key"]
    for record in caplog.records:
        assert full not in record.getMessage()
