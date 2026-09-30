"""Decision D21 — OIDC JWT as a SECOND credential for the same per-user
registry identity.

The ratified model: alongside the opaque registry keys (D15/D17), a client
may present a signed OIDC JWT (``Authorization: Bearer <jwt>``).  The JWT is
verified (RS256 only — alg-confusion guard) against a cached JWKS, its
``iss``/``aud``(azp)/``exp``/``nbf`` are validated, and the identity claim
(default ``preferred_username``, fallback ``sub``) resolves THROUGH THE SAME
per-user registry: same ``RAG_DATASET_ACLS`` grants, same admin overlay
(including the ``oidc`` alias and ``blocked`` fields), same
``key:<name>`` throttle/unlock identity.  A JWT NEVER resolves to admin;
a user absent from the ACLs gets an EMPTY dataset set (fail-closed), and a
blocked overlay entry resolves to ``None``.

Every RSA key, JWKS document, and token in this module is SYNTHETICALLY
generated in-test (``cryptography`` + a loopback ``http.server`` JWKS
endpoint on 127.0.0.1).  No external network, no real issuer, no real token
material — ``iss`` is claim-compared, never fetched; only the JWKS URL is
fetched (loopback only).

Run::

    pytest tests/security/test_oidc_identity.py -q
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr
import multimodal_rag.utils.oidc_sso as oidc_sso_module
from multimodal_rag.utils import oidc_identity as oidc

# ===========================================================================
# Synthetic crypto: keys, JWK encoding, token minting (no jwt lib needed)
# ===========================================================================

#: A fixed fake https issuer — ``iss`` is COMPARED to ``RAG_OIDC_ISSUER``,
#: never fetched, so a non-existent https URL is safe; only the JWKS URL is
#: fetched (always 127.0.0.1 here).
ISSUER = "https://oidc.test/realm-test"
AUDIENCE = "ua"
KID = "synthetic-test-key"


def _b64url(data: bytes) -> str:
    """Unpadded base64url (RFC 7515 §2 — no ``=`` padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64uint(value: int) -> str:
    """An unsigned integer as a Base64urlUInt (RFC 7518 §6.3.1.1): big-endian
    minimal octets, unpadded base64url."""
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return _b64url(raw)


def _b64json(obj) -> str:
    return _b64url(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _generate_rsa(key_id: str = KID) -> dict:
    """Generate one synthetic RSA keypair; return {"key", "jwk", "kid"}."""
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


def _mint(key: dict, claims: dict, header: dict | None = None, sign=None) -> str:
    """An RS256 compact JWS: header.claims.signature (base64url, unpadded).

    ``header`` may override alg/kid (the alg-confusion tests) and ``sign``
    replaces the RSA signer (the HS256 confusion test signs with HMAC).
    """
    hdr = {"alg": "RS256", "typ": "JWT", "kid": key["kid"]}
    if header:
        hdr.update(header)
    signing_input = f"{_b64json(hdr)}.{_b64json(claims)}"
    if sign is None:
        signature = key["sign"](signing_input.encode("ascii"))
    else:
        signature = sign(signing_input.encode("ascii"))
    return f"{signing_input}.{_b64url(signature)}"


def _now() -> int:
    return int(time.time())


def _claims(
    *,
    sub="andrew-bydlon",
    preferred="andrew-bydlon",  # False = omit the claim entirely
    iss=ISSUER,
    aud=AUDIENCE,
    exp=None,
    nbf=None,
    azp=None,
    **extra,
) -> dict:
    c = {
        "iss": iss,
        "aud": aud,
        "exp": exp if exp is not None else _now() + 600,
        "iat": _now() - 5,
    }
    if sub is not None:
        c["sub"] = sub
    if preferred is not False:
        c["preferred_username"] = preferred
    if nbf is not None:
        c["nbf"] = nbf
    if azp is not None:
        c["azp"] = azp
    c.update(extra)
    return c


class _JwksServer:
    """A minimal local JWKS endpoint: 127.0.0.1, ephemeral port, daemon
    thread.  Tests swap the served document / status code at will."""

    def __init__(self):
        holder = self
        self._doc: dict = {"keys": []}
        self._status = 200
        self.hits = 0  # served GET count — proves fetch-frequency guarantees

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                holder.hits += 1
                body = json.dumps(holder._doc).encode("utf-8")
                self.send_response(holder._status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # silence the test log
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/protocol/openid-connect/certs"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def set(self, doc: dict) -> None:
        self._doc = doc

    def set_status(self, status: int) -> None:
        self._status = status

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(scope="module")
def jwks_server():
    server = _JwksServer()
    yield server
    server.stop()


# ===========================================================================
# House hygiene: full env isolation (the D15/D17 fixture pattern)
# ===========================================================================

_OIDC_ENV_NAMES = (
    "RAG_OIDC_ENABLED",
    "RAG_OIDC_ISSUER",
    "RAG_OIDC_AUDIENCE",
    "RAG_OIDC_IDENTITY_CLAIM",
    "RAG_OIDC_JWKS_URL",
    "RAG_OIDC_JWKS_REFRESH_SECONDS",
    "RAG_OIDC_OBSERVED_TTL",
    "RAG_OIDC_FETCH_TIMEOUT_SECONDS",
    "RAG_OIDC_CLOCK_SKEW_SECONDS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Every test starts with NO registry, NO keys, NO OIDC, NO store —
    everything a test needs is set explicitly inside it."""
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
        "RAG_ACCESS_STORE",
        "MEMORY_DATASET",
        "RAG_MEMORY_DATASET",
        "RAG_MCP_SHARED_UNLOCK",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in _OIDC_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    api._UNLOCK_CACHE.clear()
    # The oidc_identity module caches per process; a test must never inherit
    # another test's JWKS/negative-cache/throttle state (mirrors the
    # ar._mtime_cache.clear() house pattern).
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
    """Default per-test OIDC wiring: enabled, the loopback JWKS, the fake
    https issuer, refetch-on-every-verify (TTL 0) so tests are order-
    independent regardless of how the implementation caches.  Cache-behaviour
    tests below override the TTL and mint with their OWN issuer (so a cache
    keyed by URL or issuer alike cannot leak between tests)."""
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("RAG_OIDC_JWKS_URL", jwks_server.url)
    monkeypatch.setenv("RAG_OIDC_JWKS_REFRESH_SECONDS", "0")


def _setup_key(server: _JwksServer, key_id: str = KID) -> dict:
    """Generate a keypair, publish it in the server's JWKS, return it."""
    key = _generate_rsa(key_id)
    server.set(_jwks_doc(key))
    return key


# ===========================================================================
# 1 · Key/JWT helpers and format detection
# ===========================================================================


def test_b64uint_is_unpadded_and_roundtrips():
    for value in (65537, 0, 1, (1 << 2047) | 0x3FF):
        enc = _b64uint(value)
        assert "=" not in enc
        raw = base64.urlsafe_b64decode(enc + "=" * (-len(enc) % 4))
        assert int.from_bytes(raw, "big") == value
    assert _b64uint(65537) == "AQAB"  # the canonical RSA exponent encoding


def test_jwt_format_detection():
    assert oidc.is_jwt_format("aaaa.bbbb.cccc") is True  # three b64url segments
    assert oidc.is_jwt_format("alice-key") is False  # an opaque registry key
    assert oidc.is_jwt_format("") is False
    assert oidc.is_jwt_format("a.b") is False  # two segments
    assert oidc.is_jwt_format("a.b.c.d") is False  # four segments
    assert oidc.is_jwt_format("!!$.###.$$$") is False  # not base64url chars
    # And the same strings are unconditionally unverifiable.
    for s in ("alice-key", "", "a.b", "!!$.###.$$$", "aaaa.bbbb"):
        assert oidc.verify_and_decode(s) is None, s


# ===========================================================================
# 2 · Verification: signature, iss, aud/azp, exp/nbf, alg guard
# ===========================================================================


def test_valid_token_verifies_and_returns_claims(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims())
    claims = oidc.verify_and_decode(token)
    assert claims is not None
    assert claims["sub"] == "andrew-bydlon"
    assert claims["iss"] == ISSUER
    assert claims["aud"] == AUDIENCE


def test_wrong_signing_key_rejected(monkeypatch, jwks_server):
    published = _setup_key(jwks_server)
    other = _generate_rsa("other-key")
    token = _mint(other, _claims())
    assert oidc.verify_and_decode(token) is None
    # once the attacker's key IS published, the same token verifies — the
    # rejection above is the signature check, not an accident.
    jwks_server.set(_jwks_doc(published, other))
    assert oidc.verify_and_decode(token) is not None


def test_iss_mismatch_rejected(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(iss="https://evil.example/realm"))
    assert oidc.verify_and_decode(token) is None


def test_expired_token_rejected(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(exp=_now() - 3600))
    assert oidc.verify_and_decode(token) is None


def test_expired_within_clock_skew_still_accepted(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(exp=_now() - 30))  # skew window is 60s
    assert oidc.verify_and_decode(token) is not None


def test_nbf_in_future_rejected(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(nbf=_now() + 600))
    assert oidc.verify_and_decode(token) is None


def test_aud_mismatch_rejected(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    assert oidc.verify_and_decode(_mint(key, _claims(aud="other-app"))) is None


def test_azp_mismatch_rejected(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(azp="other-client"))
    assert oidc.verify_and_decode(token) is None


def test_aud_list_containing_audience_verifies(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(aud=[AUDIENCE, "other-app"]))
    assert oidc.verify_and_decode(token) is not None


def test_alg_none_token_rejected(monkeypatch, jwks_server):
    """The classic JWS 'alg: none' downgrade — an UNSIGNED token must never
    verify, whatever its claims say.  (A non-empty dummy signature keeps the
    token well-formed so the ALG GUARD is what rejects it, not the shape.)"""
    key = _setup_key(jwks_server)
    token = _mint(key, _claims(), header={"alg": "none"}, sign=lambda data: b"unsigned")
    assert oidc.is_jwt_format(token)
    assert oidc.verify_and_decode(token) is None


def test_alg_hs256_confusion_rejected(monkeypatch, jwks_server):
    """The alg-confusion attack: sign with HS256 using the (public) RSA key
    material as the HMAC secret.  The JWKS only ever publishes RSA keys, so
    an HS256 header must be refused BEFORE any key lookup."""
    key = _setup_key(jwks_server)

    secret = key["jwk"]["n"].encode("ascii")  # public material as HMAC key

    def _hmac_sign(data: bytes) -> bytes:
        return hmac.new(secret, data, hashlib.sha256).digest()

    token = _mint(key, _claims(), header={"alg": "HS256", "kid": KID}, sign=_hmac_sign)
    assert oidc.verify_and_decode(token) is None


def test_garbage_signed_payload_rejected(monkeypatch, jwks_server):
    """Three well-formed segments whose payload is not a valid claims set."""
    _setup_key(jwks_server)
    assert oidc.verify_and_decode("aaaa.bbbb.cccc") is None


def test_issuer_unset_while_enabled_is_inert(monkeypatch, capsys):
    """Enabled WITHOUT RAG_OIDC_ISSUER: the resolver is inert — ``None``
    everywhere, no exception — and the misconfiguration screams once at
    startup (the implementation prints the warning via
    ``warn_if_misconfigured`` rather than logging it; the behaviour, not the
    log plumbing, is what is pinned)."""
    monkeypatch.delenv("RAG_OIDC_ISSUER", raising=False)
    assert oidc.oidc_enabled() is False
    key = _generate_rsa()
    token = _mint(key, _claims())
    assert oidc.verify_and_decode(token) is None
    assert oidc.resolve_jwt(token) is None
    assert cr.resolve_presented([token]) is None
    # The startup warning fires exactly for this state.
    assert oidc.warn_if_misconfigured("unit-test") is True
    assert "RAG_OIDC_ISSUER" in capsys.readouterr().out


# ===========================================================================
# 3 · Identity resolution through the shared registry
# ===========================================================================


def test_resolve_jwt_identity_datasets_and_client_id(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1,ds2")
    ident = oidc.resolve_jwt(_mint(key, _claims()))
    assert ident is not None
    assert ident == cr.Identity(kind="client", name="andrew-bydlon", datasets=frozenset({"ds1", "ds2"}))
    assert ident.client_id == "key:andrew-bydlon"  # the shared unlock-cache key
    assert not ident.is_admin


def test_resolve_jwt_never_admin(monkeypatch, jwks_server):
    """Even with the deployment key as the identity claim's name and '*' in
    the ACL — a JWT is a client credential, full stop."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "deployment-key:*")
    monkeypatch.setenv("RAG_API_KEY", "deployment-key")
    ident = oidc.resolve_jwt(_mint(key, _claims(preferred="deployment-key")))
    assert ident is not None and not ident.is_admin
    assert ident.name == "deployment-key"


def test_resolve_jwt_unknown_user_empty_datasets_fail_closed(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "someoneelse:ds1")
    ident = oidc.resolve_jwt(_mint(key, _claims()))
    assert ident is not None, "an authenticated user with no ACL entry is still an identity"
    assert ident.kind == "client" and ident.name == "andrew-bydlon"
    assert ident.datasets == frozenset()
    assert cr.dataset_allowed(ident, "ds1") is False


def test_resolve_jwt_disabled_is_none(monkeypatch, jwks_server):
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    key = _setup_key(jwks_server)
    assert oidc.resolve_jwt(_mint(key, _claims())) is None


def test_identity_claim_override_to_sub(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_OIDC_IDENTITY_CLAIM", "sub")
    monkeypatch.setenv(cr.ACLS_ENV, "upstream-sub-123:ds9")
    token = _mint(key, _claims(sub="upstream-sub-123", preferred="andrew-bydlon"))
    ident = oidc.resolve_jwt(token)
    assert ident is not None and ident.name == "upstream-sub-123"
    assert ident.datasets == frozenset({"ds9"})


def test_sub_fallback_when_preferred_username_absent(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1")
    token = _mint(key, _claims(preferred=False))
    ident = oidc.resolve_jwt(token)
    assert ident is not None and ident.name == "andrew-bydlon"
    assert ident.datasets == frozenset({"ds1"})


def test_oidc_status_reports_state(monkeypatch, jwks_server):
    status = oidc.oidc_status()
    assert isinstance(status, dict)


# ===========================================================================
# 4 · D19 delegation precedence with a JWT candidate
# ===========================================================================


def test_delegation_jwt_in_authorization_loses_to_opaque_x_api_key(monkeypatch, jwks_server):
    """The gateway topology: Authorization carries a JWT (the gateway's own
    auth), X-API-Key carries the caller's delegated opaque key.  The DELEGATED
    key wins — the JWT never resolves in delegation mode via Authorization."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;andrew-bydlon:andrew-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;andrew-bydlon:ds2")
    jwt_token = _mint(key, _claims())
    ident = cr.resolve_presented(
        [jwt_token, "alice-key"], ["authorization", "x-api-key"]
    )
    assert ident is not None
    assert ident.name == "alice", "the delegated X-API-Key identity must win"
    assert cr.dataset_allowed(ident, "ds1") and not cr.dataset_allowed(ident, "ds2")


def test_delegation_jwt_via_x_api_key_resolves(monkeypatch, jwks_server):
    """The mirror: the JWT is what the caller delegated (X-API-Key), the
    Authorization header carries an opaque platform key.  Now the JWT
    resolves — the same D19 rule, symmetric for the JWT credential."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;andrew-bydlon:ds2")
    jwt_token = _mint(key, _claims())
    ident = cr.resolve_presented(
        ["alice-key", jwt_token], ["authorization", "x-api-key"]
    )
    assert ident is not None
    assert ident.name == "andrew-bydlon"
    assert ident.datasets == frozenset({"ds2"})


def test_single_authorization_jwt_resolves_without_delegation(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1,ds2")
    ident = cr.resolve_presented([_mint(key, _claims())])
    assert ident is not None and ident.name == "andrew-bydlon"
    assert ident.datasets == frozenset({"ds1", "ds2"})


def test_admin_opaque_key_wins_in_non_delegation_mode(monkeypatch, jwks_server):
    """Legacy single-header order: an admin deployment key presented alongside
    a JWT still resolves as admin (JWT fall-through happens only after the
    admin check misses)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1")
    ident = cr.resolve_presented(["deployment-key", _mint(key, _claims())])
    assert ident is not None and ident.is_admin


def test_delegation_unknown_x_api_key_does_not_fall_through_to_jwt(monkeypatch, jwks_server):
    """D19's no-escalation rule extends to JWTs: an unknown X-API-Key resolves
    alone → None; the co-forwarded Authorization JWT must NOT rescue it."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1")
    ident = cr.resolve_presented(
        [_mint(key, _claims()), "unknown-key"], ["authorization", "x-api-key"]
    )
    assert ident is None


def test_jwt_in_authorization_alongside_x_api_key_never_resolves_via_oidc(monkeypatch, jwks_server):
    """The headline D21/D19 interaction: a JWT riding Authorization NEXT to an
    X-API-Key must not resolve via OIDC at all (only x-api-key-sourced
    candidates may use the OIDC path in delegation mode)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;andrew-bydlon:ds2")
    # unknown x-api-key + valid Authorization JWT → None (pinned above); and
    # a KNOWN opaque x-api-key resolves as ITS OWN identity, not the JWT's.
    ident = cr.resolve_presented(
        [_mint(key, _claims()), "alice-key"], ["authorization", "x-api-key"]
    )
    assert ident is not None and ident.name == "alice"


# ===========================================================================
# 5 · registry_configured() engagement
# ===========================================================================


def test_registry_configured_with_oidc_enabled_alone(monkeypatch):
    """OIDC enabled (issuer set) is enough to engage per-identity enforcement
    — no RAG_API_KEY_CLIENTS, no overlay, no store."""
    assert cr.registry_configured() is True


def test_registry_configured_all_disabled(monkeypatch):
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    assert cr.registry_configured() is False


def test_registry_configured_reads_env_per_request(monkeypatch):
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    assert cr.registry_configured() is False
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    assert cr.registry_configured() is True
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    assert cr.registry_configured() is False


# ===========================================================================
# 6 · Overlay integration: blocked entries and the "oidc" alias
# ===========================================================================


def _write_overlay(entry: dict) -> None:
    """Hand-write the D17 overlay file (mint_client does not know the D21
    fields) and drop the mtime cache so it is re-read."""
    from pathlib import Path

    path = Path(os.environ["DATA_PATH"]) / "access" / "clients.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"clients": {entry["name"]: entry}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    ar._mtime_cache.clear()


def test_overlay_blocked_user_jwt_denied(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    _write_overlay(
        {"name": "alice", "key": "blocked-alice-opaque-key", "datasets": ["ds1"], "blocked": True}
    )
    token = _mint(key, _claims(preferred="alice"))
    assert oidc.resolve_jwt(token) is None
    assert cr.resolve_presented([token]) is None, "blocked must hold on the middleware path too"


def test_overlay_blocked_user_minted_key_denied_too(monkeypatch, jwks_server):
    """D21 (lead regression, 2026-10): ``blocked: true`` kills BOTH credentials.

    The JWT path was pinned above; this pins the opaque-key path — the
    blocked client's minted key must resolve to ``None`` (admin_registry's
    docstring has always promised "a minted key stops authenticating"; a
    runtime check found the key surviving the registry merge and keeping
    its datasets, a fail-open gap).
    """
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    _write_overlay(
        {"name": "alice", "key": "blocked-alice-opaque-key", "datasets": ["ds1"], "blocked": True}
    )
    assert cr.resolve_presented(["blocked-alice-opaque-key"]) is None, (
        "a blocked client's minted key must stop authenticating (docstring contract)"
    )
    assert cr.registry_clients() == {}, "the blocked key must not survive the registry merge"
    # Sanity: unblocking restores the key path.
    _write_overlay({"name": "alice", "key": "blocked-alice-opaque-key", "datasets": ["ds1"]})
    ident = cr.resolve_presented(["blocked-alice-opaque-key"])
    assert ident is not None and ident.name == "alice" and ident.datasets == frozenset({"ds1"})


def test_overlay_oidc_alias_resolves_jwt_to_registry_identity(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    _write_overlay(
        {
            "name": "prod-alice",
            "key": "prod-alice-opaque-key",
            "oidc": "andrew-bydlon",
            "datasets": ["ds9"],
        }
    )
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1,ds2")
    ident = oidc.resolve_jwt(_mint(key, _claims()))
    assert ident is not None
    assert ident.kind == "client" and ident.name == "prod-alice"
    assert ident.datasets == frozenset({"ds9"})
    assert ident.client_id == "key:prod-alice"
    assert cr.dataset_allowed(ident, "ds9") and not cr.dataset_allowed(ident, "ds1")


# ===========================================================================
# 7 · Observed identities
# ===========================================================================


def test_observed_identities_recorded_on_success(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1")
    oidc.resolve_jwt(_mint(key, _claims()))
    entries = [e for e in oidc.observed_identities() if e.get("name") == "andrew-bydlon"]
    assert entries, "a successful resolution must be observed"
    entry = entries[0]
    assert "first_seen" in entry and "last_seen" in entry


def test_observed_ttl_throttles_last_seen(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "andrew-bydlon:ds1")
    monkeypatch.setenv("RAG_OIDC_OBSERVED_TTL", "3600")
    token = _mint(key, _claims())

    def _entry():
        matches = [e for e in oidc.observed_identities() if e.get("name") == "andrew-bydlon"]
        assert matches
        return matches[0]

    oidc.resolve_jwt(token)
    first = _entry()
    oidc.resolve_jwt(token)
    second = _entry()
    assert second["last_seen"] == first["last_seen"], (
        "within RAG_OIDC_OBSERVED_TTL an immediate re-resolution must not bump last_seen"
    )


# ===========================================================================
# 8 · Middleware end-to-end (REST + MCP surfaces)
# ===========================================================================


@pytest.fixture
def rest_client(monkeypatch):
    """The real FastAPI app with a stubbed manager (the test_dataset_acls
    full-stack pattern)."""
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


def test_rest_valid_jwt_authenticates_and_acl_is_enforced(rest_client, monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    headers = {"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"}
    r = rest_client.get("/api/datasets/ds1", headers=headers)
    assert r.status_code == 200
    r = rest_client.get("/api/datasets/ds2", headers=headers)
    assert r.status_code == 403
    assert "not permitted" in r.json()["detail"]
    r = rest_client.get("/api/admin/health", headers=headers)
    assert r.status_code == 403
    assert "no admin access" in r.json()["detail"]


def test_rest_opaque_key_and_jwt_behave_as_one_identity(rest_client, monkeypatch, jwks_server):
    """The same registry user, two credentials — byte-identical access matrix."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    jwt_headers = {"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"}
    key_headers = {"X-RAG-Api-Key": "alice-key"}
    for headers in (jwt_headers, key_headers):
        assert rest_client.get("/api/datasets/ds1", headers=headers).status_code == 200
        assert rest_client.get("/api/datasets/ds2", headers=headers).status_code == 403
        assert rest_client.get("/api/admin/health", headers=headers).status_code == 403


def test_rest_invalid_jwt_is_401(rest_client, monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    r = rest_client.get(
        "/api/datasets/ds1", headers={"Authorization": f"Bearer {_mint(key, _claims(exp=_now() - 9999))}"}
    )
    assert r.status_code == 401


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
    """Run the MCP middleware the way production wires it (mirrors
    test_dataset_acls.py); the app body observes the bound identity."""
    messages = []
    captured = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
        captured["identity"] = cr.current_identity()
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


def _bearer(token: str):
    return [(b"authorization", f"Bearer {token}".encode())]


def test_mcp_middleware_valid_jwt_binds_identity(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    messages, captured = asyncio.run(
        _drive(_mw(), _scope(headers=_bearer(_mint(key, _claims(preferred="alice")))))
    )
    assert messages[0]["status"] == 200, "a valid JWT must not 401 on /mcp"
    ident = captured["identity"]
    assert ident is not None and ident.kind == "client" and ident.name == "alice"
    assert ident.datasets == frozenset({"ds1"})


def test_mcp_middleware_invalid_jwt_is_401(monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    messages, _ = asyncio.run(
        _drive(_mw(), _scope(headers=_bearer(_mint(key, _claims(exp=_now() - 9999)))))
    )
    assert messages[0]["status"] == 401


def test_mcp_delegation_jwt_via_x_api_key_over_bearer(monkeypatch, jwks_server):
    """On the MCP surface the D19 header rule holds with a JWT too: the
    X-API-Key candidate resolves, the co-forwarded Bearer does not."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;andrew-bydlon:ds2")
    headers = [(b"x-api-key", b"alice-key")] + _bearer(_mint(key, _claims()))
    _, captured = asyncio.run(_drive(_mw(), _scope(headers=headers)))
    assert captured["identity"] is not None and captured["identity"].name == "alice"


def test_d20_anon_behaviour_unchanged_when_oidc_disabled(monkeypatch, rest_client, jwks_server):
    """With OIDC disabled and nothing else configured, the D20 fail-closed
    default is byte-identical: REST unconfigured anon (empty listing / 403 on
    dataset paths), MCP anonymous memory identity."""
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    r = rest_client.get("/api/datasets")
    assert r.status_code == 200
    assert r.json()["datasets"] == []
    r = rest_client.get("/api/datasets/ds1")
    assert r.status_code == 403
    messages, captured = asyncio.run(_drive(_mw(), _scope()))
    assert messages[0]["status"] == 200
    ident = captured["identity"]
    assert ident is not None and ident.name == "__anonymous__"
    assert ident.datasets == frozenset()


# ===========================================================================
# 9 · Admin-minted grants apply to the JWT path
# ===========================================================================


def test_admin_mint_grants_flow_to_jwt_identity(tmp_path, monkeypatch, jwks_server):
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    res = ar.mint_client("jwt-user", datasets=["k"])
    assert res["key"]
    ident = oidc.resolve_jwt(_mint(key, _claims(preferred="jwt-user")))
    assert ident is not None and ident.name == "jwt-user"
    assert ident.datasets == frozenset({"k"})
    # …and through the middleware fall-through as well.
    ident = cr.resolve_presented([_mint(key, _claims(preferred="jwt-user"))])
    assert ident is not None and cr.dataset_allowed(ident, "k")


# ===========================================================================
# 10 · JWKS robustness: fetch failure, recovery, unknown-kid refetch
# ===========================================================================


def test_jwks_server_500_then_recovery(monkeypatch, jwks_server):
    """A down IdP must fail closed AND not be hammered: the implementation
    arms a negative cache on a failed fetch, so requests within the backoff
    window are refused WITHOUT another HTTP attempt.  Recovery (the backoff
    window elapsed — simulated deterministically; it is 60 s, not sleepable)
    picks the JWKS back up."""
    key = _setup_key(jwks_server)
    token = _mint(key, _claims())
    jwks_server.set_status(500)
    assert oidc.verify_and_decode(token) is None
    assert oidc.resolve_jwt(token) is None
    hits_after_failure = jwks_server.hits
    assert hits_after_failure >= 1, "the failed fetch must have reached the server once"
    # Still inside the backoff window: fail fast, NO further HTTP attempt.
    assert oidc.verify_and_decode(token) is None
    assert jwks_server.hits == hits_after_failure, "a down IdP must not be hammered per request"
    # The backoff window elapses (simulated) and the IdP is fixed: recovery.
    oidc._jwks_negative.clear()
    jwks_server.set_status(200)
    assert oidc.verify_and_decode(token) is not None


def test_unknown_kid_refetches_within_cache_ttl(monkeypatch, jwks_server):
    """With a warm cache (TTL 3600), a token whose kid is NOT in the cached
    JWKS must trigger a refetch — a rotated IdP signing key is honoured
    without waiting for the TTL."""
    cache_issuer = "https://oidc.test/realm-cache"
    monkeypatch.setenv("RAG_OIDC_ISSUER", cache_issuer)
    monkeypatch.setenv("RAG_OIDC_JWKS_REFRESH_SECONDS", "3600")
    stale = _generate_rsa("stale-kid")
    rotated = _generate_rsa("rotated-kid")
    jwks_server.set(_jwks_doc(stale))
    token = _mint(rotated, _claims(iss=cache_issuer))
    # warm the cache with the stale key set; the rotated token must fail…
    assert oidc.verify_and_decode(_mint(stale, _claims(iss=cache_issuer))) is not None
    assert oidc.verify_and_decode(token) is None
    # …then the IdP publishes the rotated key: the unknown kid forces a
    # refetch and the same token now verifies WITHOUT the TTL expiring.
    jwks_server.set(_jwks_doc(rotated))
    assert oidc.verify_and_decode(token) is not None


# ===========================================================================
# 11 · Browser SSO (lead increment, 2026-10): the auth-proxy forwarded
#     access token envelope + /api/oidc-session + admin-key meta suppression
# ===========================================================================


def test_forwarded_token_header_resolves_identity(monkeypatch, jwks_server):
    """X-Auth-Request-Access-Token (the oauth2-proxy envelope) resolves the
    JWT inside it — the browser SSO path.  Envelope only: a FORGED token in
    the header is still refused (the crypto trusts, not the header)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    token = _mint(key, _claims(preferred="alice"))
    ident = cr.resolve_presented([token], ["forwarded"])
    assert ident is not None and ident.name == "alice"
    assert ident.client_id == "key:alice"
    # A forged token in the same envelope: refused.
    forged = _mint(_generate_rsa("forged-kid"), _claims(preferred="alice"))
    assert cr.resolve_presented([forged], ["forwarded"]) is None


def test_forwarded_token_never_outranks_delegated_x_api_key(monkeypatch, jwks_server):
    """D19 unchanged: a co-forwarded access token cannot override an
    explicit X-API-Key delegated identity (the browser page's own key must
    never be silently replaced by a proxy-injected one)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;bob:ds2")
    candidates = [_mint(key, _claims(preferred="bob")), "alice-key"]
    presenters = ["forwarded", "x-api-key"]
    ident = cr.resolve_presented(candidates, presenters)
    assert ident is not None and ident.name == "alice", (
        "the explicit X-API-Key delegated identity must win over the forwarded JWT"
    )
    # …and the reverse: X-API-Key missing → the forwarded JWT governs.
    ident = cr.resolve_presented([_mint(key, _claims(preferred="bob"))], ["forwarded"])
    assert ident is not None and ident.name == "bob"


def test_mcp_forwarded_token_envelope(monkeypatch, jwks_server):
    """The MCP middleware accepts the same envelope (scope-level check of
    presented_keys / presented_keys_with_source)."""
    from multimodal_rag.utils.mcp_auth import presented_keys, presented_keys_with_source

    key = _setup_key(jwks_server)
    token = _mint(key, _claims(preferred="alice"))
    scope = {"headers": [(b"x-auth-request-access-token", token.encode())]}
    assert presented_keys(scope) == [token]
    pairs = presented_keys_with_source(scope)
    assert pairs == [(token, "forwarded")], "the forwarded source must NOT read as x-api-key (D19)"
    # Bearer-prefixed envelope variant (some proxies set it).
    scope = {"headers": [(b"x-auth-request-access-token", f"Bearer {token}".encode())]}
    assert presented_keys(scope) == [token]


def test_rest_oidc_session_whoami(rest_client, monkeypatch, jwks_server):
    """GET /api/oidc-session: identity preview for the presented JWT — and
    the SAME anonymous shape on every failure mode (not an oracle)."""
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1,ds2")
    r = rest_client.get(
        "/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"}
    )
    body = r.json()
    assert r.status_code == 200 and body["authenticated"] is True
    assert body["identity"] == "alice" and body["source"] == "oidc"
    assert body["datasets"] == ["ds1", "ds2"]
    assert body["oidc"]["enabled"] is True
    # Anonymous-shaped on: no credentials, invalid JWT, disabled OIDC.
    assert rest_client.get("/api/oidc-session").json() == {"authenticated": False}
    bad = rest_client.get(
        "/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims(exp=_now() - 9999))}"}
    )
    assert bad.json() == {"authenticated": False}
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")
    off = rest_client.get(
        "/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims())}"}
    )
    assert off.json() == {"authenticated": False}


def test_rest_index_suppresses_admin_key_meta_for_sso_visitor(rest_client, monkeypatch, jwks_server):
    """The critical escalation guard: a request that itself presents a
    resolvable JWT must get the homepage WITHOUT the embedded admin key
    meta (the page would otherwise send the admin key on every call and
    outrank the forwarded JWT → every SSO visitor becomes admin)."""
    key = _setup_key(jwks_server)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-admin-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    # SSO visitor (forwarded token): no meta.
    r = rest_client.get("/", headers={"X-Auth-Request-Access-Token": _mint(key, _claims(preferred="alice"))})
    assert r.status_code == 200
    assert 'meta name="rag-api-key"' not in r.text, "an SSO visitor must not receive the admin key"
    # Same page, no JWT: meta present (embedded-key mode unchanged).
    r = rest_client.get("/")
    assert 'meta name="rag-api-key"' in r.text and "deployment-admin-key" in r.text
    # A FORGED token also suppresses nothing useful — verify it does NOT
    # suppress (fail-closed meta retention: the page keeps its key mode).
    forged = _mint(_generate_rsa("forged-kid"), _claims(preferred="alice"))
    r = rest_client.get("/", headers={"X-Auth-Request-Access-Token": forged})
    assert 'meta name="rag-api-key"' in r.text


def test_rest_oidc_session_with_admin_key_stays_anonymous_shaped(rest_client, monkeypatch, jwks_server):
    """The whoami NEVER reveals that a presented key is the admin key —
    the payload is the anonymous shape (the homepage probe with the
    embedded key must look exactly like the anonymous probe)."""
    _setup_key(jwks_server)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-admin-key")
    monkeypatch.setenv("RAG_OIDC_ISSUER", "")  # OIDC off: the simplest shape
    r = rest_client.get("/api/oidc-session", headers={"X-RAG-Api-Key": "deployment-admin-key"})
    assert r.json() == {"authenticated": False}


# ===========================================================================
# 12 · D22 Browser SSO — the OIDC authorization-code flow (the OWUI pattern)
# ===========================================================================


@pytest.fixture
def sso_idp(monkeypatch, jwks_server):
    """A loopback fake IdP: discovery doc + token endpoint.

    ``POST {token}`` validates only that the presented code matches the one
    minted (a real realm would); returns an access_token minted by the D21
    test helpers so the callback verifies it through the SAME machinery.
    """

    key = _setup_key(jwks_server)  # publish the signing key (D21 verification)
    minted = {"code": "auth-code-123", "token": _mint(key, _claims(preferred="alice"))}
    hits = {"token": 0}

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence the test runner
            pass

        def _json(self, doc, status=200):
            body = json.dumps(doc).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/.well-known/openid-configuration"):
                port = self.server.server_address[1]
                self._json(
                    {
                        "authorization_endpoint": f"http://127.0.0.1:{port}/authorize",
                        "token_endpoint": f"http://127.0.0.1:{port}/token",
                    }
                )
            else:
                self._json({"error": "not_found"}, 404)

        def do_POST(self):
            if self.path == "/token":
                hits["token"] += 1
                length = int(self.headers.get("Content-Length", "0"))
                form = dict(
                    kv.split("=", 1) for kv in self.rfile.read(length).decode().split("&")
                )
                from urllib.parse import unquote_plus

                form = {k: unquote_plus(v) for k, v in form.items()}
                if form.get("code") != minted["code"]:
                    self._json({"error": "invalid_grant"}, 400)
                    return
                self._json({"access_token": minted["token"], "expires_in": 900})
            else:
                self._json({"error": "not_found"}, 404)

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("RAG_OIDC_SSO_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_ID", "ua")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_SECRET", "sso-secret")
    monkeypatch.setenv("RAG_OIDC_SSO_REDIRECT_URI", "https://rag.example/oauth/oidc/callback")
    monkeypatch.setenv("RAG_OIDC_SSO_PROVIDER_URL", f"{base}/.well-known/openid-configuration")
    monkeypatch.setattr(oidc_sso_module, "_discovery_cache", {})
    monkeypatch.setattr(oidc_sso_module, "_discovery_negative", {})
    yield {"base": base, "minted": minted, "hits": hits}
    server.shutdown()
    server.server_close()



def test_sso_inert_by_default(monkeypatch, rest_client):
    """Without the SSO env vars the routes 404 and no cookie is honoured —
    the deployment is byte-identical to pre-D22."""
    assert rest_client.get("/oauth/login").status_code == 404
    r = rest_client.get(
        "/api/oidc-session",
        cookies={"pcai-sso": "any.token.here"},
    )
    assert r.json() == {"authenticated": False}


def test_sso_login_redirects_with_state_cookie(monkeypatch, rest_client, sso_idp):
    r = rest_client.get("/oauth/login", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert "response_type=code" in loc and "client_id=ua" in loc and "state=" in loc
    set_cookie = r.headers["set-cookie"]
    assert "HttpOnly" in set_cookie and "SameSite=Lax" in set_cookie
    assert "pcai-sso-state=" in set_cookie


def test_sso_callback_happy_path_plants_session_cookie(monkeypatch, rest_client, sso_idp):
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    state = "st4te"
    r = rest_client.get(
        f"/oauth/oidc/callback?code={sso_idp['minted']['code']}&state={state}",
        cookies={"pcai-sso-state": f"{state}|/access"},
        follow_redirects=False,
    )
    assert r.status_code == 302 and r.headers["location"] == "/access"
    set_cookie = r.headers.get("set-cookie", "")
    assert "pcai-sso=" in set_cookie and "HttpOnly" in set_cookie
    # The planted cookie authenticates: the whoami (and middleware) resolve.
    token = set_cookie.split("pcai-sso=")[1].split(";")[0]
    r2 = rest_client.get("/api/oidc-session", cookies={"pcai-sso": token})
    body = r2.json()
    assert body["authenticated"] is True and body["identity"] == "alice"
    assert body["source"] == "sso-cookie"
    # …and the cookie resolves through the SAME identity machinery (D15 ACLs):
    ident = cr.resolve_presented([token])
    assert ident is not None and ident.name == "alice" and ident.datasets == frozenset({"ds1"})
    assert sso_idp["hits"]["token"] == 1


def test_sso_callback_state_mismatch_is_rejected(monkeypatch, rest_client, sso_idp):
    r = rest_client.get(
        f"/oauth/oidc/callback?code={sso_idp['minted']['code']}&state=evil",
        cookies={"pcai-sso-state": "st4te|/access"},
        follow_redirects=False,
    )
    assert r.status_code == 302 and "sso=error" in r.headers["location"]
    assert "pcai-sso=" not in r.headers.get("set-cookie", "")
    assert sso_idp["hits"]["token"] == 0, "a state mismatch must never reach the token endpoint"


def test_sso_callback_bad_code_fails_closed(monkeypatch, rest_client, sso_idp):
    r = rest_client.get(
        "/oauth/oidc/callback?code=wrong-code&state=st4te",
        cookies={"pcai-sso-state": "st4te|/"},
        follow_redirects=False,
    )
    assert r.status_code == 302 and "sso=error" in r.headers["location"]
    assert "pcai-sso=" not in r.headers.get("set-cookie", "")


def test_sso_cookie_is_lowest_priority_envelope(monkeypatch, rest_client, sso_idp):
    """An explicit API key outranks the SSO cookie (the page's own credential
    governs), and an admin key still wins over everything."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;bob:ds2")
    cookie_token = sso_idp["minted"]["token"]  # resolves to 'alice'
    # cookie alone → alice
    r = rest_client.get("/api/datasets", cookies={"pcai-sso": cookie_token})
    assert r.status_code == 200
    # cookie + explicit key for bob → bob wins (explicit beats cookie)
    ident = cr.resolve_presented([cookie_token, "bob-key"], ["cookie", "x-api-key"])
    assert ident is not None and ident.name == "bob"
    # delegation: an X-API-Key + cookie → the X-API-Key candidate alone
    ident = cr.resolve_presented(["bob-key", cookie_token], ["x-api-key", "cookie"])
    assert ident is not None and ident.name == "bob"


def test_sso_safe_next_path_blocks_open_redirects(monkeypatch):
    from multimodal_rag.utils.oidc_sso import safe_next_path

    assert safe_next_path("/access") == "/access"
    assert safe_next_path("/") == "/"
    assert safe_next_path("//evil.example") == "/"
    assert safe_next_path("https://evil.example") == "/"
    assert safe_next_path("/\\evil") == "/"


def test_sso_logout_clears_cookies(monkeypatch, rest_client):
    r = rest_client.get("/oauth/logout", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/"
    assert "pcai-sso=;" in r.headers.get("set-cookie", "") or "Max-Age=0" in r.headers.get("set-cookie", "")


# ===========================================================================
# 13 · D24 — the enforcing proxy's identity headers as a credential
#          (the Clearwing/DSH pattern: edge auth + X-Auth-Request-User)
# ===========================================================================


def test_proxy_identity_headers_credential(monkeypatch):
    """RAG_TRUST_PROXY_IDENTITY + the proxy's identity headers = a
    credential candidate resolving DIRECTLY to the registry identity
    (name = the header value)."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "andrew:andrew-key")
    monkeypatch.setenv(cr.ACLS_ENV, "andrew:ds1,ds2")
    ident = cr.resolve_presented(["andrew"], ["proxy-identity"])
    assert ident is not None and ident.name == "andrew"
    assert ident.datasets == frozenset({"ds1", "ds2"})
    assert ident.client_id == "key:andrew"


def test_proxy_identity_unknown_user_fail_closed(monkeypatch):
    """An edge-authenticated name with no registry entry yields a
    zero-dataset identity (fail-closed — same as the JWT path), not None."""
    monkeypatch.setenv("RAG_ACCESS_STORE", "1")
    ident = cr.resolve_presented(["francesco"], ["proxy-identity"])
    assert ident is not None and ident.name == "francesco"
    assert ident.datasets == frozenset()
    assert not ident.is_admin


def test_proxy_identity_never_admin_and_lowest_precedence(monkeypatch):
    """Explicit credentials outrank the proxy identity; the proxy path can
    never mint an admin."""
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;bob:ds2")
    # proxy-identity says alice; explicit X-API-Key says bob → bob wins
    ident = cr.resolve_presented(["alice", "bob-key"], ["proxy-identity", "x-api-key"])
    assert ident is not None and ident.name == "bob"
    # proxy-identity alone never admin
    ident = cr.resolve_presented(["anyone"], ["proxy-identity"])
    assert ident is not None and not ident.is_admin


def test_proxy_identity_ignored_without_trust_flag(monkeypatch):
    """Without RAG_TRUST_PROXY_IDENTITY the header is NOT a credential
    (fail-closed against spoofing on deployments without an enforcing
    proxy)."""
    import importlib

    import multimodal_rag.api_server as api
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "")
    importlib.reload(api)
    try:
        from starlette.testclient import TestClient

        r = TestClient(api.app).get(
            "/api/datasets", headers={"X-Auth-Request-User": "andrew"}
        )
        assert r.status_code in (401, 403), "the header must not authenticate"
    finally:
        monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "true")
        importlib.reload(api)


def test_proxy_identity_survives_explicit_credential_absence(monkeypatch, rest_client):
    """D24 end-to-end through the REST middleware: a browser request with
    ONLY the proxy's identity headers (the edge-authenticated shape) resolves
    the registry identity — dataset reads work, admin is denied."""
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "true")
    monkeypatch.setenv(cr.CLIENTS_ENV, "andrew:andrew-key")
    monkeypatch.setenv(cr.ACLS_ENV, "andrew:ds1,ds2")
    r = rest_client.get(
        "/api/datasets/ds1",
        headers={"X-Auth-Request-Preferred-Username": "andrew"},
    )
    assert r.status_code == 200
    r = rest_client.get(
        "/api/datasets/admin-only-check",
        headers={"X-Auth-Request-Preferred-Username": "andrew"},
    )
    assert r.status_code == 403  # D15 ACL denial fires before existence checks
    r = rest_client.get(
        "/api/admin/clients",
        headers={"X-Auth-Request-Preferred-Username": "andrew"},
    )
    assert r.status_code == 403, "the proxy identity must never reach /api/admin/*"


def test_proxy_identity_spoof_rejected_without_trust(monkeypatch, rest_client):
    """Without RAG_TRUST_PROXY_IDENTITY, a spoofed identity header does NOT
    authenticate (fail-closed — the whole point of the trust gate)."""
    import importlib

    import multimodal_rag.api_server as api
    monkeypatch.setenv(cr.CLIENTS_ENV, "andrew:andrew-key")
    monkeypatch.setenv(cr.ACLS_ENV, "andrew:ds1,ds2")
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "")
    importlib.reload(api)
    try:
        from fastapi.testclient import TestClient

        r = TestClient(api.app).get(
            "/api/datasets/ds1", headers={"X-Auth-Request-User": "andrew"}
        )
        assert r.status_code in (401, 403)
    finally:
        monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "true")
        importlib.reload(api)


def test_oauth_logged_out_landing_page(rest_client):
    """D24: /oauth/logged-out exists, renders the signed-out confirmation,
    and is public (no credentials, gate-exempt)."""
    r = rest_client.get("/oauth/logged-out")
    assert r.status_code == 200
    assert "Signed out" in r.text
    assert "Back to Multimodal RAG" in r.text
