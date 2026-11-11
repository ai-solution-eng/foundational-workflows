"""Zone B (security / identity / API + MCP) remediation regression tests.

Audit 2026-11-10, findings P1-3, P1-4, P1-5, P2-5, P2-6, P2-7, P3-3, P3-4, P3-5.
Each test below fails on the pre-remediation code — the class docstring names the
finding, the assertion names the property.

Fully offline: no network, no model, no Qdrant.  The SSO handlers are driven
directly (their coroutines return the Response), so no app startup either.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from starlette.requests import Request

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp
from multimodal_rag.utils import oidc_sso, url_policy


def _req(headers=None, client=("10.9.0.1", 4000), path="/"):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": headers or [],
            "client": client,
            "query_string": b"",
        }
    )


def _cookie_value(response) -> str:
    """The single Set-Cookie header value the response carries (attributes
    included — the control-character scan must cover the WHOLE header)."""
    values = response.headers.getlist("set-cookie")
    assert len(values) == 1, values
    return values[0]


def _cookie_pair(response) -> str:
    """Just the name=value pair of that Set-Cookie header."""
    return _cookie_value(response).split(";", 1)[0]


# ---------------------------------------------------------------------------
# P1-3 — /oauth/login must never put a decoded 'next' into a header value
# ---------------------------------------------------------------------------


@pytest.fixture
def sso_env(monkeypatch):
    """SSO configured (env only — no discovery, no network) and the
    authorization URL stubbed, so /oauth/login can be driven directly."""
    monkeypatch.setenv("OIDC_ENABLED", "1")
    monkeypatch.setenv("OIDC_ISSUER", "https://idp.example/realms/ua")
    monkeypatch.setenv("OIDC_SSO_ENABLED", "1")
    monkeypatch.setenv("OIDC_SSO_CLIENT_ID", "ua")
    monkeypatch.setenv("OIDC_SSO_CLIENT_SECRET", "sso-secret")
    monkeypatch.setenv("OIDC_SSO_REDIRECT_URI", "https://rag.example/oauth/oidc/callback")
    monkeypatch.delenv("OIDC_SSO_COOKIE_NAME", raising=False)
    monkeypatch.setattr(oidc_sso, "provider_config", lambda: ("https://idp.example/auth", "https://idp.example/token"))
    return monkeypatch


@pytest.mark.parametrize("hostile", ["\r\n", "\n", "\x00", "\t"])
def test_oauth_login_never_emits_control_chars_in_set_cookie(sso_env, hostile):
    """P1-3: the decoded 'next' query parameter used to be interpolated RAW
    into the state cookie's Set-Cookie value, so '%0d%0a' in the URL split the
    response headers (and a TAB re-formed the '//host' open-redirect form
    after the browser stripped it)."""
    next_value = f"/x{hostile}Set-Cookie: evil=1"
    resp = asyncio.run(api.oauth_login(next=next_value))
    value = _cookie_value(resp)
    assert not any(not ch.isprintable() for ch in value), repr(value)
    # the hostile path is not a legal target at all -> the neutral root
    assert _cookie_pair(resp).endswith("|/")


def test_safe_next_path_rejects_non_printables():
    """The guard itself (oidc_sso) — the single choke point both the login
    route and the callback re-apply."""
    for hostile in ("/x\r\nY", "/x\nY", "/x\x00Y", "/x\tY", "/x\u2028Y"):
        assert oidc_sso.safe_next_path(hostile) == "/"
    # ordinary same-site paths survive, including one with a space
    assert oidc_sso.safe_next_path("/access") == "/access"
    assert oidc_sso.safe_next_path("/a b") == "/a b"
    # the open-redirect refusals are unchanged
    assert oidc_sso.safe_next_path("//evil.example") == "/"
    assert oidc_sso.safe_next_path("https://evil.example") == "/"
    assert oidc_sso.safe_next_path("/a\\b") == "/"


def test_oauth_login_percent_encodes_the_target(sso_env):
    """P1-3: the target rides the cookie percent-encoded and is decoded at the
    callback, so a legal-but-special path survives the round trip without ever
    reaching the header raw."""
    resp = asyncio.run(api.oauth_login(next="/a b"))
    target = _cookie_pair(resp).split("|", 1)[1]
    assert target == "/a%20b"
    from urllib.parse import unquote

    assert oidc_sso.safe_next_path(unquote(target)) == "/a b"


def test_oauth_login_keeps_the_generated_state(sso_env, monkeypatch):
    """P3-3: the state was recovered with url.split("state=")[-1], which works
    only while 'state' is the LAST parameter.  Here the stub puts a parameter
    AFTER it — the cookie must still carry exactly the generated nonce."""
    monkeypatch.setattr(
        oidc_sso,
        "build_authorization_url",
        lambda state: f"https://idp.example/auth?state={state}&extra=1",
    )
    resp = asyncio.run(api.oauth_login(next="/access"))
    from urllib.parse import parse_qs, urlparse

    sent_state = parse_qs(urlparse(resp.headers["location"]).query)["state"][0]
    cookie_state = _cookie_pair(resp).split("pcai-sso-state=", 1)[1].split("|", 1)[0]
    assert cookie_state == sent_state
    assert "&" not in cookie_state


# ---------------------------------------------------------------------------
# P1-4 — the D10 caller id (throttle bucket + unlock-cache key)
# ---------------------------------------------------------------------------


def test_rest_caller_id_ignores_a_bare_identity_header(monkeypatch):
    """P1-4a: with the flag on but NO enforcing-proxy assertion, a bare
    X-Auth-Request-User must not become the throttle/unlock bucket key — an
    attacker reaching the pod directly would otherwise mint a fresh
    brute-force bucket per request."""
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.delenv("RAG_ENFORCING_PROXY_SIGNAL", raising=False)
    request = _req([(b"x-auth-request-user", b"alice")])
    assert api._unlock_client_id(request) == "10.9.0.1"


def test_rest_caller_id_trusts_the_header_behind_an_enforcing_proxy(monkeypatch):
    """P1-4a, positive side: the operator-configured assertion restores the
    per-user bucket (the feature is not lost, it is gated)."""
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.setenv("RAG_ENFORCING_PROXY_SIGNAL", "s3cret")
    request = _req(
        [
            (b"x-auth-request-user", b"alice"),
            (b"x-enforcing-proxy", b"s3cret"),
        ]
    )
    assert api._unlock_client_id(request) == "alice"


def test_rest_unlock_cache_is_bounded(monkeypatch):
    """P1-4b: the cache key is caller-derived, so it must be capped — one
    caller rotating identities used to pin unbounded plaintext passwords."""
    monkeypatch.setattr(api, "_get_redis", lambda: None)
    monkeypatch.setattr(api, "_MAX_UNLOCK_ENTRIES", 8)
    api._UNLOCK_CACHE.clear()
    try:
        for i in range(40):
            api._unlock_cache_set("ds", f"cid-{i}", f"pw-{i}", ttl=300)
        assert len(api._UNLOCK_CACHE) <= 8
        # the surviving entries are the most recent ones
        assert ("ds", "cid-39") in api._UNLOCK_CACHE
    finally:
        api._UNLOCK_CACHE.clear()


# ---------------------------------------------------------------------------
# P2-7 — the trust flag is read per call, not frozen at import
# ---------------------------------------------------------------------------


def test_trust_flag_is_read_per_call_on_both_surfaces(monkeypatch):
    """P2-7: flipping the env off at runtime (the documented no-restart
    remediation) must take effect on the next request, not the next restart."""
    monkeypatch.setattr(api, "_TRUST_PROXY_IDENTITY", True)
    monkeypatch.setattr(mcp, "_TRUST_PROXY_IDENTITY", True)
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "false")
    assert api._trust_proxy_flag() is False
    assert mcp._trust_proxy_flag() is False
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    assert api._trust_proxy_flag() is True
    assert mcp._trust_proxy_flag() is True
    # unset keeps the import-time default (the offline tests' patch seam)
    monkeypatch.delenv("RAG_TRUST_PROXY_IDENTITY")
    assert api._trust_proxy_flag() is True
    assert mcp._trust_proxy_flag() is True


# ---------------------------------------------------------------------------
# P3-4 — the D25 self-mint proof must go through the per-request reader
# ---------------------------------------------------------------------------


def test_self_mint_proof_requires_the_enforcing_proxy_assertion(monkeypatch):
    """P3-4: a bare proxy-identity header is not a credential (P0-1) and so
    must not be an SSO proof for a durable key either."""
    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.delenv("RAG_ENFORCING_PROXY_SIGNAL", raising=False)
    monkeypatch.delenv("OIDC_ENABLED", raising=False)
    ident = SimpleNamespace(name="alice", is_admin=False)
    assert api._sso_proof_for_identity(_req([(b"x-auth-request-user", b"alice")]), ident) is None
    # ...and with the operator assertion it IS the proof
    monkeypatch.setenv("RAG_ENFORCING_PROXY_SIGNAL", "s3cret")
    proof = api._sso_proof_for_identity(
        _req([(b"x-auth-request-user", b"alice"), (b"x-enforcing-proxy", b"s3cret")]),
        ident,
    )
    assert proof == "proxy-identity"


# ---------------------------------------------------------------------------
# P3-5 — __Host- cookie support (the shipped default name is unchanged)
# ---------------------------------------------------------------------------


def test_host_prefixed_session_cookie_is_secure_and_domainless(monkeypatch):
    monkeypatch.setenv("OIDC_SSO_COOKIE_NAME", "__Host-pcai-sso")
    # even with the Secure override explicitly disabled: the prefix mandates it
    value = api._sso_cookie_headers("tok", 60, secure=False)[0][1]
    assert value.startswith("__Host-pcai-sso=tok")
    assert "Secure" in value
    assert "Domain" not in value and "Path=/" in value
    # the deletion must carry the same attributes or the browser ignores it
    cleared = [v for _h, v in api._sso_clear_cookie_headers()]
    assert all("Secure" in v for v in cleared)
    assert all("Domain" not in v for v in cleared)


def test_plain_cookie_name_still_works_and_warns_once(monkeypatch, caplog):
    monkeypatch.delenv("OIDC_SSO_COOKIE_NAME", raising=False)
    monkeypatch.setattr(api, "_COOKIE_PREFIX_WARNED", set())
    with caplog.at_level("WARNING", logger=api.logger.name):
        api._sso_cookie_headers("tok", 60, secure=True)
        api._sso_cookie_headers("tok", 60, secure=True)
    hits = [r for r in caplog.records if "__Host-" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# P2-5 — the INGEST_ALLOW_HOSTS allowlist must win over the own-media exemption
# ---------------------------------------------------------------------------


def test_ingest_allow_hosts_is_authoritative_for_media(monkeypatch):
    """P2-5: the own-media-host exemption returned BEFORE the allowlist, so a
    strict egress allowlist silently admitted the MEDIA_BASE_URL host."""
    monkeypatch.setenv("MEDIA_BASE_URL", "https://rag.example.com")
    monkeypatch.setattr(url_policy, "_INGEST_ALLOW_HOSTS", ("cdn.corp",))
    with pytest.raises(ValueError, match="INGEST_ALLOW_HOSTS"):
        url_policy._check_media_url_policy("https://rag.example.com/api/datasets/d/files/x.png")
    # an allowlisted host still passes
    url_policy._check_media_url_policy("https://cdn.corp/x.png")
    # and with NO allowlist the own host keeps its exemption
    monkeypatch.setattr(url_policy, "_INGEST_ALLOW_HOSTS", ())
    url_policy._check_media_url_policy("https://rag.example.com/api/datasets/d/files/x.png")


# ---------------------------------------------------------------------------
# P2-6 — every password-writing route revokes the old grants
# ---------------------------------------------------------------------------


def test_revoke_password_grants_covers_both_legs(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_unlock_cache_del_all", lambda name: calls.append(("unlock", name)) or 3)
    monkeypatch.setattr(api, "_clear_selection_passwords", lambda name: calls.append(("selection", name)) or 2)
    revoked, cleared = asyncio.run(api._revoke_password_grants("ds1"))
    assert (revoked, cleared) == (3, 2)
    assert calls == [("unlock", "ds1"), ("selection", "ds1")]


def test_every_password_writer_revokes_grants():
    """P2-6: the reset route always did; the IMPORT route (which can replace an
    existing dataset under the same name) did not."""
    assert "_revoke_password_grants(" in inspect.getsource(api.api_admin_reset_dataset_password)
    assert "_revoke_password_grants(" in inspect.getsource(api.api_import_dataset)


# ---------------------------------------------------------------------------
# Wave-2 (factual-integrity cross-review) — follow-ups on the above
# ---------------------------------------------------------------------------


def test_caller_id_token_holder_cannot_rotate_the_identity_header(monkeypatch):
    """Wave-2: a token only ever vouches for the identity it NAMES.

    With no operator signal configured, the forwarded-JWT branch was called
    with no claimed name, so ANY caller holding SOME valid realm token (their
    own) satisfied the precondition and could then rotate X-Auth-Request-User
    to arbitrary values — the fresh-bucket rotation was back for token-holders.
    """
    from types import SimpleNamespace

    from multimodal_rag.utils import oidc_identity

    monkeypatch.setenv("RAG_TRUST_PROXY_IDENTITY", "1")
    monkeypatch.delenv("RAG_ENFORCING_PROXY_SIGNAL", raising=False)
    monkeypatch.setattr(oidc_identity, "is_jwt_format", lambda t: t.count(".") == 2)
    monkeypatch.setattr(
        oidc_identity,
        "resolve_jwt",
        lambda t: SimpleNamespace(name="alice") if t == "alice.token.here" else None,
    )
    token = (b"x-auth-request-access-token", b"alice.token.here")
    # the holder of alice's token may NOT assert a victim name
    assert api._unlock_client_id(_req([(b"x-auth-request-user", b"victim"), token])) == "10.9.0.1"
    # ...but it may assert the identity it actually names
    assert api._unlock_client_id(_req([(b"x-auth-request-user", b"alice"), token])) == "alice"
    # the forwarded-for chain is never token-vouched (require_signal)
    assert api._unlock_client_id(_req([(b"x-forwarded-for", b"1.2.3.4"), token])) == "10.9.0.1"


def test_sso_clear_cookie_mirrors_the_set_attributes(monkeypatch):
    """Wave-2: the callback's state-cookie clear lacked Secure, so a
    Secure/__Host- state cookie survived a successful sign-in until Max-Age."""
    monkeypatch.setenv("OIDC_SSO_COOKIE_NAME", "__Host-pcai-sso")
    value = api._sso_clear_cookie("__Host-pcai-sso-state")
    assert value.startswith("__Host-pcai-sso-state=;")
    assert "Max-Age=0" in value
    assert "Secure" in value
    assert "Domain" not in value
    # the callback goes THROUGH the helper (not a hand-rolled clear)
    assert "_sso_clear_cookie(" in inspect.getsource(api.oauth_callback)
    # the logout path clears both names under the same rule
    cleared = [v for _h, v in api._sso_clear_cookie_headers()]
    assert len(cleared) == 2
    assert all("Secure" in v for v in cleared)


def test_ingest_allow_hosts_warns_once_when_it_excludes_the_own_host(monkeypatch, caplog):
    """Wave-2: making the allowlist authoritative is right, but a deployment
    that sets it without its own endpoint host silently loses self-media
    fetches — so say it once, with both values named."""
    monkeypatch.setenv("MEDIA_BASE_URL", "https://rag.example.com")
    monkeypatch.setattr(url_policy, "_INGEST_ALLOW_HOSTS", ("cdn.corp",))
    monkeypatch.setattr(url_policy, "_own_host_allowlist_warned", False)
    with caplog.at_level("WARNING", logger=url_policy.logger.name):
        for _ in range(3):
            with pytest.raises(ValueError, match="INGEST_ALLOW_HOSTS"):
                url_policy._check_media_url_policy("https://other.example/x.png")
    hits = [r for r in caplog.records if "MEDIA_BASE_URL" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    message = hits[0].getMessage()
    assert "rag.example.com" in message
    assert "cdn.corp" in message
    # a correctly-configured allowlist stays silent
    monkeypatch.setattr(url_policy, "_own_host_allowlist_warned", False)
    monkeypatch.setattr(url_policy, "_INGEST_ALLOW_HOSTS", ("rag.example.com",))
    caplog.clear()
    with caplog.at_level("WARNING", logger=url_policy.logger.name):
        url_policy._check_media_url_policy("https://rag.example.com/x.png")
    assert not [r for r in caplog.records if "MEDIA_BASE_URL" in r.getMessage()]
