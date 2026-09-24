"""The /access page and the unlock TTL bounds (RAG_UNLOCK_MAX_TTL).

The /access page is the per-user key-holder's counterpart to the operator
dashboard: public like / and /manage (the page is its own auth boundary — the
user pastes their own key, the server injects none), and it drives the same
unlock/lock endpoints.

RAG_UNLOCK_MAX_TTL bounds every explicit unlock TTL (default 86400, the
historical hard cap).  The special value 0 opts the deployment into
NO-EXPIRY unlocks: ttl=0 persists until an explicit lock (REST) or eviction
(MCP's bounded in-memory cache).  Bounded by default, opt-in to open — the
same values-gated posture as every other knob in this repo.

Run::

    pytest tests/security/test_access_page.py -q
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp

_ACCESS_TEMPLATE = Path(api.__file__).parent / "templates" / "access.html"


# ---------------------------------------------------------------------------
# /access page
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ACCESS_TEMPLATE.exists(), reason="access.html not yet present")
def test_access_page_is_public_under_auth(monkeypatch):
    """With the REST key set, /access must behave like / and /manage: public."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    client = TestClient(api.app)
    for path in ("/", "/manage", "/access"):
        assert client.get(path).status_code == 200, path


@pytest.mark.skipif(not _ACCESS_TEMPLATE.exists(), reason="access.html not yet present")
def test_access_page_injects_no_api_key():
    """The operator dashboard embeds RAG_API_KEY as a meta tag (its own auth
    boundary context); /access must NEVER carry any key material — under D15
    it is used with per-user registry keys and an injected deployment key
    would leak admin credentials onto a public page.  The meta tag NAME may
    legitimately appear in the page's own JS (it must not read it); no KEY
    VALUE may ever appear."""
    page = asyncio.run(api.access())
    # No injected meta tag (the route never adds one).
    assert '<meta name="rag-api-key"' not in page
    # And no key material even if a deployment env leaks into the render.
    assert "deployment-key" not in page


@pytest.mark.skipif(not _ACCESS_TEMPLATE.exists(), reason="access.html not yet present")
def test_access_page_carries_ttl_bound_meta(monkeypatch):
    """The page's TTL selector mirrors the deployment's configured bound."""
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "3600")
    try:
        page = asyncio.run(api.access())
    finally:
        os.environ.pop("RAG_UNLOCK_MAX_TTL", None)
    assert 'name="rag-unlock-ttl-max" content="3600"' in page


@pytest.mark.skipif(not _ACCESS_TEMPLATE.exists(), reason="access.html not yet present")
def test_access_page_consumes_ttl_bound_meta():
    """The injected bound must be CONSUMED, not just carried: the page gates
    its "No expiry (0)" option on the meta so a default deployment (bound
    86400, ttl=0 rejected by the server) never shows a doomed option."""
    page = asyncio.run(api.access())
    assert 'meta[name="rag-unlock-ttl-max"]' in page
    assert "applyTtlPolicy" in page
    assert "noExpiryEnabled" in page


def test_access_page_falls_back_to_index_when_template_missing(monkeypatch):
    """A deployment whose wheel predates access.html still serves something
    sane (the dashboard) instead of 500ing."""
    monkeypatch.setattr(api, "_load_html", lambda name: None)
    page = asyncio.run(api.access())
    assert "Frontend not found" in page or len(page) > 0


# ---------------------------------------------------------------------------
# RAG_UNLOCK_MAX_TTL — the bound resolver
# ---------------------------------------------------------------------------


def test_ttl_max_default_is_24h():
    assert api._unlock_ttl_max() == 86400
    assert mcp._unlock_ttl_max() == 86400


def test_ttl_max_reads_env_per_call(monkeypatch):
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "600")
    assert api._unlock_ttl_max() == 600
    assert mcp._unlock_ttl_max() == 600
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    assert api._unlock_ttl_max() == 0
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "-5")
    assert api._unlock_ttl_max() == 86400  # malformed/negative → default
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "banana")
    assert api._unlock_ttl_max() == 86400


# ---------------------------------------------------------------------------
# ttl=0 no-expiry unlock — REST cache primitives
# ---------------------------------------------------------------------------


def test_rest_unlock_cache_no_expiry_entry_survives():
    api._UNLOCK_CACHE.clear()
    try:
        api._unlock_cache_set("ds", "cid", "secret-a", ttl=0)
        # An entry set "now" with no expiry must still be readable long
        # after any real TTL would have lapsed — simulate by asserting the
        # stored expiry is the no-expiry sentinel.
        expiry, pw = api._UNLOCK_CACHE[("ds", "cid")]
        assert expiry == api._UNLOCK_NO_EXPIRY
        assert pw == "secret-a"
        assert api._unlock_cache_get("ds", "cid") == "secret-a"
    finally:
        api._UNLOCK_CACHE.clear()


def test_rest_unlock_cache_ttl_entry_still_expires(monkeypatch):
    api._UNLOCK_CACHE.clear()
    try:
        api._unlock_cache_set("ds", "cid", "secret-a", ttl=300)
        expiry, _ = api._UNLOCK_CACHE[("ds", "cid")]
        assert expiry != api._UNLOCK_NO_EXPIRY
        # Force expiry and confirm the read path evicts it.
        api._UNLOCK_CACHE[("ds", "cid")] = (0.0, "secret-a")
        assert api._unlock_cache_get("ds", "cid") is None
    finally:
        api._UNLOCK_CACHE.clear()


# ---------------------------------------------------------------------------
# ttl=0 no-expiry unlock — REST endpoint behaviour (full stack)
# ---------------------------------------------------------------------------


class _FakeDM:
    """The minimum DatasetManager surface the unlock endpoints touch."""

    def get_dataset(self, name, sync_count=False):
        if name == "missing":
            raise FileNotFoundError(f"Dataset '{name}' not found")
        return {"name": name, "document_count": 0}

    def has_password(self, name):
        return name == "protected"

    def verify_password(self, name, pw):
        return pw == "correct-pw"


@pytest.fixture
def unlock_client(monkeypatch):
    from fastapi.testclient import TestClient

    async def _fake_manager():
        return _FakeDM()

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    api._UNLOCK_CACHE.clear()
    yield TestClient(api.app)
    api._UNLOCK_CACHE.clear()


def test_rest_unlock_rejects_ttl_zero_by_default(unlock_client):
    """Bounded by default: ttl=0 without the opt-in is a 400, not a silent
    forever-unlock."""
    r = unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": 0},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 400
    assert "RAG_UNLOCK_MAX_TTL" in r.json()["detail"]


def test_rest_unlock_accepts_ttl_zero_when_opted_in(unlock_client, monkeypatch):
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    r = unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": 0},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 200
    assert r.json()["ttl_seconds"] == 0
    assert "no expiry" in r.json()["message"]
    # And the cache entry really is no-expiry (tuple shape: (expiry, password)).
    entries = [v for k, v in api._UNLOCK_CACHE.items() if k[0] == "protected"]
    assert entries and entries[0][0] == api._UNLOCK_NO_EXPIRY
    assert entries[0][1] == "correct-pw"


def test_rest_unlock_clamps_to_configured_max(unlock_client, monkeypatch):
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "3600")
    r = unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": 7200},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 400
    assert "3600" in r.json()["detail"]


def test_rest_unlock_default_bound_still_applies(unlock_client, monkeypatch):
    """No env → the historical 24h cap (byte-identical default behaviour)."""
    monkeypatch.delenv("RAG_UNLOCK_MAX_TTL", raising=False)
    r = unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": 86401},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 400


def test_rest_unlock_rejects_non_integer_ttl(unlock_client):
    r = unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": "soon"},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    assert r.status_code == 400


def test_rest_lock_still_revokes_no_expiry_unlock(unlock_client, monkeypatch):
    """The point of ttl=0: revocation stays caller-controlled."""
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    unlock_client.post(
        "/api/datasets/protected/unlock",
        json={"password": "correct-pw", "ttl": 0},
        headers={"X-RAG-Api-Key": "deployment-key"},
    )
    cid = "testclient"  # TestClient presents itself as this socket peer
    assert api._unlock_cache_get("protected", cid) is not None
    r = unlock_client.post("/api/datasets/protected/lock", headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200
    assert api._unlock_cache_get("protected", cid) is None


# ---------------------------------------------------------------------------
# ttl=0 no-expiry unlock — MCP tool semantics
# ---------------------------------------------------------------------------


class _StubDM:
    def get_dataset(self, name, check_embedder=True):
        return {"name": name}

    def has_password(self, name):
        return True

    def verify_password(self, name, pw):
        return pw == "correct-pw"


def _run_unlock(ttl, monkeypatch):
    """Drive the MCP unlock_dataset tool body offline; return its message."""
    monkeypatch.setattr(mcp, "get_manager", lambda: _StubDM())
    token = mcp._client_id_ctx.set("caller-T")
    try:
        return asyncio.run(mcp.unlock_dataset(dataset_name="ds", password="correct-pw", ttl=ttl))
    finally:
        mcp._client_id_ctx.reset(token)


def test_mcp_unlock_rejects_ttl_zero_by_default(monkeypatch):
    mcp._unlocked.clear()
    try:
        with pytest.raises(Exception, match="RAG_UNLOCK_MAX_TTL"):
            _run_unlock(0, monkeypatch)
        assert not mcp._unlocked
    finally:
        mcp._unlocked.clear()


def test_mcp_unlock_accepts_ttl_zero_when_opted_in(monkeypatch):
    mcp._unlocked.clear()
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "0")
    try:
        msg = _run_unlock(0, monkeypatch)
        assert "no expiry" in msg
        entries = list(mcp._unlocked.values())
        assert entries and entries[0][0] == mcp._MCP_UNLOCK_NO_EXPIRY
    finally:
        mcp._unlocked.clear()
        os.environ.pop("RAG_UNLOCK_MAX_TTL", None)


def test_mcp_unlock_honours_configured_max(monkeypatch):
    mcp._unlocked.clear()
    monkeypatch.setenv("RAG_UNLOCK_MAX_TTL", "600")
    try:
        with pytest.raises(Exception, match="600"):
            _run_unlock(601, monkeypatch)
    finally:
        mcp._unlocked.clear()
        os.environ.pop("RAG_UNLOCK_MAX_TTL", None)


def test_mcp_no_expiry_entry_is_readable_via_is_unlocked(monkeypatch):
    """A no-expiry cached unlock actually satisfies the check helpers."""
    mcp._unlocked.clear()
    try:
        token = mcp._client_id_ctx.set("caller-NX")
        try:
            mcp._cache_unlock("ds", "correct-pw", ttl=0)
            assert mcp._is_unlocked("ds") == "correct-pw"
        finally:
            mcp._client_id_ctx.reset(token)
    finally:
        mcp._unlocked.clear()
