"""Audit P1-7 / P1-10: the shared MCP password-verification throttle + the
SSE auth predicate.

P1-7 — ``_check_unlocked_or_password`` used to call ``dm.verify_password``
directly, so ``search_dataset``, ``get_dataset_files``, ``get_dataset_info``,
``dataset_add_documents``, ``dataset_delete_documents`` and
``dataset_replace_document`` were UNTHROTTLED password oracles: each wrong
guess cost a PBKDF2-600k verification (~0.2–0.5 s) and pinned one of the MCP
pool's 64 threads.  Verification now runs through the single funnel
``_verify_password_throttled`` (check bucket BEFORE verifying, record on
failure, reset on success) which ``unlock_dataset`` also uses.

P1-10 — the ``sse`` transport's routes are ``/sse`` + the ``/messages/``
mount, but the shared auth predicate was ``p.startswith("/mcp")``, which
matched NEITHER: an anonymous caller reached the SSE transport even with keys
configured.  The predicates are now transport-specific module-level
functions (``_protected_sse_path`` / ``_protected_streamable_path``) and the
gate is applied by WRAPPING the built app (``app = _RagClientAuthMiddleware(
app, ...)``) instead of ``app.add_middleware(...)`` after the stack was built.

Everything here is offline: the DatasetManager is a stub (no Qdrant, no
models), the middleware is driven over raw ASGI scopes, and the env knobs
(``PW_MAX_FAILURES`` / ``PW_FAIL_WINDOW``) are read at MODULE import time —
see the ``monkeypatch``/``importlib.reload`` dance in ``_throttle_env``.

Run::

    pytest tests/security/test_mcp_password_throttle.py -q
"""

import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp

# A key set the auth middleware accepts — configured per test so the 401 path
# is exercised (with NO keys the middleware is open by design, dev mode).
_KEY_ENV = "MCP_API_KEYS"
_KEY = "throttle-test-key"
_PEER = ("10.9.0.1", 5000)  # direct peer → deterministic D10 identity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Fresh unlock cache + throttle buckets around every test.

    The D10 identity environment is pinned too: no proxy trust, no shared
    escape hatch, no registry/OIDC, so the identity is the socket peer (or
    the contextvar the middleware set).
    """
    monkeypatch.setattr(mcp, "_TRUST_PROXY_IDENTITY", False)
    monkeypatch.delenv("RAG_MCP_SHARED_UNLOCK", raising=False)
    monkeypatch.delenv("RAG_API_KEY_CLIENTS", raising=False)
    monkeypatch.delenv("RAG_DATASET_ACLS", raising=False)
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("RAG_OIDC_ENABLED", raising=False)
    monkeypatch.delenv("RAG_ACCESS_STORE", raising=False)
    mcp._unlocked.clear()
    mcp._mcp_pw_fail_buckets.clear()
    yield
    mcp._unlocked.clear()
    mcp._mcp_pw_fail_buckets.clear()


@pytest.fixture
def _throttle_env(monkeypatch):
    """PW_MAX_FAILURES=2 / PW_FAIL_WINDOW=300, as the task pins them.

    ``_MCP_PW_MAX_FAILURES`` / ``_MCP_PW_FAIL_WINDOW`` are module constants
    evaluated at import time, so the env vars alone would not move them:
    set the env (for anything that re-reads it) AND reload the module, then
    hand the reloaded module to the caller and to the other helpers through
    ``sys.modules`` (a reload rebinds the module object's globals in place —
    ``importlib.reload`` mutates the SAME module object — so the fixture's
    reload is visible everywhere).
    """
    monkeypatch.setenv("PW_MAX_FAILURES", "2")
    monkeypatch.setenv("PW_FAIL_WINDOW", "300")
    reloaded = importlib.reload(mcp)
    assert reloaded._MCP_PW_MAX_FAILURES == 2, "PW_MAX_FAILURES env did not take effect"
    assert reloaded._MCP_PW_FAIL_WINDOW == 300.0, "PW_FAIL_WINDOW env did not take effect"
    reloaded._unlocked.clear()
    reloaded._mcp_pw_fail_buckets.clear()
    yield reloaded
    reloaded._unlocked.clear()
    reloaded._mcp_pw_fail_buckets.clear()
    # Put the module's original import-time constants back for later tests.
    monkeypatch.undo()
    importlib.reload(mcp)
    assert mcp._MCP_PW_MAX_FAILURES == max(1, int(os.environ.get("PW_MAX_FAILURES", "10"))), (
        "teardown must restore the module's import-time throttle constants"
    )


class _StubDM:
    """Minimal DatasetManager surface used by the unlock gate."""

    def __init__(self, correct="correct-pw", protected=True):
        self.correct = correct
        self.protected = protected
        self.calls = 0

    def has_password(self, name):
        return self.protected

    def verify_password(self, name, password):
        self.calls += 1
        return password == self.correct


# ---------------------------------------------------------------------------
# ASGI scaffolding (same shape as tests/security/test_dataset_acls.py)
# ---------------------------------------------------------------------------


def _scope(path="/sse", headers=(), client=_PEER):
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": list(headers),
        "client": client,
        "query_string": b"",
    }


async def _drive(middleware, scope, app_body=None):
    """Run *middleware* over *scope*; return (status, body_text, captured).

    ``app_body`` runs INSIDE the wrapped app, i.e. with the request's
    ContextVars bound, exactly like an MCP tool body.
    """
    messages = []
    captured = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
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
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, body.decode(), captured


def _run(coro):
    return asyncio.run(coro)


def _auth_mw(auth_mod, protected):
    return auth_mod._RagClientAuthMiddleware(
        app=None, env_names=auth_mod.AUTH_ENV_NAMES, protected=protected
    )


# ===========================================================================
# 1 · The funnel: throttle → verify → record / reset
# ===========================================================================


def test_funnel_records_failure_and_resets_on_success(_throttle_env):
    mod = _throttle_env
    dm = _StubDM()
    cid = "caller-A"

    assert mod._mcp_pw_failure_count(cid, "ds") == 0
    assert mod._verify_password_throttled(cid, dm, "ds", "nope") is False
    assert mod._mcp_pw_failure_count(cid, "ds") == 1, "a wrong password must be recorded"
    # A correct password (bucket still under the limit) RESETS the failures.
    assert mod._verify_password_throttled(cid, dm, "ds", "correct-pw") is True
    assert mod._mcp_pw_failure_count(cid, "ds") == 0, "success must clear the caller's failures"
    mod._mcp_pw_check_throttle(cid, "ds")  # no longer refuses

    # Now exhaust the bucket (PW_MAX_FAILURES=2): the next call refuses
    # BEFORE verification — the verifier must not even see the guess, and
    # that holds for a CORRECT password too (check-before-verify is the
    # deliberate fail-closed order; the caller waits out the window).
    assert mod._verify_password_throttled(cid, dm, "ds", "nope") is False
    assert mod._verify_password_throttled(cid, dm, "ds", "nope") is False
    assert mod._mcp_pw_failure_count(cid, "ds") == 2
    before = dm.calls
    with pytest.raises(mod.ToolError, match="Too many password attempts"):
        mod._verify_password_throttled(cid, dm, "ds", "correct-pw")
    assert dm.calls == before, "throttled calls must not pay for a PBKDF2 verification"


def test_funnel_buckets_are_per_caller(_throttle_env):
    mod = _throttle_env
    dm = _StubDM()
    for _ in range(mod._MCP_PW_MAX_FAILURES):
        assert mod._verify_password_throttled("attacker", dm, "ds", "nope") is False
    with pytest.raises(mod.ToolError, match="Too many password attempts"):
        mod._verify_password_throttled("attacker", dm, "ds", "nope")
    # Another caller is untouched.
    assert mod._verify_password_throttled("victim", dm, "ds", "correct-pw") is True
    assert mod._mcp_pw_failure_count("victim", "ds") == 0


# ===========================================================================
# 2 · The tools' gate (_check_unlocked_or_password) is throttled
# ===========================================================================


def test_wrong_passwords_through_the_gate_refuse_after_max_failures(_throttle_env):
    """N wrong passwords through _check_unlocked_or_password: the first
    PW_MAX_FAILURES raise the password error, then the throttle."""
    mod = _throttle_env
    dm = _StubDM()  # protected, correct-pw only

    def _attempt(pw):
        return mod._check_unlocked_or_password(dm, "secret", pw)

    tok = mod._client_id_ctx.set("caller-A")
    try:
        for i in range(mod._MCP_PW_MAX_FAILURES):
            with pytest.raises(mod.ToolError, match="Incorrect password") as exc:
                _attempt(f"guess-{i}")
            # byte-identical wording to the pre-audit refusal (pinned)
            assert str(exc.value) == "Incorrect password for dataset 'secret'."
        # Bucket full → the throttle refusal, not a verification.
        calls = dm.calls
        with pytest.raises(mod.ToolError, match="Too many password attempts") as exc:
            _attempt("guess-again")
        assert str(exc.value) == "Too many password attempts — try again later."
        assert dm.calls == calls
        # …and the throttle fires for EVERY tool that shares the funnel.
        for entry in (
            lambda: mod._check_unlocked_or_password(dm, "secret", "guess"),
            lambda: mod._resolve_and_unlock(dm, "secret", "guess"),
        ):
            with pytest.raises(mod.ToolError, match="Too many password attempts"):
                entry()
    finally:
        mod._client_id_ctx.reset(tok)


def test_gate_ignores_no_password_paths_when_bucket_exhausted(_throttle_env):
    """Regression: unlocked / unprotected reads must not start consulting the
    (unrelated) password bucket — no password argument means no verification,
    so no throttle and no failure recorded."""
    mod = _throttle_env
    dm = _StubDM(protected=False)

    def _read():
        return mod._check_unlocked_or_password(dm, "open-ds", None)

    tok = mod._client_id_ctx.set("caller-A")
    try:
        assert _read() is None
        assert mod._mcp_pw_failure_count("caller-A", "ds") == 0
        # Even with a full bucket, a passwordless read of an unlocked dataset
        # still works (the throttle guards verification, not reads).
        for _ in range(mod._MCP_PW_MAX_FAILURES):
            mod._mcp_pw_record_failure("caller-A", "ds")
        assert _read() is None
        mod._cache_unlock("open-ds", "pw", ttl=300)
        assert mod._check_unlocked_or_password(dm, "open-ds", None) == "pw"
    finally:
        mod._client_id_ctx.reset(tok)


def test_correct_password_after_failures_works_once_bucket_cleared(_throttle_env):
    """Wrong guesses are counted; after the bucket is cleared (or the window
    lapses / the caller succeeds) the correct password goes through and the
    unlock is cached."""
    mod = _throttle_env
    dm = _StubDM()

    tok = mod._client_id_ctx.set("caller-A")
    try:
        for _ in range(mod._MCP_PW_MAX_FAILURES):
            with pytest.raises(mod.ToolError, match="Incorrect password"):
                mod._check_unlocked_or_password(dm, "secret", "wrong")
        with pytest.raises(mod.ToolError, match="Too many password attempts"):
            mod._check_unlocked_or_password(dm, "secret", "correct-pw")

        # Clear the buckets (the fixture's "between tests" reset, applied
        # mid-test): the correct password then works and warms the cache.
        mod._mcp_pw_fail_buckets.clear()
        assert mod._check_unlocked_or_password(dm, "secret", "correct-pw") == "correct-pw"
        assert mod._mcp_pw_failure_count("caller-A", "ds") == 0
        assert mod._is_unlocked("secret") == "correct-pw"
    finally:
        mod._client_id_ctx.reset(tok)


def test_correct_password_without_prior_failures_resets_nothing_else(_throttle_env):
    """Another caller's bucket survives a different caller's success."""
    mod = _throttle_env
    dm = _StubDM()
    mod._mcp_pw_record_failure("attacker", "ds")

    tok = mod._client_id_ctx.set("victim")
    try:
        assert mod._check_unlocked_or_password(dm, "secret", "correct-pw") == "correct-pw"
    finally:
        mod._client_id_ctx.reset(tok)
    assert mod._mcp_pw_failure_count("victim", "ds") == 0
    assert mod._mcp_pw_failure_count("attacker", "ds") == 1, "unrelated buckets are untouched"


# ===========================================================================
# 3 · The SSE auth predicate covers both transports (P1-10)
# ===========================================================================


def test_sse_predicate_covers_sse_and_messages():
    sse = mcp._protected_sse_path
    assert sse("/sse") is True
    assert sse("/messages/") is True
    assert sse("/messages/x") is True
    assert sse("/messages/abc?x=1") is True  # path only, query kept out
    assert sse("/mcp") is True
    # Probes stay public; unrelated paths are not gated by THIS predicate.
    assert sse("/healthz") is False
    assert sse("/readyz") is False
    assert sse("/") is False


def test_streamable_predicate_unchanged():
    pred = mcp._protected_streamable_path
    assert pred("/mcp") is True
    assert pred("/mcp/") is True
    assert pred("/healthz") is False
    assert pred("/sse") is False  # that transport has no SSE route


def test_auth_gate_401s_anonymous_sse_and_messages(monkeypatch):
    """The middleware classes the production wiring installs: with keys
    configured, an anonymous /sse and /messages/x are refused (401) and a
    keyed request passes; the probes stay open."""
    monkeypatch.setenv(_KEY_ENV, _KEY)
    mw = _auth_mw(mcp, mcp._protected_sse_path)

    status, _, _ = _run(_drive(mw, _scope("/sse")))
    assert status == 401
    status, _, _ = _run(_drive(mw, _scope("/messages/x")))
    assert status == 401
    status, _, _ = _run(_drive(mw, _scope("/mcp")))
    assert status == 401
    # Authorized forms pass.
    for headers in (
        [(b"authorization", f"Bearer {_KEY}".encode())],
        [(b"x-api-key", _KEY.encode())],
    ):
        status, _, _ = _run(_drive(mw, _scope("/sse", headers=headers)))
        assert status == 200, headers
        status, _, _ = _run(_drive(mw, _scope("/messages/x", headers=headers)))
        assert status == 200, headers
    # Probes and unrelated paths are not gated by this predicate.
    status, _, _ = _run(_drive(mw, _scope("/healthz")))
    assert status == 200
    status, _, _ = _run(_drive(mw, _scope("/other")))
    assert status == 200


def test_auth_gate_streamable_401s_anonymous_mcp(monkeypatch):
    monkeypatch.setenv(_KEY_ENV, _KEY)
    mw = _auth_mw(mcp, mcp._protected_streamable_path)

    status, _, _ = _run(_drive(mw, _scope("/mcp")))
    assert status == 401
    status, _, _ = _run(
        _drive(mw, _scope("/mcp", headers=[(b"authorization", f"Bearer {_KEY}".encode())]))
    )
    assert status == 200
    # The SSE route is NOT gated on this transport (it does not exist there).
    status, _, _ = _run(_drive(mw, _scope("/sse")))
    assert status == 200


def test_wiring_predicates_are_structural_not_add_middleware():
    """P1-10 structural pin: the transport apps are wrapped, not mutated.

    Assertions are on CODE LINES only — the explanatory comments in ``main``
    deliberately mention ``add_middleware`` to record why it is gone.
    """
    import inspect

    code_lines = [
        line.split("#", 1)[0].strip()
        for line in inspect.getsource(mcp.main).splitlines()
    ]
    code = "\n".join(code_lines)
    assert "add_middleware(" not in code, "middleware must be applied by wrapping the built app"
    assert "protected=_protected_sse_path" in code
    assert "protected=_protected_streamable_path" in code
    # The gate wraps a memory-header-wrapped app; the health probes are added
    # to the transport app FIRST (a middleware wrapper has no add_route).
    assert code.count("_memory_header_wrapped(app)") == 2
    assert code.count("_with_mcp_health(app)") == 2
    assert "_with_mcp_health(_memory_header_wrapped" not in code, (
        "health routes must attach to the Starlette app, not to a wrapper"
    )
    assert code.count("uvicorn.run(app,") == 2, "both HTTP transports run the wrapped app"


def test_health_routes_reachable_through_the_wrapped_stack(monkeypatch):
    """The probes must survive the middleware wrapping — the wiring order is
    load-bearing: ``_with_mcp_health`` has to run on the Starlette transport
    app, because a middleware wrapper has no ``add_route`` (it fails loudly
    now instead of silently 404ing the probes)."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    monkeypatch.setenv(_KEY_ENV, _KEY)
    app = mcp._with_mcp_health(Starlette())
    assert {r.path for r in app.routes} >= {"/healthz", "/readyz"}
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200

    # Calling it on a bare middleware wrapper is a loud startup error.
    with pytest.raises(AttributeError):
        mcp._with_mcp_health(mcp._memory_header_wrapped(Starlette()))


def test_health_routes_are_attached_before_the_gate(monkeypatch):
    """Alias kept for readability: see the reachability test above."""
    from starlette.applications import Starlette

    monkeypatch.setenv(_KEY_ENV, _KEY)
    app = mcp._with_mcp_health(Starlette())
    assert {r.path for r in app.routes} >= {"/healthz", "/readyz"}


# ===========================================================================
# 4 · End-to-end: both P1-7 and P1-10 through the real middleware stack
# ===========================================================================


def test_end_to_end_throttle_via_middleware_and_admin_key(monkeypatch, _throttle_env):
    """Drive the real stack (RagClientAuth → MemoryHeader → app) with a
    deployment key and let the tool body call the gate: the caller's bucket
    fills and then refuses, proving the gate now throttles on the request
    path (not just in a unit call)."""
    mod = _throttle_env
    monkeypatch.setenv(_KEY_ENV, _KEY)
    dm = _StubDM()
    mw = _auth_mw(mod, mod._protected_sse_path)
    headers = [(b"authorization", f"Bearer {_KEY}".encode())]

    def _gate(pw):
        """Call the gate the way a tool body does — RETURN the raised
        ToolError so the harness can inspect it (the real MCP layer would
        serialise it back to the caller)."""
        try:
            return mod._check_unlocked_or_password(dm, "secret", pw)
        except mod.ToolError as exc:
            return exc

    # Two wrong guesses (PW_MAX_FAILURES=2) then the throttle — all through
    # the authenticated SSE surface.
    status, _, captured = _run(_drive(mw, _scope("/messages/x", headers=headers), lambda: _gate("wrong")))
    assert status == 200
    assert isinstance(captured.get("result"), mod.ToolError)
    assert captured["cid"] == "10.9.0.1"  # peer identity (admin key → D10 identity)

    _run(_drive(mw, _scope("/messages/x", headers=headers), lambda: _gate("wrong")))
    status, _, captured = _run(
        _drive(mw, _scope("/messages/x", headers=headers), lambda: _gate("correct-pw"))
    )
    assert status == 200
    err = captured.get("result")
    assert isinstance(err, mod.ToolError)
    assert "Too many password attempts" in str(err)

    # Anonymous remains blocked (P1-10) on the same surface.
    status, _, _ = _run(_drive(mw, _scope("/messages/x")))
    assert status == 401
