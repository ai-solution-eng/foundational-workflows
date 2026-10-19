"""P1-8 (audit 2026-10-16): an admin password reset must REVOKE what the old
password bought.

The defect: ``POST /api/admin/datasets/{name}/reset-password`` changed the
hash and invalidated the has_password cache, but every OTHER grant the old
password had produced stayed live —

  * unlock-cache entries (``_unlock_cache_del`` had exactly one caller,
    ``api_lock_dataset``, and it only ever deletes the ACTING identity's
    entry), so a third party's cached unlock kept serving reads after the
    reset; the docstring claimed revocation "where the client id is known";
  * D16 saved selection passwords, which act as implicit unlocks (step 0 of
    ``_require_dataset_password``).

Pinned here (negative + positive):

  * ``_unlock_cache_del_all`` purges EVERY caller's entry for the dataset,
    in-process AND through the Redis SCAN path, and leaves other datasets
    untouched;
  * the reset handler calls it and clears saved selection passwords, while
    KEEPING the selections themselves (a revocation must not silently
    re-write a user's world);
  * a password-protected dataset stops serving through a stale unlock after
    the reset (the reproduction the audit ran end-to-end).

Run::

    pytest tests/security/test_admin_reset_revocation.py -q
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Fresh DATA_PATH; registry/keys/store/trust knobs OFF unless set."""
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
        "RAG_TRUST_PROXY_IDENTITY",
        acc.STORE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    api._UNLOCK_CACHE.clear()
    acc._mtime_cache.clear()
    yield
    api._UNLOCK_CACHE.clear()
    acc._mtime_cache.clear()


class _StubDM:
    """The four DatasetManager calls the reset handler makes."""

    def __init__(self, root, password="pw-old"):
        self.root = root
        self.password = password
        self.calls = []

    def get_dataset(self, name, sync_count=True):
        if name not in ("reports", "notes"):
            raise FileNotFoundError(name)
        return {"name": name}

    def has_password_fresh(self, name):
        return bool(self.password)

    def set_password(self, name, new_password):
        self.calls.append(("set_password", name, new_password))
        self.password = new_password

    def _invalidate_has_password(self, name):
        self.calls.append(("invalidate", name))


def _request():
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/admin/datasets/reports/reset-password",
            "headers": [],
            "query_string": b"",
        }
    )


def _reset(name="reports", new_password="pw-new", dm=None):
    """Drive the handler directly, with the ADMIN identity bound the way the
    middleware binds it for a deployment key (the endpoint's defense-in-depth
    check reads the contextvar)."""

    async def _run():
        token = cr.set_current_identity(cr.Identity(kind="admin", name=None, datasets=None))
        try:
            return await api.api_admin_reset_dataset_password(name, _request(), {"new_password": new_password})
        finally:
            cr.reset_current_identity(token)

    return asyncio.run(_run())


def _stub_manager(monkeypatch, dm):
    async def _fake_manager():
        return dm

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)


# ===========================================================================
# 1 · the purge primitive
# ===========================================================================


def test_unlock_cache_del_all_purges_every_caller_in_process():
    """Every caller's entry for the dataset goes; other datasets stay."""
    api._UNLOCK_CACHE[("reports", "cid-a")] = (float("inf"), "pw-old")
    api._UNLOCK_CACHE[("reports", "cid-b")] = (float("inf"), "pw-old")
    api._UNLOCK_CACHE[("notes", "cid-a")] = (float("inf"), "pw-notes")
    try:
        removed = api._unlock_cache_del_all("reports")
        assert removed == 2
        assert api._unlock_cache_get("reports", "cid-a") is None
        assert api._unlock_cache_get("reports", "cid-b") is None
        assert api._unlock_cache_get("notes", "cid-a") == "pw-notes"
        assert api._unlock_cache_del_all("reports") == 0  # idempotent
    finally:
        api._UNLOCK_CACHE.clear()


def test_unlock_cache_del_all_uses_the_redis_scan_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Redis path deletes by the ${B}unlock:{dataset}:*${B} key prefix — the
    per-id delete cannot enumerate caller ids, so a SCAN is the only correct
    purge on a shared cache."""
    deleted: list[str] = []

    class _FakeRedis:
        def scan(self, cursor: int = 0, match: str = "", count: int = 0) -> tuple[int, list[str]]:
            assert match == "unlock:reports:*", match
            return (0, ["unlock:reports:cid-a", "unlock:reports:cid-b"])

        def delete(self, *keys: str) -> int:
            deleted.extend(keys)
            return len(keys)

    monkeypatch.setattr(api, "_get_redis", lambda: _FakeRedis())
    assert api._unlock_cache_del_all("reports") == 2
    assert sorted(deleted) == ["unlock:reports:cid-a", "unlock:reports:cid-b"]


# ===========================================================================
# 2 · the handler
# ===========================================================================


def test_reset_revokes_unlocks_and_saved_selection_passwords(monkeypatch, tmp_path):
    """THE P1-8 regression: after a reset, a stale unlock AND a saved
    selection password are both gone, the selection itself survives, and the
    response reports the revocation counts."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    dm = _StubDM(tmp_path)
    _stub_manager(monkeypatch, dm)
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset())
    acc.select_dataset(identity, "reports", "pw-old")  # saves the password
    assert acc.selection_password(identity, "reports") == "pw-old"
    api._UNLOCK_CACHE[("reports", "key:alice")] = (float("inf"), "pw-old")
    api._UNLOCK_CACHE[("reports", "203.0.113.7")] = (float("inf"), "pw-old")
    try:
        result = _reset(dm=dm)
        assert result["status"] == "ok"
        assert result["unlocks_revoked"] == 2
        assert result["selection_passwords_cleared"] == 1
        assert api._unlock_cache_get("reports", "key:alice") is None
        assert api._unlock_cache_get("reports", "203.0.113.7") is None
        assert acc.selection_password(identity, "reports") is None
        # The selection itself (and the user's world) is untouched.
        assert "reports" in acc.selections_for(identity)
        assert acc.dataset_allowed(identity, "reports") is True
    finally:
        api._UNLOCK_CACHE.clear()


def test_reset_leaves_other_datasets_alone(monkeypatch, tmp_path):
    """Negative control: a reset of 'reports' must not touch 'notes'."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    dm = _StubDM(tmp_path)
    _stub_manager(monkeypatch, dm)
    identity = cr.Identity(kind="client", name="alice", datasets=frozenset())
    acc.select_dataset(identity, "reports", "pw-old")
    acc.select_dataset(identity, "notes", "pw-notes")
    api._UNLOCK_CACHE[("notes", "key:alice")] = (float("inf"), "pw-notes")
    try:
        result = _reset(dm=dm)
        assert result["unlocks_revoked"] == 0
        assert result["selection_passwords_cleared"] == 1
        assert api._unlock_cache_get("notes", "key:alice") == "pw-notes"
        assert acc.selection_password(identity, "notes") == "pw-notes"
        assert acc.selection_password(identity, "reports") is None
    finally:
        api._UNLOCK_CACHE.clear()


def test_reset_with_store_off_still_revokes_unlocks(monkeypatch, tmp_path):
    """The unlock purge is store-independent (a single-user deployment has no
    selection store but does have an unlock cache)."""
    dm = _StubDM(tmp_path)
    _stub_manager(monkeypatch, dm)
    api._UNLOCK_CACHE[("reports", "default")] = (float("inf"), "pw-old")
    try:
        result = _reset(dm=dm)
        assert result["unlocks_revoked"] == 1
        assert result["selection_passwords_cleared"] == 0
    finally:
        api._UNLOCK_CACHE.clear()


def test_reset_to_passwordless_also_revokes(monkeypatch, tmp_path):
    """Removing the password entirely (new_password=null) is a reset too —
    it must not leave a stale unlock behind either."""
    dm = _StubDM(tmp_path)
    _stub_manager(monkeypatch, dm)
    api._UNLOCK_CACHE[("reports", "cid")] = (float("inf"), "pw-old")
    try:
        result = _reset(new_password=None, dm=dm)
        assert result["action"] == "removed"
        assert result["unlocks_revoked"] == 1
        assert api._unlock_cache_get("reports", "cid") is None
    finally:
        api._UNLOCK_CACHE.clear()


# ===========================================================================
# 3 · end-to-end: the stale unlock stops serving
# ===========================================================================


def test_stale_unlock_stops_serving_after_reset(monkeypatch, tmp_path):
    """The audit's reproduction: unlock the dataset, reset the password as
    an admin, then re-use the OLD unlock — it must no longer read.  Driven
    through the real REST middleware + handler (offline stub manager)."""
    from fastapi.testclient import TestClient

    class _ServingDM(_StubDM):
        def has_password(self, name):
            return bool(self.password)

        def verify_password(self, name, password):
            return password == self.password

        def list_datasets(self):
            return [{"name": "reports"}]

    dm = _ServingDM(tmp_path)
    _stub_manager(monkeypatch, dm)
    monkeypatch.setattr(api, "_RAG_API_KEY", "admin-key")
    client = TestClient(api.app)
    admin = {"X-RAG-Api-Key": "admin-key"}
    # 1 · unlock with the OLD password (warms the cache for this caller).
    r = client.post("/api/datasets/reports/unlock", json={"password": "pw-old"}, headers=admin)
    assert r.status_code == 200, r.text
    cid = "testclient"
    assert api._unlock_cache_get("reports", cid) is not None
    # 2 · admin reset.
    r = client.post(
        "/api/admin/datasets/reports/reset-password",
        json={"new_password": "pw-new"},
        headers={"X-RAG-Api-Key": "admin-key"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["unlocks_revoked"] >= 1
    # 3 · the old unlock is gone: the password is demanded again.
    assert api._unlock_cache_get("reports", cid) is None
    r = client.post("/api/datasets/reports/verify-password", json={"password": "pw-old"}, headers=admin)
    assert r.status_code == 403, r.text
    r = client.post("/api/datasets/reports/verify-password", json={"password": "pw-new"}, headers=admin)
    assert r.status_code == 200, r.text
    # …and a credential-less caller gets nothing at all (never a silent serve).
    r = client.post("/api/datasets/reports/verify-password", json={"password": "pw-old"})
    assert r.status_code == 401
    api._UNLOCK_CACHE.clear()
