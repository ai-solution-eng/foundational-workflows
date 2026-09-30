"""Public-to-all-keys datasets (2026-10): admin flag, client grant, guards.

Pinned semantics:

* A dataset stamped "public": true in meta.json is AVAILABLE to every
  minted key for password-free self-selection (D16) - deliberately NOT an
  automatic grant (2026-10 revision: the user's checked set is their
  world; public datasets are opt-in and excludable like any other).
* ADMIN-ONLY toggle: the REST surface is POST /api/admin/datasets/{name}/public;
  registry clients are 403ed on /api/admin/* by the middleware, and the
  generic PATCH /api/datasets/{name} actively REJECTS the 'public' key
  (a client must never publish a dataset - capabilities travel with the
  credential).
* A password-protected dataset can NEVER be public - refused at write time
  (set_public -> 409) AND read time (is_public_dataset ignores the flag
  when a password_hash is present, so a hand-edited meta cannot publish).
* The D20 anonymous fallback identity is NOT a minted key: public datasets
  are invisible to it (fail-closed default preserved).
* Recreated datasets come back private (the flag dies with the dataset dir).

Run::

    pytest tests/security/test_public_datasets.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
from multimodal_rag.utils import access_store as acc
from multimodal_rag.utils import clients_registry as cr


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Deterministic registry + meta-cache state per test."""
    monkeypatch.setenv("RAG_API_KEY", "deployment-key")
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:private-reports")
    monkeypatch.setenv("DATA_PATH", "/nonexistent-data-path-for-tests")
    cr._PUBLIC_META_CACHE.clear()


# ---------------------------------------------------------------------------
# 1 - is_public_dataset: meta read + defense in depth
# ---------------------------------------------------------------------------


def test_public_flag_read_from_meta(monkeypatch, tmp_path):
    d = tmp_path / "datasets" / "reports"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"name": "reports", "public": True}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("reports") is True


def test_missing_or_corrupt_meta_is_private(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("ghost") is False
    d = tmp_path / "datasets" / "broken"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{not json")
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("broken") is False


def test_password_hash_never_public_even_if_stamped(monkeypatch, tmp_path):
    """Read-time guard: a hand-edited meta cannot publish a protected dataset."""
    d = tmp_path / "datasets" / "vault"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"public": True, "password_hash": "x"}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("vault") is False


def test_meta_cache_respects_mtime(monkeypatch, tmp_path):
    d = tmp_path / "datasets" / "reports"
    d.mkdir(parents=True)
    meta = d / "meta.json"
    meta.write_text(json.dumps({"name": "reports"}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("reports") is False
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    meta.write_text(json.dumps({"name": "reports", "public": True}))
    st2 = meta.stat()
    os.utime(meta, ns=(st2.st_atime_ns, st2.st_mtime_ns + 1_000_000))
    assert cr.is_public_dataset("reports") is True


# ---------------------------------------------------------------------------
# 2 - dataset_allowed: identity matrix
# ---------------------------------------------------------------------------


def test_public_dataset_is_availability_not_grant(monkeypatch, tmp_path):
    """2026-10 revision: public makes the dataset SELECTABLE (password-free),
    it does NOT force it into every key's world."""
    d = tmp_path / "datasets" / "reports"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"public": True}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    ident = cr.Identity(kind="client", name="alice", datasets=frozenset())
    assert cr.dataset_allowed(ident, "reports") is False  # not auto-granted
    assert cr.is_public_dataset("reports") is True        # but available


def test_anonymous_identity_denied_public(monkeypatch, tmp_path):
    """D20 preserved: the anonymous fallback is NOT a minted key."""
    d = tmp_path / "datasets" / "reports"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"public": True}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    anon = cr.Identity(kind="client", name="__anonymous__", datasets=frozenset())
    assert cr.dataset_allowed(anon, "reports") is False
    # ...but an explicit memory-dataset grant still works
    anon_mem = cr.Identity(kind="client", name="__anonymous__", datasets=frozenset({"mem"}))
    assert cr.dataset_allowed(anon_mem, "mem") is True


def test_admin_and_inactive_unchanged(monkeypatch, tmp_path):
    d = tmp_path / "datasets" / "reports"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"public": True}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    cr._PUBLIC_META_CACHE.clear()
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    assert cr.dataset_allowed(admin, "anything") is True
    assert cr.dataset_allowed(None, "reports") is True  # D15 inactive


# ---------------------------------------------------------------------------
# 3 - REST: admin toggle + PATCH rejection + end-to-end grant
# ---------------------------------------------------------------------------


@pytest.fixture
def rest(monkeypatch):
    """The real FastAPI app with a stubbed manager (offline)."""
    from fastapi.testclient import TestClient

    class _FakeDM:
        def __init__(self):
            self.public_flags = {}

        def list_datasets(self):
            return [
                {"name": "reports", "public": True},
                {"name": "notes", "public": False},
            ]

        def set_public(self, name, public):
            if name not in ("reports", "notes"):
                raise FileNotFoundError("Dataset not found: " + name)
            self.public_flags[name] = public
            return {"name": name, "public": public, "has_password": False}

        def get_dataset(self, name, check_embedder=True):
            if name not in ("reports", "notes"):
                raise FileNotFoundError("Dataset not found: " + name)
            return {"name": name}

        def has_password(self, name):
            return False

        def verify_password(self, name, pw):
            return False

        def update_dataset(self, name, updates):
            if "public" in updates:
                raise ValueError("The 'public' flag is admin-only: POST /api/admin/datasets/{name}/public")

    dm = _FakeDM()

    async def _fake_manager():
        return dm

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-key")
    return TestClient(api.app), dm


def test_admin_can_publish_and_unpublish(rest):
    client, dm = rest
    r = client.post("/api/admin/datasets/notes/public", json={"public": True},
                    headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200 and r.json()["public"] is True
    assert dm.public_flags["notes"] is True
    r = client.post("/api/admin/datasets/notes/public", json={"public": False},
                    headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 200 and r.json()["public"] is False


def test_client_key_cannot_reach_admin_toggle(rest):
    """The middleware 403s registry clients on /api/admin/*."""
    client, _ = rest
    r = client.post("/api/admin/datasets/reports/public", json={"public": True},
                    headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403


def test_generic_patch_rejects_public_key(rest):
    """HARD BAR: a client cannot publish via the generic metadata PATCH.

    2026-10 posture: the middleware denies a registry client BEFORE the
    handler — alice has no grant on ``reports``, and the D15 fail-closed
    path filter 403s any dataset-scoped route outside her world (the deny
    message says nothing about 'public', which is correct: the ACL layer
    must not advertise WHY a dataset is interesting).  The admin-only
    ``public``-key rejection (400, ValueError → HTTPException) sits one
    layer deeper and is pinned separately below via the deployment key.
    """
    client, _ = rest
    r = client.patch("/api/datasets/reports", json={"public": True},
                     headers={"X-RAG-Api-Key": "alice-key"})
    assert r.status_code == 403
    assert "not permitted for this API key" in r.json()["detail"]


def test_generic_patch_public_key_rejected_at_handler(rest):
    """Second layer: a caller who CAN reach the handler (the deployment key
    binds the admin identity) still cannot set ``public`` through the
    generic PATCH — the admin-only ValueError maps to 400 with the
    designated admin route named."""
    client, _ = rest
    r = client.patch("/api/datasets/notes", json={"public": True},
                     headers={"X-RAG-Api-Key": "deployment-key"})
    assert r.status_code == 400
    assert "admin-only" in r.json()["detail"]


def test_public_dataset_catalog_visibility_and_self_selection(rest, tmp_path, monkeypatch):
    """End-to-end, 2026-10 semantics (see module docstring): public is
    AVAILABILITY, not inclusion.

    * The plain listing keeps the D15 world-only filter — an unselected
      public dataset never force-enters a key's world (the ratified
      2026-09-24 ruling: a listing never shows names the key cannot use;
      the 2026-10 revision made public datasets opt-in like any other).
    * The /access catalog view (``?catalog=true``) ADVERTISES the public
      dataset (world ∪ public-available ∪ excluded) so the user can opt in,
      stamped ``public: true``.
    * A password-free self-selection (POST .../select) brings it INTO the
      world; deselecting EXCLUDES it again (hidden everywhere, but the
      catalog keeps it visible stamped ``excluded: true`` so the user can
      re-include).

    The flag is written to a real meta.json under tmp DATA_PATH because
    ``is_public_dataset`` (the middleware's defense-in-depth read) reads
    DISK, not the manager's in-memory rows — in production both read the
    same meta, so the fixture mirrors that.
    """
    (tmp_path / "datasets" / "reports").mkdir(parents=True)
    (tmp_path / "datasets" / "reports" / "meta.json").write_text(
        json.dumps({"name": "reports", "public": True})
    )
    (tmp_path / "datasets" / "notes").mkdir(parents=True)
    (tmp_path / "datasets" / "notes" / "meta.json").write_text(json.dumps({"name": "notes"}))
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.setenv(acc.STORE_ENV, "1")
    cr._PUBLIC_META_CACHE.clear()

    client, _ = rest
    H = {"X-RAG-Api-Key": "alice-key"}

    # Before selection: plain listing is world-only (alice's ACL names
    # 'private-reports', which the manager does not list here → empty);
    # the catalog advertises exactly the public dataset.
    names = [d["name"] for d in client.get("/api/datasets", headers=H).json()["datasets"]]
    assert names == []
    cat = client.get("/api/datasets", params={"catalog": "true"}, headers=H).json()["datasets"]
    assert [d["name"] for d in cat] == ["reports"]
    assert cat[0]["public"] is True

    # Password-free self-selection admits it (availability → inclusion).
    r = client.post("/api/datasets/reports/select", headers=H)
    assert r.status_code == 200
    names = [d["name"] for d in client.get("/api/datasets", headers=H).json()["datasets"]]
    assert names == ["reports"]
    assert "notes" not in names  # private stays hidden throughout

    # Deselect → exclusion hides it everywhere...
    r = client.post("/api/datasets/reports/deselect", headers=H)
    assert r.status_code == 200
    names = [d["name"] for d in client.get("/api/datasets", headers=H).json()["datasets"]]
    assert names == []
    # ...but the catalog keeps advertising it, stamped excluded, for re-inclusion.
    cat = client.get("/api/datasets", params={"catalog": "true"}, headers=H).json()["datasets"]
    by_name = {d["name"]: d for d in cat}
    assert by_name["reports"]["excluded"] is True


def test_anonymous_rest_denied_public_listing(rest):
    """No key presented with the registry configured -> not authenticated as
    a minted client -> the public flag grants nothing over the wire."""
    client, _ = rest
    r = client.get("/api/datasets")
    assert r.status_code in (401, 403)
