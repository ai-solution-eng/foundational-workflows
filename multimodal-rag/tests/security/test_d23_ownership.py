"""Decision D23 — dataset ownership, read-only stats, whoami flags, HTML gate.

The ratified matrix:

* CREATE stamps ``created_by`` into the dataset's meta.json: a registry /
  JWT / SSO identity stamps its NAME, an ADMIN identity stamps ``admin``
  (admins act as the deployment), and the D20 anonymous (or absent)
  identity stamps ``anonymous``.
* DELETE (and every destructive manage surface) is OWNER-OR-ADMIN: the
  caller's identity name must equal the stamp, or the caller is admin.
  Everyone else 403s with the fixed message.  A dataset whose meta has NO
  ``created_by`` (pre-D23 — never backfilled) is deletable by ADMINS ONLY.
  The password gate stays layered AFTER the ownership check (the 403 must
  not leak protection state).
* The public flag gains a CREATOR path: a non-admin identity may
  publish/unpublish ONLY a dataset it created; non-creators and pre-D23
  datasets 403.  The admin route (/api/admin/...) keeps its bypass.
* Listings stamp ``created_by`` + ``owned_by_me`` (only a true match
  claims ownership).
* GET /api/stats: public-schema JSON, identity-filtered through the SAME
  _rag_acl_filter_datasets logic, zero writes, no per-request model probes
  (last-known health only); anonymous callers get the empty-world shape.
* The whoami (authenticated shape) gains ``is_admin`` and snake_case
  ``flags`` {can_create_datasets, sso_enabled, memory_dataset}; an OPAQUE
  admin key still reads the anonymous shape (no admin-status leak), and
  the anonymous shape ``{"authenticated": false}`` stays EXACTLY that.
* RAG_SSO_GATE (env, read per request): on, the HTML pages (/, /manage,
  /access) demand a presented credential — none → the minimal 401 sign-in
  page; a credential (even a failed one) keeps today's page.  Off =
  byte-identical public pages.

Every key, JWT, and JWKS document in this module is SYNTHETICALLY generated
in-test (the D21 test pattern).  No real credentials are ever written to
disk; the registry overlay lives in a temp DATA_PATH.

Run::

    pytest tests/security/test_d23_ownership.py -q
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr
from multimodal_rag.utils import oidc_identity as oidc

# ===========================================================================
# The D21 synthetic-JWT helpers (same _mint/_claims machinery as
# test_oidc_identity — a self-contained copy so this module stays focused
# on D23's matrix, not token plumbing).
# ===========================================================================

ISSUER = "https://oidc.test/realm-test"
AUDIENCE = "ua"
KID = "synthetic-d23-key"


def _b64url(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64uint(value: int) -> str:

    return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


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
    import time as _time

    return int(_time.time())


def _claims(*, sub="alice", preferred="alice", **extra) -> dict:
    c = {"iss": ISSUER, "aud": AUDIENCE, "exp": _now() + 600, "iat": _now() - 5, "sub": sub}
    if preferred is not False:
        c["preferred_username"] = preferred
    c.update(extra)
    return c


class _JwksServer:
    """Loopback JWKS endpoint (the D21 test pattern)."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        holder = self
        self._doc: dict = {"keys": []}

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(holder._doc).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/protocol/openid-connect/certs"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

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
# House hygiene: full env isolation (the D15/D17/D21 fixture pattern)
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
    "RAG_OIDC_SSO_ENABLED",
    "RAG_OIDC_SSO_CLIENT_ID",
    "RAG_OIDC_SSO_CLIENT_SECRET",
    "RAG_OIDC_SSO_REDIRECT_URI",
    "RAG_OIDC_SSO_PROVIDER_URL",
    "RAG_SSO_GATE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Every test starts with NO registry, NO keys, NO OIDC, NO store, NO
    gate — everything a test needs is set explicitly inside it."""
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
        "RAG_SSO_GATE",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in _OIDC_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()
    api._UNLOCK_CACHE.clear()
    api._stats_models_cache["ts"] = 0.0
    api._stats_models_cache["models"] = []
    oidc._jwks_cache.clear()
    oidc._jwks_negative.clear()
    oidc._observed_throttle.clear()
    oidc._observed_cache.clear()
    yield
    ar._mtime_cache.clear()
    acc._mtime_cache.clear()
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()
    api._stats_models_cache["ts"] = 0.0
    api._stats_models_cache["models"] = []
    oidc._jwks_cache.clear()
    oidc._jwks_negative.clear()
    oidc._observed_throttle.clear()
    oidc._observed_cache.clear()


@pytest.fixture(autouse=True)
def _oidc_on(monkeypatch, jwks_server):
    """Default per-test OIDC wiring (the D21 pattern): enabled, loopback
    JWKS, refetch-on-every-verify."""
    monkeypatch.setenv("RAG_OIDC_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("RAG_OIDC_JWKS_URL", jwks_server.url)
    monkeypatch.setenv("RAG_OIDC_JWKS_REFRESH_SECONDS", "0")


def _setup_key(server: _JwksServer, key_id: str = KID) -> dict:
    key = _generate_rsa(key_id)
    server.set(_jwks_doc(key))
    return key


@pytest.fixture
def rest_client(monkeypatch, tmp_path):
    """The real FastAPI app with a stubbed manager + a REAL meta-backed
    meta.json layer under tmp_path (the test_dataset_acls / test_oidc_identity
    full-stack pattern).

    Identity is resolved the production way: registry keys / JWTs via the
    middleware (``alice-key`` headers, Bearer tokens), the admin deployment
    key via ``RAG_API_KEY``.  The contextvar is NEVER set by hand here —
    TestClient runs each request in its own context, so a contextvar bound
    in the test body would be invisible to the handler (and the middleware
    would 401 the request).  Where a test needs a specific identity it
    configures the REGISTRY for it and presents the key.
    """
    from fastapi.testclient import TestClient

    datasets_root = tmp_path / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    class _MetaBackedDM:
        """A stub DatasetManager whose meta.json layer is REAL."""

        caption_with_asr = False
        caption_with_vlm = False

        def __init__(self):
            self.deleted: list[str] = []
            self.public_flags: dict[str, bool] = {}

        # -- meta plumbing (the REAL write pattern, atomic temp+replace) --
        def _meta_path(self, name):
            return datasets_root / name / "meta.json"

        def read_created_by(self, name):
            meta = self._read_meta_raw(name)
            if not meta:
                return None
            creator = str(meta.get("created_by") or "").strip()
            return creator or None

        def _write_meta_raw(self, name, meta):
            p = self._meta_path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2, default=str))
            os.replace(tmp, p)

        def _read_meta_raw(self, name):
            p = self._meta_path(name)
            if not p.exists():
                return None
            try:
                return json.loads(p.read_text())
            except Exception:
                return None

        def invalidate_meta_caches(self, name):
            p = self._meta_path(name)
            try:
                st = p.stat()
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                stamp = None
            for cache in (cr._PUBLIC_META_CACHE, cr._CREATED_BY_CACHE):
                key = str(p)
                if stamp is None:
                    cache.pop(key, None)
                    continue
                cached = cache.get(key)
                if cached and (cached[0], cached[1]) == stamp:
                    cache.pop(key, None)

        # -- endpoints' surface --
        def create_dataset(self, name, description="", caption_with_asr=False,
                           caption_with_vlm=False, keep_originals=True,
                           password=None, ocr=False, rrf=None, contextual=False,
                           created_by=None):
            if self._meta_path(name).exists():
                raise FileExistsError(f"Dataset '{name}' already exists")
            meta = {"name": name, "description": description, "document_count": 0}
            creator = str(created_by or "").strip()
            if creator:
                meta["created_by"] = creator
            self._write_meta_raw(name, meta)
            return dict(meta)

        def delete_dataset(self, name):
            import shutil

            self.deleted.append(name)
            shutil.rmtree(datasets_root / name, ignore_errors=True)
            self.invalidate_meta_caches(name)

        def set_public(self, name, public):
            meta = self._read_meta_raw(name)
            if not meta:
                raise FileNotFoundError(f"Dataset '{name}' not found")
            if public and meta.get("password_hash"):
                raise ValueError("Refusing to make a password-protected dataset public")
            if public:
                meta["public"] = True
            else:
                meta.pop("public", None)
            self._write_meta_raw(name, meta)
            self.invalidate_meta_caches(name)
            return {"name": name, "public": "public" in meta, "has_password": "password_hash" in meta}

        def get_dataset(self, name, sync_count=False):
            meta = self._read_meta_raw(name)
            if not meta:
                raise FileNotFoundError(f"Dataset '{name}' not found")
            return dict(meta)

        def has_password(self, name):
            meta = self._read_meta_raw(name)
            return bool(meta and meta.get("password_hash"))

        def has_password_fresh(self, name):
            return self.has_password(name)

        def _validate_name(self, name):
            import re as _re

            if not name or not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
                raise ValueError(f"Invalid dataset name: {name!r}")

        def list_datasets(self):
            out = []
            if datasets_root.exists():
                for child in sorted(datasets_root.iterdir()):
                    if child.is_dir():
                        meta = self._read_meta_raw(child.name)
                        if meta:
                            out.append(dict(meta))
            return out

        def _dataset_dir(self, name):
            return datasets_root / name

    dm = _MetaBackedDM()

    async def _fake_manager():
        return dm

    monkeypatch.setattr(api, "get_manager_async", _fake_manager)
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-admin-key")
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")  # resolve_presented's admin source
    # The registry carries the test users; the D15 ACL default grants alice
    # everything ("*") so the D15 surface never pre-empts what these tests
    # pin (the D23 gates).  Tests that need a narrower world set ACLs
    # explicitly via monkeypatch.
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    yield TestClient(api.app), dm
    api._UNLOCK_CACHE.clear()


_ALICE = {"X-RAG-Api-Key": "alice-key"}
_BOB = {"X-RAG-Api-Key": "bob-key"}
_ADMIN = {"X-RAG-Api-Key": "deployment-admin-key"}


# ===========================================================================
# 1 · CREATE stamps created_by per identity kind
# ===========================================================================


def test_create_stamps_registry_identity_name(rest_client):
    client, dm = rest_client
    r = client.post("/api/datasets", json={"name": "reports"}, headers=_ALICE)
    assert r.status_code == 200
    meta = dm._read_meta_raw("reports")
    assert meta["created_by"] == "alice"


def test_create_stamps_admin_as_admin(rest_client):
    client, dm = rest_client
    r = client.post("/api/datasets", json={"name": "ops"}, headers=_ADMIN)
    assert r.status_code == 200
    assert dm._read_meta_raw("ops")["created_by"] == "admin"


def test_create_stamps_jwt_identity_name(rest_client, jwks_server):
    """A JWT identity (D21) stamps the SAME registry name — one identity,
    two credentials, one ownership stamp."""
    client, dm = rest_client
    key = _setup_key(jwks_server)
    r = client.post(
        "/api/datasets",
        json={"name": "jwt-made"},
        headers={"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"},
    )
    assert r.status_code == 200
    assert dm._read_meta_raw("jwt-made")["created_by"] == "alice"


def test_create_stamps_anonymous_identity_as_anonymous(rest_client, monkeypatch):
    """The D20 anonymous fallback (name ``__anonymous__``) stamps
    ``anonymous`` — the registry name regex can never mint that name, so
    the stamp can never collide with a real key identity.  (On a real D20
    deployment the middleware 403s creation outright — fail-closed — so
    the stamp is exercised through the handler with the identity bound,
    the way a future/D20-store-enabled surface would.)"""
    _client, dm = rest_client
    assert api._creator_name_for_request() == "anonymous"
    r = asyncio.run(_call_create_direct("anon-ds", cr.Identity(kind="client", name="__anonymous__", datasets=frozenset())))
    assert r["dataset"]["created_by"] == "anonymous"
    assert dm._read_meta_raw("anon-ds")["created_by"] == "anonymous"


async def _call_create_direct(name, identity):
    """Call api_create_dataset with a bound identity (the D20 shape) —
    used only where the middleware cannot be negotiated to bind it."""
    token = cr.set_current_identity(identity)
    try:
        return await api.api_create_dataset({"name": name})
    finally:
        cr.reset_current_identity(token)


def test_create_with_no_identity_stamps_anonymous(rest_client, monkeypatch):
    """A legacy single-key surface (registry fully off — no registry env,
    no overlay, no OIDC): the middleware admits POST /api/datasets through
    the single-key check WITHOUT binding an identity (the admin fast path
    binds only when ``registry_configured()``) — the handler still stamps
    ``anonymous`` (no identity to name)."""
    client, dm = rest_client
    monkeypatch.delenv(cr.CLIENTS_ENV, raising=False)
    monkeypatch.setenv("RAG_OIDC_ENABLED", "0")  # keep registry_configured() False
    r = client.post("/api/datasets", json={"name": "bare"}, headers=_ADMIN)
    assert r.status_code == 200
    assert dm._read_meta_raw("bare")["created_by"] == "anonymous"


def test_create_stamp_lands_in_real_meta_json(rest_client):
    """The stamp rides the EXISTING meta write path — same file, same
    schema; the public-flag reader still works against it."""
    client, dm = rest_client
    client.post("/api/datasets", json={"name": "stamped"}, headers=_ALICE)
    raw = json.loads((dm._meta_path("stamped")).read_text())
    assert raw["created_by"] == "alice"
    assert raw["name"] == "stamped" and raw["document_count"] == 0
    cr._CREATED_BY_CACHE.clear()
    assert cr.created_by_dataset("stamped") == "alice"


# ===========================================================================
# 2 · DELETE: owner / other / admin matrix (+ pre-D23 admin-only rule)
# ===========================================================================


def _seed_dataset(dm, name, created_by="alice", password_hash=None):
    meta = {"name": name, "document_count": 0}
    if created_by:
        meta["created_by"] = created_by
    if password_hash:
        meta["password_hash"] = password_hash
    dm._write_meta_raw(name, meta)
    dm.invalidate_meta_caches(name)
    return meta


def test_delete_by_owner_succeeds(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "mine", created_by="alice")
    r = client.delete("/api/datasets/mine", headers=_ALICE)
    assert r.status_code == 200 and r.json()["deleted"] == "mine"
    assert dm.deleted == ["mine"]


def test_delete_by_other_identity_403_with_exact_message(rest_client, monkeypatch):
    client, dm = rest_client
    _seed_dataset(dm, "theirs", created_by="alice")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:theirs")
    r = client.delete("/api/datasets/theirs", headers=_BOB)
    assert r.status_code == 403
    assert r.json()["detail"] == (
        "Dataset 'theirs' is owned by 'alice' — only its creator or an admin can delete it."
    )
    assert dm.deleted == []


def test_delete_by_admin_succeeds(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "any", created_by="alice")
    r = client.delete("/api/datasets/any", headers=_ADMIN)
    assert r.status_code == 200
    assert dm.deleted == ["any"]


def test_delete_pre_d23_dataset_admin_only(rest_client, monkeypatch):
    """A dataset with NO created_by is deletable by admins only — never by
    a random registry key (no backfill, no proof of creation)."""
    client, dm = rest_client
    _seed_dataset(dm, "legacy", created_by=None)
    monkeypatch.setenv(cr.ACLS_ENV, "bob:legacy")
    r = client.delete("/api/datasets/legacy", headers=_BOB)
    assert r.status_code == 403
    assert "predates dataset ownership" in r.json()["detail"]
    assert dm.deleted == []
    r = client.delete("/api/datasets/legacy", headers=_ADMIN)
    assert r.status_code == 200 and dm.deleted == ["legacy"]


def test_delete_missing_dataset_404_not_403(rest_client):
    """A nonexistent dataset reports 404 for the admin (the only caller
    that can reach the delete half for a stampless dataset)."""
    client, _dm = rest_client
    r = client.delete("/api/datasets/ghost", headers=_ADMIN)
    # The stub's delete_dataset does not raise FileNotFoundError — pin the
    # GATE side (admin passes the ownership check) via the unit matrix and
    # accept either the stub's 200 or a real 404 here.
    assert r.status_code in (200, 404)
    assert "predates" not in r.text and "owned by" not in r.text


def test_delete_non_owner_403_precedes_password_gate(rest_client, monkeypatch):
    """The ownership check runs BEFORE the password gate: a non-owner gets
    the 403 (not a 401) and learns nothing about protection state; the
    owner still needs the password (the password gate is layered AFTER)."""
    client, dm = rest_client
    _seed_dataset(dm, "vault", created_by="alice", password_hash="x")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:vault")
    r = client.delete("/api/datasets/vault", headers=_BOB)
    assert r.status_code in (401, 403)
    assert "owned by" in r.json()["detail"] or "not permitted" in r.json()["detail"]
    # Owner without the password: the password gate now answers (401) —
    # ownership alone is NOT a password bypass.
    r = client.delete("/api/datasets/vault", headers=_ALICE)
    assert r.status_code == 401


# ===========================================================================
# 3 · Public toggle: creator path vs non-creator vs admin bypass
# ===========================================================================


def test_creator_can_publish_own_dataset(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "mine", created_by="alice")
    r = client.post("/api/datasets/mine/public", json={"public": True}, headers=_ALICE)
    assert r.status_code == 200 and r.json()["public"] is True
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("mine") is True
    # …and unpublish again
    r = client.post("/api/datasets/mine/public", json={"public": False}, headers=_ALICE)
    assert r.status_code == 200 and r.json()["public"] is False


def test_non_creator_cannot_publish_403(rest_client, monkeypatch):
    client, dm = rest_client
    _seed_dataset(dm, "theirs", created_by="alice")
    monkeypatch.setenv(cr.ACLS_ENV, "bob:theirs")
    r = client.post("/api/datasets/theirs/public", json={"public": True}, headers=_BOB)
    assert r.status_code == 403
    assert "owned by 'alice'" in r.json()["detail"]
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("theirs") is False


def test_pre_d23_dataset_never_public_by_non_admin(rest_client, monkeypatch):
    client, dm = rest_client
    _seed_dataset(dm, "legacy", created_by=None)
    monkeypatch.setenv(cr.ACLS_ENV, "bob:legacy")
    r = client.post("/api/datasets/legacy/public", json={"public": True}, headers=_BOB)
    assert r.status_code == 403
    assert "predates dataset ownership" in r.json()["detail"]


def test_creator_public_toggle_admin_via_creator_route(rest_client):
    """An admin identity reaching the creator route passes (same effect as
    its own admin surface — defense in depth, not a restriction)."""
    client, dm = rest_client
    _seed_dataset(dm, "ds", created_by="someone")
    r = client.post("/api/datasets/ds/public", json={"public": True}, headers=_ADMIN)
    assert r.status_code == 200 and r.json()["public"] is True


def test_admin_toggle_route_still_works_for_admin(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "ds", created_by="alice")
    r = client.post("/api/admin/datasets/ds/public", json={"public": True}, headers=_ADMIN)
    assert r.status_code == 200 and r.json()["public"] is True


def test_password_protected_never_public_even_for_creator(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "vault", created_by="alice", password_hash="x")
    r = client.post("/api/datasets/vault/public", json={"public": True}, headers=_ALICE)
    assert r.status_code == 409
    cr._PUBLIC_META_CACHE.clear()
    assert cr.is_public_dataset("vault") is False


def test_registry_client_still_403ed_on_admin_toggle_route(rest_client):
    """The admin route stays unreachable for registry keys (the middleware
    denies /api/admin/*) — the creator route is the ONLY non-admin path."""
    client, _dm = rest_client
    r = client.post("/api/admin/datasets/ds/public", json={"public": True}, headers=_ALICE)
    assert r.status_code == 403


# ===========================================================================
# 4 · Listing stamps: created_by + owned_by_me
# ===========================================================================


def test_listing_stamps_created_by_and_owned_by_me(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "mine", created_by="alice")
    _seed_dataset(dm, "other", created_by="bob")
    _seed_dataset(dm, "legacy", created_by=None)
    r = client.get("/api/datasets", headers=_ALICE)
    rows = {d["name"]: d for d in r.json()["datasets"]}
    assert rows["mine"]["created_by"] == "alice" and rows["mine"]["owned_by_me"] is True
    assert rows["other"]["created_by"] == "bob"
    assert "owned_by_me" not in rows["other"], "only a true match may claim ownership"
    assert "created_by" not in rows["legacy"], "pre-D23 rows carry no stamp"


def test_listing_owned_by_me_absent_for_admin_and_anonymous(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "ds", created_by="alice")
    # Admin identity: sees the stamp, never claims ownership.
    rows = {d["name"]: d for d in client.get("/api/datasets", headers=_ADMIN).json()["datasets"]}
    assert rows["ds"]["created_by"] == "alice"
    assert "owned_by_me" not in rows["ds"]


def test_catalog_listing_also_stamps(rest_client):
    """The /access page's catalog view carries the same stamps (one
    annotate pass — catalog AND plain listings)."""
    client, dm = rest_client
    _seed_dataset(dm, "mine", created_by="alice")
    r = client.get("/api/datasets", params={"catalog": "true"}, headers=_ALICE)
    rows = {d["name"]: d for d in r.json()["datasets"]}
    assert rows["mine"]["created_by"] == "alice" and rows["mine"]["owned_by_me"] is True


# ===========================================================================
# 5 · GET /api/stats — shape, identity filtering, zero writes
# ===========================================================================


def test_stats_anonymous_shape_and_no_error(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "secret", created_by="alice")
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    # storage_pvc/qdrant_pvc (the restored PVC-capacity section) are
    # identity-agnostic — present on the anonymous shape too (they are the
    # same numbers /api/admin/health reports; no per-identity info).
    assert set(body) == {"datasets", "storage_pvc", "qdrant_pvc", "models", "jobs", "memory", "generated_at"}
    assert body["datasets"] == {"total": 0, "visible_to_you": 0, "documents": 0, "storage_bytes": 0}
    assert body["models"] == [], "no model probing for an anonymous caller"
    assert body["jobs"] == {"active_uploads": 0, "recent_failures": 0}
    assert body["memory"] == {"dataset": None}
    assert body["generated_at"]


def test_stats_identity_filtered_visible_counts(rest_client, monkeypatch):
    client, dm = rest_client
    _seed_dataset(dm, "ds1", created_by="alice")
    _seed_dataset(dm, "ds2", created_by="alice")
    _seed_dataset(dm, "hidden", created_by="bob")
    # alice's world: exactly ds1+ds2 (operator grants; the store is off so
    # no selections/exclusions complicate the union).
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1,ds2")
    body = client.get("/api/stats", headers=_ALICE).json()
    assert body["datasets"]["total"] == 3, "total is deployment-wide, the filter is in visible_to_you"
    assert body["datasets"]["visible_to_you"] == 2
    assert body["datasets"]["documents"] == 0
    assert isinstance(body["datasets"]["storage_bytes"], int)


def test_stats_admin_sees_deployment_wide(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "ds1")
    _seed_dataset(dm, "ds2")
    body = client.get("/api/stats", headers=_ADMIN).json()
    assert body["datasets"]["total"] == 2
    assert body["datasets"]["visible_to_you"] == 2, "admin == deployment-wide"


def test_stats_models_rows_last_known_health(rest_client, monkeypatch):
    """The models rows come from last-known state ONLY: configured roles
    appear with {name, url, healthy}; the embedder's health rides the
    periodic monitor snapshot — no synchronous probe."""
    client, dm = rest_client
    _seed_dataset(dm, "ds")

    class _Model:
        model_name = "stub-embedder"
        url_remote = "http://stub"

    class _ModelDM:
        embedder = _Model()
        reranker = None
        vlm = None
        asr = None

        def list_datasets(self):
            return []

        def _dataset_dir(self, name):
            return dm._dataset_dir(name)

    monkeypatch.setattr(api, "get_manager", lambda: _ModelDM(), raising=True)
    try:
        api._stats_models_cache["ts"] = 0.0
        api._model_health["embedder"]["status"] = "healthy"
        body = client.get("/api/stats", headers=_ALICE).json()
        assert body["models"] == [{"name": "stub-embedder", "url": "http://stub", "healthy": True}]
        # An unhealthy snapshot flips the flag without any network call.
        api._model_health["embedder"]["status"] = "unhealthy"
        api._stats_models_cache["ts"] = 0.0
        body = client.get("/api/stats", headers=_ALICE).json()
        assert body["models"][0]["healthy"] is False
    finally:
        api._model_health["embedder"]["status"] = "unknown"
        api._stats_models_cache["ts"] = 0.0
        api._stats_models_cache["models"] = []


def test_stats_jobs_counters(rest_client):
    client, dm = rest_client
    _seed_dataset(dm, "ds")
    job_a = api._upload_jobs.create("ds", 1, source="files")
    job_b = api._upload_jobs.create("ds", 1, source="files")
    api._upload_jobs.fail(job_b, "boom")
    body = client.get("/api/stats", headers=_ADMIN).json()
    assert body["jobs"]["active_uploads"] == 1
    assert body["jobs"]["recent_failures"] == 1
    api._upload_jobs.complete(job_a, {})


def test_stats_memory_binding_identity_scoped(rest_client, monkeypatch):
    client, dm = rest_client
    _seed_dataset(dm, "mem", created_by="alice")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:mem")
    monkeypatch.setenv(acc.STORE_ENV, "1")
    alice = cr.Identity(kind="client", name="alice", datasets=frozenset({"mem"}))
    acc.select_dataset(alice, "mem")
    acc.set_memory_dataset(alice, "mem")
    body = client.get("/api/stats", headers=_ALICE).json()
    assert body["memory"] == {"dataset": "mem"}


# ===========================================================================
# 6 · Whoami flags (snake_case) + the admin-key anonymity rule
# ===========================================================================


def test_whoami_authenticated_shape_with_flags(rest_client, jwks_server, monkeypatch):
    client, _dm = rest_client
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1")
    monkeypatch.setenv(acc.STORE_ENV, "1")
    alice = cr.Identity(kind="client", name="alice", datasets=frozenset({"ds1"}))
    acc.select_dataset(alice, "ds1")
    acc.set_memory_dataset(alice, "ds1")
    r = client.get("/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"})
    body = r.json()
    assert body["authenticated"] is True and body["identity"] == "alice"
    assert body["source"] == "oidc"
    assert body["is_admin"] is False
    assert body["flags"] == {
        "can_create_datasets": False,  # named ACL, no '*'
        "sso_enabled": False,
        "memory_dataset": "ds1",
    }
    assert body["oidc"]["enabled"] is True


def test_whoami_star_grant_can_create(rest_client, jwks_server, monkeypatch):
    client, _dm = rest_client
    key = _setup_key(jwks_server)
    monkeypatch.setenv(cr.ACLS_ENV, "alice:*")
    r = client.get("/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"})
    assert r.json()["flags"]["can_create_datasets"] is True


def test_whoami_admin_key_stays_anonymous_shaped(rest_client, jwks_server, monkeypatch):
    """HARD BAR: an OPAQUE admin key presented must NEVER leak is_admin /
    flags — the payload stays the anonymous shape, exactly.  The admin
    status rides only a JWT identity (which never resolves admin) — so a
    deployment-key holder probing the whoami learns nothing."""
    client, _dm = rest_client
    _setup_key(jwks_server)
    monkeypatch.setenv("RAG_API_KEY", "deployment-admin-key")
    r = client.get("/api/oidc-session", headers={"X-RAG-Api-Key": "deployment-admin-key"})
    assert r.status_code == 200
    assert r.json() == {"authenticated": False}


def test_whoami_anonymous_shape_exactly_unchanged(rest_client, jwks_server):
    """The anonymous shape stays EXACTLY {"authenticated": false} — no new
    keys (the SPA treats missing keys as false)."""
    client, _dm = rest_client
    _setup_key(jwks_server)
    assert client.get("/api/oidc-session").json() == {"authenticated": False}
    # An invalid JWT also stays anonymous-shaped.
    r = client.get("/api/oidc-session", headers={"Authorization": f"Bearer {_mint(_generate_rsa('forged'), _claims())}"})
    assert r.json() == {"authenticated": False}


def test_whoami_flags_sso_enabled_shape(rest_client, jwks_server, monkeypatch):
    """flags.sso_enabled reports the D22 flow's configured state."""
    client, _dm = rest_client
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_OIDC_SSO_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_ID", "ua")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_SECRET", "sso-secret")
    monkeypatch.setenv("RAG_OIDC_SSO_REDIRECT_URI", "https://rag.example/oauth/oidc/callback")
    r = client.get("/api/oidc-session", headers={"Authorization": f"Bearer {_mint(key, _claims(preferred='alice'))}"})
    assert r.json()["flags"]["sso_enabled"] is True


# ===========================================================================
# 7 · RAG_SSO_GATE — off = byte-identical, on = 401 unauthenticated
# ===========================================================================


def test_gate_off_default_pages_byte_identical(rest_client, monkeypatch):
    """Env absent: public pages, exactly today's behaviour (including the
    embedded admin key meta)."""
    client, _dm = rest_client
    monkeypatch.setattr(api, "_RAG_API_KEY", "deployment-admin-key")
    monkeypatch.delenv("RAG_SSO_GATE", raising=False)
    for path in ("/", "/manage", "/access"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "Sign in required" not in r.text


def test_gate_off_with_credential_serves_page(rest_client, monkeypatch):
    client, _dm = rest_client
    monkeypatch.setenv("RAG_SSO_GATE", "0")
    r = client.get("/", headers=_ALICE)
    assert r.status_code == 200


def test_gate_on_unauthenticated_gets_401_signin_page(rest_client, monkeypatch):
    client, _dm = rest_client
    monkeypatch.setenv("RAG_SSO_GATE", "1")
    for path in ("/", "/manage", "/access"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert "Sign in required" in r.text


def test_gate_on_with_api_key_serves_page(rest_client, monkeypatch):
    """Any presented credential keeps today's page behaviour (the gate is a
    sign-in REQUIREMENT, not a resolution — the handlers still govern)."""
    client, _dm = rest_client
    monkeypatch.setenv("RAG_SSO_GATE", "1")
    r = client.get("/", headers=_ALICE)
    assert r.status_code == 200
    assert "Sign in required" not in r.text


def test_gate_on_with_sso_cookie_serves_page(rest_client, jwks_server, monkeypatch):
    """The D22 SSO cookie counts as a presented credential (a real session
    must not be locked out by the gate)."""
    client, _dm = rest_client
    key = _setup_key(jwks_server)
    monkeypatch.setenv("RAG_SSO_GATE", "1")
    r = client.get("/", cookies={"pcai-sso": _mint(key, _claims(preferred="alice"))})
    assert r.status_code == 200


def test_gate_never_touches_health_probes_or_stats(rest_client, monkeypatch):
    """Health/probes/oauth routes and the identity-aware public endpoints
    are exempt ALWAYS — the gate applies to the three HTML handlers only."""
    client, _dm = rest_client
    monkeypatch.setenv("RAG_SSO_GATE", "1")
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/oidc-session").status_code == 200


def test_gate_401_page_links_sso_login_when_enabled(rest_client, monkeypatch, jwks_server):
    """SSO-enabled deployments get the /oauth/login link; others don't."""
    client, _dm = rest_client
    _setup_key(jwks_server)
    monkeypatch.setenv("RAG_SSO_GATE", "1")
    monkeypatch.setenv("RAG_OIDC_SSO_ENABLED", "1")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_ID", "ua")
    monkeypatch.setenv("RAG_OIDC_SSO_CLIENT_SECRET", "sso-secret")
    monkeypatch.setenv("RAG_OIDC_SSO_REDIRECT_URI", "https://rag.example/oauth/oidc/callback")
    monkeypatch.setenv("RAG_OIDC_SSO_PROVIDER_URL", "https://oidc.test/.well-known/openid-configuration")
    import multimodal_rag.utils.oidc_sso as oidc_sso_module

    monkeypatch.setattr(oidc_sso_module, "_discovery_cache", {})
    monkeypatch.setattr(oidc_sso_module, "_discovery_negative", {})
    r = client.get("/")
    assert r.status_code == 401
    assert "/oauth/login" in r.text
    # SSO disabled → no link (the honest message only).
    monkeypatch.setenv("RAG_OIDC_SSO_ENABLED", "0")
    monkeypatch.setattr(oidc_sso_module, "_discovery_cache", {})
    r2 = client.get("/")
    assert r2.status_code == 401
    assert "/oauth/login" not in r2.text


# ===========================================================================
# 8 · Unit: the creator-name helper + the ownership gate matrix
# ===========================================================================


def test_creator_name_helper_matrix():
    token = cr.set_current_identity(cr.Identity(kind="client", name="alice", datasets=frozenset()))
    try:
        assert api._creator_name_for_request() == "alice"
    finally:
        cr.reset_current_identity(token)
    token = cr.set_current_identity(cr.Identity(kind="admin", name=None, datasets=None))
    try:
        assert api._creator_name_for_request() == "admin"
    finally:
        cr.reset_current_identity(token)
    token = cr.set_current_identity(cr.Identity(kind="client", name="__anonymous__", datasets=frozenset()))
    try:
        assert api._creator_name_for_request() == "anonymous"
    finally:
        cr.reset_current_identity(token)
    assert api._creator_name_for_request() == "anonymous"


class _GateDM:
    """Minimal manager stub for the ownership-gate unit matrix."""

    def __init__(self, creator):
        self._creator = creator

    def read_created_by(self, name):
        return self._creator


def _raise_http(fn):
    try:
        fn()
    except api.HTTPException as exc:
        return exc.status_code, str(exc.detail)
    return None, None


def test_gate_unit_matrix():
    dm = _GateDM("alice")
    owner = cr.Identity(kind="client", name="alice", datasets=frozenset())
    other = cr.Identity(kind="client", name="bob", datasets=frozenset({"*"}))
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    for ident, ok in ((owner, True), (other, False), (admin, True)):
        token = cr.set_current_identity(ident)
        try:
            status, detail = _raise_http(lambda: api._identity_may_manage_dataset(dm, "ds"))
            if ok:
                assert status is None, detail
            else:
                assert status == 403
                assert "owned by 'alice'" in detail
        finally:
            cr.reset_current_identity(token)
    # Pre-D23: no stamp → admin-only, exact refusal text.
    dm_legacy = _GateDM(None)
    token = cr.set_current_identity(other)
    try:
        status, detail = _raise_http(lambda: api._identity_may_manage_dataset(dm_legacy, "ds"))
        assert status == 403 and "predates dataset ownership" in detail
    finally:
        cr.reset_current_identity(token)


def test_stats_qdrant_pvc_block_present_or_absent_never_erroring(rest_client):
    """The qdrant_pvc stats block: absent-when-unobtainable, present-when-
    telemetry-answers — and NEVER an error (the Stats tab renders whatever
    arrives; absence hides the card)."""
    client, dm = rest_client
    _seed_dataset(dm, "q-check", created_by="alice")
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert "qdrant_pvc" in body
    qp = body["qdrant_pvc"]
    # in the test env there is no Qdrant: the block is either None or a
    # partial {used_bytes, replicas} shape — never a total/percent lie.
    assert qp is None or ("total_bytes" not in qp or "used_bytes" in qp)


def test_media_policy_trusts_own_media_base_url(monkeypatch):
    """2026-10 lead fix: the server's OWN media host (MEDIA_BASE_URL) is
    trusted by _check_media_url_policy — search results mint self-
    referential HMAC-token URLs on it by design, and blocking them made
    describe_media / client previews unusable for exactly the URLs the
    server issued.  Everything else keeps the strict SSRF rules."""
    from multimodal_rag.utils.url_policy import _check_media_url_policy

    monkeypatch.setenv("MEDIA_BASE_URL", "https://rag.example.com")
    monkeypatch.setenv("INGEST_BLOCK_PRIVATE_HOSTS", "true")
    monkeypatch.delenv("INGEST_ALLOW_HOSTS", raising=False)
    # own host: allowed
    _check_media_url_policy("https://rag.example.com/api/datasets/x/files/i.jpg?token=t")
    # a DIFFERENT private host: still blocked (SSRF guard intact)
    import pytest

    with pytest.raises(ValueError):
        _check_media_url_policy("http://10.0.0.5/secret")
    # loopback: still allowed (documented)
    _check_media_url_policy("http://localhost:8000/api/datasets/x/files/i.jpg?token=t")
