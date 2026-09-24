"""File-backed admin overlay for the D15 clients registry (fleet decision D17).

The operator-side counterpart to the D16 access store: per-user keys are
minted and granted from the ``/access`` page (with an ADMIN key) instead of
hand-editing env vars — which a pod cannot do.  The overlay is a single JSON
file on the shared PVC:

    {DATA_PATH}/access/clients.json
    {
      "clients": {
        "alice": {"key": "…", "datasets": ["reports", "notes"]},
        "bob":   {"key": "…", "datasets": ["*"]}
      }
    }

Resolution (the union is computed by ``clients_registry.resolve_presented``
via :func:`overlay_clients` / :func:`overlay_acls` — both re-read per call):

    effective registry = RAG_API_KEY_CLIENTS env  ∪  overlay file
    effective ACLs     = RAG_DATASET_ACLS env     ∪  overlay file

An overlay entry NEVER widens admin power: only the deployment keys
(``RAG_API_KEY`` + the MCP key set) are admin — overlay clients are registry
clients with an ACL, exactly like env clients.  A name present in BOTH env
and overlay keeps its ENV key (env wins on conflict — the operator's
hand-written config is authoritative; delete the env entry to let the page
own the name).  ACLs are merged per name (env ∪ overlay datasets) rather
than replaced, so neither source silently revokes the other.

Security posture:

* The file holds PLAINTEXT keys — the same trust posture as
  ``RAG_API_KEY_CLIENTS`` (which is env/Secret material anyway) and the D16
  store's saved passwords.  0600, atomic temp+replace under the
  cross-process fcntl lock, mtime-cached reads.
* Key generation uses ``secrets.token_urlsafe`` (128-bit entropy).
* Key material is NEVER returned by any listing endpoint in full — mint
  responses carry it ONCE (the only time the operator can copy it);
  listings show a masked prefix (``0HeC9a…``).
* ``RAG_ACCESS_ADMIN_FILE`` (default: enabled when ``RAG_ACCESS_STORE`` is
  on) gates the whole module; when off it is inert and resolves nothing.

This module is RAG-local, stdlib-only, dependency-free; it never logs key
material.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
from pathlib import Path

from multimodal_rag.utils import access_store as _base

ADMIN_FILE_ENABLED_ENV = "RAG_ACCESS_ADMIN_FILE"
CLIENTS_FILE = "clients.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_mtime_cache: dict[str, tuple[int, int, dict]] = {}
_mtime_lock = threading.Lock()
_mtime_stamp: dict[str, tuple[int, int]] = {}

# Keep the writer lock machinery from access_store (per-path threading lock
# serializing in-process writers before they reach the fcntl lock).
_writer_locks: dict[str, threading.Lock] = {}
_writer_locks_guard = threading.Lock()


def admin_file_enabled() -> bool:
    """True when the admin overlay is switched on.

    Default: follows ``RAG_ACCESS_STORE`` (the D16 knob) — the overlay is
    meaningless without the per-identity store directory anyway.  An
    explicit ``RAG_ACCESS_ADMIN_FILE=0`` turns it off even then.
    """
    raw = os.environ.get(ADMIN_FILE_ENABLED_ENV, "").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    return _base.store_enabled()


def _file_path() -> Path:
    return Path(os.environ.get("DATA_PATH", "/data")) / "access" / CLIENTS_FILE


def _validate_name(name: str) -> str:
    name = str(name).strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "client name must be 1-64 chars of letters, digits, '.', '_' or '-' "
            "(must start alphanumeric)"
        )
    return name


def _load() -> dict:
    """Load the overlay file (mtime-cached; missing/corrupt → empty doc)."""
    path = _file_path()
    try:
        st = path.stat()
    except OSError:
        with _mtime_lock:
            _mtime_cache.pop(str(path), None)
        return {}
    key = str(path)
    stamp = (st.st_mtime_ns, st.st_size)
    with _mtime_lock:
        cached = _mtime_cache.get(key)
        if cached and (cached[0], cached[1]) == stamp:
            return cached[2]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    if not isinstance(doc.get("clients"), dict):
        doc["clients"] = {}
    with _mtime_lock:
        _mtime_cache[key] = (stamp[0], stamp[1], doc)
    return doc


def _writer_lock(key: str) -> threading.Lock:
    with _writer_locks_guard:
        lk = _writer_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _writer_locks[key] = lk
        return lk


def _mutate(fn) -> dict:
    """Read-modify-write the overlay under the file lock (atomic replace)."""
    path = _file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    with _writer_lock(key), _base._cross_process_lock(path.with_suffix(".lock")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            doc = {}
        except (json.JSONDecodeError, OSError):
            doc = {}
        if not isinstance(doc, dict) or not isinstance(doc.get("clients"), dict):
            doc = {"clients": {}}
        if not fn(doc):
            return doc
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            st = path.stat()
            with _mtime_lock:
                _mtime_cache[key] = (st.st_mtime_ns, st.st_size, doc)
        except OSError:
            pass
        return doc


# ---------------------------------------------------------------------------
# Public surface: the overlay view of the registry (merged into clients_registry)
# ---------------------------------------------------------------------------


def overlay_clients() -> dict:
    """``{key: name}`` from the overlay file (empty when disabled).

    Merged into the env registry by callers — on key conflicts with the env
    registry the ENV key wins (callers apply env first; a duplicate key with
    a different name is resolved by the env registry taking precedence).
    """
    if not admin_file_enabled():
        return {}
    clients: dict[str, str] = {}
    for name, entry in _load().get("clients", {}).items():
        if isinstance(entry, dict) and entry.get("key"):
            clients[str(entry["key"])] = str(name)
    return clients


def overlay_acls() -> dict:
    """``{name: frozenset(datasets)}`` from the overlay file (empty when off)."""
    if not admin_file_enabled():
        return {}
    acls: dict[str, frozenset] = {}
    for name, entry in _load().get("clients", {}).items():
        if isinstance(entry, dict):
            ds = entry.get("datasets")
            if isinstance(ds, list):
                acls[str(name)] = frozenset(str(d) for d in ds if d)
            else:
                acls[str(name)] = frozenset()
    return acls


# ---------------------------------------------------------------------------
# Admin operations (the /access page's admin panel)
# ---------------------------------------------------------------------------


def generate_key() -> str:
    """A fresh 128-bit client key (the only time key material is handed out)."""
    return secrets.token_urlsafe(18)


def mint_client(name: str, datasets: list | None = None, key: str | None = None) -> dict:
    """Create (or rotate the key of) an overlay client.

    *name* must pass :func:`_validate_name`.  *datasets* is the client's ACL
    (list of names; ``["*"]`` = everything; empty = nothing until granted).
    When *key* is given it replaces the stored one (rotation — e.g. the
    operator pastes a known value); otherwise a fresh key is generated and
    returned in the result dict (the ONLY copy the caller ever gets).
    """
    if not admin_file_enabled():
        raise _base.SelectionDenied("The admin key registry is not enabled on this deployment.")
    name = _validate_name(name)
    ds = [str(d).strip() for d in (datasets or []) if str(d).strip()]
    for d in ds:
        _validate_dataset_token(d)
    new_key = (str(key).strip() if key else "") or generate_key()
    if ":" in new_key or ";" in new_key or len(new_key) < 8:
        raise ValueError("custom key must be >=8 chars without ':' or ';'")
    result: dict = {}

    def _apply(doc: dict) -> bool:
        existing = doc["clients"].get(name)
        rotated = bool(existing and existing.get("key") != new_key)
        doc["clients"][name] = {"key": new_key, "datasets": ds}
        result["rotated"] = rotated
        return True

    _mutate(_apply)
    return {"name": name, "key": new_key, "datasets": ds, "rotated": result.get("rotated", False)}


def revoke_client(name: str) -> bool:
    """Remove an overlay client entirely (its key stops authenticating on the
    next request — everything is re-read per call).  Returns True when the
    name existed.  An ENV client with the same name is NOT touched (the env
    registry keeps authority over its own entries)."""
    if not admin_file_enabled():
        raise _base.SelectionDenied("The admin key registry is not enabled on this deployment.")
    name = _validate_name(name)
    removed = {"v": False}

    def _apply(doc: dict) -> bool:
        if name in doc["clients"]:
            del doc["clients"][name]
            removed["v"] = True
            return True
        return False

    _mutate(_apply)
    return removed["v"]


def grant_datasets(name: str, datasets: list) -> dict:
    """Replace an overlay client's dataset grant (the checkbox set)."""
    if not admin_file_enabled():
        raise _base.SelectionDenied("The admin key registry is not enabled on this deployment.")
    name = _validate_name(name)
    ds = sorted({str(d).strip() for d in (datasets or []) if str(d).strip()})
    for d in ds:
        _validate_dataset_token(d)

    def _apply(doc: dict) -> bool:
        entry = doc["clients"].get(name)
        if not isinstance(entry, dict):
            raise KeyError(name)
        entry["datasets"] = ds
        return True

    try:
        _mutate(_apply)
    except KeyError:
        raise KeyError(f"Client '{name}' does not exist in the admin registry (mint it first).")
    return {"name": name, "datasets": ds}


def _validate_dataset_token(d: str) -> None:
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", d) and d != "*":
        raise ValueError(f"invalid dataset token {d!r}")


def list_clients() -> list:
    """The overlay registry for the admin UI — key material MASKED."""
    from multimodal_rag.utils.clients_registry import CLIENTS_ENV, dataset_acls, parse_clients

    env_clients = parse_clients(os.environ.get(CLIENTS_ENV, ""))
    doc = _load()
    out = []
    for name in sorted(doc.get("clients", {}).keys()):
        entry = doc["clients"][name]
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "")
        env_name = env_clients.get(key)
        out.append(
            {
                "name": name,
                "key_masked": (key[:6] + "…" + key[-2:]) if len(key) > 10 else "…",
                "datasets": list(entry.get("datasets") or []),
                "source": "env+overlay" if env_name == name else "overlay",
                "created": entry.get("created"),
            }
        )
    # Env-only clients are listed too (visible, not editable here — the env
    # is authoritative for them).
    for key, env_name in env_clients.items():
        if env_name in doc.get("clients", {}):
            continue
        env_acls = dataset_acls().get(env_name, frozenset())
        out.append(
            {
                "name": env_name,
                "key_masked": (key[:6] + "…" + key[-2:]) if len(key) > 10 else "…",
                "datasets": sorted(env_acls),
                "source": "env",
                "created": None,
            }
        )
    return sorted(out, key=lambda c: c["name"])


def verify_client_key(name: str, key: str) -> bool:
    """Constant-time check that *key* is the stored key of overlay *name*
    (used by tests; the servers authenticate via the merged registry)."""
    entry = _load().get("clients", {}).get(name)
    if not isinstance(entry, dict):
        return False
    return hmac.compare_digest(str(entry.get("key") or ""), str(key))
