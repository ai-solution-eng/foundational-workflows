"""Per-identity dataset selection + memory binding (fleet decision D16, OPT-IN).

The self-service counterpart to the operator-side D15 registry.  A
registry-key user (from ``RAG_API_KEY_CLIENTS``) selects datasets on the
``/access`` page — public datasets freely, password-protected ones only with
the correct password as proof — and the selection persists per identity so
BOTH surfaces (MCP tools and the REST API) honor it.  Effective access is
``operator ACL ∪ self-selections`` (the ACL is a guaranteed FLOOR; selections
can only widen, never narrow, and can never reach the admin surface).

Storage: one JSON file per identity under ``{DATA_PATH}/access/<identity>.json``
(the shared RWX PVC — the same store the upload history and dataset metadata
live on), written atomically (temp + ``os.replace``) under a cross-process
fcntl lock, read with an mtime cache (per-process) so hot paths don't re-stat
NFS on every request.  Passwords are stored AS THE CALLER PROVIDED THEM — the
store's trust posture is identical to the unlock caches' (plaintext passwords
at rest in an operator-controlled store); it is NOT a new class of secret.

Configuration:

* ``RAG_ACCESS_STORE`` — set to ``"1"``/``"true"``/``"yes"`` to enable
  (default OFF; when OFF the module is inert and every predicate reports
  "nothing extra").  Server-side personalization without touching D15's
  default posture.
* ``RAG_ACCESS_DENY_SELECT`` — comma-separated dataset names that can never
  be self-selected (operator floor/ceiling protection, e.g. ``hr-data``).
  Re-read per call (rotation without restart).
* ``RAG_MEMORY_DEFAULT`` — deployment-wide fallback memory dataset (used when
  the caller has no memory binding; admin/single-user convenience).

Effective access (``dataset_allowed``):
  admins / no-registry / store-off  → operator behaviour, byte-identical
  registry client                   → operator ACL ∪ selections (denylist
                                       still enforced at selection time;
                                       ``*`` in the operator ACL still wins)

Memory resolution (``memory_dataset_for``):
  explicit arg (caller) → request header (client) → the caller's SAVED
  binding (page star) → ``RAG_MEMORY_DEFAULT`` → ``MEMORY_DATASET`` env.
  The saved binding is consulted only when the bound dataset is actually
  accessible to the identity (fail-soft to the next source).

This module is RAG-local, stdlib-only, dependency-free; it never logs key or
password material.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Reuse the dataset_manager's cross-process lock helper (fcntl on the shared
# PVC) — imported lazily to keep this module import-light and to avoid a
# cycle (dataset_manager does not import this module).
_lock_mod = None


def _cross_process_lock(lock_path: Path):
    global _lock_mod
    if _lock_mod is None:
        from multimodal_rag.dataset_manager import _cross_process_lock as fn

        _lock_mod = fn
    return _lock_mod(lock_path)


STORE_ENV = "RAG_ACCESS_STORE"
DENY_ENV = "RAG_ACCESS_DENY_SELECT"
MEMORY_DEFAULT_ENV = "RAG_MEMORY_DEFAULT"

# Reserved directory names under DATA_PATH — a dataset directory can never
# start with "." so an identity file can never collide with a dataset dir;
# the store dir is a sibling of datasets/, not inside it.
_STORE_DIRNAME = "access"
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# mtime-based read cache (per process): path → (mtime_ns, size, parsed dict)
_mtime_cache: dict[str, tuple[int, int, dict]] = {}
_mtime_lock = threading.Lock()
# Per-path threading lock so concurrent in-process writers serialize BEFORE
# they reach fcntl (flock is per-fd: two threads in one process would both
# "hold" it and race the read-modify-write).
_writer_locks: dict[str, threading.Lock] = {}
_writer_locks_guard = threading.Lock()


def store_enabled() -> bool:
    """True when RAG_ACCESS_STORE opts the deployment in.  Read per call."""
    return os.environ.get(STORE_ENV, "").strip().lower() in ("1", "true", "yes")


def _store_dir() -> Path:
    return Path(os.environ.get("DATA_PATH", "/data")) / _STORE_DIRNAME


def _identity_filename(identity: str) -> str:
    """Sanitized per-identity file name.

    The caller id is ``key:<name>`` (clients_registry.client_id) with *name*
    constrained by the registry parser to a printable no-``:``/``;`` form;
    we additionally whitelist a strict charset so a hostile operator config
    can never escape the store directory (defense in depth — the name comes
    from the operator's own env, but the store treats it as data).
    """
    name = identity.split(":", 1)[1] if identity.startswith("key:") else identity
    if not _IDENTITY_RE.match(name):
        raise ValueError(f"identity {identity!r} does not match the access-store name rules")
    return f"{name}.json"


def _path_for(identity: str) -> Path:
    return _store_dir() / _identity_filename(identity)


def _denylist() -> frozenset:
    raw = os.environ.get(DENY_ENV, "")
    return frozenset(d.strip() for d in raw.split(",") if d.strip())


def _load(identity: str) -> dict:
    """Load one identity's store file (mtime-cached; missing → empty doc)."""
    path = _path_for(identity)
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
    doc.setdefault("datasets", {})
    doc.setdefault("memory_dataset", "")
    if not isinstance(doc.get("datasets"), dict):
        doc["datasets"] = {}
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


def _mutate(identity: str, fn: Callable[[dict], bool]) -> dict:
    """Read-modify-write one identity's store under the cross-process lock.

    *fn* receives the loaded doc (freshly read under the lock — the mtime
    cache is bypassed for the read to avoid a stale-merge) and returns True
    when it changed anything; nothing is written when it returns False.
    """
    path = _path_for(identity)
    key = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(key), _cross_process_lock(path.with_suffix(".lock")):
        # Read the FILE (not the cache) for a true current state.
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            doc = {}
        except (json.JSONDecodeError, OSError):
            doc = {}
        if not isinstance(doc, dict):
            doc = {}
        doc.setdefault("datasets", {})
        doc.setdefault("memory_dataset", "")
        if not isinstance(doc.get("datasets"), dict):
            doc["datasets"] = {}
        if not fn(doc):
            return doc
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass  # e.g. some network FS refuse chmod — file still works
        os.replace(tmp, path)
        # Refresh the mtime cache from what we just wrote.
        try:
            st = path.stat()
            with _mtime_lock:
                _mtime_cache[key] = (st.st_mtime_ns, st.st_size, doc)
        except OSError:
            pass
        return doc


# ---------------------------------------------------------------------------
# Selections (the D16 core)
# ---------------------------------------------------------------------------


def selections_for(identity: Any | None) -> frozenset:
    """The identity's self-selected dataset names (empty for admins/None/off)."""
    if identity is None or not store_enabled():
        return frozenset()
    if getattr(identity, "is_admin", False):
        return frozenset()
    return frozenset(str(k) for k in _load(identity.client_id).get("datasets", {}))


def selection_entry(identity: Any | None, dataset_name: str) -> dict | None:
    """The identity's stored entry for *dataset_name*, or None."""
    if identity is None or not store_enabled() or getattr(identity, "is_admin", False):
        return None
    entry = _load(identity.client_id).get("datasets", {}).get(dataset_name)
    return dict(entry) if isinstance(entry, dict) else None


def selection_password(identity: Any | None, dataset_name: str) -> str | None:
    """The password saved with the identity's selection, or None.

    Returned ONLY for the matching identity — the caller resolved the
    identity from the presented key already; this is not an oracle (an
    unknown dataset name returns None like anything else).
    """
    entry = selection_entry(identity, dataset_name)
    if entry is None:
        return None
    pw = entry.get("password")
    return str(pw) if pw else None


def dataset_allowed(identity: Any | None, dataset_name: str) -> bool:
    """D16-effective access: operator ACL ∪ self-selections.

    Semantics preserved exactly when the store is off / identity is None or
    admin (byte-identical default).  For a registry client: the operator ACL
    is a FLOOR — a selection widens access (it was made with proof), the
    denylist can never be selected in the first place.
    """
    from multimodal_rag.utils.clients_registry import dataset_allowed as operator_allows

    if operator_allows(identity, dataset_name):
        return True
    return dataset_name in selections_for(identity)


def filter_dataset_names(identity: Any | None, names) -> tuple[list, int]:
    """D16-effective ``(visible, hidden_count)`` over dataset *names*."""
    visible = [n for n in names if dataset_allowed(identity, str(n))]
    return visible, len(names) - len(visible)


def select_dataset(identity: Any | None, dataset_name: str, password: str | None = None) -> dict:
    """Self-select *dataset_name* for *identity* (D16).

    Rules:
      * admin identity or store off → refused (nothing to select INTO; admins
        already have everything and the store is inert).
      * denylisted dataset → refused regardless of proof.
      * already accessible via the operator ACL → accepted WITHOUT saving the
        password (the ACL is the stronger grant; keep the store minimal).
      * protected dataset → requires the CORRECT password (verified by the
        caller's DatasetManager and passed in pre-verified); saved with the
        selection.
      * public dataset → accepted, no password saved.

    Returns the stored entry (``{"dataset", "password", "selected_at",
    "source"}``); raises :class:`SelectionDenied` with a caller-safe message
    when refused.
    """
    if identity is None or getattr(identity, "is_admin", False):
        raise SelectionDenied("Dataset selection is available for per-user API keys.")
    if not store_enabled():
        raise SelectionDenied("Dataset selection is not enabled on this deployment (RAG_ACCESS_STORE).")
    from multimodal_rag.utils.clients_registry import dataset_allowed as operator_allows

    dataset_name = str(dataset_name)
    if dataset_name in _denylist():
        raise SelectionDenied(f"Dataset '{dataset_name}' cannot be self-selected on this deployment.")
    already = operator_allows(identity, dataset_name)
    selected_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    stored_source = {"v": "self"}

    def _apply(doc: dict) -> bool:
        entry = {"dataset": dataset_name, "password": password or "", "selected_at": selected_at}
        if already:
            if password:
                # ACL-granted BUT the caller proved the password anyway —
                # save it as a password-only sidecar entry (source:
                # "acl+pw").  This is the memory-binding case: the ★ binding
                # resolves the dataset via the ACL, and the memory tools
                # resolve the PASSWORD from this entry (without it an
                # ACL-granted protected dataset would still demand a
                # password on every memory call).  A later deselect removes
                # the sidecar; the ACL grant itself is untouched.
                entry["source"] = "acl+pw"
                stored_source["v"] = "acl+pw"
                doc["datasets"][dataset_name] = entry
                return True
            # No proof offered: keep the store minimal (the ACL already
            # grants; shadowing it with an empty entry would break nothing
            # but carries no information).
            stored_source["v"] = "acl"
            return False
        entry["source"] = "self"
        doc["datasets"][dataset_name] = entry
        return True

    _mutate(identity.client_id, _apply)
    return {
        "dataset": dataset_name,
        "password": password or "",
        "selected_at": selected_at,
        "source": stored_source["v"],
    }


def deselect_dataset(identity: Any | None, dataset_name: str) -> bool:
    """Remove the identity's self-selection (and its saved password).

    Returns True when a stored selection was removed.  An operator-ACL grant
    is NOT touched (deselecting what the operator granted is meaningless —
    access comes from the ACL itself).
    """
    if identity is None or getattr(identity, "is_admin", False) or not store_enabled():
        return False

    removed = {"v": False}

    def _apply(doc: dict) -> bool:
        if dataset_name in doc["datasets"]:
            del doc["datasets"][dataset_name]
            removed["v"] = True
            return True
        return False

    _mutate(identity.client_id, _apply)
    return removed["v"]


def memory_dataset_for(identity: Any | None, fallback: str | None = None) -> str | None:
    """The identity's saved memory dataset (the page's ★), when accessible.

    Fail-soft: a binding to a dataset the identity can no longer reach (ACL
    shrunk, selection removed, dataset deleted) is ignored — the caller
    falls through to the deployment default.  ``fallback`` is the caller's
    own next source (e.g. ``MEMORY_DATASET`` env); when the store is off the
    function is a passthrough to *fallback*.
    """
    if identity is None or not store_enabled() or getattr(identity, "is_admin", False):
        return fallback
    bound = str(_load(identity.client_id).get("memory_dataset") or "")
    if not bound:
        return fallback
    if not dataset_allowed(identity, bound):
        return fallback
    return bound


def set_memory_dataset(identity: Any | None, dataset_name: str | None) -> str | None:
    """Star/unstar the identity's memory dataset (the page's ★, server-side).

    Setting a name requires it to be ACCESSIBLE (operator ACL or selection);
    clearing (None) always works.  Returns the effective binding afterwards.
    """
    if identity is None or getattr(identity, "is_admin", False):
        raise SelectionDenied("Memory binding is available for per-user API keys.")
    if not store_enabled():
        raise SelectionDenied("Memory binding is not enabled on this deployment (RAG_ACCESS_STORE).")
    name = str(dataset_name).strip() if dataset_name else ""
    if name and not dataset_allowed(identity, name):
        raise SelectionDenied(
            f"Dataset '{name}' is not accessible to this key — select it first."
        )
    doc = _mutate(identity.client_id, lambda d: _set_memory(d, name))
    return str(doc.get("memory_dataset") or "") or None


def _set_memory(doc: dict, name: str) -> bool:
    if doc.get("memory_dataset") == name:
        return False
    doc["memory_dataset"] = name
    return True


def stats(identity: Any | None) -> dict:
    """A small summary for the page/status (never includes passwords)."""
    if identity is None or not store_enabled():
        return {"enabled": store_enabled(), "selections": 0, "memory_dataset": None}
    doc = _load(identity.client_id)
    sels = doc.get("datasets", {})
    mem = str(doc.get("memory_dataset") or "")
    return {
        "enabled": True,
        "selections": len(sels),
        "memory_dataset": mem or None,
        "with_password": sum(1 for e in sels.values() if isinstance(e, dict) and e.get("password")),
    }


class SelectionDenied(PermissionError):
    """The caller may not select / bind (admin, store off, or denylist)."""


# ---------------------------------------------------------------------------
# Verification helper (shared by both servers)
# ---------------------------------------------------------------------------


def verify_and_select(
    identity: Any | None,
    dataset_name: str,
    password: str | None,
    has_password: Callable[[str], bool],
    verify_password: Callable[[str, str], bool],
) -> dict:
    """Verify the proof, then select.  Shared by the REST endpoint and (in
    tool form) the MCP ``select_dataset`` tool.

    Order matters: the denylist/admin/store checks run BEFORE any password
    work, a public dataset needs no password, and a WRONG password on a
    protected dataset raises ``ValueError`` (the caller maps it to 403 with
    its own throttle accounting — the caller owns its throttle buckets).
    """
    # Existence + protection status are the caller's DatasetManager calls;
    # has_password/verify_password are passed in so this stays manager-free.
    if has_password(dataset_name):
        if not password:
            raise ValueError(f"Dataset '{dataset_name}' is password protected — provide the password to select it.")
        if not verify_password(dataset_name, password):
            raise ValueError("Incorrect password")
        entry = select_dataset(identity, dataset_name, password=password)
    else:
        entry = select_dataset(identity, dataset_name)
    return entry
