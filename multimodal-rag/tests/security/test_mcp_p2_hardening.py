"""Audit P2 hardening (mcp_server.py) — dataset-name gate, memory binding,
password-oracle.

Pinned semantics:

* ``_require_dataset_acl`` — the common entry gate of every MCP dataset tool —
  validates the dataset NAME before anything else: a name that
  ``DatasetManager._validate_name`` would reject (path separators, traversal,
  leading dot/dash) is refused with a generic ``ToolError("Invalid dataset
  name.")`` that never echoes the name.  This closes the
  ``../<other>``-reaches-``_dataset_dir`` path (``DatasetManager._dataset_dir``
  itself never validates).
* ``_resolve_memory_dataset`` — the per-identity binding (``X-Memory-Dataset``
  header → ★ binding → ``MEMORY_DATASET`` env) now OUTRANKS an explicit
  ``dataset_name`` tool argument: an argument that disagrees with a resolved
  binding is refused (a prompt-injected argument can no longer retarget a
  memory call at another dataset).  An argument that AGREES is accepted, and
  an argument with NO binding still resolves as before (the headerless
  single-user path).
* ``_check_unlocked_or_password`` — the omitted-password refusal is now the
  SAME string as the wrong-password refusal, so it no longer reveals which
  names are password protected.

Run::

    pytest tests/security/test_mcp_p2_hardening.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.mcp_server as mcp
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr

# ===========================================================================
# 1 · Dataset-name validation at the common ACL gate
# ===========================================================================


@pytest.mark.parametrize(
    "bad",
    [
        "../../datasets/<other>",  # the audit's path-traversal probe
        "../escape",
        "a/b",
        "..",
        ".",
        ".hidden",
        "-dash-first",
        "",
        "  ",
        "a\\b",
        None,
    ],
)
def test_require_dataset_acl_rejects_malformed_names(bad):
    """Generic refusal, in every identity mode — the no-identity passthrough
    (the default single-key deployment) is exactly the case the audit
    reached ``_dataset_dir`` through, so it must refuse too."""
    with pytest.raises(mcp.ToolError) as exc:
        mcp._require_dataset_acl(bad)
    assert str(exc.value) == "Invalid dataset name."
    assert "escape" not in str(exc.value)  # never echoes the submitted name


@pytest.mark.parametrize("bad", ["../escape", "a/b", "../../datasets/<other>"])
def test_require_dataset_acl_denies_before_the_acl_check(bad):
    """The name check comes FIRST: a client identity that would have been
    allowed (or denied) never reaches the ACL predicate with a malformed
    name — the message stays the generic one either way."""
    token = cr.set_current_identity(cr.Identity(kind="client", name="bob", datasets=frozenset({"*"})))
    try:
        with pytest.raises(mcp.ToolError) as exc:
            mcp._require_dataset_acl(bad)
        assert str(exc.value) == "Invalid dataset name."
    finally:
        cr.reset_current_identity(token)


def test_require_dataset_acl_accepts_valid_names_across_modes():
    """Byte-identical behaviour for well-formed names: no identity → silent;
    granted identity → silent; ungranted identity → the ACL refusal."""
    mcp._require_dataset_acl("reports-2026.v2_final")  # no identity: passthrough
    token = cr.set_current_identity(cr.Identity(kind="client", name="bob", datasets=frozenset({"notes"})))
    try:
        mcp._require_dataset_acl("notes")
        with pytest.raises(mcp.ToolError, match="not permitted"):
            mcp._require_dataset_acl("reports")
    finally:
        cr.reset_current_identity(token)


def test_require_dataset_acl_matches_dataset_manager_rules():
    """The local rule must stay a mirror of DatasetManager._validate_name —
    drive both over a sample matrix and require the same accept/reject."""
    from multimodal_rag.dataset_manager import DatasetManager

    for name in ["ok", "Ok.dot-dash_1", "9lives", ".hidden", "-lead", "a/b", "..", "a b", ""]:
        try:
            DatasetManager._validate_name(name)
            dm_ok = True
        except ValueError:
            dm_ok = False
        try:
            mcp._require_dataset_acl(name)
            mcp_ok = True
        except mcp.ToolError:
            mcp_ok = False
        assert dm_ok == mcp_ok, name


def test_search_dataset_rejects_traversal_before_touching_the_store(monkeypatch):
    """The audit's exact probe: ``search_dataset(dataset_name="../../datasets/
    <other>")`` used to reach ``_dataset_dir`` before any existence check.  The
    name gate now refuses it before the store is addressed at all.

    (``get_manager()`` itself is a name-agnostic cached-singleton accessor and
    may still be called first by the tool body; nothing that takes the dataset
    name may be reached.)"""
    import asyncio

    class _NeverDM:
        def __getattr__(self, item):  # pragma: no cover - only fires on a regression
            raise AssertionError(f"DatasetManager.{item} must not be reached for a malformed name")

    monkeypatch.setattr(mcp, "get_manager", lambda: _NeverDM())
    with pytest.raises(mcp.ToolError) as exc:
        asyncio.run(mcp.search_dataset(dataset_name="../../datasets/<other>", query="x"))
    assert str(exc.value) == "Invalid dataset name."


# ===========================================================================
# 2 · Memory-dataset binding is authoritative over the tool argument
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_memory_env(tmp_path, monkeypatch):
    """No ambient memory configuration: DATA_PATH isolated under tmp, no
    MEMORY_DATASET / header, store OFF unless a test turns it on.

    (Also pins the access-store directory at tmp so nothing in this file can
    touch a real deployment's /data — the store path is DATA_PATH-derived.)"""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    monkeypatch.delenv("MEMORY_DATASET", raising=False)
    monkeypatch.delenv("RAG_MEMORY_DATASET", raising=False)
    monkeypatch.delenv("RAG_MEMORY_DEFAULT", raising=False)
    monkeypatch.delenv("RAG_MEMORY_PASSWORD", raising=False)
    monkeypatch.delenv(acc.STORE_ENV, raising=False)
    acc._mtime_cache.clear()
    ds_token = mcp._memory_dataset_ctx.set(None)
    pw_token = mcp._memory_password_ctx.set(None)
    identity_token = cr.set_current_identity(None)
    yield
    cr.reset_current_identity(identity_token)
    mcp._memory_password_ctx.reset(pw_token)
    mcp._memory_dataset_ctx.reset(ds_token)
    acc._mtime_cache.clear()


def test_header_binding_wins_and_mismatched_argument_is_refused(monkeypatch):
    monkeypatch.setenv("MEMORY_DATASET", "env-mem")
    token = mcp._memory_dataset_ctx.set("bound-mem")
    try:
        # The header binding outranks the env fallback AND the argument.
        assert mcp._resolve_memory_dataset(None) == "bound-mem"
        assert mcp._resolve_memory_dataset("bound-mem") == "bound-mem"  # agreeing arg: allowed
        with pytest.raises(mcp.ToolError, match="does not match your memory binding"):
            mcp._resolve_memory_dataset("other-mem")
    finally:
        mcp._memory_dataset_ctx.reset(token)
    assert mcp._resolve_memory_dataset(None) == "env-mem"  # env source intact


def test_star_binding_wins_and_mismatched_argument_is_refused(monkeypatch):
    """The ★ binding (D16 access store) — the injection case the audit
    describes: an identity bound to X may not be retargeted at Y."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = cr.Identity(kind="client", name="alice", datasets=frozenset({"X"}))
    token = cr.set_current_identity(ident)
    try:
        acc.select_dataset(ident, "X")
        acc.set_memory_dataset(ident, "X")
        assert mcp._resolve_memory_dataset(None) == "X"
        assert mcp._resolve_memory_dataset("X") == "X"  # argument == binding: allowed
        with pytest.raises(mcp.ToolError, match="does not match your memory binding"):
            mcp._resolve_memory_dataset("Y")
    finally:
        cr.reset_current_identity(token)

    # No binding (★ cleared) → the explicit argument works as before.
    token = cr.set_current_identity(ident)
    try:
        acc.set_memory_dataset(ident, None)
        assert mcp._resolve_memory_dataset("Y") == "Y"
        assert mcp._resolve_memory_dataset("X") == "X"  # the selection is not a binding
    finally:
        cr.reset_current_identity(token)


def test_store_bound_identity_without_env_still_refuses_mismatch(monkeypatch):
    """Store ON, no MEMORY_DATASET env, ★ bound: the binding resolves and the
    mismatched argument is refused (the ordering is binding-first even when
    the only binding source is the store)."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ident = cr.Identity(kind="client", name="alice", datasets=frozenset({"X"}))
    token = cr.set_current_identity(ident)
    try:
        acc.select_dataset(ident, "X")
        acc.set_memory_dataset(ident, "X")
        with pytest.raises(mcp.ToolError, match="does not match your memory binding"):
            mcp._resolve_memory_dataset("Y")
    finally:
        cr.reset_current_identity(token)


def test_no_binding_explicit_argument_is_todays_behaviour():
    """No header, no ★, no env — the headerless single-user path the audit
    says must keep working."""
    assert mcp._resolve_memory_dataset("solo-mem") == "solo-mem"


def test_no_binding_no_argument_still_refuses_with_guidance():
    with pytest.raises(mcp.ToolError, match="No memory dataset specified"):
        mcp._resolve_memory_dataset(None)


def test_memory_tool_refuses_a_retargeted_argument(monkeypatch):
    """End-to-end through a memory tool: ``add_memory(dataset_name=…)`` on a
    connection bound to another dataset refuses BEFORE any store access."""
    token = mcp._memory_dataset_ctx.set("bound-mem")
    try:
        with pytest.raises(mcp.ToolError, match="does not match your memory binding"):
            import asyncio

            asyncio.run(mcp.add_memory(text="hello", dataset_name="other-mem"))
    finally:
        mcp._memory_dataset_ctx.reset(token)


# ===========================================================================
# 3 · The omitted-password refusal is the wrong-password refusal
# ===========================================================================


class _StubDM:
    """Minimal DatasetManager surface for ``_check_unlocked_or_password``
    (plus ``get_dataset`` so it can stand in for a tool body)."""

    def __init__(self, protected: bool):
        self.protected = protected

    def get_dataset(self, name: str, sync_count: bool = True) -> dict:
        return {"name": name, "document_count": 0, "has_password": self.protected}

    def has_password(self, name: str) -> bool:
        return self.protected

    def verify_password(self, name: str, password: str) -> bool:
        return password == "right"


def test_omitted_and_wrong_password_errors_are_identical(monkeypatch):
    monkeypatch.setenv(acc.STORE_ENV, "1")  # a saved password is a real unlock source
    tracked = cr.Identity(kind="client", name="carol", datasets=frozenset({"secret"}))
    token = cr.set_current_identity(tracked)
    dm = _StubDM(protected=True)
    try:
        with pytest.raises(mcp.ToolError) as wrong:
            mcp._check_unlocked_or_password(dm, "secret", "wrong")
        with pytest.raises(mcp.ToolError) as omitted:
            mcp._check_unlocked_or_password(dm, "secret", None)
        assert str(wrong.value) == str(omitted.value)
        assert str(omitted.value) == "Incorrect password for dataset 'secret'."
        # Correct password / unprotected dataset behave as before.
        assert mcp._check_unlocked_or_password(dm, "secret", "right") == "right"
        assert mcp._check_unlocked_or_password(_StubDM(protected=False), "open", None) is None
    finally:
        mcp._unlocked.clear()
        cr.reset_current_identity(token)


def test_password_oracle_closed_at_the_tool_surface(monkeypatch):
    """The same equality through a real tool body, where a caller actually
    sees it: two calls that differ only in password-correctness must return
    the identical ToolError message (no 'this dataset is protected' tell)."""
    import asyncio

    monkeypatch.setenv(acc.STORE_ENV, "1")
    token = cr.set_current_identity(
        cr.Identity(kind="client", name="carol", datasets=frozenset({"secret"}))
    )
    monkeypatch.setattr(mcp, "get_manager", lambda: _StubDM(protected=True))
    try:
        with pytest.raises(mcp.ToolError) as wrong:
            asyncio.run(mcp.get_dataset_info(dataset_name="secret", password="wrong"))
        with pytest.raises(mcp.ToolError) as omitted:
            asyncio.run(mcp.get_dataset_info(dataset_name="secret"))
        assert str(wrong.value) == str(omitted.value)
        assert "protected" not in str(omitted.value)
    finally:
        mcp._unlocked.clear()
        cr.reset_current_identity(token)


def test_unprotected_omission_is_not_mistaken_for_a_bad_password(monkeypatch):
    """The shared string must not turn the old 'no password set' passthrough
    into a refusal (the message says a password is wrong; an unprotected
    dataset still passes through)."""
    dm = _StubDM(protected=False)
    assert mcp._check_unlocked_or_password(dm, "public-ds", None) is None
