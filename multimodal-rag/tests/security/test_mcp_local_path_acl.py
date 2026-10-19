"""P1-2 (audit 2026-10-16): MEDIA_ALLOW_PATH_PREFIXES is a PATH allowlist,
not an ACL.

The defect: ``describe_media`` / ``transcribe_audio`` /
``_mcp_ingest_local_path`` / ``dataset_replace_document`` read local files
that merely had to sit under the media-path prefixes — which span EVERY
tenant's ``datasets/`` tree.  Reproduced: an ACL-denied tenant's file bytes
reached the VLM, and a cross-tenant copy into the caller's own dataset
succeeded.

Pinned here (negative + positive):

  * a local path inside another dataset's directory is refused for an
    identity that does not reach that dataset (D15/D16), for both the media
    tools and the ingest tools;
  * the SAME path is accepted for the dataset's owner / an ACL-granted key;
  * paths outside any dataset directory (staging) and the identity-less
    single-user deployment are unchanged — the prefix policy alone.

Run::

    pytest tests/security/test_mcp_local_path_acl.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.clients_registry as cr
import multimodal_rag.utils.media_paths as mp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in (cr.CLIENTS_ENV, cr.ACLS_ENV, acc.STORE_ENV, "MEDIA_ALLOW_PATH_PREFIXES", "MEMORY_DATASET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    acc._mtime_cache.clear()
    yield
    acc._mtime_cache.clear()


@pytest.fixture()
def mcp_env(monkeypatch, tmp_path):
    """The MCP module with the media-path policy pointed at tmp_path.

    The prefix list is read from the environment per call by the tools, so
    setting DATA_PATH + MEDIA_ALLOW_PATH_PREFIXES is enough to make the
    fixture's dataset tree the allowed world.
    """
    import multimodal_rag.mcp_server as mcp

    root = tmp_path / "datasets"
    (root / "mine" / "files").mkdir(parents=True)
    (root / "theirs" / "files").mkdir(parents=True)
    (root / "mine" / "files" / "ok.jpg").write_bytes(b"\xff\xd8\xffmine")
    (root / "theirs" / "files" / "secret.jpg").write_bytes(b"\xff\xd8\xffsecret")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "handoff.jpg").write_bytes(b"\xff\xd8\xffstaged")
    # media_paths resolves the prefix list ONCE at import (module constants),
    # so the suite patches them directly — the sibling suites' pattern.
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_PATH_PREFIXES", (str(root), str(staging)))
    monkeypatch.setattr(mp, "_MEDIA_ALLOW_ANY", False)

    # A stub manager so the tool bodies reach the ACL step offline: the
    # refusal under test happens BEFORE any model work (that is the point),
    # and no real DatasetManager can be built without an embedder endpoint.
    class _StubDM:
        vlm = None

        class embedder:  # attribute-style stub
            mm_processor_kwargs = {}  # noqa: RUF012 — attribute-style stub

        def get_dataset(self, name, sync_count=True):
            if name not in ("mine", "theirs"):
                raise FileNotFoundError(name)
            return {"name": name}

        def has_password(self, name):
            return False

    monkeypatch.setattr(mcp, "get_manager", lambda: _StubDM())
    return mcp, root, staging


def _acl_helper(mcp):
    """The nested ``_require_local_path_acl`` the tool bodies close over.

    It is defined inside the server factory, so the test reaches it the same
    way the tools do: through the registered tool call for the end-to-end
    assertions below, and through the module-level helper factory for the
    unit assertions.  ``_dataset_for_local_path`` is the pure half — it is
    exposed on the module for exactly this kind of pinning.
    """
    return mcp._dataset_for_local_path


def _call(mcp, tool, args):
    import asyncio

    async def _run():
        try:
            result = await mcp.mcp.call_tool(tool, args)
            if getattr(result, "isError", False):
                return ("error", str(result.content[0].text))
            return ("ok", str(result.content[0].text))
        except mcp.ToolError as exc:
            return ("error", str(exc))

    return asyncio.run(_run())


# ===========================================================================
# 1 · the pure reverse index
# ===========================================================================


def test_dataset_for_local_path_resolves_and_picks_the_longest_root(mcp_env):
    mcp, root, staging = mcp_env
    find = _acl_helper(mcp)
    assert find(str((root / "theirs" / "files" / "secret.jpg").resolve())) == "theirs"
    assert find(str((root / "mine" / "files").resolve())) == "mine"
    # outside every dataset root → None (staging is governed by the prefix
    # policy alone).
    assert find(str((staging / "handoff.jpg").resolve())) is None
    assert find("/etc/passwd") is None
    # a sibling with a shared prefix must NOT match ("mine" vs "mine2").
    (root / "mine2" / "files").mkdir(parents=True)
    assert find(str((root / "mine2" / "files").resolve())) == "mine2"


# ===========================================================================
# 2 · the tool surface
# ===========================================================================


def test_media_tool_refuses_another_datasets_file(mcp_env, monkeypatch):
    """THE P1-2 regression: an ACL-denied key must not hand another tenant's
    file to the VLM, even though the path passes the prefix allowlist."""
    mcp, root, _staging = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:mine;bob:theirs")
    secret = str((root / "theirs" / "files" / "secret.jpg").resolve())
    token = cr.set_current_identity(cr.resolve_presented(["alice-key"]))
    try:
        status, text = _call(mcp, "describe_media", {"media_url": f"file://{secret}"})
        assert status == "error", text
        assert "not permitted for this API key" in text
        # The VLM must never have been reached (the refusal is pre-read).
        assert "VLM description failed" not in text
        # …and alice's OWN dataset path passes the ACL step (it then fails
        # later for an unrelated reason: no VLM configured on the stub —
        # which is exactly the proof that the ACL no longer refused it).
        own = str((root / "mine" / "files" / "ok.jpg").resolve())
        status, text = _call(mcp, "describe_media", {"media_url": f"file://{own}"})
        assert "not permitted for this API key" not in text
    finally:
        cr.reset_current_identity(token)


def test_ingest_tool_refuses_another_datasets_file(mcp_env, monkeypatch):
    """The ingest twin: a cross-tenant copy must not succeed."""
    mcp, root, _staging = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key;bob:bob-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:mine;bob:theirs")
    secret = str((root / "theirs" / "files" / "secret.jpg").resolve())
    token = cr.set_current_identity(cr.resolve_presented(["alice-key"]))
    try:
        status, text = _call(mcp, "dataset_add_documents", {"dataset_name": "mine", "paths": [secret]})
        assert status == "error", text
        assert "not permitted for this API key" in text
    finally:
        cr.reset_current_identity(token)


def test_staging_paths_and_identityless_callers_unchanged(mcp_env, monkeypatch):
    """The preservation half: staging paths (outside every dataset root) and
    a deployment with no registry identity behave exactly as before — the
    prefix policy alone decides."""
    mcp, _root, staging = mcp_env
    staged = str((staging / "handoff.jpg").resolve())
    # identity-less (D15 inactive): the legacy default.
    _status, text = _call(mcp, "describe_media", {"media_url": f"file://{staged}"})
    assert "not permitted for this API key" not in text
    # and with an identity bound: staging is still not dataset-governed.
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:mine")
    token = cr.set_current_identity(cr.resolve_presented(["alice-key"]))
    try:
        _status, text = _call(mcp, "describe_media", {"media_url": f"file://{staged}"})
        assert "not permitted for this API key" not in text
    finally:
        cr.reset_current_identity(token)


def test_prefix_policy_still_applies_first(mcp_env, monkeypatch):
    """The ACL step is ADDITIVE — a path outside the prefixes is still
    refused by the prefix policy, with its own message."""
    mcp, _root, _staging = mcp_env
    monkeypatch.setenv(cr.CLIENTS_ENV, "alice:alice-key")
    monkeypatch.setenv(cr.ACLS_ENV, "alice:mine")
    token = cr.set_current_identity(cr.resolve_presented(["alice-key"]))
    try:
        status, text = _call(mcp, "describe_media", {"media_url": "file:///etc/passwd"})
        assert status == "error"
        assert "MEDIA_ALLOW_PATH_PREFIXES" in text
    finally:
        cr.reset_current_identity(token)


def test_dataset_dir_roots_ignores_hidden_entries(mcp_env):
    """A dotted sibling under datasets/ is not a dataset (no ACL to apply)."""
    mcp, root, _staging = mcp_env
    hidden = root / ".hidden" / "files"
    hidden.mkdir(parents=True)
    (hidden / "x.jpg").write_bytes(b"\xff\xd8\xff")
    assert _acl_helper(mcp)(str((hidden / "x.jpg").resolve())) is None
