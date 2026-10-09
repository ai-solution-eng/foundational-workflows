"""Fleet decision D26 (read-only dataset grants) — grant-mode enforcement matrix.

Pinned semantics (the implementation in the working tree is the source of
truth — src/multimodal_rag/utils/clients_registry.py, utils/admin_registry.py,
utils/access_store.py, and the two server gates):

* ``RAG_DATASET_ACLS`` dataset tokens carry an optional ``:ro``/``:rw`` mode
  suffix (bare tokens stay rw — byte-compatible with D15); malformed entries
  raise ``ClientConfigError`` (fail loud).
* ``Identity.grants`` carries ``{dataset: mode}``; ``None`` (the legacy
  frozenset form, every pre-D26 construction site) resolves all-rw; the MOST
  SPECIFIC grant beats the ``*`` wildcard in either direction; admins and
  identity-less callers have no mode semantics at all (write allowed).
* env ∪ overlay merge keeps the STRONGEST restriction across sources
  (env rw ∪ overlay ro ⇒ ro; env ro ∪ overlay rw ⇒ ro).
* The overlay file stores plain strings (rw) or ``{"name","mode"}`` objects;
  unknown modes fail closed (the grant is dropped); a non-list ``datasets``
  yields no grants.
* ``_normalize_grants`` is the REST/PATCH grant writer: ``"name:mode"``
  shorthand normalizes to the object form, duplicates last-wins, invalid
  modes raise ``ValueError``; a bare ``*`` keeps the legacy string form.
* access_store: a 🌐 public dataset is included for every authenticated
  identity (global-by-default) at READ-ONLY mode unless an explicit grant
  covers it; exclusions hide even globals; a self-selection never upgrades a
  global to rw; the memory ★ binding refuses a read-only dataset.
* Both servers gate writes through the same predicate: the MCP
  ``_require_dataset_write`` (ToolError, "read-only") and the REST middleware
  deny-checker ``_rag_acl_path_denial`` (the ro denial string; the D16
  /select and D23 /public POST exemptions ahead of the gate; GETs never
  gated).

Run::

    pytest tests/security/test_grant_modes.py -q
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import multimodal_rag.api_server as api
import multimodal_rag.mcp_server as mcp
import multimodal_rag.utils.access_store as acc
import multimodal_rag.utils.admin_registry as ar
import multimodal_rag.utils.clients_registry as cr

# ===========================================================================
# Fixtures / helpers
# ===========================================================================


@pytest.fixture(autouse=True)
def _fresh_env(tmp_path, monkeypatch):
    """Fresh DATA_PATH per test + a quiet registry/store environment.

    Mirrors the sibling security suites: the registry, ACLs, keys, store and
    overlay knobs are all OFF unless a test turns one on, and the module
    caches (public-meta, created-by, both mtime stores) are cleared so a
    stamped meta.json is always read fresh.
    """
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    for name in (
        cr.CLIENTS_ENV,
        cr.ACLS_ENV,
        "RAG_API_KEY",
        "MCP_API_KEYS",
        "RAG_API_KEYS",
        "MEMORY_DATASET",
        "RAG_TRUST_PROXY_IDENTITY",
        acc.STORE_ENV,
        acc.DENY_ENV,
        acc.MEMORY_DEFAULT_ENV,
        ar.ADMIN_FILE_ENABLED_ENV,
        ar.SELF_MINT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()
    acc._mtime_cache.clear()
    ar._mtime_cache.clear()
    yield
    cr._PUBLIC_META_CACHE.clear()
    cr._CREATED_BY_CACHE.clear()
    acc._mtime_cache.clear()
    ar._mtime_cache.clear()


def _stamp_public(name):
    """Stamp a dataset GLOBAL (the D26 public flag) via its meta.json — the
    same path ``clients_registry.is_public_dataset`` reads."""
    d = Path(os.environ["DATA_PATH"]) / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"public": True}), encoding="utf-8")


def _ro(name="reports"):
    return cr.Identity(kind="client", name="ro-" + name, datasets=frozenset({name}), grants={name: cr.GRANT_RO})


def _rw(name="reports"):
    return cr.Identity(kind="client", name="rw-" + name, datasets=frozenset({name}), grants={name: cr.GRANT_RW})


# ===========================================================================
# 1 · parse_acls: the D26 dataset-token forms
# ===========================================================================


def test_parse_acls_bare_list_is_all_rw():
    """``a:ds1,ds2`` — the historical form — keeps its exact meaning: every
    bare token grants rw."""
    assert cr.parse_acls("a:ds1,ds2") == {"a": {"ds1": cr.GRANT_RW, "ds2": cr.GRANT_RW}}


def test_parse_acls_mixed_modes():
    """``a:ds1:ro,ds2:rw,ds3`` — per-token modes mix freely with bare
    tokens (a bare token is rw)."""
    assert cr.parse_acls("a:ds1:ro,ds2:rw,ds3") == {"a": {"ds1": cr.GRANT_RO, "ds2": cr.GRANT_RW, "ds3": cr.GRANT_RW}}


def test_parse_acls_star_modes():
    """``*:ro`` → the wildcard carries the ro mode; ``*:rw`` is the explicit
    rw spelling of the historical ``*``."""
    assert cr.parse_acls("a:*:ro") == {"a": {"*": cr.GRANT_RO}}
    assert cr.parse_acls("a:*:rw") == {"a": {"*": cr.GRANT_RW}}
    assert cr.parse_acls("a:*") == {"a": {"*": cr.GRANT_RW}}


def test_parse_acls_empty_list_entry_grants_nothing():
    """``a:`` — an entry with an empty dataset list grants nothing (the D15
    fail-closed rule, mode-aware but unchanged)."""
    assert cr.parse_acls("a:") == {"a": {}}
    assert cr.parse_acls("") == {}


# ===========================================================================
# 2 · parse_acls: fail loud on malformed entries
# ===========================================================================


def test_parse_acls_unknown_mode_fails_loud():
    """An unknown mode word is a typo'd registry — ClientConfigError, never a
    silent degrade to 'no ACLs enforced'."""
    with pytest.raises(cr.ClientConfigError, match="unknown mode"):
        cr.parse_acls("a:ds1:admin")


def test_parse_acls_empty_token_fails_loud():
    """An empty dataset token (``a:ds1,:ro``) is malformed — fail loud."""
    with pytest.raises(cr.ClientConfigError, match="empty dataset token"):
        cr.parse_acls("a:ds1,:ro")


# ===========================================================================
# 3 · Identity.grants: defaults, specificity, missing entries
# ===========================================================================


def test_identity_without_grants_resolves_all_rw():
    """The legacy frozenset form (``grants`` omitted = None) has NO mode
    semantics: grant_mode None, write allowed — every pre-D26 identity is
    byte-identical."""
    legacy = cr.Identity(kind="client", name="alice", datasets=frozenset({"ds1"}))
    assert legacy.grants is None
    assert cr.grant_mode(legacy, "ds1") is None
    assert cr.dataset_write_allowed(legacy, "ds1") is True
    assert cr.grant_mode(legacy, "anything") is None


def test_grant_mode_specific_beats_wildcard_both_directions():
    """The most specific grant wins in either direction: ``*:ro`` + a
    per-dataset rw ⇒ rw; ``*:rw`` + a per-dataset ro ⇒ ro."""
    star_ro = cr.Identity(kind="client", name="u", datasets=frozenset({"ds1", "ds2"}), grants={"*": "ro", "ds1": "rw"})
    assert cr.grant_mode(star_ro, "ds1") == cr.GRANT_RW
    assert cr.dataset_write_allowed(star_ro, "ds1") is True
    assert cr.grant_mode(star_ro, "ds2") == cr.GRANT_RO
    assert cr.dataset_write_allowed(star_ro, "ds2") is False

    star_rw = cr.Identity(kind="client", name="v", datasets=frozenset({"ds1", "ds2"}), grants={"*": "rw", "ds1": "ro"})
    assert cr.grant_mode(star_rw, "ds1") == cr.GRANT_RO
    assert cr.dataset_write_allowed(star_rw, "ds1") is False
    assert cr.grant_mode(star_rw, "ds2") == cr.GRANT_RW


def test_grant_mode_no_matching_entry_is_none():
    """A grants dict with no entry for the dataset (no specific, no
    wildcard) → None — write allowed (the entry-missing case, e.g. a
    dataset reached via the D16 selection store)."""
    ident = cr.Identity(kind="client", name="u", datasets=frozenset({"ds2"}), grants={"ds2": "ro"})
    assert cr.grant_mode(ident, "other") is None
    assert cr.dataset_write_allowed(ident, "other") is True
    # an empty grants dict behaves the same way
    empty = cr.Identity(kind="client", name="e", datasets=frozenset(), grants={})
    assert cr.grant_mode(empty, "ds1") is None


def test_grant_mode_admin_and_identity_none_have_no_modes():
    """Admins (deployment keys) and identity-less callers (D15/D26 inactive)
    have no mode semantics: grant_mode None, write allowed — the
    byte-identical default."""
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    assert cr.grant_mode(admin, "ds1") is None
    assert cr.dataset_write_allowed(admin, "ds1") is True
    assert cr.grant_mode(None, "ds1") is None
    assert cr.dataset_write_allowed(None, "ds1") is True
    # require_dataset_write raises ONLY for an explicit ro mode
    cr.require_dataset_write(admin, "ds1")
    cr.require_dataset_write(None, "ds1")
    with pytest.raises(cr.DatasetAccessDenied, match="read-only"):
        cr.require_dataset_write(_ro(), "reports")


# ===========================================================================
# 4 · dataset_acls: env ∪ overlay merge (strongest restriction wins)
# ===========================================================================


def test_dataset_acls_env_overlay_merge_ro_wins(monkeypatch):
    """The merged registry keeps the STRONGEST restriction per dataset:
    env rw ∪ overlay ro ⇒ ro, and an operator ro is a floor the admin panel
    cannot silently lift (env ro ∪ overlay rw ⇒ ro)."""
    monkeypatch.setenv(cr.ACLS_ENV, "alice:ds1;bob:ds1:ro;carol:envonly")
    monkeypatch.setattr(
        ar,
        "overlay_acls",
        lambda: {
            "alice": {"ds1": "ro", "ds2": "rw"},  # env rw ∪ overlay ro → ro; overlay-only ds added
            "bob": {"ds1": "rw"},  # env ro ∪ overlay rw → ro (overlay cannot lift)
            "dave": {"*": "ro"},  # overlay-only name added wholesale
        },
    )
    merged = cr.dataset_acls()
    assert merged["alice"] == {"ds1": "ro", "ds2": "rw"}
    assert merged["bob"] == {"ds1": "ro"}
    assert merged["carol"] == {"envonly": "rw"}
    assert merged["dave"] == {"*": "ro"}
    # disabled overlay → exactly the env parse.  The real overlay_acls
    # short-circuits to {} when disabled (pinned against the real function
    # in the overlay test below); that empty overlay merges into the env
    # grants untouched.
    monkeypatch.setattr(ar, "overlay_acls", dict)
    assert cr.dataset_acls() == cr.parse_acls(os.environ.get(cr.ACLS_ENV, ""))


# ===========================================================================
# 5 · overlay_acls: the file's D26 grant forms
# ===========================================================================


def test_overlay_acls_file_grant_forms(tmp_path, monkeypatch):
    """The overlay file accepts plain strings (rw — every pre-D26 file is
    byte-compatible) and ``{"name","mode"}`` objects; unknown modes are
    DROPPED (fail-closed — a hand-edited file can never widen a write by
    typo); a non-list ``datasets`` yields no grants."""
    monkeypatch.setenv(ar.ADMIN_FILE_ENABLED_ENV, "1")
    path = tmp_path / "access" / "clients.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clients": {
                    "legacy": {"key": "key-legacy-0001", "datasets": ["reports", "notes"]},
                    "modes": {
                        "key": "key-modes-0002",
                        "datasets": [
                            {"name": "vault", "mode": "ro"},
                            {"name": "open", "mode": "rw"},
                            {"name": "loud", "mode": "RW"},  # case-normalized
                        ],
                    },
                    "typo": {"key": "key-typo-0003", "datasets": [{"name": "x", "mode": "admin"}]},
                    "badshape": {"key": "key-badsh-0004", "datasets": {"reports": True}},
                }
            }
        ),
        encoding="utf-8",
    )
    ar._mtime_cache.clear()
    acls = ar.overlay_acls()
    assert acls["legacy"] == {"reports": "rw", "notes": "rw"}
    assert acls["modes"] == {"vault": "ro", "open": "rw", "loud": "rw"}
    assert acls["typo"] == {}, "unknown mode must be dropped, never widened"
    assert acls["badshape"] == {}, "a non-list datasets field grants nothing"
    # explicitly disabled overlay → inert even with the file present
    monkeypatch.setenv(ar.ADMIN_FILE_ENABLED_ENV, "0")
    assert ar.overlay_acls() == {}


# ===========================================================================
# 6 · _normalize_grants: the REST/PATCH grant writer
# ===========================================================================


def test_normalize_grants_forms_and_duplicates():
    """Legacy strings pass through as strings; ``name:mode`` shorthand and
    ``{name, mode}`` objects normalize to the stored object form; duplicates
    last-wins; a bare ``*`` keeps the string form.  Output is sorted by
    dataset name (deterministic)."""
    # legacy wire form passes through verbatim
    assert ar._normalize_grants(["notes", "reports"]) == ["notes", "reports"]
    # shorthand → object
    assert ar._normalize_grants(["reports:ro"]) == [{"name": "reports", "mode": "ro"}]
    assert ar._normalize_grants(["reports:rw"]) == [{"name": "reports", "mode": "rw"}]
    # objects normalize (mode defaults to rw); sorted by name
    assert ar._normalize_grants([{"name": "vault", "mode": "ro"}, {"name": "open"}]) == [
        {"name": "open", "mode": "rw"},
        {"name": "vault", "mode": "ro"},
    ]
    # duplicates: last entry wins, in either spelling
    assert ar._normalize_grants(["ds1", "ds1:ro"]) == [{"name": "ds1", "mode": "ro"}]
    assert ar._normalize_grants([{"name": "ds1", "mode": "ro"}, "ds1"]) == ["ds1"]
    # the wildcard: bare * keeps the legacy string, *:mode becomes the object
    assert ar._normalize_grants(["*"]) == ["*"]
    assert ar._normalize_grants(["*:ro"]) == [{"name": "*", "mode": "ro"}]
    # empty tokens are skipped
    assert ar._normalize_grants(["", "  ", "ds1"]) == ["ds1"]


@pytest.mark.parametrize(
    ("raw", "exc"),
    [
        (["ds1:admin"], ValueError),  # shorthand with an unknown mode
        ([{"name": "ds1", "mode": "admin"}], ValueError),  # object with an unknown mode
        ([{"name": "", "mode": "ro"}], ValueError),  # object without a name
        ([42], TypeError),  # neither a string nor an object (TRY004: a type problem)
    ],
)
def test_normalize_grants_invalid_raises(raw, exc):
    with pytest.raises(exc):
        ar._normalize_grants(raw)


# ===========================================================================
# 7 · access_store: D26 global inclusion (dataset_allowed)
# ===========================================================================


def test_access_store_dataset_allowed_global_inclusion(monkeypatch):
    """A 🌐 public dataset is in EVERY authenticated identity's world
    (global-by-default, D26): an empty-grants client sees it without any
    checkbox; a deselect records an exclusion that hides even the global;
    admins see everything regardless; and the global rule rides the
    identity path — for ``identity None`` the predicate returns the
    unrestricted operator behaviour (unrestricted True) before any public
    check, and a store-off client is still included via the flag."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    _stamp_public("reports")
    u1 = cr.Identity(kind="client", name="u1", datasets=frozenset())
    u2 = cr.Identity(kind="client", name="u2", datasets=frozenset())
    assert acc.dataset_allowed(u1, "reports") is True
    assert acc.dataset_allowed(u2, "reports") is True, "every authenticated identity"
    # the exclusion escape hatch hides even a global until re-selected
    assert acc.deselect_dataset(u1, "reports") is True
    assert acc.dataset_allowed(u1, "reports") is False
    assert acc.dataset_allowed(u2, "reports") is True, "exclusions are per identity"
    # admins are never affected by the store
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    assert acc.dataset_allowed(admin, "reports") is True
    # identity None: the D26 global-inclusion branch is unreachable — the
    # None branch returns the (unrestricted) operator behaviour first, so
    # the public flag adds nothing for it either way.
    assert acc.dataset_allowed(None, "reports") is True
    # globals ride the IDENTITY path, not the store path: even with the
    # store OFF a client identity is included via the flag...
    monkeypatch.delenv(acc.STORE_ENV)
    plain = cr.Identity(kind="client", name="p", datasets=frozenset())
    assert acc.dataset_allowed(plain, "reports") is True


# ===========================================================================
# 8 · access_store: D26 write gate (dataset_write_allowed)
# ===========================================================================


def test_access_store_dataset_write_allowed_matrix(monkeypatch):
    """The ro sources, in precedence order: an explicit grant mode first
    (ro refuses, rw allows — specificity already resolved by
    clients_registry.grant_mode), then the 🌐 global flag (ro unless an
    explicit grant covers it — a self-selection NEVER upgrades it)."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    _stamp_public("pub")
    assert acc.dataset_write_allowed(_ro(), "reports") is False
    assert acc.dataset_write_allowed(_rw(), "reports") is True
    # no grant + public → ro (global-by-default)
    plain = cr.Identity(kind="client", name="plain", datasets=frozenset())
    assert acc.dataset_write_allowed(plain, "pub") is False
    # self-selecting the global records the ro path — still no write power
    # (D26 cross-validation fix: the entry is stamped write_mode "ro", so
    # the ro SURVIVES a later unpublish; the source tag says self-ro)
    assert acc.select_dataset(plain, "pub")["source"] == "self-ro"
    assert acc.dataset_write_allowed(plain, "pub") is False
    # an explicit rw grant DOES cover a public dataset (globals are ro by
    # DEFAULT, not by ceiling)
    rw_global = cr.Identity(kind="client", name="rwg", datasets=frozenset({"pub"}), grants={"pub": "rw"})
    assert acc.dataset_write_allowed(rw_global, "pub") is True
    # admin / identity-less → write allowed
    admin = cr.Identity(kind="admin", name=None, datasets=None)
    assert acc.dataset_write_allowed(admin, "pub") is True
    # store off: identity-less and grant-less non-public datasets write as
    # before (the byte-identical default)...
    monkeypatch.delenv(acc.STORE_ENV)
    other = cr.Identity(kind="client", name="o", datasets=frozenset({"mine"}))
    assert acc.dataset_write_allowed(other, "mine") is True
    assert acc.dataset_write_allowed(None, "mine") is True
    # ...but a client facing a PUBLIC dataset is still ro — the global flag
    # is store-independent (the write gate, not the store, enforces ro).
    assert acc.dataset_write_allowed(plain, "pub") is False


# ===========================================================================
# 9 · access_store: the ★ memory binding refuses read-only datasets
# ===========================================================================


def test_set_memory_dataset_refuses_read_only_binding(monkeypatch):
    """The ★ binding is the identity's ONE writable memory store: a dataset
    the key can only reach read-only refuses with a caller-safe message; an
    rw grant binds fine."""
    monkeypatch.setenv(acc.STORE_ENV, "1")
    ro = cr.Identity(kind="client", name="mem-ro", datasets=frozenset({"reports"}), grants={"reports": "ro"})
    with pytest.raises(acc.SelectionDenied, match="read-only for your key"):
        acc.set_memory_dataset(ro, "reports")
    assert acc.memory_dataset_for(ro) is None, "a refused binding must not stick"
    # rw → binds (and clearing works)
    rw = cr.Identity(kind="client", name="mem-rw", datasets=frozenset({"reports"}), grants={"reports": "rw"})
    assert acc.set_memory_dataset(rw, "reports") == "reports"
    assert acc.memory_dataset_for(rw) == "reports"
    assert acc.set_memory_dataset(rw, None) is None


# ===========================================================================
# 10 · MCP gate: _require_dataset_write
# ===========================================================================


def test_mcp_require_dataset_write_matrix():
    """The MCP write gate raises ToolError ("read-only") for a ro grant and
    stays silent for admins, identity-less callers and rw grants — the
    identity comes from the request contextvar (set/reset around each
    probe, mirroring test_dataset_acls.py)."""
    token = cr.set_current_identity(_ro())
    try:
        with pytest.raises(mcp.ToolError, match="read-only"):
            mcp._require_dataset_write("reports")
    finally:
        cr.reset_current_identity(token)
    # admin → silent
    token = cr.set_current_identity(cr.Identity(kind="admin", name=None, datasets=None))
    try:
        mcp._require_dataset_write("reports")
    finally:
        cr.reset_current_identity(token)
    # identity None (the store-off/legacy default) → silent
    token = cr.set_current_identity(None)
    try:
        mcp._require_dataset_write("reports")
    finally:
        cr.reset_current_identity(token)
    # rw → silent
    token = cr.set_current_identity(_rw())
    try:
        mcp._require_dataset_write("reports")
    finally:
        cr.reset_current_identity(token)


# ===========================================================================
# 11 · REST middleware gate: _rag_acl_path_denial
# ===========================================================================


def test_rest_path_denial_ro_matrix():
    """The REST deny-checker refuses the mutating dataset surfaces for a ro
    grant with the D26 denial string; the D16 select and D23 creator-publish
    POSTs are exempt (they are self-service / ownership-gated, not writes);
    GETs are never gated."""
    ro = _ro()
    ro_msg = "read-only for this API key"
    assert ro_msg in api._rag_acl_path_denial("/api/datasets/reports/documents", "POST", ro)
    assert ro_msg in api._rag_acl_path_denial("/api/datasets/reports/files", "POST", ro)
    assert ro_msg in api._rag_acl_path_denial("/api/datasets/reports", "DELETE", ro)
    # D16 exemption: selecting is how access is GAINED — never pre-empted
    assert api._rag_acl_path_denial("/api/datasets/reports/select", "POST", ro) is None
    # D23 exemption: the creator-publish toggle (ownership gate applies instead)
    assert api._rag_acl_path_denial("/api/datasets/reports/public", "POST", ro) is None
    # reads are never gated by the mode
    assert api._rag_acl_path_denial("/api/datasets/reports", "GET", ro) is None
    assert api._rag_acl_path_denial("/api/datasets/reports/documents", "GET", ro) is None
    # control: the same POST with an rw grant passes
    assert api._rag_acl_path_denial("/api/datasets/reports/documents", "POST", _rw()) is None
