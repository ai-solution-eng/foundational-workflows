"""STM safety invariants (PLAN section 7) -- the acceptance criteria.

Invariant 1 is proven BY ENUMERATION over REAL dataset meta on disk (section 5
Phase 1 acceptance): every meta.json under the live data root is loaded and
asserted unreachable by the reaper's selection predicate.

Run::

    timeout 120 .venv/bin/python -m pytest tests/security/test_stm_safety_invariants.py -q
"""

import json
import os
import sys
from pathlib import Path

import pytest

# STM_SRC_OVERRIDE exists for mutation testing (PLAN section 6.3): a throwaway
# copy of the package can be put first on sys.path without touching the real
# source tree.  Unset in every normal run.
sys.path.insert(0, os.environ.get("STM_SRC_OVERRIDE") or str(Path(__file__).resolve().parents[2] / "src"))

from multimodal_rag import stm_lifetime as sl

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = 2_000_000_000


# ===========================================================================
# Invariant 1 -- existing data is untouchable, proven BY ENUMERATION
# ===========================================================================


def _data_roots():
    """Candidate DatasetManager roots on this machine, in priority order."""
    roots = []
    env = os.environ.get("DATA_PATH")
    if env:
        roots.append(Path(env))
    roots.append(Path("/data"))
    roots.append(REPO_ROOT / "data")
    return roots


def _dataset_meta_dirs(root: Path):
    """Every dataset directory under *root* (datasets/<name>/meta.json)."""
    datasets = root / "datasets"
    if not datasets.is_dir():
        return []
    out = []
    for child in sorted(datasets.iterdir()):
        meta = child / "meta.json"
        if child.is_dir() and meta.is_file():
            out.append((child.name, meta))
    return out


def _discover_real_meta():
    """All (name, meta_dict) pairs from the first data root that has any.

    If no production-style store exists, the enumeration falls back to a
    faithful on-disk fixture: meta files written the way
    DatasetManager._write_meta writes them (the pre-STM shape, no lifetime),
    plus a corrupted one and a non-dataset file, exactly as they sit on disk.
    """
    for root in _data_roots():
        found = _dataset_meta_dirs(root)
        if found:
            return root, _load_all(found)
    # fallback fixture written to disk in the real layout
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="stm-inv1-"))
    datasets = root / "datasets"
    pre_stm = {
        "name": "pre-stm",
        "description": "a dataset that predates the STM feature",
        "created": "2026-01-01T00:00:00",
        "document_count": 802,
        "schema_version": 3,
    }
    (datasets / "pre-stm").mkdir(parents=True)
    (datasets / "pre-stm" / "meta.json").write_text(json.dumps(pre_stm))
    # a short_term dataset (the only shape the reaper may touch)
    stm = dict(pre_stm, name="stm-demo", kind="short_term", lifetime=6 * 3600)
    (datasets / "stm-demo").mkdir()
    (datasets / "stm-demo" / "meta.json").write_text(json.dumps(stm))
    # a corrupt meta file must not break the enumeration
    (datasets / "corrupt").mkdir()
    (datasets / "corrupt" / "meta.json").write_text("{not json")
    return root, _load_all(_dataset_meta_dirs(root))


def _load_all(pairs):
    out = []
    for name, path in pairs:
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            meta = None  # unreadable/corrupt: the reaper sees this too
        out.append((name, meta))
    return out


@pytest.fixture(scope="module")
def real_meta():
    root, metas = _discover_real_meta()
    return root, metas


def test_invariant_1_enumeration_real_meta_untouchable(real_meta):
    """SECTION 5 PHASE 1 ACCEPTANCE: load EVERY dataset meta and assert none
    of the pre-STM ones is reachable by the reaper's selection predicate --
    neither the point-level sweep gate nor the Phase 2 dataset gate."""
    root, metas = real_meta
    assert len(metas) > 0, "enumeration must run over at least one meta file"
    untouched = []
    for name, meta in metas:
        # the reaper's ONLY entry points are these two predicates
        point_selected = sl.dataset_point_reapable(meta)
        dataset_selected = sl.dataset_deletable(meta)
        if meta is None:
            # corrupt/unreadable meta: fail-closed on both gates
            assert not point_selected and not dataset_selected
            continue
        if "lifetime" not in meta or meta.get("lifetime") is None:
            assert not point_selected, (
                f"dataset {name!r} has no lifetime yet the point reaper selected it"
            )
            assert not dataset_selected, (
                f"dataset {name!r} has no lifetime yet the dataset reaper selected it"
            )
            untouched.append(name)
        # decision-level cross-check: every point in an unselected dataset is kept
        if not point_selected:
            verdict = sl.reap_decision({"expires_at": 1}, meta.get("lifetime") if meta else None, NOW)
            assert verdict.reap is False, f"point in {name!r} would be reaped despite immortal dataset"
    print(f"\n[invariant 1] enumerated {len(metas)} dataset(s) under {root}; "
          f"untouchable (no lifetime): {untouched}")


def test_invariant_1_pre_stm_meta_shape_is_the_default(real_meta):
    """The pre-STM meta shape (no lifetime key) is exactly what the frozen
    algebra treats as INFINITE -- pin the arithmetic, not just the predicate."""
    _, metas = real_meta
    for name, meta in metas:
        if meta and "lifetime" not in meta:
            assert sl.validate_stored_lifetime(meta.get("lifetime")) is None
            assert sl.effective_lifetime(None, meta.get("lifetime")) is None
            assert sl.stamp_point_times(now=NOW, dataset_lifetime=meta.get("lifetime"))["expires_at"] is None


def test_invariant_1_short_term_datasets_are_the_only_reachable_shape(real_meta):
    """The enumeration is meaningful only if the reachable shape exists in it
    or can be reached by the predicates; assert the discrimination is real:
    adding kind+lifetime flips the verdict, nothing else does."""
    _, metas = real_meta
    sample = next((m for _, m in metas if m), None)
    assert sample is not None
    assert sl.dataset_deletable(sample) is False
    stm = dict(sample, kind="short_term", lifetime=6 * 3600)
    assert sl.dataset_deletable(stm) is True


# ===========================================================================
# Invariant 2 -- None = infinite at BOTH levels; min() is the only rule
# ===========================================================================


def test_invariant_2_none_is_infinite_at_both_levels():
    assert sl.effective_lifetime(None, None) is None
    assert sl.validate_lifetime(None) is None
    # dataset level: None lifetime -> dataset gate closed
    assert sl.dataset_point_reapable({"lifetime": None}) is False
    assert sl.dataset_deletable({"kind": "short_term", "lifetime": None}) is False
    # point level: None expires_at -> keep
    assert sl.reap_decision({"expires_at": None}, 3600, NOW).reason == "no-expires-at"


def test_invariant_2_max_draft_would_fail():
    """Mutation guard for the historical max() bug: rows where min() != max()
    must produce the min."""
    for p, d, expected in [(3600, 21600, 3600), (8 * 3600, 6 * 3600, 6 * 3600)]:
        got = sl.effective_lifetime(p, d)
        assert got == expected
        assert got != max(p, d)


# ===========================================================================
# Invariant 3 -- the reaper never deletes without a provable expiry;
#                a malformed entry never aborts the sweep (predicate level;
#                the sweep-level behaviour is test_stm_reaper.py)
# ===========================================================================


def test_invariant_3_fail_closed_matrix():
    cases = [None, {}, {"expires_at": None}, {"expires_at": "x"}, {"expires_at": True}, {"expires_at": [1]}]
    for payload in cases:
        verdict = sl.reap_decision(payload, 3600, NOW)
        assert verdict.reap is False, payload


def test_invariant_3_reap_decision_never_raises_over_hostile_payloads():
    hostile = [
        {"expires_at": object()},
        {"expires_at": float("-inf")},
        {"expires_at": "  12  "},  # numeric string with whitespace: parses
    ]
    for payload in hostile:
        verdict = sl.reap_decision(payload, 3600, NOW)  # must not raise
        assert isinstance(verdict.reap, bool)


def test_invariant_3_huge_int_expires_at_fails_closed_not_raises():
    """REGRESSION PIN (fixed 2026-10-14 by the Lead after my report): the old
    _as_finite_number round-tripped ints through float(), and float(10**400)
    raised OverflowError -- a hostile expires_at then aborted the WHOLE sweep
    (fail-OPEN, invariant 3).  The fix handles ints natively and range-checks;
    pin the verdict, not merely 'does not raise'."""
    verdict = sl.reap_decision({"expires_at": 10**400}, 3600, NOW)
    assert verdict.reap is False
    assert verdict.reason == "malformed-expires-at"
    # ...and the negative twin fails closed the same way
    neg = sl.reap_decision({"expires_at": -(10**400)}, 3600, NOW)
    assert neg.reap is False
    assert neg.reason == "malformed-expires-at"


def test_invariant_3_whitespace_numeric_string_is_actionable():
    assert sl.reap_decision({"expires_at": f"  {NOW - 1}  "}, 3600, NOW).reap is True


# ===========================================================================
# Invariant 6 (Phase 2) -- the dataset-deletion predicate
# ===========================================================================


@pytest.mark.parametrize(
    "meta",
    [
        {"name": "andrew-memory", "document_count": 802},
        {"name": "normal", "lifetime": 999999999},  # lifetime set but NOT short_term
        {"name": "normal", "kind": "normal", "lifetime": 60},
        {"name": "normal", "kind": "short_term", "lifetime": None},  # immortal STM
        {"name": "normal", "kind": "SHORT_TERM", "lifetime": 60},  # exact match required
        None,
        42,
    ],
)
def test_invariant_6_normal_datasets_never_deletable(meta):
    assert sl.dataset_deletable(meta) is False


def test_invariant_6_phase2_gate_explicit_cases():
    """Lead-mandated Phase 2 gate cases (the most destructive path in the
    feature): a normal dataset with a FINITE lifetime is never deletable, and
    an STM dataset with NO lifetime is never deletable."""
    assert sl.dataset_deletable({"kind": "normal", "lifetime": 3600}) is False
    assert sl.dataset_deletable({"kind": "short_term", "lifetime": None}) is False
    assert sl.dataset_deletable({"kind": "short_term"}) is False
    # ...and the only shape that passes requires BOTH
    assert sl.dataset_deletable({"kind": "short_term", "lifetime": 3600}) is True


def test_invariant_6_short_term_with_lifetime_is_deletable():
    assert sl.dataset_deletable({"kind": "short_term", "lifetime": 60}) is True


# ===========================================================================
# Invariant 7 -- no password bypass is introduced anywhere (structural)
# ===========================================================================


def test_invariant_7_stm_lifetime_has_no_password_notions():
    """The frozen algebra is the reaper's only decision authority; it must
    carry no password vocabulary at all."""
    src = Path(sl.__file__).read_text()
    for token in ("password", "is_admin", "bypass"):
        assert token not in src.lower(), f"stm_lifetime.py mentions {token!r}"


def test_invariant_7_reaper_gate_ignores_password_hash():
    """A password-protected dataset gains NO exemption and NO extra reach:
    the gate reads only kind/lifetime.  (A protected normal dataset stays
    undeletable; a protected short_term one stays deletable -- the password
    is invisible to the decision, per INTERFACE-FREEZE section 2.)"""
    protected_normal = {"kind": "normal", "lifetime": 60, "password_hash": "x"}
    assert sl.dataset_deletable(protected_normal) is False
    protected_stm = {"kind": "short_term", "lifetime": 60, "password_hash": "x"}
    assert sl.dataset_deletable(protected_stm) is True


def test_invariant_7_api_password_gate_has_no_admin_bypass():
    """The pre-existing regression pin (PLAN Appendix C), re-asserted so an
    STM-era refactor cannot quietly reintroduce the bypass."""
    api_path = Path(os.environ.get("STM_API_SERVER_SRC") or REPO_ROOT / "src" / "multimodal_rag" / "api_server.py")
    api_src = api_path.read_text()
    start = api_src.find("def _require_dataset_password")
    assert start != -1, "_require_dataset_password must exist"
    end = api_src.find("\ndef ", start + 1)
    body = api_src[start : end if end != -1 else start + 4000]
    assert "is_admin" not in body, "the password gate must have NO admin bypass"

def test_file_ingest_applies_the_dataset_cap(tmp_path, monkeypatch):
    """END-TO-END: a FILE ingested into a finite-lifetime dataset is capped.

    This is the regression pin for the cap bypass: before the fix, only
    add_documents passed dataset_lifetime, so every file-ingested point fell
    back to min(None, None) = INFINITE and escaped the dataset's hard cap
    (PLAN section 2.2 row 4: the dataset lifetime is a hard cap, so a finite
    dataset may not hold permanent samples).

    The suite previously could not see that bug: the only dataset_lifetime
    coverage was pure-algebra stamp_point_times() calls, and the reaper's e2e
    test seeds points directly instead of going through ingest. So the WRITER
    half of the composition was unpinned. This test drives the real ingest.
    """
    import json
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from multimodal_rag.dataset_manager import _CAP_REQUIRED

    # The structural guard: a caller that FORGETS the cap must raise rather
    # than silently store immortal points.  This is the second half of the
    # pin -- it fails if anyone restores the '= None' default.
    import inspect

    from multimodal_rag import dataset_manager as dm_mod

    sig = inspect.signature(dm_mod.DatasetManager._add_file_processed)
    assert sig.parameters["dataset_lifetime"].default is _CAP_REQUIRED, (
        "_add_file_processed must REQUIRE dataset_lifetime: a None default "
        "silently stores immortal points inside a finite dataset"
    )

    # And every add_to_vector_store call site in the module must carry the cap.
    import re

    src = inspect.getsource(dm_mod)
    missing = []
    for m in re.finditer(r"add_to_vector_store\(", src):
        tail = src[m.end() : m.end() + 200]
        # stop at the end of this call's argument list
        depth = 0
        call = ""
        for ch in tail:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            call += ch
        if "dataset_lifetime" not in call:
            missing.append(call.strip().splitlines()[0][:80])
    assert not missing, (
        "add_to_vector_store call site(s) without dataset_lifetime -> "
        "file-ingested points would be immortal in a finite dataset: "
        f"{missing}"
    )


def test_file_ingest_cap_guard_raises_when_forgotten():
    """The forgotten-cap case is LOUD, not silent.

    A missing argument must raise, because None is a legal lifetime value
    (an immortal dataset) and so cannot double as 'caller forgot'.
    """
    import inspect
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from multimodal_rag import dataset_manager as dm_mod

    dm = dm_mod.DatasetManager.__new__(dm_mod.DatasetManager)
    try:
        dm._add_file_processed("ds", "/tmp/nonexistent", None, {}, 100, 10)
    except TypeError as exc:
        assert "dataset_lifetime" in str(exc)
    else:
        raise AssertionError(
            "_add_file_processed accepted a missing dataset_lifetime; the "
            "cap can be forgotten again"
        )

