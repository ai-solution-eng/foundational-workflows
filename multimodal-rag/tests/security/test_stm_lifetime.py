"""stm_lifetime -- the frozen lifetime algebra (stmem/INTERFACE-FREEZE.md section 0).

RATIFIED SEMANTICS: PLAN section 2.2 truth table, min() with None = INFINITY.

Run::

    timeout 120 .venv/bin/python -m pytest tests/security/test_stm_lifetime.py -q
"""

import os
import sys
from pathlib import Path

import pytest

# STM_SRC_OVERRIDE exists for mutation testing (PLAN section 6.3): a throwaway
# copy of the package can be put first on sys.path without touching the real
# source tree.  Unset in every normal run.
sys.path.insert(0, os.environ.get("STM_SRC_OVERRIDE") or str(Path(__file__).resolve().parents[2] / "src"))

from multimodal_rag import stm_lifetime as sl

H = 3600


# ===========================================================================
# 1. The 2.2 truth table -- all five rows, parametrized
# ===========================================================================


@pytest.mark.parametrize(
    ("point", "dataset", "expected"),
    [
        (None, None, None),  # infinite -- today's behaviour, nothing may be deleted
        (None, 6 * H, 6 * H),  # point inherits the dataset cap
        (1 * H, 6 * H, 1 * H),  # short point stays short
        (8 * H, 6 * H, 6 * H),  # dataset is a HARD CAP (the max() draft was wrong)
        (1 * H, None, 1 * H),  # point caps itself in an immortal dataset
    ],
    ids=[
        "inf+inf=inf",
        "none+6h=6h",
        "1h+6h=1h",
        "8h+6h=6h (dataset cap)",
        "1h+none=1h",
    ],
)
def test_truth_table_effective_lifetime(point, dataset, expected):
    assert sl.effective_lifetime(point, dataset) == expected


def test_truth_table_min_is_the_only_combination_rule():
    """Invariant 2, exhaustive over a small symmetric grid: for every
    combination, the effective lifetime is min() with None = INFINITY -- never
    max(), never the point value alone, never the dataset value alone."""
    lifetimes = [None, 1, H, 8 * H, 10**9]
    for p in lifetimes:
        for d in lifetimes:
            expected = _min_inf(p, d)
            assert sl.effective_lifetime(p, d) == expected, (p, d)
            # symmetry is a consequence of min(); assert it explicitly
            assert sl.effective_lifetime(d, p) == expected


def _min_inf(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


# ===========================================================================
# 2. stamp_point_times -- server-stamped, expiry computed, audit fields kept
# ===========================================================================


def test_stamp_point_times_full_shape():
    stamp = sl.stamp_point_times(now=1_000_000, point_lifetime=H, dataset_lifetime=6 * H)
    assert stamp == {"lifetime": H, "created_at": 1_000_000, "expires_at": 1_000_000 + H}


def test_stamp_immortal_dataset_inherits_dataset_cap():
    stamp = sl.stamp_point_times(now=5, point_lifetime=None, dataset_lifetime=6 * H)
    assert stamp["lifetime"] == 6 * H
    assert stamp["expires_at"] == 5 + 6 * H


def test_stamp_all_none_is_explicitly_immortal():
    """None (rather than an omitted key) keeps the payload self-describing."""
    stamp = sl.stamp_point_times(now=5, point_lifetime=None, dataset_lifetime=None)
    assert stamp == {"lifetime": None, "created_at": 5, "expires_at": None}


def test_stamp_row5_point_in_immortal_dataset_is_stored_not_enforced():
    """Row 5's expires_at is WRITTEN (truthful and auditable) even though the
    reaper skips immortal datasets entirely (INTERFACE-FREEZE section 3)."""
    stamp = sl.stamp_point_times(now=10, point_lifetime=H, dataset_lifetime=None)
    assert stamp["expires_at"] == 10 + H


def test_stamp_uses_only_the_server_clock():
    """The client never supplies created_at: the only clock input is 'now'."""
    stamp = sl.stamp_point_times(now=77, point_lifetime=3)
    assert stamp["created_at"] == 77


def test_stamp_rejects_client_style_float_now():
    with pytest.raises(ValueError):
        sl.stamp_point_times(now=1_000.5, point_lifetime=H)


# ===========================================================================
# 3. compute_expires_at + the i64 range bound
# ===========================================================================


def test_compute_expires_at_none_lifetime_is_none():
    assert sl.compute_expires_at(1000, None) is None


def test_compute_expires_at_overflow_refused():
    with pytest.raises(ValueError, match="64-bit"):
        sl.compute_expires_at(sl.I64_MAX, 1)
    # boundary: exactly at the bound is fine
    assert sl.compute_expires_at(sl.I64_MAX - 1, 1) == sl.I64_MAX


# ===========================================================================
# 4. Write-time strictness (fail LOUD at the boundary)
# ===========================================================================


@pytest.mark.parametrize(
    "bad",
    [
        True,  # bool is an int subclass; must never pass as 1 second
        False,
        -1,
        0,
        -H,
        1.5,  # fractional seconds
        float("nan"),
        float("inf"),
        "3600",  # a string is a caller bug at WRITE time (strict)
        [3600],
        {"seconds": 3600},
    ],
)
def test_validate_lifetime_rejects_malformed(bad):
    with pytest.raises(ValueError):
        sl.validate_lifetime(bad)


@pytest.mark.parametrize("good", [None, 1, H, 6 * H, 3600.0])  # whole float OK
def test_validate_lifetime_accepts_valid(good):
    assert sl.validate_lifetime(good) == (None if good is None else int(good))


def test_validate_lifetime_field_name_in_error():
    with pytest.raises(ValueError, match="dataset lifetime"):
        sl.validate_lifetime(-2, field="dataset lifetime")


def test_effective_lifetime_propagates_malformed_input():
    """Malformed lifetimes must not silently become INFINITY at write time."""
    for bad in (True, -1, 0, 1.5, "x"):
        with pytest.raises(ValueError):
            sl.effective_lifetime(bad, 6 * H)
        with pytest.raises(ValueError):
            sl.effective_lifetime(6 * H, bad)


# ===========================================================================
# 5. validate_stored_lifetime -- the fail-closed read-back variant
# ===========================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),  # INFINITE is legitimate
        (6 * H, 6 * H),
        (3600.0, 3600),
        (True, None),  # bool -> malformed -> None = KEEP, never raises
        (-1, None),
        (0, None),
        ("nonsense", None),
        ([], None),
        (float("nan"), None),
    ],
)
def test_validate_stored_lifetime_never_raises(value, expected):
    assert sl.validate_stored_lifetime(value) == expected


def test_stored_never_raises_where_strict_raises():
    for value in (True, -1, 0, 1.5, "x", {"a": 1}):
        with pytest.raises(ValueError):
            sl.validate_lifetime(value)
        assert sl.validate_stored_lifetime(value) is None


# ===========================================================================
# 6. The boundary: expires_at == now is NOT expired
# ===========================================================================


def test_boundary_expires_equal_now_is_kept():
    """expires_at < now reaps; expires_at == now does NOT.  A point that dies
    at T is still alive AT T."""
    now = 1_000_000
    d = sl.reap_decision({"expires_at": now}, 6 * H, now)
    assert d.reap is False
    assert d.reason == "not-expired"


def test_boundary_one_second_past_is_reaped():
    now = 1_000_000
    d = sl.reap_decision({"expires_at": now - 1}, 6 * H, now)
    assert d.reap is True
    assert d.reason == "expired"


# ===========================================================================
# 7. The malformed fail-closed matrix (reap time)
# ===========================================================================


NOW = 2_000_000
DS = 6 * H


@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (None, "no-payload"),
        ({}, "no-expires-at"),
        ({"expires_at": None}, "no-expires-at"),
        ({"expires_at": "garbage"}, "malformed-expires-at"),
        ({"expires_at": float("nan")}, "malformed-expires-at"),
        ({"expires_at": float("inf")}, "malformed-expires-at"),
        ({"expires_at": True}, "malformed-expires-at"),  # bool is not a timestamp
        ({"expires_at": [NOW]}, "malformed-expires-at"),
        ({"expires_at": {"t": NOW}}, "malformed-expires-at"),
    ],
)
def test_malformed_payload_fails_closed(payload, expected_reason):
    d = sl.reap_decision(payload, DS, NOW)
    assert d.reap is False
    assert d.reason == expected_reason


def test_absurdly_huge_expires_at_fails_closed():
    """A value outside any sane range (here: beyond float's range, and beyond
    the i64 window) reads as MALFORMED -> KEEP.  Never raises, never reaps."""
    d = sl.reap_decision({"expires_at": "99999999999999999999999"}, DS, NOW)
    assert d.reap is False
    assert d.reason == "malformed-expires-at"
    d2 = sl.reap_decision({"expires_at": 10**400}, DS, NOW)
    assert d2.reap is False
    assert d2.reason == "malformed-expires-at"


def test_numeric_string_expires_at_is_honoured():
    """A numeric string is unambiguous; failing it would leak points a buggy
    writer stored as text.  Finite numeric strings parse and are actionable."""
    d = sl.reap_decision({"expires_at": str(NOW - 5)}, DS, NOW)
    assert d.reap is True
    d2 = sl.reap_decision({"expires_at": str(NOW + 5)}, DS, NOW)
    assert d2.reap is False


def test_float_expires_at_is_honoured():
    d = sl.reap_decision({"expires_at": float(NOW - 5)}, DS, NOW)
    assert d.reap is True


def test_reap_decision_returns_expires_at_on_expired():
    d = sl.reap_decision({"expires_at": NOW - 1}, DS, NOW)
    assert d.expires_at == NOW - 1


def test_reap_decision_rejects_malformed_now():
    for bad in (None, -1, "now", 1.5, True):
        with pytest.raises(ValueError):
            sl.reap_decision({"expires_at": NOW - 1}, DS, bad)


# ===========================================================================
# 8. Dataset-level gates (the reaper's selection predicates)
# ===========================================================================


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"lifetime": 6 * H}, True),
        ({"lifetime": None}, False),  # immortal -- never selected
        ({}, False),  # pre-feature dataset
        ({"lifetime": "6h"}, False),  # malformed -- fail closed
        ({"lifetime": -1}, False),
        ({"lifetime": True}, False),
        (None, False),  # non-dict
        ("lifetime", False),
        ([("lifetime", H)], False),
    ],
)
def test_dataset_point_reapable(meta, expected):
    assert sl.dataset_point_reapable(meta) is expected


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"kind": "short_term", "lifetime": 6 * H}, True),
        ({"kind": "short_term", "lifetime": None}, False),  # kind alone is never enough
        ({"kind": "short_term"}, False),
        ({"lifetime": 6 * H}, False),  # lifetime alone is never enough
        ({"kind": "normal", "lifetime": 6 * H}, False),  # invariant 6: a normal dataset, whatever its fields
        ({}, False),
        (None, False),
    ],
)
def test_dataset_deletable(meta, expected):
    assert sl.dataset_deletable(meta) is expected


def test_immortal_dataset_short_circuits_every_point():
    """Invariant 1, decision-level: a dataset with no lifetime keeps points
    whose own expires_at is long past -- the dataset gate is checked FIRST."""
    d = sl.reap_decision({"expires_at": 1}, None, NOW)
    assert d.reap is False
    assert d.reason == "dataset-immortal"


def test_reap_decision_reason_tokens_are_stable():
    """Tests and logs match on reason tokens, not prose; pin the full set."""
    reasons = set(ReapDecision := sl.ReapDecision._fields)  # noqa: F841
    observed = {
        sl.reap_decision(None, DS, NOW).reason,
        sl.reap_decision({}, None, NOW).reason,
        sl.reap_decision({}, DS, NOW).reason,
        sl.reap_decision({"expires_at": "x"}, DS, NOW).reason,
        sl.reap_decision({"expires_at": NOW - 1}, DS, NOW).reason,
        sl.reap_decision({"expires_at": NOW + 1}, DS, NOW).reason,
    }
    assert observed == {"no-payload", "dataset-immortal", "no-expires-at", "malformed-expires-at", "expired", "not-expired"}
