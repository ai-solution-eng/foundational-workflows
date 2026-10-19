"""STM reaper end-to-end: REAL payload shapes against a real DatasetManager.

Written after a real defect (2026-10-14): the reaper passed the OUTER payload
({page_content, metadata}) to stm_lifetime.reap_decision, which reads
expires_at at the TOP level -- while the candidate filter keys on the NESTED
metadata.expires_at.  Against real points the authority therefore answered
"no-expires-at" for everything and the feature was silently INERT (fail-
closed, so no data loss, but nothing was ever reaped).

The regression test is deliberately NOT built on flat synthetic dicts: every
point here is written in the shape the real writer uses
(rag_system.py: payload={"page_content": ..., "metadata": doc.metadata}).

Run::

    timeout 120 .venv/bin/python -m pytest tests/security/test_stm_reaper_e2e.py -q
"""

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.stm_reaper import reap_once
from multimodal_rag.utils import bm25 as bm25_lane

NOW = 2_000_000_000
H = 3600
_DIM = 8  # tests/conftest.py stub-embedder dimension


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def dm(base_path, embedder) -> DatasetManager:
    """Real offline DatasetManager rooted in a temp dir (house harness)."""
    return DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)


def _real_payload(meta: dict) -> dict:
    """The EXACT payload shape rag_system.py writes (rag_system.py:2653)."""
    return {"page_content": "some text", "metadata": meta}


def _seed(dm: DatasetManager, name: str, payloads: list[dict]):
    """Insert points directly, bypassing ingest (no embedder round-trip)."""
    from qdrant_client.models import PointStruct

    store = dm._get_rag(name, check_embedder=False).vector_store
    # Explicit dict[str, Any]: the bare literal infers dict[Any, ...] (the key
    # is a module constant) and dict is INVARIANT in its value type, so neither
    # dict[Any, ...] nor dict[str, list[float]] satisfies PointStruct.vector's
    # named-vector mapping union.
    vector: dict[str, Any] = {bm25_lane.DENSE_VECTOR_NAME: [0.1] * _DIM}
    store._client.upsert(
        store.collection_name,
        points=[PointStruct(id=uuid.uuid4().hex, vector=vector, payload=p) for p in payloads],
    )
    return store


def _count(store) -> int:
    return store._client.count(store.collection_name).count


def _result(report, name: str):
    return next(r for r in report.results if r.dataset == name)


def _short_term(dm: DatasetManager, name: str, lifetime: int = 6 * H):
    return dm.create_dataset(name, lifetime=lifetime, kind="short_term")


# ---------------------------------------------------------------------------
# THE REGRESSION: a REAL-shape expired point must actually be reaped
# ---------------------------------------------------------------------------


def test_real_payload_shape_expired_point_is_reaped(dm):
    """Nested {"metadata": {"expires_at": <past>}} -> deleted, end to end."""
    _short_term(dm, "stm")
    store = _seed(
        dm,
        "stm",
        [
            _real_payload({"expires_at": NOW - 10, "lifetime": H}),
            _real_payload({"expires_at": NOW - 1, "lifetime": H}),
            _real_payload({"expires_at": NOW + 10, "lifetime": H}),
        ],
    )
    assert _count(store) == 3

    report = reap_once(dm, now=NOW)

    result = _result(report, "stm")
    assert result.selected is True
    assert result.reaped == 2, "real-shape expired points must be reaped (the inert-reaper bug)"
    assert result.errors == []
    assert _count(store) == 1, "only the live point may remain"


def test_real_payload_shape_survivor_is_the_live_point(dm):
    """The point that survives is the one whose expiry is in the future --
    proving the reaper read the NESTED stamp rather than guessing."""
    _short_term(dm, "stm")
    store = _seed(
        dm,
        "stm",
        [
            _real_payload({"expires_at": NOW - 10, "tag": "dead"}),
            _real_payload({"expires_at": NOW + 10, "tag": "alive"}),
        ],
    )
    reap_once(dm, now=NOW)
    left = store._client.scroll(store.collection_name, limit=10, with_payload=True)[0]
    assert [p.payload["metadata"]["tag"] for p in left] == ["alive"]


def test_flat_payload_is_never_a_candidate(dm):
    """The candidate filter keys on the NESTED metadata.expires_at, so a flat
    point is never returned by the scroll and is therefore never deleted."""
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [{"expires_at": NOW - 10}])
    report = reap_once(dm, now=NOW)
    result = _result(report, "stm")
    assert result.scanned == 0, "a flat point must not even be selected"
    assert result.reaped == 0
    assert _count(store) == 1


def test_point_payload_prefers_nested_metadata_with_flat_fallback():
    """_point_payload is what makes the authority and the filter agree: it
    hands reap_decision the SAME mapping the filter indexed, and still passes a
    flat payload through unchanged."""
    from multimodal_rag.stm_reaper import _point_payload

    class P:
        def __init__(self, payload):
            self.payload = payload

    nested = {"page_content": "t", "metadata": {"expires_at": 7}}
    assert _point_payload(P(nested)) == {"expires_at": 7}
    assert _point_payload(P({"expires_at": 7})) == {"expires_at": 7}
    assert _point_payload(P({"metadata": None, "expires_at": 7})) == {"metadata": None, "expires_at": 7}
    assert _point_payload(P(None)) is None


# ---------------------------------------------------------------------------
# Invariant 3 -- fail closed, and one bad point never aborts the sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "garbage",
        True,
        [NOW - 1],
        {"t": NOW - 1},
        float("nan"),
        float("inf"),
        10**400,  # huge int: float() would raise OverflowError
        None,
    ],
)
def test_malformed_expiry_is_kept_and_never_aborts_the_sweep(dm, bad):
    """A malformed point is KEPT, and the good points in the SAME sweep are
    still reaped (invariant 3: fail-closed, non-aborting)."""
    _short_term(dm, "stm")
    store = _seed(
        dm,
        "stm",
        [
            _real_payload({"expires_at": NOW - 10}),
            _real_payload({"expires_at": bad}),
            _real_payload({"expires_at": NOW - 5}),
        ],
    )
    report = reap_once(dm, now=NOW)
    result = _result(report, "stm")
    assert result.reaped == 2, "malformed expiry must not stop the other two being reaped"
    assert result.errors == []
    assert _count(store) == 1


def test_point_with_no_expiry_stamp_is_kept(dm):
    """A point with no stamp is not even a candidate (the Range never
    matches a missing value), so it is never touched."""
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [_real_payload({"source": "no-stamp"})])
    report = reap_once(dm, now=NOW)
    assert _result(report, "stm").reaped == 0
    assert _count(store) == 1


def test_boundary_expires_equal_now_is_kept(dm):
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [_real_payload({"expires_at": NOW})])
    reap_once(dm, now=NOW)
    assert _count(store) == 1


# ---------------------------------------------------------------------------
# Invariant 1 -- existing data is untouchable: never even OPENED
# ---------------------------------------------------------------------------


def test_immortal_dataset_is_never_scrolled_or_opened(dm: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset with no lifetime is skipped BEFORE any Qdrant traffic:
    neither _get_rag nor scroll is called for it, even though it holds a
    point whose expiry is far in the past."""
    dm.create_dataset("legacy")  # no lifetime -> immortal
    store = _seed(dm, "legacy", [_real_payload({"expires_at": NOW - 99999})])

    opened: list[str] = []
    scrolled: list[str] = []
    real_get_rag = DatasetManager._get_rag
    real_scroll = type(store._client).scroll

    def spy_get_rag(self: Any, name: Any, *a: Any, **k: Any) -> Any:
        opened.append(name)
        return real_get_rag(self, name, *a, **k)

    def spy_scroll(self: Any, collection_name: Any, *a: Any, **k: Any) -> Any:
        scrolled.append(collection_name)
        return real_scroll(self, collection_name, *a, **k)

    monkeypatch.setattr(DatasetManager, "_get_rag", spy_get_rag)
    monkeypatch.setattr(type(store._client), "scroll", spy_scroll)

    report = reap_once(dm, now=NOW)

    result = _result(report, "legacy")
    assert result.selected is False
    assert result.skipped_reason == "not-point-reapable"
    assert result.scanned == 0
    assert opened == [], "an ineligible dataset must not even be opened"
    assert scrolled == [], "an ineligible dataset must not be scrolled"
    assert _count(store) == 1, "the ancient point in an immortal dataset is untouchable"


def test_corrupt_meta_is_invisible_to_the_sweep(dm):
    """A dataset whose meta.json cannot be parsed is not even LISTED by
    DatasetManager, so the sweep never sees it -- fail-closed by construction."""
    dm.create_dataset("broken")
    meta_path = Path(dm.base_path) / "datasets" / "broken" / "meta.json"
    meta_path.write_text("{not json")

    report = reap_once(dm, now=NOW)

    assert [r.dataset for r in report.results] == [], "an unreadable dataset is never swept"
    assert report.reaped == 0


def test_corrupt_meta_named_explicitly_is_skipped_fail_closed(dm):
    """The same dataset reached by NAME (the CronJob's --dataset path) is
    skipped with the no-meta token rather than swept on a guess."""
    from multimodal_rag.stm_reaper import reap_dataset

    dm.create_dataset("broken")
    (Path(dm.base_path) / "datasets" / "broken" / "meta.json").write_text("{not json")

    result = reap_dataset(dm, "broken", NOW)

    assert result.selected is False
    assert result.skipped_reason == "no-meta"
    assert result.scanned == 0


def test_password_protected_dataset_is_skipped_not_bypassed(dm):
    """Invariant 7: the reaper never bypasses a password.  It SKIPS the
    dataset (even a reaping-eligible one) and says so."""
    dm.create_dataset("locked", lifetime=6 * H, kind="short_term", password="s3cret")
    store = _seed(dm, "locked", [_real_payload({"expires_at": NOW - 10})])

    report = reap_once(dm, now=NOW)

    result = _result(report, "locked")
    assert result.selected is False
    assert result.skipped_reason == "password-protected"
    assert _count(store) == 1, "nothing may be deleted from a protected dataset"


# ---------------------------------------------------------------------------
# Invariant 4 -- the sweep always terminates
# ---------------------------------------------------------------------------


def test_repeating_pagination_cursor_terminates(dm, monkeypatch):
    """A cursor that never advances must not loop forever."""
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [_real_payload({"expires_at": NOW + 10**6})])

    from qdrant_client.models import Record

    sticky = Record(id=uuid.uuid4().hex, payload=_real_payload({"expires_at": NOW + 10**6}))
    calls = {"n": 0}

    def sticky_scroll(self, collection_name, *a, **k):
        calls["n"] += 1
        assert calls["n"] < 10, "the sweep did not terminate against a repeating cursor"
        return [sticky], "same-offset-forever"

    monkeypatch.setattr(type(store._client), "scroll", sticky_scroll)

    report = reap_once(dm, now=NOW)

    result = _result(report, "stm")
    assert result.truncated is True
    assert any("repeating pagination offset" in n for n in result.notes)


def test_page_cap_bounds_the_sweep(dm):
    _short_term(dm, "stm")
    _seed(dm, "stm", [_real_payload({"expires_at": NOW - 10}) for _ in range(5)])
    report = reap_once(dm, now=NOW, batch_size=1, max_batches=2)
    result = _result(report, "stm")
    assert result.batches == 2
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Invariant 5 -- a failed delete does not abandon the remaining deletes
# ---------------------------------------------------------------------------


def test_failed_delete_does_not_abandon_remaining_deletes(dm, monkeypatch):
    """Batch 1 fails; batch 2 must still be attempted and must succeed."""
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [_real_payload({"expires_at": NOW - 10}) for _ in range(3)])

    real_delete = type(store._client).delete
    calls = {"n": 0}

    def flaky_delete(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transport failure")
        return real_delete(self, *a, **k)

    monkeypatch.setattr(type(store._client), "delete", flaky_delete)

    report = reap_once(dm, now=NOW, batch_size=1)

    result = _result(report, "stm")
    assert result.delete_failures == 1
    assert calls["n"] >= 2, "a failed delete must not abandon the remaining deletes"
    assert result.reaped == 2, "the two batches after the failure are still deleted"
    assert any("delete of" in e for e in result.errors)


# ---------------------------------------------------------------------------
# dry run -- selection proof with no mutation
# ---------------------------------------------------------------------------


def test_dry_run_reports_but_deletes_nothing(dm):
    _short_term(dm, "stm")
    store = _seed(dm, "stm", [_real_payload({"expires_at": NOW - 10}) for _ in range(3)])

    report = reap_once(dm, now=NOW, dry_run=True)

    result = _result(report, "stm")
    assert result.reaped == 3, "dry-run counts what WOULD be deleted"
    assert result.delete_attempts == 0
    assert _count(store) == 3, "dry-run must not delete anything"


def test_reap_report_totals(dm):
    _short_term(dm, "stm")
    _seed(
        dm,
        "stm",
        [_real_payload({"expires_at": NOW - 10}), _real_payload({"expires_at": NOW + 10})],
    )
    report = reap_once(dm, now=NOW)
    assert report.reaped == 1
    assert report.kept == 0, "the live point is not a candidate (filtered out)"
    assert report.datasets == 1
    assert "reaped=1" in report.summary_line()


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["now", -1, 1.5, True])
def test_reap_once_rejects_a_malformed_clock(dm, bad):
    """now is an operator input, not stored data: it fails LOUD."""
    with pytest.raises(ValueError):
        reap_once(dm, now=bad)


def test_reap_once_uses_the_wall_clock_when_now_is_omitted(dm):
    """now=None is legitimate: the server clock is the only clock trusted."""
    report = reap_once(dm, now=None)
    assert report.now > 1_600_000_000
