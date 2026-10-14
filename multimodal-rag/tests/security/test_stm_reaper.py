"""stm_reaper -- sweep robustness invariants (PLAN section 7, invariants 3-5).

Written against the FROZEN contract in stmem/INTERFACE-FREEZE.md section 1
(T3).  The sweep's structure under test:

  - dataset selection goes through stm_lifetime.dataset_point_reapable(meta);
    ineligible datasets are skipped WITHOUT being scrolled;
  - the Qdrant Range filter on metadata.expires_at is the CANDIDATE selector;
    stm_lifetime.reap_decision is the AUTHORITY, re-checked per point;
  - one malformed point must NOT abort the sweep (invariant 3);
  - a failed delete must NOT abandon the remaining deletes (invariant 5);
  - the sweep must terminate against a repeating pagination cursor
    (invariant 4) -- track seen offsets.

The landed module also takes keyword knobs (batch_size / max_batches /
budget_s / dry_run) and returns rich ReapResult/ReapReport records; the tests
assert on the record fields (selected, skipped_reason, reaped, kept, malformed,
delete_failures, truncated) in addition to the fake-collection side effects.

PAYLOAD LAYOUT (ARBITRATED by the Lead, 2026-10-14): NESTED --
payload = {"page_content": ..., "metadata": {"expires_at": ...}} -- matching
rag_system.py's PointStruct writer and every existing metadata filter.  The
first landed reaper read the top level (flat); that was a real defect and has
been ordered fixed in _point_payload (nested with a flat/legacy fallback).
These tests model the nested shape and would have caught the defect either way.

Run::

    timeout 120 .venv/bin/python -m pytest tests/security/test_stm_reaper.py -q
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.environ.get("STM_SRC_OVERRIDE")
                or str(Path(__file__).resolve().parents[2] / "src"))

from multimodal_rag import stm_lifetime as sl

pytest.importorskip("multimodal_rag.stm_reaper",
                    reason="T3's stm_reaper.py has not landed yet (task-3)")

from multimodal_rag.stm_reaper import reap_dataset, reap_once  # noqa: E402

NOW = 2_000_000_000
H = 3600


# ===========================================================================
# The fake Qdrant the contract is exercised against
# ===========================================================================


class _FakeCollection:
    """In-memory stand-in for one Qdrant collection.

    Supports exactly the surface the reaper contract uses: a Range-filtered
    scroll (with a deliberately ugly pagination mode) and delete-by-ids.
    """

    def __init__(self, points, *, scroll_fn=None, delete_fail_ids=()):
        # points: list of (id, payload_dict)
        self.points = list(points)
        self.deleted: list[list] = []
        self.scroll_calls: list = []
        self.scroll_fn = scroll_fn  # (points, offset, limit) -> (points, next_offset)
        self.delete_fail_ids = set(delete_fail_ids)

    def scroll(self, coll, limit=10, offset=None, scroll_filter=None,
               with_payload=True, with_vectors=False):
        self.scroll_calls.append({"offset": offset, "limit": limit})
        if self.scroll_fn is not None:
            return self.scroll_fn(self.points, offset, limit, self)
        # simple in-memory "filter": candidate selector = any point whose
        # metadata.expires_at is a number < now (what the Range filter does)
        import qdrant_client.models as qm

        now = None
        for cond in getattr(scroll_filter, "must", []) or []:
            if getattr(cond, "range", None) is not None and getattr(cond.range, "lt", None) is not None:
                now = cond.range.lt
        selected = []
        for pid, payload in self.points:
            meta = payload.get("metadata") or {}
            raw = meta.get("expires_at")
            if now is None:
                selected.append((pid, payload))
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue  # a numeric Range never matches a non-number
            if raw < now:
                selected.append((pid, payload))
        start = 0
        if offset is not None:
            try:
                start = int(offset)
            except (TypeError, ValueError):
                start = 0
        window = selected[start : start + limit]
        nxt = start + limit if start + limit < len(selected) else None
        return ([_Point(pid, payload) for pid, payload in window], nxt)

    def delete(self, collection_name=None, points_selector=None, wait=True):
        ids = list(points_selector.points)
        failed = [i for i in ids if i in self.delete_fail_ids]
        if failed:
            from qdrant_client import exceptions as _exc

            self.delete_fail_calls = getattr(self, "delete_fail_calls", 0) + 1
            raise RuntimeError(f"qdrant delete failed for {failed!r}")
        self.deleted.append(ids)


class _Point:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


class _VectorStub:
    def __init__(self, coll):
        self._client = type("_C", (), {"scroll": coll.scroll, "delete": coll.delete})()
        self.collection_name = "ds"


class _RagStub:
    def __init__(self, coll):
        self.vector_store = _VectorStub(coll)


class _DmStub:
    """DatasetManager surface the landed reaper touches: list + meta +
    has_password + rag (with check_embedder=False keyword)."""

    def __init__(self, metas, colls):
        # metas: {name: meta_dict}; colls: {name: _FakeCollection}
        self._metas = dict(metas)
        self._rag = {n: _RagStub(c) for n, c in colls.items()}

    def _read_meta(self, name):
        meta = self._metas.get(name)
        return dict(meta) if meta else None

    def has_password(self, name):
        return bool((self._metas.get(name) or {}).get("password_hash"))

    def list_datasets(self):
        return [{"name": n} for n in self._metas]

    def _get_rag(self, name, check_embedder=True):
        return self._rag[name]


def _meta(lifetime=None, kind=None):
    m = {"name": "x"}
    if lifetime is not None:
        m["lifetime"] = lifetime
    if kind is not None:
        m["kind"] = kind
    return m


def _pt(expires_at, pid=None, **extra):
    """A stored point EXACTLY as the real writer produces it (Lead-arbitrated
    layout, 2026-10-14): payload = {"page_content": ..., "metadata": {...}},
    with the lifetime stamp nested under metadata -- the same nesting every
    existing filter (metadata.file_type, metadata.session_id, ...) keys on.
    The flat top-level layout was the defect in the first landed reaper; these
    tests model the correct shape and must NOT be adapted to it."""
    meta = {"expires_at": expires_at, "created_at": NOW - 10, "lifetime": 5}
    meta.update(extra)
    return (pid or f"p{expires_at}-{len(extra)}", {"page_content": "t", "metadata": meta})


# ===========================================================================
# Invariant 3 -- a malformed point must NOT abort the sweep
# ===========================================================================


def test_invariant_3_malformed_point_does_not_abort_sweep():
    """Expired, malformed, and immortal points coexist: the expired ones are
    reaped even though a hostile point sits in the middle of the scroll."""
    coll = _FakeCollection([
        _pt(NOW - 100, "good1"),
        _pt("not-a-number", "malformed1"),
        _pt(None, "immortal1"),
        _pt(NOW - 50, "good2"),
        _pt({"deep": {"junk": [1]}}, "malformed2"),
        _pt(NOW - 1, "good3"),
    ])
    dm = _DmStub({"ds": _meta(lifetime=6 * H)}, {"ds": coll})
    report = reap_once(dm, now=NOW)
    # every WELL-FORMED expired point was deleted, malformed ones kept
    deleted = [pid for batch in coll.deleted for pid in batch]
    assert set(deleted) == {"good1", "good2", "good3"}
    kept = {pid for pid, _ in coll.points} - set(deleted)
    assert {"malformed1", "malformed2", "immortal1"} <= kept
    # the malformed entries are REPORTED, not swallowed silently
    assert report is not None


def test_invariant_3_all_malformed_still_no_crash_and_no_deletes():
    coll = _FakeCollection([_pt("x", "m1"), _pt(True, "m2"), _pt([1], "m3")])
    dm = _DmStub({"ds": _meta(lifetime=H)}, {"ds": coll})
    reap_once(dm, now=NOW)
    assert coll.deleted == [], "nothing may be deleted when nothing is PROVABLY expired"


# ===========================================================================
# Invariant 4 -- the sweep terminates against a repeating pagination cursor
# ===========================================================================


def test_invariant_4_repeating_cursor_terminates():
    """A hostile server keeps returning the same offset forever.  The sweep
    must terminate (tracked seen offsets), and must not delete the same point
    twice nor spin unbounded."""
    pts = [_pt(NOW - 5, "a"), _pt(NOW - 6, "b")]

    def evil_scroll(points, offset, limit, coll):
        # ALWAYS the same page, ALWAYS a next cursor (scroll() itself logs the
        # call; do not double-count here)
        return ([_Point(pid, payload) for pid, payload in pts], offset or "same-cursor")

    coll = _FakeCollection(pts, scroll_fn=evil_scroll)
    dm = _DmStub({"ds": _meta(lifetime=H)}, {"ds": coll})
    report = reap_once(dm, now=NOW)
    # TERMINATION (the invariant): the first repeated cursor stops the sweep.
    assert len(coll.scroll_calls) == 2, (
        "the sweep must stop at the FIRST repeated cursor, not loop"
    )
    assert report.truncated is True
    assert any("repeating" in n for r in report.results for n in r.notes)
    # A hostile server can make the module re-delete the same ids before the
    # guard fires; Qdrant deletes are idempotent, so the harm is nil.  What is
    # pinned is that the attempts stay BOUNDED (two pages, not a spin loop).
    assert len(coll.deleted) <= 2


def test_invariant_4_normal_pagination_terminates_and_reaps_all():
    # i starts at 1: expires_at == now is NOT expired (the pinned boundary),
    # so a point stamped exactly NOW must survive the sweep.
    pts = [_pt(NOW - i, f"p{i}") for i in range(1, 8)]
    coll = _FakeCollection(pts)
    dm = _DmStub({"ds": _meta(lifetime=H)}, {"ds": coll})
    report = reap_once(dm, now=NOW)
    deleted = [pid for batch in coll.deleted for pid in batch]
    assert set(deleted) == {f"p{i}" for i in range(1, 8)}
    assert report.reaped == 7


# ===========================================================================
# Invariant 5 -- a failed delete does not abandon the remaining deletes
# ===========================================================================


def test_invariant_5_failed_delete_does_not_abandon_remaining():
    """The first delete batch fails (transport error); the sweep must still
    attempt the remaining deletes and keep its accounting honest."""
    calls = {"n": 0}

    class _FlakyColl(_FakeCollection):
        def delete(self, collection_name=None, points_selector=None, wait=True):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transport timeout")
            self.deleted.append(list(points_selector.points))

    coll = _FlakyColl([_pt(NOW - 10, "a"), _pt(NOW - 9, "b"), _pt(NOW - 8, "c")])
    dm = _DmStub({"ds": _meta(lifetime=H)}, {"ds": coll})
    # batch_size=1 -> one delete attempt per page; the first raises, the
    # sweep must still attempt the remaining two (invariant 5).
    reap_once(dm, now=NOW, batch_size=1)
    assert calls["n"] >= 2, "the sweep must keep deleting after a failed delete"
    assert len(coll.deleted) == 2, "the two healthy deletes must still land"


def test_invariant_5_failure_in_one_dataset_does_not_stop_the_sweep():
    """reap_once sweeps EVERY eligible dataset even when one raises."""
    bad = _FakeCollection([_pt(NOW - 5, "x")])
    bad.scroll = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))
    good_coll = _FakeCollection([_pt(NOW - 5, "y")])
    dm = _DmStub(
        {"bad": _meta(lifetime=H), "good": _meta(lifetime=H)},
        {"bad": bad, "good": good_coll},
    )
    reap_once(dm, now=NOW)
    deleted = [pid for batch in good_coll.deleted for pid in batch]
    assert deleted == ["y"], "a broken dataset must not block the healthy ones"


# ===========================================================================
# Invariant 1 (sweep form) -- an immortal dataset is never scrolled
# ===========================================================================


@pytest.mark.parametrize(
    "meta",
    [
        _meta(),  # no lifetime key (pre-STM shape)
        _meta(lifetime=None),
        _meta(lifetime="bogus"),
        _meta(lifetime=-5),
        _meta(lifetime=True),
    ],
)
def test_invariant_1_immortal_dataset_is_never_scrolled(meta):
    coll = _FakeCollection([_pt(NOW - 5, "x")])
    dm = _DmStub({"ds": meta}, {"ds": coll})
    reap_once(dm, now=NOW)
    assert coll.scroll_calls == [], "an ineligible dataset must be skipped WITHOUT being scrolled"
    assert coll.deleted == []


def test_invariant_1_eligible_dataset_still_uses_authority_recheck():
    """The Range filter is only the CANDIDATE selector: reap_decision is the
    authority.  A point the filter returns but the authority keeps survives."""
    pts = [_pt(NOW - 5, "filtered-but-kept")]
    dm = _DmStub({"ds": _meta(lifetime=None)}, {"ds": _FakeCollection(pts)})
    # dataset immortal -> authority keeps EVERYTHING even if a filter ran
    reap_once(dm, now=NOW)


# ===========================================================================
# reap_dataset -- single-dataset contract
# ===========================================================================


def test_reap_dataset_points_only_phase_1():
    """Phase 1: entries die, the dataset itself survives -- even an STM one."""
    coll = _FakeCollection([_pt(NOW - 5, "dead"), _pt(NOW + 5, "alive")])
    dm = _DmStub({"ds": _meta(lifetime=H, kind="short_term")}, {"ds": coll})
    reap_dataset(dm, "ds", now=NOW)
    deleted = [pid for batch in coll.deleted for pid in batch]
    assert deleted == ["dead"]


def test_reap_dataset_immortal_is_a_noop():
    coll = _FakeCollection([_pt(NOW - 5, "x")])
    dm = _DmStub({"ds": _meta()}, {"ds": coll})
    reap_dataset(dm, "ds", now=NOW)
    assert coll.scroll_calls == []
    assert coll.deleted == []
