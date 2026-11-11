"""Feature B tests: memory consolidation (decay, archive, contradictions).

The policy is pure (plan_consolidation takes an explicit now), so the decay
math, the three-condition archive gate and the report-only contradiction rule
are asserted exactly.  apply_plan and the consolidate_memories MCP tool are
driven through doubles that record every write, so "dry run writes nothing" is
verified against the calls, not against a promise in a docstring.

Offline: no cluster, no models, no network.

Run::

    pytest tests/test_memory_consolidation.py -q
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import multimodal_rag.mcp_server as mcp
from multimodal_rag import memory_consolidation as mc

NOW = datetime(2026, 11, 10, tzinfo=UTC).timestamp()
DAY = 86400.0


def _iso(days_ago: float) -> str:
    return datetime.fromtimestamp(NOW - days_ago * DAY, UTC).isoformat()


def _record(
    memory_id, kind="note", text="a durable note about the deployment", *, days_ago=0.0, access_count=0, **extra
):
    record = {
        "memory_id": memory_id,
        "kind": kind,
        "text": text,
        "memory_ts": _iso(days_ago),
    }
    if access_count:
        record["access_count"] = access_count
    record.update(extra)
    return record


def _plan(records, **config):
    return mc.plan_consolidation(records, now=NOW, config=config or None)


def _ids(rows):
    return [row["memory_id"] for row in rows]


# ===========================================================================
# The archive gate
# ===========================================================================


def test_protected_kinds_is_the_playbook_only():
    assert mc.PROTECTED_KINDS == ("playbook",)


def test_old_never_accessed_note_is_archived():
    plan = _plan([_record("old", days_ago=365)])
    assert _ids(plan["archive"]) == ["old"]
    assert plan["decay"] == []
    assert plan["archive"][0]["action"] == "archive"


def test_old_but_accessed_note_is_scored_not_archived():
    """Access evidence is the second gate: something recalled is still in use."""
    plan = _plan([_record("used", days_ago=365, access_count=3)])
    assert plan["archive"] == []
    assert _ids(plan["decay"]) == ["used"]
    assert plan["decay"][0]["action"] == "score"


def test_old_playbook_is_never_archived():
    """A playbook's value is its counters — archiving it would destroy the evidence."""
    plan = _plan([_record("pb", kind="playbook", days_ago=365)])
    assert plan["archive"] == []
    assert _ids(plan["decay"]) == ["pb"]
    assert plan["decay"][0]["protected"] is True


def test_fresh_note_is_in_no_action_list():
    plan = _plan([_record("fresh", days_ago=1)])
    assert plan["archive"] == [] and plan["decay"] == []
    assert plan["stats"]["above_floor"] == 1


def test_undated_record_is_not_archived_on_a_guess():
    plan = _plan([{"memory_id": "undated", "kind": "note", "text": "no timestamp"}])
    assert plan["archive"] == []
    assert plan["stats"]["undated"] == 1


def test_stats_describe_the_scan():
    plan = _plan(
        [
            _record("old", days_ago=400),
            _record("used", days_ago=400, access_count=2),
            _record("pb", kind="playbook", days_ago=400),
            _record("fresh", days_ago=2),
        ]
    )
    stats = plan["stats"]
    assert stats["total"] == 4
    assert stats["archive"] == 1
    assert stats["decay"] == 2
    assert stats["above_floor"] == 1
    assert stats["protected"] == 1
    assert stats["by_kind"] == {"note": 3, "playbook": 1}
    assert stats["protected_kinds"] == ["playbook"]
    assert stats["half_life_days"] == 30.0 and stats["floor"] == 0.15
    assert stats["now"].startswith("2026-11-10T00:00:00")


def test_older_record_has_a_lower_decayed_score():
    # A high floor puts EVERY record below it, so the action list carries the
    # scores of records that would otherwise be considered healthy.
    plan = _plan([_record("young", days_ago=10), _record("older", days_ago=60)], floor=10.0)
    by_id = {row["memory_id"]: row for row in plan["archive"]}
    assert by_id["older"]["decayed"] < by_id["young"]["decayed"]
    assert by_id["older"]["age_days"] > by_id["young"]["age_days"]


def test_decay_follows_the_half_life_curve():
    """Two half-lives must cut the score by roughly four (bounded by the mild
    recency boost inside importance, which is why the bounds are wide)."""
    plan = _plan([_record("t0", days_ago=0), _record("t2", days_ago=60)], floor=10.0)
    by_id = {row["memory_id"]: row for row in plan["archive"]}
    ratio = by_id["t2"]["decayed"] / by_id["t0"]["decayed"]
    assert 0.16 < ratio < 0.35


def test_floor_zero_archives_nothing():
    plan = _plan([_record("old", days_ago=1000)], floor=0.0)
    assert plan["archive"] == []
    assert plan["stats"]["archive"] == 0


def test_kind_weights_order_decisions_above_session_history():
    plan = _plan(
        [_record("decision", kind="decision", days_ago=20), _record("hist", kind="session_history", days_ago=20)],
        floor=10.0,
    )
    by_id = {row["memory_id"]: row for row in plan["archive"]}
    assert by_id["decision"]["decayed"] > by_id["hist"]["decayed"]


def test_raw_qdrant_payload_shape_is_accepted():
    payload = {
        "memory_id": "raw",
        "page_content": "an old note stored as a qdrant payload",
        "metadata": {"memory_kind": "note", "memory_ts": _iso(365)},
    }
    plan = _plan([payload])
    assert _ids(plan["archive"]) == ["raw"]
    assert "old note" in plan["archive"][0]["text_preview"]


def test_text_preview_is_truncated():
    plan = _plan([_record("long", text="x" * 500, days_ago=365)], text_preview_chars=40)
    assert len(plan["archive"][0]["text_preview"]) == 40


# ===========================================================================
# Config
# ===========================================================================


@pytest.mark.parametrize(
    "config",
    [
        {"half_life_days": 0},
        {"half_life_days": -1},
        {"floor": -0.1},
        {"contradiction_threshold": 0},
        {"contradiction_threshold": 2},
    ],
)
def test_invalid_config_raises(config):
    with pytest.raises(ValueError):
        mc.plan_consolidation([], now=NOW, config=config)


def test_non_mapping_config_raises():
    with pytest.raises(ValueError):
        mc.plan_consolidation([], now=NOW, config=["not", "a", "mapping"])


def test_unknown_config_keys_are_reported_not_silently_ignored():
    plan = _plan([], half_life_days=7, halflife_days=7)
    assert plan["stats"]["ignored_config_keys"] == ["halflife_days"]
    assert plan["stats"]["half_life_days"] == 7.0


# ===========================================================================
# Contradictions (report-only)
# ===========================================================================


def test_near_duplicate_with_opposing_markers_is_reported():
    left = _record("a", text="always enable the cache for the api endpoint in production at startup")
    right = _record("b", text="never enable the cache for the api endpoint in production at startup")
    plan = _plan([left, right])
    assert len(plan["contradictions"]) == 1
    contradiction = plan["contradictions"][0]
    assert {contradiction["memory_id_a"], contradiction["memory_id_b"]} == {"a", "b"}
    assert contradiction["similarity"] >= 0.92
    assert contradiction["action"] == "report-only"
    assert set(contradiction["polarity"]) == {"always", "never"}


def test_near_duplicate_without_opposing_markers_is_not_a_contradiction():
    left = _record("a", text="always enable the cache for the api endpoint in production at startup")
    right = _record("b", text="always enable the cache for the api endpoint in production at startup please")
    plan = _plan([left, right])
    assert plan["contradictions"] == []


def test_unrelated_records_are_not_contradictions():
    plan = _plan([_record("a", text="always enable the cache"), _record("b", text="never write a changelog by hand")])
    assert plan["contradictions"] == []


def test_embeddings_are_preferred_over_the_lexical_fallback():
    left = _record("a", text="use the cache", embedding=[1.0, 0.0, 0.0])
    right = _record("b", text="avoid the cache", embedding=[1.0, 0.0, 0.0])
    plan = _plan([left, right])
    assert len(plan["contradictions"]) == 1
    assert plan["contradictions"][0]["similarity"] == 1.0


def test_contradictions_are_never_in_the_archive_set():
    left = _record("a", text="always enable the cache for the api endpoint in production at startup", days_ago=400)
    right = _record("b", text="never enable the cache for the api endpoint in production at startup", days_ago=400)
    plan = _plan([left, right])
    assert len(plan["contradictions"]) == 1
    archive_ids = set(_ids(plan["archive"]))
    assert archive_ids <= {"a", "b"}, "the report is independent of the archive decision"
    assert plan["contradictions"][0]["action"] == "report-only"


# ===========================================================================
# apply_plan
# ===========================================================================


class _PlanDM:
    def __init__(self, fail_delete: bool = False, touched: int | None = None) -> None:
        self.deleted: list[list[str]] = []
        self.patches: list[tuple[list[str], dict]] = []
        self.fail_delete = fail_delete
        self.touched = touched

    def delete_documents(self, dataset: str, ids: list[str]) -> int:
        if self.fail_delete:
            raise RuntimeError("storage down")
        self.deleted.append(list(ids))
        return len(ids)

    def update_memory_payload(self, dataset: str, ids: list[str], patch: dict) -> int:
        self.patches.append((list(ids), dict(patch)))
        return len(ids) if self.touched is None else self.touched


def _plan_with_actions():
    return _plan(
        [
            _record("old", days_ago=365),
            _record("used", days_ago=365, access_count=2),
            _record("pb", kind="playbook", days_ago=365),
        ]
    )


def test_apply_plan_is_a_dry_run_by_default():
    dm = _PlanDM()
    result = mc.apply_plan(dm, "mem", _plan_with_actions())
    assert result["applied"] is False
    assert result["would_archive"] == ["old"]
    assert set(result["would_score"]) == {"used", "pb"}
    assert result["archived"] == 0 and result["scored"] == 0
    assert dm.deleted == [] and dm.patches == [], "a dry run must not touch the store"


def test_apply_plan_applies_archive_and_scores():
    dm = _PlanDM()
    result = mc.apply_plan(dm, "mem", _plan_with_actions(), dry_run=False)
    assert result["applied"] is True
    assert dm.deleted == [["old"]]
    assert result["archived"] == 1
    assert result["scored"] == 2
    scored_ids = {patch[0][0] for patch in dm.patches}
    assert scored_ids == {"used", "pb"}
    patch = dm.patches[0][1]
    assert "decay_score" in patch and "decay_checked_at" in patch
    assert patch["decay_floor"] == 0.15


def test_apply_plan_refuses_to_delete_a_protected_kind_even_if_the_plan_says_so():
    """Defence in depth: the executor, not only the planner, enforces protection."""
    plan = {
        "archive": [{"memory_id": "pb", "kind": "playbook"}],
        "decay": [],
        "contradictions": [],
        "stats": {"floor": 0.15},
    }
    dm = _PlanDM()
    result = mc.apply_plan(dm, "mem", plan, dry_run=False)
    assert dm.deleted == []
    assert result["archived"] == 0
    assert result["skipped_protected"] == ["pb"]


def test_apply_plan_reports_errors_without_aborting_the_rest():
    dm = _PlanDM(fail_delete=True)
    result = mc.apply_plan(dm, "mem", _plan_with_actions(), dry_run=False)
    assert result["archived"] == 0
    assert any("delete_documents" in error for error in result["errors"])
    assert result["scored"] == 2, "a failed delete must not skip the score writes"


def test_apply_plan_reports_a_no_op_patch():
    dm = _PlanDM(touched=0)
    result = mc.apply_plan(dm, "mem", _plan_with_actions(), dry_run=False)
    assert result["scored"] == 0
    assert any("no ids touched" in error for error in result["errors"])


# ===========================================================================
# consolidate_memories MCP tool
# ===========================================================================


class _ConsolidationDM:
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.deleted: list[list[str]] = []
        self.patches: list[tuple[list[str], dict]] = []

    def get_dataset(self, name: str, sync_count: bool = True) -> dict:
        return {"name": name, "document_count": len(self.points)}

    def has_password(self, name: str) -> bool:
        return False

    def scroll_documents(self, name: str, scroll_filter=None, limit: int = 500) -> list[tuple[str, dict]]:
        return list(self.points.items())[:limit]

    def delete_documents(self, dataset: str, ids: list[str]) -> int:
        self.deleted.append(list(ids))
        return len(ids)

    def update_memory_payload(self, dataset: str, ids: list[str], patch: dict) -> int:
        self.patches.append((list(ids), dict(patch)))
        return len(ids)


@pytest.fixture()
def consolidation_dm(monkeypatch):
    dm = _ConsolidationDM()
    monkeypatch.setattr(mcp, "_dm", dm, raising=False)
    monkeypatch.setattr(mcp, "get_manager", lambda: dm, raising=False)
    monkeypatch.setattr(mcp, "_require_dataset_acl", lambda name: None, raising=False)
    monkeypatch.setattr(mcp, "_require_dataset_write", lambda name: None, raising=False)
    monkeypatch.setattr(mcp, "_resolve_and_unlock", lambda dm_, name, pw: None, raising=False)
    monkeypatch.setattr(mcp, "_resolve_memory_dataset", lambda name: name or "test-memory", raising=False)
    monkeypatch.setattr(mcp, "_resolve_memory_password", lambda pw: pw, raising=False)
    monkeypatch.delenv("MEMORY_DATASET", raising=False)
    dm.points["old"] = {"page_content": "an old note", "metadata": {"memory_kind": "note", "memory_ts": _iso(365)}}
    dm.points["pb"] = {
        "page_content": "## Playbook",
        "metadata": {"memory_kind": "playbook", "memory_ts": _iso(365), "task_signature": "x"},
    }
    return dm


def _consolidate(**kwargs):
    return json.loads(asyncio.run(mcp.consolidate_memories(**kwargs)))


def test_consolidate_tool_is_a_dry_run_by_default(consolidation_dm):
    out = _consolidate()
    assert out["dry_run"] is True
    assert out["scanned"] == 2
    assert out["apply"]["applied"] is False
    assert out["apply"]["would_archive"] == ["old"]
    assert out["apply"]["would_score"] == ["pb"]
    assert consolidation_dm.deleted == [] and consolidation_dm.patches == []


def test_consolidate_tool_applies_when_asked(consolidation_dm):
    out = _consolidate(dry_run=False)
    assert out["dry_run"] is False
    assert consolidation_dm.deleted == [["old"]]
    assert out["apply"]["archived"] == 1
    assert out["apply"]["scored"] == 1
    assert consolidation_dm.patches[0][1]["decay_score"] is not None


def test_consolidate_tool_requires_write_access_to_apply(consolidation_dm, monkeypatch):
    def _read_only(name):
        raise mcp.ToolError("read-only")

    monkeypatch.setattr(mcp, "_require_dataset_write", _read_only, raising=False)
    with pytest.raises(mcp.ToolError):
        _consolidate(dry_run=False)
    assert consolidation_dm.deleted == []


def test_consolidate_tool_reports_stats_and_contradictions(consolidation_dm):
    consolidation_dm.points["a"] = {
        "page_content": "always enable the cache for the api endpoint in production at startup",
        "metadata": {"memory_kind": "note", "memory_ts": _iso(1)},
    }
    consolidation_dm.points["b"] = {
        "page_content": "never enable the cache for the api endpoint in production at startup",
        "metadata": {"memory_kind": "note", "memory_ts": _iso(1)},
    }
    out = _consolidate()
    assert out["stats"]["total"] == 4
    assert len(out["contradictions"]) == 1
    assert out["apply"]["would_archive"] == ["old"]


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  {name} ... OK")
        except Exception:
            failures += 1
            print(f"  {name} ... FAIL")
            traceback.print_exc()
    print(f"{'All tests passed!' if not failures else f'{failures} test(s) failed'}")
    sys.exit(1 if failures else 0)
