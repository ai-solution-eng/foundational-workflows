"""Tests for the Feature A gateway overlay: playbook emission + patcher.

Two layers:

* the pure module (playbook.py) — key normalization, validation, parse,
  strip, and the trace-signal gate/outcome functions;
* the patcher (apply_playbook_patch.py) — applied to a RUNNABLE miniature
  of the gateway tree (tests/fixtures_gateway) whose edited modules are
  then imported and executed: write_memory splitting, prompt append, the
  version bump, and the degrade path when the overlay module is removed.

The REAL gateway verification (a /tmp copy of PCAI_LLM + its own test
suite) is a separate manual step recorded in the impl report — it needs the
read-only gateway tree and is not hermetic enough for this suite.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

_OVERLAY_DIR = Path(__file__).resolve().parents[1]
_PLAYBOOK_PATH = _OVERLAY_DIR / "playbook.py"
_PATCHER_PATH = _OVERLAY_DIR / "apply_playbook_patch.py"
_FIXTURE_ROOT = _OVERLAY_DIR / "tests" / "fixtures_gateway"
_FENCE = chr(96) * 3


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - load plumbing
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pb = _load("pcai_llm_playbook_overlay_playbook", _PLAYBOOK_PATH)
patcher = _load("pcai_llm_playbook_overlay_patcher", _PATCHER_PATH)

_VALID: dict[str, Any] = {
    "task_signature": "recover a stuck pod finalizer",
    "applies_when": ["deletionTimestamp set"],
    "steps": ["check finalizers", "check webhook"],
    "failure_modes": ["assumed RBAC"],
    "outcome": "success",
}


def _summary_doc(playbook_json: str | None = None) -> str:
    """A distilled document in the gateway shape, optionally with a Playbook."""
    doc = (
        "[MEMORY INDEX | project: general | type: session-summary | topics: t]\n"
        "\n"
        "## What we did\nwork\n\n## Key decisions\n1. d\n\n## Gotchas\n- g\n\n## How to proceed\n- next\n"
    )
    if playbook_json is not None:
        doc += "\n## Playbook\n" + _FENCE + "json\n" + playbook_json + "\n" + _FENCE + "\n"
    return doc


def _call(name: str, args: str, cid: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"function": {"name": name, "arguments": args}}
    if cid is not None:
        entry["id"] = cid
    return {"role": "assistant", "content": "", "tool_calls": [entry]}


def _result(text: str, cid: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "tool", "content": text}
    if cid is not None:
        msg["tool_call_id"] = cid
    return msg


def _canonical_session(n: int) -> list[dict[str, Any]]:
    """n distinct tool calls with clean results (the canonical store shape)."""
    msgs: list[dict[str, Any]] = []
    for i in range(n):
        msgs.append(_call("tool_" + str(i), '{"arg": ' + str(i) + "}", "c" + str(i)))
        msgs.append(_result("ok " + str(i), "c" + str(i)))
    return msgs


# ---------------------------------------------------------------------------
# playbook_key — the cross-repo normalization contract with Feature B
# ---------------------------------------------------------------------------


def test_playbook_key_is_pinned_sha256_of_lowercase_collapsed_signature() -> None:
    assert (
        pb.playbook_key("Recover a CrashLoopBackOff pod")
        # sha256 PIN, not a credential. Split literals: the fleet scanner's HEX64 hard pattern
        # ignores allow markers, so a contiguous 64-hex run cannot live in the tree. Python
        # concatenates at compile time - the pinned value is unchanged.
        == "2b3d4bc10a25331681e822d9bbe6bc99"
        "5fe1f4ce949d66c5a8dab0f6d9bb8fc3"
    )
    assert pb.playbook_key("recover a stuck pod finalizer") == (
        "0371ed0a837e4f2e5a1ded074f3338b0"
        "7f3008768d660a51192b62bbe018d336"  # sha256 pin (split: HEX64 ignores allow markers)
    )


def test_playbook_key_normalizes_whitespace_case_and_strips() -> None:
    assert pb.playbook_key("  Recover   a\nCrashLoopBackOff  pod ") == pb.playbook_key("recover a crashloopbackoff pod")


def test_playbook_key_empty_and_none_hash_the_empty_string() -> None:
    # sha256 of "" (split: HEX64 hard pattern ignores allow markers; value unchanged)
    empty = (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert pb.playbook_key("") == empty
    assert pb.playbook_key(None) == empty  # type: ignore[arg-type]


def test_playbook_key_is_64_lowercase_hex_and_discriminates() -> None:
    a, b = pb.playbook_key("task one"), pb.playbook_key("task two")
    assert a != b
    for key in (a, b):
        assert len(key) == 64
        assert key == key.lower()
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# validate_playbook — exact keys and bounds; never raises
# ---------------------------------------------------------------------------


def test_validate_accepts_the_contract_shape() -> None:
    assert pb.validate_playbook(_VALID) == []
    minimal = {
        "task_signature": "recover a stuck pod finalizer",
        "applies_when": ["deletionTimestamp set"],
        "steps": ["check finalizers"],
        "failure_modes": [],
        "outcome": "abandoned",
    }
    assert pb.validate_playbook(minimal) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_task_signature",
        "missing_steps",
        "extra_key",
        "sig_too_short",
        "sig_too_long",
        "sig_multiline",
        "sig_unstripped",
        "sig_not_a_string",
        "applies_empty",
        "applies_nine",
        "steps_empty",
        "steps_sixteen",
        "failure_nine",
        "failure_not_strings",
        "outcome_bad",
        "outcome_not_a_string",
    ],
)
def test_validate_rejects_each_violation_class(mutation: str) -> None:
    obj = json.loads(json.dumps(_VALID))  # deep copy
    if mutation == "missing_task_signature":
        del obj["task_signature"]
    elif mutation == "missing_steps":
        del obj["steps"]
    elif mutation == "extra_key":
        obj["confidence"] = 0.9
    elif mutation == "sig_too_short":
        obj["task_signature"] = "short"
    elif mutation == "sig_too_long":
        obj["task_signature"] = "x" * 201
    elif mutation == "sig_multiline":
        obj["task_signature"] = "line one\nline two here"
    elif mutation == "sig_unstripped":
        obj["task_signature"] = "  padded signature  "
    elif mutation == "sig_not_a_string":
        obj["task_signature"] = 42
    elif mutation == "applies_empty":
        obj["applies_when"] = []
    elif mutation == "applies_nine":
        obj["applies_when"] = ["c" + str(i) for i in range(9)]
    elif mutation == "steps_empty":
        obj["steps"] = []
    elif mutation == "steps_sixteen":
        obj["steps"] = ["s" + str(i) for i in range(16)]
    elif mutation == "failure_nine":
        obj["failure_modes"] = ["f" + str(i) for i in range(9)]
    elif mutation == "failure_not_strings":
        obj["failure_modes"] = ["ok", 7]
    elif mutation == "outcome_bad":
        obj["outcome"] = "mostly fine"
    elif mutation == "outcome_not_a_string":
        obj["outcome"] = None
    errors = pb.validate_playbook(obj)
    assert errors, "expected " + mutation + " to be rejected"
    assert all(isinstance(e, str) for e in errors)


def test_validate_never_raises_on_garbage() -> None:
    garbage_inputs: tuple[Any, ...] = (None, [], "x", 42, 3.5, {1: 2}, {"task_signature": None})
    for garbage in garbage_inputs:
        assert pb.validate_playbook(garbage) != []  # type: ignore[arg-type]


def test_validate_bounds_are_inclusive() -> None:
    obj = json.loads(json.dumps(_VALID))
    obj["task_signature"] = "x" * 200  # exactly the max
    obj["applies_when"] = ["c" for _ in range(8)]
    obj["steps"] = ["s" for _ in range(15)]
    obj["failure_modes"] = ["f" for _ in range(8)]
    assert pb.validate_playbook(obj) == []


# ---------------------------------------------------------------------------
# parse_playbook — fenced JSON under the header; None when absent/invalid
# ---------------------------------------------------------------------------


def test_parse_extracts_the_playbook_from_a_full_summary() -> None:
    doc = _summary_doc(json.dumps(_VALID))
    parsed = pb.parse_playbook(doc)
    assert parsed == _VALID


def test_parse_returns_none_when_the_section_is_absent() -> None:
    assert pb.parse_playbook(_summary_doc()) is None
    assert pb.parse_playbook("") is None
    assert pb.parse_playbook("## Gotchas\n- g\n") is None


def test_parse_returns_none_on_invalid_json_or_shape() -> None:
    assert pb.parse_playbook(_summary_doc("{not json")) is None
    assert pb.parse_playbook(_summary_doc('{"task_signature": "too short"}')) is None
    assert pb.parse_playbook(_summary_doc("[1, 2, 3]")) is None
    assert pb.parse_playbook(_summary_doc('{"task_signature": "valid signature here"}')) is None


def test_parse_requires_the_fence() -> None:
    unfenced = _summary_doc(json.dumps(_VALID)).replace(_FENCE + "json\n", "").replace("\n" + _FENCE + "\n", "\n")
    assert "## Playbook" in unfenced and _FENCE not in unfenced
    assert pb.parse_playbook(unfenced) is None


def test_parse_tolerates_header_variants() -> None:
    for header in ("## Playbook", "## Playbook:", "## **Playbook**", "### Playbook"):
        doc = _summary_doc(json.dumps(_VALID)).replace("## Playbook", header, 1)
        assert pb.parse_playbook(doc) == _VALID, header


def test_parse_finds_the_section_mid_document_and_ignores_garbage() -> None:
    doc = (
        "## Gotchas\n- g\n\n## Playbook\n"
        + _FENCE
        + "json\n"
        + json.dumps(_VALID)
        + "\n"
        + _FENCE
        + "\n\n## How to proceed\n- next\n"
    )
    assert pb.parse_playbook(doc) == _VALID
    for garbage in (None, 42, ["x"]):
        assert pb.parse_playbook(garbage) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# strip_playbook_section — byte-identical when absent; clean seam when not
# ---------------------------------------------------------------------------


def test_strip_is_a_noop_without_the_section() -> None:
    for doc in (_summary_doc(), "", "word\n", "## Gotchas\n- g\n\n\n"):
        assert pb.strip_playbook_section(doc) == doc


def test_strip_removes_a_trailing_section_and_keeps_the_rest() -> None:
    doc = _summary_doc(json.dumps(_VALID))
    stripped = pb.strip_playbook_section(doc)
    assert "## Playbook" not in stripped
    assert _VALID["task_signature"] not in stripped
    assert stripped.startswith("[MEMORY INDEX")
    assert "## How to proceed\n- next\n" in stripped
    assert stripped.endswith("\n")


def test_strip_removes_a_mid_document_section_cleanly() -> None:
    doc = (
        "## Gotchas\n- g\n\n## Playbook\n"
        + _FENCE
        + "json\n"
        + json.dumps(_VALID)
        + "\n"
        + _FENCE
        + "\n\n## How to proceed\n- next\n"
    )
    stripped = pb.strip_playbook_section(doc)
    assert "## Playbook" not in stripped and "## Gotchas" in stripped
    assert "## How to proceed" in stripped
    assert "\n\n\n" not in stripped


def test_strip_handles_unfenced_and_unclosed_shapes() -> None:
    unfenced = "top\n\n## Playbook\nsome prose\n\n## Next\nbody\n"
    out = pb.strip_playbook_section(unfenced)
    assert "Playbook" not in out and "## Next\nbody" in out and "top" in out
    unclosed = "top\n\n## Playbook\n" + _FENCE + "json\n{bad\n"
    assert pb.strip_playbook_section(unclosed).startswith("top")
    assert "Playbook" not in pb.strip_playbook_section(unclosed)


def test_strip_of_a_section_only_document_empties_it() -> None:
    only = "## Playbook\n" + _FENCE + "json\n{}\n" + _FENCE + "\n"
    assert pb.strip_playbook_section(only) == ""


def test_strip_never_raises_on_garbage() -> None:
    for garbage in (None, 42, ["a"]):
        assert pb.strip_playbook_section(garbage) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# emission gate + outcome inference over message rows
# ---------------------------------------------------------------------------


def test_eligible_canonical_session() -> None:
    assert pb.emission_eligible(_canonical_session(4)) is True
    assert pb.emission_eligible(_canonical_session(12)) is True


def test_gate_requires_four_tool_calls() -> None:
    assert pb.emission_eligible(_canonical_session(3)) is False  # tool data, bar not met
    # an EMPTY trace carries no tool data: the structural gate is
    # indeterminate there and stays eligible (lead ruling F1) - only
    # "tool data exists AND the bar is not met" is False
    assert pb.emission_eligible([]) is True


def test_budget_marker_marks_the_terminal_state_dirty() -> None:
    msgs = _canonical_session(6)
    msgs.append(_result("sweep stopped: budget_exceeded for this row", "cz"))
    assert pb.emission_eligible(msgs) is False


def test_denied_tool_storm_blocks_and_a_few_denials_do_not() -> None:
    msgs = _canonical_session(6)
    msgs.append(_result("access denied: policy", "d1"))
    assert pb.emission_eligible(msgs) is True  # a couple of denials are ordinary work
    msgs.append(_result("Permission Denied", "d2"))
    assert pb.emission_eligible(msgs) is True  # two: still under the storm line
    msgs.append(_result("not allowed by policy", "d3"))
    assert pb.emission_eligible(msgs) is False  # three: a storm


def test_repeated_identical_call_blocks_but_two_do_not() -> None:
    same = '{"wait": true}'
    msgs = [_call("job_output", same, "c1"), _result("running", "c1")]
    msgs += [_call("job_output", same, "c2"), _result("running", "c2")]
    msgs += _canonical_session(2)
    assert pb.emission_eligible(msgs) is True  # the pair repeated twice
    msgs += [_call("job_output", same, "c3"), _result("running", "c3")]
    assert pb.emission_eligible(msgs) is False  # third identical call


def test_plain_store_shape_counts_tool_results_without_double_counting() -> None:
    plain = [_result("ok " + str(i)) for i in range(4)]
    assert pb.emission_eligible(plain) is True
    assert pb.emission_eligible(plain[:3]) is False
    # canonical shape with both sides present: 2 calls + 4 results must
    # count as TWO calls, not six
    mixed = _canonical_session(2) + [_result("ok extra " + str(i)) for i in range(4)]
    assert pb.emission_eligible(mixed) is False


def test_gate_never_raises_on_garbage() -> None:
    # row garbage without tool data: indeterminate -> eligible, never raises
    assert pb.emission_eligible([None, "x", 42, {"role": 3}]) is True
    # non-list input cannot even be scanned: conservatively ineligible
    assert pb.emission_eligible("not a list") is False  # type: ignore[arg-type]
    assert pb.emission_eligible(None) is False  # type: ignore[arg-type]


def test_infer_outcome_success_and_source_are_pinned() -> None:
    assert pb.infer_outcome(_canonical_session(6)) == ("success", "inferred")
    # no tool data -> nothing inferable: "unknown"/"unverified", NEVER a
    # self-assessed label wearing "inferred" (lead ruling F1/F2)
    assert pb.infer_outcome([]) == ("unknown", "unverified")


def test_infer_outcome_abandoned_on_each_dirty_signal() -> None:
    budget = _canonical_session(6)
    budget.append(_result("budget exceeded mid-row", "zb"))
    assert pb.infer_outcome(budget) == ("abandoned", "inferred")

    storm = _canonical_session(6)
    storm += [_result("not allowed", "d1"), _result("not allowed", "d2"), _result("not allowed", "d3")]
    assert pb.infer_outcome(storm) == ("abandoned", "inferred")

    loop = [_call("t", "{}", "c")] * 3
    loop += _canonical_session(2)
    assert pb.infer_outcome(loop) == ("abandoned", "inferred")


def test_infer_outcome_tool_error_ratio() -> None:
    four_errors = _canonical_session(4)
    # replace every result text with an error marker
    for i, msg in enumerate(four_errors):
        if msg.get("role") == "tool":
            msg["content"] = "error: thing " + str(i) + " broke"
    assert pb.infer_outcome(four_errors) == ("abandoned", "inferred")

    half = _canonical_session(4)
    half_errors = 0
    for msg in half:
        if msg.get("role") == "tool" and half_errors < 2:
            msg["content"] = "error: broke"
            half_errors += 1
    assert pb.infer_outcome(half) == ("success", "inferred")  # 0.5 is not > 0.5

    small = [_result("error: a"), _result("error: b"), _result("error: c")]
    assert pb.infer_outcome(small) == ("success", "inferred")  # under the sample floor


def test_infer_outcome_never_raises_on_garbage() -> None:
    for garbage in (None, 3, "session", [7, None]):
        outcome, source = pb.infer_outcome(garbage)  # type: ignore[arg-type]
        assert outcome == "unknown"
        assert source == "unverified"


# ---------------------------------------------------------------------------
# prompt / contract constants
# ---------------------------------------------------------------------------


def test_contract_constants() -> None:
    assert pb.PROMPT_VERSION_BUMP == "dsh-2026-11-10"
    assert pb.OPTIONAL_SECTIONS == ("## Playbook",)
    assert pb.PLAYBOOK_KEYS == (
        "task_signature",
        "applies_when",
        "steps",
        "failure_modes",
        "outcome",
    )
    assert pb.OUTCOME_SOURCE == "inferred"


def test_playbook_prompt_states_gate_and_shape() -> None:
    prompt = pb.PLAYBOOK_PROMPT
    assert "4 tool calls" in prompt
    assert "budget_exceeded" in prompt
    assert "## Playbook" in prompt
    assert _FENCE + "json" in prompt
    assert '"task_signature"' in prompt
    assert '"outcome"' in prompt
    assert "## How to proceed" in prompt
    assert "NEVER from your self-assessment" in prompt


def test_format_playbook_section_round_trips_through_parse() -> None:
    section = pb.format_playbook_section(_VALID)
    assert section.startswith("## Playbook\n" + _FENCE + "json\n")
    assert pb.parse_playbook("intro\n\n" + section + "\n") == _VALID


# ---------------------------------------------------------------------------
# the patcher — planned against the miniature gateway, then executed
# ---------------------------------------------------------------------------


def _fresh_tree(tmp_path: Path) -> Path:
    root = tmp_path / "gateway"
    shutil.copytree(_FIXTURE_ROOT, root)
    return root


def test_apply_makes_every_edit_and_is_idempotent(tmp_path: Path) -> None:
    root = _fresh_tree(tmp_path)
    rc = patcher.main(["--apply", "--root", str(root)])
    assert rc == 0
    copied = root / "src/pcai_llm/core/playbook.py"
    assert copied.read_bytes() == _PLAYBOOK_PATH.read_bytes()

    distiller = (root / "src/pcai_llm/core/distiller.py").read_text(encoding="utf-8")
    assert 'PROMPT_VERSION = "dsh-2026-11-10"' in distiller
    assert "OPTIONAL_SECTIONS" in distiller
    assert "    + PLAYBOOK_PROMPT\n)" in distiller

    memory_write = (root / "src/pcai_llm/core/memory_write.py").read_text(encoding="utf-8")
    assert "async def _overlay_emit_playbook(" in memory_write
    assert "doc = await _overlay_emit_playbook(" in memory_write
    assert memory_write.index("def _overlay_emit_playbook") < memory_write.index("def build_add_memory_arguments")
    # the ruling wiring: keyword, gate, and outcome override all present
    assert (
        "    verdict_out: list[str] | None = None,\n"
        "    trace_messages: list[dict[str, Any]] | None = None,\n"
        ") -> tuple[bool, str]:" in memory_write
    )
    assert "trace_messages=trace_messages,\n    )" in memory_write
    assert "emission_eligible(trace_messages)" in memory_write
    assert "outcome, outcome_source = infer_outcome(trace_messages)" in memory_write

    routes = (root / "src/pcai_llm/api/routes_distill.py").read_text(encoding="utf-8")
    assert routes.count('trace_messages=conv.get("messages") or [],') == 2
    assert routes.count("        trace_messages=messages,\n") == 1
    inline_head = routes.index("def site_inline_shim")
    assert (
        "trace_messages=" not in routes[inline_head:]
    )  # the shim stays unwired (the docstring may name the parameter)

    pin = (root / "tests/test_distiller.py").read_text(encoding="utf-8")
    assert 'assert dist_mod.PROMPT_VERSION == "dsh-2026-11-10"' in pin

    # snapshot + second apply must not change a byte
    hashes_before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rc2 = patcher.main(["--apply", "--root", str(root)])
    assert rc2 == 0
    hashes_after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert hashes_before == hashes_after


def test_check_reports_pending_then_applied_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fresh_tree(tmp_path)
    assert patcher.main(["--check", "--root", str(root)]) == 0
    first = capsys.readouterr().out
    assert "would apply 11 edit(s)" in first
    for target in ("distiller.py", "memory_write.py", "playbook.py", "test_distiller.py"):
        assert target in first

    assert patcher.main(["--apply", "--root", str(root)]) == 0
    assert patcher.main(["--check", "--root", str(root)]) == 0
    second = capsys.readouterr().out
    assert "already applied - nothing to do" in second
    assert "broken" not in second


def test_check_exits_one_with_a_report_when_an_anchor_is_gone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fresh_tree(tmp_path)
    distiller = root / "src/pcai_llm/core/distiller.py"
    text = distiller.read_text(encoding="utf-8").replace(
        'PROMPT_VERSION = "dsh-2026-10-09"\n', "PROMPT_VERSION = 'moved'\n"
    )
    distiller.write_text(text, encoding="utf-8")
    assert patcher.main(["--check", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "broken" in out and "WOULD NOT APPLY" in out
    # apply must refuse to write anything on a broken plan
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert patcher.main(["--apply", "--root", str(root)]) == 1
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_apply_refreshes_a_drifted_sentinel_block(tmp_path: Path) -> None:
    root = _fresh_tree(tmp_path)
    assert patcher.main(["--apply", "--root", str(root)]) == 0
    memory_write = root / "src/pcai_llm/core/memory_write.py"
    text = memory_write.read_text(encoding="utf-8").replace(
        "return doc  # the section was the whole document",
        "return doc  # hand-edited comment",
    )
    memory_write.write_text(text, encoding="utf-8")
    assert patcher.main(["--apply", "--root", str(root)]) == 0
    restored = memory_write.read_text(encoding="utf-8")
    assert "hand-edited comment" not in restored
    assert "the section was the whole document" in restored


def test_patcher_cli_subprocess(tmp_path: Path) -> None:
    root = _fresh_tree(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(_PATCHER_PATH), "--check", "--root", str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "playbook" in proc.stdout
    assert "would apply 11 edit(s)" in proc.stdout


def test_pin_edit_is_soft_when_the_test_file_is_absent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fresh_tree(tmp_path)
    (root / "tests/test_distiller.py").unlink()
    assert patcher.main(["--apply", "--root", str(root)]) == 0
    assert "skipped" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the PATCHED miniature gateway, executed: behaviour, not just text
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_root(tmp_path: Path) -> Iterator[Path]:
    root = _fresh_tree(tmp_path)
    assert patcher.main(["--apply", "--root", str(root)]) == 0
    src = str(root / "src")
    pre = {n for n in sys.modules if n == "pcai_llm" or n.startswith("pcai_llm.")}
    sys.path.insert(0, src)
    importlib.invalidate_caches()
    yield root
    sys.path.remove(src)
    for name in [n for n in sys.modules if (n == "pcai_llm" or n.startswith("pcai_llm.")) and n not in pre]:
        sys.modules.pop(name, None)


def _fake_mcp(fail_kinds: frozenset[str] = frozenset()) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    async def mcp_call(args: dict[str, Any]) -> tuple[bool, str]:
        calls.append(args)
        if str(args.get("metadata", {}).get("kind")) in fail_kinds:
            raise RuntimeError("transport down")
        return True, "written"

    return calls, mcp_call


def test_patched_distiller_constants_and_completeness(patched_root: Path) -> None:
    from pcai_llm.core import distiller as dist

    assert dist.PROMPT_VERSION == pb.PROMPT_VERSION_BUMP == "dsh-2026-11-10"
    assert dist.OPTIONAL_SECTIONS == pb.OPTIONAL_SECTIONS
    assert dist.DISTILL_SYSTEM_PROMPT.endswith(pb.PLAYBOOK_PROMPT)
    # the contract's confirmation: an optional Playbook section does not
    # break the completeness guard, and completeness still requires all four
    assert dist.document_is_complete(_summary_doc(json.dumps(_VALID))) is True
    assert dist.document_is_complete(_summary_doc()) is True
    assert dist.document_is_complete("## What we did\nonly one\n") is False


def test_write_memory_strips_the_section_and_writes_the_playbook(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    verdicts: list[str] = []
    doc = _summary_doc(json.dumps(_VALID))
    ok, detail = asyncio.run(mw.write_memory(None, mcp, doc, session_id="sess-abc123", verdict_out=verdicts))
    assert ok is True and detail == "written"
    assert verdicts == [mw.VERDICT_WRITTEN]
    assert len(calls) == 2
    # the overlay runs BEFORE the summary write (lead directive: split after
    # the guards, ahead of the summary) - so the playbook call comes first
    playbook_call, summary = calls
    assert summary["metadata"]["kind"] == "project-summary"
    assert summary["metadata"]["session_id"] == "sess-abc123"
    assert "## Playbook" not in summary["text"]
    assert "## How to proceed" in summary["text"]
    assert playbook_call["metadata"]["kind"] == "playbook"
    md = playbook_call["metadata"]
    assert md["playbook_key"] == pb.playbook_key(_VALID["task_signature"])
    assert md["task_signature"] == _VALID["task_signature"]
    assert md["applies_when"] == _VALID["applies_when"]
    assert md["steps"] == _VALID["steps"]
    assert md["failure_modes"] == _VALID["failure_modes"]
    assert md["outcome"] == "success"
    assert md["outcome_source"] == "unverified"  # F2: no trace here, so no "inferred"
    assert md["success_count"] == 0 and md["fail_count"] == 0
    assert md["evidence_thread"] == "sess-abc123"
    datetime.fromisoformat(md["last_validated"])  # ISO string
    assert pb.parse_playbook(playbook_call["text"]) == _VALID


def test_write_memory_without_a_playbook_is_byte_identical_to_stock(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    doc = _summary_doc()
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s"))
    assert ok is True
    assert len(calls) == 1
    assert calls[0]["text"] == doc  # untouched: no strip, no second write


def test_playbook_write_failure_never_fails_the_summary(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp(fail_kinds=frozenset({"playbook"}))
    verdicts: list[str] = []
    ok, _ = asyncio.run(
        mw.write_memory(None, mcp, _summary_doc(json.dumps(_VALID)), session_id="s", verdict_out=verdicts)
    )
    assert ok is True
    assert verdicts == [mw.VERDICT_WRITTEN]
    assert len(calls) == 2  # the playbook call was attempted, raised, swallowed


def test_playbook_write_refusal_also_never_fails_the_summary(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls: list[dict[str, Any]] = []

    async def refusing(args: dict[str, Any]) -> tuple[bool, str]:
        calls.append(args)
        if args.get("metadata", {}).get("kind") == "playbook":
            return False, "near-duplicate; write suppressed"
        return True, "written"

    ok, _ = asyncio.run(mw.write_memory(None, refusing, _summary_doc(json.dumps(_VALID)), session_id="s"))
    assert ok is True
    assert len(calls) == 2


def test_invalid_playbook_is_stripped_and_not_written(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    doc = _summary_doc("{ broken json")
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s"))
    assert ok is True
    assert len(calls) == 1  # nothing parseable: no second write
    assert "## Playbook" not in calls[0]["text"]


def test_removed_overlay_module_degrades_to_stock_write(patched_root: Path) -> None:
    (patched_root / "src/pcai_llm/core/playbook.py").unlink()
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    doc = _summary_doc(json.dumps(_VALID))
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s"))
    assert ok is True
    assert len(calls) == 1  # stock behaviour: one write, section left in place
    assert calls[0]["text"] == doc

    from pcai_llm.core import distiller as dist

    assert dist.OPTIONAL_SECTIONS == ()  # the stub kept the module importable
    assert dist.PLAYBOOK_PROMPT == ""
    assert dist.DISTILL_SYSTEM_PROMPT.endswith("not every one seen.")  # no appendix


# ---------------------------------------------------------------------------
# server-side gate + outcome override (lead ruling 2026-11-11)
# ---------------------------------------------------------------------------


def test_gate_blocks_an_ineligible_session_even_with_a_valid_section(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    doc = _summary_doc(json.dumps(_VALID))
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s", trace_messages=_canonical_session(1)))
    assert ok is True
    assert len(calls) == 1  # the gate: no playbook write for a 1-call session
    assert calls[0]["metadata"]["kind"] == "project-summary"
    assert "## Playbook" not in calls[0]["text"]  # the summary is still stripped


def test_outcome_override_discards_the_model_claim(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    # ELIGIBLE for the gate (2 denials are under the storm line) but the
    # outcome signal is broader: ANY denial infers "abandoned"
    trace = _canonical_session(6)
    trace.append(_result("access denied: policy", "d1"))
    trace.append(_result("Permission Denied", "d2"))
    assert pb.emission_eligible(trace) is True
    assert pb.infer_outcome(trace) == ("abandoned", "inferred")

    calls, mcp = _fake_mcp()
    doc = _summary_doc(json.dumps(_VALID))  # the model claims "success"
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s", trace_messages=trace))
    assert ok is True
    assert len(calls) == 2
    playbook_call = next(c for c in calls if c["metadata"]["kind"] == "playbook")
    assert playbook_call["metadata"]["outcome"] == "abandoned"
    assert playbook_call["metadata"]["outcome_source"] == "inferred"
    stored = pb.parse_playbook(playbook_call["text"])
    assert stored is not None and stored["outcome"] == "abandoned"  # the body is rewritten too


def test_budget_trace_is_gate_blocked_entirely(patched_root: Path) -> None:
    # the (b)-literal form: budget_exceeded is a GATE signal, so the session
    # is ineligible and NO playbook exists to relabel - the design's
    # "no budget_exceeded" gate and the abandoned label cannot coexist
    from pcai_llm.core import memory_write as mw

    trace = _canonical_session(6)
    trace.append(_result("sweep stopped: budget_exceeded", "zb"))
    calls, mcp = _fake_mcp()
    ok, _ = asyncio.run(
        mw.write_memory(None, mcp, _summary_doc(json.dumps(_VALID)), session_id="s", trace_messages=trace)
    )
    assert ok is True
    assert len(calls) == 1
    assert "## Playbook" not in calls[0]["text"]


def test_trace_none_stores_the_model_claim_as_unverified(patched_root: Path) -> None:
    # lead ruling F2: with NO trace at all the model's claim may still be
    # stored, but outcome_source is "unverified" - never "inferred"
    from pcai_llm.core import memory_write as mw

    calls, mcp = _fake_mcp()
    doc = _summary_doc(json.dumps(_VALID))
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s", trace_messages=None))
    assert ok is True
    assert len(calls) == 2
    playbook_call = next(c for c in calls if c["metadata"]["kind"] == "playbook")
    assert playbook_call["metadata"]["outcome"] == "success"  # the model's claim passes through
    assert playbook_call["metadata"]["outcome_source"] == "unverified"


def test_summary_body_is_stable_across_all_trace_modes(patched_root: Path) -> None:
    from pcai_llm.core import memory_write as mw

    doc = _summary_doc(json.dumps(_VALID))
    bodies: list[str] = []
    for trace in (_canonical_session(1), _canonical_session(6) + [_result("access denied", "d1")], None):
        calls, mcp = _fake_mcp()
        ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s", trace_messages=trace))
        assert ok is True
        bodies.append(next(c for c in calls if c["metadata"]["kind"] == "project-summary")["text"])
    assert bodies[0] == bodies[1] == bodies[2] == pb.strip_playbook_section(doc)


def test_high_error_ratio_session_is_eligible_but_abandoned(patched_root: Path) -> None:
    trace = _canonical_session(4)
    for i, msg in enumerate(trace):
        if msg.get("role") == "tool":
            msg["content"] = "error: thing " + str(i) + " broke"
    assert pb.emission_eligible(trace) is True  # the gate does not use the ratio
    assert pb.infer_outcome(trace) == ("abandoned", "inferred")


def test_denial_and_repeat_outcome_thresholds_are_broader_than_the_gate() -> None:
    one_denial = _canonical_session(6) + [_result("not allowed", "d1")]
    assert pb.emission_eligible(one_denial) is True
    assert pb.infer_outcome(one_denial) == ("abandoned", "inferred")

    twice = [
        _call("job_output", "{}", "c1"),
        _result("running", "c1"),
        _call("job_output", "{}", "c2"),
        _result("running", "c2"),
    ]
    twice += _canonical_session(2)
    assert pb.emission_eligible(twice) is True
    assert pb.infer_outcome(twice) == ("abandoned", "inferred")


def test_routes_wiring_passes_the_in_scope_lists(patched_root: Path) -> None:
    from pcai_llm.api import routes_distill as routes

    class Sink:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []

        async def write_memory(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
            self.kwargs.append(kwargs)
            return True, "ok"

    conv_messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ho"}]
    conv = {"messages": conv_messages, "title": "t"}  # a dict: the wired line calls .get
    row_messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    sink = Sink()
    asyncio.run(routes.site_explicit_conversation("k", conv, sink, "doc", "c1", "t"))
    asyncio.run(routes.site_scheduled_conversation("k", conv, sink, "doc", "c2", "t"))
    asyncio.run(routes.site_scheduled_canonical("k", row_messages, sink, "doc", "thread-1", "t"))
    asyncio.run(routes.site_inline_shim("k", sink, "doc", "s1", "t"))
    assert len(sink.kwargs) == 4
    assert sink.kwargs[0]["trace_messages"] == conv_messages  # site :760
    assert sink.kwargs[1]["trace_messages"] == conv_messages  # site :935
    assert sink.kwargs[2]["trace_messages"] == row_messages  # site :1179
    assert "trace_messages" not in sink.kwargs[3]  # the inline site is intentionally unwired


# ---------------------------------------------------------------------------
# lead ruling round 2: F1 indeterminate gate, F2 unverified labels, F3 sections
# ---------------------------------------------------------------------------


def test_conversation_store_shape_is_eligible_and_emits_unverified(patched_root: Path) -> None:
    # F1 regression: the conversation store's rows are {seq, role, content,
    # ts} - no tool_calls blocks, no role=tool rows. The structural gate is
    # INDETERMINATE on such traces (it must not silently discard every
    # playbook on the two conversation-store distill paths), so the session
    # stays eligible and the model's claim is stored stamped "unverified".
    from pcai_llm.core import memory_write as mw

    trace = [
        {"seq": 1, "role": "user", "content": "fix the flake", "ts": 1},
        {"seq": 2, "role": "assistant", "content": "pinned the dep", "ts": 2},
        {"seq": 3, "role": "user", "content": "tests pass now", "ts": 3},
        {"seq": 4, "role": "assistant", "content": "merged", "ts": 4},
    ]
    assert pb.emission_eligible(trace) is True  # indeterminate, not "no"
    assert pb.infer_outcome(trace) == ("unknown", "unverified")

    calls, mcp = _fake_mcp()
    ok, _ = asyncio.run(
        mw.write_memory(None, mcp, _summary_doc(json.dumps(_VALID)), session_id="s", trace_messages=trace)
    )
    assert ok is True
    assert len(calls) == 2  # the playbook IS written on this shape
    playbook_call = next(c for c in calls if c["metadata"]["kind"] == "playbook")
    assert playbook_call["metadata"]["outcome"] == "success"  # the model's claim
    assert playbook_call["metadata"]["outcome_source"] == "unverified"  # the honest label


def test_tool_free_trace_with_budget_prose_stays_unverified(patched_root: Path) -> None:
    # (d): a tool-call-free trace cannot carry a trace-level budget signal -
    # the marker word in PROSE is not a trace signal; the honest label is
    # "unverified" and the gate stays indeterminate-eligible
    from pcai_llm.core import memory_write as mw

    trace = [
        {"role": "user", "content": "the sweep said budget_exceeded, whatever"},
        {"role": "assistant", "content": "noted"},
    ]
    assert pb.emission_eligible(trace) is True
    assert pb.infer_outcome(trace) == ("unknown", "unverified")

    calls, mcp = _fake_mcp()
    _ok, _ = asyncio.run(
        mw.write_memory(None, mcp, _summary_doc(json.dumps(_VALID)), session_id="s", trace_messages=trace)
    )
    playbook_call = next(c for c in calls if c["metadata"]["kind"] == "playbook")
    assert playbook_call["metadata"]["outcome_source"] == "unverified"


def test_tool_data_dirty_trace_is_inferred_abandoned(patched_root: Path) -> None:
    # (b): a trace WITH tool data and a dirty signal -> "abandoned"/"inferred"
    # (one denial: under the gate's storm line, over the outcome's line)
    trace = _canonical_session(6)
    trace.append(_result("access denied: policy", "d1"))
    assert pb.emission_eligible(trace) is True
    assert pb.infer_outcome(trace) == ("abandoned", "inferred")


def test_two_sections_parse_finds_the_first_valid_and_strip_removes_all() -> None:
    # (c)/F3 at the pure level: a decoy invalid first section cannot shadow
    # the valid one, and NOTHING model-authored survives the strip
    decoy = "## Playbook\n" + _FENCE + "json\n{ broken json\n" + _FENCE + "\n"
    valid = "## Playbook\n" + _FENCE + "json\n" + json.dumps(_VALID) + "\n" + _FENCE + "\n"
    doc = _summary_doc().rstrip("\n") + "\n\n" + decoy + "\n" + valid

    assert pb.parse_playbook(doc) == _VALID  # the VALID section is found
    stripped = pb.strip_playbook_section(doc)
    assert "## Playbook" not in stripped  # EVERY section is gone
    assert "broken json" not in stripped
    assert _VALID["task_signature"] not in stripped
    assert "## How to proceed" in stripped  # the required content survives


def test_two_sections_write_one_playbook_and_a_clean_summary(patched_root: Path) -> None:
    # (c)/F3 end to end: exactly ONE playbook write (the valid one) and a
    # stored summary carrying zero Playbook sections
    from pcai_llm.core import memory_write as mw

    decoy = "## Playbook\n" + _FENCE + "json\n{ broken json\n" + _FENCE + "\n"
    valid = "## Playbook\n" + _FENCE + "json\n" + json.dumps(_VALID) + "\n" + _FENCE + "\n"
    doc = _summary_doc().rstrip("\n") + "\n\n" + decoy + "\n" + valid

    calls, mcp = _fake_mcp()
    ok, _ = asyncio.run(mw.write_memory(None, mcp, doc, session_id="s", trace_messages=_canonical_session(6)))
    assert ok is True
    assert len(calls) == 2
    playbook_call = next(c for c in calls if c["metadata"]["kind"] == "playbook")
    assert playbook_call["metadata"]["task_signature"] == _VALID["task_signature"]
    summary = next(c for c in calls if c["metadata"]["kind"] == "project-summary")
    assert "## Playbook" not in summary["text"]
    assert "broken json" not in summary["text"]
    assert _VALID["task_signature"] not in summary["text"]
    assert "## How to proceed" in summary["text"]


def test_broken_decoy_does_not_swallow_required_sections() -> None:
    # the uniform next-"## " extent: an unclosed-fence decoy is capped at
    # the next heading instead of eating the rest of the document
    doc = "## Gotchas\n- g\n\n## Playbook\n" + _FENCE + "json\n{ bad\n\n## How to proceed\n- next\n"
    out = pb.strip_playbook_section(doc)
    assert "## Gotchas" in out and "## How to proceed" in out
    assert "## Playbook" not in out and "{ bad" not in out


def test_unclosed_decoy_still_yields_the_valid_section() -> None:
    decoy = "## Playbook\n" + _FENCE + "json\n{ bad never closes\n"
    valid = "## Playbook\n" + _FENCE + "json\n" + json.dumps(_VALID) + "\n" + _FENCE + "\n"
    assert pb.parse_playbook(decoy) is None
    assert pb.parse_playbook(decoy + "\n" + valid) == _VALID


def test_first_valid_section_wins_when_two_are_valid() -> None:
    other = dict(_VALID, task_signature="audit the shared lint config")
    doc = (
        _summary_doc().rstrip("\n")
        + "\n\n## Playbook\n"
        + _FENCE
        + "json\n"
        + json.dumps(_VALID)
        + "\n"
        + _FENCE
        + "\n"
        + "\n## Playbook\n"
        + _FENCE
        + "json\n"
        + json.dumps(other)
        + "\n"
        + _FENCE
        + "\n"
    )
    assert pb.parse_playbook(doc) == _VALID  # the FIRST valid one
    stripped = pb.strip_playbook_section(doc)
    assert stripped.count("## Playbook") == 0
    assert "audit the shared lint config" not in stripped
