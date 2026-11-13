#!/usr/bin/env python3
"""Idempotent patcher: teach the PCAI-LLM gateway distiller playbook emission.

Feature A overlay (wave 2026-11-10; server-side gate + outcome wired per the
lead ruling of 2026-11-11). The gateway repo is mounted READ-ONLY from the
build workspace, so the overlay lives in integrations/pcai_llm_playbook/ and
is applied to a WRITABLE copy of the gateway tree (typically a /tmp copy;
the real repo when it is writable):

    python apply_playbook_patch.py --check --root /tmp/pcai-patch   # dry run
    python apply_playbook_patch.py --apply --root /tmp/pcai-patch

Edits, in order:
  1. copy playbook.py -> <root>/src/pcai_llm/core/playbook.py (verbatim)
  2. distiller.py: PROMPT_VERSION -> playbook.PROMPT_VERSION_BUMP
  3. distiller.py: OPTIONAL_SECTIONS + PLAYBOOK_PROMPT import block (sentinel)
  4. distiller.py: append PLAYBOOK_PROMPT to DISTILL_SYSTEM_PROMPT
  5. memory_write.py: _overlay_emit_playbook helper block (sentinel)
  6. memory_write.py: write_memory gains trace_messages keyword
  7. memory_write.py: call the helper after the guards, before the summary
  8. routes_distill.py: conversation route passes trace_messages=conv...
  9. routes_distill.py: scheduled conversations sweep passes the same list
 10. routes_distill.py: canonical sweep passes the row's messages
 11. routes_distill.py INLINE SHIM: NO EDIT (informational, always skipped):
     the /v1/distill payload carries only a caller-rendered transcript
     STRING - there are no message rows in scope, so trace_messages keeps
     its None default there and the pre-ruling behaviour applies.
 12. tests/test_distiller.py: the PROMPT_VERSION pin -> the bumped version
     (DISCLOSED DEVIATION: the gateway pins the old constant at
     tests/test_distiller.py:62; the pin UPDATE is the correct change for a
     real constant change - it rewrites the expected value, it never skips
     or xfails the assertion. Soft: a missing file/line is "skipped".)

Idempotency: edits on the same file are planned against ONE cumulative
buffer (each edit's new_text includes every earlier edit of that file), so
one --apply writes the final state once. Sentinel-marked blocks are
recognized and compared byte-for-byte (identical -> already applied,
different -> refreshed); line edits match the exact known-old text and are
no-ops once the new text is present. --check never writes and exits 1 only
when an edit is BROKEN (its anchor is gone or ambiguous) - that means the
gateway file drifted and a human must look, never that the patch should
force its way in.

document_is_complete() needs NO edit: it checks REQUIRED_SECTIONS only, so
a document that carries the optional "## Playbook" section already passes
completeness unchanged (verified against the real gateway module).

Lint contract: the generated code must pass the gateway's release gate
(`ruff check src/ tests/ scripts/`). Hence the sentinel blocks below carry no
redundant `noqa` (RUF100), use the `datetime.UTC` alias (UP017), and the import
block is separated from the sentinel comment by a blank line (I001) - that
blank line lives outside the span, so _ensure_import_separation() repairs it
on every pass rather than baking it into the block constant.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
PLAYBOOK_SOURCE_PATH = MODULE_DIR / "playbook.py"

DISTILLER_REL = "src/pcai_llm/core/distiller.py"
MEMORY_WRITE_REL = "src/pcai_llm/core/memory_write.py"
ROUTES_REL = "src/pcai_llm/api/routes_distill.py"
PIN_TEST_REL = "tests/test_distiller.py"
PLAYBOOK_TARGET_REL = "src/pcai_llm/core/playbook.py"

BEGIN_MARK = "# --- pcai-llm-playbook overlay BEGIN (idempotent patch; integrations/pcai_llm_playbook) ---"
END_MARK = "# --- pcai-llm-playbook overlay END ---"

# Edit 2 - the version bump is pinned to the ONE known transition. A gateway
# that has moved past dsh-2026-10-09 on its own reports "broken" here and a
# human reconciles; the overlay never overwrites a version it does not know.
PROMPT_VERSION_OLD_LINE = 'PROMPT_VERSION = "dsh-2026-10-09"' + "\n"
PROMPT_VERSION_COMMENT = "  # playbook overlay bump"

# Edit 3 - the distiller import block, inserted right after the typing import.
DISTILLER_IMPORT_ANCHOR_RE = r"(?m)^from typing import Any, cast\b.*$"
DISTILLER_IMPORT_BLOCK = (
    BEGIN_MARK
    + "\n"
    + "# Optional-section contract + the prompt addendum. The overlay module is\n"
    + "# copied next to this file by apply_playbook_patch.py; if it is ever\n"
    + "# removed, the stubs below degrade the distiller to its stock behaviour.\n"
    + "try:\n"
    + "    from .playbook import OPTIONAL_SECTIONS, PLAYBOOK_PROMPT\n"
    + "except ImportError:  # pragma: no cover - overlay module removed: stock behaviour\n"
    + "    OPTIONAL_SECTIONS = ()  # type: ignore[assignment]\n"
    + '    PLAYBOOK_PROMPT = ""\n'
    + END_MARK
)

# Edit 4 - the exact tail of the gateway DISTILL_SYSTEM_PROMPT construction.
PROMPT_TAIL_ANCHOR = (
    '    "- Preserve exact commands, paths, digests, error messages, and config values"\n'
    '    " verbatim - but pick the few that carry recall value, not every one seen."\n' + ")"
)
PROMPT_TAIL_NEW = (
    '    "- Preserve exact commands, paths, digests, error messages, and config values"\n'
    '    " verbatim - but pick the few that carry recall value, not every one seen."\n'
    + "\n"
    + "    + PLAYBOOK_PROMPT\n"
    + ")"
)
PROMPT_APPEND_MARK = "    + PLAYBOOK_PROMPT\n)"

# Edit 5 - the memory_write overlay helper, inserted before
# build_add_memory_arguments. Imports are lazy (inside the function) on
# purpose: no module-level import lands mid-file, and a deleted overlay
# module degrades to stock write behaviour instead of breaking the import.
MEMORY_WRITE_BLOCK = (
    BEGIN_MARK
    + "\n"
    + "async def _overlay_emit_playbook(\n"
    + "    mcp_call: McpCall,\n"
    + "    doc: str,\n"
    + "    *,\n"
    + "    session_id: str,\n"
    + "    source: str,\n"
    + "    key_id: uuid.UUID | None,\n"
    + "    trace_messages: list[dict[str, Any]] | None = None,\n"
    + ") -> str:\n"
    + '    """Split the optional Playbook section off a distilled summary.\n'
    + "\n"
    + '    Overlay behaviour (Feature A): the "## Playbook" section is stripped\n'
    + "    from the summary body, and when it parses as a valid playbook a\n"
    + '    SECOND, best-effort memory of kind "playbook" is written. Never\n'
    + "    raises and never changes the summary write's (ok, detail) result or\n"
    + "    its verdict: a playbook-write failure is logged and swallowed.\n"
    + "\n"
    + "    Server-side gate + outcome (lead rulings 2026-11-11): when\n"
    + "    trace_messages carries TOOL DATA, the session must pass\n"
    + "    emission_eligible() or the playbook is NOT written (the section is\n"
    + "    still stripped from the summary), and the outcome label is\n"
    + "    infer_outcome(trace_messages) - the model's own claimed outcome is\n"
    + "    DISCARDED, never merged, and the stored body is rewritten with the\n"
    + "    server-side label. A trace with no tool data, or trace_messages=None,\n"
    + "    gives the structural gate nothing to judge: the model's claimed\n"
    + '    outcome may be stored but is stamped outcome_source="unverified" -\n'
    + '    "inferred" is reserved for labels derived from trace signals.\n'
    + '    """\n'
    + "    from datetime import UTC, datetime\n"
    + "\n"
    + "    try:\n"
    + "        from .playbook import (\n"
    + "            OUTCOME_UNKNOWN,\n"
    + "            OUTCOME_UNVERIFIED,\n"
    + "            emission_eligible,\n"
    + "            format_playbook_section,\n"
    + "            infer_outcome,\n"
    + "            parse_playbook,\n"
    + "            playbook_key,\n"
    + "            strip_playbook_section,\n"
    + "        )\n"
    + "    except ImportError:  # pragma: no cover - overlay module removed\n"
    + "        return doc\n"
    + "    body = doc\n"
    + "    try:\n"
    + "        body = strip_playbook_section(doc)\n"
    + "        if not body.strip():\n"
    + "            return doc  # the section was the whole document: keep the original\n"
    + "        playbook = parse_playbook(doc)\n"
    + "        if playbook is None:\n"
    + "            return body\n"
    + "        if trace_messages is not None and not emission_eligible(trace_messages):\n"
    + "            log.debug(\n"
    + '                "playbook gate: session not eligible (key=%s, session=%s)"\n'
    + '                " - section stripped from the summary, no playbook written",\n'
    + "                key_id,\n"
    + "                session_id[:16],\n"
    + "            )\n"
    + "            return body\n"
    + "        if trace_messages is not None:\n"
    + "            # SERVER-SIDE OUTCOME (design note point 4): derived from\n"
    + "            # trace signals, never the model's self-assessment.\n"
    + "            outcome, outcome_source = infer_outcome(trace_messages)\n"
    + "            if outcome == OUTCOME_UNKNOWN:\n"
    + "                # the trace carried NO tool data: nothing was inferable -\n"
    + '                # store the model\'s claim stamped "unverified", never\n'
    + '                # "inferred" (lead ruling F1/F2)\n'
    + '                outcome = str(playbook.get("outcome") or "")\n'
    + "                outcome_source = OUTCOME_UNVERIFIED\n"
    + "        else:\n"
    + "            # no trace at all: the same honesty rule - the model's claim\n"
    + '            # is stored but stamped "unverified" (lead ruling F2)\n'
    + '            outcome = str(playbook.get("outcome") or "")\n'
    + "            outcome_source = OUTCOME_UNVERIFIED\n"
    + '        playbook["outcome"] = outcome\n'
    + '        signature = str(playbook.get("task_signature") or "")\n'
    + "        args = build_add_memory_arguments(\n"
    + "            format_playbook_section(playbook),\n"
    + "            session_id,\n"
    + "            source,\n"
    + '            "playbook",\n'
    + "        )\n"
    + '        args["metadata"].update(\n'
    + "            {\n"
    + '                "playbook_key": playbook_key(signature),\n'
    + '                "task_signature": signature,\n'
    + '                "applies_when": list(playbook.get("applies_when") or []),\n'
    + '                "steps": list(playbook.get("steps") or []),\n'
    + '                "failure_modes": list(playbook.get("failure_modes") or []),\n'
    + '                "outcome": outcome,\n'
    + '                "outcome_source": outcome_source,\n'
    + '                "last_validated": datetime.now(UTC).isoformat(timespec="seconds"),\n'
    + '                "success_count": 0,\n'
    + '                "fail_count": 0,\n'
    + '                "evidence_thread": session_id,\n'
    + "            }\n"
    + "        )\n"
    + "        try:\n"
    + "            ok, detail = await mcp_call(args)\n"
    + "            if not ok:\n"
    + "                log.warning(\n"
    + '                    "playbook memory write refused (key=%s, session=%s): %s",\n'
    + "                    key_id,\n"
    + "                    session_id[:16],\n"
    + "                    str(detail)[:120],\n"
    + "                )\n"
    + "        except Exception as exc:\n"
    + "            log.warning(\n"
    + '                "playbook memory write failed (key=%s, session=%s): %s",\n'
    + "                key_id,\n"
    + "                session_id[:16],\n"
    + "                type(exc).__name__,\n"
    + "            )\n"
    + "    except Exception:  # pragma: no cover - the overlay must never break a write\n"
    + '        log.warning("playbook emission fault (key=%s)", key_id, exc_info=True)\n'
    + "    return body\n"
    + END_MARK
)

# Edit 6 - write_memory gains the optional trace_messages keyword (backward
# compatible: keyword-only with a None default, so every existing caller
# and test keeps its exact behaviour).
SIGNATURE_ANCHOR = "    verdict_out: list[str] | None = None,\n) -> tuple[bool, str]:"
SIGNATURE_NEW = (
    "    verdict_out: list[str] | None = None,\n"
    "    trace_messages: list[dict[str, Any]] | None = None,\n"
    ") -> tuple[bool, str]:"
)
SIGNATURE_MARK = (
    "    verdict_out: list[str] | None = None,\n"
    "    trace_messages: list[dict[str, Any]] | None = None,\n"
    ") -> tuple[bool, str]:"
)

# Edit 7 - the call site: after the empty/floor guards, before the summary
# write (lead directive). The anchor is the summary mcp_call try block.
SUMMARY_CALL_ANCHOR = (
    "    try:\n        ok, detail = await mcp_call(build_add_memory_arguments(doc, session_id, source, kind))"
)
OVERLAY_CALL_NEW = (
    "\n"
    "    # --- pcai-llm-playbook overlay: split the optional Playbook section off ---\n"
    "    doc = await _overlay_emit_playbook(\n"
    "        mcp_call,\n"
    "        doc,\n"
    "        session_id=session_id,\n"
    "        source=source,\n"
    "        key_id=key_id,\n"
    "        trace_messages=trace_messages,\n"
    "    )\n"
    "\n"
    "    try:\n"
    "        ok, detail = await mcp_call(build_add_memory_arguments(doc, session_id, source, kind))"
)
OVERLAY_CALL_MARK = "pcai-llm-playbook overlay: split the optional Playbook section off"

# Edits 8-10 - the routes_distill call sites that HAVE a message list in
# scope. Anchors are the unique tail kwargs of each real call (verified
# against PCAI_LLM/src/pcai_llm/api/routes_distill.py); the inserted line
# uses the variable actually in scope at each site - no invented fetch.
ROUTES_SITE_EXPLICIT_ANCHOR = (
    "        fallback_session_id=str(cid),\n"
    "        mcp_search=await _mcp_search_for(key),\n"
    "        verdict_out=verdict,\n"
    "    )"
)
ROUTES_SITE_EXPLICIT_NEW = (
    "        fallback_session_id=str(cid),\n"
    "        mcp_search=await _mcp_search_for(key),\n"
    "        verdict_out=verdict,\n"
    '        trace_messages=conv.get("messages") or [],\n'
    "    )"
)
ROUTES_SITE_EXPLICIT_MARK = '        trace_messages=conv.get("messages") or [],\n'

ROUTES_SITE_SCHED_CONV_ANCHOR = "        fallback_session_id=str(cid),\n        verdict_out=verdict,\n    )"
ROUTES_SITE_SCHED_CONV_NEW = (
    "        fallback_session_id=str(cid),\n"
    "        verdict_out=verdict,\n"
    '        trace_messages=conv.get("messages") or [],\n'
    "    )"
)
# the scheduled-conv site is the only one whose tail goes str(cid) ->
# verdict_out -> trace_messages with NO mcp_search line between, so the
# mark is anchored to that context (a bare inserted-line mark would
# false-positive against edit 8's insertion of the same line)
ROUTES_SITE_SCHED_CONV_MARK = (
    "        fallback_session_id=str(cid),\n"
    "        verdict_out=verdict,\n"
    '        trace_messages=conv.get("messages") or [],\n'
    "    )"
)

ROUTES_SITE_CANONICAL_ANCHOR = (
    "        fallback_session_id=thread_id[:64],\n"
    "        mcp_search=await _mcp_search_for(key),\n"
    "        verdict_out=verdict,\n"
    "    )"
)
ROUTES_SITE_CANONICAL_NEW = (
    "        fallback_session_id=thread_id[:64],\n"
    "        mcp_search=await _mcp_search_for(key),\n"
    "        verdict_out=verdict,\n"
    "        trace_messages=messages,\n"
    "    )"
)
ROUTES_SITE_CANONICAL_MARK = "        trace_messages=messages,\n"

# Edit 12 - the gateway's own pinned-version test (soft edit).
PIN_OLD_LINE = '    assert dist_mod.PROMPT_VERSION == "dsh-2026-10-09"\n'
PIN_NEW_LINE = '    assert dist_mod.PROMPT_VERSION == "dsh-2026-11-10"  # playbook overlay bump\n'
PIN_APPLIED_MARK = '    assert dist_mod.PROMPT_VERSION == "dsh-2026-11-10"'

COMPLETENESS_NOTE = (
    "note: document_is_complete() needs no edit - it checks REQUIRED_SECTIONS only,"
    " so an optional '## Playbook' section already passes completeness"
)


@dataclass
class Edit:
    """One planned edit: a status plus, when it would change bytes, the text.

    new_text is the CUMULATIVE content of the target file after this edit
    and every earlier edit on the same file - writing the pending edits in
    order leaves the tree in exactly the planned state.
    """

    num: int
    name: str
    target: str
    status: str  # pending | applied | update | broken | skipped
    detail: str = ""
    new_text: str | None = None
    new_bytes: bytes | None = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _block_span_status(text: str, block: str) -> str:
    """absent | applied | update | broken for one sentinel-marked block."""
    begins = text.count(BEGIN_MARK)
    ends = text.count(END_MARK)
    if begins == 1 and ends == 1:
        b = text.index(BEGIN_MARK)
        e = text.index(END_MARK)
        return "applied" if text[b : e + len(END_MARK)] == block else "update"
    if begins or ends:
        return "broken"  # marks present but mangled: a human must look
    return "absent"


def _replace_block_span(text: str, block: str) -> str:
    b = text.index(BEGIN_MARK)
    e = text.index(END_MARK)
    return text[:b] + block + text[e + len(END_MARK) :]

def _ensure_import_separation(text: str) -> str:
    """Keep one blank line between the sorted imports and the sentinel.

    ruff's isort (I001) parses the sentinel's leading comment as part of the
    import section, so a comment glued to the last import reads as an
    un-sorted block. The blank line sits OUTSIDE the sentinel span, so it
    cannot ride along in DISTILLER_IMPORT_BLOCK: it is normalised here on
    every pass instead, which also repairs an older application.
    """
    b = text.index(BEGIN_MARK)
    if text[:b].endswith("\n\n"):
        return text
    return text[:b] + "\n" + text[b:]


def _unified(old: str, new: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="current",
        tofile="patched",
        n=1,
    )
    return "".join(diff)


def _edit_copy_module(root: Path) -> Edit:
    target = root / PLAYBOOK_TARGET_REL
    edit = Edit(1, "copy overlay module", PLAYBOOK_TARGET_REL, "pending")
    try:
        source_bytes = PLAYBOOK_SOURCE_PATH.read_bytes()
    except OSError as exc:
        edit.status = "broken"
        edit.detail = "playbook.py unreadable next to the patcher: " + type(exc).__name__
        return edit
    edit.new_bytes = source_bytes
    if not target.exists():
        edit.detail = "would copy (" + str(len(source_bytes)) + " bytes)"
        return edit
    if target.read_bytes() == source_bytes:
        edit.status = "applied"
        edit.detail = "module already present and byte-identical"
        return edit
    edit.status = "update"
    edit.detail = "module source changed since the copy - will overwrite"
    return edit


def _edit_version_bump(text: str | None, bump: str) -> tuple[Edit, str | None]:
    edit = Edit(2, "bump PROMPT_VERSION", DISTILLER_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    applied_line = 'PROMPT_VERSION = "' + bump + '"'
    for line in text.splitlines():
        if line == applied_line or line.startswith(applied_line + PROMPT_VERSION_COMMENT):
            edit.status = "applied"
            edit.detail = "already " + bump
            return edit, text
    if text.count(PROMPT_VERSION_OLD_LINE) != 1:
        edit.status = "broken"
        edit.detail = "expected exactly one '" + PROMPT_VERSION_OLD_LINE.strip() + "' line"
        return edit, None
    new_line = applied_line + PROMPT_VERSION_COMMENT + "\n"
    edit.new_text = text.replace(PROMPT_VERSION_OLD_LINE, new_line, 1)
    edit.detail = PROMPT_VERSION_OLD_LINE.strip()[17:] + " -> " + bump
    return edit, edit.new_text


def _edit_import_block(text: str | None) -> tuple[Edit, str | None]:
    edit = Edit(3, "add OPTIONAL_SECTIONS block", DISTILLER_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    status = _block_span_status(text, DISTILLER_IMPORT_BLOCK)
    if status == "absent":
        m = re.search(DISTILLER_IMPORT_ANCHOR_RE, text)
        if m is None:
            edit.status = "broken"
            edit.detail = "anchor 'from typing import Any, cast' not found"
            return edit, None
        edit.new_text = _ensure_import_separation(
            text[: m.end()] + "\n\n" + DISTILLER_IMPORT_BLOCK + text[m.end() :]
        )
        edit.detail = "would insert after the typing import"
        return edit, edit.new_text
    edit.status = status
    if status == "broken":
        edit.detail = "sentinel marks mangled (BEGIN/END missing or duplicated)"
        return edit, text
    candidate = text
    if status == "update":
        candidate = _replace_block_span(text, DISTILLER_IMPORT_BLOCK)
        edit.detail = "sentinel block drifted - will refresh"
    candidate = _ensure_import_separation(candidate)
    if candidate != text:
        edit.status = "update"
        edit.new_text = candidate
        if status == "applied":
            edit.detail = "block current; inserting the blank line ruff's isort requires"
        return edit, candidate
    edit.detail = "sentinel block already present and identical"
    return edit, text


def _edit_prompt_append(text: str | None) -> tuple[Edit, str | None]:
    edit = Edit(4, "append PLAYBOOK_PROMPT", DISTILLER_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    if PROMPT_APPEND_MARK in text:
        edit.status = "applied"
        edit.detail = "prompt already ends with + PLAYBOOK_PROMPT"
        return edit, text
    if text.count(PROMPT_TAIL_ANCHOR) != 1:
        edit.status = "broken"
        edit.detail = "DISTILL_SYSTEM_PROMPT tail anchor not found or ambiguous"
        return edit, None
    edit.new_text = text.replace(PROMPT_TAIL_ANCHOR, PROMPT_TAIL_NEW, 1)
    edit.detail = "would append the playbook addendum to the system prompt"
    return edit, edit.new_text


def _edit_memory_block(text: str | None) -> tuple[Edit, str | None]:
    edit = Edit(5, "add write-memory helper block", MEMORY_WRITE_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    status = _block_span_status(text, MEMORY_WRITE_BLOCK)
    if status == "absent":
        m = re.search(r"(?m)^def build_add_memory_arguments\(", text)
        if m is None:
            edit.status = "broken"
            edit.detail = "anchor 'def build_add_memory_arguments(' not found"
            return edit, None
        edit.new_text = text[: m.start()] + MEMORY_WRITE_BLOCK + "\n\n\n" + text[m.start() :]
        edit.detail = "would insert _overlay_emit_playbook before build_add_memory_arguments"
        return edit, edit.new_text
    edit.status = status
    if status == "update":
        edit.new_text = _replace_block_span(text, MEMORY_WRITE_BLOCK)
        edit.detail = "sentinel block drifted - will refresh"
        return edit, edit.new_text
    edit.detail = "sentinel block already present and identical"
    return edit, text


def _edit_signature(text: str | None) -> tuple[Edit, str | None]:
    edit = Edit(6, "add trace_messages keyword", MEMORY_WRITE_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    if SIGNATURE_MARK in text:
        edit.status = "applied"
        edit.detail = "keyword already present"
        return edit, text
    if text.count(SIGNATURE_ANCHOR) != 1:
        edit.status = "broken"
        edit.detail = "write_memory signature tail anchor not found or ambiguous"
        return edit, None
    edit.new_text = text.replace(SIGNATURE_ANCHOR, SIGNATURE_NEW, 1)
    edit.detail = "would add the optional keyword-only trace_messages parameter"
    return edit, edit.new_text


def _edit_call_site(text: str | None) -> tuple[Edit, str | None]:
    edit = Edit(7, "call helper before summary write", MEMORY_WRITE_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    if OVERLAY_CALL_MARK in text:
        edit.status = "applied"
        edit.detail = "call site already wired"
        return edit, text
    if text.count(SUMMARY_CALL_ANCHOR) != 1:
        edit.status = "broken"
        edit.detail = "summary mcp_call try block not found or ambiguous"
        return edit, None
    edit.new_text = text.replace(SUMMARY_CALL_ANCHOR, OVERLAY_CALL_NEW, 1)
    edit.detail = "would wire _overlay_emit_playbook ahead of the summary write"
    return edit, edit.new_text


def _edit_routes_site(
    num: int,
    name: str,
    text: str | None,
    anchor: str,
    new: str,
    mark: str,
    detail_pending: str,
) -> tuple[Edit, str | None]:
    edit = Edit(num, name, ROUTES_REL, "pending")
    if text is None:
        edit.status = "broken"
        edit.detail = "file not found under --root"
        return edit, None
    if mark in text:
        edit.status = "applied"
        edit.detail = "trace_messages already wired here"
        return edit, text
    if text.count(anchor) != 1:
        edit.status = "broken"
        edit.detail = "call-site tail anchor not found or ambiguous"
        return edit, None
    edit.new_text = text.replace(anchor, new, 1)
    edit.detail = detail_pending
    return edit, edit.new_text


def _edit_pin_test(text: str | None, bump: str) -> Edit:
    edit = Edit(12, "move version pin in gateway test", PIN_TEST_REL, "pending")
    if text is None:
        edit.status = "skipped"
        edit.detail = "tests/test_distiller.py absent - pin edit skipped"
        return edit
    if PIN_APPLIED_MARK in text:
        edit.status = "applied"
        edit.detail = "pin already at " + bump
        return edit
    if text.count(PIN_OLD_LINE) != 1:
        edit.status = "skipped"
        edit.detail = "pinned line not found (test file changed) - NOT touched"
        return edit
    edit.new_text = text.replace(PIN_OLD_LINE, PIN_NEW_LINE, 1)
    edit.detail = "would update the pinned version assertion (rewrites the expected value)"
    return edit


def plan(root: Path, bump: str) -> list[Edit]:
    """Plan every edit against cumulative per-file buffers, in order."""
    edits: list[Edit] = []
    buffers: dict[str, str | None] = {}

    def buffer(rel: str) -> str | None:
        if rel not in buffers:
            buffers[rel] = _read_text(root / rel)
        return buffers[rel]

    edits.append(_edit_copy_module(root))

    text = buffer(DISTILLER_REL)
    edit, text = _edit_version_bump(text, bump)
    if text is not None:
        buffers[DISTILLER_REL] = text
    edits.append(edit)
    edit, new = _edit_import_block(text)
    if new is not None:
        text = new
    edits.append(edit)
    edit, new = _edit_prompt_append(text)
    if new is not None:
        text = new
    edits.append(edit)

    mtext = buffer(MEMORY_WRITE_REL)
    edit, new = _edit_memory_block(mtext)
    if new is not None:
        mtext = new
    edits.append(edit)
    edit, new = _edit_signature(mtext)
    if new is not None:
        mtext = new
    edits.append(edit)
    edit, new = _edit_call_site(mtext)
    if new is not None:
        mtext = new
    edits.append(edit)

    rtext = buffer(ROUTES_REL)
    edit, new = _edit_routes_site(
        8,
        "route: explicit conversation trace",
        rtext,
        ROUTES_SITE_EXPLICIT_ANCHOR,
        ROUTES_SITE_EXPLICIT_NEW,
        ROUTES_SITE_EXPLICIT_MARK,
        'would pass trace_messages=conv.get("messages") or [] (site :760)',
    )
    if new is not None:
        rtext = new
    edits.append(edit)
    edit, new = _edit_routes_site(
        9,
        "route: scheduled conversation trace",
        rtext,
        ROUTES_SITE_SCHED_CONV_ANCHOR,
        ROUTES_SITE_SCHED_CONV_NEW,
        ROUTES_SITE_SCHED_CONV_MARK,
        'would pass trace_messages=conv.get("messages") or [] (site :935)',
    )
    if new is not None:
        rtext = new
    edits.append(edit)
    edit, new = _edit_routes_site(
        10,
        "route: canonical sweep trace",
        rtext,
        ROUTES_SITE_CANONICAL_ANCHOR,
        ROUTES_SITE_CANONICAL_NEW,
        ROUTES_SITE_CANONICAL_MARK,
        "would pass trace_messages=messages (site :1179)",
    )
    if new is not None:
        rtext = new
    edits.append(edit)
    edits.append(
        Edit(
            11,
            "route: inline shim - intentionally NOT wired",
            ROUTES_REL,
            "skipped",
            "/v1/distill has no message rows in scope (caller-rendered transcript"
            " string); trace_messages keeps its None default there, so the"
            " pre-ruling behaviour applies at that one site",
        )
    )

    edits.append(_edit_pin_test(buffer(PIN_TEST_REL), bump))
    return edits


def _load_playbook_module() -> Any:
    spec = importlib.util.spec_from_file_location("pcai_llm_playbook_overlay_source", PLAYBOOK_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot locate playbook.py next to apply_playbook_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print_edits(edits: list[Edit]) -> None:
    for e in edits:
        line = "  " + str(e.num) + ". " + e.name.ljust(42) + " " + e.target.ljust(36) + " " + e.status
        if e.detail:
            line += "  | " + e.detail
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the playbook-emission overlay to a writable copy of the PCAI-LLM gateway tree."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="dry run; exit 1 only when an edit is broken")
    mode.add_argument("--apply", action="store_true", help="write the edits (idempotent)")
    parser.add_argument("--root", type=Path, default=None, help="gateway tree root (default: cwd)")
    args = parser.parse_args(argv)

    root = (args.root or Path.cwd()).resolve()
    try:
        pb = _load_playbook_module()
    except Exception as exc:
        print("ERROR: cannot load playbook.py: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1
    bump = str(getattr(pb, "PROMPT_VERSION_BUMP", ""))
    if not bump:
        print("ERROR: playbook.py does not define PROMPT_VERSION_BUMP", file=sys.stderr)
        return 1

    edits = plan(root, bump)
    mode_label = "check" if args.check else "apply"
    print("pcai-llm-playbook overlay - root: " + str(root) + " (mode: " + mode_label + ", bump: " + bump + ")")
    _print_edits(edits)
    print(COMPLETENESS_NOTE)

    broken = [e for e in edits if e.status == "broken"]
    skipped = [e for e in edits if e.status == "skipped"]
    if args.check:
        for e in edits:
            if e.status in ("pending", "update") and e.new_text is not None:
                path = root / e.target
                if not path.exists():
                    continue
                diff = _unified(path.read_text(encoding="utf-8"), e.new_text)
                if diff:
                    print("--- per-edit diff [" + str(e.num) + ". " + e.name + "]")
                    print(diff, end="")
        if broken:
            print("WOULD NOT APPLY: " + str(len(broken)) + " broken edit(s) - gateway file drifted")
            return 1
        pending = [e for e in edits if e.status in ("pending", "update")]
        applied = [e for e in edits if e.status == "applied"]
        if pending:
            print(
                "would apply "
                + str(len(pending))
                + " edit(s); already applied: "
                + str(len(applied))
                + ("; skipped: " + str(len(skipped)) if skipped else "")
            )
        else:
            print(
                "already applied - nothing to do"
                + (" (" + str(len(skipped)) + " informational skip)" if skipped else "")
            )
        return 0

    if broken:
        print("ABORT - no files written; fix or reconcile the broken anchor(s) above")
        return 1
    written = 0
    for e in edits:
        if e.status == "applied" or e.status == "skipped":
            continue
        target = root / e.target
        target.parent.mkdir(parents=True, exist_ok=True)
        if e.new_bytes is not None:
            target.write_bytes(e.new_bytes)
        elif e.new_text is not None:
            target.write_text(e.new_text, encoding="utf-8")
        else:
            continue
        written += 1
        print("wrote " + str(target) + " (" + str(e.num) + ". " + e.name + ")")
    print("done: " + str(written) + " edit(s) written, " + str(len(edits) - written) + " already applied/skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
