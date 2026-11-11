"""MINIATURE fixture of the gateway distill routes (patcher tests only).

Mirrors the write_memory call-site shapes of the real
src/pcai_llm/api/routes_distill.py so the patcher's route-wiring edits and
their anchors are exercised. NOT the real module: the real verification
runs against a /tmp copy of the actual gateway tree.
"""

from __future__ import annotations

from typing import Any


async def _mcp_search_for(key: Any) -> None:
    return None


async def site_explicit_conversation(
    key: Any,
    conv: Any,
    sink: Any,
    doc: str,
    cid: str,
    transcript: str,
) -> tuple[bool, str]:
    """Real site :760 (/v1/distill/conversations/{id}): conv is in scope."""
    verdict: list[str] = []
    ok, detail = await sink.write_memory(
        None,
        None,
        doc,
        session_id=str(cid),
        kind="project-summary",
        fallback_doc=transcript,
        fallback_session_id=str(cid),
        mcp_search=await _mcp_search_for(key),
        verdict_out=verdict,
    )
    return ok, detail


async def site_scheduled_conversation(
    key: Any,
    conv: Any,
    sink: Any,
    doc: str,
    cid: str,
    transcript: str,
) -> tuple[bool, str]:
    """Real site :935 (scheduled conversations sweep): conv is in scope."""
    verdict: list[str] = []
    ok, detail = await sink.write_memory(
        None,
        None,
        doc,
        session_id=str(cid),
        kind="project-summary",
        fallback_doc=transcript,
        fallback_session_id=str(cid),
        verdict_out=verdict,
    )
    return ok, detail


async def site_scheduled_canonical(
    key: Any,
    messages: list[dict[str, Any]],
    sink: Any,
    doc: str,
    thread_id: str,
    transcript: str,
) -> tuple[bool, str]:
    """Real site :1179 (canonical sweep): the row's messages are in scope."""
    verdict: list[str] = []
    ok, detail = await sink.write_memory(
        None,
        None,
        doc,
        session_id=thread_id[:64],
        kind="project-summary",
        fallback_doc=transcript,
        fallback_session_id=thread_id[:64],
        mcp_search=await _mcp_search_for(key),
        verdict_out=verdict,
    )
    return ok, detail


async def site_inline_shim(
    key: Any,
    sink: Any,
    doc: str,
    session_id: str,
    transcript: str,
) -> tuple[bool, str]:
    """Real site :688 (/v1/distill): NO message rows in scope - a
    caller-rendered transcript string is all the payload carries - so this
    site is intentionally NOT wired to trace_messages."""
    ok, detail = await sink.write_memory(
        None,
        None,
        doc,
        session_id=session_id,
        kind="project-summary",
        fallback_doc=transcript,
        fallback_session_id=session_id,
        mcp_search=await _mcp_search_for(key),
    )
    return ok, detail
