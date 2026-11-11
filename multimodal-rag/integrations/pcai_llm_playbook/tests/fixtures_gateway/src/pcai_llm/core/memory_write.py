"""MINIATURE fixture of the gateway memory-write module (patcher tests only).

A runnable stand-in for PCAI_LLM/src/pcai_llm/core/memory_write.py carrying
the exact anchors apply_playbook_patch.py edits. NOT the real module.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import settings
from .distiller import KIND_FALLBACK, KIND_SUMMARY, PROMPT_VERSION, memory_body_chars

log = logging.getLogger(__name__)

SOURCE_LABEL = "gateway:distiller"
VERDICT_WRITTEN = "written"
VERDICT_FAILED = "failed"

McpCall = Callable[[dict[str, Any]], Awaitable[tuple[bool, str]]]


def build_add_memory_arguments(
    doc: str,
    session_id: str,
    source: str,
    kind: str,
) -> dict[str, Any]:
    """The add_memory tool arguments (miniature of the gateway shape)."""
    return {
        "text": doc,
        "metadata": {
            "kind": kind,
            "session_id": session_id,
            "source": source,
            "prompt_version": PROMPT_VERSION,
        },
    }


async def write_memory(
    key_id: uuid.UUID | None,
    mcp_call: McpCall,
    doc: str,
    *,
    session_id: str,
    kind: str = KIND_SUMMARY,
    source: str = SOURCE_LABEL,
    fallback_doc: str = "",
    fallback_session_id: str = "",
    mcp_search: McpCall | None = None,
    verdict_out: list[str] | None = None,
) -> tuple[bool, str]:
    """Write ONE memory document; miniature of the gateway contract."""

    def _verdict(v: str) -> None:
        if verdict_out is not None:
            verdict_out.append(v)

    if not doc or not doc.strip():
        _verdict(VERDICT_FAILED)
        return False, "empty document"

    min_chars = settings.distiller_min_doc_chars
    if min_chars > 0 and memory_body_chars(doc) < min_chars:
        _verdict(VERDICT_FAILED)
        return False, f"document body under the {min_chars}-char floor"

    try:
        ok, detail = await mcp_call(build_add_memory_arguments(doc, session_id, source, kind))
        if ok:
            _verdict(VERDICT_WRITTEN)
            return True, detail
    except Exception as exc:
        log.warning("memory write failed (key=%s): %s", key_id, type(exc).__name__, exc_info=True)
        ok, detail = False, "memory write failed: " + type(exc).__name__

    if fallback_doc and fallback_doc.strip():
        try:
            ok2, detail2 = await mcp_call(
                build_add_memory_arguments(
                    fallback_doc,
                    fallback_session_id or session_id,
                    source,
                    KIND_FALLBACK,
                )
            )
            if ok2:
                _verdict(VERDICT_WRITTEN)
                return True, "summary failed; wrote raw transcript as session_history"
            detail = detail + "; fallback: " + detail2
        except Exception as exc:
            log.warning("memory fallback write failed (key=%s)", key_id, exc_info=True)
            detail = detail + "; fallback failed: " + type(exc).__name__
    _verdict(VERDICT_FAILED)
    return False, detail
