"""MINIATURE fixture of the gateway distiller (patcher tests only).

A runnable stand-in for PCAI_LLM/src/pcai_llm/core/distiller.py carrying the
exact anchors apply_playbook_patch.py edits. NOT the real module: the real
verification runs against a /tmp copy of the actual gateway tree.
"""

from __future__ import annotations

from typing import Any, cast  # noqa: F401 - anchor line for the overlay patcher

PROMPT_VERSION = "dsh-2026-10-09"

KIND_SUMMARY = "project-summary"
KIND_FALLBACK = "session_history"

REQUIRED_SECTIONS = (
    "## What we did",
    "## Key decisions",
    "## Gotchas",
    "## How to proceed",
)


def document_is_complete(doc: str) -> bool:
    """Miniature of the gateway guard: requires the four sections only."""
    if not isinstance(doc, str):
        return False
    return all(s in doc for s in REQUIRED_SECTIONS)


def memory_body_chars(doc: str) -> int:
    """Length of the document content excluding the index header."""
    if not isinstance(doc, str):
        return 0
    text = doc.lstrip()
    if not text.startswith("[MEMORY INDEX"):
        return len(text.strip())
    close = text.find("]")
    remainder = text[close + 1 :] if close != -1 else text
    return len(remainder.strip())


DISTILL_SYSTEM_PROMPT = (
    "You distill AI-coding-assistant session transcripts into durable memory documents.\n"
    "OUTPUT FORMAT (markdown, exactly this shape):\n"
    "[MEMORY INDEX | project: <project-slug> | type: session-summary | topics: <topics>]\n"
    "## What we did\n"
    "## Key decisions & rationale\n"
    "## Gotchas\n"
    "## How to proceed\n"
    "RULES:\n"
    "- Preserve exact commands, paths, digests, error messages, and config values"
    " verbatim - but pick the few that carry recall value, not every one seen."
)
