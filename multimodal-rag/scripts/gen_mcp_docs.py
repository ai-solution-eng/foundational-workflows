#!/usr/bin/env python3
"""Generate documentation/MCP.md §2's tool table from the MCP server source.

AST-only: parses src/multimodal_rag/mcp_server.py with the stdlib ast module —
NO import of multimodal_rag, no network, no side effects. Prints (to stdout)
the "## 2. Available tools (N)" heading, the markdown table (tool name,
signature with defaults, first docstring sentence as the purpose, and the
hand-maintained dataset_name/password columns), and a provenance note.

Run from the repo root:

    python scripts/gen_mcp_docs.py

Paste the output between §2's hand-written prose blocks (the D26 read-only
grant note and the memory-tools footnote). The table cannot drift from the
code because it is regenerated from it; edit the docstrings, not the docs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_SOURCE = REPO_ROOT / "src" / "multimodal_rag" / "mcp_server.py"

# The two "Needs …?" columns are semantic, not derivable from the signature
# alone (e.g. set_memory_dataset takes dataset_name but "no" — omitting it
# clears the binding). They live HERE, keyed by tool name, so adding or
# renaming a tool fails this script loudly instead of drifting in the docs.
NEEDS_COLUMNS: dict[str, tuple[str, str]] = {
    "list_datasets": ("—", "—"),
    "unlock_dataset": ("yes", "yes"),
    "select_dataset": ("yes", "if protected"),
    "deselect_dataset": ("yes", "—"),
    "create_short_term_memory": ("no", "never"),
    "end_short_term_memory": ("no", "—"),
    "memory_status": ("no", "—"),
    "set_memory_dataset": ("no", "—"),
    "search_dataset": ("yes", "if protected"),
    "search_datasets": ("no", "never"),
    "get_dataset_files": ("yes", "if protected"),
    "get_dataset_info": ("yes", "if protected"),
    "dataset_add_documents": ("yes", "if protected"),
    "dataset_delete_documents": ("yes", "if protected"),
    "dataset_replace_document": ("yes", "if protected"),
    "inspect_media": ("—", "—"),
    "add_memory": ("optional¹", "optional¹"),
    "search_memory": ("optional¹", "optional¹"),
    "delete_memory": ("optional¹", "optional¹"),
    "list_memories": ("optional¹", "optional¹"),
    "forget_session": ("optional¹", "optional¹"),
    # Wave 2026-11-10 — playbook memory (Feature B) and late interaction (Feature D).
    # recall_playbooks / record_playbook_outcome / consolidate_memories are memory-store
    # tools: the dataset resolves server-side exactly like the other memory tools.
    # set_late_interaction names its dataset explicitly and is a WRITE surface (it
    # mutates the dataset's retrieval config), so it takes the password like a write.
    "recall_playbooks": ("optional¹", "optional¹"),
    "record_playbook_outcome": ("optional¹", "optional¹"),
    "consolidate_memories": ("optional¹", "optional¹"),
    "set_late_interaction": ("yes", "if protected"),
}

SENTENCE_END = re.compile(r"[.!?](?=\s)")

# Dots that do not end a sentence (so the extractor survives "e.g." / "..." later on).
_NOT_SENTENCE_END = ("e.g", "i.e", "etc", "vs", "Dr", "No", "St")


def _is_tool_decorator(dec: ast.expr) -> bool:
    """Match @mcp.tool and @mcp.tool()."""
    if isinstance(dec, ast.Attribute) and dec.attr == "tool":
        return True
    return isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "tool"


def _signature(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> str:
    """Render name(arg, arg=default, …) from the AST — defaults verbatim."""
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parts = []
    for arg, default in zip(positional, defaults):
        parts.append(arg.arg if default is None else f"{arg.arg}={ast.unparse(default)}")
    return f"{fn.name}({', '.join(parts)})"


def _first_sentence(docstring: str) -> str:
    """First sentence of a docstring, abbreviation-tolerant."""
    text = " ".join(docstring.split())
    for match in SENTENCE_END.finditer(text):
        head = text[: match.start()]
        last_word = head.rsplit(" ", 1)[-1] if head.rsplit(" ", 1) else ""
        if last_word in _NOT_SENTENCE_END:
            continue
        return text[: match.end()].strip()
    return text.strip()


def _cell(text: str) -> str:
    """Escape a string for use inside a markdown table cell."""
    return text.replace("|", "\\|")


def _tool_functions(tree: ast.Module) -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    """@mcp.tool functions in source order."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_tool_decorator(dec) for dec in node.decorator_list
        ):
            found.append(node)
    return found


def generate() -> str:
    tree = ast.parse(SERVER_SOURCE.read_text(encoding="utf-8"))
    tools = _tool_functions(tree)

    known = {fn.name for fn in tools}
    unmapped = known - set(NEEDS_COLUMNS)
    stale = set(NEEDS_COLUMNS) - known
    problems = [f"tool '{name}' has no NEEDS_COLUMNS entry" for name in sorted(unmapped)]
    problems += [f"NEEDS_COLUMNS entry '{name}' matches no registered tool" for name in sorted(stale)]
    if problems:
        raise SystemExit("gen_mcp_docs: update NEEDS_COLUMNS —\n  " + "\n  ".join(problems))

    rows = []
    for fn in tools:
        docstring = ast.get_docstring(fn, clean=True) or ""
        purpose = _first_sentence(docstring)
        needs_dataset, needs_password = NEEDS_COLUMNS[fn.name]
        rows.append(f"| `{_cell(_signature(fn))}` | {_cell(purpose)} | {needs_dataset} | {needs_password} |")

    header = (
        f"## 2. Available tools ({len(tools)})",
        "",
        "| Tool | Purpose | Needs `dataset_name`? | Needs `password`? |",
        "|------|---------|----------------------|-------------------|",
        *rows,
        "",
        "> This table (and the tool count) is generated by `scripts/gen_mcp_docs.py` from the MCP",
        "> server source — AST-only, no server import, so it cannot drift. Regenerate with",
        "> `python scripts/gen_mcp_docs.py` after any `@mcp.tool()` signature or docstring change;",
        "> do not edit the table by hand.",
    )
    return "\n".join(header)


if __name__ == "__main__":
    print(generate())
