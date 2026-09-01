#!/usr/bin/env python3
"""Reflow Markdown prose to a fixed wrap width (default 200).

House doc style: paragraphs are hard-wrapped (render-neutral — Markdown joins
wrapped lines) so `git diff` stays line-local and files read well in
terminals. This tool re-wraps prose to the chosen width.

Untouched, by design:
  * fenced code blocks (``` / ~~~)
  * tables (lines starting with `|`) — rows are already single lines
  * headings, horizontal rules, HTML-ish lines (`<div>`, `<img>`, comments)
  * reference-link definitions (`[id]: url`)
  * lines ending in a hard break (2+ trailing spaces)
  * blockquote `>` prefixes (inner text is reflowed)

Safety: a whitespace invariant — all whitespace runs collapsed — must hold
between input and output, or the file is left untouched. Reflowing can
therefore never change wording, only where lines break.

Writes happen in place (same inode), preserving any hardlink twins.

Usage:
  scripts/reflow_docs.py [PATH ...] [--width N] [--check]

With no PATHs, reflows the repo's documentation set (README, USAGE,
CHANGELOG, documentation/*.md, openwebui_extension/README.md).
`--check` reports what would change without writing.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_PATHS = [
    "README.md",
    "USAGE.md",
    "CHANGELOG.md",
    "documentation/AGENTS.md",
    "documentation/API.md",
    "documentation/DEPLOYMENT.md",
    "documentation/DEVELOPMENT_NOTES.md",
    "documentation/FEATURES.md",
    "documentation/MCP.md",
    "documentation/MEMORY.md",
    "documentation/ROADMAP.md",
    "documentation/SCALE.md",
    "openwebui_extension/README.md",
]

FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
HEADING_RE = re.compile(r"^#{1,6}\s")
TABLE_RE = re.compile(r"^\s*\|")
HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
HTML_RE = re.compile(r"^\s*<")
REFLINK_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S")
HARD_BREAK_RE = re.compile(r"\s{2,}$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])(\s+)(\S.*)$")
QUOTE_RE = re.compile(r"^\s*>")
QUOTE_PREF_RE = re.compile(r"^(?:\s?> ?)+")
# 4+ indented non-list line: assume an indented code block — never touch it.
INDENT_CODE_RE = re.compile(r"^\s{4,}\S")

ATOMIC_RES = (HEADING_RE, TABLE_RE, HR_RE, HTML_RE, REFLINK_RE)


def _atomic(line: str) -> bool:
    if INDENT_CODE_RE.match(line) and not LIST_RE.match(line):
        return True
    return any(r.match(line) for r in ATOMIC_RES)


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# A wrapped line may never START with one of these (except an intentional
# list marker on a first line): re-wrap would re-parse it — and Markdown may
# render it — as a new list item / heading / quote / table row / fence.
BAD_START_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|>|\||```|~~~)")


def _wrap(text: str, width: int, **kw) -> list[str]:
    """textwrap.wrap, but widened until no continuation line starts a construct.

    Only non-first lines are checked: a first line carries whatever marker the
    caller put in ``initial_indent`` on purpose. Widening by one char per try
    moves the break point until the offending token is mid-line; capped so a
    paragraph that genuinely cannot avoid it is emitted anyway (the caller's
    structure/idempotency checks would flag it).
    """
    w = width
    lines: list[str] = []
    for _ in range(80):
        lines = textwrap.wrap(
            text,
            width=w,
            break_long_words=False,
            break_on_hyphens=False,
            **kw,
        )
        if not any(BAD_START_RE.match(ln) for ln in lines[1:]):
            return lines
        w += 1
    return lines


def reflow(text: str, width: int) -> str:
    lines = text.split("\n")
    out: list[str] = []
    para: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal para
        if para:
            joined = " ".join(s.strip() for s in para)
            out.extend(_wrap(joined, width) or [""])
            para = []

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        fence = FENCE_RE.match(line)

        if in_fence:
            out.append(line)
            if fence:
                in_fence = False
            i += 1
            continue
        if fence:
            flush()
            out.append(line)
            in_fence = True
            i += 1
            continue
        if not line.strip():
            flush()
            out.append(line)
            i += 1
            continue
        if _atomic(line):
            flush()
            out.append(line)
            i += 1
            continue
        if HARD_BREAK_RE.search(line):
            # Explicit <br>: keep the line (and the break) exactly as-is.
            flush()
            out.append(line)
            i += 1
            continue
        if QUOTE_RE.match(line):
            flush()
            pref = QUOTE_PREF_RE.match(line).group(0)
            group = [line]
            fences = 1 if re.match(r"^\s*(```|~~~)", line[len(pref):]) else 0
            j = i + 1
            while j < n and lines[j].strip():
                if FENCE_RE.match(lines[j]) or HARD_BREAK_RE.search(lines[j]):
                    break
                m = QUOTE_PREF_RE.match(lines[j])
                if not m or m.group(0) != pref:
                    break
                if re.match(r"^\s*(```|~~~)", lines[j][len(pref):]):
                    fences += 1
                group.append(lines[j])
                j += 1
                if fences >= 2:  # consumed a complete fenced block
                    break
            if fences:
                out.extend(group)  # quote containing a fence: emit verbatim
                i = j
                continue
            inner = [ln[len(pref):] for ln in group]
            if any(LIST_RE.match(s) for s in inner):
                out.extend(lines[i:j])  # quoted list: markers must survive
                i = j
                continue
            joined = " ".join(s.strip() for s in inner)
            out.extend(pref + w for w in (_wrap(joined, width) or [""]))
            i = j
            continue
        m = LIST_RE.match(line)
        if m:
            flush()
            indent, marker, sp, rest = m.groups()
            hang = " " * (len(indent) + len(marker) + len(sp))
            pieces = [rest]
            j = i + 1
            while j < n and lines[j].strip():
                l2 = lines[j]
                if (
                    FENCE_RE.match(l2)
                    or LIST_RE.match(l2)
                    or _atomic(l2)
                    or QUOTE_RE.match(l2)
                    or HARD_BREAK_RE.search(l2)
                ):
                    break
                if _leading_spaces(l2) == 0:
                    break  # lazy continuation: leave for its own paragraph
                pieces.append(l2.strip())
                j += 1
            joined = " ".join(p.strip() for p in pieces)
            first = indent + marker + sp
            out.extend(
                _wrap(joined, width, initial_indent=first, subsequent_indent=hang)
                or [first.rstrip()]
            )
            i = j
            continue
        para.append(line)
        i += 1

    flush()
    return "\n".join(out)


def _norm(text: str) -> str:
    # Drop blockquote markers first: re-wrapping a quote legitimately moves
    # its per-line `> ` prefixes, and they are not content.
    text = re.sub(r"(?m)^\s*(?:>\s?)+", "", text)
    return " ".join(text.split())


def process(path: Path, width: int, check: bool) -> tuple[str, str]:
    old = path.read_text(encoding="utf-8")
    new = reflow(old, width)
    if _norm(old) != _norm(new):
        return "SKIP", f"whitespace invariant FAILED — not written: {path}"
    if new == old:
        return "OK", f"already conforming: {path}"
    over = sum(1 for ln in new.split("\n") if len(ln) > width)
    if not check:
        # In-place write keeps the inode (and any hardlink twins) intact.
        with path.open("w", encoding="utf-8") as fh:
            fh.write(new)
    return "WROTE" if not check else "DIFF", (
        f"{'[check] ' if check else ''}reflowed: {path} "
        f"({len(old.splitlines())} -> {len(new.splitlines())} lines, "
        f"{over} line(s) still > {width} chars: tables/HTML/URLs)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="Markdown files (default: repo doc set)")
    ap.add_argument("--width", type=int, default=200, help="wrap width (default 200)")
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    rel = args.paths or DEFAULT_PATHS
    rc = 0
    for r in rel:
        p = r if r.startswith("/") else REPO / r
        if not p.is_file():
            print(f"MISSING: {p}")
            rc = 1
            continue
        status, msg = process(p, args.width, args.check)
        print(msg)
        if status == "SKIP":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
