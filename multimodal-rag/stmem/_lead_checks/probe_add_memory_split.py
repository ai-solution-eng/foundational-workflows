import json, re



def _split_memory_header(text: str) -> tuple[str, str]:
    """Return ``(header, body)`` for a memory document.

    The header is the document prologue: everything before the first
    markdown section heading.  For session histories this is the
    provenance block (title, session id, git info, file list) — the
    ``## ``-level heading for the opencode format, or the leading
    ``### ``-level section for the dsh format, whose body carries no
    ``## `` headings.  The header is prepended to **every** chunk, so a
    split document keeps its identifying block in each chunk's text.
    For free-form notes with no section headings there is no header and
    the whole text is the body.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## ") or line.startswith("### "):
            header = "\n".join(lines[:i]).strip()
            body = "\n".join(lines[i:])
            return (header + "\n") if header else "", body
    return "", text


_MEMORY_SECTION_HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"### (?:User|Assistant)(?: .*)?"
    r"|### Tool — .+"
    r"|## (?:Transcript|Files changed)(?: .*)?"
    r")$"
)



def _split_memory_sections(body: str) -> list[str]:
    """Split a memory body into sections at the document's own headings.

    The memory-text analogue of the code processor's definition-boundary
    splitting: each section keeps its heading (the analogue of a top-level
    definition), so packed chunks never break mid-section.  Only the two
    session-history formats' own signatures match — ``### User`` /
    ``### Assistant`` / ``### Tool`` for the dsh format, ``## Transcript``
    / ``## Files changed`` for the opencode format — because a session
    that reads markdown files embeds *their* headings inside tool output,
    and those content headings must never split a section.  A bare
    heading with no content of its own (e.g. ``## Transcript``) is
    attached to the section that follows it.
    """
    matches = list(_MEMORY_SECTION_HEADING_RE.finditer(body))
    if not matches:
        return [body]
    sections: list[str] = []
    if matches[0].start() > 0:
        sections.append(body[: matches[0].start()])
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(body[match.start():end])
    attached: list[str] = []
    for section in sections:
        stripped = section.strip()
        if attached and "\n" not in stripped and stripped.startswith("#"):
            attached[-1] += section
            continue
        attached.append(section)
    return attached



def _pack_memory_sections(
    sections: list[str],
    measure: Any,
    budget: int,
    split_oversized: Any,
) -> list[str]:
    """Greedy whole-section packing within a token budget.

    Mirrors the code splitting pattern: pack semantic sections so chunks
    never break mid-section unless one section alone exceeds the budget —
    that outsized section is positionally split inside itself, the only
    place boundary context is lost.  A final chunk holding fewer than a
    quarter of the budget folds into the previous chunk when the two fit,
    so no tiny trailing chunk is emitted.

    ``measure`` counts tokens for one section (real tokenizer or the
    ~4 chars/token estimate); ``split_oversized`` splits one oversized
    section's body within the budget.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for section in sections:
        tokens = measure(section)
        if tokens > budget:
            if current:
                groups.append(current)
                current, current_tokens = [], 0
            groups.extend([c] for c in split_oversized(section) if c.strip())
            continue
        if current and current_tokens + tokens > budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(section)
        current_tokens += tokens
    if current:
        groups.append(current)
    if len(groups) > 1:
        tail_tokens = sum(measure(s) for s in groups[-1])
        prev_tokens = sum(measure(s) for s in groups[-2])
        if tail_tokens < budget // 4 and tail_tokens + prev_tokens <= budget:
            groups = groups[:-2] + [groups[-2] + groups[-1]]
    return ["".join(group) for group in groups]




# stub the tokenizer factory: force the CHARACTER FALLBACK path
def _memory_splitter(max_tokens):
    return None

def _split_memory_text(text: str, max_tokens: int) -> tuple[list[str], bool]:
    """Split *text* into memory chunks of at most *max_tokens* tokens each.

    Splitting is **context-aware** (the session-history analogue of the
    code processor's definition-boundary splitting): the body is packed
    from whole sections at markdown heading boundaries, so a chunk never
    breaks a ``### User`` / ``### Assistant`` / ``### Tool`` section
    mid-body unless that one section alone exceeds the budget.  Only a
    single oversized section is positionally split inside itself.

    The document header (e.g. the session-history provenance block) is
    prepended to **every** chunk so each split stays identifiable — the
    ``session_id`` etc. travels with each chunk.  Uses the same tokenizer
    logic as dataset-side text splitting; falls back to ~4 chars/token.

    Returns ``(chunks, was_split)`` where each chunk is header + sections.
    """
    if not text or max_tokens <= 0:
        return ([text] if text else []), False

    header, body = _split_memory_header(text)
    if not body:
        return [text], False

    splitter = _memory_splitter(max_tokens)
    try:
        header_tokens = splitter.count_tokens(header) if header and splitter is not None else 0
    except Exception:
        header_tokens = 0
    if not header_tokens:
        header_tokens = max(0, len(header) // 4)

    content_budget = max(1, max_tokens - header_tokens)
    sections = _split_memory_sections(body)
    # Skip splitting only when the body measurably fits one chunk; without a
    # tokenizer the character fallback below does its own measuring, so an
    # oversized single section still splits.
    if len(sections) <= 1 and splitter is not None and splitter.count_tokens(body) <= content_budget:
        return [text], False

    if splitter is not None:
        try:
            body_splitter = _memory_splitter(content_budget)
            chunks = _pack_memory_sections(
                sections,
                body_splitter.count_tokens,
                content_budget,
                body_splitter.split_text,
            )
            chunks = [c for c in chunks if c.strip()]
            if header:
                chunks = [header + c for c in chunks]
            return chunks, len(chunks) > 1
        except Exception:
            pass  # fall through to the character-based estimate

    # Character-based fallback: ~4 chars per token, same section packing.
    # The packing budget is the token budget with a ~4 chars/token measure;
    # the oversized slice width is the same budget expressed in characters.
    chunks = _pack_memory_sections(
        sections,
        lambda section: len(section) // 4,
        content_budget,
        lambda section: [
            section[i: i + content_budget * 4]
            for i in range(0, len(section), content_budget * 4)
        ],
    )
    chunks = [c for c in chunks if c.strip()]
    if header:
        chunks = [header + c for c in chunks]
    return chunks, len(chunks) > 1




env = json.dumps({"stmm": 1, "schema": "stmm/v1", "run_id": "runA", "agent": "a1", "step": 1,
                  "epistemic": "observed", "created_at": "2026-10-11T12:00:00Z", "ttl_hours": 24,
                  "supersedes": []})

body = "\n".join("fact line %d padded with enough words to consume budget here" % i for i in range(400))
text = env + "\n" + body
chunks, was_split = _split_memory_text(text, 200)
print("heading-less oversized entry: was_split=%s chunks=%d" % (was_split, len(chunks)))
missing = [i for i, c in enumerate(chunks) if '"stmm"' not in c]
for i, c in enumerate(chunks[:4]):
    print("  chunk[%d] envelope=%s | starts: %r" % (i, '"stmm"' in c, c[:34]))
print("chunks lacking envelope:", missing)
print("VERDICT:", "F4 CONFIRMED" if missing else "F4 NOT reproduced (heading-less case)")

