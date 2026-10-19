# andrew-bydlon-memory — distilled theme set (2026-10-17)

Synthesized from the raw memory export andrew-bydlon-memory-documents.md.

## What this is
47 themed documents replacing 682 memory units (1,638 underlying memory records),
2,556,144 chars -> 236,468 chars (10.8x compression).

Each document merges every memory that covered the same theme. The sources were
heavily redundant: typically 5-15 memories retold the SAME session or decision in
different words, which is what polluted recall. Each theme doc states each fact
once, in its strongest form.

## Files
- <group>.txt — one themed memory per file, 1.4k-9.5k chars
- MANIFEST.csv — group, project, sizes, compression, source counts, sha256
- PROVENANCE.json — group -> every source unit and memory id it was built from

## Format
Every document opens with the standard memory header:
    [MEMORY INDEX | project: <slug> | type: theme | topics: a,b,c]

## Conventions used
- Exact paths, pod names, image digests, versions, namespaces, error strings and
  config keys are preserved verbatim — that is the recall value.
- Conflicting sources resolved toward the NEWEST state, with the change noted.
- Session logistics (who asked, retries, tool narration) were dropped.
- Ephemeral identifiers (one-off diagnostic pod suffixes) were dropped when they
  carry no future value; durable object names were kept.

## Known limits
- 15 units resisted topic classification and landed in misc-notes.
- Some groups are thin by nature (single-source themes) and compress poorly.
- 126 headerless fragment units were grouped by keyword, so a few may sit in a
  less-than-ideal theme. PROVENANCE.json makes every grouping auditable.