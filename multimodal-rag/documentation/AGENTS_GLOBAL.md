# Agent Instructions — Long-term memory

When the `rag-memory` and `rag-knowledge` MCP namespaces are connected,
you have **long-term memory** of past interactions:

- **`rag-memory`** — personal, per-user long-term memory. Tools:
  `rag-memory_add_memory`, `rag-memory_search_memory`. The memory dataset
  and its password are supplied by opencode via request headers, so you do
  NOT pass `dataset_name` or `password` to these tools.
- **`rag-knowledge`** — access to shared project/knowledge datasets.
  Tools: `rag-knowledge_search_dataset`, `rag-knowledge_list_datasets`,
  `rag-knowledge_get_dataset_files`, `rag-knowledge_get_dataset_info`,
  `rag-knowledge_unlock_dataset`, `rag-knowledge_describe_media`,
  `rag-knowledge_transcribe_audio`. For these you DO pass `dataset_name`
  explicitly; pass `password` only for protected datasets.

> If neither namespace is connected, ignore this file — the memory tools
> are unavailable and you should proceed normally.

## When to RECALL — `rag-memory_search_memory`
- At the **start of any non-trivial task** (a task likely to span multiple
  steps or touch existing code), call `rag-memory_search_memory` with a
  concise summary of the task. This surfaces relevant past decisions,
  preferences, gotchas, and prior work before you act.
- Whenever the **user references prior work** ("remember when…", "like we
  did before", "last time"), call `rag-memory_search_memory` with the
  described topic.
- Keep `top_k` at the default (5). Turn the reranker on only if the first
  recall feels off.

## When to WRITE — `rag-memory_add_memory`
A full record of each session (prompts, responses, tool calls, file
changes) is captured automatically by the `session-memory-logger` plugin
(`kind: session_history`), so don't reproduce the session here — only
durable, distilled facts.

After completing a non-trivial task, store a memory **only if** something
durable was learned. Worth remembering:
- A **decision** and its rationale.
- A confirmed **preference** of the user.
- A **gotcha / fix** that took real effort to find and could recur.
- A non-obvious **architectural fact** about a codebase.

Do NOT store: transient debugging steps, trivial Q&A, restatements of what
is already in committed docs/code, or anything that would be obvious from
reading the repo. When in doubt, don't write — noise degrades recall.

## How to WRITE a good memory
- Make it **standalone**: a future session with zero other context must
  understand it.
- Be **specific and concrete**: names of files, commands, error messages.
- Pass `metadata={"kind": "decision"|"preference"|"gotcha"|"fact",
  "tags": [...], "session_id": "<this session>"}` so memories are
  attributable.
- One memory per call; call multiple times if several distinct facts
  deserve saving.

## Rules
- Never mention the memory dataset name or password to the user.
- Near-duplicate memories are auto-skipped at cosine ≥ 0.995, so re-saving
  a learned fact in a later session is a harmless no-op.
- There is no `delete_memory` tool in v1. If you must correct a wrong
  memory, write a new, corrected one and tell the user the old one is
  superseded.
- Memory is per-user. Do not assume a teammate's memory is present.
