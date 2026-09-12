# opencode — Memory Setup & Agent Policy

How opencode gets per-user long-term memory from the Multimodal RAG server: a two-connection MCP config, optional plugins for automatic session history, and the recall/write policy the model follows.
Store mechanics, operations, and troubleshooting: [README.md](README.md).

## 1. One-time (cluster side)

Create one password-protected dataset via the RAG HTML frontend (e.g. `andrew-memory`). That name + password become your `RAG_MEMORY_DATASET` / `RAG_MEMORY_PASSWORD`.

## 2. Config — two connections to the same server

Use the template [`opencode.jsonc`](opencode.jsonc) (copy it to your project root, or load via `OPENCODE_CONFIG=documentation/memory/opencode.jsonc`). It defines:

- **`rag-memory`** — same `/mcp` URL, sends the `X-Memory-Dataset` + `X-Dataset-Password` headers, with every non-memory tool disabled.
- **`rag-knowledge`** — same URL, no memory headers, with every memory tool disabled.

Splitting the namespaces keeps the memory password off every knowledge request and keeps dataset tools out of the memory connection — the server additionally scopes the memory headers to the memory
tools only, so a memory password can never silently unlock another dataset.

```jsonc
"mcp": {
  "rag-memory": {
    "type": "remote",
    "url": "https://rag-mcp-server.<YOUR-DOMAIN>/mcp",
    "headers": {
      "Authorization": "Bearer {env:RAG_INGRESS_TOKEN}",
      "X-Memory-Dataset": "{env:RAG_MEMORY_DATASET}",
      "X-Dataset-Password": "{env:RAG_MEMORY_PASSWORD}"
    }
  },
  "rag-knowledge": { "type": "remote", "url": "https://rag-mcp-server.<YOUR-DOMAIN>/mcp", "headers": { "Authorization": "Bearer {env:RAG_INGRESS_TOKEN}" } }
}
```

### Env vars (export before launching opencode)

| Var | Required | Purpose |
|---|---|---|
| `RAG_MEMORY_DATASET` | yes | Your memory dataset name (e.g. `andrew-memory`) |
| `RAG_MEMORY_PASSWORD` | yes | That dataset's password |
| `RAG_INGRESS_TOKEN` | only via ingress | Platform bearer token — only when reaching the server through the oauth2-proxy ingress. Drop for a local `kubectl port-forward` (`http://localhost:8001/mcp`). |

Verify: `opencode mcp list` — both `rag-memory` and `rag-knowledge` should connect; confirm the prefixed tool names match the `tools` globs in the template.

## 3. Agent recall/write policy

The successor of the old `documentation/AGENTS.md` — point opencode's `instructions` at this file (the `opencode.jsonc` template already does), or copy the policy into your global
`~/.config/opencode/AGENTS.md` for memory across all projects. If neither namespace is connected, ignore this section.

**Namespaces:** `rag-memory_*` = personal per-user memory (headers supply dataset/password — never pass them yourself). `rag-knowledge_*` = shared project datasets (pass `dataset_name` explicitly;
`password` only for protected datasets). Prefer `rag-knowledge_search_dataset` over `get_dataset_files` for finding content — datasets can contain tens of thousands of files and listing them wastes
context.

### When to RECALL — `rag-memory_search_memory`

- At the **start of any non-trivial task** (a task likely to span multiple steps or touch existing code), call `rag-memory_search_memory` with a concise summary of the task. This surfaces relevant
  past decisions, preferences, gotchas, and prior work before you act.
- Whenever the **user references prior work** ("remember when…", "like we did before", "last time"), call `rag-memory_search_memory` with the described topic.
- Keep `top_k` at the default (5). Turn the reranker on only if the first recall feels off.

### When to WRITE — `rag-memory_add_memory`

A full record of the session (prompts, responses, tool calls, file changes) is captured automatically by the `session-memory-logger` plugin (`kind: session_history`), so don't reproduce the session —
only durable, distilled facts. After completing a non-trivial task, store a memory **only if** something durable was learned:

- A **decision** and its rationale ("chose tabs over spaces for repo style", "auth uses oauth2-proxy bearer tokens, not cookies").
- A confirmed **preference** of the user.
- A **gotcha / fix** that took real effort to find and could recur.
- A non-obvious **architectural fact** about the codebase.

Do NOT store: transient debugging steps, trivial Q&A, restatements of what is already in committed docs/code, or anything obvious from reading the repo. When in doubt, don't write — noise degrades
recall.

### How to WRITE a good memory

- Make it **standalone**: a future session with zero other context must understand it. "User prefers tabs over spaces because of repo style" beats "prefers tabs".
- Be **specific and concrete**: names of files, commands, error messages.
- Pass `metadata={"kind": "decision"|"preference"|"gotcha"|"fact", "tags": [...], "session_id": "<this session>"}` so memories are attributable. One memory per call; call multiple times for distinct
  facts.

### Rules

- Never mention the memory dataset name or password to the user — they are resolved silently server-side and are none of the user's concern.
- Near-duplicate memories are auto-skipped at cosine ≥ 0.995, so re-saving a learned fact later is a harmless no-op.
- Wrong or outdated memories can be removed with `rag-memory_delete_memory` (by explicit `memory_id` from `search_memory`/`list_memories` results) — prefer correcting with a new memory when the fact
  is still useful, and mention the supersession. `rag-memory_forget_session` wipes one session's stored history.
- Memory is per-user (per dataset). Do not assume a teammate's memory is present; recall only ever searches your own store.

## 4. Client-side plugins & tools (shipped in this repo)

`memory/opencode/` ships the opencode-side glue for automatic memory as **templates** — they are not auto-loaded, so a clone of this repo won't change your opencode until you install them. They are
no-ops until `RAG_MEMORY_DATASET` is exported:

| File | What it does |
|---|---|
| `memory/opencode/plugins/session-memory-logger.ts` | Watches each session and writes a structured history (`kind: "session_history"`) to the memory dataset — prompts, responses, tool calls, and file changes — after the conversation goes quiet, and flushes anything pending when opencode exits. |
| `memory/opencode/plugins/memory-provenance.ts` | Auto-attaches git + session provenance (HEAD before/after, branch, repo, diff stat, session id) to every `add_memory` call. |
| `memory/opencode/tools/session-id.ts` | Exposes a `session-id` tool the model can call to get the current session id for tagging memories. |

Install them into your global opencode config to get automatic memory in **every** project:

```bash
mkdir -p ~/.config/opencode/plugins ~/.config/opencode/tools
cp documentation/memory/opencode/plugins/*.ts ~/.config/opencode/plugins/
cp documentation/memory/opencode/tools/*.ts   ~/.config/opencode/tools/
```

(Plugins load at opencode startup — restart after copying.)

Notes:

- The session-history plugin talks directly to the memory MCP server over HTTP; it defaults to the `rag-memory` URL from `opencode.jsonc` and can be overridden with `RAG_MEMORY_URL`. It relies on the
  same private CA opencode already needs to reach these servers (`NODE_EXTRA_CA_CERTS`).
- Keeping them only in the global config means exactly **one** copy loads; don't also drop them into a project's `.opencode/plugins/` / `.opencode/tools/` or they load twice in that repo (harmless —
  server-side dedup skips the second write — but wasteful).

## 5. Sharing one store across clients

Datasets are client-agnostic, so one shared dataset per user works across opencode + Open WebUI + DSH with no code changes — point both clients at the same dataset name + password (opencode: the env
vars above; OWUI: a matching `MEMORY_DATASET_PREFIX` or a fixed dataset; DSH: the same env rows in its profile). Provenance `source` values distinguish writers; see [README.md](README.md).
