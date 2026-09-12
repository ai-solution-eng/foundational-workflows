# Long-term Memory

Per-user long-term memory of past LLM interactions, backed by the Multimodal RAG server. Three clients are covered in this folder — **opencode** ([opencode.md](opencode.md), MCP tools, LLM-curated),
**Open WebUI** ([owui.md](owui.md), filter inlet/outlet with auto-recall + LLM-distilled writes), and **DSH** ([dsh/](dsh/README.md), MCP tools + a host-plugin session logger). The server-side
building blocks are the same for all of them.

> MCP tool signatures and connection configs (including the opencode/Claude/OWUI/stdio client JSON) live in [FEATURES.md](../FEATURES.md) § MCP server. This folder is about the memory store itself and
> per-client setup.

## The two tool namespaces

The MCP server exposes memory in two namespaces, so a client can scope what the model sees:

- **`rag-memory_*`** — personal, per-user long-term memory: `add_memory`, `search_memory`, `delete_memory`, `list_memories`, `forget_session`. The memory dataset and its password are supplied by the
  client via request headers, so the model does NOT pass `dataset_name` or `password` to these tools.
- **`rag-knowledge_*`** — access to the shared project/knowledge datasets: `search_dataset`, `search_datasets`, `list_datasets`, `get_dataset_files`, `get_dataset_info`, `unlock_dataset`,
  `describe_media`, `transcribe_audio`. For these you DO pass `dataset_name` explicitly; pass `password` only for protected datasets.

## Dataset / password model

Memories live in password-protected RAG datasets — one Qdrant collection per user. Identity resolution:

- `dataset_name` ← `X-Memory-Dataset` request header (or `MEMORY_DATASET` env); `password` ← `X-Dataset-Password` request header. The client (opencode config, DSH profile, OWUI filter) injects these
  per request; the user never sees them and **should never be shown them** — they are resolved silently server-side.
- The headers are read **only inside the memory tools**, so a memory password can never silently unlock another dataset. The general dataset tools require an explicit `password=` argument or a prior
  `unlock_dataset`.
- Per-user isolation **is** the dataset password: separate dataset + separate password per user. In OWUI both are derived server-side from the SSO identity (see [owui.md](owui.md)); in opencode/DSH
  they come from the user's own environment.

## How memories are written

| | opencode | Open WebUI | DSH |
|---|---|---|---|
| **Write trigger** | Auto: `session-memory-logger` plugin writes a structured session history (`kind: session_history`) after the conversation goes quiet (flushed on exit) **and** the model calls `add_memory` for distilled notes | `outlet()` filter asks a distillation LLM after each reply | Auto: DSH host-plugin session logger (`kind: session_history`) **and** model `add_memory` calls |
| **Recall trigger** | Model calls `rag-memory_search_memory` (proactive policy: [opencode.md](opencode.md)) | `inlet()` filter auto-searches at conversation start | Model calls `mcp__rag-memory__search_memory` (policy lives in the agent preset) |
| **Transport** | MCP (streamable-http, stateless) | REST API (direct HTTP from filter) | MCP (streamable-http) |
| **Dataset/password** | `{env:}` headers in `opencode.jsonc` | HMAC-derived from SSO `__user__` (or shared valve) | `!!js process.env.*` rows in the DSH profile |
| **Provenance** | `source: "opencode:memory"` | `source: "openwebui:memory"` | `source: "dsh:memory"` |

All paths land in the same Qdrant collection via `DatasetManager.add_documents`. Raw transcripts are never stored verbatim — session histories are reconstructed into structured `### User` / `###
Assistant` / `### Tool` summaries with a provenance header. Near-duplicates are auto-skipped at cosine ≥ `RAG_DEDUP_THRESHOLD` (default `0.995`), so re-saving a learned fact is a harmless no-op.

Because datasets are client-agnostic, **one shared dataset per user works across all three clients with no code changes** (recommended). Provenance (`source`) tells memories apart; recall currently
surfaces all of them together — a server-side `source` filter is the noted follow-up in [FEATURES.md](../FEATURES.md) § Roadmap.

## Operations

**Recall** — `rag-memory_search_memory` with a natural-language description of what you're looking for (optionally `image`/`video`/`audio`, `top_k` default 5, optional reranker).

**Write** — `rag-memory_add_memory` with standalone, specific text; pass `metadata={"kind": "decision"|"preference"|"gotcha"|"fact", "tags": [...], "session_id": "..."}` so memories are attributable.
One memory per call. The recall/write *policy* (when an agent should do this) lives per client: [opencode.md](opencode.md) for opencode, the DSH agent preset for DSH (template in
[dsh/README.md](dsh/README.md)).

**Delete** — `rag-memory_delete_memory(memory_ids=[...])` by explicit point IDs from `search_memory`/`list_memories` results (explicit-IDs-only by design — no query/similarity-directed deletion, so an
LLM can never delete what it hasn't seen listed). Prefer correcting with a new memory when the fact is still useful, and mention the supersession. `rag-memory_forget_session(session_id=...)` wipes one
session's stored history (never touches curated memories).

**Size cap** — every memory (session histories and curated notes) is split into docs of at most `MEMORY_MAX_TOKENS` tokens each (MCP container env; default 8192, set via `extraEnv` in the chart — keep
at or below the embedder's context window). The header is prepended to every chunk; splits are marked `memory_chunks` / `chunk_index` / `chunk_total` / `memory_truncated`. Sessions are **replaced in
place** per `session_id` when re-flushed, so the store keeps one current history per session.

**Dedup tuning** — `RAG_DEDUP_THRESHOLD` (env on the MCP/API container, default 0.995): raise toward 0.998 for a curated store (keeps more distinct facts), lower toward 0.97 to collapse more
aggressively.

**Rotation** — changing a dataset's password via the REST API and updating the client (`RAG_MEMORY_PASSWORD` env, or re-derivation in OWUI) is the whole story; see [owui.md](owui.md) for the OWUI
`MEMORY_SECRET` case.

**Confirming it works:**

- **opencode:** `opencode mcp list` shows both connections; watch the tool-call stream for `rag-memory_search_memory` / `rag-memory_add_memory`. The session-history plugin writes `[session-memory] …`
  diagnostics to `~/.local/share/opencode/log/session-memory.log` (override the directory with `SESSION_MEMORY_LOG_DIR`) and never writes to the terminal.
- **OWUI:** check the filter logs for `Memory recall: N hit(s)` and `Memory stored in 'owui-memory-…': …`.
- **Server:** MCP container logs show tool invocations; `GET /api/datasets/{name}` returns `document_count` to confirm writes are landing.
- **API-key note:** `security.apiKey` does not affect the memory plugins — the opencode and DSH session-memory plugins talk MCP (`/mcp`, `tools/call`) and the MCP server has no API-key middleware;
  only direct REST callers need `X-RAG-Api-Key`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| opencode: `ToolError: No memory dataset specified` | `RAG_MEMORY_DATASET` not exported | Export it in the shell that launches opencode |
| opencode: no `[session-memory]` lines in `session-memory.log` | Plugin not loaded | Plugins load at startup — quit and restart opencode; confirm the plugin file exists (see [opencode.md](opencode.md)) |
| opencode: `Incorrect password for dataset` | `RAG_MEMORY_PASSWORD` wrong / stale | Re-verify against the dataset's password |
| opencode: `rag-memory` connection not listed | URL unreachable / ingress token missing | Check `RAG_INGRESS_TOKEN` and the URL; try `opencode mcp debug rag-memory` |
| opencode: `Session not found` (404) on MCP calls | Multi-replica deployment running pre-stateless MCP mode | v1.3.0+ runs `stateless_http=True`; upgrade the image ([DEPLOYMENT.md](../DEPLOYMENT.md)) |
| Any: `MEDIA_TOKEN_SECRET is required` at startup | Shared HMAC secret unset | Set `security.mediaTokenSecret` (chart) — both containers refuse to start without it |
| OWUI: no memories recalled for a new user | Dataset doesn't exist yet | Set `MEMORY_AUTO_CREATE=true`, or pre-create the user's dataset |
| OWUI: `Memory store failed` in logs | Derived password ≠ dataset's password (e.g. `MEMORY_SECRET` changed) | Re-create the dataset or restore the old secret |
| OWUI: distillation never writes | `DISTILL_LLM_*` not set, or replies < `DISTILL_MIN_REPLY_CHARS` | Configure distillation LLM; lower the char threshold if needed |
| Both: recall feels noisy | Paraphrastic dups accumulating | Raise `RAG_DEDUP_THRESHOLD` toward 0.998 |

## Where the agent policy lives

The former `documentation/AGENTS.md` (the opencode agent recall/write policy that opencode auto-loads) now lives as part of [opencode.md](opencode.md) — point opencode's `instructions` at that file
(the `opencode.jsonc` template in this folder already does). DSH agents get the equivalent policy text from the DSH preset template in [dsh/README.md](dsh/README.md).

## Reference — file map

| File | What's in it |
|---|---|
| `src/multimodal_rag/mcp_server.py` | The 13 MCP tools, `_MemoryHeaderMiddleware`, `_run_retrieval`, memory splitting |
| `openwebui_extension/filter.py` | OWUI inlet (recall) + outlet (distil/store), per-user HMAC, auto-create |
| `memory/opencode.md` | opencode setup + agent recall/write policy (successor of the old AGENTS.md) |
| `memory/opencode.jsonc` | Two-entry MCP config template for opencode |
| `memory/opencode/` | opencode plugin/tool templates (session logger, provenance, session-id) |
| `memory/dsh/` | DSH integration design + host-plugin template |
| `memory/owui.md` | Open WebUI filter memory setup |
| `../FEATURES.md` | MCP tool reference + all client connection configs |
| `../DEPLOYMENT.md` | Chart variants, values, deployment targets |
| `../../openwebui_extension/README.md` | OWUI filter valves, per-user isolation, setup |
