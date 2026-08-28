# Long-term Memory — Setup & Operations Guide

Per-user long-term memory of past LLM interactions, backed by the
Multimodal RAG server. Two clients are supported — **opencode** (MCP
tools, LLM-curated) and **Open WebUI** (filter inlet/outlet,
auto-recall + LLM-distilled writes).

> **See also:** [MCP.md](MCP.md) for the MCP tool reference and
> connection configs; [FEATURES.md](FEATURES.md) § "MCP Server" for
> the server-side internals (`_MemoryHeaderMiddleware`, `_run_retrieval`,
> query-vector caching).

---

## 1. How it works

Both clients write memories into password-protected RAG datasets (one
Qdrant collection per user). On the opencode side memories come from two
places: a `session-memory-logger` plugin that **automatically stores a
structured history of each session** (prompts, responses, tool calls,
file changes), plus the model's own `add_memory` calls for distilled
decisions, preferences, and gotchas. Open WebUI stores memories via a
separate distillation LLM. Raw transcripts are never stored verbatim —
session histories are reconstructed into structured summaries.

| | opencode | Open WebUI |
|---|---|---|
| **Write trigger** | Auto: `session-memory-logger` plugin writes a structured session history ~45s after the conversation goes quiet (flushed on exit) **and** the model calls `add_memory` for distilled notes | `outlet()` filter asks a distillation LLM after each reply |
| **Recall trigger** | Model calls `search_memory` MCP tool (proactive, per `AGENTS.md`) | `inlet()` filter auto-searches at conversation start |
| **Transport** | MCP (streamable-http, stateless) | REST API (direct HTTP from filter) |
| **Dataset/password** | `{env:}` headers in `opencode.jsonc` | HMAC-derived from SSO `__user__` (or shared valve) |
| **Provenance** | `source: "opencode:memory"` | `source: "openwebui:memory"` |

Both paths land in the same Qdrant collection via
`DatasetManager.add_documents`. Near-duplicates are auto-skipped at
cosine ≥ `RAG_DEDUP_THRESHOLD` (default `0.995`).

---

## 2. Setup — opencode

### One-time (cluster side)
Create one password-protected dataset via the RAG HTML frontend (e.g.
`andrew-memory`).

### Config
Use the two-connection pattern in [`opencode.jsonc`](opencode.jsonc) —
`rag-memory` (sends memory headers, exposes only `add_memory` /
`search_memory`) and `rag-knowledge` (general dataset tools). See
[MCP.md](MCP.md) § 3.2 for the full config.

### Env vars (export before launching opencode)

| Var | Required | Purpose |
|---|---|---|
| `RAG_MEMORY_DATASET` | yes | Your memory dataset name (e.g. `andrew-memory`) |
| `RAG_MEMORY_PASSWORD` | yes | That dataset's password |
| `RAG_INGRESS_TOKEN` | only via ingress | Platform bearer token — only if reaching the server through the oauth2-proxy ingress. Drop for a local `kubectl port-forward` (`http://localhost:8001/mcp`). |

### Agent behavior
[`AGENTS.md`](AGENTS.md) tells the model:
- **Recall** at the start of any non-trivial task and when the user references prior work.
- **Write** after a non-trivial task only if something durable was learned.
- Never mention the dataset name or password to the user.

For memory across all projects, copy the memory section of `AGENTS.md`
into your global `~/.config/opencode/AGENTS.md`.

### Client-side plugins & tools (shipped in this repo)

`documentation/opencode-memory/` ships the opencode-side glue for
automatic memory as **templates** — they are not auto-loaded, so a clone
of this repo won't change your opencode until you install them. They are
no-ops until `RAG_MEMORY_DATASET` is exported:

| File | What it does |
|---|---|
| `documentation/opencode-memory/plugins/session-memory-logger.ts` | Watches each session and writes a structured history (`kind: "session_history"`) to the memory dataset — prompts, responses, tool calls, and file changes — ~45s after the conversation goes quiet, and flushes anything pending when opencode exits. |
| `documentation/opencode-memory/plugins/memory-provenance.ts` | Auto-attaches git + session provenance (HEAD before/after, branch, repo, diff stat, session id) to every `add_memory` call. |
| `documentation/opencode-memory/tools/session-id.ts` | Exposes a `session-id` tool the model can call to get the current session id for tagging memories. |

Install them into your global opencode config to get automatic memory in
**every** project:

```bash
mkdir -p ~/.config/opencode/plugins ~/.config/opencode/tools
cp documentation/opencode-memory/plugins/*.ts ~/.config/opencode/plugins/
cp documentation/opencode-memory/tools/*.ts  ~/.config/opencode/tools/
```

(Plugins load at opencode startup — restart after copying.)

Notes:
- The session-history plugin talks directly to the memory MCP server over
  HTTP; it defaults to the `rag-memory` URL from `opencode.jsonc` and can
  be overridden with `RAG_MEMORY_URL`. It relies on the same private CA
  opencode already needs to reach these servers (`NODE_EXTRA_CA_CERTS`).
- Keeping them only in the global config means exactly **one** copy loads;
  don't also drop them into a project's `.opencode/plugins/` /
  `.opencode/tools/` or they load twice in that repo (harmless — the
  server's near-duplicate dedup skips the second write — but wasteful).

### Verify
```bash
opencode mcp list     # both rag-memory and rag-knowledge should connect
```

---

## 3. Setup — Open WebUI

OWUI uses a filter (`openwebui_extension/filter.py`) with an **inlet**
(auto-recall at conversation start) and an **outlet** (LLM-distilled
write after each reply). Memory goes through the RAG REST API directly
— no MCP needed. Full valve reference: [`openwebui_extension/README.md`](../openwebui_extension/README.md).

> The extension dir also ships a memory-free variant, `filter_media_strip.py`
> (strips image/video/audio parts for text-only LLMs, no RAG at all). Use it
> when you want media handling without a memory store. The former
> `filter_no_memory.py` fork was removed — the full `filter.py` covers that
> case by disabling its memory/SQL-lesson valves (`MEMORY_ENABLED`,
> `SQL_LESSONS_ENABLED`).

### Per-user isolation (SSO)

OWUI filter Valves are **global** (admin-configured once), so per-user
passwords can't live in Valves. The filter derives **two** per-user
secrets from the SSO-authenticated `__user__` identity at runtime:

```
dataset_name = MEMORY_DATASET_PREFIX + sanitised(__user__.id)
             e.g. "owui-memory-a1b2c3d4"

password     = HMAC-SHA256(MEMORY_SECRET, __user__.id)[:18]   (base64url, 24 chars)
             e.g. "c82tY2vCJGCxRjTwr7MDxYxs"
```

Because OWUI populates `__user__` **after** SSO authentication, a user
cannot forge another user's id — the derivation is sound.

| | `MEMORY_SECRET` set (recommended) | `MEMORY_SECRET` empty (fallback) |
|---|---|---|
| Dataset name | per-user (from `__user__.id`) | per-user (from `__user__.id`) |
| Dataset password | per-user (HMAC-derived, unpredictable) | shared (`MEMORY_PASSWORD`) |
| If password leaks | one user exposed | all users exposed |
| Admin setup | one `MEMORY_SECRET` string | one `MEMORY_PASSWORD` string |

### Setup steps (SSO-enabled, recommended path)

1. Generate a random secret:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Install the filter (Admin Panel → Functions → paste `filter.py`).
3. In the filter ⚙️ set:
   - `MEMORY_DATASET_PREFIX` (default `owui-memory-` is fine)
   - `MEMORY_SECRET` (your random secret)
   - `MEMORY_AUTO_CREATE = true` (zero per-user provisioning)
   - `DISTILL_LLM_URL` / `DISTILL_LLM_MODEL` / `DISTILL_LLM_API_KEY`
4. Leave `MEMORY_ENABLED = true`. Recall starts immediately (empty for
   users whose dataset doesn't exist yet); writes start once the
   distillation LLM is configured (and auto-create the dataset first).

The distillation LLM can be any small fast model — it just decides "is
this worth remembering?" and writes 1-3 standalone sentences. It does
NOT need to be the same model the user is chatting with.

---

## 4. Sharing one memory store across both apps

Datasets are client-agnostic (just a Qdrant collection + PVC files), so
**one shared dataset per user works across opencode + OWUI with no code
changes**. This is the recommended structure.

Point both clients at the same dataset name + password:

- **opencode:** `RAG_MEMORY_DATASET=andrew-memory`, `RAG_MEMORY_PASSWORD=<pw>`
- **OWUI:** set `MEMORY_DATASET_PREFIX` so the derived name matches, or
  run OWUI single-user with a fixed dataset.

### Provenance

Memories carry a `source` field so you can tell them apart:
- opencode → `"source": "opencode:memory"`
- OWUI → `"source": "openwebui:memory"`

Recall currently surfaces memories from both apps (no `source` filter at
search time). If cross-app recall becomes noisy, a server-side search
filter on `source` is a small follow-up — not needed today.

---

## 5. Operations

### Deleting memories

There is **no `delete_memory` tool in v1**. To remove a memory:
- **REST:** `DELETE /api/datasets/{name}/documents/{doc_id}` (find the
  `doc_id` via `GET /api/datasets/{name}/documents`).
- To correct a wrong memory, write a new corrected one and tell the user
  the old one is superseded (dedup won't catch it because the text
  differs).

### Tuning dedup aggressiveness

`RAG_DEDUP_THRESHOLD` (env var on the MCP/API container, default `0.995`):
- **Raise to 0.998** — more conservative, keeps more distinct facts
  (good for a curated memory store).
- **Lower to 0.97** — collapses more aggressively, smaller store (risk:
  merges distinct facts that happen to be semantically close).

### Capping memory size (tokens)

Every memory — session histories *and* LLM-curated notes — is split into
documents of at most **`MEMORY_MAX_TOKENS`** tokens each (MCP container env
var, default `8192`), mirroring dataset-side text splitting. The split uses
the bundled tokenizer (`RAG_TOKENIZER_PATH`; falls back to ~4 chars/token).
The **header is prepended to every chunk**, so each split carries the
session/provenance info (`session_id` in the payload, header block in the
text). Split memories are marked with `memory_chunks` / `chunk_index` /
`chunk_total` / `memory_truncated` payload fields.

Set the budget via `extraEnv` in the helm chart (e.g. `extraEnv:
{MEMORY_MAX_TOKENS: "8192"}`). Keep it at or below the embedder's context
window (the image bundles an 8192-token embedding model) so each chunk
embeds fully.

**Sessions are replaced in place.** When a `session_history` memory is
re-written (a session grows and is re-flushed), `add_memory` first deletes
the previous chunks for that session, then stores the new ones — so the
store keeps **one current history per session** instead of accumulating
copies every time the session is restarted.

### Rotating the OWUI `MEMORY_SECRET`

Changing `MEMORY_SECRET` re-derives all per-user passwords. Existing
datasets (hashed with the old derived passwords) become inaccessible.
To rotate: re-create each user's dataset, or update each dataset's
password via the REST API.

For opencode, "rotation" = changing the dataset's password via the REST
API and updating `RAG_MEMORY_PASSWORD` in users' env.

### Confirming it's working

- **opencode:** `opencode mcp list` shows both connections; watch the
  tool-call stream for `rag-memory_search_memory` / `rag-memory_add_memory`.
  The session-history plugin writes `[session-memory] …` diagnostics to
  `~/.local/share/opencode/log/session-memory.log` (override the directory
  with `SESSION_MEMORY_LOG_DIR`) and stores records with
  `kind: session_history`. The plugin never writes to the terminal — TUI
  output from plugins would overlap the display.
- **OWUI:** check the filter logs for `Memory recall: N hit(s)` and
  `Memory stored in 'owui-memory-…': …`. Recall injects a system
  message; writes happen silently in the outlet.
- **Server:** `kubectl logs -l app=rag-mcp-server -c rag-mcp-server`
  shows tool invocations; `GET /api/datasets/{name}` returns
  `document_count` to confirm writes are landing.
- **Multi-replica:** in stateless mode (v1.3.0+), each request logs as a
  fresh transport (no "Created new transport with session ID" line).
  Unlock state is shared via Redis, not in-memory — verify with
  `kubectl exec deploy/rag-mcp-server-redis -- redis-cli KEYS '*unlock*'`.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| opencode: `ToolError: No memory dataset specified` | `RAG_MEMORY_DATASET` not exported | Export it in the shell that launches opencode |
| opencode: no `[session-memory]` lines in `session-memory.log` | Plugin not loaded | Plugins load at startup — quit and restart opencode; confirm `documentation/opencode-memory/plugins/session-memory-logger.ts` (or the global copy) exists |
| opencode: `Incorrect password for dataset` | `RAG_MEMORY_PASSWORD` wrong / stale | Re-verify against the dataset's password |
| opencode: `rag-memory` connection not listed | URL unreachable / ingress token missing | Check `RAG_INGRESS_TOKEN` and the URL; try `opencode mcp debug rag-memory` |
| opencode: `Session not found` (404) on MCP calls | Multi-replica deployment running stateful MCP mode (pre-v1.3.0) | Upgrade to v1.3.0+ which enables `stateless_http=True`; see [SCALE.md](SCALE.md) |
| Both servers exit at startup: `MEDIA_TOKEN_SECRET is required` | The legacy `?password=` media URLs were removed; the shared HMAC secret isn't set | Set `security.mediaTokenSecret` in the helm chart (shared by both containers) before deploying |
| OWUI: no memories recalled for a new user | Dataset doesn't exist yet | Set `MEMORY_AUTO_CREATE=true`, or pre-create the user's dataset |
| OWUI: `Memory store failed` in logs | Derived password ≠ dataset's password (e.g. `MEMORY_SECRET` changed) | Re-create the dataset or restore the old secret |
| OWUI: distillation never writes | `DISTILL_LLM_*` not set, or replies < `DISTILL_MIN_REPLY_CHARS` | Configure distillation LLM; lower the char threshold if needed |
| Both: recall feels noisy | Paraphrastic dups accumulating | Raise `RAG_DEDUP_THRESHOLD` toward 0.998 |

---

## 6. Reference — file map

| File | What's in it |
|---|---|
| `src/multimodal_rag/mcp_server.py` | `add_memory` / `search_memory` tools, `_MemoryHeaderMiddleware`, `_run_retrieval` |
| `openwebui_extension/filter.py` | OWUI inlet (recall) + outlet (distil/store), per-user HMAC, auto-create |
| `documentation/opencode-memory/plugins/session-memory-logger.ts` | Template — auto-writes a structured per-session history (`kind: session_history`) to the memory dataset |
| `documentation/opencode-memory/plugins/memory-provenance.ts` | Template — attaches git + session provenance to every `add_memory` call |
| `documentation/opencode-memory/tools/session-id.ts` | Template — `session-id` custom tool |
| `documentation/opencode.jsonc` | Two-entry MCP config template for opencode |
| `documentation/AGENTS.md` | opencode proactive recall/write policy |
| `documentation/MCP.md` | MCP tool reference + all connection configs |
| `documentation/DEPLOYMENT.md` | Build, helm chart, verify, troubleshoot |
| `documentation/FEATURES.md` | Deep technical reference (formats, embedding, MCP server internals) |
| `documentation/MEMORY.md` | This file — per-app memory setup + operations |
| `openwebui_extension/README.md` | OWUI filter valves, per-user isolation, setup |
