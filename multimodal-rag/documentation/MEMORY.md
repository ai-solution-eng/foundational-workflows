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
Qdrant collection per user). Memories are **LLM-curated** — the model
(or a separate distillation LLM) decides what's durable enough to
store, not raw transcripts.

| | opencode | Open WebUI |
|---|---|---|
| **Write trigger** | Model calls `add_memory` MCP tool (proactive, per `AGENTS.md`) | `outlet()` filter asks a distillation LLM after each reply |
| **Recall trigger** | Model calls `search_memory` MCP tool (proactive, per `AGENTS.md`) | `inlet()` filter auto-searches at conversation start |
| **Transport** | MCP (streamable-http) | REST API (direct HTTP from filter) |
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
- **OWUI:** check the filter logs for `Memory recall: N hit(s)` and
  `Memory stored in 'owui-memory-…': …`. Recall injects a system
  message; writes happen silently in the outlet.
- **Server:** `kubectl logs -l app=rag-mcp-server -c rag-mcp-server`
  shows tool invocations; `GET /api/datasets/{name}` returns
  `document_count` to confirm writes are landing.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| opencode: `ToolError: No memory dataset specified` | `RAG_MEMORY_DATASET` not exported | Export it in the shell that launches opencode |
| opencode: `Incorrect password for dataset` | `RAG_MEMORY_PASSWORD` wrong / stale | Re-verify against the dataset's password |
| opencode: `rag-memory` connection not listed | URL unreachable / ingress token missing | Check `RAG_INGRESS_TOKEN` and the URL; try `opencode mcp debug rag-memory` |
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
| `documentation/opencode.jsonc` | Two-entry MCP config template for opencode |
| `documentation/AGENTS.md` | opencode proactive recall/write policy |
| `documentation/MCP.md` | MCP tool reference + all connection configs |
| `documentation/DEPLOYMENT.md` | Build, helm chart, verify, troubleshoot |
| `documentation/FEATURES.md` | Deep technical reference (formats, embedding, MCP server internals) |
| `documentation/MEMORY.md` | This file — per-app memory setup + operations |
| `openwebui_extension/README.md` | OWUI filter valves, per-user isolation, setup |
