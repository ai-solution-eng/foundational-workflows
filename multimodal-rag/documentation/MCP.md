# Configuring MCP Access

How to connect any MCP-compatible client (opencode, Claude Desktop, Open WebUI, etc.) to the Multimodal RAG server's MCP sidecar.

> **See also:** [MEMORY.md](MEMORY.md) for long-term memory configuration (which uses a specialised two-connection pattern on top of the basics here), [DEPLOYMENT.md](DEPLOYMENT.md) for cluster
> deployment, [FEATURES.md](FEATURES.md) for the deep technical reference.

---

## 1. Server endpoint

When `mcp.enabled=true` (default in the helm chart), the MCP server runs as a sidecar container in the same pod as the API server. Transport is **`streamable-http`** on port `9090` (default) at path `/mcp`.

| Access method | URL |
|---|---|
| Via cluster ingress (production) | `https://rag-mcp-server.<your-domain>/mcp` |
| Via `kubectl port-forward` (local) | `http://localhost:8001/mcp` (after `kubectl port-forward deployment/rag-mcp-server 8001:9090`) |

The MCP container shares the `/data` PVC with the API server, so `file://` paths in staged uploads are directly readable by the MCP tools. See [DEPLOYMENT.md](DEPLOYMENT.md) § Architecture for the pod layout.

> **`MEDIA_TOKEN_SECRET` is required.** The server refuses to start without it. Converted media URLs always carry a short-lived HMAC `?token=` (never the dataset password), which the API server verifies
> when serving files.

> The streamable-http transport always runs `stateless_http=True` + `json_response=True`, so any pod can handle any request — no in-memory session state.
> Note that the MCP unlock cache is **per-process** (an in-process dict, unlike the REST unlock cache which is Redis-backed when `redis.enabled=true`): an MCP `unlock_dataset` on pod A is not visible to
> pod B — callers should pass `password=` per tool call on multi-replica deployments (helm-scale-medium / helm-scale-large). See [SCALE.md](SCALE.md) for the full scale-chart architecture.

> **API-key auth on MCP (opt-in).** The MCP transport has its own middleware, separate from the REST one: if `RAG_API_KEYS` (or the fleet-universal `MCP_API_KEYS`) is set, every `/mcp` request must carry `X-API-Key` or `Authorization: Bearer <key>`; when neither is set the MCP server runs open with a loud startup warning (local-development mode). Keys are re-read per request, so rotation needs no restart. The REST `RAG_API_KEY` is *not* accepted here by default — MCP clients use the MCP key envs (or a D15 registry key, below; with the registry configured, `RAG_API_KEY` is honored as an admin key). Dataset protection still comes from the per-tool `password=` / unlock flow on top. The REST-side auth semantics are documented in `API.md`.

> **Multi-user keys → dataset ACLs (decision D15 — opt-in).** Setting `RAG_API_KEY_CLIENTS="name:key;name:key"` mints per-user keys accepted on **both** the MCP and REST surfaces, and `RAG_DATASET_ACLS="name:ds1,ds2;name2:*"` binds each name to its datasets (`*` = all). Access is **fail-closed** (a registry key with no ACL entry sees no datasets), while the plain deployment keys (`RAG_API_KEY` + `MCP_API_KEYS` / `RAG_API_KEYS`) keep full admin access. Default (registry unset): single-key behaviour, unchanged. See [API.md](API.md) § 1 for the REST side.

> **Self-service dataset selection (decision D16 — opt-in via `RAG_ACCESS_STORE=1`).** With the access store enabled, registry-key users grow their own dataset set: `select_dataset(dataset_name, password?)` adds a dataset to the key's set (protected datasets demand the correct password, which is then saved server-side — every tool afterwards works with **no** `password` argument), `deselect_dataset` removes it, and `set_memory_dataset` binds the key's ★ memory dataset so `add_memory` / `search_memory` need neither `dataset_name` nor `password`. Effective access = operator ACL ∪ selections (the ACL is a floor; `RAG_ACCESS_DENY_SELECT` datasets refuse selection outright). `list_datasets` shows only the key's EFFECTIVE datasets (operator grants ∪ selections) — access isolation: a listing never shows names the key cannot use (ratified 2026-09-24, reversing the earlier discovery-mode flip). See [API.md](API.md) § 1 for the REST endpoints and the `/access` page.

> **Unlock TTL bounds (`RAG_UNLOCK_MAX_TTL`, default `86400`).** `unlock_dataset(ttl=…)` is clamped to 60..86400 seconds by default — the deployment may set `RAG_UNLOCK_MAX_TTL` to a different cap. The special value `0` opts the deployment into **no-expiry unlocks**: `ttl=0` caches the unlock without a deadline (until the MCP process restarts — the in-memory cache also evicts under its bounded-entry guard). Read per request (rotation without restart). The same knob bounds the REST `POST /api/datasets/{name}/unlock` (where `ttl=0` lasts until an explicit `/lock`); the `/access` page ([API.md](API.md) § 1) surfaces the opt-in as its "No expiry (0)" TTL option.

If the cluster ingress uses `oauth2-proxy` (EZUA), include a bearer token in the `Authorization` header:

```json
"headers": { "Authorization": "Bearer <token>" }
```

---

## 2. Available tools (19)

| Tool | Purpose | Needs `dataset_name`? | Needs `password`? |
|------|---------|----------------------|-------------------|
| `list_datasets()` | List the caller's datasets with metadata — only the key's EFFECTIVE datasets (operator grants ∪ selections; access isolation) | — | — |
| `unlock_dataset(dataset_name, password, ttl)` | Verify a dataset password; cached per-process (default 30 min; `ttl` bounded by `RAG_UNLOCK_MAX_TTL`, `0` = no expiry when the deployment opts in) — pass `password=` per call on multi-replica deployments, or select the dataset once instead (below) | yes | yes |
| `select_dataset(dataset_name, password?)` | **D16** (opt-in): add a dataset to YOUR key's set — protected ones require the correct password, saved server-side so no tool needs `password=` afterwards | yes | if protected |
| `deselect_dataset(dataset_name)` | **D16**: remove one of YOUR selections (and its saved password); operator-ACL grants untouched | yes | — |
| `set_memory_dataset(dataset_name?)` | **D16**: bind your ★ memory dataset (omit the arg to clear) — `add_memory`/`search_memory` then need no `dataset_name`/`password` | no | — |
| `search_dataset(dataset_name, query, image?, video?, audio?, top_k?, use_reranker?, reranker_top_k?, base_llm_modalities?, password?, media_base_url?, file_types?, severities?, source_prefix?, date_from?, date_to?)` | Full multimodal retrieval with post-processing; optional metadata filters (`file_types`, `severities`, `source_prefix`, `date_from`/`date_to`) applied server-side before ranking | yes | if protected |
| `get_dataset_files(dataset_name, file_path?, limit?, offset?, password?)` | List or retrieve files in a dataset | yes | if protected |
| `get_dataset_info(dataset_name, password?)` | Dataset metadata | yes | if protected |
| `dataset_add_documents(dataset_name, paths?, texts?, password?)` | Add documents to a dataset — local file paths (inside the media allowlist), `http(s)://`/`s3://` URLs, or raw text; the MCP twin of the REST ingest endpoints, same pipeline and download policy | yes | if protected |
| `dataset_delete_documents(dataset_name, doc_ids?, filter?, limit?, password?)` | Delete documents **by explicit point ID(s)** or by `{"source_prefix": …}` filter (exactly one selector); the MCP twin of the REST delete-document endpoint, extended with batch IDs and prefix filter | yes | if protected |
| `dataset_replace_document(dataset_name, doc_id, path, password?)` | Replace one document: ingest the new `path` **first**, then delete the old point — a failed ingest leaves the old document untouched; a failed final delete reports `"status": "partial"` with both versions present | yes | if protected |
| `describe_media(media_url, query?, media_type?)` | Standalone VLM description of an image/video (no dataset needed) | — | — |
| `transcribe_audio(audio_url, max_seconds?)` | Standalone ASR transcription (no dataset needed) | — | — |
| `add_memory(text, image?, video?, audio?, metadata?, dataset_name?, password?)` | Store a memory into the caller's memory dataset | optional¹ | optional¹ |
| `search_memory(query, image?, video?, audio?, top_k?, use_reranker?, reranker_top_k?, base_llm_modalities?, dataset_name?, password?)` | Recall from the caller's memory dataset | optional¹ | optional¹ |
| `delete_memory(memory_ids, dataset_name?, password?)` | Delete memories **by explicit point ID** (from `search_memory`/`list_memories` results); reports a preview of each deleted memory and unknown ids. No similarity/query-directed deletion exists. | optional¹ | optional¹ |
| `list_memories(limit?, kind?, tags?, include_session_history?, dataset_name?, password?)` | List stored memories newest-first with `memory_id`, kind, timestamp, tags and a preview — the source of ids for `delete_memory` | optional¹ | optional¹ |
| `forget_session(session_id, dataset_name?, password?)` | Delete the `session_history` memory for one session (never touches curated memories) | optional¹ | optional¹ |
| `search_datasets(datasets \| "all", query, image?, video?, audio?, top_k?, use_reranker?, reranker_top_k?, base_llm_modalities?, file_types?, severities?, source_prefix?, date_from?, date_to?)` | **Federated search**: concurrent per-dataset fan-out, dataset-labelled merged results, dataset-qualified dedup, one optional rerank over the pool. Password-protected datasets without a cached unlock are skipped with a note — there is deliberately **no `password` parameter** | no | never |

¹ The memory tools resolve `dataset_name` / `password` from request headers (`X-Memory-Dataset` / `X-Dataset-Password`) or the `MEMORY_DATASET` env var when omitted — see [MEMORY.md](MEMORY.md).

### `search_dataset` example

```json
{
  "dataset_name": "my-dataset",
  "query": "aurora borealis over snowy mountains",
  "top_k": 10,
  "use_reranker": false,
  "reranker_top_k": 3,
  "base_llm_modalities": ["text"]
}
```

Returns JSON with:
- **`context`** — formatted text ready for LLM consumption (unsupported media is auto-described by VLM/ASR when `base_llm_modalities` doesn't include that modality)
- **`results`** — raw result array with scores and content

### Modality conversion

If the calling LLM doesn't support a modality (set via `base_llm_modalities`), retrieved media is automatically converted to text:
- **images/video** → VLM (Gemma 4 31B) describes them → text
- **audio** → ASR (Cohere Transcribe) transcribes → text

---

## 3. Connecting from an MCP client

### 3.1 Any remote client (streamable-http)

```json
{
  "mcpServers": {
    "multimodal-rag": {
      "url": "https://rag-mcp-server.your-domain.com/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

All 19 tools are exposed on every connection. The client (or its `tools` config) can disable specific tools it doesn't want the model to see.

### 3.2 opencode (two-connection pattern for memory isolation)

opencode connects **twice** to the same URL, splitting memory tools from knowledge tools so the memory password only rides requests to the memory connection:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["documentation/AGENTS.md"],
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
    "rag-knowledge": {
      "type": "remote",
      "url": "https://rag-mcp-server.<YOUR-DOMAIN>/mcp",
      "headers": {
        "Authorization": "Bearer {env:RAG_INGRESS_TOKEN}"
      }
    }
  },
  "tools": {
    "rag-memory_search_dataset": false,
    "rag-memory_search_datasets": false,
    "rag-memory_list_datasets": false,
    "rag-memory_get_dataset_files": false,
    "rag-memory_get_dataset_info": false,
    "rag-memory_unlock_dataset": false,
    "rag-memory_dataset_add_documents": false,
    "rag-memory_dataset_delete_documents": false,
    "rag-memory_dataset_replace_document": false,
    "rag-memory_describe_media": false,
    "rag-memory_transcribe_audio": false,
    "rag-knowledge_add_memory": false,
    "rag-knowledge_search_memory": false
  }
}
```

The full template is at [`opencode.jsonc`](opencode.jsonc). The agent policy (when to recall / write) is in [`AGENTS.md`](AGENTS.md). See [MEMORY.md](MEMORY.md) § 3 for the complete opencode setup guide.

> **Verify after connecting:** run `opencode mcp list` — both `rag-memory` and `rag-knowledge` should appear. Confirm the prefixed tool names match the `tools` globs above.

### 3.3 Open WebUI

OWUI does **not** use MCP for memory — the filter handles recall/write via the RAG REST API directly (see [MEMORY.md](MEMORY.md) § 4). To let OWUI search knowledge datasets via MCP, attach the MCP server to the model in **Admin Panel → Models → (your model) → Connections / Tools**.

### 3.4 stdio transport (local development)

```bash
python -m multimodal_rag.mcp_server --transport stdio
```

Client config:
```json
{
  "mcpServers": {
    "multimodal-rag": {
      "command": "python",
      "args": ["-m", "multimodal_rag.mcp_server", "--transport", "stdio",
               "--data-path", "/data", "--qdrant-host", "localhost"]
    }
  }
}
```

Requires local access to the embedder/reranker/VLM/ASR endpoints (set via `MODEL_*_URL` env vars). No GPU needed locally — models stay remote.

---

## 4. Query-vector caching

`search_dataset` / `search_memory` never re-embed the same query media twice:
1. If the query media is already in the target dataset, its stored Qdrant vector is reused (zero model calls).
2. Otherwise, the file hash + embedder model + query text form a cache key in an in-process LRU. Same file in a later turn — cache hit.

Audio queries are auto-transcribed via ASR before embedding (the embedder doesn't support audio natively); transcripts are cached too.
