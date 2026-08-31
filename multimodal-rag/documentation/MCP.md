# Configuring MCP Access

How to connect any MCP-compatible client (opencode, Claude Desktop,
Open WebUI, etc.) to the Multimodal RAG server's MCP sidecar.

> **See also:** [MEMORY.md](MEMORY.md) for long-term memory configuration
> (which uses a specialised two-connection pattern on top of the basics
> here), [DEPLOYMENT.md](DEPLOYMENT.md) for cluster deployment,
> [FEATURES.md](FEATURES.md) for the deep technical reference.

---

## 1. Server endpoint

When `mcp.enabled=true` (default in the helm chart), the MCP server runs
as a sidecar container in the same pod as the API server. Transport is
**`streamable-http`** on port `9090` (default) at path `/mcp`.

| Access method | URL |
|---|---|
| Via cluster ingress (production) | `https://rag-mcp-server.<your-domain>/mcp` |
| Via `kubectl port-forward` (local) | `http://localhost:8001/mcp` (after `kubectl port-forward deployment/rag-mcp-server 8001:9090`) |

The MCP container shares the `/data` PVC with the API server, so
`file://` paths in staged uploads are directly readable by the MCP
tools. See [DEPLOYMENT.md](DEPLOYMENT.md) § Architecture for the pod
layout.

> **`MEDIA_TOKEN_SECRET` is required.** The server refuses to start without
> it. Converted media URLs always carry a short-lived HMAC `?token=` (never
> the dataset password), which the API server verifies when serving files.

> **Multi-replica deployments (helm-scale-medium / helm-scale-large):** the MCP
> server runs `stateless_http=True` + `json_response=True` so any pod
> can handle any request — no in-memory session state. Note that the MCP
> unlock cache is **per-process** (an in-process dict, unlike the REST
> unlock cache which is Redis-backed when `redis.enabled=true`): with
> `stateless_http=True` any pod can handle any request, but an MCP
> `unlock_dataset` on pod A is not visible to pod B — callers should pass
> `password=` per tool call on multi-replica deployments. This is
> required for horizontal scaling; the default stateful mode would
> return `404 Session not found` when the K8s Service load-balances a
> request to a different pod than the one that initialized the session.
> See [SCALE.md](SCALE.md) for the full scale-chart architecture.

> **API-key auth does not apply to the MCP server.** `RAG_API_KEY`
> (`security.apiKey`) is enforced only by the REST API server's middleware.
> The MCP transport has no API-key middleware — MCP clients (opencode, DSH
> session-memory plugins, other MCP hosts) need nothing beyond whatever the
> gateway requires; dataset protection comes from the per-tool
> `password=` / unlock flow. The REST-side auth semantics are documented in
> `API.md`.

If the cluster ingress uses `oauth2-proxy` (EZUA), include a bearer
token in the `Authorization` header:

```json
"headers": { "Authorization": "Bearer <token>" }
```

---

## 2. Available tools (9)

| Tool | Purpose | Needs `dataset_name`? | Needs `password`? |
|------|---------|----------------------|-------------------|
| `list_datasets()` | List all datasets with metadata | — | — |
| `unlock_dataset(dataset_name, password, ttl)` | Verify a dataset password; cached per-process (default 30 min) — pass `password=` per call on multi-replica deployments | yes | yes |
| `search_dataset(dataset_name, query, image?, video?, audio?, top_k?, use_reranker?, reranker_top_k?, base_llm_modalities?, password?, media_base_url?)` | Full multimodal retrieval with post-processing | yes | if protected |
| `get_dataset_files(dataset_name, file_path?, limit?, offset?, password?)` | List or retrieve files in a dataset | yes | if protected |
| `get_dataset_info(dataset_name, password?)` | Dataset metadata | yes | if protected |
| `describe_media(media_url, query?, media_type?)` | Standalone VLM description of an image/video (no dataset needed) | — | — |
| `transcribe_audio(audio_url, max_seconds?)` | Standalone ASR transcription (no dataset needed) | — | — |
| `add_memory(text, image?, video?, audio?, metadata?, dataset_name?, password?)` | Store a memory into the caller's memory dataset | optional¹ | optional¹ |
| `search_memory(query, image?, video?, audio?, top_k?, use_reranker?, reranker_top_k?, base_llm_modalities?, dataset_name?, password?)` | Recall from the caller's memory dataset | optional¹ | optional¹ |

¹ The memory tools resolve `dataset_name` / `password` from request
headers (`X-Memory-Dataset` / `X-Dataset-Password`) or the
`MEMORY_DATASET` env var when omitted — see [MEMORY.md](MEMORY.md).

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
- **`context`** — formatted text ready for LLM consumption (unsupported
  media is auto-described by VLM/ASR when `base_llm_modalities` doesn't
  include that modality)
- **`results`** — raw result array with scores and content

### Modality conversion

If the calling LLM doesn't support a modality (set via
`base_llm_modalities`), retrieved media is automatically converted to
text:
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

All 9 tools are exposed on every connection. The client (or its
`tools` config) can disable specific tools it doesn't want the model to
see.

### 3.2 opencode (two-connection pattern for memory isolation)

opencode connects **twice** to the same URL, splitting memory tools
from knowledge tools so the memory password only rides requests to the
memory connection:

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
    "rag-memory_list_datasets": false,
    "rag-memory_get_dataset_files": false,
    "rag-memory_get_dataset_info": false,
    "rag-memory_unlock_dataset": false,
    "rag-memory_describe_media": false,
    "rag-memory_transcribe_audio": false,
    "rag-knowledge_add_memory": false,
    "rag-knowledge_search_memory": false
  }
}
```

The full template is at [`opencode.jsonc`](opencode.jsonc). The agent
policy (when to recall / write) is in [`AGENTS.md`](AGENTS.md). See
[MEMORY.md](MEMORY.md) § 3 for the complete opencode setup guide.

> **Verify after connecting:** run `opencode mcp list` — both
> `rag-memory` and `rag-knowledge` should appear. Confirm the prefixed
> tool names match the `tools` globs above.

### 3.3 Open WebUI

OWUI does **not** use MCP for memory — the filter handles recall/write
via the RAG REST API directly (see [MEMORY.md](MEMORY.md) § 4). To let
OWUI search knowledge datasets via MCP, attach the MCP server to the
model in **Admin Panel → Models → (your model) → Connections / Tools**.

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

Requires local access to the embedder/reranker/VLM/ASR endpoints (set
via `MODEL_*_URL` env vars). No GPU needed locally — models stay remote.

---

## 4. Query-vector caching

`search_dataset` / `search_memory` never re-embed the same query media
twice:
1. If the query media is already in the target dataset, its stored
   Qdrant vector is reused (zero model calls).
2. Otherwise, the file hash + embedder model + query text form a cache
   key in an in-process LRU. Same file in a later turn — cache hit.

Audio queries are auto-transcribed via ASR before embedding (the
embedder doesn't support audio natively); transcripts are cached too.
