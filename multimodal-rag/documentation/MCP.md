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

> **OIDC JWT as a second credential (decision D21 — opt-in via `OIDC_ENABLED=true`).** When the deployment enables OIDC, a JWT-only MCP client sends its Keycloak/UA-realm token as the bearer — the server verifies it (RS256 against the realm JWKS; `iss` / `aud` / `exp`) and resolves it to the **same per-user registry identity** as the user's minted API key (join key = `preferred_username`, fallback `sub`). A JWT-only user starts with **zero datasets (fail-closed)** and self-selects via the D16 tools above — no API key is needed at all. Note the tokens rotate (15-day validity on G2): the client should re-read the notebook secret per session, not bake a token into the config.

> **Dataset ownership on delete (decision D23).** The D23 ownership rule applies to the MCP delete tools identically: deleting a dataset (or its documents) requires the dataset's `created_by` identity or an admin — same owner-or-admin rule as the REST DELETE endpoint, enforced on both surfaces; pre-D23 datasets (no recorded creator) are admin-only. See API.md § 1 and § 4.3.

```jsonc
{
  "mcpServers": {
    "multimodal-rag": {
      "url": "https://rag-mcp-server.<YOUR-DOMAIN>/mcp",
      "headers": {
        "Authorization": "Bearer {env:RAG_INGRESS_TOKEN}"
      }
    }
  }
}
```

> **Unlock TTL bounds (`RAG_UNLOCK_MAX_TTL`, default `86400`).** `unlock_dataset(ttl=…)` is clamped to 60..86400 seconds by default — the deployment may set `RAG_UNLOCK_MAX_TTL` to a different cap. The special value `0` opts the deployment into **no-expiry unlocks**: `ttl=0` caches the unlock without a deadline (until the MCP process restarts — the in-memory cache also evicts under its bounded-entry guard). Read per request (rotation without restart). The same knob bounds the REST `POST /api/datasets/{name}/unlock` (where `ttl=0` lasts until an explicit `/lock`); the `/access` page ([API.md](API.md) § 1) surfaces the opt-in as its "No expiry (0)" TTL option.

If the cluster ingress uses `oauth2-proxy` (EZUA), include a bearer token in the `Authorization` header:

```json
"headers": { "Authorization": "Bearer <token>" }
```

---

## 2. Available tools (25)

| Tool | Purpose | Needs `dataset_name`? | Needs `password`? |
|------|---------|----------------------|-------------------|
| `list_datasets(cursor=None, limit=None)` | List the datasets available to you with document counts and access status — ALWAYS call this FIRST when you don't know a dataset's exact name. | — | — |
| `unlock_dataset(dataset_name, password, ttl=1800)` | Unlock a password-protected dataset for the current session: other tools (search_dataset, get_dataset_files, get_dataset_info) then accept requests without a password parameter for the TTL duration. | yes | yes |
| `select_dataset(dataset_name, password=None)` | Add a dataset to YOUR key's accessible set: a protected dataset's CORRECT password is verified and saved server-side — other tools need no password argument. | yes | if protected |
| `deselect_dataset(dataset_name)` | Remove a dataset from YOUR key's accessible set. | yes | — |
| `create_short_term_memory(session_id, ttl_seconds=None, name=None)` | Create a short-term-memory dataset for one agent session: a NON-GUESSABLE name is minted and HIDDEN from the catalog — KEEP the returned name, the only handle. | no | never |
| `end_short_term_memory(dataset_name, password=None)` | Explicitly tear down (delete) a short-term-memory dataset. | no | — |
| `set_memory_dataset(dataset_name=None)` | Mark which dataset is YOUR long-term memory store (the /access page's ★). | no | — |
| `search_dataset(dataset_name, query='', image=None, video=None, audio=None, top_k=10, use_reranker=False, reranker_top_k=3, base_llm_modalities=None, password=None, media_base_url=None, file_types=None, severities=None, source_prefix=None, date_from=None, date_to=None, dense_weight=None, sparse_weight=None, k=None, agentic=False, agentic_max_rounds=2)` | Search ONE multimodal RAG dataset by MEANING: images/videos match what they show (captions + image↔text embeddings — text queries DO find pictures), audio its transcript, documents their text. | yes | if protected |
| `search_datasets(datasets, query='', image=None, video=None, audio=None, top_k=5, use_reranker=False, reranker_top_k=3, base_llm_modalities=None, file_types=None, severities=None, source_prefix=None, date_from=None, date_to=None)` | Search SEVERAL datasets with ONE query and merge the ranked results. | no | never |
| `set_late_interaction(dataset_name, enabled, token_dim=None, password=None)` | Per-dataset opt-in to multi-vector late interaction (Feature D): MaxSim rerank over token-level vectors stored in a '<dataset>__late' sidecar collection. | yes | if protected |
| `memory_status()` | Report the calling key's memory-write capability. | no | — |
| `add_memory(text, image=None, video=None, audio=None, metadata=None, dataset_name=None, password=None)` | Store a memory (durable fact / decision / preference) for recall via search_memory; distil a concise, self-contained record. | optional¹ | optional¹ |
| `search_memory(query, image=None, video=None, audio=None, top_k=5, use_reranker=False, reranker_top_k=3, base_llm_modalities=None, dataset_name=None, password=None)` | Recall relevant memories from your personal long-term memory store. | optional¹ | optional¹ |
| `recall_playbooks(task, k=3, dataset_name=None, password=None)` | Recall stored playbooks (reusable procedures) for a task, ranked by empirical success rate. | optional¹ | optional¹ |
| `record_playbook_outcome(memory_id, outcome, note='', dataset_name=None, password=None)` | Record how a playbook actually performed: increment its success/fail counter and stamp last_validated. | optional¹ | optional¹ |
| `consolidate_memories(dry_run=True, limit=500, dataset_name=None, password=None)` | Plan (and only on request apply) memory consolidation: decay scores, archive set, contradictions. | optional¹ | optional¹ |
| `delete_memory(memory_ids, dataset_name=None, password=None)` | Delete memories from your personal long-term memory store by point ID — pass memory_id values from search_memory / list_memories. | optional¹ | optional¹ |
| `list_memories(limit=20, kind=None, tags=None, include_session_history=False, dataset_name=None, password=None)` | List stored memories — newest first, with memory_id (→ delete_memory), kind, timestamp, tags, preview; session histories hidden unless requested. | optional¹ | optional¹ |
| `forget_session(session_id, dataset_name=None, password=None)` | Delete the stored session-history memory for one session. | optional¹ | optional¹ |
| `get_dataset_files(dataset_name, file_path=None, limit=100, offset=0, password=None, return_bytes=False)` | Browse a dataset's files by NAME — paginated listing, or one file by path; to find files by CONTENT, use search_dataset. | yes | if protected |
| `get_dataset_info(dataset_name, password=None)` | Return metadata (including document count and password status) for a dataset. | yes | if protected |
| `dataset_add_documents(dataset_name, paths=None, texts=None, password=None)` | Add documents to a dataset: local file paths, http(s)/s3 URLs, or raw text — REST-ingest twin (same pipeline and download policy). | yes | if protected |
| `dataset_delete_documents(dataset_name, doc_ids=None, filter=None, limit=10000, password=None)` | Delete documents by point ID(s) or source-prefix filter — REST-delete twin. | yes | if protected |
| `dataset_replace_document(dataset_name, doc_id, path, password=None)` | Ingest the new path, then delete the old point — new content goes FIRST: a failed ingest leaves the old document untouched; a failed final delete leaves both versions ("status": "partial"). | yes | if protected |
| `inspect_media(media_url, query=None, media_type=None, max_seconds=None)` | Inspect ONE media file whose URL/path you already have: image/video are described/answered by the vision-language model (VLM), audio transcribed by ASR. | — | — |

> This table (and the tool count) is generated by `scripts/gen_mcp_docs.py` from the MCP
> server source — AST-only, no server import, so it cannot drift. Regenerate with
> `python scripts/gen_mcp_docs.py` after any `@mcp.tool()` signature or docstring change;
> do not edit the table by hand.

> **Read-only grants (D26):** when the caller's key carries mode ACLs (`RAG_DATASET_ACLS` token form `name:ds1:ro` or an overlay `{"name": …, "mode": "ro"}`), the write tools — `dataset_add_documents`, `dataset_delete_documents`, `dataset_replace_document`, `add_memory`, `delete_memory`, `forget_session`, `record_playbook_outcome`, `consolidate_memories` (only when applying), `set_late_interaction` — require an **rw** grant for the target dataset: an ro (or ro-only) dataset refuses with a clear `ToolError`; read tools are unchanged.

¹ The memory tools resolve `dataset_name` from the `X-Memory-Dataset` header → the caller's ★ binding → `RAG_MEMORY_DEFAULT` → `MEMORY_DATASET` (an explicit `dataset_name` argument is honoured only when nothing was bound — a disagreeing one is refused), and `password` from an explicit argument → the `X-Dataset-Password` header → the saved selection → `RAG_MEMORY_PASSWORD` — see [MEMORY.md](MEMORY.md).

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
- **images/video** → VLM (Qwen3.8-27B-FP8 by chart default) describes them → text
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

All 21 tools are exposed on every connection. The client (or its `tools` config) can disable specific tools it doesn't want the model to see.

### 3.2 opencode (unified single connection, 2026-10)

The historical two-connection pattern existed to keep the memory **password** out of knowledge-connection headers. Under D16/D21 that secret no longer rides any header — the per-identity **★ memory binding + saved selection** (from the /access page) resolves dataset and password server-side — so opencode connects **once** with one bearer:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["documentation/AGENTS.md"],
  "mcp": {
    "rag": {
      "type": "remote",
      "url": "https://rag-mcp-server.<YOUR-DOMAIN>/mcp",
      "headers": {
        // Keycloak OIDC access token (D21) → the SAME per-user identity as
        // your browser SSO; or a minted per-user registry key (D15/D17).
        // Rotate per session — tokens have bounded validity on G2.
        "Authorization": "Bearer {env:MM_RAG_BEARER_TOKEN}"
      }
    }
  }
}
```

The full template is at [`opencode.jsonc`](opencode.jsonc). The agent policy (when to recall / write) is in [`AGENTS.md`](AGENTS.md). See [MEMORY.md](MEMORY.md) § 2 for the one-time /access selection that replaces the env headers.

> **Legacy (key-unconfigured single-user) deployments** — or if you prefer explicit headers — the two-connection pattern still works byte-for-byte: `rag-memory` with `X-Memory-Dataset: {env:RAG_MEMORY_DATASET}` + `X-Dataset-Password: {env:RAG_MEMORY_PASSWORD}` headers, `rag-knowledge` with the bearer only, and the `tools:` disable-list splitting the namespaces (memory tools off on rag-knowledge, dataset tools off on rag-memory). Keep `Authorization` OUT of rag-memory's env headers in that mode only if the deployment has no keys configured; with keys/OIDC configured the memory connection ALSO needs the bearer (an unauthenticated write 401s — D20 fail-closed).
>
> **Verify after connecting:** run `opencode mcp list` — the connection should appear and tool calls should resolve. A 401 means the bearer is stale/absent; an "No memory dataset specified" ToolError means the identity has no ★ binding and no header supplied.

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

---

## 5. The Inspector tab (D27)

The unified frontend (`/`, `RAG_SSO_GATE`-gated) ships a fifth tab, **Inspector**, next to Datasets / Access / (Admin) / Stats: a browser playground for the MCP tool calls. Pick a tool from the catalog (fetched live via MCP `tools/list`), edit the JSON arguments (seeded from the tool's `inputSchema` defaults, with `dataset_name` prefilled from your first visible dataset), and run it — the result's text content blocks render directly, with the raw MCP result in a collapsible JSON view.

**No new trust path.** The browser never talks to the MCP endpoint itself — the sidecar is off the ingress path and page scripts never see an SSO-derived key (the D23 hard rule). Calls go through the API server, which forwards EXACTLY the credential envelope the request authenticated with, re-expressed in the headers the MCP middleware accepts:

| Session authenticated via | Forwarded to MCP as |
|---|---|
| `X-RAG-Api-Key` / `X-API-Key` (paste, embedded, delegated) | `X-API-Key` |
| `Authorization: Bearer <key>` | `Authorization: Bearer` |
| `X-Auth-Request-Access-Token` (proxy-forwarded JWT) | `X-Auth-Request-Access-Token` |
| SSO session cookie (realm access-token JWT) | `Authorization: Bearer <jwt>` |
| Trusted-proxy identity headers (D24) | nothing — no forwardable secret; a keyed sidecar replies 401 and the tab says so |

The sidecar then resolves identity itself from the forwarded credential against the SAME registry store (D15 ACLs ∪ D16 selections, ★ memory binding, per-caller unlock scoping) — the Inspector exercises the real MCP path, not a shortcut around it. Tool-level failures render as the tool's own error text.

**Env:** `RAG_INSPECTOR_MCP_URL` (default `http://127.0.0.1:9090/mcp` — the chart's sidecar topology; point it at `http://localhost:8001/mcp` for `kubectl port-forward` dev setups) and `RAG_INSPECTOR_TIMEOUT_S` (default 300 — tool calls embed and search). Read per request.

---

## 6. Rendering media URLs in a browser that cannot reach the app host

`MEDIA_BASE_URL` points at the PCAI application endpoint, and **that host sits behind
the platform oauth2-proxy SSO gate** (a `CUSTOM` `AuthorizationPolicy` on the
istio-ingressgateway — see `helm-scale-large/templates/authorizationpolicy.yaml`).
The gate runs *in front of* the app: every request on the host, including
`/api/datasets/{ds}/files/{f}?token=...`, is answered `302` to Keycloak before FastAPI
sees it. That breaks two things at once:

1. **The media token never gets evaluated.** The HMAC `?token=` exists precisely so
   `<img>`/`<video>`/`<audio>` tags — which cannot set headers — can fetch protected
   media. The proxy does not know that token, so a signed URL behaves exactly like an
   unsigned one: `302`.
2. **A live SSO session does not rescue it.** The gateway cookie is
   `Domain=pcai-se-ai-application…; SameSite=Lax`. `Lax` cookies are **not** sent on
   cross-site subresource loads, so an `<img>` embedded in a page on any other origin
   (a local GUI on `127.0.0.1:3080`, say) fires cookieless and is `302`d again.

A page served from the **same registrable domain** (Open WebUI on
`open-webui.pcai-se-ai-application…`) is same-site, so the cookie rides along and media
renders there. Anything on a different origin does not.

### `scripts/media-proxy.py`

A localhost shim for exactly that case. It fetches bytes through the MCP gateway —
which authenticates with a **bearer key**, not a cookie — and re-serves them from a
local origin the browser loads without any SSO involvement:

```bash
python3 scripts/media-proxy.py --port 8788     # key from PCAI_LLM_API_KEY or ~/.dsh/.env
```

Then swap the origin of any RAG media URL:

```text
https://rag-mcp-server.<domain>/api/datasets/andrew-personal-dataset/files/DSC01378_preprocessed.JPG
                          ↓
http://127.0.0.1:8788/api/datasets/andrew-personal-dataset/files/DSC01378_preprocessed.JPG
```

| Behaviour | Detail |
|---|---|
| Accepted paths | `/api/datasets/{ds}/files/{f}` (the real route) and `/{ds}/{f}` (shorthand) |
| Filenames | both the stored `{32-hex}_name` and the short human name resolve (uuid prefix stripped) |
| `?token=` | accepted and ignored — the proxy re-authorises through the gateway key, so stale/absent tokens are fine |
| Caching | in-memory plus a disk cache (default `~/.dsh/tmp-rag-media-proxy`, falling back to `.scratch/`) |
| Status codes | `200` ok, `404` no such file, `413` file exceeds `MCP_FILE_BYTES_MAX`, `502` gateway error |
| Bind | `127.0.0.1` only; reads nothing beyond what the configured gateway key may already read; exposes no filesystem path |

> **Size limit — the one real constraint.** `get_dataset_files(return_bytes=True)` refuses files over
> `MCP_FILE_BYTES_MAX` (default **256 KiB**), so the proxy can serve most preprocessed
> image previews and small documents but not full-size photos, video, or audio. That is a
> transport cap, not a permission problem, and the proxy answers `413` to say so. To serve
> bulk media, raise `MCP_FILE_BYTES_MAX` on the deployment (`extraEnv` in the chart values;
> note it is *not* currently a first-class chart value) — or serve those files over REST
> from a same-site origin instead.

### Why not just fix the deployment?

Three alternatives, in increasing order of blast radius:

1. **Same-origin front end** — render the media from a page on the app's own domain
   (Open WebUI already works this way). No config change; nothing to bypass.
2. **`oauth2-proxy --skip-auth-route`** for `/api/datasets/.*/files/.*` — lets the app's
   HMAC media token be the only gate, which is the design `api_server.py` already
   documents. This is a gateway-policy change on a shared PCAI unit: coordinate it.
3. **Server-side proxy with credentials** — have the consuming front end fetch media
   through an authenticated backend and re-serve it from its own origin (what
   `scripts/media-proxy.py` does, locally).

The proxy is option 3 in its smallest form: no cluster change, no shared-policy edit, and
it reuses the credential the MCP client already holds.

