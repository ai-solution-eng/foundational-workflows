# Multimodal RAG

End-to-end multimodal retrieval-augmented generation: ingest documents in 17+ formats (text, PDF, images, video, audio, code, tables, office docs, and more), embed them into a joint multimodal vector space, and retrieve at query time with optional cross-encoder reranking — all exposed via a REST API, an HTML frontend, and an MCP server.

[Video Demonstration](https://storage.googleapis.com/ai-solution-engineering-videos/public/MultimodalRag.mkv) with chapters and subtitles. Highlights models, dataset ingestion, open webui integration, and the opencode longterm memory implementation.

<div align="center"><img src="./documentation/rag_system_flow-1.png" width="700" alt="RAG system flow: dataset building (left) feeding a shared vector store, queried by query-time retrieval (right), with dynamic batching annotations throughout"></div>

---

## Features

- **Joint multimodal embedding** (text, image, video) via Qwen3-VL-Embedding-8B — search with any combination of modalities
- **Dual-embedding "twins"** — PDFs get a text-only twin so text queries match; images/videos/audio get a caption twin (media + caption) so caption wording is searchable alongside the raw-media embedding; unsupported media degrades to caption-only or is skipped
- **Contextual retrieval (opt-in per dataset)** — ingest-time LLM context injection: one small call per text chunk writes 1–2 sentences of document-level context prepended before embedding (with cost preview + fail-open degradation; affects new ingests — Recreate adopts existing content)
- **Audio support** via ASR transcription (Cohere Transcribe) — audio is converted to text before embedding
- **17+ file formats** with format-specific chunking: PDF (page-by-page + image extraction), images, video (overlapping segments), audio, text/markdown, JSON, XML, YAML, CSV/Excel, code (16 languages), HTML, Office docs, Jupyter notebooks, EPUB, log files, archives
- **Cross-encoder reranking** via Qwen3-VL-Reranker-8B for improved precision at the cost of latency
- **Modality conversion** — retrieved media the LLM doesn't support is auto-converted (images/video → VLM description, audio → ASR transcript)
- **Dataset management** — password-protected datasets, per-dataset Qdrant collections, dedup (cosine ≥ 0.995), S3/HTTP URL ingestion with sync reconciliation
- **Watched S3 sources** (opt-in) — a reconciler CronJob keeps datasets in sync with S3/MinIO prefixes on a schedule: new/changed objects ingest, objects deleted upstream are pruned, unchanged objects are skipped via a per-dataset ETag/Size state (no re-download)
- **MCP server** — 19 tools (search, federated search, list, document management add/delete/replace, self-service dataset selection + ★ memory binding, recall + memory management, describe media, transcribe audio) over streamable-http / stdio / sse
- **Long-term memory** — per-user LLM-curated memory store for opencode and Open WebUI, with SSO-backed isolation
- **Open WebUI extension** — filter that routes unsupported modalities to the RAG MCP tool, plus inlet/outlet memory hooks
- **Helm chart** — 2-container pod (API + MCP sidecar), Qdrant StatefulSet, Istio/EZUA ingress with oauth2-proxy

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Pod                                                 │
│                                                      │
│  ┌──────────────────┐   ┌──────────────────┐        │
│  │  API Server       │   │  MCP Server       │       │
│  │  port 8000        │   │  port 9090        │       │
│  │  (REST + Web UI)  │   │  (MCP tools)      │       │
│  └──────┬───────────┘   └──────┬───────────┘        │
│         │                      │                     │
│         └──────────┬───────────┘                     │
│                    │                                 │
│             ┌──────┴──────┐                          │
│             │   PVC /data  │  ← datasets + files    │
│             └─────────────┘                          │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴────────┐
              │  Qdrant          │  ← StatefulSet
              │  (port 6333)     │
              └──────────────────┘
```

Both the API server and MCP server connect to the same Qdrant instance and share the same PVC, so datasets created through the web UI are immediately searchable via MCP tools and vice versa.

---

## Documentation

| Document | What it covers |
|---|---|
| **[USAGE.md](USAGE.md)** | HTML frontend usage + programmatic Python API |
| **[documentation/API.md](documentation/API.md)** | REST API reference with `curl`/Python examples — create datasets, add/delete files |
| **[documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md)** | Build the image, install the helm chart, verify, troubleshoot |
| **[documentation/MCP.md](documentation/MCP.md)** | All 19 MCP tools + connection configs for opencode, Claude Desktop, OWUI, stdio |
| **[documentation/MEMORY.md](documentation/MEMORY.md)** | Long-term memory setup per client (opencode + Open WebUI), multi-user isolation, operations |
| **[documentation/FEATURES.md](documentation/FEATURES.md)** | Deep technical reference: every format, chunking strategy, embedding, reranking, storage |
| **[documentation/AGENTS.md](documentation/AGENTS.md)** | opencode agent policy: when to recall / write memories |
| **[documentation/opencode.jsonc](documentation/opencode.jsonc)** | opencode MCP config template (two-connection memory pattern) |
| **[documentation/DEVELOPMENT_NOTES.md](documentation/DEVELOPMENT_NOTES.md)** | Embedding/reranker validation, pipeline benchmarks, model setup debugging |
| **[openwebui_extension/README.md](openwebui_extension/README.md)** | Open WebUI filter: media routing, memory valves, per-user HMAC isolation |

---

## Quick start

> **PCAI is a Helm wrapper — you never run `helm` or `kubectl`.** Import the packaged chart into PCAI once, then drive the deployment by setting the chart's **`values.yaml`** in the PCAI *Helm Values*
> editor (or via the PCAI API). Every `--set` in the upstream docs maps 1:1 to a key in `values.yaml`.

To deploy on PCAI:
1. Pick the chart variant you need and import it into PCAI. a. There are 3 variants: `helm/` (single replica), `helm-scale-medium/`, `helm-scale-large/`. b. Scale variants use multiple API and Qdrant replicas to improve throughput. Requests are still routed jointly (for text) to a single request to improve performance.
2. Set the model endpoints (deployed through MLIS) as `models.*` values: a. The embedder (`models.embedder.url`) is the only required endpoint. b. A VLM/ASR model is often recommended to give images/videos or audios (including video-embedded audio) respectively to the base LLM. c. A reranker can be helpful as well, but often the LLM will just call `top_k` with sufficient performance. I have once
   seen it fail to retrieve with only `top_k`; increase the value to 100 with reranking and it succeeds.
3. **Required secrets — front and center.** The chart ships NO key material (P0-7): set `security.existingSecret` to a Secret you own carrying BOTH `RAG_API_KEY` and `MEDIA_TOKEN_SECRET` (the containers refuse to start without the media secret) —
   ```bash
   kubectl -n <ns> create secret generic rag-platform-keys \
     --from-literal=RAG_API_KEY="$(openssl rand -hex 16)" \
     --from-literal=MEDIA_TOKEN_SECRET="$(openssl rand -hex 32)"
   ```
   — or leave the `security` block empty and let the chart auto-generate both keys into `<deployment.name>-model-keys` on first install (retrieve the API key with `kubectl -n <ns> get secret <deployment.name>-model-keys -o jsonpath='{.data.RAG_API_KEY}' | base64 -d`). Inline `security.apiKey` / `security.mediaTokenSecret` still work as back-compat. Model API tokens go to `modelSecrets.*ApiKey` (values → the model-keys Secret). Precedence chain and the optional rotation runbook: `helm/ROTATION.md`.

The image does not bundle any ML models — it connects to remote model endpoints (embedder, reranker, VLM, ASR) configured at runtime via the chart's values. See `documentation/DEPLOYMENT.md` for details.

---

## Security hardening

The charts ship auth **on by default** (`security.apiKey` / `mediaTokenSecret` are wired to a Secret — see the Quick start) and the deployment sits behind an ingress auth proxy (Istio + oauth2-proxy). Since D20 (2026-09-24), an unconfigured deployment is **fail-closed**: unauthenticated callers bind an anonymous identity whose only grant is the deployment's memory dataset — dataset paths, `/api/admin/*` and dataset creation answer 403. Setting any key below (or the MCP keyset) restores the normal full-access resolution. Per-process env vars, or first-class Helm `security` values, control the rest:

| Env var | Purpose | Default |
|---|---|---|
| `RAG_API_KEY` | Require `Authorization: Bearer <key>` (or `X-RAG-Api-Key`) on all `/api/*` routes. Exempt: health/probes, the HTML pages (the served page embeds the key so the browser UI keeps working), dataset media serving, and staged media. Chart wiring: `security.apiKey` / `security.existingSecret` — the chart generates a key on first install when nothing is set; direct/scripted callers must send the header, and the Open WebUI filter takes it via its `RAG_API_KEY` valve. **With NO key configured anywhere the D20 fail-closed posture applies (anonymous = memory-dataset-only).** | charts: auto-generated (auth on) |
| `RAG_API_KEYS` (or fleet-universal `MCP_API_KEYS`) | **MCP endpoints** (streamable-http/SSE): comma-separated key list; when EITHER var is set, every `/mcp` request needs `X-API-Key` or `Bearer`. OPTIONAL per fleet decision 2026-09 — unset → MCP runs open (loud startup warning; REST-side D20 fail-closed still applies). Rotation: append the new key, move clients, drop the old — env re-read per request, no restart. Chart wiring: `mcp.apiKey.existingSecret` (empty default = not wired). | unset → MCP open |
| `MEDIA_TOKEN_SECRET` | **Required.** Secret shared by API + MCP; returned media URLs carry short-lived HMAC `?token=` (expiry `MEDIA_TOKEN_TTL`) — the legacy clear `?password=` suffix was removed. Both servers refuse to start without it. | unset → startup refused |
| `INGEST_ALLOW_HOSTS` | Comma-separated host allowlist for `/batch-urls` http(s) ingestion (`.example.com` matches subdomains). | unset (all hosts) |
| `INGEST_BLOCK_PRIVATE_HOSTS` | Reject http(s) URLs (ingest **and query-time media**) that resolve to private/link-local ranges (query-time still allows loopback — clients pass the server's own media URLs back). Every fetch — ingest downloads and media refs, redirect hops included — connects to the IP the policy validated (DNS-rebinding pin: check-time DNS = fetch-time DNS; a redirect into private/metadata space is refused). When an HTTP(S) proxy env is set the pin is skipped (the proxy performs egress DNS — the check-time denylist still ran). | `true` |
| `MAX_REMOTE_DOWNLOAD_BYTES` | Cap per remote/S3 download (streamed, aborted past this). | 536870912 |
| `ARCHIVE_MAX_TOTAL_BYTES` / `ARCHIVE_MAX_MEMBER_BYTES` / `ARCHIVE_MAX_ENTRIES` | Zip/tar/rar unpacked-size caps (incl. nested archives). | 2 GiB / 1 GiB / 10000 |
| `MEDIA_ALLOW_PATH_PREFIXES` | Allowlist of `file://` prefixes the MCP `describe_media`/`transcribe_audio`/audio-query tools may read (`:`-separated). | `DATA_PATH/datasets:DATA_PATH/staging` (fail-closed when unset) |
| `PW_MAX_FAILURES` / `PW_FAIL_WINDOW` | Password-failure throttle (returns 429 per identity). | 10 / 300 s |
| `RAG_MCP_SHARED_UNLOCK` | Opt OUT of per-caller unlock scoping (fleet decision D10): one shared unlock cache + throttle bucket for every caller. Only for gateway-fronted SINGLE-USER deployments. Unset (default) = per-caller: provided identity → `X-Forwarded-For` chain → socket peer, so one caller's `unlock_dataset` never opens a dataset for other callers. | unset (per-caller) |
| `MODEL_HEALTH_INTERVAL` / `MODEL_HEALTH_FAIL_THRESHOLD` | Background embedder probe: interval in seconds, and consecutive failures before a warning is logged. The result surfaces in `/api/admin/health` and the manage page — `/healthz` and `/readyz` deliberately do **not** gate on the embedder (a remote-model outage is not fixed by restarting this pod). | 60 / 3 |
| `CONFIG_DIR` | `:`-separated dirs of mounted ConfigMap/Secret files (one file per env key). When set, model config is **live-reloaded** on file change — no rollout needed (charts mount `-config` and `-model-keys` at `/etc/rag/config:/etc/rag/secrets`). The new embedder is verified before swap; an unreachable one is rejected and the old config is kept. | unset (env-only, rollout required) |
| `QUERY_EMB_CACHE_MAX`, `FILE_HASH_CACHE_MAX`, `ASR_TRANSCRIPT_CACHE_MAX`, `UNLOCK_CACHE_MAX` | Bounded sizes for the in-process caches (query vectors are stored packed — `array('f')`). | 4096 / 4096 / 512 / 4096 |
| `RAG_TRUST_PROXY_IDENTITY` | Trust the auth proxy's identity headers (`X-Auth-Request-*`/`X-Email`/`X-User`) — and, when no identity header is present, the `X-Forwarded-For` chain — as the per-caller unlock-cache/throttle identity (decision D10). Keep **off** unless an enforcing auth proxy overwrites these headers on every request — otherwise clients can spoof them to hijack unlocks or rotate identities past the throttle. Direct (no proxy) deployments don't need it: the socket peer is the identity. | `false` (socket peer; charts default `true`) |
| `RAG_METRICS_AUTH` | Require an API key on `/metrics` (accepted: `X-RAG-Api-Key` / `X-API-Key` / `Authorization: Bearer`, validated against `RAG_API_KEY` + the MCP key set). For in-cluster scrapers pair with the chart's `metrics.serviceMonitorBearerSecret`. | `false` (public in-cluster exposition, unchanged) |
| `QDRANT_CLIENT_TIMEOUT` | Hard timeout (s) for sync Qdrant calls, so a hung Qdrant cannot pin `sync_pool`/`qdrant-io` threads forever. | unset (no timeout; charts set 30) |
| `QDRANT_POOL_SIZE` / `MEDIA_POOL_SIZE` | Dedicated thread-pool sizes for Qdrant I/O (batched searches, upserts) and ffmpeg/media work — kept off the event loop and out of the default executor. Size `QDRANT_POOL_SIZE` comfortably above the concurrent searches a single process serves — it is the Qdrant-side bottleneck under saturation. | 4 / 2 (charts wire 16) |
| `EMBEDDING_QUERY_IDLE_WAIT_MS` (+ `EMBEDDING_QUERY_IDLE_MAX_BATCH`) | Opt-in idle early-flush for the embedding query batchers: a **small, stalled** queue (≤ `IDLE_MAX_BATCH`, default 2) flushes after `IDLE_WAIT_MS` instead of the full `EMBEDDING_QUERY_BATCH_WAIT_MS` window, cutting interactive-search latency ~4x. A growing queue (any burst) always waits for the window/cap — burst batching is structurally untouched. **Default 0 (disabled)** — the unguarded v1 (10ms, no batch guard) was benchmark-convicted: it shattered batches at load and collapsed throughput. Enable only after a benchmark pass. | 0 (disabled) |
| `RAG_EMBED_BATCH_URL` | Optional URL of a shared (cross-process) embedding query batcher: text-only queries are POSTed there instead of the per-process local batcher, so batch size is independent of worker/pod count. Falls back to local batching when unreachable. | unset (local batching) |
| `MODEL_EMBED_MAX_CONCURRENCY` | Per-event-loop bound on concurrent multimodal embedding POSTs (one request per converted doc; concurrent ingests multiply this). `0` disables the bound. | 32 |
| `RAG_DEFER_COUNT_SYNC` | **Decision D12 (Wave-4, default flipped ON).** `list_datasets`/`get_dataset` skip the per-request Qdrant count sync — previously N sequential round-trips (plus a meta.json write) on the hottest endpoint per list call. Counts are still maintained incrementally on every ingest/delete, so they are exact for in-band changes; they may lag by the defer window after out-of-band point changes (direct Qdrant writes or another replica). Set `false` to restore live counting (exact counts, N sequential round-trips per call). The scale charts already pinned `true`. | `true` (deferred; `false` = live counting) |
| `RAG_INGEST_CONCURRENCY` | Wave-4: how many files inside one batch ingest (`add_files_batch`) are preprocessed concurrently (store/copy/hash/classify/PDF/image/video/audio extraction) before embedding. Document order, batch composition and per-file results are deterministic regardless of completion order (an ordered sequencer absorbs each file's output). `1` restores the strictly sequential pre-Wave-4 behaviour. | 4 |
| `RAG_CONTEXTUAL_DEFAULT` | **Contextual retrieval (feature: DECISIONS 2026-09) — create-time default, disabled by default.** When `true`, NEW datasets are created with ingest-time contextual retrieval on: one small LLM call per real-text chunk writes 1–2 sentences of document-level context, prepended as a `[Document context]:` line before embedding (better chunk retrieval for large documents; the line also enters the BM25 lane and the stored payload, visibly labeled). An explicit `contextual` in the `POST /api/datasets` body always wins; per-dataset flip later via `PATCH /api/datasets/{name}`. Affects NEW ingests only — Recreate re-contextualizes existing files (content-hash dedup would skip a plain re-upload). Preview the cost first: `POST /api/admin/datasets/{name}/contextual-preview` (labeled estimate: point count ×2 for hybrid collections, token math, configured VLM). No-op without a VLM model. Chart: `rag.contextual` (gated rendering — a disabled deployment renders no config key). | `false` |
| `RAG_CONTEXTUAL_CONCURRENCY` | Max in-flight context LLM calls during a contextualized ingest (per-loop semaphore, the `MODEL_EMBED_MAX_CONCURRENCY` pattern). `0` disables the bound. Only read when a dataset has contextual retrieval enabled. | 8 |
| `RAG_CONTEXTUAL_DIGEST_CHARS` | Size of the document excerpt (chars) in the shared prompt preamble — stable per document so the serving engine's prefix cache absorbs it across a document's chunks. | 512 |
| `RAG_RERANK_MEDIA_LITE` | Wave-4 media-lite rerank: the reranker scores candidates on their text/caption representation — the candidate pool is fetched WITHOUT the heavy tier-3 base64 `image`/`video` payloads, and the full payloads are back-filled onto the surviving top_k docs only (one point-retrieve after scoring). Combined with the rerank over-fetch (`max(top_k, 4 × reranker_top_k)` candidates scored instead of `top_k`; output size unchanged). Set `false` to restore full-media scoring (every fetched candidate carries its base64 to the reranker — pre-Wave-4, much heavier transfer for media-heavy datasets). | `true` (media-lite) |
| `RAG_WEBHOOK_URL` (+ `RAG_WEBHOOK_SECRET` / `RAG_WEBHOOK_TIMEOUT`) | **Ingest webhooks (opt-in, Wave-5).** When set, every completed ingest (REST or MCP: raw documents, single file, batch files, URL/S3 batches) POSTs one small JSON event to the URL — `{"dataset", "doc_count", "status", "timestamp"}` with header `X-RAG-Webhook-Secret` when `RAG_WEBHOOK_SECRET` is set. The POST is timeout-capped (`RAG_WEBHOOK_TIMEOUT`, default 5.0 s, clamped ≥ 0.1) and **failures are logged, never fatal** — a dead receiver cannot fail an ingest that already succeeded. Dataset restore/import replays are muted (one logical ingest, not one event per batch). Unset (default) = zero behaviour: no request, no latency, no log line. For the chart, pass via `extraEnv` (the values are not secrets, but treat the webhook URL as semi-sensitive). | unset (off) |
| `RAG_API_KEY_CLIENTS` (+ `RAG_DATASET_ACLS`) | **Decision D15 — multi-user API keys → dataset ACLs (explicitly OPT-IN; enabling changes the auth model).** `RAG_API_KEY_CLIENTS="name:key;name:key"` (parsed like `K8S_MCP_CLIENTS`; keys must not contain `:` or `;`) mints per-user keys accepted by BOTH the REST and MCP surfaces; `RAG_DATASET_ACLS="name:ds1,ds2;name2:*"` binds each name to its datasets (`*` = all). A registry key matching NO ACL entry gets **NO datasets (fail-closed)**; dataset access (list/read/search/unlock/manage) is enforced per identity on both surfaces, per-key unlock caches and password-throttle buckets ride the D10 per-identity machinery (`key:<name>`), federated search restricts its fan-out to granted datasets, and ACL'd keys cannot reach `/api/admin/*` nor create datasets (unless granted `*`). The plain deployment keys — `RAG_API_KEY` plus the MCP key set (`MCP_API_KEYS` / `RAG_API_KEYS`) — keep FULL (admin) access. **Default (registry unset): today's single-key behaviour, byte-identical.** Env re-read per request (rotation without restart). Chart wiring: `mcp.apiKeyClients` / `mcp.datasetAcls` — rendered ONLY when set; the values are key material, pass them from a Secret pipeline. | unset (single-key, unchanged) |
| `RAG_ACCESS_STORE` (+ `RAG_ACCESS_DENY_SELECT` / `RAG_MEMORY_DEFAULT`) | **Decision D16 — self-service dataset selection + per-identity access store (OPT-IN; requires D15 registry keys to be useful).** `RAG_ACCESS_STORE=1` enables the /access page's checkbox model: registry keys SELECT their own datasets — public ones freely, protected ones only with the correct password, which is then saved per identity (`{DATA_PATH}/access/<name>.json`, 0600, shared PVC) so every REST/MCP call works password-free; `list_datasets` shows only a key's EFFECTIVE datasets (grants ∪ selections — access isolation, no name discovery). Effective access = operator ACL ∪ selections (the ACL is a floor). `RAG_ACCESS_DENY_SELECT="ds1,ds2"` datasets refuse self-selection outright; `RAG_MEMORY_DEFAULT` is the deployment-wide fallback memory dataset (resolution order: arg → header → the caller's ★ binding → this → `MEMORY_DATASET`). MCP tools: `select_dataset` / `deselect_dataset` / `set_memory_dataset`. **Default (off): byte-identical D15 behaviour** — ACL'd names only, fail-closed, selection endpoints 409. Env re-read per request. Chart wiring: `security.accessStore` (chart default `true` — the base-chart render always carries the key; set `false` to opt out) / `security.accessDenySelect` / `security.memoryDefault` — the latter two rendered ONLY when set. | chart default: on (`accessStore: true`); raw env default off |
| `RAG_UNLOCK_MAX_TTL` | Upper bound for explicit unlock TTLs on BOTH surfaces (REST `POST /unlock`, MCP `unlock_dataset(ttl=…)`; default 86400 = the historical 24 h cap). The special value `0` opts the deployment into **no-expiry unlocks** (`ttl=0` lasts until an explicit `POST /lock` on REST / until process restart on MCP) — the /access page's "No expiry (0)" option becomes meaningful only then. Read per request (rotation without restart). Chart wiring: `rag.unlockMaxTtl` (always rendered; the chart default equals the built-in default, so behaviour is unchanged). | `86400` |

Some defaults have deliberately shifted from permissive to strict since v1.9 (`MEDIA_TOKEN_SECRET` now required, private-host ingest blocking on, media path allowlist fail-closed). `helm/`, `helm-scale-large/` and `helm-scale-medium/` ship a `security:` values block wired to these flags. In PCAI you set them in `values.yaml` (the *Helm Values* editor) — the charts ship no key material, so point at a Secret you own (see Quick start step 3) or let the chart auto-generate:

```yaml
# values.yaml
security:
  existingSecret: "rag-platform-keys"   # Secret carrying RAG_API_KEY + MEDIA_TOKEN_SECRET
  blockPrivateHosts: true
```

### MCP document-management tools (Wave-5)

The dataset **write** path is no longer REST-only — the MCP server exposes the same
operations as thin tools over the identical `DatasetManager` logic (same validation, the
same password gate, the same upload-history events, the same 404 semantics):

| Tool | REST twin | What it does |
|---|---|---|
| `dataset_add_documents(dataset_name, paths, texts?, password?)` | `POST /api/datasets/{name}/batch-urls` · `POST …/batch-files` · `POST …/documents` | Ingests local file paths, `http(s)://`/`s3://` URLs (S3 prefixes expanded per object) and/or raw text documents into an existing dataset. Local paths must sit inside `MEDIA_ALLOW_PATH_PREFIXES` (default `DATA_PATH/datasets` + `DATA_PATH/staging` — the SQL-export staging hook), so an MCP caller cannot ingest arbitrary server files and read them back via search. Returns the REST twins' result shapes (batch: `{"status","file_count","files":[…]}`; texts: `{"status","stored_ids","count"}`). |
| `dataset_delete_documents(dataset_name, doc_ids?, filter?, limit?, password?)` | `DELETE /api/datasets/{name}/documents/{doc_id}` | Deletes by point ID(s) or by `{"source_prefix": …}` filter (the S3-sync prune semantic, server-side `MatchPrefix` scroll, `limit`-capped). Returns `{"status","deleted":[…],"count"}` — the REST twin's shape extended to the batch. Exactly one of `doc_ids`/`filter`. |
| `dataset_replace_document(dataset_name, doc_id, path, password?)` | `POST …/files` + `DELETE …/documents/{id}` composed | Ingests the replacement FIRST (a failed ingest leaves the old document untouched — no data loss), then deletes the old point; a failed delete returns an honest `"status": "partial"` with both versions present. The old point ID must exist (no silent degrade into a plain add). |

All three are dataset-auth gated exactly like the read tools: the dataset must exist
(the REST 404 message as a `ToolError`), password-protected datasets need `password` (or
a cached `unlock_dataset`), and — when D15 is enabled — the caller's registry key must
have the dataset in its ACL.

### Multi-user API keys → dataset ACLs (D15 — opt-in)

```bash
# mint per-user keys + bind datasets (one env each, re-read per request)
RAG_API_KEY_CLIENTS="alice:key-a;bob:key-b"
RAG_DATASET_ACLS="alice:reports,notes;bob:*"

# helm (values rendered only when set; default render byte-identical)
helm upgrade ... --set-string mcp.apiKeyClients="alice:key-a;bob:key-b" \
                  --set-string mcp.datasetAcls="alice:reports,notes;bob:*"
```

Registry keys authenticate on BOTH surfaces and are ACL-enforced per request; the plain
`RAG_API_KEY` / `MCP_API_KEYS` keys keep full (admin) access; a key with no ACL entry
sees no datasets. See the `RAG_API_KEY_CLIENTS` row above for the full semantics.

> **Behind the PCAI LLM gateway (D19):** the gateway authenticates itself upstream with an
> admin `Authorization: Bearer` token and forwards the caller's delegated key via
> `X-API-Key` — the explicit `X-API-Key` wins (de-escalation-only), so per-user identity,
> ACL filtering and ★ memory binding work through the gateway unchanged.

> Keep deployment secret material (e.g. `helm*/local/values.*.yaml`, which contain model-serving API keys and site credentials) out of version control — `.gitignore` covers `helm*/local/*` except `README.md` and `values.example.yaml`.

### Watched S3 sources (opt-in)

A values-gated reconciler CronJob keeps datasets continuously synced with S3/MinIO prefixes:

```yaml
# values.yaml (all three charts; default renders NOTHING — deployments without S3 are untouched)
watchedSources:
  enabled: true
  schedule: "*/30 * * * *"     # cron schedule
  sources:
    - dataset: reports
      prefixes:
        - s3://mm-rag-drop/reports/
    - dataset: logs
      prefixes:
        - s3://mm-rag-drop/logs/
        - s3://mm-rag-drop/trace-exports/
```

Each tick, serially per source, the job POSTs the prefixes to
`POST /api/datasets/{dataset}/batch-urls` with `{"urls": [...], "sync": true}` (header
`X-RAG-Api-Key` from the model-keys Secret) and polls the job to completion — so the full
sync semantics apply: new/changed objects are ingested, objects deleted upstream are pruned
(points deleted, counter decremented), and **unchanged objects are skipped before any
download** via the per-dataset ETag/Size state (`files/.watched_state.json`): an object whose
ETag+Size still matches its last successful ingest costs zero I/O. Config is validated at
`helm template` time — every prefix must be an `s3://` directory URL (single-object and
http(s) URLs fail the render, not the 3 a.m. cron log); a `type:` field is reserved for
future sources. The bucket allowlist `INGEST_ALLOW_S3_BUCKETS` still applies server-side as
the multi-tenant guard (env-only — set via `extraEnv` in values when multi-tenant MinIO
makes it relevant), `concurrencyPolicy: Forbid` prevents pile-up, and one failing source
never kills the run. Delete the dataset (or Recreate it) and the state resets with it.

### Self-service dataset selection (D16 — opt-in)

The `/access` page's checkbox model — chart default ON (`security.accessStore: true`), raw-env default OFF. With `RAG_ACCESS_STORE=1` (chart:
`security.accessStore: true`), registry-key users manage their own dataset set: every key
SELECTS the datasets it needs (a listing shows only a key's effective datasets — isolation, no name discovery) — public datasets freely,
password-protected ones only with their correct password, which is then saved per identity
(`{DATA_PATH}/access/<name>.json`, 0600, on the shared PVC) so every REST/MCP call works
without sending it again. Effective access = **operator ACL ∪ selections** (the ACL is a
guaranteed floor; deselect removes only self-added widening). `security.accessDenySelect`
(`RAG_ACCESS_DENY_SELECT`) names datasets that can never be self-selected, and
`security.memoryDefault` (`RAG_MEMORY_DEFAULT`) is the deployment-wide fallback memory
dataset for callers without a personal ★ binding. MCP gains `select_dataset` /
`deselect_dataset` / `set_memory_dataset`; the memory resolution order becomes arg →
header → the caller's ★ binding → `RAG_MEMORY_DEFAULT` → `MEMORY_DATASET` — so an opencode
user needs only their API key: select your memory dataset once (password once), star it,
done.

```yaml
# values.yaml (all three charts; denylist/memoryDefault render only when set)
security:
  accessStore: true
  accessDenySelect: ""          # e.g. "hr-data,payroll" — never self-selectable
  memoryDefault: ""             # e.g. "team-knowledge" — fallback ★ for callers without one
rag:
  unlockMaxTtl: 86400           # cap for explicit unlock TTLs; 0 = allow no-expiry unlocks
```

Requires per-user keys to be useful (`mcp.apiKeyClients` — see D15 above). The store file
is a sibling of `datasets/` on the same PVC and is included in existing PVC backups;
dataset passwords in it are stored as-provided (same posture as the unlock caches).

**Minting keys from the page (admin key registry).** With the store on, an ADMIN key on
`/access` gets a "User keys" panel: mint a per-user key (shown once — copy it), set the
user's dataset grants (`*` = all), rotate, or revoke — against
`{DATA_PATH}/access/clients.json` (0600, atomic, cross-process-locked). Effective
registry = `RAG_API_KEY_CLIENTS` env ∪ overlay (env authoritative on conflicts; grants
union per name), and everything applies on the next request — no restart. Registry keys
can never reach the panel (the D15 admin-surface gate denies `/api/admin/*` to them).
