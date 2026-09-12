# Multimodal RAG

End-to-end multimodal retrieval-augmented generation for HPE Private Cloud AI (PCAI): ingest documents in 17+ formats — text, PDF, images, video, audio, code, tables, office docs, notebooks, archives — embed everything into one joint multimodal vector space, and retrieve with hybrid dense + BM25 fusion and optional cross-encoder reranking. The same pipeline is exposed through a REST API, an HTML frontend, and an MCP server (13 tools) that doubles as a per-user long-term memory store for opencode and Open WebUI — all deployed as a single PCAI Helm chart (2-container pod + Qdrant behind the Istio gateway with SSO).

[Video Demonstration](https://storage.googleapis.com/ai-solution-engineering-videos/public/MultimodalRag.mkv) with chapters and subtitles. Highlights models, dataset ingestion, Open WebUI integration, and the opencode long-term-memory implementation.

<div align="center"><img src="./documentation/rag_system_flow-1.png" width="700" alt="RAG system flow: dataset building (left) feeding a shared vector store, queried by query-time retrieval (right), with dynamic batching annotations throughout"></div>

---

## What problem(s) it solves

- **Searchable multimodal corpora behind one API.** Drop PDFs, decks, screenshots, screen recordings, call audio, notebooks, and code into a dataset and search all of it through one REST endpoint or one MCP tool — no per-format pipelines, no separate image/video/audio indexes.
- **Text, image, video, and audio in one vector space.** A single multimodal embedder (Qwen3-VL-Embedding-8B) embeds every modality jointly, so a text query surfaces the right video segment, slide image, or scanned page. Media the consumer can't handle is auto-converted at query time (images/video → VLM description, audio → ASR transcript).
- **Agents that remember.** The MCP server is also an LLM-curated long-term memory store — per-user isolation (SSO-backed identity), recall/write tools plus `delete_memory` / `list_memories` / `forget_session` management — wired into opencode and Open WebUI via a filter extension.
- **Deploys as a single PCAI chart.** No kubectl: import the packaged chart into PCAI once, then drive everything from the Helm Values editor. The chart wires the API + MCP containers, Qdrant, (scale charts) Redis and the shared embed-batcher, Istio ingress with oauth2-proxy SSO, PVCs, optional backups and Prometheus metrics.

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
              │  Qdrant          │  ← StatefulSet (scale charts: sharded)
              │  (port 6333)     │
              └──────────────────┘
```

Both the API server and MCP server connect to the same Qdrant instance and share the same PVC, so datasets created through the web UI are immediately searchable via MCP tools and vice versa. The containers hold no models: they call remote model endpoints (embedder, reranker, VLM, ASR) deployed through PCAI's model serving (MLIS), configured entirely through chart values.

## Features

**Ingestion**
- **17+ file formats** with format-specific chunking: PDF (page-by-page + image extraction), images, video (overlapping segments), audio, text/markdown, JSON, XML, YAML, CSV/Excel, code (16 languages), HTML, Office docs, Jupyter notebooks, EPUB, log files, archives
- **OCR fallback for scanned PDFs** (tesseract in-image; per-dataset flag, chart-configurable default) — scanned archives become text-searchable and BM25-indexable
- **S3/HTTP URL ingestion** with batch jobs, size caps, archive-bomb guards, and S3-prefix sync with pruning (`sync` reconciles deletions upstream)
- **Dual-embedding "twins"** — PDFs get a text-only twin so text queries match; images/videos/audio get a caption twin (media + caption) so caption wording is searchable alongside the raw-media embedding; unsupported media degrades to caption-only or is dropped and its stored file cleaned up
- **Audio support** via ASR transcription — audio (including video soundtracks) is transcribed before embedding

**Retrieval**
- **Joint multimodal embedding** (text, image, video) — search with any combination of modalities; text-only queries are dynamically batched (idle early-flush, shared embed-batcher on the scale chart)
- **Hybrid dense + BM25 (RRF fusion)** on new collections; results carry honest `score_kind` labels (`rrf` | `cosine` | `reranker`) plus true dense cosines recomputed from the stored vectors — no more RRF rank arithmetic mislabeled as a similarity
- **Metadata-filtered search** — `file_types`, `severities`, `source_prefix`, `date_from`/`date_to` filters applied server-side on every search surface (REST + MCP), payload indexes managed automatically
- **Federated multi-dataset search** — concurrent fan-out across datasets (or "all"), merged and deduplicated results, one optional rerank over the pool
- **Cross-encoder reranking** via Qwen3-VL-Reranker-8B for precision at the cost of latency
- **Modality conversion** — retrieved media the LLM doesn't support is auto-converted (images/video → VLM description, audio → ASR transcript)

**Interfaces**
- **REST API + web UI** — dataset create/manage, uploads, search, document download (Markdown/JSONL), backup export/import with re-embedding, live model-connection health, password-protected datasets
- **MCP server** — 13 tools (search, federated search, list, recall + memory management, describe media, transcribe audio) over streamable-http / stdio / sse, with health probes
- **Long-term memory** — per-user LLM-curated memory store for opencode and Open WebUI, SSO-backed isolation, session-history management
- **Open WebUI extension** — filter that routes unsupported modalities to the RAG MCP tools, plus inlet/outlet memory hooks

**Operations**
- **Prometheus `/metrics`** on the API container (request/latency per route, ingest throughput, Qdrant op latency, cache hit/miss) with an opt-in ServiceMonitor
- **Opt-in backup CronJob** to S3/MinIO with retention pruning; restore re-embeds from exported archives
- **Multi-replica correctness** — job progress mirrored to Redis so poll responses answer from any pod; cross-pod unlock cache; gRPC + int8-quantized Qdrant clients on the scale charts
- **Hardened defaults** — signed HMAC media tokens (mandatory), API-key auth on `/api/*`, SSRF guards on ingest and query-time media, password-gated destructive routes, brute-force throttling, read-only rootfs / dropped capabilities / no SA token

## Deployment on PCAI

PCAI (HPE Private Cloud AI, the Ezmeral Unified Analytics distribution) is a Kubernetes wrapper with a Helm-based catalog: users **never run `helm install` or `kubectl apply`**. The packaged chart is imported into PCAI once; from then on the deployment is driven by editing the chart's `values.yaml` in the PCAI **Helm Values** editor (or via the PCAI API) and applying. Every `helm --set a.b=c` from the upstream docs maps 1:1 to a values key, and PCAI substitutes `${DOMAIN_NAME}` before rendering. The application image tag ships with the chart — you never set it by hand.

### 1. Pick the chart variant

| Chart | Shape | Choose when |
|---|---|---|
| `helm/` | 1 API replica (+ MCP sidecar), single-replica Qdrant | Pilots and single-team use; smallest footprint |
| `helm-scale-medium/` | 2 API replicas × 2 workers, 2-shard Qdrant, Redis unlock cache | Small concurrent teams; drop-in test of the scale architecture |
| `helm-scale-large/` | 4 API replicas × 4 workers, 3-shard Qdrant, Redis, shared embed-batcher | Throughput deployments — many simultaneous searches/ingests |

All three are versioned together (currently 3.6.3) and share the same top-level values; the scale charts add `app.*` batching/pool tuning, `qdrant.replicas` + `qdrant.client` (gRPC, int8 quantization), `redis`, and — on large only — the `embedBatcher` singleton that keeps embedding batch size independent of process count. Benchmark reference (v3.1.8 shape): 49.3 req/s @ N=100 and 72.8 req/s @ N=250 with 100% success on the large variant; see [documentation/BENCHMARKS.md](documentation/BENCHMARKS.md).

### 2. Required values

```yaml
# The one required model endpoint — without it the pod never passes readiness.
models:
  embedder:
    url: https://<embedder>.<project>.serving.<cluster-domain>   # MLIS endpoint

security:
  # REQUIRED: both the API and MCP containers refuse to start without it
  # (media URLs are served via short-lived HMAC tokens). Generate:
  #   python -c "import secrets; print(secrets.token_hex(32))"
  mediaTokenSecret: "<64 hex chars>"
  # The charts ship a chart-known default API key — replace it.
  apiKey: "<random string>"
```

That is genuinely all a working deployment needs. The image (`ghcr.io/ai-solution-eng/multimodal-rag-mcp`, tagged with the chart version) is packaged with the chart and bundles no models — it connects to remote endpoints configured at runtime.

### 3. Optional values

```yaml
# Additional model roles — leave url "" to disable a role.
models:
  reranker: { url: https://<reranker>... }   # precision ↑, latency ↑; often unnecessary (raise top_k instead)
  vlm:      { url: https://<vlm>... }        # captions images/videos at ingest, converts them at query time
  asr:      { url: https://<asr>... }        # transcribes audio + video soundtracks
modelSecrets:                                # → MODEL_*_API_KEY env (rendered into the -model-keys Secret)
  embedderApiKey: ""
  rerankerApiKey: ""
  vlmApiKey: ""
  asrApiKey: ""

persistence:                                 # size to the corpus; data should be RWX
  data:   { size: 50Gi, storageClass: gl4f-filesystem, accessMode: ReadWriteMany }
  qdrant: { size: 50Gi, storageClass: gl4f-filesystem, accessMode: ReadWriteMany }

ezua:                                        # PCAI ingress (charts default to this)
  virtualService:
    endpoint: rag-mcp-server.${DOMAIN_NAME}  # PCAI substitutes ${DOMAIN_NAME}
    timeout: 300s                            # normal API calls
    longTimeout: 3600s                       # batch uploads, SSE, MCP

resources:                                   # per-container requests/limits (app, qdrant, redis, embedBatcher)
  app: { requests: { memory: 2Gi, cpu: 2 }, limits: { memory: 8Gi, cpu: 4 } }

rag:                                         # pipeline defaults
  ocr: false                                 # default OCR flag for new datasets
  captionWithAsr: true                       # caption video audio tracks at ingest
  captionWithVlm: true                       # caption images/videos at ingest
  remote: false                              # false = ".serving." URLs rewritten to in-cluster .svc addresses

s3:                                          # ingest/export bucket credentials (→ S3_* env in the -model-keys Secret)
  endpointUrl: http://minio.minio.svc.cluster.local:9000
  accessKeyId: ""
  secretAccessKey: ""

metrics:  { serviceMonitor: false }          # Prometheus Operator ServiceMonitor
backups:  { enabled: false, schedule: "0 3 * * *", bucket: "", retentionDays: 0 }
extraEnv: {}                                 # any additional env var for both containers, e.g. MEMORY_MAX_TOKENS
```

How values become environment (verified in `helm/templates/`): `models.*.url/name/extra` → `MODEL_EMBEDDER_URL` & co. via the `-config` ConfigMap; `modelSecrets.*` → `MODEL_EMBEDDER_API_KEY` & co., `security.apiKey` → `RAG_API_KEY`, and `security.mediaTokenSecret` → `MEDIA_TOKEN_SECRET` via the `-model-keys` Secret; `security.*` guard knobs map to `INGEST_BLOCK_PRIVATE_HOSTS`, `INGEST_ALLOW_HOSTS`, `MAX_REMOTE_DOWNLOAD_BYTES`, `ARCHIVE_MAX_*`, `MEDIA_ALLOW_PATH_PREFIXES`, `MEDIA_TOKEN_TTL`, `PW_MAX_FAILURES`/`PW_FAIL_WINDOW`, `RAG_TRUST_PROXY_IDENTITY`; `rag.ocr` → `RAG_OCR_DEFAULT`, `rag.remote` → `RAG_REMOTE`; `ezua.virtualService.endpoint` → `MEDIA_BASE_URL` (signed media URLs point here). The scale charts add `app.*` → `SYNC_POOL_SIZE` / `MCP_POOL_SIZE` / `EMBEDDING_QUERY_*` / Qdrant batcher and pool vars, `qdrant.client.*` → `QDRANT_PREFER_GRPC` / `QDRANT_CLIENT_TIMEOUT` / `QDRANT_QUANTIZATION*`, and `modelPool.*` → `MODEL_POOL_MAX_CONNECTIONS` / `MODEL_POOL_MAX_KEEPALIVE_CONNECTIONS`. You normally never touch env vars — they are listed only to explain what a values key changes.

Ready-made, paste-ready values documents for both deployment targets live in each chart's [`values-examples/`](helm/values-examples/) folder.

### Deployment targets

**SE G2 (HPE internal cluster)** — cluster domain `pcai-se-ai-application.hst.rdlabs.hpecorp.net`, namespaces `project-user-<name>`, model endpoints under `https://<model>.project-user-<name>.serving.pcai-se-ai-application.hst.rdlabs.hpecorp.net`. The ezaf-gateway applies HPE SSO centrally, so the chart does not create its own AuthorizationPolicy (`ezua.authorizationPolicy.enabled: false`); keep `rag.remote: false` so `.serving.` URLs are rewritten to their in-cluster form. Real model JWTs and other secrets stay in `helm*/local/` (gitignored, hardlink-ignored, never packaged) — start from `helm/values-examples/values.g2.yaml` and copy the real credentials into your local file.

**Hosted trial (customer PCAI)** — use the `${DOMAIN_NAME}` placeholder in `ezua.virtualService.endpoint` (PCAI substitutes it), keep the oauth2-proxy AuthorizationPolicy enabled (`ezua.authorizationPolicy.enabled: true`, `providerName: oauth2-proxy`) and `security.trustProxyIdentity: true` so unlock-cache identity follows the authenticated user. Model endpoints come from the customer's own model-serving deployment. Note the chart installs a Kyverno vendor-label ClusterPolicy as a pre-install hook (`add-vendor-app-labels-<release>-<chart>`, labeling Pods/Deployments/Services with `hpe-ezua/*`); on locked-down customer clusters the admin may need to permit ClusterPolicy creation. Start from `helm/values-examples/values.hosted-trial.yaml`.

## Security

Security is configured through `security.*` values keys (rendered into a Kubernetes Secret plus ConfigMap); the env vars they produce are secondary detail. The core server is **unauthenticated by default** by design and is meant to sit behind the ingress auth proxy (Istio + oauth2-proxy).

| Values key | Env var | Purpose |
|---|---|---|
| `security.mediaTokenSecret` | `MEDIA_TOKEN_SECRET` | **Required.** Shared by API + MCP; media URLs carry short-lived HMAC `?token=` (TTL `mediaTokenTtl`) — the legacy clear `?password=` suffix was removed. Both containers refuse to start without it. |
| `security.apiKey` | `RAG_API_KEY` | Require `Authorization: Bearer <key>` (or `X-RAG-Api-Key`) on all `/api/*` routes. Exempt: health/probes, the HTML pages (the served page embeds the key so the browser UI keeps working), dataset media serving, staged media. MCP clients are unaffected. Charts ship a default — change it for real deployments. |
| `security.trustProxyIdentity` | `RAG_TRUST_PROXY_IDENTITY` | Trust `X-Auth-Request-*`/`X-Email`/`X-User` headers for unlock-cache scoping and password throttling. Keep on only behind an enforcing auth proxy (charts set `true`); otherwise clients can spoof these headers to hijack unlocks or rotate identities past the throttle. |
| `security.blockPrivateHosts` | `INGEST_BLOCK_PRIVATE_HOSTS` | Reject http(s) URLs (ingest **and** query-time media) resolving to private/link-local ranges (incl. cloud metadata); loopback stays allowed at query time so clients can pass the server's own media URLs back. Unresolved hosts fail closed. |
| `security.ingestAllowHosts` | `INGEST_ALLOW_HOSTS` | Comma-separated host allowlist for `/batch-urls` ingestion (`.example.com` matches subdomains); when set it is authoritative — hosts not listed are rejected, listed hosts are allowed even when private (how in-cluster MinIO endpoints are permitted). |
| `security.maxDownloadBytes` | `MAX_REMOTE_DOWNLOAD_BYTES` | Per-download cap for remote/S3 ingest (Content-Length pre-check + streamed abort). |
| `security.archiveMaxTotalBytes` / `archiveMaxMemberBytes` / `archiveMaxEntries` | `ARCHIVE_MAX_*` | Zip/tar/rar unpacked-size and entry-count caps (incl. nested archives), audited from headers before extraction. `0` disables a check. |
| `security.mediaAllowPathPrefixes` | `MEDIA_ALLOW_PATH_PREFIXES` | `:`-separated `file://` prefixes the MCP media tools may read (`describe_media`, `transcribe_audio`, audio queries) — realpath-resolved, fail-closed (`"*"` is the dev/test escape hatch). Media refs inside user documents are validated centrally too. |
| `security.pwMaxFailures` / `security.pwFailWindow` | `PW_MAX_FAILURES` / `PW_FAIL_WINDOW` | Password-failure throttle: max failures per identity within the sliding window before 429s. |
| `security.mediaTokenTtl` | `MEDIA_TOKEN_TTL` | Signed media-URL lifetime in seconds; short = safer, but links embedded in old LLM replies stop working sooner. |

Additional opt-in runtime knobs (set via `extraEnv`): `INGEST_ALLOW_S3_BUCKETS` (S3 bucket allowlist so ingest credentials can't be aimed at other tenants' buckets), `MAX_MEDIA_FETCH_BYTES`, `QDRANT_CLIENT_TIMEOUT` (charts set 30 s so a hung Qdrant can't pin worker threads), `QDRANT_POOL_SIZE` / `MEDIA_POOL_SIZE` (dedicated I/O pools), `CONFIG_DIR` (mounted-config live reload — the charts mount `-config` and `-model-keys` at `/etc/rag/config:/etc/rag/secrets`; a new embedder is verified before swap, an unreachable one is rejected), `MODEL_HEALTH_INTERVAL` / `MODEL_HEALTH_FAIL_THRESHOLD` (background embedder probe surfaced in `/api/admin/health` and readiness — `/healthz` deliberately does not gate on it), `RAG_HYBRID_SEARCH` / `RAG_BM25_K1` / `RAG_BM25_B` / `RAG_HYBRID_EMBEDDING_SCORES` (hybrid-search knobs), `OCR_LANG` / `OCR_DPI` / `OCR_TIMEOUT_S`, and the bounded-cache caps (`QUERY_EMB_CACHE_MAX`, `FILE_HASH_CACHE_MAX`, `ASR_TRANSCRIPT_CACHE_MAX`, `UNLOCK_CACHE_MAX`, `RAG_CACHE_MAX`).

Some defaults deliberately shifted from permissive to strict since v1.9: `MEDIA_TOKEN_SECRET` is required, private-host ingest blocking is on, media path reads are fail-closed, and destructive routes (delete/recreate/migrate/import-overwrite) require the dataset password.

## Documentation

| Document | What it covers |
|---|---|
| **[USAGE.md](USAGE.md)** | HTML frontend usage + programmatic Python API |
| **[documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md)** | Chart variants, values reference, PCAI deployment walkthrough |
| **[documentation/VERIFICATION.md](documentation/VERIFICATION.md)** | Post-deployment verification checklist |
| **[documentation/FEATURES.md](documentation/FEATURES.md)** | Deep technical reference: every format, chunking strategy, embedding, reranking, storage |
| **[documentation/BENCHMARKS.md](documentation/BENCHMARKS.md)** | Throughput/latency benchmarks per chart variant |
| **[documentation/memory/](documentation/memory/)** | Long-term memory: overview README + per-client setup docs (opencode, Open WebUI, DSH) |
| **[openwebui_extension/README.md](openwebui_extension/README.md)** | Open WebUI filter: media routing, memory valves, per-user HMAC isolation |
