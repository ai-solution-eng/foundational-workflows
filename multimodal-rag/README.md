# Multimodal RAG

End-to-end multimodal retrieval-augmented generation: ingest documents
in 17+ formats (text, PDF, images, video, audio, code, tables, office
docs, and more), embed them into a joint multimodal vector space, and
retrieve at query time with optional cross-encoder reranking — all
exposed via a REST API, an HTML frontend, and an MCP server.

[Video Demonstration](https://storage.googleapis.com/ai-solution-engineering-videos/public/MultimodalRag.mkv) with chapters and subtitles. Highlights models, dataset ingestion, open webui integration, and the opencode longterm memory implementation.

<div align="center"><img src="./documentation/rag_system_flow-1.png" width="700" alt="RAG system flow: dataset building (left) feeding a shared vector store, queried by query-time retrieval (right), with dynamic batching annotations throughout"></div>

---

## Features

- **Joint multimodal embedding** (text, image, video) via Qwen3-VL-Embedding-8B — search with any combination of modalities
- **Dual-embedding "twins"** — PDFs get a text-only twin so text queries match; images/videos/audio get a caption twin (media + caption) so caption wording is searchable alongside the raw-media embedding; unsupported media degrades to caption-only or is skipped
- **Audio support** via ASR transcription (Cohere Transcribe) — audio is converted to text before embedding
- **17+ file formats** with format-specific chunking: PDF (page-by-page + image extraction), images, video (overlapping segments), audio, text/markdown, JSON, XML, YAML, CSV/Excel, code (16 languages), HTML, Office docs, Jupyter notebooks, EPUB, log files, archives
- **Cross-encoder reranking** via Qwen3-VL-Reranker-8B for improved precision at the cost of latency
- **Modality conversion** — retrieved media the LLM doesn't support is auto-converted (images/video → VLM description, audio → ASR transcript)
- **Dataset management** — password-protected datasets, per-dataset Qdrant collections, dedup (cosine ≥ 0.995), S3/HTTP URL ingestion
- **MCP server** — 9 tools (search, list, recall, store memory, describe media, transcribe audio) over streamable-http / stdio / sse
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

Both the API server and MCP server connect to the same Qdrant instance
and share the same PVC, so datasets created through the web UI are
immediately searchable via MCP tools and vice versa.

---

## Documentation

| Document | What it covers |
|---|---|
| **[USAGE.md](USAGE.md)** | HTML frontend usage + programmatic Python API |
| **[documentation/API.md](documentation/API.md)** | REST API reference with `curl`/Python examples — create datasets, add/delete files |
| **[documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md)** | Build the image, install the helm chart, verify, troubleshoot |
| **[documentation/MCP.md](documentation/MCP.md)** | All 9 MCP tools + connection configs for opencode, Claude Desktop, OWUI, stdio |
| **[documentation/MEMORY.md](documentation/MEMORY.md)** | Long-term memory setup per client (opencode + Open WebUI), multi-user isolation, operations |
| **[documentation/FEATURES.md](documentation/FEATURES.md)** | Deep technical reference: every format, chunking strategy, embedding, reranking, storage |
| **[documentation/AGENTS.md](documentation/AGENTS.md)** | opencode agent policy: when to recall / write memories |
| **[documentation/opencode.jsonc](documentation/opencode.jsonc)** | opencode MCP config template (two-connection memory pattern) |
| **[documentation/DEVELOPMENT_NOTES.md](documentation/DEVELOPMENT_NOTES.md)** | Embedding/reranker validation, pipeline benchmarks, model setup debugging |
| **[openwebui_extension/README.md](openwebui_extension/README.md)** | Open WebUI filter: media routing, memory valves, per-user HMAC isolation |

---

## Quick start

> **PCAI is a Helm wrapper — you never run `helm` or `kubectl`.** Import the
> packaged chart into PCAI once, then drive the deployment by setting the
> chart's **`values.yaml`** in the PCAI *Helm Values* editor (or via the PCAI
> API). Every `--set` in the upstream docs maps 1:1 to a key in `values.yaml`.

To deploy on PCAI:
1. Pick the chart variant you need and import it into PCAI.
   a. There are 3 variants: `helm/` (single replica), `helm-scale-medium/`,
      `helm-scale-large/`.
   b. Scale variants use multiple API and Qdrant replicas to improve
      throughput. Requests are still routed jointly (for text) to a single
      request to improve performance.
2. Set the model endpoints (deployed through MLIS) as `models.*` values:
   a. The embedder (`models.embedder.url`) is the only required endpoint.
   b. A VLM/ASR model is often recommended to give images/videos or audios
      (including video-embedded audio) respectively to the base LLM.
   c. A reranker can be helpful as well, but often the LLM will just call
      `top_k` with sufficient performance. I have once seen it fail to
      retrieve with only `top_k`; increase the value to 100 with reranking
      and it succeeds.
3. Recommended: generate a `security.mediaTokenSecret` value with
   `python -c "import secrets; print(secrets.token_hex(32))"` and set it in
   `values.yaml`. There is a default one in the charts, but it is recommended
   to change it for security; it governs token generation for password
   protected datasets.

The image does not bundle any ML models — it connects to remote model
endpoints (embedder, reranker, VLM, ASR) configured at runtime via the chart's
values. See `documentation/DEPLOYMENT.md` for details.

---

## Security hardening (opt-in)

The core server is **unauthenticated by default** and is designed to sit
behind an ingress auth proxy (Istio + oauth2-proxy). For additional,
opt-in protection (per-process env vars, or first-class [Helm
`security` values]), set any of the following:

| Env var | Purpose | Default |
|---|---|---|
| `RAG_API_KEY` | Require `Authorization: Bearer <key>` (or `X-RAG-Api-Key`) on all `/api/*` routes. Exempt: health/probes, the HTML pages (the served page embeds the key so the browser UI keeps working), dataset media serving, and staged media. MCP clients are unaffected (the MCP server has no key middleware). The charts ship a default key — change it for real deployments; direct/scripted callers must send the header, and the Open WebUI filter takes it via its `RAG_API_KEY` valve. | charts: shipped default (auth on); unset = no auth |
| `MEDIA_TOKEN_SECRET` | **Required.** Secret shared by API + MCP; returned media URLs carry short-lived HMAC `?token=` (expiry `MEDIA_TOKEN_TTL`) — the legacy clear `?password=` suffix was removed. Both servers refuse to start without it. | unset → startup refused |
| `INGEST_ALLOW_HOSTS` | Comma-separated host allowlist for `/batch-urls` http(s) ingestion (`.example.com` matches subdomains). | unset (all hosts) |
| `INGEST_BLOCK_PRIVATE_HOSTS` | Reject http(s) URLs (ingest **and query-time media**) that resolve to private/link-local ranges (query-time still allows loopback — clients pass the server's own media URLs back). | `true` |
| `MAX_REMOTE_DOWNLOAD_BYTES` | Cap per remote/S3 download (streamed, aborted past this). | 536870912 |
| `ARCHIVE_MAX_TOTAL_BYTES` / `ARCHIVE_MAX_MEMBER_BYTES` / `ARCHIVE_MAX_ENTRIES` | Zip/tar/rar unpacked-size caps (incl. nested archives). | 2 GiB / 1 GiB / 10000 |
| `MEDIA_ALLOW_PATH_PREFIXES` | Allowlist of `file://` prefixes the MCP `describe_media`/`transcribe_audio`/audio-query tools may read (`:`-separated). | `DATA_PATH/datasets:DATA_PATH/staging` (fail-closed when unset) |
| `PW_MAX_FAILURES` / `PW_FAIL_WINDOW` | Password-failure throttle (returns 429 per identity). | 10 / 300 s |
| `MODEL_HEALTH_INTERVAL` / `MODEL_HEALTH_FAIL_THRESHOLD` | Background embedder probe: interval in seconds, and consecutive failures before a warning is logged. The result surfaces in `/api/admin/health` and the manage page — `/healthz` and `/readyz` deliberately do **not** gate on the embedder (a remote-model outage is not fixed by restarting this pod). | 60 / 3 |
| `CONFIG_DIR` | `:`-separated dirs of mounted ConfigMap/Secret files (one file per env key). When set, model config is **live-reloaded** on file change — no rollout needed (charts mount `-config` and `-model-keys` at `/etc/rag/config:/etc/rag/secrets`). The new embedder is verified before swap; an unreachable one is rejected and the old config is kept. | unset (env-only, rollout required) |
| `QUERY_EMB_CACHE_MAX`, `FILE_HASH_CACHE_MAX`, `ASR_TRANSCRIPT_CACHE_MAX`, `UNLOCK_CACHE_MAX` | Bounded sizes for the in-process caches (query vectors are stored packed — `array('f')`). | 4096 / 4096 / 512 / 4096 |
| `RAG_TRUST_PROXY_IDENTITY` | Trust `X-Auth-Request-*`/`X-Email`/`X-User` headers for unlock-cache scoping and password throttling. Keep **off** unless an enforcing auth proxy overwrites these headers on every request — otherwise clients can spoof them to hijack unlocks or rotate identities past the throttle. | `false` (socket peer) |
| `QDRANT_CLIENT_TIMEOUT` | Hard timeout (s) for sync Qdrant calls, so a hung Qdrant cannot pin `sync_pool`/`qdrant-io` threads forever. | unset (no timeout; charts set 30) |
| `QDRANT_POOL_SIZE` / `MEDIA_POOL_SIZE` | Dedicated thread-pool sizes for Qdrant I/O (batched searches, upserts) and ffmpeg/media work — kept off the event loop and out of the default executor. | 4 / 2 |
| `EMBEDDING_QUERY_IDLE_WAIT_MS` | Idle early-flush for the embedding query batchers: a batch flushes early when no new query arrives for this long, so an isolated interactive search doesn't wait out the full `EMBEDDING_QUERY_BATCH_WAIT_MS` window (bursts still coalesce). `0` restores the old full-window behavior. | 10 |
| `RAG_EMBED_BATCH_URL` | Optional URL of a shared (cross-process) embedding query batcher: text-only queries are POSTed there instead of the per-process local batcher, so batch size is independent of worker/pod count. Falls back to local batching when unreachable. | unset (local batching) |
| `MODEL_EMBED_MAX_CONCURRENCY` | Per-event-loop bound on concurrent multimodal embedding POSTs (one request per converted doc; concurrent ingests multiply this). `0` disables the bound. | 32 |

Some defaults have deliberately shifted from permissive to strict since v1.9 (`MEDIA_TOKEN_SECRET` now required, private-host ingest blocking on, media path allowlist fail-closed). `helm/`, `helm-scale-large/` and
`helm-scale-medium/` ship a `security:` values block wired to these flags. In
PCAI you set them in `values.yaml` (the *Helm Values* editor):

```yaml
# values.yaml
security:
  mediaTokenSecret: "$RANDOM"
  apiKey: "change-me"
  blockPrivateHosts: true
```

> Keep deployment secret material (e.g. `helm/.values.yaml`, which contains
> Kubernetes service-account tokens) out of version control — it is not
> covered by `.gitignore`.
