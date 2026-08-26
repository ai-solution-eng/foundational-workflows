# Changelog

All notable changes to this project are tracked here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — audit-hardening series (targets v1.9.0)

Committed as three waves over the `checkpoint-pre-audit-fixes` baseline.

### Wave 1 — MCP/REST hardening (no behavior change by default)

- **MCP tool-limit clamps** (`search_dataset`, `search_memory`): `top_k` is
  coerced into `[1, 100]` and `reranker_top_k` into `[1, min(50, top_k)]`,
  preventing a single tool call from ballooning into a huge
  Qdrant/VLM/memory request. Non-integer or `<=0` values raise a `ToolError`.
- **Per-client unlock scoping**: the MCP server's dataset unlock cache is now
  keyed by the caller's identity (auth-proxy headers → `X-Forwarded-For`;
  otherwise a shared `"default"`), so unlocking a dataset no longer opens it
  for every MCP client on the pod. Configurable size cap via `UNLOCK_CACHE_MAX`.
- **Bounded caches**: `_query_emb_cache`, `_file_hash_cache`,
  `_asr_transcript_cache` are capped (`QUERY_EMB_CACHE_MAX`,
  `FILE_HASH_CACHE_MAX`, `ASR_TRANSCRIPT_CACHE_MAX`) and evict LRU-style so a
  long-running server never accumulates unbounded memory.
- **Single-query dataset-vector lookup**: replaced the per-field Qdrant
  scrolls (up to 10 round-trips) with one OR-filtered scroll.
- **Media output escaping**: filenames/labels in the generated markdown
  image/audio/document blocks are escaped to prevent crafted filenames from
  breaking out of markup.
- **File serving**: non-media files (and SVG) are served with
  `Content-Disposition: attachment` instead of inline, mitigating stored-XSS.
- **Staging `DATA_PATH` fix**: the staging/sweep/serve paths now resolve
  `DATA_PATH` from the environment at call time, fixing `--data-path` in CLI mode.

### Wave 2 — ingestion + transport guards (opt-in via env)

- **Download size caps**: remote and S3 ingests stream to disk and abort past
  `MAX_REMOTE_DOWNLOAD_BYTES` (`Content-Length` pre-check + streamed check).
- **URL policy**: `INGEST_ALLOW_HOSTS` allowlist and
  `INGEST_BLOCK_PRIVATE_HOSTS` private-range block for http(s) ingestion.
- **Archive-bomb bounds**: `ARCHIVE_MAX_TOTAL_BYTES`, `ARCHIVE_MAX_MEMBER_BYTES`,
  `ARCHIVE_MAX_ENTRIES` are audited from the archive headers *before*
  extraction (recursively for nested archives).
- **Media path allowlist**: `MEDIA_ALLOW_PATH_PREFIXES` constrains which
  `file://` / local paths the MCP media tools read.
- **Signed media tokens**: when `MEDIA_TOKEN_SECRET` is set, media URLs carry
  a short-lived HMAC `?token=<expiry>.<sig>` (TTL `MEDIA_TOKEN_TTL`) instead
  of the clear dataset password; the file-serving endpoint validates tokens.
- **REST API auth**: `RAG_API_KEY` enables `Bearer`/`X-RAG-Api-Key`
  enforcement over `/api/*`, with probes/pages/media-serving exemptions.
- **Password throttling**: per-identity failure counters
  (`PW_MAX_FAILURES` / `PW_FAIL_WINDOW`) rate-limit unlock/verify/search flows.

### Post-audit follow-ups

- **PyMuPDF import migration**: PDF extraction now uses `import pymupdf`
  instead of the deprecated `fitz`, silencing the "`fitz` API is
  deprecated" startup warning (PyMuPDF ≥ 1.24 exposes the new module under
  `pymupdf`).
- **`qdrant-client` floor pinned to ≥1.19**: the gRPC auth interceptor only
  guards its deprecated `asyncio.iscoroutinefunction()` call since 1.19;
  older releases emit a `DeprecationWarning` once per Qdrant search on
  Python 3.14 (re-logged at INFO by the MCP SDK). `qdrant-client>=1.19.0,<2`
  prevents a future build from regressing to a warning-spamming release.
- **MCP sidecar probes**: the MCP server now serves `/healthz` (SSE and
  streamable-http transports) and the Helm deployment configures
  startup/liveness/readiness probes for the sidecar container.
- **`describe_media` modality detection**: URLs are classified by their
  *path* extension (query/fragment ignored), so `…/photo.jpg?hmac=…` is
  recognised as an image even though the string does not end in `.jpg`; for
  extension-less CDN URLs it falls back to a bounded network probe
  (Content-Type + first-chunk magic bytes), and extension-less local files /
  `data:` URLs are sniffed via magic bytes / MIME prefix. A new optional
  `media_type` parameter ("image"/"video") lets the caller force the
  expected modality, and the query wording (e.g. "describe the image") is
  used as a low-confidence hint when detection is ambiguous. The response
  now reports the resolved `media_type`.
- **Hash-index cache**: `.hashes.json` reads are mtime-cached per process so
  large file sets are not re-read/re-parsed on every upload.
- **No base64 media in the LLM result JSON**: `search_dataset`/`search_memory`
  result JSON no longer carries heavy tier-3 `image`/`video`/`audio` base64
  data URLs. The Postprocessor still consumes them internally to generate
  descriptions, but the JSON handed back to the LLM drops a media key when
  no tier-2 `preprocessed_*` ref exists to attach a viewable URL — a
  matched video segment previously dumped megabytes of
  `data:video/mp4;base64,...` into a text-only LLM's context (tens of
  thousands of wasted tokens). Tier-2 refs are still substituted and
  converted to signed HTTP URLs as before.
- **Model connectivity monitoring**: the embedder is probed in the background
  every `MODEL_HEALTH_INTERVAL` (default 60 s); `/api/admin/health` exposes
  `models.embedder` status. The embedder gate drives the **readiness** probes
  (`/readyz` on the API and a new `/readyz` on the MCP sidecar, which the Helm
  charts now use for the MCP readiness probe) after
  `MODEL_HEALTH_FAIL_THRESHOLD` (default 3) consecutive failures — liveness is
  untouched, so a remote vLLM/SGLang embedder outage drops the pod out of
  rotation without a restart loop. A new `GET /api/admin/connections` endpoint
  live-checks every configured model (`healthy` / `not_provided` /
  `unhealthy`), and the management page gained a "Test connections" button
  plus a live embedder status indicator.

### Known limitations / future work

- The RAG `need_media` flag is still on for *every* search when a VLM is
  configured, even when ingest-time captions let the Postprocessor skip the
  VLM call (the skip happens after the payload transfer). A two-pass fetch
  (lightweight first, then re-fetch only the docs that will actually hit the
  VLM) would eliminate the residual traffic — deferred.
- `.hashes.json` is kept as a JSON + fcntl format (not SQLite) because the
  RWX PVC is NFS-backed, where in-process SQLite locking is unreliable.

## [1.8.1] — previous release

Text-only twins for multimodal documents at retrieval de-duplication, `keep_originals`
flag, VLM captioning of videos when `caption_with_vlm` is enabled, and
4-image VLM batching fixes. See git history for details.