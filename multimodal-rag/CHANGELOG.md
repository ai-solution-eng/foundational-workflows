# Changelog

All notable changes to this project are tracked here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.5.0] — 2026-08-27

Audit-hardening release: caption twins for media, unified "skip entirely"
drop + on-disk cleanup, ingest-warning surfacing to the UI, and Low-severity
audit clean-ups. Committed over the `checkpoint-pre-audit-fixes` baseline.

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

### Caption twins for media (dual-embedding ingest)

- **Caption twins for image/video/audio**: pure-media docs (whose text is a
  media placeholder + an ingest-time VLM/ASR caption) now get a *second*
  embedding — the same media embedded **with** the caption text — in addition
  to the base media-only embedding (which strips captions via
  `_strip_embed_caption`). Caption wording is therefore searchable by text
  queries that the raw-media embedding would miss. Gated by
  `_media_caption_twin_needed()`: created only when the embedder supports the
  doc's media modality AND a caption is present. If the embedder can't embed
  the media but a VLM/ASR module can, the Preprocessor collapses the media to
  caption text and embeds that (existing behaviour, unchanged); if neither
  supports it, the media is skipped entirely. Twins share the parent's
  `(source, page, chunk_index)` identity and are tagged `_twin=True`, so
  retrieval `_dedup_twins()` still prefers the parent when both match.
- New offline tests: `tests/full_pipeline/test_twins.py` (helper gating +
  end-to-end ingest through the in-memory store with a stub embedder).

### Ingest-warning surfacing fix

- **Skip/caption warnings now reach the UI**: ingest warnings (media dropped
  because neither the embedder nor a VLM/ASR supports it, ASR/VLM unavailable,
  caption skipped) are collected per request and returned to the frontend as
  `warnings` — single-file uploads and `POST /documents` return them in the
  response body; batch-files/batch-urls attach them to the job result the UI
  polls. Previously the collector (a `contextvars.ContextVar`) was invisible
  to the thread-pool/background-loop workers that actually run ingestion, so
  the `warnings` list always came back empty in every path. The four ingest
  call sites in `api_server.py` now submit through
  `_submit_with_context`, which copies the request context into the worker
  threads. Regression test: `tests/full_pipeline/test_ingest_warnings.py`.

### Audit clean-up (Low-severity)

- **PIL file handle released** (`dataset_manager.py`): `Image.open(...)` in
  `_preprocess_image_file` now runs under a `with` block, so the decoded
  image's file handle is closed right after the size check instead of being
  left to GC.
- **Media-payload strip no longer silent** (`dataset_manager.py`):
  `_strip_media_payloads` logs a warning when the Qdrant retrieve fails or a
  `set_payload` group fails, instead of bare `return`/`continue` swallowing
  the error.
- **Deprecated FastAPI startup hooks removed**: `api_server.py` and
  `embed_batcher.py` replaced `@app.on_event("startup")` with a
  `lifespan=` context manager (the `on_event` API is removed in FastAPI
  0.99+). Both apps still run the same eager init / config-watcher / health
  loop / upload-prune work at startup; verified via `app.router.lifespan_context`.

### Unified "skip entirely" drop + on-disk cleanup

- **Uniform drop rule for all media types** (`rag_system.py`): audio, image
  and video now behave identically when the embedder can't ingest the media
  and no VLM/ASR can convert it to caption text — the media is removed and,
  if nothing embeddable remains (no supported media, no caption text, no
  real text — a bare `[Video: x.mp4] [0s–32s]` placeholder doesn't count),
  the whole document is **omitted** and logged + surfaced as an ingest
  warning. Previously image (kept but never embedded) and video (left a
  placeholder-only doc) diverged from audio's drop.
- **On-disk cleanup of dropped files** (`dataset_manager.py`): when every
  document a file produced is dropped, the stored copy in `files/` (and its
  `*_preprocessed` tier-2 sibling) is deleted and its content-hash entry
  forgotten — a file that can never be used no longer occupies PVC space.
  Deleting is guarded by a vector-store reference check (`_file_referenced`),
  so a file still referenced by any Qdrant point is never orphaned; remote
  (URL) ingests are never deleted. Wired into both the batch consumer and the
  single-file/URL ingest path. New offline tests:
  `tests/full_pipeline/test_ingest_drop_cleanup.py`.

### Deployment fixes (found in mm-rag production)

- **Recreate now actually re-embeds** (`dataset_manager.py`):
  `recreate_dataset` clears the per-dataset ingest-dedup index
  (`.ingested_hashes.json`) before re-processing, so it no longer skips every
  previously-ingested file as "already ingested" and leaves the freshly-dropped
  collection empty. Previously, "Recreate" dropped the Qdrant collection and
  then stored **0** documents (files are skipped by content-hash dedup on the
  second pass).
- **Job progress is now shared across workers/pods** (`api_server.py`):
  `_UploadJobTracker` mirrors each job to Redis (when `REDIS_URL` is set) with
  a 6h TTL, so the poll-based `upload-status` endpoint answers from any API
  process. The in-memory-only tracker returned `404 Job … not found` under the
  scale chart (4 replicas × 4 gunicorn workers) whenever the load balancer
  routed the progress poll to a different process than the one that created
  the job — breaking progress for batch uploads and recreate. Falls back to
  in-memory-only when Redis is unavailable.
- New offline tests: `tests/full_pipeline/test_job_tracker_recreate.py`.

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