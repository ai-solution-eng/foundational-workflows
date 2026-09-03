# Changelog

All notable changes to this project are tracked here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [3.5.1] — 2026-09-03

### Security
- **Document media refs are now validated** (input-handling audit H-1): media refs inside user-supplied documents (`POST /api/datasets/{name}/documents`, MCP `add_memory` — the `image`/`video`/`audio`/`preprocessed_*` keys) are checked before the pipeline reads them. The server fetches these refs at embed time and again at query time, so a bare path, `file:///etc/...`, or a blocked http(s) URL in a document dict was an arbitrary-file-read / internal-SSRF channel. New shared modules: `utils/url_policy.py` (the SSRF guards, re-exported from `dataset_manager`) and `utils/media_paths.py` (the `MEDIA_ALLOW_PATH_PREFIXES` allowlist, re-exported from `mcp_server`, plus `MediaRefError`). Enforcement: at `aadd_to_vector_store` entry (validation; `MediaRefError` → HTTP 400 on REST, `ToolError` on MCP) and at the read primitives (`_afetch_media_bytes`, `_file_url_to_data_url`, `_resize_media_in_docs`) so legacy stored payloads fail closed too. http(s) media fetches additionally gained the redirect policy and a `MAX_MEDIA_FETCH_BYTES` (default 512 MB) streamed cap. `s3://` refs in document media fields are rejected (ingest S3 via `/batch-urls`); `MEDIA_ALLOW_PATH_PREFIXES="*"` is the dev/test escape hatch.
- **Backup-import overwrite is password-gated** (audit H-2): `POST /api/admin/datasets/import` with `overwrite=true` deletes the target dataset — now only after the target name (from `new_name` or the archive's `meta.json`, peeked via the new `DatasetManager.peek_backup_meta` before anything is touched) passes the same `_require_dataset_password` gate as delete/recreate/migrate/backfill. Previously the API key alone could replace a password-protected dataset.
- **S3 bucket allowlist** (audit M-5): `INGEST_ALLOW_S3_BUCKETS` (comma-separated) — when set, `_download_s3`/`_list_s3_prefix` refuse any bucket not on the list, so ingest credentials can't be pointed at other tenants' buckets on a shared MinIO. Unset preserves current behavior.

### Fixed
- **CSV/TSV row-grouping budget was dead code** (audit M-2): `_build_docs` measured `r.get("text")` on raw row dicts (always empty for a normal CSV), so the flush condition never fired and an entire CSV collapsed into ONE document — a 500k-row CSV produced a single ~100 MB doc for the embedder. Row groups are now measured against the serialized text actually emitted (verified: 60 rows → 3 budget-sized docs).
- **Backup import is bounded** (audit M-1): `prepare_import` now audits declared member sizes (`MAX_IMPORT_EXTRACT_BYTES`, default 8 GiB) and member count (`MAX_IMPORT_MEMBERS`, default 20000) before extracting, caps `meta.json` reads at 4 MB, and streams the `documents.jsonl` row count instead of buffering the whole member — closing the gzip-bomb-to-PVC path the ingested-archive path already guarded against.
- **EPUB decompression bounds** (audit M-3): every `zf.read()` in `ebook_processor` is now gated by declared-size caps (`MAX_EPUB_MEMBER_BYTES`, default 256 MB per member; `MAX_EPUB_TOTAL_BYTES`, default 1 GiB cumulative) — a ZIP-bomb EPUB fails the file instead of ballooning RAM, and the image-read `except (KeyError, Exception): pass` no longer swallows bound violations.
- **PDF raster geometry guard** (audit M-4): OCR page rasters and the `Matrix(2,2)` page-render image fallback skip rendering when the estimated pixel count exceeds `PDF_MAX_RASTER_PIXELS` (default 64 M) — a crafted huge-MediaBox page can no longer drive a giant pixmap allocation.
- **Text chunking robustness** (audits L-2/L-3): `_split_by_chars` caps the overlap at 80 % of the chunk size (an overlap ≥ chunk size previously looped forever on a pathological config — the same guard `pdf_processor` already had); `json_processor`/`notebook_processor` handle pathologically deep nesting gracefully (`RecursionError` from both the scanner and the flattener) instead of failing the file with a raw traceback.

### Changed
- `image_processor.process_url`/`_read_file` are documented as internal-only (they open any path and must never be wired to user-controlled input — API/MCP media refs are validated centrally in `rag_system`).

## [3.4.0] — 2026-09-01

### Added
- **Metadata-filtered search**: optional filter dict on every search surface (REST GET/POST dataset search, MCP `search_dataset`, federated search) — `file_types`, `severities`, `source_prefix` (Qdrant `MatchPrefix`), `date_from`/`date_to` — AND-combined and applied server-side before ranking. `metadata.file_type` is stamped at ingest; new collections get payload indexes automatically; older datasets via the idempotent `POST /api/admin/datasets/{name}/backfill-search-metadata`.
- **Hybrid dense + BM25 (RRF)**: new collections carry a named `dense` vector plus a sparse `bm25` vector (`schema_version: 2`); text queries run server-side RRF fusion over both lanes (batcher semantics unchanged; multimodal queries stay dense-only). Per-dataset BM25 document frequencies in `files/.bm25_stats.json`; `RAG_HYBRID_SEARCH` / `RAG_BM25_K1` / `RAG_BM25_B` knobs; legacy collections keep flat dense search with a one-time recreate nudge.
- **Backup restore/import**: `POST /api/admin/datasets/import` accepts an export `.tar.gz` (multipart) or `{"s3_uri": ...}`. Exports carry no vectors, so restore **re-embeds** — datasets with `files/` go through the recreate flow (twins/captions/tiers rebuilt); raw-documents backups replay their `documents.jsonl` text. Optional `new_name`/`overwrite` (409 without it)/`password` (exports strip the hash by design). Backup CronJob gained `backups.retentionDays` pruning (default 0 = keep forever).
- **S3 sync with pruning**: `sync: true` on `/batch-urls` reconciles an S3 prefix — sources deleted upstream are pruned from the dataset; URLs without stored points are force-re-ingested. `sync_dry_run: true` reports the `would_ingest`/`would_prune` diff. Known limitation: content changed under the same key keeps its old points (source-keyed pruning cannot see versions).
- **Prometheus `/metrics`** (API container): HTTP request count/latency by route template, ingest files/chunks/jobs, Qdrant op latency (the batched-search bottleneck, now continuous), cache hit/miss/eviction, hybrid-vs-dense search counts. Bounded labels only. `prometheus-client` added to requirements (import-optional at runtime); charts ship an opt-in ServiceMonitor (`metrics.serviceMonitor`).
- **OCR fallback for scanned PDFs**: dataset-level `ocr` flag (create/PATCH, default off) — pages with images but an empty text layer are rasterized and OCR'd through the tesseract CLI (added to the image) into `[OCR page N]` text blocks, making scanned archives text-searchable and BM25-indexable. Env: `OCR_LANG`/`OCR_DPI`/`OCR_TIMEOUT_S`. New-dataset default is server-configurable via chart values `rag.ocr` -> `RAG_OCR_DEFAULT` (default false; explicit per-dataset flag still wins).
- **MCP memory management tools**: `delete_memory` (explicit point IDs only — no similarity-directed deletion), `list_memories` (newest-first, kind/tag filters, previews + ids), `forget_session` (wipes one session's `session_history`). Same header-based identity resolution as the other memory tools.
- **Federated multi-dataset search**: MCP `search_datasets(datasets | "all", ...)` and `POST /api/search` — concurrent per-dataset fan-out, dataset-labelled merged results with per-dataset scores, dataset-qualified dedup, one optional rerank over the merged pool. Password-protected datasets without a cached unlock are skipped with a note; there is deliberately no password parameter (v3.0.0 rule preserved).

### Changed
- **ToC/Index noise filtering hardened**: PyMuPDF flattens whole ToC/Index pages into run-on lines that defeated the line-oriented heuristic — detection is now content-based (dot-leader runs, header prefixes incl. Index, section-ref + page-number repetition, index-entry patterns), with the flattened DeepSeek-paper sample as a regression test. Existing chunks persist until re-ingest/Recreate.
- **OCR fallback in the web UI**: create-dialog + manage-page checkboxes, pre-checked from the server default; new-dataset default via `rag.ocr` → `RAG_OCR_DEFAULT`.
- MCP tool count 9 → 13 (`delete_memory`, `list_memories`, `forget_session`, `search_datasets`).
- `_arun_retrieval` split into retrieval-core / postprocess / formatter stages (output-preserving) so federated search can reuse the single-dataset pipeline.
- Docker image now includes `tesseract-ocr`; `prometheus-client` added to requirements.

## [3.2.0] — 2026-08-31

### Added
- **Documents download**: `GET /api/datasets/{name}/documents/download?format=md|jsonl` — every document as one streamed file (readable Markdown by default, JSONL for scripts), no binary files, heavy
  tier-3 base64 stripped, tier-2 refs kept; UI "Download Docs" button on the manage page (named to avoid the existing "Documents" viewer button).
- **Opt-in backup CronJob** (`backups.enabled`): exports non-protected datasets to an S3/MinIO bucket on a schedule.

### Fixed
- **API-key middleware exemptions never matched**: `scope["route"]` is not set inside an http middleware (routing happens later), so once `security.apiKey` shipped a default (v3.1.0), every exempt
  route — dataset media serving, staged media, the HTML pages' JS fetches — returned 401 `Missing or invalid API key`. The middleware now resolves the matched endpoint explicitly.
- **Default API key + auto-wiring** (`security.apiKey`): charts ship a default key; the served UI embeds it for the browser, and the Open WebUI filter gained an `RAG_API_KEY` valve — direct/scripted
  callers send `X-RAG-Api-Key`. Same chart-known-default tradeoff as the token secret.
- **Optional Redis auth** (`redis.password`) and container hardening (read-only rootfs, no privilege escalation, dropped capabilities, no SA token automount) across all charts.

### Performance
- Two-phase media fetch: searches skip the tier-3 base64 transfer when the VLM conversion would be caption-skipped anyway (`_media_payloads_needed`).
- `_rag_cache` is now LRU-capped (`RAG_CACHE_MAX`, default 64) and closes evicted Qdrant clients.
- File copies during ingest no longer hold the cross-process hash lock.
- **Idle early-flush for the embedding query batchers (v2, batch-guarded)**: when the query queue stops growing and holds at most 2 items, it flushes after `EMBEDDING_QUERY_IDLE_WAIT_MS` (chart
  default 50ms) instead of waiting the full 200ms window; bursts never split, because the idle check only fires on a stalled small queue.  (The unguarded v1 — 10ms, no guard — was benchmark-convicted:
  mid-burst splitting collapsed N=100 from 53 to 15 r/s — and reverted.)  v3.1.8 A/B: single-query mean 238ms (was ~370-550 with the full window), N=100 49.3 r/s and N=250 72.8 r/s (p99 5.2s vs v1's
  51s), 100% success at every load.

### Changed
- **Chart defaults**: the idle early-flush is promoted into the `helm-scale-large` / `helm-scale-medium` values (`app.embeddingQueryIdleWaitMs: 50`) and scale-large is restored to **4 replicas × 4
  workers** — the old "4 workers → ~8 req/s" result predates the shared embed-batcher singleton and does not reproduce (the same 4×4 shape measured 49.3 / 72.8 r/s on v3.1.8).  The single-replica
  `helm/` chart has no batcher tuning surface and keeps the disabled (0) default.
- **Release tooling**: `automation.sh` bumps versions BEFORE the docker build (previously the image baked the previous release's version), and the package carries `__version__` (surfaced in FastAPI
  `/docs`; in-cluster self-identification: `python3 -c "import multimodal_rag; print(multimodal_rag.__version__)"`).

## [3.0.0] — 2026-08-31

*(Shipped as 3.0.0: mandatory `MEDIA_TOKEN_SECRET` and password-gated destructive routes are breaking changes.)*

Second audit round (documentation / performance / security). Supersedes two descriptions in the 2.5.0 notes below: **signed media tokens are mandatory, not opt-in** (both servers refuse to start
without `MEDIA_TOKEN_SECRET`, and the legacy `?password=` suffix was fully removed — commit `c27ddab`), and **per-client unlock scoping no longer keys on `X-Forwarded-For`** (dropped as spoofable;
identity now comes from auth-proxy headers when `RAG_TRUST_PROXY_IDENTITY` is set, else the socket peer).

### Security

- **Destructive routes are now password-gated**: `DELETE`/`PATCH /api/datasets/{name}`, `/api/admin/datasets/{name}/recreate` and `/migrate-tier-schema` require the dataset password
  (`X-Dataset-Password` header or a cached unlock) — previously a password-protected dataset could be deleted or rewritten by a caller who never knew the password. The frontend passes
  `passwordHeader()` on delete/recreate/poll/patch.
- **Query-time SSRF guard**: media URLs supplied at query time (REST search `image`/`video`/`audio`, MCP `search_dataset`, `describe_media`, `transcribe_audio`, and `add_memory` at write time) are now
  checked with the same host policy as ingest (`_check_media_url_policy`) — private and link-local ranges (incl. cloud metadata) are blocked by default; loopback is allowed so clients can pass the
  server's own media URLs back; `INGEST_ALLOW_HOSTS` remains authoritative. `_afetch_media_bytes` also gained a bounded timeout (previously none).
- **Proxy identity headers are opt-in** (`RAG_TRUST_PROXY_IDENTITY`, helm `security.trustProxyIdentity`): `X-Auth-Request-*`/`X-Email`/`X-User` are client-spoofable, so they are only trusted when the
  operator confirms an enforcing auth proxy overwrites them; the default is the socket peer. Charts set it `true` (oauth2-proxy deployments).
- **RAR fallback hardening**: the `unrar` CLI fallback now pre-lists members (entry cap + traversal check), rejects symlink members and enforces the size caps on the extracted bytes
  (`_sweep_extracted_tree`), fails closed when no RAR tooling exists, and checks the CLI exit code. Bare `.gz`/`.bz2`/`.xz` (non-tar) files now decompress correctly with a streamed byte cap instead of
  always failing in tarfile.
- Startup warning when `RAG_API_KEY` is unset (admin surface unauthenticated).
- Frontend `escHtml` escapes quotes — dataset-controlled values interpolated into attributes (`href=`/`src=`/`onclick=`) can no longer break out (stored XSS).

### Performance

- Dedicated thread pools: `_QDRANT_IO_POOL` (`QDRANT_POOL_SIZE`, 4) for Qdrant upserts/scrolls/dedup/batched searches and `_MEDIA_POOL` (`MEDIA_POOL_SIZE`, 2) for ffmpeg work — previously sync Qdrant
  upserts ran on the event loop (a media-heavy sub-batch froze every concurrent request) and long ingest jobs shared the default executor with the search batcher.
- `_require_dataset_password` is async: PBKDF2-600k verification (~0.2–0.5 s CPU) and NFS `has_password`/`get_dataset` meta reads run in `sync_pool` instead of on the event loop (`get_dataset`'s
  cross-pod retry loop could block the loop up to 2 s).
- MCP query-embedding cache stores packed `array('f')` vectors — ~65 MB at cap instead of ~530 MB for 4096-dim Python lists.
- Batch ingest coalesces `.hashes.json` writes into one merged write per batch (`_flush_hash_index_writes`) and caches `.ingested_hashes.json` reads — both were full-file rewrites/re-parses per stored
  file (O(n²) NFS I/O).
- Charts set `QDRANT_CLIENT_TIMEOUT=30` (code default remains no-timeout).

### Documentation

- README security table corrected: `MEDIA_TOKEN_SECRET` required (not optional-with-fallback), `INGEST_BLOCK_PRIVATE_HOSTS` defaults `true`, `MEDIA_ALLOW_PATH_PREFIXES` defaults fail-closed, embedder
  health does not gate `/healthz`/`/readyz`; new env vars documented (`RAG_TRUST_PROXY_IDENTITY`, `QDRANT_CLIENT_TIMEOUT`, `QDRANT_POOL_SIZE`, `MEDIA_POOL_SIZE`).
- DEPLOYMENT.md: image tag `v2.5.1` (was `v2.5.0`), only the embedder is required (reranker/vlm/asr optional), VLM example updated to the shipped default, `captionWith*` chart defaults (`true`),
  VirtualService timeout tiers (300s/3600s, was 660s).
- SCALE.md: `helm-scale-large` is 2 replicas × 2 workers (was documented as 4×4/1024-concurrency; actual 2×2 = 256, matching the shipped values and their benchmarking note); summary/resource totals
  recomputed.
- MCP.md / FEATURES.md / MEMORY.md: the MCP unlock cache is per-process, not Redis-backed (Redis backs the REST unlock cache only).
- USAGE.md: programmatic quickstart now passes the required embedder; create-dataset fields corrected (`caption_with_asr`/`caption_with_vlm`/ `keep_originals`).
- API.md: `POST /lock` and `POST /media-token` added to the endpoint table; create-dataset defaults follow server config.
- Open WebUI README: `REPAIR_MEDIA_URLS` valve documented; relative link and troubleshooting row fixed.

## [2.5.0] — 2026-08-27

Audit-hardening release: caption twins for media, unified "skip entirely" drop + on-disk cleanup, ingest-warning surfacing to the UI, and Low-severity audit clean-ups. Committed over the
`checkpoint-pre-audit-fixes` baseline.

### Wave 1 — MCP/REST hardening (no behavior change by default)

- **MCP tool-limit clamps** (`search_dataset`, `search_memory`): `top_k` is coerced into `[1, 100]` and `reranker_top_k` into `[1, min(50, top_k)]`, preventing a single tool call from ballooning into
  a huge Qdrant/VLM/memory request. Non-integer or `<=0` values raise a `ToolError`.
- **Per-client unlock scoping**: the MCP server's dataset unlock cache is now keyed by the caller's identity (auth-proxy headers → `X-Forwarded-For`; otherwise a shared `"default"`), so unlocking a
  dataset no longer opens it for every MCP client on the pod. Configurable size cap via `UNLOCK_CACHE_MAX`.
- **Bounded caches**: `_query_emb_cache`, `_file_hash_cache`, `_asr_transcript_cache` are capped (`QUERY_EMB_CACHE_MAX`, `FILE_HASH_CACHE_MAX`, `ASR_TRANSCRIPT_CACHE_MAX`) and evict LRU-style so a
  long-running server never accumulates unbounded memory.
- **Single-query dataset-vector lookup**: replaced the per-field Qdrant scrolls (up to 10 round-trips) with one OR-filtered scroll.
- **Media output escaping**: filenames/labels in the generated markdown image/audio/document blocks are escaped to prevent crafted filenames from breaking out of markup.
- **File serving**: non-media files (and SVG) are served with `Content-Disposition: attachment` instead of inline, mitigating stored-XSS.
- **Staging `DATA_PATH` fix**: the staging/sweep/serve paths now resolve `DATA_PATH` from the environment at call time, fixing `--data-path` in CLI mode.

### Wave 2 — ingestion + transport guards (opt-in via env)

- **Download size caps**: remote and S3 ingests stream to disk and abort past `MAX_REMOTE_DOWNLOAD_BYTES` (`Content-Length` pre-check + streamed check).
- **URL policy**: `INGEST_ALLOW_HOSTS` allowlist and `INGEST_BLOCK_PRIVATE_HOSTS` private-range block for http(s) ingestion.
- **Archive-bomb bounds**: `ARCHIVE_MAX_TOTAL_BYTES`, `ARCHIVE_MAX_MEMBER_BYTES`, `ARCHIVE_MAX_ENTRIES` are audited from the archive headers *before* extraction (recursively for nested archives).
- **Media path allowlist**: `MEDIA_ALLOW_PATH_PREFIXES` constrains which `file://` / local paths the MCP media tools read.
- **Signed media tokens**: when `MEDIA_TOKEN_SECRET` is set, media URLs carry a short-lived HMAC `?token=<expiry>.<sig>` (TTL `MEDIA_TOKEN_TTL`) instead of the clear dataset password; the file-serving
  endpoint validates tokens.
- **REST API auth**: `RAG_API_KEY` enables `Bearer`/`X-RAG-Api-Key` enforcement over `/api/*`, with probes/pages/media-serving exemptions.
- **Password throttling**: per-identity failure counters (`PW_MAX_FAILURES` / `PW_FAIL_WINDOW`) rate-limit unlock/verify/search flows.

### Post-audit follow-ups

- **PyMuPDF import migration**: PDF extraction now uses `import pymupdf` instead of the deprecated `fitz`, silencing the "`fitz` API is deprecated" startup warning (PyMuPDF ≥ 1.24 exposes the new
  module under `pymupdf`).
- **`qdrant-client` floor pinned to ≥1.19**: the gRPC auth interceptor only guards its deprecated `asyncio.iscoroutinefunction()` call since 1.19; older releases emit a `DeprecationWarning` once per
  Qdrant search on Python 3.14 (re-logged at INFO by the MCP SDK). `qdrant-client>=1.19.0,<2` prevents a future build from regressing to a warning-spamming release.
- **MCP sidecar probes**: the MCP server now serves `/healthz` (SSE and streamable-http transports) and the Helm deployment configures startup/liveness/readiness probes for the sidecar container.
- **`describe_media` modality detection**: URLs are classified by their *path* extension (query/fragment ignored), so `…/photo.jpg?hmac=…` is recognised as an image even though the string does not end
  in `.jpg`; for extension-less CDN URLs it falls back to a bounded network probe (Content-Type + first-chunk magic bytes), and extension-less local files / `data:` URLs are sniffed via magic bytes /
  MIME prefix. A new optional `media_type` parameter ("image"/"video") lets the caller force the expected modality, and the query wording (e.g. "describe the image") is used as a low-confidence hint
  when detection is ambiguous. The response now reports the resolved `media_type`.
- **Hash-index cache**: `.hashes.json` reads are mtime-cached per process so large file sets are not re-read/re-parsed on every upload.
- **No base64 media in the LLM result JSON**: `search_dataset`/`search_memory` result JSON no longer carries heavy tier-3 `image`/`video`/`audio` base64 data URLs. The Postprocessor still consumes
  them internally to generate descriptions, but the JSON handed back to the LLM drops a media key when no tier-2 `preprocessed_*` ref exists to attach a viewable URL — a matched video segment
  previously dumped megabytes of `data:video/mp4;base64,...` into a text-only LLM's context (tens of thousands of wasted tokens). Tier-2 refs are still substituted and converted to signed HTTP URLs as
  before.
- **Model connectivity monitoring**: the embedder is probed in the background every `MODEL_HEALTH_INTERVAL` (default 60 s); `/api/admin/health` exposes `models.embedder` status. The embedder gate drives
  the **readiness** probes (`/readyz` on the API and a new `/readyz` on the MCP sidecar, which the Helm charts now use for the MCP readiness probe) after `MODEL_HEALTH_FAIL_THRESHOLD` (default 3)
  consecutive failures — liveness is untouched, so a remote vLLM/SGLang embedder outage drops the pod out of rotation without a restart loop. A new `GET /api/admin/connections` endpoint live-checks
  every configured model (`healthy` / `not_provided` / `unhealthy`), and the management page gained a "Test connections" button plus a live embedder status indicator.

### Caption twins for media (dual-embedding ingest)

- **Caption twins for image/video/audio**: pure-media docs (whose text is a media placeholder + an ingest-time VLM/ASR caption) now get a *second* embedding — the same media embedded **with** the
  caption text — in addition to the base media-only embedding (which strips captions via `_strip_embed_caption`). Caption wording is therefore searchable by text queries that the raw-media embedding
  would miss. Gated by `_media_caption_twin_needed()`: created only when the embedder supports the doc's media modality AND a caption is present. If the embedder can't embed the media but a VLM/ASR
  module can, the Preprocessor collapses the media to caption text and embeds that (existing behaviour, unchanged); if neither supports it, the media is skipped entirely. Twins share the parent's
  `(source, page, chunk_index)` identity and are tagged `_twin=True`, so retrieval `_dedup_twins()` still prefers the parent when both match.
- New offline tests: `tests/full_pipeline/test_twins.py` (helper gating + end-to-end ingest through the in-memory store with a stub embedder).

### Ingest-warning surfacing fix

- **Skip/caption warnings now reach the UI**: ingest warnings (media dropped because neither the embedder nor a VLM/ASR supports it, ASR/VLM unavailable, caption skipped) are collected per request and
  returned to the frontend as `warnings` — single-file uploads and `POST /documents` return them in the response body; batch-files/batch-urls attach them to the job result the UI polls. Previously the
  collector (a `contextvars.ContextVar`) was invisible to the thread-pool/background-loop workers that actually run ingestion, so the `warnings` list always came back empty in every path. The four
  ingest call sites in `api_server.py` now submit through `_submit_with_context`, which copies the request context into the worker threads. Regression test:
  `tests/full_pipeline/test_ingest_warnings.py`.

### Audit clean-up (Low-severity)

- **PIL file handle released** (`dataset_manager.py`): `Image.open(...)` in `_preprocess_image_file` now runs under a `with` block, so the decoded image's file handle is closed right after the size
  check instead of being left to GC.
- **Media-payload strip no longer silent** (`dataset_manager.py`): `_strip_media_payloads` logs a warning when the Qdrant retrieve fails or a `set_payload` group fails, instead of bare
  `return`/`continue` swallowing the error.
- **Deprecated FastAPI startup hooks removed**: `api_server.py` and `embed_batcher.py` replaced `@app.on_event("startup")` with a `lifespan=` context manager (the `on_event` API is removed in FastAPI
  0.99+). Both apps still run the same eager init / config-watcher / health loop / upload-prune work at startup; verified via `app.router.lifespan_context`.

### Unified "skip entirely" drop + on-disk cleanup

- **Uniform drop rule for all media types** (`rag_system.py`): audio, image and video now behave identically when the embedder can't ingest the media and no VLM/ASR can convert it to caption text —
  the media is removed and, if nothing embeddable remains (no supported media, no caption text, no real text — a bare `[Video: x.mp4] [0s–32s]` placeholder doesn't count), the whole document is
  **omitted** and logged + surfaced as an ingest warning. Previously image (kept but never embedded) and video (left a placeholder-only doc) diverged from audio's drop.
- **On-disk cleanup of dropped files** (`dataset_manager.py`): when every document a file produced is dropped, the stored copy in `files/` (and its `*_preprocessed` tier-2 sibling) is deleted and its
  content-hash entry forgotten — a file that can never be used no longer occupies PVC space. Deleting is guarded by a vector-store reference check (`_file_referenced`), so a file still referenced by
  any Qdrant point is never orphaned; remote (URL) ingests are never deleted. Wired into both the batch consumer and the single-file/URL ingest path. New offline tests:
  `tests/full_pipeline/test_ingest_drop_cleanup.py`.

### Deployment fixes (found in mm-rag production)

- **Recreate now actually re-embeds** (`dataset_manager.py`): `recreate_dataset` clears the per-dataset ingest-dedup index (`.ingested_hashes.json`) before re-processing, so it no longer skips every
  previously-ingested file as "already ingested" and leaves the freshly-dropped collection empty. Previously, "Recreate" dropped the Qdrant collection and then stored **0** documents (files are
  skipped by content-hash dedup on the second pass).
- **Job progress is now shared across workers/pods** (`api_server.py`): `_UploadJobTracker` mirrors each job to Redis (when `REDIS_URL` is set) with a 6h TTL, so the poll-based `upload-status`
  endpoint answers from any API process. The in-memory-only tracker returned `404 Job … not found` under the scale chart (4 replicas × 4 gunicorn workers) whenever the load balancer routed the
  progress poll to a different process than the one that created the job — breaking progress for batch uploads and recreate. Falls back to in-memory-only when Redis is unavailable.
- New offline tests: `tests/full_pipeline/test_job_tracker_recreate.py`.

### Known limitations / future work

- The RAG `need_media` flag is still on for *every* search when a VLM is configured, even when ingest-time captions let the Postprocessor skip the VLM call (the skip happens after the payload
  transfer). A two-pass fetch (lightweight first, then re-fetch only the docs that will actually hit the VLM) would eliminate the residual traffic — deferred.
- `.hashes.json` is kept as a JSON + fcntl format (not SQLite) because the RWX PVC is NFS-backed, where in-process SQLite locking is unreliable.

## [1.8.1] — previous release

Text-only twins for multimodal documents at retrieval de-duplication, `keep_originals` flag, VLM captioning of videos when `caption_with_vlm` is enabled, and 4-image VLM batching fixes. See git
history for details.