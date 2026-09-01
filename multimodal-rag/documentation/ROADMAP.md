# Roadmap — Candidate Features (post-audit)

Status: proposal, not committed scope. Authored 2026-08-31, after the v3.0.0 security round and v3.2.0 performance/hardening round. The security and performance surfaces are deliberately left alone
here — these items extend retrieval quality, operability, and the MCP/memory product surface, in preference order. Effort: S (≤1 day), M (2–4 days), L (a week+).

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 1 | Metadata-filtered search | S/M | High for log/code/table corpora |
| 2 | ~~Hybrid dense + BM25 (RRF)~~ — **shipped in 3.4.0** | — | Biggest retrieval-quality lever |
| 3 | OCR for scanned PDFs | S/M | Unlocks scanned archives |
| 4 | Backup restore/import | S/M | Closes the DR loop |
| 5 | S3 sync with pruning | S/M | Completes S3 ingest story |
| 6 | Memory management tools | S | Fixes a documented MCP gap |
| 7 | Prometheus `/metrics` | S/M | Makes the next audit continuous |
| 8 | Federated multi-dataset search | M | Removes agent-side dataset guessing |

---

## 1. Metadata-filtered search

**Problem.** Qdrant filters are used internally (`dataset_manager.py` dedup-by-source, twin collapse, `session_history` replacement) but no user surface can filter. "Only PDFs", "only ERROR, last
24h", "under `reports/2025/`" are impossible today.

**Design.**

- Payloads already carry `metadata.source`, `page`, `chunk_index`, `timestamp_start/end`, `severities`. Add one field at ingest: `metadata.file_type` (the same `_classify_file()` label used for
  `file_type_counts`) — new datasets get it free; existing datasets via an idempotent backfill endpoint (scroll + `set_payload`, the `migrate-tier-schema` pattern).
- Create payload indexes at dataset creation: `metadata.file_type` (KEYWORD), `metadata.severities` (KEYWORD), `metadata.timestamp_start` (DATETIME). Index creation is cheap and idempotent; add it to
  the backfill too.
- Thread an optional `qdrant.Filter` through the retrieval path: `VectorStore.asimilarity_search_with_score_by_vector(emb, k, need_media)` gains `qfilter=None` → `_QdrantBatcher.submit()` →
  `QueryRequest(filter=...)` (`vector_store.py`). `InMemoryVectorStore` applies the predicate in Python.
- Surface a small validated set, not raw filters, on both faces:
  - REST `GET/POST /api/datasets/{name}/search`:
    `file_types=pdf,log`, `severities=ERROR`, `date_from`/`date_to`,
    `source_prefix=` (prefix match on `metadata.source`).
  - MCP `search_dataset(...)`: same params.
- Reranker path is unchanged (filters apply before rerank).

**Files.** `vector_store.py`, `rag_system.py`, `dataset_manager.py` (ingest metadata + backfill), `api_server.py`, `mcp_server.py`, templates.

**Risks.** Existing collections lack `metadata.file_type` until backfilled — the backfill endpoint doubles as the migration. `source_prefix` needs a text index on `metadata.source` (Qdrant `MatchText`
prefix); if that proves flaky, v1 ships without it.

---

## 2. Hybrid dense + BM25 retrieval (RRF fusion) — SHIPPED in 3.4.0

**Problem.** Retrieval is dense-cosine only. Dense embeddings are weakest exactly where this corpus is strongest: code (identifiers, function names), logs (error codes), JSON/YAML keys. Exact-token
recall needs a lexical lane.

**Design.**

- **Schema**: each collection gains a second named vector `bm25` (`SparseVectorParams`, `MULTI`?) alongside the dense named vector already selected via `vector_name`. New collections get it at create
  time; existing collections adopt it through the existing Recreate flow — bump a `schema_version` in `meta.json` so the embedder-fingerprint guard can warn "recreate to enable hybrid".
- **Ingest**: compute a BM25-weighted sparse vector per document that has real text (skip bare `[Image: x]` placeholders; include caption-twin text). No new model dependency: reuse the bundled
  `tokenizer.json` (`token_text_splitter.py`) for term extraction; keep per-dataset df counts in a `.bm25_stats.json` next to the existing `.hashes.json`. Store `SparseVector(indices, values)` in the
  point. Knobs: `RAG_HYBRID_SEARCH` (default on for new datasets), `RAG_BM25_K1`, `RAG_BM25_B`.
- **Query**: build the query sparse vector the same way and replace the flat dense `QueryRequest` with a fusion request — `prefetch=[dense(using=vector_name), sparse(using=bm25)], fusion=RRF`. The
  batcher keeps working: fusion is per-request, so `query_batch_points` semantics are unchanged.
- **Rerank**: unchanged — the cross-encoder reranks the fused top-k pool.
- **Validation**: rerun `evaluations/eval_pipelines/run_eval.py` (embed-only vs embed+rerank vs hybrid) across the 8 ViDoRe domains, plus a hand-built code/log spot-check set (identifier lookups where
  dense fails).

**Files.** `vector_store.py` (query shape), `rag_system.py` (collection create + upsert + query), `dataset_manager.py` (df stats, schema_version), `evaluations/`.

**Risks.** Sparse index grows collection size (~1 sparse vector per chunk); df stats must tolerate multi-writer ingest (same lock pattern as `.hashes.json`). Datasets that skip recreate silently keep
dense-only behaviour — hence the `schema_version` warning.

---

## 3. OCR fallback for scanned PDFs

**Problem.** Scanned PDFs yield near-zero text; they are only searchable via their image chunks and (when configured) VLM captions. VLM prose misses exact strings (names, part numbers, dates).

**Design.**

- In `pdf_processor.extract_chunks_iter()`: per page, if extracted text is below `OCR_MIN_TEXT_CHARS` (default ~32) and the page has images, mark the page `needs_ocr`.
- Two OCR backends, tried in order:
  1. **Tesseract** (new apt layer, offline, CPU): OCR the page raster at
     tier-2 resolution; text is prepended with provenance `{"ocr": true,
     "ocr_lang": ...}`.
  2. **VLM page transcription**: when no tesseract but a VLM is configured,
     reuse the caption machinery for a faithful page transcription.
- Gate per dataset like the existing `caption_with_asr` / `caption_with_vlm` flags: `ocr: true` on `POST /api/datasets` (default off — OCR on an 8k-page scan is expensive and someone must opt in).
  Auto mode (`OCR_ENABLED=auto`) only triggers on near-empty pages, never on healthy ones.
- OCR'd text flows through the normal chunk/twin pipeline (it is real text, so it also strengthens feature 2's BM25 lane).

**Files.** `pdf_processor.py`, `dataset_manager.py` (flag + Dockerfile apt layer), docs.

**Risks.** CPU cost on pathological PDFs — mitigated by the opt-in flag and per-page char threshold.

---

## 4. Backup restore / import

**Problem.** `GET /api/datasets/{name}/export` and the v3.2.0 backup CronJob write backups, but nothing reads them back. The CronJob also skips password-protected datasets, so restore is the only
recovery path for those. The export contains `meta.json` (hash stripped), `documents.jsonl` (`{"id","payload"}`, **no vectors**) and `files/`.

**Design.**

- `POST /api/admin/datasets/import` — multipart `tar.gz` or `{"s3_uri": "s3://bucket/key.tar.gz"}` (reuses the S3 client).
- **Primary path (full restore)**: unpack to the dataset dir (restore `meta.json` + `files/`, excluding `.hashes.json`), then trigger the existing `recreate` flow — re-embeds from on-disk originals
  with the current embedder, rebuilding twins/captions/tiers correctly. Vectors are absent from the export, so re-embedding is the only faithful restore.
  - If the stored `embedder_model`/`embedder_dim` fingerprint mismatches the
    current embedder, proceed but record the *current* fingerprint (recreate
    semantics), with a loud warning in the response.
- **Fallback path (text replay)**: when `files/` is empty but `documents.jsonl` has rows (raw `POST /documents` datasets), re-embed each row's `page_content` as raw documents. Media tiers are not
  rebuilt in this mode — acceptable, since such datasets have no media by construction.
- Name collision: refuse unless `overwrite=true` (drops + recreates, and requires the destructive-route password if the target is protected).
- Password: exports strip `password_hash`, so `import` accepts an optional `password=` to protect the restored dataset (unset → unprotected, stated in the response).
- CronJob follow-up: `backups.retentionDays` to prune old exports.

**Files.** `api_server.py` (endpoint + SSE progress like batch jobs), `dataset_manager.py` (unpack/restore helper), backup CronJob template, docs.

**Effort.** S/M — the heavy lifting (recreate, S3 client, SSE job status) already exists.

---

## 5. S3 sync with pruning

**Problem.** `POST /batch-urls` with an `s3://prefix/` dedups additions, but objects deleted upstream remain in the dataset (and PVC) forever.

**Design.**

- `sync: true` on the `batch-urls` body (S3 prefix sources only): after a successful listing+ingest, compute the listed source set and compare with the dataset's stored sources (paginated scroll with
  `PayloadSelectorInclude(["metadata.source"])` — the storage-stats pattern).
- Prune: delete Qdrant points whose source is absent, delete the PVC copy (and `*_preprocessed` sibling), forget the `.hashes.json` entry — the exact inverse of the existing unreferenced-file cleanup,
  which already guards "no point references this source".
- `sync_dry_run: true` returns the would-be prune list without deleting.
- Media-serving caveat: grouped per original file source, since one file fans out to many chunk points.

**Files.** `dataset_manager.py` (listing diff + prune), `api_server.py` (body param), docs.

---

## 6. Memory management tools (MCP)

**Problem.** `AGENTS.md` documents it: there is no `delete_memory`, so correcting a wrong memory means writing a superseding one. The filter plumbing exists (`metadata.memory_kind` / `session_id`,
`dataset_manager.py:3314`).

**Design.** Three tools, same header-based identity resolution as `add_memory`/`search_memory` (memory headers only ever resolve inside memory tools):

- `delete_memory(memory_ids: list[str])` — explicit ids; `search_memory` results gain the point `id` so the agent can echo it back. No query-directed deletion in v1 (an LLM deleting by similarity is
  too trigger-happy); ids-only is auditable.
- `list_memories(limit, kind?, tags?)` — filtered scroll (`memory_kind != session_history` unless requested); returns id, ts, kind, tags, first ~200 chars.
- `forget_session(session_id)` — reuses the session-history filter to wipe a session's stored history.

**Files.** `mcp_server.py`, `dataset_manager.py` (delete-by-filter helper), `MCP.md`/`MEMORY.md`/`AGENTS.md` (drop the "no delete_memory" caveat).

**Effort.** S.

---

## 7. Prometheus `/metrics`

**Problem.** Zero instrumentation (verified). Both audit rounds relied on manual benchmarks; metrics make the next one continuous and catch regressions the moment they land.

**Design.**

- `prometheus-client`, `/metrics` route exempt from `RAG_API_KEY` exactly like `/healthz` (explicit-route exemption already exists).
- Instrument in three layers:
  1. **HTTP**: request count + duration histogram by route/method/status
     (ASGI middleware; use the route template, not the raw path, to bound
     label cardinality).
  2. **Pipeline**: ingest jobs by state, chunks embedded, warnings emitted,
     embed batcher queue depth/flush size, `EMBEDDING_QUERY_IDLE_*` flush
     counts — the batcher behaviour v3.2.0 tuned would have been visible
     here instead of benchmark-convicted.
  3. **Backends**: Qdrant call duration by op (upsert/scroll/query_batch),
     embedder/reranker/VLM/ASR latency + error counts, cache hit/miss
     (query-embedding, unlock, rag-cache evictions), `_QDRANT_IO_POOL` /
     `_MEDIA_POOL` saturation gauges.
- Charts: `metrics.enabled` + ServiceMonitor annotations; MCP sidecar gets the same endpoint on 9090.

**Files.** `api_server.py`, `mcp_server.py`, `vector_store.py`, `model_adapters.py`, chart values + ServiceMonitor, docs.

**Effort.** S for the middleware + core counters; M if every cache and pool is wired.

---

## 8. Federated multi-dataset search

**Problem.** `search_dataset` takes exactly one dataset, so an agent must already know where the answer lives; wrong guesses waste turns or produce "not found".

**Design.**

- MCP `search_datasets(datasets: list[str] | "all", ...)` and REST `POST /api/search` `{datasets: [...], q, ...}`.
- Fan out concurrently via `asyncio.gather` over the existing `_run_retrieval` helper, per-dataset `top_k` (default 5), merge into one pool, optional single rerank pass over the merged pool
  (content-based, so cross-dataset pairs are fine). Every hit is labeled with its dataset and keeps the per-dataset score breakdown.
- Protected datasets: locked datasets are **skipped with a note** (never accept passwords in federated tool args — that would undo the v3.0.0 decision to keep passwords out of tool signatures). "all"
  expands to unlocked datasets only.
- Result identity: `(dataset, source, page, chunk_index, time-window)` keys — the twin-collapse logic already keys this way, so cross-dataset dedup collapses on the dataset-qualified tuple.

**Files.** `mcp_server.py`, `api_server.py`, `rag_system.py` (merge helper), docs.

**Effort.** M.

---

## Deliberately not planned

- **Server-side RAG generation endpoint** — generation lives cleanly in the clients (Open WebUI, opencode) and the server stays a retrieval/memory service. Revisit only if a headless/automation
  consumer needs one-shot answers.
- **Per-dataset embedder overrides** — would multiply the fingerprint-guard matrix for little gain; Recreate already handles embedder migrations.

## Suggested shipping order

Preference order is 1→8, but front-loading the small items in each tier gets value out early without blocking the big ones:

1. **#6 memory tools** (S) and **#1 filtered search** (S/M) — independent, quick wins.
2. **#4 import** + **#5 sync** (S/M each) — close the two ops loops.
3. **#2 hybrid BM25** (M/L) — the deep one; do it with a ViDoRe A/B.
4. **#3 OCR** (S/M) — pairs naturally with #2 (OCR text feeds the BM25 lane).
5. **#7 metrics** (S/M) and **#8 federated search** (M) — metrics ideally land *before* #2 so its ingest-cost impact is measurable from day one.
