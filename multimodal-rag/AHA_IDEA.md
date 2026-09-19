# Aha! Idea — Built-in Multimodal RAG + Agent Long-Term Memory for HPE Private Cloud AI

> Paste-ready Aha! idea submission. One-line summary for the idea **Name** field; everything else maps to
> the **Description** / **Business value** fields.

**Idea name:** Built-in multimodal RAG service for PCAI — hybrid sparse+dense retrieval, 17+ input formats, and per-user agent long-term memory

**Submitted by:** AI Solutions Engineering — Multimodal RAG v4.0.3 (chart `rag-mcp-server`), deployed and validated on PCAI

---

## Description (the ask)

Make Multimodal RAG a **built-in foundational workflow of HPE Private Cloud AI**: a Kubernetes-native retrieval service that turns PCAI's stored documents *and media* — PDFs (including scanned), images, video, audio, code, logs, office docs — into a citation-grounded knowledge layer, with hybrid **sparse (BM25) + dense** retrieval and a **per-user long-term memory store** for agents.

The application already runs on PCAI today via the standard helm-first flow: import the packaged chart once, configure everything through the **Helm Values** editor (model endpoints come from MLIS; ingress/auth flow through the ezaf-gateway + oauth2-proxy SSO), and apply. The ask is to promote it from "importable package" to **first-class platform capability**, with:

1. **Catalog placement** as a PCAI foundational workflow (single chart import, values-driven configuration, three scale variants: single-replica / medium / large).
2. **MLIS recipes** for the validated embedding stack (Qwen3-VL-Embedding-8B, Qwen3-VL-Reranker-8B) so the required embedder endpoint is a one-click model deployment.
3. **Native MCP tool registry in PCAI Open WebUI** (today the integration ships as an Open WebUI filter; a platform MCP registry removes the extension entirely).
4. **A platform SSO-identity propagation pattern** (trusted auth-proxy identity headers) so per-user data isolation — already implemented — becomes a PCAI-wide convention.

## Why this merits being built in

### 1. Performance — measured, not promised

- **Dynamic request batching** for query embeddings, with a batch-guarded idle early-flush: A/B measured on the scale shape (4 replicas × 4 workers) shows single-query mean latency of **238 ms (down from ~370–550 ms)** and **49.3 req/s @ N=100, 72.8 req/s @ N=250 (p99 5.2 s vs 51 s for the unguarded variant)** at 100% success — bursts stay batched; only stalled small queues flush early.
- **Hybrid retrieval in one round-trip**: dense + sparse BM25 lanes fuse server-side in Qdrant (RRF) — no client-side re-ranking pass, no second query; true dense cosines are recomputed from vectors returned in the same fusion request, at zero extra query cost.
- **Media-lite reranking**: the cross-encoder scores candidates on their text/caption representation without transferring heavy base64 media payloads; full media is back-filled only onto the surviving top-k. Two-phase media fetch and bounded LRU caches extend the same principle ingest-to-query.
- **Concurrent ingest** (configurable file-level preprocessing parallelism) with deterministic ordering; dedicated Qdrant/media thread pools keep sync I/O off the event loop; deferred dataset-count sync removes N sequential Qdrant round-trips from the hottest listing endpoint.
- **Full Prometheus `/metrics`** (request latency by route, ingest throughput, Qdrant op latency, cache hit/miss, hybrid-vs-dense counts) with an opt-in ServiceMonitor — capacity planning is observable out of the box, and three chart variants cover single-node PCAI through multi-replica Qdrant clusters.

### 2. Security — built for regulated deployments

- **Short-lived HMAC-signed media tokens** are mandatory (both servers refuse to start without the token secret); the legacy clear-text `?password=` URL suffix was removed entirely. Password-protected datasets, per-identity unlock scoping, and a per-identity password-failure throttle (429) come standard.
- **SSRF-hardened by default**: private/link-local/cloud-metadata targets are blocked for every fetched URL at ingest *and* query time, with DNS-rebinding pinning (the fetch connects to the IP the policy validated), redirect re-validation, and streamed size caps for remote downloads, media fetches, archives, EPUBs, and PDF rasters — the zip-bomb / gzip-bomb / huge-pixmap classes of attack fail closed.
- **Destructive routes are password-gated**; multi-user **per-identity API keys bound to dataset ACLs** (opt-in) enforce fail-closed dataset access on both the REST and MCP surfaces, with rotation without restart. Key material can ride the chart's Secret wiring rather than values in plain text.
- **Deploys hardened**: read-only rootfs, no privilege escalation, dropped capabilities, no service-account token automount; metrics endpoints can require auth; S3 ingestion supports bucket allowlists.
- **Fits the PCAI security model natively**: traffic enters through the ezaf-gateway with oauth2-proxy SSO, and user identity (when the enforcing proxy is confirmed) propagates into per-user data isolation — including memory isolation (below).

### 3. Feature set — inputs and embeddings

- **17+ input formats** with format-specific chunking: PDF (page-by-page text + figure/chart extraction + ToC/index noise filtering + **OCR fallback for scanned pages**), images, video (overlapping segments), **audio via ASR transcription**, text/markdown, JSON, XML, YAML, CSV/Excel, code (16 languages), HTML, Office documents, Jupyter notebooks, EPUB, log files, and nested archives — plus http(s)/S3/URL ingestion with S3 reconciliation (prune deleted sources) and ingest webhooks.
- **Joint multimodal embedding** (text + image + video in one vector space, Qwen3-VL-Embedding-8B), with token-aware dynamic chunk budgets for prose vs structured content.
- **Sparse + dense, together**: every new collection carries a **dense vector and a sparse BM25 vector**; text queries run server-side **RRF fusion** over both lanes — dense semantic matching where paraphrase wins, lexical BM25 where it can't (code identifiers, log error codes, JSON/YAML keys). BM25 document frequencies are maintained per dataset, including decrementing on deletes, so sparse quality doesn't drift.
- **Dual-embedding "twins"**: PDFs get a text-only twin (text queries match), media gets a caption twin (caption wording is searchable alongside the raw-media embedding) — a practical answer to the classic multimodal-RAG recall gap.
- **Retrieval-quality surface**: optional cross-encoder reranking, server-side metadata filters (file type, severity, source prefix, date range), federated multi-dataset search with one merged rerank, and automatic modality conversion (media → VLM description, audio → transcript) for any downstream LLM. Results carry honest `score_kind` labeling (`rrf` / `cosine` / `reranker`) so downstream agents and auditors can trust what a score means.
- **Operational completeness**: dataset export/import with re-embedding restore, backup CronJob to S3/MinIO, dedup (cosine ≥ 0.995), and a 16-tool MCP server (streamable-http/SSE/stdio) alongside the REST API and HTML dashboard.

### 4. Long-term memory — PCAI agents that remember

- A **per-user, LLM-curated long-term memory store** exposed through the same MCP server: curated memories plus automatic session histories, with recall, list/filter (kind, tags), explicit-ID deletion (no similarity-directed deletion — nothing is removed that wasn't seen listed), and per-session wipe.
- **SSO-backed multi-user isolation**: identity resolves from the auth proxy (or per-identity API keys), so each Open WebUI or opencode user's memories, unlocks, and throttle buckets are their own — a shared PCAI deployment behaves like a personal memory per user, not a shared notepad.
- Inlet/outlet hooks in the Open WebUI extension give every PCAI chat session durable memory without user setup — currently the only memory path of its kind on the platform.

## Business value

- **Broadens what PCAI can be sold on**: enterprise knowledge today is mostly PDFs, scans, recordings, and code. This makes all of it retrievable — with citations — from one platform service.
- **Regulated-industry ready by design**: private deployment keeps data in-cluster; signed tokens, password gates, ACLs, and SSRF hardening map directly onto the compliance reviews we hit in finance, healthcare, and government engagements. One live pattern: AI-assisted document review where every drafted answer carries a page-level citation and confidence — turning review-time savings into measurable FTE hours per 1,000 reviews.
- **Differentiated agent experience**: Open WebUI + per-user long-term memory is a demo that lands with customers in minutes and is a tangible reason to choose PCAI over assembled point tools.
- **Zero new infrastructure**: reuses MLIS model endpoints, platform SSO, PVC storage, and the PCAI observability stack; the image ships no models, so model refreshes are independent of the service.

## Evidence / demand

- Deployed and running on PCAI with a North American industrial-technology customer (alongside Open WebUI, SQL Handler, and Model Downloader packages).
- Validated against a hosted-trial engagement in a FISMA-High government market where citation-grounded review and private-data posture were the deciding criteria.
- [Video demonstration](https://storage.googleapis.com/ai-solution-engineering-videos/public/MultimodalRag.mkv) (models, dataset ingestion, Open WebUI integration, long-term memory).

## Success criteria

- Package importable and Ready on PCAI from values alone (already true — target: catalog-listed).
- Hybrid (sparse+dense) retrieval measurable in `/metrics` (`SEARCH_HYBRID`) on every new dataset (already true).
- Per-user memory isolation verified under the platform SSO proxy (already true when identity headers are enforced).
- MLIS one-click embedder/reranker deployment replaces manual endpoint setup (new).

## References

- Package README + `documentation/FEATURES.md`, `DEPLOYMENT.md`, `MCP.md`, `MEMORY.md` (format, deployment, tooling, and memory references)
- `CHANGELOG.md` v3.x–4.x for the measured A/B performance figures and the security-audit trail

---

## Aha! form fields (paste-ready)

### What solution do you suggest?

Build Multimodal RAG into PCAI as a first-class foundational workflow: a packaged, values-driven service that turns PCAI's documents *and media* into a citation-grounded knowledge layer with agent memory.

- **Ingest 17+ formats** — PDF (including OCR for scanned pages), images, video segments, audio via ASR, code, logs, office docs, archives — into one joint multimodal vector space (embedder/reranker deployed as MLIS endpoints).
- **Hybrid retrieval**: dense + sparse BM25 with server-side RRF fusion, optional cross-encoder reranking, metadata filters, federated multi-dataset search.
- **Per-user long-term memory** for Open WebUI and opencode agents (LLM-curated memories + session histories, SSO-backed isolation), exposed through a 16-tool MCP server alongside REST and a web dashboard.
- **Enterprise security built in**: HMAC-signed short-lived media tokens, password-gated destructive routes, per-identity API keys bound to dataset ACLs, SSRF/zip-bomb guards, Prometheus metrics, backup/restore.
- **Zero new infrastructure**: import the chart once and configure via Helm Values; three scale variants (single-replica → multi-replica Qdrant); SSO flows through the platform gateway.

### What is the problem that the user cannot solve with today's features?

PCAI already ships a built-in RAG workflow, but it is **text-only** — "chat with clean-text documents" works, and everything past that hits a wall. With today's features users cannot:

- **Search anything that isn't text.** Images, video, and audio are invisible to retrieval, and scanned PDFs yield nothing (no text layer to extract, no OCR). A customer's media library — training videos, call recordings, engineering photos, fax-grade scans — simply isn't searchable.
- **Match on exact identifiers.** Retrieval is dense-only, so it misses what keyword search exists for — error codes, part/claim numbers, JSON/YAML keys, function names. There is no hybrid sparse+dense lane.
- **See what the model is talking about.** Retrieved figures and images aren't passed to the LLM (no VLM conversion) and audio isn't transcribed — the text around a chart is answerable, the chart itself isn't.
- **Give agents memory.** Chats start from zero each session; user preferences, decisions, and prior context can't persist per user anywhere on the platform.
- **Trust results in regulated settings.** No citation-grade provenance (page-level sources) or honest score labeling to audit an answer against.

### How do they work around and/or solve the problem? (If they can)

- **Pre-convert everything to text, then feed the built-in RAG.** Scripted/manual OCR for scans, third-party transcription for audio/video, captions written for images — then upload the flattened text. Slow and lossy (layout, figures, and tone are gone), creates duplicates that drift from the source, and collapses at any real volume.
- **Scope use cases down to clean text.** Teams quietly exclude media-heavy archives — recordings, scans, image decks — from AI search entirely. The built-in feature is fine for meeting notes and well-formed PDFs and unusable beyond that.
- **DIY multimodal stack** (notebooks + open-source chunking/embedding + a separately deployed vector DB). Viable for demos; weeks of effort, unsupported, and all of the hardening (SSRF, multi-tenant isolation, media access control) lands on the customer — who still ends up dense-only unless they build hybrid retrieval too.
- **External SaaS RAG/embedding APIs** — fastest to try, but content leaves the cluster; a non-starter for the regulated customers PCAI targets.
- **Memory**: re-paste context and re-state preferences every session, or maintain one shared notes file with no per-user isolation.
- **Validated alternative**: Multimodal RAG v4.0.3 already runs on PCAI via chart import + Helm Values — 17+ formats incl. OCR/ASR, hybrid sparse+dense with server-side RRF, reranking, per-user MCP memory, signed-token security — and is deployed with a North American industrial-technology customer. The workaround is proven, which is exactly why we're proposing it be built in.
