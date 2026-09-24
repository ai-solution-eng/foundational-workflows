# REST API Reference

How to drive the Multimodal RAG API server directly from `curl`, Python, or any HTTP client — no HTML frontend required. You can create datasets, upload files, ingest URLs, add raw documents, delete content, and search.

The API server also ships an interactive **Swagger UI** at `/docs` on the running server (e.g. `http://localhost:8000/docs`) with every endpoint, request schema, and a "Try it out" button.

---

## 1. Basics

**Base URL:** the API server listens on port `8000`. Locally (after `kubectl port-forward deployment/rag-mcp-server 8000:8000`) it is `http://localhost:8000`. In-cluster it is the `rag-mcp-server-api` service (see [DEPLOYMENT.md](DEPLOYMENT.md)).

**Authentication:**

- **Dataset password** — for password-protected datasets send the password in the `X-Dataset-Password` request header on every call (JSON endpoints), or as a `password` form field on multipart uploads. `POST /api/datasets/{name}/unlock` verifies the password once and caches it for ~30 min (Redis across pods), so subsequent calls can omit it. Unprotected datasets need none of this. Unlock TTLs are bounded by `RAG_UNLOCK_MAX_TTL` (default `86400` = 24 h; a deployment may set it to `0` to opt into **no-expiry** unlocks — `ttl=0` then persists until `POST /api/datasets/{name}/lock`).
- **Optional API key** — if `RAG_API_KEY` is set on the server, every `/api/*` request must carry `Authorization: Bearer <key>` or `X-RAG-Api-Key: <key>`. Health/probe routes, the HTML pages (the served page embeds the key for its JS), dataset media serving, and staged media stay open. The MCP server runs its **own** middleware instead (keyed by `RAG_API_KEYS` / `MCP_API_KEYS` — see [MCP.md](MCP.md) § 1).
- **Multi-user keys → dataset ACLs (decision D15 — opt-in).** Setting `RAG_API_KEY_CLIENTS="name:key;name:key"` mints per-user keys accepted on **both** the REST and MCP surfaces, and `RAG_DATASET_ACLS="name:ds1,ds2;name2:*"` binds each name to its datasets (`*` = all). Access is **fail-closed** (a registry key with no ACL entry sees no datasets); ACL'd keys cannot reach `/api/admin/*` or create datasets unless granted `*`, while the plain deployment keys (`RAG_API_KEY` + `MCP_API_KEYS` / `RAG_API_KEYS`) keep full admin access. Default (registry unset): single-key behaviour, unchanged. Envs are re-read per request (rotation without restart).
- **Self-service dataset selection (decision D16 — opt-in, gated by `RAG_ACCESS_STORE=1`).** With the access store enabled, the checkbox model replaces operator-only grants: a key's listing shows only its EFFECTIVE datasets (grants ∪ selections — access isolation; names a key cannot use are hidden) and **selects** the ones it wants — public datasets freely, password-protected ones only with their correct password, which is then **saved per identity** (one JSON per key under `{DATA_PATH}/access/`, 0600, atomic writes under a cross-process lock) so every REST and MCP call works without sending the password again. Effective access = **operator ACL ∪ selections** (the ACL is a guaranteed floor; deselect removes only the self-added widening; `RAG_ACCESS_DENY_SELECT="ds1,ds2"` datasets can never be self-selected). Both surfaces enforce the union: the MCP tools' `_require_dataset_acl` and the REST middleware use it, and federated `"all"` expands over it. REST endpoints: `GET /api/access/selections` (the caller's state — never returns passwords), `POST /api/datasets/{name}/select` (optional `{"password": …}` — the one path exempt from the per-dataset ACL pre-check, since it is how access is gained), `POST /api/datasets/{name}/deselect`, `POST /api/access/memory-dataset` (bind/unbind the caller's ★ memory dataset server-side). MCP tools: `select_dataset`, `deselect_dataset`, `set_memory_dataset`. When the store is off: byte-identical D15 behaviour (ACL'd names only, fail-closed).

### The /access page

`GET /access` serves a per-user key page — the key-holder's counterpart to the operator dashboard (`/`). Like `/` and `/manage` it is public: the page IS its own auth boundary. The user pastes **their own** API key (a D15 registry key, or the deployment key), and the page then makes every API call with that key as `X-RAG-Api-Key`. Unlike `/`, the server injects **no** key meta tag into this page — under D15 it is used with per-user keys, and embedding the deployment key would leak admin credentials onto a public page.

What the page does, per key:

- **Dataset list** — `GET /api/datasets` renders exactly the caller's view: ACL-filtered for registry keys (with an "n hidden by access policy" note), each row showing lock state (the `unlocked` flag reflects the caller's own unlock cache).
- **Unlock with a TTL** — per-dataset unlock form calling `POST /api/datasets/{name}/unlock` (TTL options 30 min … 24 h, plus **No expiry (0)** when the deployment opts in via `RAG_UNLOCK_MAX_TTL=0`), and a **Lock** button (`POST /api/datasets/{name}/lock`) to revoke immediately.
- **Memory-dataset star** — a per-dataset ★ stored in the browser (`localStorage['rag-memory-dataset']`) marking which dataset the user's MCP client should use for long-term memory. Client-side preference in this release (the server does not read it yet — see MCP.md); dataset passwords are never stored by the page.

Because the unlock cache is keyed per-caller (D10/D15: a registry key resolves to the stable `key:<name>` identity), an unlock made on this page is visible only to the same key — use the *same* key in your MCP client for the page and the client to share unlock state on the REST surface. The MCP surface keeps its own in-process unlock cache (see MCP.md § notes).

Throughout this document `BASE=http://localhost:8000` and `DATASET=my_dataset`.

---

## 2. Create a dataset

```bash
curl -X POST "$BASE/api/datasets" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "my_dataset",
    "description": "My research papers",
    "caption_with_asr": false,
    "caption_with_vlm": true,
    "keep_originals": true,
    "password": "secret"
  }'
```

Response: `{"status":"ok","dataset":{...}}`

Fields (all optional except `name`):

| Field | Default | Meaning |
|---|---|---|
| `name` | — | Must match `[A-Za-z0-9._-]` and start alphanumeric |
| `description` | `""` | Free-text description |
| `caption_with_asr` | server config (`RAG_CAPTION_WITH_ASR`, chart default `true`) | Transcribe audio tracks from uploaded videos during ingestion (auto-disables when no ASR model is configured) |
| `caption_with_vlm` | server config (`RAG_CAPTION_WITH_VLM`, chart default `true`) | Describe images/videos with the VLM during ingestion (auto-disables when no VLM is configured) |
| `keep_originals` | `true` | Keep full-quality originals on disk after preprocessing |
| `password` | unset | Protect the dataset; all reads/ingests then require it |
| `rrf` | deployment default (`RAG_RRF_DEFAULT`, chart `rag.rrfDefault` — disabled by default) | Optional weighted-RRF defaults `{"dense_weight": …, "sparse_weight": …, "k": …}` (any subset; weights 0.0–10.0 clamped to 3 decimals, k 1–1000). Applied to this dataset's hybrid text searches when a search carries no explicit override. A payload equal to the global default (1.0/1.0, no k) stores nothing. |
| `contextual` | deployment default (`RAG_CONTEXTUAL_DEFAULT`, chart `rag.contextual` — disabled by default) | Enable ingest-time contextual retrieval for this dataset: one small LLM call per real-text chunk writes 1–2 sentences of document-level context, prepended as a `[Document context]:` line before embedding. Requires a VLM model (no-op without one). Affects NEW ingests only — Recreate re-contextualizes existing files. Preview the cost first: `POST /api/admin/datasets/{name}/contextual-preview`. |

> Naming note: dataset names are validated against `^[A-Za-z0-9][A-Za-z0-9._-]*$` (prevents path traversal).

**Python:**

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/api/datasets",
    json={
        "name": "my_dataset",
        "description": "My research papers",
        "caption_with_vlm": True,
        "password": "secret",
    },
)
resp.raise_for_status()
print(resp.json())
```

### Update a dataset (dynamic captioning config)

Captioning settings are **not** frozen at create time — patch them whenever you like; they apply to subsequent ingests/retrievals (no restart, no recreate). Already-ingested content keeps its stored captions.

```bash
curl -X PATCH "$BASE/api/datasets/$DATASET" \
  -H 'Content-Type: application/json' \
  -d '{"caption_with_asr": true, "caption_with_vlm": true}'
```

Any subset of `description`, `caption_with_asr`, `caption_with_vlm`, `keep_originals`, `ocr`, `contextual`, `rrf` may be sent. `rrf` is query-time only (no re-ingest): patch `{"rrf": {"sparse_weight": 3.0}}` to tilt the BM25 lane for this dataset's hybrid searches, or `{"rrf": {}}` to remove the stored default (falls back to the global 1.0/1.0). `contextual` flips ingest-time contextual retrieval (rebuilds the cached RAG like the caption flags); it affects NEW ingests only — run Recreate to re-contextualize existing files.

---

## 3. Add content

### 3.1 Single file upload (multipart)

Supported types: PDF, image (jpg/png/gif/bmp/webp), video (mp4/mkv/avi/mov), audio (mp3/wav/flac/ogg), and text files. Files are processed (chunked / transcribed / described) and embedded into the dataset's Qdrant collection.

```bash
curl -X POST "$BASE/api/datasets/$DATASET/files" \
  -F 'file=@paper.pdf' \
  -F 'password=secret'          # only for protected datasets
```

Response: `{"status":"ok","file":"paper.pdf","chunks":17,...}`.

**Python:**

```python
import httpx

with open("paper.pdf", "rb") as f:
    resp = httpx.post(
        f"http://localhost:8000/api/datasets/{DATASET}/files",
        files={"file": ("paper.pdf", f, "application/pdf")},
        data={"password": "secret"} if protected else {},
    )
    resp.raise_for_status()
```

### 3.2 Batch upload (recommended for many files)

Returns a `job_id` immediately; poll the status endpoint until `status` is `complete` or `error`.

```bash
curl -X POST "$BASE/api/datasets/$DATASET/batch-files" \
  -F 'files=@a.pdf' -F 'files=@b.png' -F 'files=@clip.mp4' \
  -F 'password=secret'
# → {"job_id":"...","status":"uploading","total_files":3}
```

Poll progress (every 2–3 s):

```bash
curl -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/$DATASET/upload-status/<job_id>"
```

The response contains `status`, per-file events, and aggregate counters; stop polling once `status` is `complete` or `error`.

### 3.3 Ingest from URLs (S3 / HTTP)

```bash
curl -X POST "$BASE/api/datasets/$DATASET/batch-urls" \
  -H 'Content-Type: application/json' \
  -H 'X-Dataset-Password: secret' \
  -d '{"urls": ["https://example.com/a.pdf", "s3://bucket/b.jpg"]}'
# → {"job_id":"...","status":"uploading","total_files":2}
```

Poll `GET /api/datasets/$DATASET/upload-status/<job_id>` as above. Note the server-side `INGEST_ALLOW_HOSTS` / `INGEST_BLOCK_PRIVATE_HOSTS` settings (see [README](../README.md) security table) can restrict which hosts are ingestible.

> **Ingest webhooks (opt-in).** Set `RAG_WEBHOOK_URL` and every completed ingest fires one small JSON event (`{"dataset", "doc_count", "status", "timestamp"}`) to that URL — a batch (files or URLs) fires exactly one event, not one per file. `RAG_WEBHOOK_SECRET` adds an `X-RAG-Webhook-Secret` header, and the POST is timeout-capped by `RAG_WEBHOOK_TIMEOUT` (seconds, default 5). Failures are logged, never fatal; unset `RAG_WEBHOOK_URL` (the default) means zero behaviour, and bulk replays (dataset restore/import) are muted.

### 3.4 Add raw text / structured documents

```bash
curl -X POST "$BASE/api/datasets/$DATASET/documents" \
  -H 'Content-Type: application/json' \
  -H 'X-Dataset-Password: secret' \
  -d '[
    "A plain text note",
    {"text": "A caption", "image": "https://example.com/i.jpg"},
    {"text": "Two images", "image": ["https://.../a.jpg", "https://.../b.jpg"]}
  ]'
```

Accepts a JSON array, or a single string/dict. Document dicts may mix `text`, `image`, `video`, `audio` keys; each media key takes a URL, data-URL, or list. Response:

```json
{"status": "ok", "stored_ids": ["...", "..."], "count": 2}
```

---

## 4. List and delete content

### 4.1 List documents (to find IDs)

```bash
curl -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/$DATASET/documents?limit=100"
# → {"documents": [{"id": "9b1d...", "payload": {...}}], "count": 42}
```

`limit` defaults to 50, max 1000.

### 4.2 Delete one document

```bash
curl -X DELETE -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/$DATASET/documents/<doc_id>"
# → {"status": "ok", "deleted": "<doc_id>"}
```

> **About "deleting files":** deletion is at the **document (Qdrant point) level** — the ID you delete is the vector/point ID returned by list-documents, not a filename. This removes the entry from
> search. The on-disk original under `/data/datasets/<name>/files/` is kept (it is reused for retrieval/media display), so there is no per-file-path delete endpoint.

### 4.3 Delete the whole dataset (collection + files)

```bash
curl -X DELETE "$BASE/api/datasets/$DATASET"
# → {"status": "ok", "deleted": "my_dataset"}
```

### 4.4 Download a full dataset backup

```bash
curl -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/$DATASET/export" -o my_dataset-backup.tar.gz
```

Streams a `.tar.gz` containing `meta.json` (password hash stripped), `documents.jsonl` (every Qdrant point as `{"id","payload"}` JSON Lines), and `files/` (all on-disk files referenced by the dataset). Restore by re-adding the documents (`POST /documents`) and files (`POST /batch-files`), or use the management page "Backup" button.

---

## 5. Embedding-model changes and dataset recreate

Vectors are embedded at ingestion time; nothing re-embeds them later, and there is **no automatic rebuild** when you swap the embedding model. Each dataset records the embedder model + dimension in its `meta.json`; if you change the embedder the server **fails loudly** (HTTP 409) on search/ingest for existing datasets instead of silently mixing incompatible vectors. Payload-only reads (dataset listing, `GET /documents`, export) are exempt from the guard — export→import stays usable as the recovery path across a swap. To treat two model ids as the same embedder (e.g. a quantized FP8 redeploy of the same base model), set `models.embedder.extra.alias` in the chart values to the other id — datasets fingerprinted under an alias keep working without a rebuild.

To rebuild a dataset with the new embedder, drop the old collection and re-ingest its on-disk originals (this also re-records the fingerprint):

```bash
curl -X POST "$BASE/api/admin/datasets/$DATASET/recreate"
# → {"job_id":"...","status":"recreating","total_files":N}
```

Poll until `status` is `complete` or `error` (same endpoint as uploads):

```bash
curl "$BASE/api/datasets/$DATASET/upload-status/<job_id>"
```

New datasets created after the swap are unaffected (fresh collection at the new model's dimension). Delete-and-re-upload works too, but `recreate` skips the upload since the originals are already on disk.

---

## 6. Verify with a search

```bash
# Text search (GET)
curl -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/$DATASET/search?q=aurora+borealis&top_k=5&use_reranker=true&reranker_top_k=3"

# Multimodal search (POST) — text + image/video/audio in one query
curl -X POST "$BASE/api/datasets/$DATASET/search" \
  -H 'Content-Type: application/json' \
  -H 'X-Dataset-Password: secret' \
  -d '{
    "query": {"text": "a green sky over mountains", "image": "https://example.com/photo.jpg"},
    "top_k": 10,
    "use_reranker": false
  }'
```

`GET` params: `q` (required), `top_k` (1–100, default 10), `use_reranker` (default false), `reranker_top_k` (1–50, default 3). `POST` accepts the same params in the body plus the `query` dict and an optional `password` field.

**Weighted RRF** (optional, both surfaces): `dense_weight` / `sparse_weight` (0.0–10.0, clamped to 3 decimals — rank-space tilts, NOT score multipliers; trust the order, not the magnitude) and `k` (ranking constant, 1–1000). Omitted params resolve per search: explicit override > the dataset's stored defaults (the `rrf` field above) > global default (1.0/1.0, unpinned k — the request then keeps the historical fusion form unchanged). The response carries `"rrf": {dense, sparse, k, applied}` reporting the EFFECTIVE parameters; `applied=false` when the search degraded to dense-only (hybrid off, no BM25 stats, multimodal query) rather than labelling dense results with weights. Federated `POST /api/search` deliberately takes no weight parameters — per-dataset defaults do not apply there by ruling.

---

## 7. Other useful endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/datasets` | List all datasets with metadata |
| `GET` | `/api/datasets/{name}` | Get one dataset's metadata |
| `PATCH` | `/api/datasets/{name}` | Update metadata (`description`, caption flags) |
| `POST` | `/api/datasets/{name}/verify-password` | Verify a password → 200 / 401 / 403 |
| `POST` | `/api/datasets/{name}/unlock` | Unlock for ~30 min (REST cache is Redis-backed when Redis is enabled) |
| `POST` | `/api/datasets/{name}/lock` | Immediately revoke a cached unlock |
| `POST` | `/api/datasets/{name}/media-token` | Mint a short-lived dataset-scoped HMAC media token (`?token=`) so the password never travels in a URL |
| `GET` | `/api/datasets/{name}/files/{path}` | Serve a stored file (header, `?password=`, or `?token=`) |
| `GET` | `/api/datasets/{name}/export` | Download full dataset backup (`.tar.gz`: `meta.json` + `documents.jsonl` + `files/`) |
| `GET` | `/api/datasets/{name}/documents/download?format=md\|jsonl` | Download every document as one file — readable Markdown (default) or `{"id","text","metadata"}` JSONL; no binary files, heavy base64 media stripped |
| `POST` | `/api/admin/datasets/{name}/recreate` | Rebuild a dataset from its on-disk files with the current embedder (drops old collection, re-embeds; poll `upload-status/{job_id}`; password-protected datasets require the `X-Dataset-Password` header) |
| `POST` | `/api/admin/datasets/{name}/contextual-preview` | Cost preview for enabling contextual retrieval — a LABELED ESTIMATE (live point count ×2 for hybrid collections, one context call's token math, configured VLM name); read-only, no side effects |
| `POST` | `/api/admin/datasets/{name}/migrate-tier-schema` | One-time migration of a dataset's points to the three-tier media schema (idempotent; password-gated) |
| `GET` | `/api/admin/health` | Health: model endpoints, Qdrant status + per-replica shard placement, PVC |
| `GET` | `/api/admin/models` | Discovered model names per role |

Full endpoint inventory: [FEATURES.md](FEATURES.md) § API Server.