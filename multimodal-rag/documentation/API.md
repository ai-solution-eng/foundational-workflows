# REST API Reference

How to drive the Multimodal RAG API server directly from `curl`, Python,
or any HTTP client — no HTML frontend required. You can create datasets,
upload files, ingest URLs, add raw documents, delete content, and search.

The API server also ships an interactive **Swagger UI** at `/docs` on the
running server (e.g. `http://localhost:8000/docs`) with every endpoint,
request schema, and a "Try it out" button.

---

## 1. Basics

**Base URL:** the API server listens on port `8000`. Locally (after
`kubectl port-forward deployment/rag-mcp-server 8000:8000`) it is
`http://localhost:8000`. In-cluster it is the `rag-mcp-server-api`
service (see [DEPLOYMENT.md](DEPLOYMENT.md)).

**Authentication:**

- **Dataset password** — for password-protected datasets send the
  password in the `X-Dataset-Password` request header on every call
  (JSON endpoints), or as a `password` form field on multipart uploads.
  `POST /api/datasets/{name}/unlock` verifies the password once and
  caches it for ~30 min (Redis across pods), so subsequent calls can
  omit it. Unprotected datasets need none of this.
- **Optional API key** — if `RAG_API_KEY` is set on the server, every
  `/api/*` request must carry `Authorization: Bearer <key>` or
  `X-RAG-Api-Key: <key>`. Health/probe routes, the HTML pages (the
  served page embeds the key for its JS), dataset media serving, and
  staged media stay open. The MCP server is not covered by this
  middleware.

Throughout this document `BASE=http://localhost:8000` and
`DATASET=my_dataset`.

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

> Naming note: dataset names are validated against
> `^[A-Za-z0-9][A-Za-z0-9._-]*$` (prevents path traversal).

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

Captioning settings are **not** frozen at create time — patch them whenever
you like; they apply to subsequent ingests/retrievals (no restart, no
recreate). Already-ingested content keeps its stored captions.

```bash
curl -X PATCH "$BASE/api/datasets/$DATASET" \
  -H 'Content-Type: application/json' \
  -d '{"caption_with_asr": true, "caption_with_vlm": true}'
```

Any subset of `description`, `caption_with_asr`, `caption_with_vlm`,
`keep_originals` may be sent.

---

## 3. Add content

### 3.1 Single file upload (multipart)

Supported types: PDF, image (jpg/png/gif/bmp/webp), video
(mp4/mkv/avi/mov), audio (mp3/wav/flac/ogg), and text files. Files are
processed (chunked / transcribed / described) and embedded into the
dataset's Qdrant collection.

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

Returns a `job_id` immediately; poll the status endpoint until
`status` is `complete` or `error`.

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

The response contains `status`, per-file events, and aggregate counters;
stop polling once `status` is `complete` or `error`.

### 3.3 Ingest from URLs (S3 / HTTP)

```bash
curl -X POST "$BASE/api/datasets/$DATASET/batch-urls" \
  -H 'Content-Type: application/json' \
  -H 'X-Dataset-Password: secret' \
  -d '{"urls": ["https://example.com/a.pdf", "s3://bucket/b.jpg"]}'
# → {"job_id":"...","status":"uploading","total_files":2}
```

Poll `GET /api/datasets/$DATASET/upload-status/<job_id>` as above. Note
the server-side `INGEST_ALLOW_HOSTS` / `INGEST_BLOCK_PRIVATE_HOSTS`
settings (see [README](../README.md) security table) can restrict which
hosts are ingestible.

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

Accepts a JSON array, or a single string/dict. Document dicts may mix
`text`, `image`, `video`, `audio` keys; each media key takes a URL,
data-URL, or list. Response:

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

> **About "deleting files":** deletion is at the **document (Qdrant
> point) level** — the ID you delete is the vector/point ID returned by
> list-documents, not a filename. This removes the entry from search.
> The on-disk original under `/data/datasets/<name>/files/` is kept
> (it is reused for retrieval/media display), so there is no
> per-file-path delete endpoint.

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

Streams a `.tar.gz` containing `meta.json` (password hash stripped),
`documents.jsonl` (every Qdrant point as `{"id","payload"}` JSON Lines), and
`files/` (all on-disk files referenced by the dataset). Restore by re-adding
the documents (`POST /documents`) and files (`POST /batch-files`), or use the
management page "Backup" button.

---

## 5. Embedding-model changes and dataset recreate

Vectors are embedded at ingestion time; nothing re-embeds them later, and
there is **no automatic rebuild** when you swap the embedding model. Each
dataset records the embedder model + dimension in its `meta.json`; if you
change the embedder the server **fails loudly** (HTTP 409) on search/ingest
for existing datasets instead of silently mixing incompatible vectors.

To rebuild a dataset with the new embedder, drop the old collection and
re-ingest its on-disk originals (this also re-records the fingerprint):

```bash
curl -X POST "$BASE/api/admin/datasets/$DATASET/recreate"
# → {"job_id":"...","status":"recreating","total_files":N}
```

Poll until `status` is `complete` or `error` (same endpoint as uploads):

```bash
curl "$BASE/api/datasets/$DATASET/upload-status/<job_id>"
```

New datasets created after the swap are unaffected (fresh collection at the
new model's dimension). Delete-and-re-upload works too, but `recreate`
skips the upload since the originals are already on disk.

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

`GET` params: `q` (required), `top_k` (1–100, default 10),
`use_reranker` (default false), `reranker_top_k` (1–50, default 3).
`POST` accepts the same params in the body plus the `query` dict and an
optional `password` field.

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
| `POST` | `/api/admin/datasets/{name}/migrate-tier-schema` | One-time migration of a dataset's points to the three-tier media schema (idempotent; password-gated) |
| `GET` | `/api/admin/health` | Health: model endpoints, Qdrant status + per-replica shard placement, PVC |
| `GET` | `/api/admin/models` | Discovered model names per role |

Full endpoint inventory: [FEATURES.md](FEATURES.md) § API Server.