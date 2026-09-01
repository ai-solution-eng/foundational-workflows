# Usage Guide

How to use the HTML frontend and the Python programmatic API. For MCP tool access see [documentation/MCP.md](documentation/MCP.md); for deployment see
[documentation/DEPLOYMENT.md](documentation/DEPLOYMENT.md).

---

## 1. HTML Frontend (Dataset Manager UI)

Start it (from the repo root, with the package on `PYTHONPATH`):

```bash
cd /home/andrew/Code/HPE/MultimodalRAG
PYTHONPATH=src python -m multimodal_rag.api_server --host 0.0.0.0 --port 8000
```

`MEDIA_TOKEN_SECRET` must be set (the server refuses to start without it): `MEDIA_TOKEN_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")`. Open `http://localhost:8000` (the HTML
frontend lives at `/`, served from `src/multimodal_rag/templates/index.html`).

| Action | How |
|---|---|
| **Create dataset** | Type a name + optional description, click Create |
| **Add raw text/JSON** | Paste text or JSON like `{"text": "...", "image": "..."}` or a JSON array |
| **Add URLs** | One per line — URLs are auto-classified as image/audio/video by extension; or paste JSON lines |
| **Upload files** | Pick a file (PDF, image, audio, video, text). PDFs are split page-by-page with images extracted alongside text blocks |
| **Search** | Enter a query with optional `top_k` / reranker params, see scored results |

Uploaded files land in `/data/datasets/<name>/files/`. Every dataset gets its own Qdrant collection.

---

## 2. Programmatic API

```python
from multimodal_rag import DatasetManager
from multimodal_rag.model_config import build_all

# An embedder is required — DatasetManager refuses to start without one
# (set MODEL_EMBEDDER_NAME / MODEL_EMBEDDER_URL first; reranker/vlm/asr
# are optional and built as None when their _URL env vars are unset).
embedder, reranker, vlm, asr = build_all()
dm = DatasetManager(base_path="/data", embedder=embedder, reranker=reranker, vlm=vlm, asr=asr)

# Create
dm.create_dataset("dataset_1", "My research papers")

# Add documents (text, URLs, or dicts with media keys)
dm.add_documents("dataset_1", [
    "The aurora borealis over a snowy mountain",
    {"image": "https://fastly.picsum.photos/id/901/3517/1726.jpg"},
    {"text": "A caption", "image": "https://..."},
])

# Add a file (PDF, image, audio, video, text)
dm.add_file("dataset_1", "/path/to/paper.pdf")

# Search
results = dm.search(
    "dataset_1",
    "aurora",
    top_k=5,
    use_reranker=True,
    reranker_top_k=3,
)
# [{"content": "...", "score": 0.92}, ...]

# List / delete
dm.list_datasets()
dm.delete_dataset("dataset_1")
```

Input format follows the HuggingFace API convention: strings are auto-detected (plain text stays text, URLs/data URIs/local paths to media files are recognised). For dicts, the `image`, `video` and
`audio` keys accept either a single URL or a list for multiple items:

```python
inputs = [
    "A caption as plain text",
    "https://example.com/image.jpg",
    {"text": "A caption", "image": "https://example.com/image.jpg"},
    {"text": "Two images", "image": ["https://.../a.jpg", "https://.../b.jpg"]},
]
```

**Preprocessing** (audio→Cohere→text, images/video→Gemma4→text):
- Happens by default when `preprocess=True` (the default). Disable with `preprocess=False` if your embedding model supports the modality natively.

---

## 3. REST API

The API server exposes 35 REST endpoints. Key ones:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Liveness/readiness |
| `POST` | `/api/datasets` | Create (name, description, caption_with_asr, caption_with_vlm, keep_originals, password) |
| `GET` | `/api/datasets` | List all |
| `GET` | `/api/datasets/{name}` | Get one (uses `X-Dataset-Password` header) |
| `DELETE` | `/api/datasets/{name}` | Delete dataset + Qdrant collection |
| `POST` | `/api/datasets/{name}/documents` | Add raw text/dict docs |
| `POST` | `/api/datasets/{name}/files` | Single file upload (multipart) |
| `POST` | `/api/datasets/{name}/batch-files` | Multi-file upload with SSE progress |
| `POST` | `/api/datasets/{name}/batch-urls` | S3/HTTP URL ingestion with SSE |
| `GET` | `/api/datasets/{name}/search` | Text search (`q`, `top_k`, `use_reranker`) |
| `POST` | `/api/datasets/{name}/search` | Multimodal search (body: text + image/video/audio) |
| `GET` | `/api/datasets/{name}/documents` | List stored docs |
| `DELETE` | `/api/datasets/{name}/documents/{doc_id}` | Delete single doc |
| `GET` | `/api/datasets/{name}/files/{filepath}` | Serve stored file |

See [documentation/FEATURES.md](documentation/FEATURES.md) for full endpoint details and search parameters.
