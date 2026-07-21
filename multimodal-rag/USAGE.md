# MultiModal RAG — Usage Guide

## 1. HTML Frontend (Dataset Manager UI)

Start it:

```bash
cd /workspace/src/multimodal_rag
python -m uvicorn html_frontend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

| Action | How |
|---|---|
| **Create dataset** | Type a name + optional description, click Create |
| **Add raw text/JSON** | Paste text or JSON like `{"text": "...", "image": "..."}` or a JSON array |
| **Add URLs** | One per line — URLs are auto-classified as image/audio/video by extension; or paste JSON lines |
| **Upload files** | Pick a file (PDF, image, audio, video, text). PDFs are split page-by-page with images extracted alongside text blocks |
| **Search** | Enter a query with optional `top_k` / reranker params, see scored results |

Uploaded files land in `/data/datasets/<name>/files/`. Every dataset gets its own Qdrant instance on disk at `/data/datasets/<name>/qdrant/`.

---

## 2. PDF Processing

```python
from pdf_processor import PDFProcessor

pp = PDFProcessor()

# Page-level (full page text + all embedded images)
pages = pp.extract_pages("doc.pdf")
# [{"page_num": 1, "text": "...", "images": [{"path": "...", "index": 0}]}, ...]

# Structured blocks (text blocks paired with nearby images)
blocks = pp.extract_text_blocks("doc.pdf")
# [{"page_num": 1, "block_num": 0, "text": "...", "nearby_images": [...]}, ...]

# RAG-ready entries (recommended)
entries = pp.extract_structured_pages("doc.pdf")
# [{"text": "...", "image": "/tmp/...png", "source": "doc.pdf", "page": 1}, ...]
```

The `DatasetManager.add_file()` method uses `extract_structured_pages` automatically for PDFs.

---

## 3. MCP Server

Start it (stdio mode):

```bash
cd /workspace/src/multimodal_rag
python -m mcp_server.server
```

### Exposed tools

**`multimodal_rag_search`**

```json
{
  "dataset_name": "dataset_1",
  "query": "aurora borealis",
  "top_k": 5,
  "use_reranker": false,
  "reranker_top_k": null,
  "base_llm_modalities": ["text"]
}
```

Returns `context` (formatted for LLM consumption) + raw `results` array. If the base LLM doesn't support images/audio/video (set via `base_llm_modalities`), unsupported media is automatically converted:

- **images/video** → Gemma4 describes them → text
- **audio** → Cohere Transcribe 2026-03 transcribes → text

**`list_datasets`**

Returns metadata for all datasets.

### Connect from an MCP client

```json
{
  "mcpServers": {
    "multimodal-rag": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/workspace/src/multimodal_rag"
    }
  }
}
```

---

## 4. Programmatic API

```python
from multimodal_rag import DatasetManager

dm = DatasetManager(base_path="/data")

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

**Preprocessing** (audio→Cohere→text, images/video→Gemma4→text):
- Happens by default when `preprocess=True` (the default). Disable with `preprocess=False` if your embedding model supports the modality natively.

---

## 5. Helm Deployment

```bash
helm install multimodal-rag ./helm \
  --set persistence.size=100Gi \
  --set ingress.hosts[0].host=rag.mycluster.io
```

| Key | Default | Notes |
|---|---|---|
| `persistence.size` | `50Gi` | PVC size for datasets + Qdrant DBs |
| `persistence.storageClass` | `""` | Set if your cluster requires one |
| `frontend.port` | `8000` | HTML UI port |
| `mcp.enabled` | `true` | Deploys MCP container alongside |
| `mcp.port` | `8001` | MCP server port |

Both containers (frontend + MCP) run in one pod sharing a PVC mounted at `/data`.

---

## 6. Data Flow

```
User Input (UI / MCP / API)
  │
  ├─ Text/JSON ──→ DatasetManager.add_documents()
  ├─ URLs ──────→ DatasetManager.add_documents()
  │
  ├─ PDF ──────→ PDFProcessor.extract_structured_pages()
  │                └─ page-by-page: text blocks + paired images/charts
  │
  ├─ Audio ────→ PreprocessingPipeline (Cohere) → text
  ├─ Image ────→ PreprocessingPipeline (Gemma4, optional) → text description
  └─ Video ────→ PreprocessingPipeline (Cohere + Gemma4) → text
                   │
                   ▼
              Qwen3-VL-Embedding-8B → Qdrant (on PVC)
                   │
Search ──────────► Embed query → Qdrant similarity
                   │
              [Optional] Qwen3-VL-Reranker-8B (re-rank top_k)
                   │
              [Optional] Gemma4/Cohere for unsupported modalities
                   │
              Context → DeepSeek-V4-Flash → Answer (with source/page refs)
```
