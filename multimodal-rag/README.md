# Multimodal RAG

End-to-end multimodal retrieval-augmented generation: ingest documents
in 17+ formats (text, PDF, images, video, audio, code, tables, office
docs, and more), embed them into a joint multimodal vector space, and
retrieve at query time with optional cross-encoder reranking — all
exposed via a REST API, an HTML frontend, and an MCP server.

<div align="center"><img src="./documentation/rag_system_flow-1.png" width="700" alt="RAG system flow"></div>

---

## Features

- **Joint multimodal embedding** (text, image, video) via Qwen3-VL-Embedding-8B — search with any combination of modalities
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

```bash
# Deploy to Kubernetes (see documentation/DEPLOYMENT.md for full guide)
export DOMAIN_NAME="your-domain.com"
envsubst < helm/values.yaml > values-resolved.yaml

helm install multimodal-rag ./helm \
  -f values-resolved.yaml \
  --set models.embedder.url="https://..." \
  --set models.reranker.url="https://..." \
  --set models.vlm.url="https://..." \
  --set models.asr.url="https://..." \
  --set modelSecrets.embedderApiKey="eyJ..."

# Verify
kubectl get pods -l app=rag-mcp-server
kubectl port-forward deployment/rag-mcp-server 8000:8000
# → http://localhost:8000
```

The image does not bundle any ML models — it connects to remote model
endpoints (embedder, reranker, VLM, ASR) configured at runtime via
environment variables. See `documentation/DEPLOYMENT.md` for details.
