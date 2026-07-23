# Deployment Guide

## Prerequisites

- Docker (or Podman) for building the image
- A container registry you can push to (Docker Hub, GHCR, internal registry, etc.)
- A Kubernetes cluster with Helm 3 installed
- `kubectl` configured to access your cluster

---

## 1. Build and push the Docker image

```bash
# From the repo root
docker build -t your-registry/rag-api-server:latest .

# Push to your registry
docker push your-registry/rag-api-server:latest
```

> **Note on models**: The image does not bundle any ML models. It connects to remote model endpoints configured at runtime via environment variables. The defaults in `values.yaml` point to models hosted on the PCAI internal cluster.

---

## 2. Configure deployment values

All configuration lives in `helm/values.yaml`. Key settings:

```bash
# Set your image (required)
image.repository=your-registry/rag-api-server
image.tag=latest

# PVC sizes — adjust for your dataset scale
persistence.data.size=200Gi        # Uploaded files + dataset metadata
persistence.qdrant.size=150Gi      # Qdrant vectors + index

# Model configuration — each model has name, url, className, and
# optional extra kwargs.  Set url to "" to disable a component.
models.embedder.name="Qwen/Qwen3-VL-Embedding-8B"
models.embedder.url="https://..."          # required
models.embedder.className="MultiModalEmbeddings"
models.reranker.name="Qwen/Qwen3-VL-Reranker-8B"
models.reranker.url="https://..."          # required
models.reranker.className="MultiModalReranker"
models.vlm.name="google/gemma-4-31B-it"
models.vlm.url="https://..."               # required
models.asr.name="CohereLabs/cohere-transcribe-03-2026"
models.asr.url="https://..."               # required

# Model API keys (stored in a Kubernetes Secret, never in the image)
modelSecrets.embedderApiKey="eyJ..."
modelSecrets.rerankerApiKey="eyJ..."
modelSecrets.vlmApiKey="eyJ..."
modelSecrets.asrApiKey="eyJ..."

# RAG pipeline defaults
rag.captionVideo=false   # Transcribe video audio during ingestion
rag.remote=true           # Use remote model URLs

# MCP server (sidecar)
mcp.enabled=true
mcp.port=8001

# PCAI / EZUA (Istio-based ingress)
ezua.enabled=true
ezua.domainName="your-domain.com"             # Must match cluster domain
ezua.virtualService.endpoint="rag-mcp-server.your-domain.com"
ezua.virtualService.istioGateway="istio-system/ezaf-gateway"
ezua.virtualService.timeout=660s
ezua.authorizationPolicy.namespace="istio-system"
ezua.authorizationPolicy.providerName="oauth2-proxy"

# Note: ${DOMAIN_NAME} in values.yaml is substituted at deploy time.
#   export DOMAIN_NAME="your-domain.com"
#   envsubst < values.yaml > values-resolved.yaml
#   helm install ... -f values-resolved.yaml

# Resources — scale based on expected Qdrant load
resources.app.limits.memory=4Gi
resources.qdrant.limits.memory=48Gi   # ~40 GiB needed for 1M × 4096-dim vectors
```

You can set these via `--set` flags or edit `values.yaml` directly.

---

## 3. Install the Helm chart

```bash
cd helm/

# 1. Resolve ${DOMAIN_NAME} in values.yaml
export DOMAIN_NAME="your-domain.com"
envsubst < values.yaml > values-resolved.yaml

# 2. Dry-run first to validate
helm install multimodal-rag . \
  --dry-run --debug -f values-resolved.yaml \
  --set image.repository=your-registry/rag-api-server

# 3. Actual install (model URLs and API keys required)
helm install multimodal-rag . \
  -f values-resolved.yaml \
  --set image.repository=your-registry/rag-api-server \
  --set image.tag=latest \
  --set models.embedder.url="https://..." \
  --set models.reranker.url="https://..." \
  --set models.vlm.url="https://..." \
  --set models.asr.url="https://..." \
  --set modelSecrets.embedderApiKey="eyJ..." \
  --set modelSecrets.rerankerApiKey="eyJ..." \
  --set modelSecrets.vlmApiKey="eyJ..." \
  --set modelSecrets.asrApiKey="eyJ..."
```

Or with many overrides:

```bash
export DOMAIN_NAME="your-domain.com"
envsubst < values.yaml > values-resolved.yaml

helm install multimodal-rag . \
  -f values-resolved.yaml \
  --set image.repository=your-registry/rag-api-server \
  --set image.tag=v1.0.0 \
  --set models.embedder.url="https://..." \
  --set models.reranker.url="https://..." \
  --set models.vlm.url="https://..." \
  --set models.asr.url="https://..." \
  --set modelSecrets.embedderApiKey="eyJ..." \
  --set modelSecrets.rerankerApiKey="eyJ..." \
  --set modelSecrets.vlmApiKey="eyJ..." \
  --set modelSecrets.asrApiKey="eyJ..." \
  --set persistence.data.size=500Gi \
  --set persistence.qdrant.size=200Gi \
  --set resources.qdrant.limits.memory=64Gi
```

> **Tip**: Put model URLs and API keys in a separate `secrets.yaml` values file and
> pass it with `-f` to keep them out of your shell history:
> ```bash
> helm install multimodal-rag . -f values.yaml -f secrets.yaml
> ```

---

## 4. Verify the deployment

```bash
# Check pods are running
kubectl get pods -l app=rag-mcp-server

# Check the API server logs (models initializing)
kubectl logs -l app=rag-mcp-server -c rag-api-server

# Check the MCP server logs (if enabled)
kubectl logs -l app=rag-mcp-server -c rag-mcp-server

# Check Qdrant is ready
kubectl get pods -l app=rag-mcp-server-qdrant

# Port-forward to test locally
kubectl port-forward deployment/rag-mcp-server 8000:8000

# Health check
curl http://localhost:8000/healthz
# → {"status": "ok"}

# List datasets (empty initially)
curl http://localhost:8000/api/datasets
# → {"datasets": []}
```

---

## 5. Access the web UI

If EZUA (Istio) is enabled, the service is available at the VirtualService
endpoint (e.g. `https://rag-mcp-server.your-domain.com`).  Authentication is
handled by the `oauth2-proxy` AuthorizationPolicy.

Otherwise port-forward:

```bash
kubectl port-forward deployment/rag-mcp-server 8000:8000
# → http://localhost:8000
```

The UI lets you:
- **Create datasets** with optional video captioning toggle
- **Upload files** (PDF, images, videos, audio, text) via drag-and-drop
- **Add text** in the standardized multimodal format
- **Search** with configurable `top_k` and reranker toggle
- **Browse stored documents** in each dataset

---

## 6. Connect an MCP client

When `mcp.enabled=true` (default), the MCP server runs as a sidecar
container exposing `streamable-http` transport on port 9090 at `/mcp`.

For the full tool list, connection configs (opencode, Claude Desktop,
Open WebUI, stdio), and the long-term memory setup, see:

- **[MCP.md](MCP.md)** — all 9 MCP tools + connection configs for any client
- **[MEMORY.md](MEMORY.md)** — per-user long-term memory setup (opencode + Open WebUI)

### Quick reference

```json
{
  "mcpServers": {
    "multimodal-rag": {
      "url": "https://rag-mcp-server.your-domain.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

---

## 7. Architecture overview

```
┌─────────────────────────────────────────────────────┐
│  Pod                                                 │
│                                                      │
│  ┌──────────────────┐   ┌──────────────────┐        │
│  │  API Server       │   │  MCP Server       │       │
│  │  port 8000        │   │  port 8001        │       │
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
              │  Qdrant (port 6333) │  ← StatefulSet
              │  ┌──────────────┐  │
              │  │ PVC /qdrant  │  │  ← vectors + index
              │  │   storage    │  │
              │  └──────────────┘  │
              └───────────────────┘
```

Both the API server and MCP server connect to the same Qdrant instance and share the same PVC, so datasets created through the web UI are immediately searchable via MCP tools and vice versa.

### EZUA / Istio integration

When `ezua.enabled=true` (default), the chart also creates:

- **VirtualService** — routes `rag-mcp-server.<domain>` through `istio-system/ezaf-gateway`
- **AuthorizationPolicy** — enforces OAuth2 authentication at the Istio ingress gateway
- **Kyverno ClusterPolicy** — auto-labels Pods/Deployments/Services in the release namespace with `hpe-ezua/type: vendor-service` and `hpe-ezua/app: rag-mcp-server` (required for the EZUA ingress to discover the service)

---

## 8. Upgrading

```bash
helm upgrade multimodal-rag . \
  --set image.tag=v1.0.0 \
  --reuse-values  # keep existing non-default values
```

To change specific values while upgrading:

```bash
helm upgrade multimodal-rag . \
  --set image.tag=v1.0.0 \
  --set persistence.data.size=500Gi
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| Pods stuck in `Pending` | PVC not binding | `kubectl describe pvc` |
| API server crash-looping | Model connection failure | `kubectl logs -c rag-api-server` |
| MCP tools return "dataset not found" | Dataset created on different Qdrant | Verify `QDRANT_HOST` matches |
| Search returns 0 results | Empty dataset or wrong collection | Check via web UI document list |
| Qdrant OOMKilled | Vector count exceeds memory | Increase `resources.qdrant.limits.memory` |
| 404 at VirtualService endpoint | Kyverno labels not applied | `kubectl describe deployment -n <ns> \| grep hpe-ezua` |
| 401 at VirtualService endpoint | OAuth2 token missing/expired | Check `oauth2-proxy` logs in `istio-system` |
| `${DOMAIN_NAME}` literal in VS endpoint | `envsubst` step skipped | `kubectl describe vs -n <ns>` shows unresolved variable |
