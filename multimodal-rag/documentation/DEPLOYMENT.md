# Deployment Guide

> **PCAI is a Kubernetes wrapper — you never run `helm` or `kubectl`.** You
> import the packaged chart (`rag-mcp-server` `.tar.gz`) into PCAI once, then
> drive the deployment by setting the chart's **`values.yaml`** in the PCAI
> *Helm Values* editor (or via the PCAI API). Every `--set` in this guide maps
> 1:1 to a key in `values.yaml`. There is no `envsubst` step — set the actual
> domain value directly in `values.yaml`.

---

> **⚠ Required config: `MEDIA_TOKEN_SECRET`**
>
> Since v1.9.5, both the API and MCP containers **refuse to start** without
> `MEDIA_TOKEN_SECRET` set. Media URLs are served via short-lived HMAC
> tokens (the legacy `?password=` URLs were removed), and this shared secret
> is what signs/verifies them. Deploying without it crashes both pods.
>
> Generate one and set it in `values.yaml` before deploying:
>
> ```bash
> python -c "import secrets; print(secrets.token_hex(32))"
> # -> values.yaml:  security.mediaTokenSecret: "<output>"
> ```

---

## Prerequisites

- A PCAI environment where you can import the packaged chart and edit its values.
- The image is public in `ghcr.io/ai-solution-eng/...` — you only need
  registry push access if you build a custom image.
- Model endpoints (embedder, reranker, VLM, ASR) deployed through MLIS, with
  their API tokens.

---

## 1. Import the chart in PCAI and set the image

The packaged charts ship with
`ghcr.io/ai-solution-eng/multimodal-rag-mcp:v2.5.0` as the default
`image.repository`/`image.tag` — no image build is required. If you need a
custom build, the Dockerfile lives at `docker/Dockerfile` and expects the repo
root as the build context; push the result to your registry and override the
values:

```yaml
# values.yaml
image:
  repository: ghcr.io/ai-solution-eng/multimodal-rag-mcp
  tag: v2.5.0
```

> **Note on models**: The image does not bundle any ML models. It connects to
> remote model endpoints configured via values; the defaults point to models
> hosted on the PCAI internal cluster.

---

## 2. Configure deployment values

All configuration lives in the chart's `values.yaml`. Key settings you edit
in the PCAI *Helm Values* editor:

```yaml
# PVC sizes — adjust for your dataset scale
persistence:
  data:
    size: 200Gi        # Uploaded files + dataset metadata
  qdrant:
    size: 150Gi        # Qdrant vectors + index

# Model configuration — each model has name, url, className, and
# optional extra kwargs. Set url to "" to disable a component.
models:
  embedder:
    name: "Qwen/Qwen3-VL-Embedding-8B"
    url: "https://..."              # required
    className: "MultiModalEmbeddings"
  reranker:
    name: "Qwen/Qwen3-VL-Reranker-8B"
    url: "https://..."              # required
    className: "MultiModalReranker"
  vlm:
    name: "RedHatAI/gemma-4-31B-it-FP8-block"
    url: "https://..."              # required
  asr:
    name: "CohereLabs/cohere-transcribe-03-2026"
    url: "https://..."              # required

# Model API keys (stored in a Kubernetes Secret, never in the image)
modelSecrets:
  embedderApiKey: "eyJ..."
  rerankerApiKey: "eyJ..."
  vlmApiKey: "eyJ..."
  asrApiKey: "eyJ..."

# RAG pipeline defaults
rag:
  captionWithAsr: false   # Transcribe video audio tracks via ASR during ingestion
  captionWithVlm: false   # VLM-describe images/videos at ingest (enables VLM-skip at retrieval)
  remote: false           # Use remote model URLs vs in-cluster .svc.cluster.local

# MCP server (sidecar)
mcp:
  enabled: true
  port: 9090

# Security — REQUIRED: a shared secret for short-lived media HMAC tokens.
# Both the API and MCP servers refuse to start without it (the legacy
# ?password= media URLs were removed). Also raises the SSRF guard defaults.
security:
  mediaTokenSecret: "<random-string-shared-by-both-containers>"
  mediaTokenTtl: 3600
  ingestAllowHosts: ""        # e.g. ".minio.svc.cluster.local" (bypasses private-block)
  blockPrivateHosts: true     # default on — blocks SSRF targets

# PCAI / EZUA (Istio-based ingress)
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-mcp-server.<your-domain>"
    istioGateway: "istio-system/ezaf-gateway"
    timeout: 660s
  authorizationPolicy:
    namespace: "istio-system"
    providerName: "oauth2-proxy"

# Resources — scale based on expected Qdrant load
resources:
  app:
    limits:
      memory: 4Gi
  qdrant:
    limits:
      memory: 48Gi   # ~40 GiB needed for 1M × 4096-dim vectors
```

> Set the actual PCAI domain in `ezua.virtualService.endpoint` (e.g.
> `rag-mcp-server.<your-domain>`) directly — there is no `${DOMAIN_NAME}`
> envsubst step in a PCAI deployment.

---

## 3. Install / update in PCAI

Import the packaged `rag-mcp-server` chart into PCAI, then set the values
above (image, models + `modelSecrets`, `security.mediaTokenSecret`,
persistence sizes, `ezua.*`) in the *Helm Values* editor and apply. Model
URLs and API keys are required.

To change a setting later, edit the values in PCAI and apply again — that is
the only "upgrade" path you need.

```yaml
# values.yaml — the keys PCAI renders from (also shown above)
image:
  repository: ghcr.io/ai-solution-eng/multimodal-rag-mcp
  tag: v2.5.0
models:
  embedder:
    name: "Qwen/Qwen3-VL-Embedding-8B"
    url: "https://..."
    className: "MultiModalEmbeddings"
  reranker:
    name: "Qwen/Qwen3-VL-Reranker-8B"
    url: "https://..."
    className: "MultiModalReranker"
  vlm:
    name: "RedHatAI/gemma-4-31B-it-FP8-block"
    url: "https://..."
  asr:
    name: "CohereLabs/cohere-transcribe-03-2026"
    url: "https://..."
modelSecrets:
  embedderApiKey: "eyJ..."
  rerankerApiKey: "eyJ..."
  vlmApiKey: "eyJ..."
  asrApiKey: "eyJ..."
persistence:
  data:
    size: 500Gi
  qdrant:
    size: 200Gi
security:
  mediaTokenSecret: "<generated>"
ezua:
  virtualService:
    endpoint: "rag-mcp-server.<your-domain>"
```

> **Tip**: keep the model URLs and API keys in a private values fragment (or
> the PCAI Secret) so they stay out of commit history; the chart reads them
> from values at apply time.

---

## 4. Verify the deployment

In the PCAI UI the deployment should reach **Ready**. To check locally
(optional, developer convenience):

```bash
# Port-forward to test locally
kubectl port-forward deployment/rag-mcp-server 8000:8000

# Health check
curl http://localhost:8000/healthz
# → {"status": "ok"}

# List datasets (empty initially)
curl http://localhost:8000/api/datasets
# → {"datasets": []}
```

Pods and logs are visible from the PCAI workload view
(\cmd{kubectl get pods -l app=rag-mcp-server} and
\cmd{kubectl logs -l app=rag-mcp-server -c rag-api-server} work the same as
ever for operators who have cluster access).

---

## 5. Access the web UI

If EZUA (Istio) is enabled, the service is available at the VirtualService
endpoint (e.g. `https://rag-mcp-server.<your-domain>`). Authentication is
handled by the `oauth2-proxy` AuthorizationPolicy.

For a local developer preview only, port-forward:

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

In PCAI, upgrading is just editing the values and applying again. To bump the
image, change `image.tag` in the *Helm Values* editor:

```yaml
# values.yaml
image:
  tag: v2.5.0
```

To change specific settings (e.g. storage or model endpoints), edit the
corresponding keys in `values.yaml` and re-apply; the rest of the values are
kept.

---

## 9. Troubleshooting

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| Pods stuck in `Pending` | PVC not binding | Check the PVC status in PCAI |
| API server crash-looping | Model connection failure | Read the API pod logs in PCAI |
| MCP tools return "dataset not found" | Dataset created on different Qdrant | Verify `QDRANT_HOST` matches |
| Search returns 0 results | Empty dataset or wrong collection | Check via web UI document list |
| Qdrant OOMKilled | Vector count exceeds memory | Increase `resources.qdrant.limits.memory` |
| 404 at VirtualService endpoint | Kyverno labels not applied | Confirm the workload is labelled `hpe-ezua` (PCAI discovery) |
| 401 at VirtualService endpoint | OAuth2 token missing/expired | Check `oauth2-proxy` logs in `istio-system` |
| Endpoint shows `rag-mcp-server.<domain>` unresolved | Domain value not set | Set `ezua.virtualService.endpoint` in values.yaml |
