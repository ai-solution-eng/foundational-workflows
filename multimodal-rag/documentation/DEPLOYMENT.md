# Deployment Guide

Deploy the Multimodal RAG server on PCAI: pick one of the three packaged chart variants, set a handful of required values (plus optional tuning) in the PCAI *Helm Values* editor, and apply.

<div align="center"><img src="./deployment_flow-1.png" width="900" alt="Deployment architecture: clients -> Istio gateway/oauth2-proxy -> API server + MCP sidecar + embed-batcher + Redis -> Qdrant cluster + PVC -> MLIS model endpoints, with security layers annotated"></div>

The image is public in `ghcr.io/ai-solution-eng/...` — no build required unless you maintain a custom image (Dockerfile at `docker/Dockerfile`, repo root as build context). The image bundles **no ML
models**: it connects to remote model endpoints (embedder, reranker, VLM, ASR — typically MLIS deployments) configured via values.

## Prerequisites

- A PCAI environment where you can import the packaged chart and edit its values.
- Model endpoints (embedder, reranker, VLM, ASR) deployed through MLIS, with their API tokens.
- For ingress: an EZUA/Istio gateway with `oauth2-proxy` (see [Deployment targets](#5-deployment-targets)).

---

## 1. Pick the chart variant

All three charts share the same templates and component layout — an API Deployment whose pod runs two containers (`rag-api-server` :8000 REST + web UI, `rag-mcp-server` :9090 MCP sidecar), a Qdrant
StatefulSet, and (scale charts only) Redis and the shared embed-batcher. They differ only in replica counts, per-container resources, and PVC sizing, all driven by `values.yaml`. Import the packaged
`.tar.gz` of the variant you need into PCAI.

| Dimension | `helm/` (base) | `helm-scale-medium/` | `helm-scale-large/` |
|---|---|---|---|
| API replicas | 1 | 2 | 4 |
| Server | `uvicorn` (1 event loop) | gunicorn + 2 `UvicornWorker`s | gunicorn + 4 `UvicornWorker`s |
| Qdrant | 1 instance (HTTP) | 2-node cluster (gRPC, sharded) | 3-node cluster (gRPC, sharded) |
| Qdrant client | HTTP, 30s timeout | gRPC, 30s timeout, INT8 quantization | gRPC, 30s timeout, INT8 quantization |
| Shared embed-batcher | — | — | ✔ (cross-process text-query batching) |
| Redis (cross-pod unlock cache + job tracker) | off | ✔ | ✔ |
| Unlock cache | in-process | Redis (cross-pod) | Redis (cross-pod) |
| Document-count sync on read | every call | deferred (admin-only) | deferred (admin-only) |
| Sync / MCP thread pools | 12 / 64 | 32 / 32 per worker | 64 / 64 per worker |
| Model HTTP pool | 30 connections | 100 connections | 200 connections |
| Pod anti-affinity | no | ✔ | ✔ |
| Data PVC (RWX, shared) | 50 Gi | 50 Gi | 100 Gi |
| Qdrant PVC | 50 Gi RWX (read-only mount on API pod) | 25 Gi RWO × 2 replicas | 100 Gi RWO × 3 replicas |

**What the scale charts change architecturally:**

- **Multiple API replicas + load balancing.** A ClusterIP Service distributes traffic round-robin; the Istio VirtualService routes external traffic from the `ezaf-gateway` with three timeout tiers so
  slow operations don't tie up the gateway: batch uploads / SSE streams (`/api/datasets/*/batch-*`) and MCP searches (`/mcp`) get `longTimeout` (3600s), all other routes `timeout` (300s).
- **Gunicorn with multiple Uvicorn workers per pod.** Effective concurrency = `replicas × workers × syncPoolSize` (large default: 4 × 4 × 64 = 1024 concurrent blocking operations cluster-wide). The
  shared embed-batcher singleton (large chart, `embedBatcher.enabled`) aggregates text queries cross-process, so per-worker batch fragmentation no longer applies — measured effect in
  [BENCHMARKS.md](BENCHMARKS.md).
- **Sharded multi-replica Qdrant.** `QDRANT__CLUSTER__ENABLED=true` + peer-to-peer port 6335 + a headless Service give each Qdrant pod stable DNS for StatefulSet peer discovery; collections shard
  across replicas so read load spreads. Each replica gets its own PVC via `volumeClaimTemplates` (RWO), sized by `persistence.qdrant.size`.
- **Redis-backed unlock cache and job tracker.** With many workers, an in-process unlock cache would force password re-entry on every pod switch; Redis (TTL 1800s, in-app default) shares unlocks
  across pods, and upload/recreate job progress is mirrored to Redis so `upload-status` polls answer from any replica. Falls back to in-memory when Redis is unavailable.
- **Deferred count sync.** Scale charts set `rag.deferCountSync: true` so dataset reads skip the per-read Qdrant count round-trip and `meta.json` write-back (write races on the shared PVC) — counts
  sync on explicit admin requests.
- **Larger pools.** `app.syncPoolSize` / `app.mcpPoolSize` (blocking RAG work and MCP tool bodies) and `modelPool.maxConnections` (embedder HTTP pool on every search's critical path) are raised;
  Qdrant runs gRPC with a hard 30s timeout so a hung node can't pin worker threads.

**Resource requirements** (each API pod runs two containers, so per-pod app resources are 2 × `resources.app`):

| Component | Chart | Replicas | Per-unit requests | Per-unit limits | Storage |
|---|---|---|---|---|---|
| App (2 ctr/pod) | `helm/` | 1 | 4 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| | `helm-scale-medium/` | 2 | 5 Gi / 3 cpu | 16 Gi / 6 cpu | — |
| | `helm-scale-large/` | 2→4 | 8 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| Qdrant | `helm/` | 1 | 16 Gi / 4 cpu | 32 Gi / 8 cpu | 50 Gi |
| | `helm-scale-medium/` | 2 | 10 Gi / 3 cpu | 20 Gi / 6 cpu | 25 Gi × 2 |
| | `helm-scale-large/` | 3 | 16 Gi / 4 cpu | 32 Gi / 8 cpu | 100 Gi × 3 |
| Redis | scale charts | 1 | 256 Mi / 100 m | 512 Mi / 500 m | — |

Cluster-wide totals: base 20 Gi mem / 8 cpu requests, medium ~30 Gi / 12 cpu (+51 %), large ~64 Gi / 20 cpu (+221 %); PVC totals 100 Gi / 100 Gi / 400 Gi. The data PVC is annotated
`helm.sh/resource-policy: keep` in all three charts (survives release removal); Qdrant PVCs come from `volumeClaimTemplates` and are not kept.

> **Variant-switch caveat:** moving a release from the base chart to either scale variant (or between scale variants) changes the Qdrant StatefulSet `volumeClaimTemplates` (access mode and/or size),
> which Kubernetes treats as immutable. The existing Qdrant StatefulSet and its PVCs must be deleted before re-applying — **Qdrant vectors are lost and must be re-indexed** (recreate each dataset from
> its on-disk originals). The data PVC is unaffected.

Measured throughput for the variants lives in [BENCHMARKS.md](BENCHMARKS.md).

---

## 2. Required values

Three values must be set deliberately before applying. The charts ship defaults for `image` and `security.mediaTokenSecret` so a first apply can succeed, but treat all three as required configuration:

| Value | Why it is required |
|---|---|
| `image.repository` / `image.tag` | The workload definition itself — defaults to the published `ghcr.io/ai-solution-eng/multimodal-rag-mcp:v3.6.3`. Override only for custom builds. |
| `models.embedder.url` | The **only required model**. The embedder is probed at startup via its OpenAI-compatible `GET /v1/models`; an unreachable embedder aborts startup (the API and MCP containers both refuse to come up without a working embedder). Vector dimension, chunk budgets, and the bundled tokenizer all derive from it. |
| `security.mediaTokenSecret` | Both containers refuse to start when unset (env-only runs): media URLs are served via short-lived HMAC tokens and this shared secret signs/verifies them. The charts carry a placeholder default — **replace it** (`python -c "import secrets; print(secrets.token_hex(32))"`), because every token the deployment mints is signed with it. |

In practice the model endpoints on PCAI/MLIS also authenticate: set `modelSecrets.embedderApiKey` (and the reranker/VLM/ASR keys as configured) — they are rendered into a Kubernetes Secret, never
baked into the image.

```yaml
# values.yaml — required
image:
  repository: ghcr.io/ai-solution-eng/multimodal-rag-mcp
  tag: v3.6.3
models:
  embedder:
    name: "Qwen/Qwen3-VL-Embedding-8B"
    url: "https://qwen3-vl-embedding-8b.project-user-<you>.serving.<cluster-domain>"
    className: "MultiModalEmbeddings"
modelSecrets:
  embedderApiKey: "eyJ..."
security:
  mediaTokenSecret: "<python -c 'import secrets; print(secrets.token_hex(32))'>"
```

---

## 3. Optional values

Everything else has a working default. The full walkthrough:

```yaml
# Optional models — url "" disables a component. Only the embedder is required:
# an unreachable reranker/VLM/ASR logs a warning at startup and the system
# degrades without them (no reranking / no captioning / no transcription).
models:
  reranker:
    name: "Qwen/Qwen3-VL-Reranker-8B"
    url: "https://..."
    className: "MultiModalReranker"
  vlm:
    name: "Qwen/Qwen3.8-27B-FP8"    # any OpenAI-compatible VLM
    url: "https://..."
  asr:
    name: "CohereLabs/cohere-transcribe-03-2026"
    url: "https://..."
modelSecrets:                        # keys for the optional models, same Secret mechanism
  rerankerApiKey: "eyJ..."
  vlmApiKey: "eyJ..."
  asrApiKey: "eyJ..."

# PVC sizes — adjust for dataset scale
persistence:
  data:
    size: 200Gi        # uploaded files + dataset metadata (RWX)
  qdrant:
    size: 150Gi        # Qdrant vectors + index (per replica on scale charts)
    # storageClass / accessMode are also settable; scale charts use RWO per replica

# RAG pipeline defaults
rag:
  captionWithAsr: true    # transcribe video audio tracks via ASR at ingest (auto-disables without ASR)
  captionWithVlm: true    # VLM-describe images/videos at ingest (auto-disables without VLM)
  remote: false           # remote model URLs vs in-cluster .svc.cluster.local
  ocr: false              # default for new datasets' OCR flag (per-dataset flag always wins)
  dedupThreshold: 0.995   # cosine above which a near-duplicate is skipped

# MCP sidecar
mcp:
  enabled: true
  port: 9090

# Security extras (beyond the required mediaTokenSecret) — chart maps these to
# the server's env-level flags, e.g. security.apiKey -> RAG_API_KEY
security:
  apiKey: "<change-me>"        # Bearer/X-RAG-Api-Key on all /api/* routes (charts ship a default — change it)
  mediaTokenTtl: 3600
  ingestAllowHosts: ""         # e.g. ".minio.svc.cluster.local" — authoritative allowlist bypassing the private-block
  blockPrivateHosts: true      # SSRF guard: reject private/loopback ingest+query media targets
  trustProxyIdentity: true     # trust X-Auth-Request-* identity headers behind oauth2-proxy
  maxDownloadBytes: 536870912
  archiveMaxTotalBytes: 2147483648
  archiveMaxMemberBytes: 1073741824
  archiveMaxEntries: 10000
  mediaAllowPathPrefixes: /data/datasets:/data/staging
  pwMaxFailures: 10            # password-failure throttle per identity
  pwFailWindow: 300

# PCAI / EZUA (Istio-based ingress)
ezua:
  enabled: true
  virtualService:
    endpoint: "rag-mcp-server.${DOMAIN_NAME}"   # keep the ${DOMAIN_NAME} placeholder on customer PCAI (see targets)
    istioGateway: "istio-system/ezaf-gateway"
    timeout: 300s          # default tier; longTimeout: 3600s covers batch uploads / SSE / MCP
  authorizationPolicy:
    enabled: true
    namespace: "istio-system"
    providerName: "oauth2-proxy"

# Resources — scale with expected Qdrant load (~40 GiB for 1M × 4096-dim vectors)
resources:
  app:
    limits:
      memory: 8Gi
  qdrant:
    limits:
      memory: 48Gi

# S3/MinIO ingest source (optional — omit to use default AWS credential chain)
s3:
  endpointUrl: http://minio.minio.svc.cluster.local:9000
  accessKeyId: ""
  secretAccessKey: ""

# Opt-in add-ons
redis:
  enabled: false          # auto-on in the scale charts (cross-pod unlock cache + job mirror)
metrics:
  serviceMonitor: false   # Prometheus ServiceMonitor for /metrics
backups:
  enabled: false          # scheduled exports of non-protected datasets to S3
  schedule: "0 3 * * *"
  bucket: ""
  retentionDays: 0

extraEnv: {}              # any additional container env, e.g. MEMORY_MAX_TOKENS: "8192"
```

Model URLs and API keys are live-reloadable: the charts mount the `-config` ConfigMap and `-model-keys` Secret as file volumes and a watcher re-applies them every 15s — a model swap takes effect
without a rollout (the new embedder is verified against `/v1/models` before the swap; unreachable, the old config is kept).

---

## 4. Apply in PCAI

1. **Import the packaged chart once** — PCAI → import the `rag-mcp-server` `.tar.gz` of your chosen variant.
2. **Edit `values.yaml`** in the PCAI *Helm Values* editor: required values (§2), then optional values (§3).
3. **Apply.** The chart renders the VirtualService, AuthorizationPolicy, and the vendor-label ClusterPolicy (see targets below), creates the PVCs/Secrets, and starts the pods.

Changing a setting later is the same loop: edit the value, apply again. Bumping the image is just `image.tag`. There is no separate upgrade procedure.

> **Tip:** keep model URLs and API keys in a private values fragment (or the PCAI Secret) so they stay out of commit history; the chart reads them from values at apply time. The `helm*/local/`
> convention in this repo holds per-cluster site values that are gitignored.

---

## 5. Deployment targets

### SE G2 (HPE internal cluster)

- **Cluster domain:** `pcai-se-ai-application.hst.rdlabs.hpecorp.net` — the EZUA ingress endpoint becomes `rag-mcp-server.pcai-se-ai-application.hst.rdlabs.hpecorp.net`.
- **HPE proxy:** keep the corporate proxy settings on for anything that reaches out (image pulls, model endpoint calls from a laptop, HuggingFace downloads at build time). In-cluster model traffic
  goes to MLIS service endpoints.
- **Namespaces:** deploy into your personal `project-user-<name>` namespace. Model endpoints follow the same convention — MLIS deployments land under
  `project-user-<owner>.serving.pcai-se-ai-application.hst.rdlabs.hpecorp.net`, which is what you put in `models.*.url`.
- **Auth:** the ezaf-gateway applies HPE SSO centrally, so set `ezua.authorizationPolicy.enabled: false` — the chart should not create its own AuthorizationPolicy here. Keep `rag.remote: false` so
  `.serving.` URLs are rewritten to their in-cluster form.
- Paste-ready values: `helm/values-examples/values.g2.yaml` (copy real model JWTs and the media token secret into a gitignored `helm*/local/` file — never commit them). The repo keeps SE G2 site
  values under `helm-scale-large/local/se_g2.yaml`. The scale benchmark in [BENCHMARKS.md](BENCHMARKS.md) was measured on this cluster.

### Hosted trial (customer PCAI)

- **Domain placeholder:** keep `ezua.virtualService.endpoint: "rag-mcp-server.${DOMAIN_NAME}"` as-is — PCAI's deployment pipeline resolves `${DOMAIN_NAME}` before rendering (verified: the deployed
  release stores the fully-resolved endpoint). `ezua.domainName` itself is a platform-convention key no chart template reads; the VirtualService is built from `ezua.virtualService.endpoint`.
- **oauth2-proxy AuthorizationPolicy:** keep `ezua.authorizationPolicy.enabled=true` (default) — the chart creates an Istio `AuthorizationPolicy` in `istio-system` that forces the `oauth2-proxy`
  provider (action CUSTOM) on requests to the endpoint host, selecting the `istio: ingressgateway` pods. Users authenticate via SSO at the gateway; set `security.trustProxyIdentity: true` (default) so
  unlock-cache identity follows the authenticated user. The API-key middleware is a separate, optional inner layer.
- **Kyverno vendor-label ClusterPolicy (pre-install hook):** the chart installs a `kyverno.io/v1 ClusterPolicy` named `add-vendor-app-labels-<release>-<chart>` as a **pre-install Helm hook** (weight
  −5, `before-hook-creation`). It mutates every Pod/Deployment/Service **in the release namespace** to carry `hpe-ezua/type: vendor-service` and `hpe-ezua/app: rag-mcp-server` — the labels the EZUA
  ingress uses to discover the service; without them the VirtualService endpoint 404s even though the pods are healthy. Customer-cluster admins may need to allow it: creating a ClusterPolicy requires
  Kyverno admission and cluster-scoped permission, so on a locked-down trial cluster the apply can be rejected until an admin whitelists the policy (or pre-creates the labels). `background: false` —
  it mutates on admission, not retroactively.
- Paste-ready values: `helm/values-examples/values.hosted-trial.yaml`.

---

## 6. Architecture overview

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
              │  (port 6333)     │     (cluster mode on scale charts)
              └──────────────────┘
```

Both the API server and the MCP sidecar connect to the same Qdrant and share the `/data` PVC, so datasets created through the web UI are immediately searchable via MCP tools and vice versa.

When `ezua.enabled=true` (default) the chart also creates:

- **VirtualService** — routes `rag-mcp-server.<domain>` through `istio-system/ezaf-gateway` with the timeout tiers above.
- **AuthorizationPolicy** — enforces OAuth2 authentication at the Istio ingress gateway via `oauth2-proxy`.
- **Kyverno ClusterPolicy** — the vendor-label mutation described in [Hosted trial](#hosted-trial-customer-pcai).

---

## Next steps

- **[VERIFICATION.md](VERIFICATION.md)** — confirm the deployment came up, exercise it, and troubleshoot.
- **[FEATURES.md](FEATURES.md)** — the full feature/technical reference: REST API, MCP server and its 13 tools, retrieval internals.
- **[memory/](memory/README.md)** — per-user long-term memory setup for opencode, Open WebUI, and DSH.
