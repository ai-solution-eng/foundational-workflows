# Scaling: the `helm-scale-medium/` and `helm-scale-large/` charts

The base `helm/` chart runs a single API replica with a single Qdrant instance — optimised for simplicity. The `helm-scale-large/` chart trades that for horizontal capacity. This document explains every layer of the scale chart and how it achieves its goal.

A medium variant, `helm-scale-medium/`, reuses the scale-chart architecture (gunicorn workers, multi-replica Qdrant, Redis unlock cache) but dials the replica counts and per-container resources back so it can be tested as a drop-in replacement on a cluster sized for the base chart. The [Resource requirements](#resource-requirements) section compares all three.

## 1. Multiple API replicas + load balancing

`values.yaml` — `replicaCount: 4` (when `autoscaling.enabled`, §12, an HPA owns this count instead and the Deployment ships without `spec.replicas`)

The Deployment runs 4 pods (vs 1 in the base chart). Traffic is distributed across them by:

- A **ClusterIP Service** (`templates/service.yaml`) with `targetPort: http`, which kube-proxy load-balances round-robin.
- An **Istio VirtualService** (`templates/virtualservice.yaml`) that routes external traffic from the `ezaf-gateway` to the `-api` service. It defines **three timeout tiers** so slow operations don't tie up the gateway:
  - Batch uploads / SSE streams (`/api/datasets/*/batch-*`): `longTimeout: 3600s`
  - MCP searches (`/mcp` prefix): `longTimeout: 3600s`
  - All other API routes: `timeout: 300s`

## 2. Gunicorn with multiple Uvicorn workers per pod

`templates/deployment.yaml` — gunicorn command

Instead of a single `uvicorn` process, the scale chart runs **gunicorn with `UvicornWorker`** (`app.workers`). The large chart defaults to **4 replicas × 4 workers** (16 event loops) (`app.workers: 4`); the medium chart dials this to **2 replicas × 2 workers**. The shared embed-batcher singleton aggregates text queries cross-process (`RAG_EMBED_BATCH_URL`), so per-worker batch fragmentation no longer applies.  (The old "4 workers → ~8 req/s" result predates
the singleton and does not reproduce: the same 4×4 shape measured 49.3 req/s @ N=100 and 72.8 req/s @ N=250 on v3.1.8, 100% success.)  Effective concurrency = `replicas × workers × app.syncPoolSize` = 4 × 4 × 64 = **1024 concurrent blocking operations** cluster-wide. Each worker process gets its own `sync_pool`, `httpx` connection pools, and Qdrant clients.

## 3. Multi-replica Qdrant cluster with sharding

`templates/qdrant-statefulset.yaml` — `qdrant.replicas` (large: 3, medium: 2) with cluster mode

This is the biggest architectural difference. The base chart runs a single Qdrant instance; the scale charts run a **multi-node Qdrant cluster** (`qdrant.replicas`: 3 on large, 2 on medium):

- `QDRANT__CLUSTER__ENABLED=true` and `QDRANT__CLUSTER__P2P_PORT=6335` enable Qdrant's distributed consensus and peer-to-peer bootstrapping.
- A **headless Service** (`clusterIP: None`, `templates/qdrant-service.yaml`) gives each Qdrant pod a stable DNS identity (`rag-mcp-server-qdrant-0`, `-1`, `-2`) — required for StatefulSet peer discovery.
- The p2p port (6335) is exposed alongside gRPC (6334) and HTTP (6333) so nodes can coordinate.
- Collections are **sharded** across the nodes, so read load (every search hits Qdrant) is spread out.

Each Qdrant replica gets its own PVC via `volumeClaimTemplates` with `ReadWriteOnce` (per-pod storage; `persistence.qdrant.accessMode` — the volumeClaimTemplate mode, distinct from the base chart's `ReadWriteMany` shared claim), and larger resources: `10–32Gi` memory, `3–8` CPU (`resources.qdrant`, the same dotted paths as the base chart). The large chart additionally retains per-replica PVCs on scale-down/uninstall (`persistentVolumeClaimRetentionPolicy: Retain`); delete the PVCs explicitly to wipe Qdrant data.

Because the per-replica Qdrant PVCs are **not** mounted on the API pod (in the base chart they are mounted read-only so `/api/admin/health` can report exact usage), the management page instead surfaces per-replica **shard placement** plus the configured per-replica size (`QDRANT_PVC_SIZE` = `persistence.qdrant.size`) from Qdrant's `/cluster` API. This gives a storage-spread estimate (which
replicas hold how many shards × PVC size) without exec/kubectl. For exact bytes per replica use `kubectl exec <qdrant-N> -- df -h /qdrant/storage`.

## 4. Redis-backed cross-pod unlock cache

`templates/redis.yaml` + `templates/configmap.yaml`

In the base chart, password-unlocked datasets are cached **in-process memory** (`_UNLOCK_CACHE` dict in `api_server.py`). With 4 replicas × 4 workers = 16 separate processes, a user would have to re-enter their password whenever routed to a different pod/worker. The scale charts add:

- A **Redis Deployment + Service** (`templates/redis.yaml`) running `redis-server` in-memory (no persistence — unlock state is ephemeral). `redis.enabled: true` on both scale charts (the base chart keeps `redis.enabled: false`, which renders nothing); `redis.image` (`redis:7-alpine`, `redis.image.repository`/`redis.image.tag`/`redis.image.pullPolicy`) and `redis.port` (6379) are overridable.
- `REDIS_URL` is injected into the ConfigMap, and the backend (`api_server.py`) lazily builds a Redis client. `_unlock_cache_get` / `_unlock_cache_set` use Redis with a TTL (in-app `UNLOCK_TTL` default 1800 s — not chart-configurable) so a single unlock works across all pods.
- `redis.password` (default empty = unauthenticated) — when set, `redis-server` starts with `--requirepass` and the app connects with the password from the model-keys Secret (`REDIS_PASSWORD`).
- Falls back to in-memory dict if Redis is unavailable.

Unlock identity is also derived from the authenticated user (oauth2-proxy headers: `X-Auth-Request-Email` / `X-Auth-Request-User`) rather than client IP, which is unreliable behind Istio/oauth2-proxy (many users may share one proxy IP).

## 5. Deferred document-count sync (avoids write races)

`values.yaml` — `rag.deferCountSync: true` → `RAG_DEFER_COUNT_SYNC=true`

In the base chart, every `get_dataset()` / `list_datasets()` call syncs the document count from Qdrant and writes it back to `meta.json` on the shared PVC. With multiple replicas, this causes:

1. **Write races** — multiple pods writing `meta.json` concurrently on the NFS PVC.
2. **Qdrant load** — every read triggers an extra Qdrant round-trip.

The scale charts set `rag.deferCountSync: true`, which makes `list_datasets()` and `get_dataset()` skip the Qdrant count sync entirely (`dataset_manager.py`). Counts are only synced on explicit admin requests. This eliminates both the race and the extra Qdrant load.

## 6. gRPC Qdrant client with hard timeout

`values.yaml` → `qdrant.client` → `templates/configmap.yaml`

The base chart uses HTTP to talk to Qdrant. The scale charts set:

- `qdrant.client.preferGrpc: true` → `QDRANT_PREFER_GRPC=true` — gRPC is more efficient than HTTP at high QPS (binary framing, multiplexed streams). Honored in `rag_system.py`.
- `qdrant.client.timeout: 30` → `QDRANT_CLIENT_TIMEOUT=30` — a hard 30s timeout so a hung Qdrant node can't pin a `sync_pool` thread indefinitely. With limited thread-pool slots, one hung request could otherwise cascade.
- `qdrant.client.quantization: int8` → `QDRANT_QUANTIZATION=int8` — new collections get int8 scalar quantization (4x vector-memory reduction, ~1–2% recall loss, faster search; no rescoring pass). `none` disables.
- `qdrant.client.quantizationAlwaysRam: true` → `QDRANT_QUANTIZATION_ALWAYS_RAM=true` — quantized vectors are pinned in RAM for fast search. Set `false` for very large datasets (2M+ points) to avoid OOM (quantized vectors then read from disk per search).

Quantization applies to *newly created* collections only — existing collections keep their original config.

## 7. Larger, tuned thread pools

`values.yaml` → `app.syncPoolSize` / `app.mcpPoolSize` → `templates/configmap.yaml`

The blocking RAG work (embedding, Qdrant calls, file processing) runs in a `ThreadPoolExecutor`. The scale charts raise:

- `app.syncPoolSize` → `SYNC_POOL_SIZE: 64` (large) / `32` (medium) — per-worker thread pool for `sync_wrapper_safe` (`utils/general_tools.py`). Effective concurrency = `replicaCount × app.workers × app.syncPoolSize`.
- `app.mcpPoolSize` → `MCP_POOL_SIZE: 64` (large) / `32` (medium) — per-pod thread pool for offloading MCP tool bodies (`mcp_server.py`), so a single slow search can't stall every concurrent MCP client.
- `app.qdrantPoolSize` → `QDRANT_POOL_SIZE: 16` — per-process thread pool around the Qdrant client (`vector_store.py`); queries and upserts fan through it so one slow shard can't serialize the rest.

## 8. Larger httpx connection pools for model endpoints

`values.yaml` → `modelPool` → `templates/configmap.yaml`

The embedder is on every search's critical path. The base chart defaults to `max_connections=30`. The scale charts raise:

- `modelPool.maxConnections` → `MODEL_POOL_MAX_CONNECTIONS: 200` (large) / `100` (medium)
- `modelPool.maxKeepaliveConnections` → `MODEL_POOL_MAX_KEEPALIVE_CONNECTIONS: 50` (large) / `25` (medium)

Honored in `utils/pcai_model_classes.py`, which builds `httpx.Limits` for the async model clients. This prevents the embedder's HTTP connection pool from becoming the bottleneck under concurrent load.

## 8b. Dynamic query batching (embedding + Qdrant)

`values.yaml` → `app.*` → `templates/configmap.yaml`

Both search-path hotspots batch concurrent queries so N simultaneous searches cost fewer model/store round-trips:

**Embedding queries** (`utils/model_adapters.py`):

- `app.embeddingQueryBatchSize` → `EMBEDDING_QUERY_BATCH_SIZE: 128` — fold up to this many concurrent text-embedding queries into one embedder call.
- `app.embeddingQueryBatchWaitMs` → `EMBEDDING_QUERY_BATCH_WAIT_MS: 200` — max ms the batcher holds a query waiting for company before flushing.
- `app.embeddingQueryIdleWaitMs` → `EMBEDDING_QUERY_IDLE_WAIT_MS: 50` — idle early-flush (batch-guarded): a small *stalled* queue (≤2 items) flushes after this many ms instead of the full window; bursts never split. `0` = always wait the full window.
- `app.embeddingMaxImagesPerPrompt` → `EMBEDDING_MAX_IMAGES_PER_PROMPT: 4` — cap on images embedded in a single multimodal prompt (bounds embedder request size).
- Large chart only: `embedBatcher.enabled: true` runs a **singleton** `multimodal_rag.embed_batcher` Deployment (`embedBatcher.port: 8002`); all API/MCP pods then route text-only queries to it (`RAG_EMBED_BATCH_URL`), so the effective batch size no longer depends on the number of app processes. Replicas must stay at 1 — more would split the batch pool again. Batcher pod sizing comes from `resources.embedBatcher` (`requests` 512Mi/500m, `limits` 1Gi/2 cpu on large). The medium chart omits the embedBatcher section entirely (per-process batching only).

**Qdrant queries** (`vector_store.py`):

- `app.qdrantQueryBatchSize` → `QDRANT_QUERY_BATCH_SIZE: 128` — fold up to this many concurrent Qdrant searches into one batched call.
- `app.qdrantQueryBatchWaitMs` → `QDRANT_QUERY_BATCH_WAIT_MS: 5` — max ms the batcher waits before flushing (kept small: Qdrant batches are cheap).

## 9. Pod anti-affinity (spreads replicas across nodes)

`templates/deployment.yaml` + `values.yaml` → `antiAffinity.enabled: true`

A `preferredDuringSchedulingIgnoredDuringExecution` anti-affinity rule with weight 100 tells the scheduler to spread the API pods across different nodes when possible. This means a node failure takes down at most one pod, and no single node becomes a CPU/memory hotspot for the RAG workload.

## 10. Shared file PVC (ReadWriteMany)

`templates/pvc.yaml` + `values.yaml` → `persistence.data`

All API replicas mount the same file PVC (`/data`) as `persistence.data.accessMode: ReadWriteMany` (NFS-backed via `persistence.data.storageClass: gl4f-filesystem`). This means any pod can serve any uploaded file — there's no need to replicate files across pods. The PVC is also annotated with `helm.sh/resource-policy: keep` so it survives `helm uninstall`. (`persistence.data.accessMode` / `persistence.data.storageClass` are the same standard-Kubernetes knobs as the base chart — see DEPLOYMENT.md.)

## 11. Standard Kubernetes knobs (scale charts)

Same families as the base chart (defaults differ — see the per-component tables
below); documented once there and repeated here so coverage is explicit:

| Key | Large default | Medium default | Effect |
|---|---|---|---|
| `deployment.appName` | `rag-mcp-server` | `rag-mcp-server` | `app` label on every rendered resource |
| `image.pullPolicy` | `IfNotPresent` | `IfNotPresent` | Pull policy for the app image (all containers + CronJobs) |
| `resources.app.requests.memory` / `requests.cpu` | 4Gi / 2 | 4Gi / 1.5 | Per-container requests (×2 containers per pod) — HPA-calibrated: utilization is computed against the request, so the request sets where the 70 % target fires (§12) |
| `resources.app.limits.memory` / `limits.cpu` | 8Gi / 4 | 8Gi / 3 | Per-container limits |
| `resources.qdrant.requests.memory` / `requests.cpu` | 4Gi / 1 | 4Gi / 1 | Per-replica Qdrant requests — burst-to-limit; a pod whose working set far exceeds its request is a node-pressure eviction candidate |
| `resources.qdrant.limits.memory` / `limits.cpu` | 32Gi / 8 | 20Gi / 6 | Per-replica Qdrant limits |
| `resources.redis.requests.memory` / `resources.redis.requests.cpu` | 256Mi / 100m | 256Mi / 100m | Redis pod requests |
| `resources.redis.limits.memory` / `resources.redis.limits.cpu` | 512Mi / 500m | 512Mi / 500m | Redis pod limits |
| `resources.embedBatcher.requests.memory` / `resources.embedBatcher.requests.cpu` | 512Mi / 500m | — (no embedBatcher) | Embed-batcher singleton requests |
| `resources.embedBatcher.limits.memory` / `resources.embedBatcher.limits.cpu` | 1Gi / 2 | — (no embedBatcher) | Embed-batcher singleton limits |
| `redis.image.repository` / `redis.image.tag` | `redis` / `7-alpine` | same | Redis server image |
| `redis.image.pullPolicy` | `IfNotPresent` | `IfNotPresent` | Redis image pull policy |

## 12. Opt-in autoscaling (HPA) and HPA-calibrated requests

Both scale charts ship an opt-in HorizontalPodAutoscaler for the API Deployment:

```yaml
# values.yaml (scale charts; default off — the default render is unchanged)
autoscaling:
  enabled: false
  minReplicas: 2        # HA floor — 1 replica is not high availability
  maxReplicas: 6        # medium (large: 8) — cap to your namespace quota
  targetCPUUtilizationPercentage: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300   # bursty RAG traffic → avoid flapping
```

When `autoscaling.enabled`, `templates/hpa.yaml` renders an autoscaling/v2 HPA targeting the API Deployment, and the Deployment renders **without `spec.replicas`** — the HPA owns the replica count, so a subsequent `helm upgrade` can no longer reset it (the classic helm/HPA fight; `replicaCount` is inert while enabled). The defaults preserve today's behavior exactly: disabled renders the HPA nothing and pins `spec.replicas: {{ .Values.replicaCount }}`.

What the HPA deliberately does **not** scale:

- **Qdrant StatefulSet** — shard topology and per-replica PVCs (`volumeClaimTemplates`) are not HPA-elastic; scale `qdrant.replicas` deliberately.
- **Embed-batcher** — a singleton by design (`replicas: 1`); more replicas would split the batch pool again (§8b). More API replicas means more queries through the same batcher — if `maxReplicas` goes high, consider raising `resources.embedBatcher`.
- **Base chart** — no autoscaling block at all: its in-process unlock cache is not multi-replica safe. Only the scale charts (Redis unlock cache, deferred count sync, shared RWX data PVC) are replicas-safe by construction (§4, §5, §10).

Scheduling shape is the other half of the contract — the app containers' **requests are HPA-calibrated** (2026-10):

| Resource | Requests (default) | Limits (unchanged) |
|---|---|---|
| App container (×2/pod) | 4Gi / 2 cpu (large) · 4Gi / 1.5 cpu (medium) | 8Gi / 4 cpu (large) · 8Gi / 3 cpu (medium) |
| Qdrant replica | 4Gi / 1 (burst-to-limit) | 20Gi / 6 cpu (medium) · 32Gi / 8 cpu (large) |

**Why the app request is NOT burst-to-limit:** the HPA's Resource-metric CPU target computes average utilization **against the request** — the request is the autoscaler's sensitivity dial, not just a scheduling reservation. An earlier all-burst-to-limit attempt (1Gi/500m app requests) made the benchmarked N=100–250 shape (7-day G2 peak: 2.37 cores on `rag-api-server`) read **474 %+ utilization**, so any real traffic slammed the HPA straight to `maxReplicas` while idle pods read ~3 % (meaningless resolution at both ends). The restored app requests put the 70 % target at **~1.4 cores (large) / ~1.05 cores (medium)** of sustained per-container CPU: interactive load (0.2–0.5 core, 10–25 % utilization) never triggers scale-up, sustained heavy load does. Memory requests cover the measured busy peak (~3.1 Gi) so a busy pod is not a node-pressure eviction candidate. Qdrant **stays burst-to-limit** — it is deliberately NOT the HPA signal (excluded from autoscaling, shard topology), and its measured CPU is ~60 m, so returning idle headroom there remains correct; its long-standing caveat stands: a Qdrant pod whose working set far exceeds its request is an eviction candidate as collections approach the memory limits.

Prerequisites and operations:

- **metrics-server** must be installed on the cluster for the Resource metric (or a prometheus-adapter + ServiceMonitor for the custom-metric path below).
- Prefer bounding by quota first: at the large chart's default shape (8Gi limit per app container, ×2 containers), `maxReplicas: 8` still reserves real capacity when busy — set `maxReplicas` to what your quota actually fits.
- Once the ServiceMonitor is on (`metrics.serviceMonitor`) and a prometheus-adapter exists, a pods metric on `rag_http_requests_total` (per-pod rate, AverageValue target) scales on true request demand instead of CPU — the API is partly I/O-bound on the embedder, so request-rate is the more faithful signal.
- An HPA pairing nicely with this: add a PodDisruptionBudget (`minAvailable: 1`) so node drains respect the HA floor (not shipped by the charts yet).

## 13. EzUA app-catalog status (vendor labels on pod templates)

The EzUA app catalog derives a card's status from **labeled pods**. This chart's Service/Deployment objects were always labeled (the cluster's vendor-label policy plus the embed-batcher Service fix), but its pod templates were not — making it the only unlabeled-pod app on the G2 cluster and the only "Unknown" card in the catalog (2026-10). All long-running workloads' pod templates (API Deployment, Qdrant StatefulSet, Redis, embed-batcher) now render `hpe-ezua/type: vendor-service` + `hpe-ezua/app: <chart>`, matching what every Ready app's pods carry.

Deliberate exception: **cron-job pods stay unlabeled**. The platform's `assign-custom-scheduler-for-ezua-user-vendor-pods` Kyverno policy stamps `schedulerName` on labeled pods at every admission, and pod updates (the Job controller's finalizer removal) then reject the immutable-field change — labeled cron pods become undeletable (G2 2026-09-24 incident). Catalog status is unaffected: it reads the long-running pods. Both bars are render-pinned in `tests/full_pipeline/test_vendor_pod_labels_render.py` (labels present on long-running pod templates; absent from selectors — immutable field — and from all CronJob pod specs).

## Summary table

| Dimension | Base chart (`helm/`) | Medium chart (`helm-scale-medium/`) | Large chart (`helm-scale-large/`) |
|---|---|---|---|
| API replicas | 1 | 2 (HPA-managed when `autoscaling.enabled`, §12) | 4 (HPA-managed when `autoscaling.enabled`, §12) |
| Server | `uvicorn` (1 event loop) | `gunicorn` + 2 `UvicornWorker`s | `gunicorn` + 4 `UvicornWorker`s |
| Qdrant | 1 instance (HTTP) | 2-node cluster (gRPC, sharded) | 3-node cluster (gRPC, sharded) |
| Qdrant client timeout | 30s | 30s | 30s |
| Unlock cache | in-process dict | Redis (cross-pod) | Redis (cross-pod) |
| Unlock identity | client IP | authenticated user (oauth2-proxy) | authenticated user (oauth2-proxy) |
| Count sync on read | every call | deferred (admin-only) | deferred (admin-only) |
| Sync thread pool (`app.syncPoolSize`) | 12 (base default) | 32 per worker | 64 per worker |
| MCP thread pool (`app.mcpPoolSize`) | 64 (default) | 32 (explicit) | 64 (explicit) |
| Model HTTP pool (`modelPool.maxConnections`) | 30 connections | 100 connections | 200 connections |
| Pod anti-affinity (`antiAffinity.enabled`) | no | yes (spread across nodes) | yes (spread across nodes) |
| File PVC | 50Gi RWMany | 50Gi RWMany | 100Gi RWMany |
| Qdrant PVC | 50Gi RWMany (shareable) | 25Gi RWOnce per replica | 100Gi RWOnce per replica |
| Embed-batcher singleton | — | no | yes (`embedBatcher.enabled`) |
| Redis unlock cache | — | `redis.enabled: true` | `redis.enabled: true` |

Everything not listed above (model URLs, `security.*` hardening, `s3.*`, `ezua.*`, `backups.*`, `watchedSources.*`, `metrics.*`, the `models.*`/`modelSecrets.*` blocks) is identical to the base chart's `values.yaml` — see [DEPLOYMENT.md](DEPLOYMENT.md) for those keys.

## Resource requirements

The three charts share the same templates and component layout (API Deployment with `rag-api-server` + `rag-mcp-server` sidecar containers, Qdrant StatefulSet, optional Redis). They differ only in replica counts, per-container resource requests/limits, and PVC sizing — all driven by `values.yaml`.

### Per-component breakdown

Each API pod runs **two** containers (`rag-api-server` + `rag-mcp-server`), so per-pod app resources are `2 × resources.app`.

| Component | Chart | Replicas | Per-unit req | Per-unit lim | Storage |
|---|---|---|---|---|---|
| **App** (2 ctr/pod) | `helm/` | 1 | 4 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| | `helm-scale-medium/` | 2 | 8 Gi / 3 cpu | 16 Gi / 6 cpu | — |
| | `helm-scale-large/` | 4 | 8 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| **Qdrant** | `helm/` | 1 | 16 Gi / 4 cpu | 32 Gi / 8 cpu | 50 Gi |
| | `helm-scale-medium/` | 2 | 4 Gi / 1 cpu | 20 Gi / 6 cpu | 25 Gi × 2 |
| | `helm-scale-large/` | 3 | 4 Gi / 1 cpu | 32 Gi / 8 cpu | 100 Gi × 3 |
| **Redis** | `helm/` | — | — | — | — |
| | `helm-scale-medium/` | 1 | 256 Mi / 100 m | 512 Mi / 500 m | — |
| | `helm-scale-large/` | 1 | 256 Mi / 100 m | 512 Mi / 500 m | — |

### Cluster-wide totals

| | `helm/` (base) | `helm-scale-medium/` | `helm-scale-large/` |
|---|---|---|---|
| **Req memory** | 20 Gi | 24.25 Gi (+21 %) | 44.75 Gi (+124 %) |
| **Req CPU** | 8.0 | 8.1 (+1 %) | 19.6 (+145 %) |
| **Lim memory** | 48 Gi | 72.5 Gi (+51 %) | 161.5 Gi (+236 %) |
| **Lim CPU** | 16.0 | 24.5 (+53 %) | 58.5 (+266 %) |
| **PVC total** | 100 Gi | 100 Gi (+0 %) | 400 Gi (+300 %) |

> The totals are the **`values.yaml` defaults** (large = 4 API replicas × 4 workers), recomputed from the per-component table after the HPA-calibration change (§12, 2026-10 — supersedes the burst-to-limit totals). Limits are unchanged by that change; the app **requests** went back up (500m/1Gi → 1.5/2 cpu, 4Gi) because the HPA reads utilization against them.

Percentages are relative to the base chart. Requests are deliberately calibrated to the HPA (§12), not minimized: at these levels the 70 % CPU target fires only on sustained real demand — at the cost of reserving more scheduling headroom per replica than a burst-to-limit shape would, which is the trade-off you accept when you run an HPA. **Total PVC stays at 100 Gi (medium)** — the data PVC is unchanged at 50 Gi (no resize needed on upgrade) and the two Qdrant PVCs are 25 Gi each (replacing the base chart's single 50 Gi Qdrant PVC).

### PVC layout

| PVC | `helm/` | `helm-scale-medium/` | `helm-scale-large/` |
|---|---|---|---|
| Data (RWX, shared) | 50 Gi | 50 Gi | 100 Gi |
| Qdrant (RWO, per-replica) | 50 Gi × 1 | 25 Gi × 2 | 100 Gi × 3 |
| **Total** | **100 Gi** | **100 Gi** | **400 Gi** |

The data PVC is annotated `helm.sh/resource-policy: keep` in all three charts, so it survives `helm uninstall`. The Qdrant PVCs are created via the StatefulSet `volumeClaimTemplates` and are **not** kept on uninstall.

> **Upgrade caveat:** moving from `helm/` to either scale variant changes the Qdrant StatefulSet `volumeClaimTemplates` (access mode and/or size), which Kubernetes treats as immutable. The existing
> Qdrant StatefulSet and its PVC must be deleted before `helm upgrade`; Qdrant vectors are lost and must be re-indexed. The data PVC is unaffected.

## Benchmarks

On our SE G2 cluster, with the helm-scale-medium chart, I was able to achieve about 50 requests / second with top_k=10 retrieval from N=100 (emulated) concurrent users. This is up from around 10 without dynamic batching on the base helm chart, or 4 in a very mutimodal dataset (pre-caching base descriptions).

```
(base) andrew-bydlon@rag-throughput-0:~$ python3 benchmark.py   --mode mcp   --url http://rag-mcp-server-mcp.mm-rag-mcp.svc.cluster.local:9090/mcp   --api-url http://rag-mcp-server-api.mm-rag-mcp.svc.cluster.local   --dataset andrew-test-dataset   --N 100 --duration 120 --top-k 10   --call-timeout 60 
Checking server health at http://rag-mcp-server-api.mm-rag-mcp.svc.cluster.local ... OK
Query pool: 40 queries

============================================================
  BENCHMARK CONFIGURATION
============================================================
  Mode:            MCP
  MCP endpoint:    http://rag-mcp-server-mcp.mm-rag-mcp.svc.cluster.local:9090/mcp
  REST API:        http://rag-mcp-server-api.mm-rag-mcp.svc.cluster.local
  Dataset:         andrew-test-dataset
  Users (N):       100
  Duration:        120.0s
  Ramp-up:         5.0s
  Top K:           10
  Reranker:        OFF
  Queries:         40
  Password:        none
  TLS verify:      ON
  Call timeout:    60s
============================================================

Launching 100 concurrent users (mcp mode) ...
   elapsed   requests    success     failed  rate (req/s)
  --------  ---------  ---------  ---------  ------------
      5.1s         52         52          0          10.3
     10.1s        305        305          0          30.3
     15.1s        573        573          0          38.0
     20.1s        802        802          0          40.0
     25.1s       1097       1097          0          43.7
     30.1s       1356       1356          0          45.0
     35.1s       1659       1659          0          47.2
     40.1s       1924       1924          0          47.9
     45.1s       2139       2139          0          47.4
     50.2s       2401       2401          0          47.9
     55.2s       2638       2638          0          47.8
     60.2s       2907       2907          0          48.3
     65.2s       3201       3201          0          49.1
     70.2s       3412       3412          0          48.6
     75.2s       3617       3617          0          48.1
     80.2s       3862       3862          0          48.2
     85.2s       4160       4160          0          48.8
     90.2s       4413       4413          0          48.9
     95.2s       4662       4662          0          49.0
    100.2s       4932       4932          0          49.2
   elapsed   requests    success     failed  rate (req/s)
  --------  ---------  ---------  ---------  ------------
    105.2s       5226       5226          0          49.7
    110.2s       5461       5461          0          49.5
    115.2s       5734       5734          0          49.8
    120.2s       5996       5996          0          49.9
    125.2s       6220       6220          0          49.7

============================================================
  BENCHMARK RESULTS
============================================================
  Duration:            127.08s
  Total requests:      6267
  Successful:          6267
  Failed:              0
  Success rate:        100.0%
  Response rate:       49.32 req/s

  Latency (ms):
    min:     325.04
    mean:    1921.83
    median:  1893.31
    p95:     2938.42
    p99:     3511.81
    max:     5021.09

  Avg results/query:   1
============================================================
```

