# Scaling: the `helm-scale-medium/` and `helm-scale-large/` charts

The base `helm/` chart runs a single API replica with a single Qdrant
instance — optimised for simplicity. The `helm-scale-large/` chart trades
that for horizontal capacity. This document explains every layer of the
scale chart and how it achieves its goal.

A medium variant, `helm-scale-medium/`, reuses the scale-chart
architecture (gunicorn workers, multi-replica Qdrant, Redis unlock cache)
but dials the replica counts and per-container resources back so it can be
tested as a drop-in replacement on a cluster sized for the base chart. The
[Resource requirements](#resource-requirements) section compares all three.

## 1. Multiple API replicas + load balancing

`values.yaml` — `replicaCount: 4`

The Deployment runs 4 pods (vs 1 in the base chart). Traffic is
distributed across them by:

- A **ClusterIP Service** (`templates/service.yaml`) with
  `targetPort: http`, which kube-proxy load-balances round-robin.
- An **Istio VirtualService** (`templates/virtualservice.yaml`) that
  routes external traffic from the `ezaf-gateway` to the `-api` service.
  It defines **three timeout tiers** so slow operations don't tie up the
  gateway:
  - Batch uploads / SSE streams (`/api/datasets/*/batch-*`):
    `longTimeout: 3600s`
  - MCP searches (`/mcp` prefix): `longTimeout: 3600s`
  - All other API routes: `timeout: 300s`

## 2. Gunicorn with multiple Uvicorn workers per pod

`templates/deployment.yaml` — gunicorn command

Instead of a single `uvicorn` process, the scale chart runs **gunicorn
with `UvicornWorker`** and `workers: 4` (`values.yaml` → `app.workers`).
This gives 4 independent event loops per pod, so effective concurrency =
`replicas × workers × syncPoolSize` = 4 × 4 × 64 = **1024 concurrent
blocking operations** cluster-wide. Each worker process gets its own
`sync_pool`, `httpx` connection pools, and Qdrant clients.

## 3. Multi-replica Qdrant cluster with sharding

`templates/qdrant-statefulset.yaml` — `replicas: 3` with cluster mode

This is the biggest architectural difference. The base chart runs a
single Qdrant instance; the scale chart runs a **3-node Qdrant cluster**:

- `QDRANT__CLUSTER__ENABLED=true` and `QDRANT__CLUSTER__P2P_PORT=6335`
  enable Qdrant's distributed consensus and peer-to-peer bootstrapping.
- A **headless Service** (`clusterIP: None`,
  `templates/qdrant-service.yaml`) gives each Qdrant pod a stable DNS
  identity (`rag-mcp-server-qdrant-0`, `-1`, `-2`) — required for
  StatefulSet peer discovery.
- The p2p port (6335) is exposed alongside gRPC (6334) and HTTP (6333)
  so nodes can coordinate.
- Collections are **sharded** across the 3 nodes, so read load (every
  search hits Qdrant) is spread out.

Each Qdrant replica gets its own PVC via `volumeClaimTemplates` with
`ReadWriteOnce` (per-pod storage), and larger resources: `16Gi–32Gi`
memory, `4–8` CPU (`values.yaml` → `resources.qdrant`).

Because the per-replica Qdrant PVCs are **not** mounted on the API pod
(in the base chart they are mounted read-only so `/api/admin/health` can
report exact usage), the management page instead surfaces per-replica
**shard placement** plus the configured per-replica size
(`QDRANT_PVC_SIZE` = `persistence.qdrant.size`) from Qdrant's `/cluster`
API. This gives a storage-spread estimate (which replicas hold how many
shards × PVC size) without exec/kubectl. For exact bytes per replica use
`kubectl exec <qdrant-N> -- df -h /qdrant/storage`.

## 4. Redis-backed cross-pod unlock cache

`templates/redis.yaml` + `templates/configmap.yaml`

In the base chart, password-unlocked datasets are cached **in-process
memory** (`_UNLOCK_CACHE` dict in `api_server.py`). With 4 replicas × 4
workers = 16 separate processes, a user would have to re-enter their
password whenever routed to a different pod/worker. The scale chart adds:

- A **Redis Deployment + Service** (`templates/redis.yaml`) running
  `redis-server` in-memory (no persistence — unlock state is ephemeral).
- `REDIS_URL` is injected into the ConfigMap, and the backend
  (`api_server.py`) lazily builds a Redis client. `_unlock_cache_get` /
  `_unlock_cache_set` use Redis with a TTL (in-app `UNLOCK_TTL` default
  1800 s — not chart-configurable) so a single unlock works across all
  pods.
- Falls back to in-memory dict if Redis is unavailable.

Unlock identity is also derived from the authenticated user
(oauth2-proxy headers: `X-Auth-Request-Email` / `X-Auth-Request-User`)
rather than client IP, which is unreliable behind Istio/oauth2-proxy
(many users may share one proxy IP).

## 5. Deferred document-count sync (avoids write races)

`values.yaml` — `rag.deferCountSync: true` → `RAG_DEFER_COUNT_SYNC=true`

In the base chart, every `get_dataset()` / `list_datasets()` call syncs
the document count from Qdrant and writes it back to `meta.json` on the
shared PVC. With 4 replicas, this causes:

1. **Write races** — multiple pods writing `meta.json` concurrently on
   the NFS PVC.
2. **Qdrant load** — every read triggers an extra Qdrant round-trip.

The scale chart sets `RAG_DEFER_COUNT_SYNC=true`, which makes
`list_datasets()` and `get_dataset()` skip the Qdrant count sync
entirely (`dataset_manager.py`). Counts are only synced on explicit
admin requests. This eliminates both the race and the extra Qdrant load.

## 6. gRPC Qdrant client with hard timeout

`values.yaml` → `qdrant.client` → `templates/configmap.yaml`

The base chart uses HTTP to talk to Qdrant (no timeout). The scale chart
sets:

- `QDRANT_PREFER_GRPC=true` — gRPC is more efficient than HTTP at high
  QPS (binary framing, multiplexed streams). Honored in `rag_system.py`.
- `QDRANT_CLIENT_TIMEOUT=30` — a hard 30s timeout so a hung Qdrant node
  can't pin a `sync_pool` thread indefinitely. With limited thread-pool
  slots, one hung request could otherwise cascade.

## 7. Larger, tuned thread pools

`values.yaml` → `app.syncPoolSize` / `app.mcpPoolSize` →
`templates/configmap.yaml`

The blocking RAG work (embedding, Qdrant calls, file processing) runs in
a `ThreadPoolExecutor`. The scale chart raises:

- `SYNC_POOL_SIZE: 64` (base default: 12) — per-worker thread pool for
  `sync_wrapper_safe` (`utils/general_tools.py`). Each of the 4 workers
  × 4 pods gets 64 threads.
- `MCP_POOL_SIZE: 64` — per-pod thread pool for offloading MCP tool
  bodies (`mcp_server.py`), so a single slow search can't stall every
  concurrent MCP client.

## 8. Larger httpx connection pools for model endpoints

`values.yaml` → `modelPool` → `templates/configmap.yaml`

The embedder is on every search's critical path. The base chart defaults
to `max_connections=30`. The scale chart raises:

- `MODEL_POOL_MAX_CONNECTIONS: 200`
- `MODEL_POOL_MAX_KEEPALIVE_CONNECTIONS: 50`

Honored in `utils/pcai_model_classes.py`, which builds `httpx.Limits`
for the async model clients. This prevents the embedder's HTTP connection
pool from becoming the bottleneck under concurrent load.

## 9. Pod anti-affinity (spreads replicas across nodes)

`templates/deployment.yaml` + `values.yaml` → `antiAffinity.enabled: true`

A `preferredDuringSchedulingIgnoredDuringExecution` anti-affinity rule
with weight 100 tells the scheduler to spread the 4 API pods across
different nodes when possible. This means a node failure takes down at
most one pod, and no single node becomes a CPU/memory hotspot for the
RAG workload.

## 10. Shared file PVC (ReadWriteMany)

`templates/pvc.yaml` + `values.yaml` → `persistence.data`

All 4 API replicas mount the same file PVC (`/data`) as `ReadWriteMany`
(NFS-backed via `gl4f-filesystem`). This means any pod can serve any
uploaded file — there's no need to replicate files across pods. The PVC
is also annotated with `helm.sh/resource-policy: keep` so it survives
`helm uninstall`.

## Summary table

| Dimension | Base chart (`helm/`) | Medium chart (`helm-scale-medium/`) | Large chart (`helm-scale-large/`) |
|---|---|---|---|
| API replicas | 1 | 2 | 4 |
| Server | `uvicorn` (1 event loop) | `gunicorn` + 2 `UvicornWorker`s | `gunicorn` + 4 `UvicornWorker`s |
| Qdrant | 1 instance (HTTP) | 2-node cluster (gRPC, sharded) | 3-node cluster (gRPC, sharded) |
| Qdrant client timeout | none | 30s | 30s |
| Unlock cache | in-process dict | Redis (cross-pod) | Redis (cross-pod) |
| Unlock identity | client IP | authenticated user (oauth2-proxy) | authenticated user (oauth2-proxy) |
| Count sync on read | every call | deferred (admin-only) | deferred (admin-only) |
| Sync thread pool | 12 | 32 per worker | 64 per worker |
| MCP thread pool | 64 (default) | 32 (explicit) | 64 (explicit) |
| Model HTTP pool | 30 connections | 100 connections | 200 connections |
| Pod anti-affinity | no | yes (spread across nodes) | yes (spread across nodes) |
| File PVC | 50Gi RWMany | 50Gi RWMany | 100Gi RWMany |
| Qdrant PVC | 50Gi RWMany (shareable) | 25Gi RWOnce per replica | 100Gi RWOnce per replica |

## Resource requirements

The three charts share the same templates and component layout (API
Deployment with `rag-api-server` + `rag-mcp-server` sidecar containers,
Qdrant StatefulSet, optional Redis). They differ only in replica counts,
per-container resource requests/limits, and PVC sizing — all driven by
`values.yaml`.

### Per-component breakdown

Each API pod runs **two** containers (`rag-api-server` +
`rag-mcp-server`), so per-pod app resources are `2 × resources.app`.

| Component | Chart | Replicas | Per-unit req | Per-unit lim | Storage |
|---|---|---|---|---|---|
| **App** (2 ctr/pod) | `helm/` | 1 | 4 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| | `helm-scale-medium/` | 2 | 5 Gi / 3 cpu | 16 Gi / 6 cpu | — |
| | `helm-scale-large/` | 4 | 8 Gi / 4 cpu | 16 Gi / 8 cpu | — |
| **Qdrant** | `helm/` | 1 | 16 Gi / 4 cpu | 32 Gi / 8 cpu | 50 Gi |
| | `helm-scale-medium/` | 2 | 10 Gi / 3 cpu | 20 Gi / 6 cpu | 25 Gi × 2 |
| | `helm-scale-large/` | 3 | 16 Gi / 4 cpu | 32 Gi / 8 cpu | 100 Gi × 3 |
| **Redis** | `helm/` | — | — | — | — |
| | `helm-scale-medium/` | 1 | 256 Mi / 100 m | 512 Mi / 500 m | — |
| | `helm-scale-large/` | 1 | 256 Mi / 100 m | 512 Mi / 500 m | — |

### Cluster-wide totals

| | `helm/` (base) | `helm-scale-medium/` | `helm-scale-large/` |
|---|---|---|---|
| **Req memory** | 20 Gi | 30.25 Gi (+51 %) | 80.25 Gi (+301 %) |
| **Req CPU** | 8.0 | 12.1 (+51 %) | 28.1 (+251 %) |
| **Lim memory** | 48 Gi | 72.5 Gi (+51 %) | 160.5 Gi (+234 %) |
| **Lim CPU** | 16.0 | 24.5 (+53 %) | 56.5 (+253 %) |
| **PVC total** | 100 Gi | 100 Gi (+0 %) | 400 Gi (+300 %) |

Percentages are relative to the base chart. The medium variant is tuned so
that **total requests are ~50 % above the base chart** while **total PVC
stays at 100 Gi** — the data PVC is unchanged at 50 Gi (no resize needed
on upgrade) and the two Qdrant PVCs are 25 Gi each (replacing the base
chart's single 50 Gi Qdrant PVC).

### PVC layout

| PVC | `helm/` | `helm-scale-medium/` | `helm-scale-large/` |
|---|---|---|---|
| Data (RWX, shared) | 50 Gi | 50 Gi | 100 Gi |
| Qdrant (RWO, per-replica) | 50 Gi × 1 | 25 Gi × 2 | 100 Gi × 3 |
| **Total** | **100 Gi** | **100 Gi** | **400 Gi** |

The data PVC is annotated `helm.sh/resource-policy: keep` in all three
charts, so it survives `helm uninstall`. The Qdrant PVCs are created via
the StatefulSet `volumeClaimTemplates` and are **not** kept on uninstall.

> **Upgrade caveat:** moving from `helm/` to either scale variant changes
> the Qdrant StatefulSet `volumeClaimTemplates` (access mode and/or size),
> which Kubernetes treats as immutable. The existing Qdrant StatefulSet
> and its PVC must be deleted before `helm upgrade`; Qdrant vectors are
> lost and must be re-indexed. The data PVC is unaffected.

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

