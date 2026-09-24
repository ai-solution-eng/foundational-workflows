# Verification & Troubleshooting

Confirm a freshly applied deployment is healthy, exercise it end-to-end, and debug model-endpoint problems — everything here assumes the chart is already applied per [DEPLOYMENT.md](DEPLOYMENT.md).

## 1. Deployment came up

In the PCAI UI the deployment should reach **Ready**. Pods and logs are visible from the PCAI workload view. For operators with cluster access, the usual inspection works unchanged (**optional,
operator-only** — not part of the PCAI deployment flow):

```bash
# Optional (operator): pod status, logs, local port-forward
kubectl get pods -l app=rag-mcp-server
kubectl logs -l app=rag-mcp-server -c rag-api-server --tail=50
kubectl logs -l app=rag-mcp-server -c rag-mcp-server --tail=50
kubectl port-forward deployment/rag-mcp-server 8000:8000   # API+UI locally
kubectl port-forward deployment/rag-mcp-server 8001:9090   # MCP locally
```

Quick local checks after a port-forward:

```bash
curl http://localhost:8000/healthz          # → {"status": "ok"}
curl http://localhost:8000/api/datasets     # → {"datasets": [...]} (empty initially)
```

## 2. Health endpoints

| Surface | What it tells you |
|---|---|
| `GET /healthz` | Liveness of API and MCP sidecar. Deliberately does **not** probe model endpoints — a remote embedder outage is not fixed by restarting this pod. |
| `GET /readyz` | Readiness. Gates on the background embedder probe after `MODEL_HEALTH_FAIL_THRESHOLD` (default 3) consecutive failures — an embedder outage drops the pod out of Service rotation without a restart loop. |
| `GET /api/admin/health` | Full picture: model endpoints, Qdrant status + per-replica shard placement (scale charts), PVC usage. Includes the live `models.embedder` probe result (checked every `MODEL_HEALTH_INTERVAL`, default 60s). |
| `GET /api/admin/connections` | Live-checks **every** configured model endpoint (`/v1/models`) → per-role `healthy` / `not_provided` / `unhealthy`. The management page's "Test connections" button runs the same check. |
| `GET /metrics` | Prometheus counters/histograms (HTTP, ingest, Qdrant ops, caches, hybrid-vs-dense). Unauthenticated like `/healthz`. |

The management page (`/manage`) shows the live embedder status indicator and a Qdrant card; on scale charts (no Qdrant PVC mounted on the API pod) it shows per-replica shard placement instead of exact
PVC bytes.

## 3. Exercise it

1. **Web UI** at the VirtualService endpoint (`https://rag-mcp-server.<domain>`; SSO via oauth2-proxy). Create a dataset, upload a PDF/image/video, and search. Local-only preview: the port-forward
   above.
2. **REST search** (full API reference in [FEATURES.md](FEATURES.md) § REST API):

```bash
BASE=http://localhost:8000   # or the VirtualService endpoint
curl -H 'X-Dataset-Password: secret' \
  "$BASE/api/datasets/my_dataset/search?q=aurora+borealis&top_k=5&use_reranker=true&reranker_top_k=3"
```

3. **MCP client**: connect to `https://rag-mcp-server.<domain>/mcp` (or `http://localhost:8001/mcp` after the port-forward) and run `list_datasets` / `search_dataset`. Connection configs in
   [FEATURES.md](FEATURES.md) § MCP server.
4. **Multi-replica check** (scale charts): `search_dataset` from any replica must work; if an MCP `unlock_dataset` on pod A isn't honored on pod B, pass `password=` per tool call (the MCP unlock cache
   is per-process by design).

## 4. Model-endpoint validation & debugging

The model endpoints are the usual source of trouble. Validation playbook:

**Embedder (`models.embedder`):**

- `GET /v1/models` on the endpoint must list the model (this is the startup probe and the live-check). A 404/401 here is an endpoint/auth problem, not a RAG problem.
- Text-only calls go through the OpenAI-compatible `POST /v1/embeddings` with `input: [str1, str2, ...]` — client-side pre-formatted with the Qwen3-VL chat template. Multimodal (image/video) inputs
  use per-doc `messages` requests; the OpenAI python client's `client.embeddings.create` cannot carry dicts, so image/joint calls only work through the `messages` path.
- Reference parity: the deployed vLLM/OpenAI-client path matches the local vLLM implementation to cosine distance ~`1e-4` for text and ~`1e-3` for image/joint embeddings; the sentence-transformers
  reference deviates more on multimodal data (up to ~0.21 in embedding deltas) but preserves rankings. Full comparison numbers in [BENCHMARKS.md](BENCHMARKS.md) § Embedding validation.
- After an embedder change, existing datasets fail loudly (HTTP 409 / `EmbedderMismatchError`) instead of mixing vectors — rebuild via the Recreate button / `POST /api/admin/datasets/{name}/recreate`.

**Reranker (`models.reranker`):**

- Served on non-standard endpoints under the base URL with `/v1` stripped: `POST /score` (`text_1` query + `text_2` documents → relevance scores) and `POST /rerank` (`query` + `documents` → ranked
  list). Both are exercised by `use_reranker=true` searches.
- Offline CrossEncoder (sentence-transformers) loading of Qwen3-VL-Reranker-8B fails with `TypeError: LogitScore.__init__() missing 1 required positional argument: 'true_token_id'` unless you drop
  `{"true_token_id": 9693, "false_token_id": 2152}` into `1_CausalScoreHead/config.json` under the model's HF snapshot dir. The vLLM deployment used on PCAI does not need this.
- Reranker scores are `[0,1]` relevance probabilities and rank-consistent with the offline vLLM reference across text/image/joint modalities (see [BENCHMARKS.md](BENCHMARKS.md)).

**VLM / ASR:** optional — if unreachable, startup logs a warning and ingestion degrades (no captioning/transcription; media may be dropped with an ingest warning surfaced to the UI). `GET
/api/admin/connections` shows them as `unhealthy` or `not_provided`.

**Score sanity:** every result carries a `score_kind` (`cosine` | `rrf` | `reranker`). On hybrid (dense+BM25) collections a fused score is **rank arithmetic, not a similarity** — recurring values like
0.5 / 0.3333 / 1.0 mean "won N of 2 lanes", not a cosine; `embedding_score` is `null` on those results (or a true recomputed dense cosine when hybrid embedding scores are enabled). Trust the order,
not the magnitude — and don't debug the embedder because a memory recall "scored 0.5".

## 5. Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| Pods stuck in `Pending` | PVC not binding | PVC status in PCAI (storage class, RWX support) |
| Containers exit at startup: media-token secret missing | `security.mediaTokenSecret` removed/empty — both servers refuse to start without it | Set the value in the *Helm Values* editor and re-apply |
| API server crash-looping at boot | Embedder unreachable (`models.embedder.url` wrong / token missing) | Container logs; `GET /v1/models` on the embedder |
| MCP tools return "dataset not found" | Dataset created against a different Qdrant/namespace | Compare the MCP and REST dataset lists; check the Qdrant service the pod targets |
| Search returns 0 results | Empty dataset, or filters exclude everything | Document list in the web UI; drop `file_types`/`date_*` filters |
| Every score looks like 0.5 / 0.3333 / 1.0 | Hybrid RRF fusion scores (rank arithmetic), not broken embeddings | `score_kind: rrf` in the result; see § Score sanity above |
| Qdrant OOMKilled | Vector count exceeds memory | Raise `resources.qdrant.limits.memory` (~40 GiB per 1M × 4096-dim vectors) |
| 404 at the VirtualService endpoint | Vendor labels not applied (Kyverno policy absent/blocked) | Confirm the workload carries `hpe-ezua/type: vendor-service`; see DEPLOYMENT.md § Hosted trial |
| 401 at the VirtualService endpoint | OAuth2 token missing/expired | `oauth2-proxy` logs in `istio-system` |
| Endpoint shows `rag-mcp-server.<domain>` unresolved | Domain value not set | `ezua.virtualService.endpoint` in values (`${DOMAIN_NAME}` resolves on customer PCAI) |
| MCP `404 Session not found` | Multi-replica deployment running pre-stateless MCP mode | v1.3.0+ runs `stateless_http=True`; upgrade the image |
| Model swap had no effect | Hot reload disabled (`CONFIG_DIR` unset) | Charts mount `-config`/`-model-keys` — check the deployment's volumes; otherwise re-apply to roll |
| Ingest warnings "media dropped" | Neither the embedder nor VLM/ASR supports the modality | Configure a VLM/ASR endpoint, or accept the drop (warning is surfaced to the UI) |

Memory-specific symptoms (opencode plugin not writing, OWUI recall empty, dataset/password errors) have their own table in [MEMORY.md](MEMORY.md) § Troubleshooting.
