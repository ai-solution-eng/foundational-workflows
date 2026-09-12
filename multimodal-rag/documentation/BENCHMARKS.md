# Benchmarks & Evaluations

Measured performance of the deployed system (throughput/latency), retrieval-quality evaluations (ViDoRe v3), and the offline validation that pinned the embedding/reranker model integrations. Numbers
are from the SE G2 cluster unless noted; methodology and scripts are linked so runs are reproducible.

## 1. Deployment throughput (concurrent searches)

Measured with the **universal endpoint benchmarker** in `../ModelBenchmarker`
(`PYTHONPATH=src python -m model_benchmarker.endpoint_benchmarker --mode mcp`; N emulated concurrent
users issuing random `search_dataset` MCP calls — the path LLM clients actually use; the retired
in-repo `tests/benchmark.py` was its hardwired predecessor and has been removed), against the
deployed release on SE G2. Query pool of 40 queries, `top_k=10`.

**Variant scaling** — `helm-scale-medium`, N=100, 120s run, no reranker: **49.3 req/s**, 100 % success (min 325 ms / mean 1.92 s / p95 2.94 s / p99 3.51 s latency). The same 4-replica × 4-worker shape
measured 72.8 req/s @ N=250 on v3.1.8 (p99 5.2 s). Context: ~10 req/s without dynamic batching on the base chart, ~4 req/s on a heavily multimodal dataset before query-media pre-caching.

**Reranker cost** — N=100, 30s runs, `helm-scale-medium`:

| Dataset | Reranker | Requests | Rate | Mean | p95 | p99 |
|---|---|---|---|---|---|---|
| Multimodal (664 docs, `andrew-test-dataset`) | off | 1647 | 44.2 r/s | 1.84 s | 2.82 s | 3.18 s |
| Multimodal | on (`reranker_top_k=3`) | 212 | 3.4 r/s | 22.0 s | 33.2 s | 34.4 s |
| Text-only (10.5k docs, stacks-project) | off | 1650 | 45.5 r/s | 1.84 s | 2.72 s | 3.37 s |
| Text-only | on (`reranker_top_k=3`) | 218 | 3.4 r/s | 22.4 s | 34.1 s | 34.6 s |

The cross-encoder dominates latency when enabled (~20× on these runs) — enable `use_reranker` deliberately, not by default. Saved raw runs: `tests/benchmark_results/*.json`.

**Dynamic-batching A/B (v3.1.8):** the opt-in idle early-flush for the embedding query batchers cut single-query mean latency from ~370–550 ms (full 200 ms batch window) to **238 ms**, with burst
batching structurally untouched. An earlier unguarded variant (10 ms, no batch-size guard) was benchmark-convicted — it split batches under load and collapsed N=100 throughput from 53 to 15 r/s — and
was reverted; the shipped version only fires when a *small, stalled* queue (≤2 items) is waiting. Details: CHANGELOG 3.2.0.

**Reproduce** (run from `../ModelBenchmarker`; the universal tool needs only the MCP URL — no second REST URL):

```bash
PYTHONPATH=src python -m model_benchmarker.endpoint_benchmarker --mode mcp \
  --url http://<release>-mcp.<ns>.svc.cluster.local:9090/mcp \
  --dataset <dataset> -N 100 --duration 120 \
  --arg top_k=10 \
  [--arg use_reranker=true --arg reranker_top_k=3] \
  --output results.json --md results.md
```

## 2. Retrieval quality — ViDoRe v3

`evaluations/eval_pipelines/` evaluates against the 8 ViDoRe v3 domains (HR, finance EN/FR, industrial, pharmaceuticals, computer science, energy, physics) with the deployed PCAI model endpoints:

- **`run_eval.py`** — embed-only retrieval per domain × modality (text / image / both): page-level corpus, pytrec_eval nDCG/recall/MAP/MRR. Results in `results/embed-only/`.
- **`rerank_eval.py`** — page-level rerank A/B: embed top-k by cosine → reranker rescores → keep `reranker_top_k` (ThreadPoolExecutor, 8 concurrent). Results in `results/reranker/`.
- **`pipeline_benchmark.py`** — the *actual* pipeline: original PDFs through `PDFProcessor.extract_chunks()` (element-level chunks with images), page qrels mapped to elements, text-only vs multimodal
  corpus embeddings. Results in `results/pipeline/`.

**Embed-only, nDCG@5 / Recall@5 by domain** (text modality, all 8 domains):

| Domain | nDCG@5 | Recall@5 |
|---|---|---|
| computer_science | 0.649 | 0.589 |
| pharmaceuticals | 0.563 | 0.537 |
| energy | 0.532 | 0.562 |
| finance_en | 0.483 | 0.457 |
| physics | 0.433 | 0.372 |
| hr | 0.420 | 0.376 |
| finance_fr | 0.351 | 0.352 |
| industrial | 0.349 | 0.337 |
| **Mean (8 domains)** | **0.472** | **0.448** |

Modality ablation on the two domains with image corpora (finance_en, hr): image-only and text+image ("both") land within ~0.02 nDCG@5 of text-only (image 0.454 / both 0.455 vs text 0.457 mean over
those two domains) — the joint multimodal embedding gives up nothing measurable on page-level retrieval while enabling cross-modal queries.

**Full-pipeline benchmark** (element-level chunks, `pipeline_benchmark.py`; nDCG@5):

| Domain | Chunks | Queries | Text-only | Multimodal |
|---|---|---|---|---|
| computer_science | 1356 | 1290 | 0.282 | 0.265 |
| pharmaceuticals | 2457 | 2184 | 0.190 | 0.194 |
| hr | 1103 | 1908 | 0.188 | 0.193 |
| energy | 2248 | 1848 | 0.201 | 0.201 |
| physics | 1781 | 1812 | 0.160 | 0.139 |
| finance_en | 2927 | 1854 | 0.142 | 0.145 |
| finance_fr | 2394 | 1920 | 0.111 | 0.112 |
| industrial | 5077 | 1698 | 0.112 | 0.100 |

Element-level scores sit below page-level by construction (page qrels mapped onto fine-grained chunks penalize partial matches). Text-only vs multimodal corpus embeddings are within noise of each
other per domain. Ingest-side timing from the same runs (computer_science, 1356 chunks): text corpus embedding 69 s vs multimodal (text+image) 270 s — the multimodal lane's per-doc `messages` requests
are the bulk of ingest cost for image-heavy corpora.

The hybrid dense+BM25 lane (v3.4.0) targets exactly the dense-weak domains here (code identifiers, error codes, keys); validate changes by re-running `run_eval.py` plus a hand-built code/log
spot-check set, as noted in [FEATURES.md](FEATURES.md) § Roadmap.

## 3. Embedding validation

Test set: 18 samples — text, image, and joint text+image for 6 reference internet images (captions authored per image), embedded with Qwen3-VL-Embedding-8B (4096-d). The block-diagonal structure of
the embedding cosine matrix behaves as expected: exact matches = 1.0, samples sharing an image and/or caption ≈ 0.64–0.83, unrelated pairs ≤ ~0.35 — i.e. the joint space separates both by content and
by modality pairing.

**Deployed (PCAI vLLM via OpenAI-compatible client) vs offline references** — max deltas over the 6-sample caption subset:

| Comparison | Text | Image | Joint | Similarities |
|---|---|---|---|---|
| vLLM local Python ↔ vLLM OpenAI client (deployed path) | 1.5e-4 | 1.1e-3 | 1.1e-3 | 3.4e-3 |
| vLLM local Python ↔ HuggingFace (sentence-transformers) | 2.7e-4 | 0.142 | 0.212 | 0.084 |

Takeaways:

- The deployed client reproduces the vLLM reference to ~1e-4 (text) / ~1e-3 (media) — the integration is faithful; this is the number that justifies sending base64 data URLs (equivalence of base64 vs
  http-link inputs was verified separately).
- HuggingFace/sentence-transformers deviates materially on **multimodal** inputs while preserving the same rankings — use vLLM-served endpoints as ground truth for this model.
- The OpenAI python client (`client.embeddings.create`) cannot carry dict/multimodal inputs at all (its API accepts strings only) — hence the `messages`-format path for image/video documents.
- The LangChain `Embeddings` adapter initially produced non-equivalent representations vs the local variants even for plain text (a deviation from OpenAI-client defaults); the custom adapter in
  `utils/pcai_models.py` matches offline text-embedding results.

## 4. Reranker validation

Test set: same 18 samples; the reranker (Qwen3-VL-Reranker-8B via `MultiModalReranker.score`) puts cross-comparisons on a `[0,1]` relevance scale. For each of the 6 reference texts, matched
image/joint candidates score 0.85–0.96 while unrelated candidates score ≤ 0.05 — clean separation (e.g. one text row: matched pair 0.960 / 0.848 vs next-best 0.236).

Ranking parity between the deployed PCAI vLLM implementation and the offline reference was checked by comparing score orderings across all modality pairings (text↔image, text↔joint, image↔joint): the
top-ranked candidates match across modalities in the strong majority of rows, and score deltas are mostly e-3–e-2. The largest observed delta (0.15, on one text↔joint pair) favors the PCAI
implementation (0.62 vs 0.47). Full matrices: the upstream comparison dumps (`tests/embeddings/comparison.txt`, `tests/reranker/comparison.txt` in the internal projects repo).

Known setup gotcha (offline CrossEncoder only): loading Qwen3-VL-Reranker-8B through sentence-transformers requires adding `{"true_token_id": 9693, "false_token_id": 2152}` to
`1_CausalScoreHead/config.json` in the HF snapshot — otherwise `TypeError: LogitScore.__init__() missing 1 required positional argument: 'true_token_id'`. The vLLM deployment used on PCAI needs
nothing; runtime debug steps live in [VERIFICATION.md](VERIFICATION.md).

## 5. End-to-end RAG pipeline example

Queries against a mixed corpus (local LLM papers, photos, a gameplay video) via `rag.generate(route=True, ...)` — the retrieval-LLM flow with routing, optional rerank, and VLM post-processing. Timing
profile per query (SE G2, pre-optimization measurements kept as a shape reference):

| Query (intent) | Route | Retrieve | Postproc (VLM) | LLM | Total |
|---|---|---|---|---|---|
| "kid getting his hair cut?" (images) | 1.7 s | 0.6 s | 5.4 s (6 media) | 5.2 s | 12.9 s |
| "child climbing through a snow tunnel?" (images, rerank) | 1.2 s | 1.7 s | 3.4 s (3 media) | 4.4 s | 10.7 s |
| "DeepSeek V4 Flash approaches" (text, top_k=20, rerank) | 1.6 s | 1.8 s | 0 s (text-only) | 14.0 s | 17.3 s |
| "videos of an old video game?" (video, rerank) | 1.1 s | 5.8 s | 10.4 s (5 media) | 5.0 s | 22.3 s |

All four retrieved the right modality (images from disk *and* from within PDFs, video segments), and the answers cite the correct sources. Current numbers are substantially better after the batching
work in § 1 — these runs predate the query-vector cache and idle early-flush.

<details>
<summary>Raw transcript with per-stage timings (click to expand)</summary>

<pre><code style="white-space: pre-wrap;">
Can you find data of or about a kid getting his hair cut? Logging Info: 1.74s route(llm) → RAG needed | 0.57s retrieve — 10 docs (top_k=10, reranker=no) 3.41s–5.39s vlm describe — 1 media item each |
5.40s postproc — VLM/ASR conversion (6 docs with media) 5.23s llm — generation (192 tokens?) [total 12.93s] → Six correct images identified (DSC01364–DSC01370), clippers/setting/interaction described.

Can you find and describe images of a child climbing through a snow tunnel? Logging Info: 1.16s route | 1.69s retrieve — 3 docs (reranker=yes) | 3.43s postproc (3 media) | 4.39s llm [total 10.67s] →
Three correct images (DSC01378–DSC01380) with clothing/scene descriptions.

Can you describe the most important new approaches in the DeepSeek V4 Flash family? Logging Info: 1.57s route | 1.75s retrieve — 5 docs (top_k=20, reranker=yes) | 0.00s postproc (text-only) | 14.00s
llm [total 17.32s] → CSA/HCA hybrid attention, FP4/FP8 precision, mHC, 256-expert MoE, on-policy distillation.

Can you show me videos of an old video game? Logging Info: 1.12s route | 5.79s retrieve — 5 docs (reranker=yes) | 10.42s postproc (5 media) | 5.02s llm [total 22.34s] → Super Smash Bros. (N64, 1999),
Yoshi vs Captain Falcon on Hyrule Castle.
</code></pre>

</details>
