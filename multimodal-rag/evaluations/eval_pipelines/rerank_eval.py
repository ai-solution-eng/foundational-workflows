#!/usr/bin/env python3
"""
Page-level reranker evaluation on ViDoRe v3.

Flow:
  1. Embed all corpus + queries with batched text API
  2. For each query, retrieve top-k by cosine similarity
  3. Reranker scores all top-k candidates against the query
  4. Keep top N (reranker_top_k) after reranking

Uses ThreadPoolExecutor to parallelize reranker calls (8 concurrent).
"""

import argparse
import concurrent.futures
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytrec_eval
from datasets import load_dataset
from tqdm import tqdm

from multimodal_rag.utils.langchain_overrides import MultiModalReranker
from multimodal_rag.utils.pcai_models import qwen3_vl_8B, qwen3_vl_reranker_8B

RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "reranker"


def compute_metrics(results, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut", "recall", "map_cut"})
    scores = evaluator.evaluate(results)
    metrics = {}
    for k in [1, 3, 5, 10, 20]:
        ndcg = [scores[q].get(f"ndcg_cut_{k}", 0) for q in scores]
        recall = [scores[q].get(f"recall_{k}", 0) for q in scores]
        mAP = [scores[q].get(f"map_cut_{k}", 0) for q in scores]
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg)) if ndcg else None
        metrics[f"recall_at_{k}"] = float(np.mean(recall)) if recall else None
        metrics[f"map_at_{k}"] = float(np.mean(mAP)) if mAP else None
    return metrics


ALL_DATASETS = [
    "vidore/vidore_v3_hr",
    "vidore/vidore_v3_finance_en",
    "vidore/vidore_v3_industrial",
    "vidore/vidore_v3_pharmaceuticals",
    "vidore/vidore_v3_computer_science",
    "vidore/vidore_v3_energy",
    "vidore/vidore_v3_physics",
    "vidore/vidore_v3_finance_fr",
]


def evaluate_reranker(dataset_name: str, embed_top_k: int, reranker_top_k: int):
    print(f"\n{'=' * 70}")
    print(f"Reranker eval: {dataset_name}")
    print(f"{'=' * 70}")

    # ── Load data ────────────────────────────────────────────────────
    t0 = time.time()
    corpus = load_dataset(dataset_name, "corpus", split="test")
    queries = load_dataset(dataset_name, "queries", split="test")
    qrels_ds = load_dataset(dataset_name, "qrels", split="test")

    corpus_ids = [str(cid) for cid in corpus["corpus_id"]]
    corpus_texts = list(corpus["markdown"])
    query_ids = [str(qid) for qid in queries["query_id"]]
    query_texts = list(queries["query"])

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["query_id"])][str(row["corpus_id"])] = row["score"]

    print(
        f"  Loaded {len(corpus_ids)} corpus, {len(query_ids)} queries in {time.time() - t0:.1f}s"
    )
    del corpus, queries, qrels_ds

    # ── First stage: text embedding ──────────────────────────────────
    print("  Embedding (batched text API) …")
    qwen3_vl_8B.remote()
    client = qwen3_vl_8B.client
    model_name = qwen3_vl_8B.model_name

    t0 = time.time()
    c_vecs = []
    for start in range(0, len(corpus_texts), 200):
        batch = corpus_texts[start : start + 200]
        resp = client.embeddings.create(
            model=model_name, input=batch, encoding_format="float"
        )
        c_vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
    C = np.array(c_vecs)
    print(f"  Corpus: {C.shape} in {time.time() - t0:.1f}s")

    t0 = time.time()
    q_vecs = []
    for start in range(0, len(query_texts), 200):
        batch = query_texts[start : start + 200]
        resp = client.embeddings.create(
            model=model_name, input=batch, encoding_format="float"
        )
        q_vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
    Q = np.array(q_vecs)
    print(f"  Queries: {Q.shape} in {time.time() - t0:.1f}s")

    # ── Cosine similarity → top-k per query ──────────────────────────
    t0 = time.time()
    Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12)
    C = C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-12)
    scores = Q @ C.T

    first_stage_results: Dict[str, Dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        first_stage_results[qid] = {
            corpus_ids[ci]: float(scores[qi, ci]) for ci in range(len(corpus_ids))
        }
    print(f"  Cosine scoring done in {time.time() - t0:.2f}s")

    metrics_first = compute_metrics(first_stage_results, qrels)
    print(f"  First-stage NDCG@5: {metrics_first.get('ndcg_at_5', 0):.4f}")

    # ── Second stage: reranker (parallelized) ────────────────────────
    print(
        f"\n  Reranking top-{embed_top_k} candidates (keeping top-{reranker_top_k}) …"
    )
    qwen3_vl_reranker_8B.remote()
    reranker = MultiModalReranker(qwen3_vl_reranker_8B)
    corpus_text_map = dict(zip(corpus_ids, corpus_texts))

    # Build candidate lists once
    all_candidates: List[tuple] = []
    for qid, qtext in zip(query_ids, query_texts):
        top = sorted(first_stage_results[qid].items(), key=lambda x: -x[1])[
            :embed_top_k
        ]
        cids = [c[0] for c in top]
        docs = [corpus_text_map.get(cid, "") for cid in cids]
        all_candidates.append((qid, qtext, cids, docs))

    def _score_one(args):
        qid, query, cids, docs = args
        t1 = time.time()
        try:
            scores_rr = reranker.score(query, docs)
            qs = scores_rr if isinstance(scores_rr, list) else scores_rr.tolist()
        except Exception as e:
            qs = [0.0] * len(docs)
        elapsed = time.time() - t1
        sorted_pairs = sorted(zip(cids, qs), key=lambda x: -x[1])[:reranker_top_k]
        return qid, {cid: float(sc) for cid, sc in sorted_pairs}, elapsed

    total_rerank_time = 0.0
    reranked_results: Dict[str, Dict[str, float]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_score_one, args) for args in all_candidates]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reranking",
            leave=False,
        ):
            qid, result, elapsed = fut.result()
            reranked_results[qid] = result
            total_rerank_time += elapsed

    n_q = len(query_ids)
    print(
        f"  Reranking: {total_rerank_time:.1f}s total, "
        f"{total_rerank_time / n_q:.2f}s/query avg, "
        f"{n_q / total_rerank_time:.1f} queries/s"
    )

    metrics_rerank = compute_metrics(reranked_results, qrels)
    nd5_first = metrics_first.get("ndcg_at_5", 0) or 0
    nd5_rerank = metrics_rerank.get("ndcg_at_5", 0) or 0
    print(
        f"  Reranked NDCG@5: {nd5_rerank:.4f}  (Δ vs first-stage: {nd5_rerank - nd5_first:+.4f})"
    )

    # ── Save ─────────────────────────────────────────────────────────
    ds_short = dataset_name.replace("/", "_")
    out_dir = RESULTS_ROOT / ds_short
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_corpus": len(corpus_ids),
        "num_queries": n_q,
        "config": {"embed_top_k": embed_top_k, "reranker_top_k": reranker_top_k},
        "metrics": {
            "first_stage_text_only": metrics_first,
            "second_stage_reranked": metrics_rerank,
        },
        "timing": {
            "total_rerank_time_s": round(total_rerank_time, 1),
            "avg_rerank_time_per_query_s": round(total_rerank_time / n_q, 2),
            "queries_per_second": round(n_q / total_rerank_time, 2),
        },
    }
    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved to {path}")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="ViDoRe v3 page-level reranker evaluation"
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--embed-top-k", type=int, default=50)
    parser.add_argument("--reranker-top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS
    print("ViDoRe v3 — Page-Level Reranker Evaluation")
    print(f"  Datasets:       {len(datasets)}")
    print(f"  Embed top-k:    {args.embed_top_k}")
    print(f"  Reranker top-k: {args.reranker_top_k}")
    print(f"  Workers:        {args.workers}")
    print()

    total_start = time.time()
    for ds_name in datasets:
        evaluate_reranker(ds_name, args.embed_top_k, args.reranker_top_k)
    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
