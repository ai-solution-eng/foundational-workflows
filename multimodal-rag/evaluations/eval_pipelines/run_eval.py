#!/usr/bin/env python3
"""
Run ViDoRe v3 pipeline evaluations using the remote PCAI model deployments.

Evaluates every combination of:
  - Dataset (all 8 ViDoRe v3 domains)
  - Modality (text, image, both)
  - Pipeline (embed-only, embed+rerank)

Results are saved under evaluations/eval_pipelines/results/<pipeline>/<modality>/
"""

import argparse
import base64
import io
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

import pytrec_eval

from multimodal_rag.utils.langchain_overrides import MultiModalReranker
from multimodal_rag.utils.pcai_models import qwen3_vl_8B, qwen3_vl_reranker_8B

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _build_corpus_input(
    text: str, image: Image.Image, modality: str
) -> str | Dict[str, Any]:
    """Build an input acceptable by MultiModalEmbeddings.embed_documents().

    The embedding model accepts:
      * ``str``                      → plain text
      * ``{"text": …, "image": …}``  → text + image URL (data URI accepted)
    """
    if modality == "text":
        return text
    uri = _pil_to_data_uri(image.convert("RGB"))
    if modality == "image":
        return {"image": uri, "text": ""}
    # both
    return {"text": text, "image": uri}


def _build_reranker_doc(
    text: str, image: Image.Image, modality: str
) -> str | Dict[str, Any]:
    """Build a document acceptable by MultiModalReranker.score()."""
    if modality == "text":
        return text
    uri = _pil_to_data_uri(image.convert("RGB"))
    if modality == "image":
        return {"image": uri, "text": ""}
    return {"text": text, "image": uri}


# ═══════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Embedding & scoring
# ═══════════════════════════════════════════════════════════════════════


def _init_embedder():
    qwen3_vl_8B.remote()
    return qwen3_vl_8B


def _init_reranker():
    qwen3_vl_reranker_8B.remote()
    return MultiModalReranker(qwen3_vl_reranker_8B)


def embed_texts_batched(
    texts: List[str],
    cfg: Any,
    label: str = "",
    batch_size: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Embed plain texts using the standard OpenAI embeddings API (batched).

    Much faster than per-item message-based embedding since the server can
    process the entire batch with internal batching.
    """
    info: Dict[str, Any] = {}
    client = cfg.client
    model_name = cfg.model_name

    t0 = time.time()
    all_vecs: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(
            model=model_name,
            input=batch,
            encoding_format="float",
        )
        # Responses are ordered by index
        all_vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)

    matrix = np.array(all_vecs, dtype=np.float32)
    info[f"{label}_embed_time_s"] = round(time.time() - t0, 2)
    info[f"num_{label}"] = len(texts)
    info[f"{label}_dim"] = matrix.shape[1]
    return matrix, info


def embed_multimodal(
    inputs: List[str | Dict[str, Any]],
    model: Any,
    label: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Embed multimodal inputs via the chat-message API (one item at a time)."""
    info: Dict[str, Any] = {}
    t0 = time.time()

    vecs = model.embed_documents(inputs)

    info[f"{label}_embed_time_s"] = round(time.time() - t0, 2)
    info[f"num_{label}"] = len(vecs)
    if vecs:
        info[f"{label}_dim"] = len(vecs[0])

    matrix = np.array(vecs, dtype=np.float32)
    return matrix, info


def cosine_score_matrix(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12)
    C = C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-12)
    return Q @ C.T


def scores_to_results(
    scores: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        results[qid] = {
            corpus_ids[ci]: float(scores[qi, ci]) for ci in range(len(corpus_ids))
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# Reranking
# ═══════════════════════════════════════════════════════════════════════


def rerank_results(
    results: Dict[str, Dict[str, float]],
    query_texts: List[str],
    query_ids: List[str],
    corpus_images: List[Image.Image],
    corpus_texts: List[str],
    corpus_ids: List[str],
    reranker: Any,
    modality: str,
    reranker_top_k: int = 20,
    top_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    reranked: Dict[str, Dict[str, float]] = {}

    for qi, qid in enumerate(tqdm(query_ids, desc="Reranking queries", leave=False)):
        query = query_texts[qi]
        candidates = sorted(results[qid].items(), key=lambda x: -x[1])[:reranker_top_k]
        cids = [c[0] for c in candidates]

        # Build documents for reranking (image + text)
        cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
        docs: List[str | Dict[str, Any]] = []
        for cid in cids:
            idx = cid_to_idx[cid]
            docs.append(
                _build_reranker_doc(corpus_texts[idx], corpus_images[idx], modality)
            )

        scores = reranker.score(query, docs)
        q_scores: List[float] = scores if isinstance(scores, list) else scores.tolist()

        sorted_pairs = sorted(zip(cids, q_scores), key=lambda x: -x[1])[:top_k]
        reranked[qid] = {cid: float(sc) for cid, sc in sorted_pairs}

    return reranked


# ═══════════════════════════════════════════════════════════════════════
# Metrics & persistence
# ═══════════════════════════════════════════════════════════════════════


def compute_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
) -> Dict[str, Optional[float]]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut", "recall", "map_cut"})
    scores = evaluator.evaluate(results)

    k_values = [1, 3, 5, 10, 20]
    metrics: Dict[str, Optional[float]] = {}
    for k in k_values:
        ndcg_vals = [scores[q].get(f"ndcg_cut_{k}", 0) for q in scores]
        recall_vals = [scores[q].get(f"recall_{k}", 0) for q in scores]
        map_vals = [scores[q].get(f"map_cut_{k}", 0) for q in scores]
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_vals)) if ndcg_vals else None
        metrics[f"recall_at_{k}"] = float(np.mean(recall_vals)) if recall_vals else None
        metrics[f"map_at_{k}"] = float(np.mean(map_vals)) if map_vals else None

    return metrics


def save_results(
    metrics: Dict[str, Optional[float]],
    info: Dict[str, Any],
    dataset_name: str,
    pipeline: str,
    modality: str,
) -> Path:
    ds_short = dataset_name.replace("/", "_")
    out_dir = RESULTS_ROOT / pipeline / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset_name,
        "pipeline": pipeline,
        "modality": modality,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "info": info,
    }
    path = out_dir / f"{ds_short}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════════════
# Evaluation per dataset
# ═══════════════════════════════════════════════════════════════════════


def load_corpus_data(
    dataset_name: str,
    need_images: bool,
) -> Tuple[List[str], Optional[List[Image.Image]], List[str]]:
    """Load corpus, optionally skipping the image column to save memory."""
    cols = ["corpus_id", "markdown"]
    if need_images:
        cols.append("image")
    corpus = load_dataset(dataset_name, "corpus", split="test")  # keep reference
    ids: List[str] = [str(cid) for cid in corpus["corpus_id"]]
    texts: List[str] = list(corpus["markdown"])
    images: Optional[List[Image.Image]] = list(corpus["image"]) if need_images else None
    return ids, images, texts


def evaluate_dataset(
    dataset_name: str,
    modalities: List[str],
    pipelines: List[str],
    top_k: int,
    reranker_top_k: int,
    embed_batch_size: int,
):
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    # ── Load data ─────────────────────────────────────────────────────
    t_load = time.time()
    query_data = load_dataset(dataset_name, "queries", split="test")
    qrels_ds = load_dataset(dataset_name, "qrels", split="test")

    query_ids: List[str] = [str(qid) for qid in query_data["query_id"]]
    query_texts: List[str] = list(query_data["query"])

    need_images = any(m in ("image", "both") for m in modalities)
    corpus_ids, corpus_images, corpus_texts = load_corpus_data(
        dataset_name, need_images
    )

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["query_id"])][str(row["corpus_id"])] = row["score"]

    print(
        f"  Loaded {len(corpus_ids)} corpus, {len(query_ids)} queries"
        f" in {time.time() - t_load:.1f}s"
    )

    # ── Embed corpus (once per modality) ──────────────────────────────
    cfg = _init_embedder()
    mm_model = cfg.model  # MultiModalEmbeddings instance
    # Limit concurrency for multimodal requests — the VLLM server serializes
    # beyond ~10 concurrent embeddings.
    mm_model.chunk_size = 10
    cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    for mod in modalities:
        print(f"\n  Embedding corpus (modality={mod}) …")
        if mod == "text":
            matrix, info = embed_texts_batched(corpus_texts, cfg, label="corpus")
        else:
            assert corpus_images is not None, "Images not loaded"
            inputs = [
                _build_corpus_input(t, img, mod)
                for img, t in zip(corpus_images, corpus_texts)
            ]
            matrix, info = embed_multimodal(inputs, mm_model, label="corpus")
            del inputs  # free memory
        cache[mod] = (matrix, info)
        print(f"    done ({info['corpus_embed_time_s']:.1f}s, {matrix.shape})")

    # Free corpus images after all modality embeddings done
    if corpus_images is not None:
        corpus_images.clear()

    # ── Embed queries (once – text is always the same) ────────────────
    print(f"\n  Embedding queries …")
    q_matrix, qinfo = embed_texts_batched(query_texts, cfg, label="query")
    print(f"    done ({qinfo['query_embed_time_s']:.1f}s, {q_matrix.shape})")

    # ── Run each pipeline × modality ──────────────────────────────────
    reranker = _init_reranker() if "embed+rerank" in pipelines else None

    for pipeline in pipelines:
        for mod in modalities:
            label = f"{pipeline}/{mod}"
            print(f"\n  --- {label} ---")

            c_matrix, cinfo = cache[mod]
            t0 = time.time()

            scores = cosine_score_matrix(q_matrix, c_matrix)
            results = scores_to_results(scores, query_ids, corpus_ids)
            retrieval_time = time.time() - t0

            info: Dict[str, Any] = {**cinfo, **qinfo}
            info["total_retrieval_time_s"] = round(retrieval_time, 2)

            if pipeline == "embed+rerank" and reranker is not None:
                t1 = time.time()
                results = rerank_results(
                    results,
                    query_texts,
                    query_ids,
                    corpus_images,
                    corpus_texts,
                    corpus_ids,
                    reranker,
                    mod,
                    reranker_top_k,
                    top_k,
                )
                info["rerank_time_s"] = round(time.time() - t1, 2)

            metrics = compute_metrics(results, qrels)
            print(
                f"  NDCG@5: {metrics.get('ndcg_at_5', 'N/A'):.4f}  |  "
                f"NDCG@10: {metrics.get('ndcg_at_10', 'N/A'):.4f}  |  "
                f"Recall@10: {metrics.get('recall_at_10', 'N/A'):.4f}"
            )

            path = save_results(metrics, info, dataset_name, pipeline, mod)
            print(f"  -> saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ViDoRe v3 evaluation using remote PCAI models"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to evaluate (default: all 8)",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=["text", "image", "both"],
        choices=["text", "image", "both"],
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["embed-only", "embed+rerank"],
        choices=["embed-only", "embed+rerank"],
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--reranker-top-k", type=int, default=20)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS

    print("ViDoRe v3 Evaluation")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  Modalities: {args.modalities}")
    print(f"  Pipelines:  {args.pipelines}")
    print()

    total_start = time.time()
    for ds_name in datasets:
        evaluate_dataset(
            dataset_name=ds_name,
            modalities=args.modalities,
            pipelines=args.pipelines,
            top_k=args.top_k,
            reranker_top_k=args.reranker_top_k,
            embed_batch_size=args.embed_batch_size,
        )

    print(f"\n{'=' * 70}")
    print(f"All done in {time.time() - total_start:.1f}s")
    print(f"Results in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
