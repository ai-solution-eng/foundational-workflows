#!/usr/bin/env python3
"""
Benchmark the actual multimodal-rag pipeline against ViDoRe v3.

Processes original PDFs through PDFProcessor.extract_chunks() to get
element-level entries (text blocks + nearby images), maps page-level
qrels to element-level, and evaluates the full pipeline (embed + reranker).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytrec_eval
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from multimodal_rag.input_processing.pdf_processor import PDFProcessor
from multimodal_rag.utils.langchain_overrides import MultiModalReranker
from multimodal_rag.utils.pcai_models import qwen3_vl_8B, qwen3_vl_reranker_8B


RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "pipeline"


# ═══════════════════════════════════════════════════════════════════════
# PDF download & chunking
# ═══════════════════════════════════════════════════════════════════════


def download_pdfs(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Download all PDFs for a dataset. Returns {doc_id: local_path}."""
    meta = load_dataset(dataset_name, "documents_metadata", split="test")
    pdfs: Dict[str, str] = {}
    for row in tqdm(meta, desc="Downloading PDFs"):
        doc_id = row["doc_id"]
        filename = row["file_name"]
        local = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=f"pdfs/{filename}",
        )
        pdfs[doc_id] = local
    return pdfs


def chunk_pdfs(pdfs: Dict[str, str]) -> List[Dict[str, Any]]:
    """Run PDFProcessor.extract_chunks() on all PDFs.

    Returns a list of chunk dicts with added ``_doc_id`` and ``_page``
    fields so we can trace back to the original document.
    """
    proc = PDFProcessor()
    all_chunks: List[Dict[str, Any]] = []
    for doc_id, path in tqdm(pdfs.items(), desc="Chunking PDFs"):
        chunks = proc.extract_chunks(path)
        for c in chunks:
            c["_doc_id"] = doc_id
        all_chunks.extend(chunks)
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════
# Corpus & qrel construction
# ═══════════════════════════════════════════════════════════════════════


def build_element_corpus(
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[Optional[str]], List[str]]:
    """Convert chunks into a corpus of (ids, images, texts).

    Each chunk becomes one corpus entry. ``image`` is a data URI string
    when the chunk has one, None otherwise.
    """
    ids: List[str] = []
    images: List[Optional[str]] = []
    texts: List[str] = []
    for i, c in enumerate(chunks):
        ids.append(f"elem_{i}")
        img = c.get("image", "")
        images.append(img if len(img) > 10 else None)
        texts.append(c.get("text", ""))
    return ids, images, texts


def map_qrels_to_elements(
    dataset_name: str,
    chunks: List[Dict[str, Any]],
    corpus_ids: List[str],
) -> Dict[str, Dict[str, int]]:
    """Map page-level qrels to element-level.

    For each qrel (query → page), ALL chunks belonging to that page are
    treated as relevant.  This is a coarse approximation — future work
    could use ``bounding_boxes`` to pinpoint which specific chunk(s) the
    query actually refers to.
    """
    # Build a lookup: (doc_id, page_number) → list of element corpus_ids
    page_to_elements: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for cid, c in zip(corpus_ids, chunks):
        doc_id = c.get("_doc_id", "")
        page = c.get("page", 0)
        page_to_elements[(doc_id, page)].append(cid)

    # Load the corpus to get (doc_id, page_number) per corpus_id
    corpus = load_dataset(dataset_name, "corpus", split="test")
    corpus_page_map: Dict[str, Tuple[str, int]] = {}
    for row in corpus:
        cid = str(row["corpus_id"])
        corpus_page_map[cid] = (row["doc_id"], row["page_number_in_doc"])

    # Load qrels
    qrels_ds = load_dataset(dataset_name, "qrels", split="test")
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)

    for row in tqdm(qrels_ds, desc="Mapping qrels to elements"):
        qid = str(row["query_id"])
        corpus_id = str(row["corpus_id"])
        doc_id, page = corpus_page_map[corpus_id]

        # All chunks on this page are relevant for this query
        element_ids = page_to_elements.get((doc_id, page), [])
        for eid in element_ids:
            qrels[qid][eid] = row["score"]

    return dict(qrels)


# ═══════════════════════════════════════════════════════════════════════
# Embedding & evaluation (reusing runner helpers)
# ═══════════════════════════════════════════════════════════════════════


def _init_embedder():
    qwen3_vl_8B.remote()
    cfg = qwen3_vl_8B
    mm_model = cfg.model
    mm_model.chunk_size = 10
    return cfg, mm_model


def _init_reranker():
    qwen3_vl_reranker_8B.remote()
    return MultiModalReranker(qwen3_vl_reranker_8B)


def embed_texts_batched(texts, cfg, label=""):
    client = cfg.client
    model_name = cfg.model_name
    t0 = time.time()
    all_vecs = []
    batch_size = 200
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(
            model=model_name, input=batch, encoding_format="float"
        )
        all_vecs.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
    matrix = np.array(all_vecs, dtype=np.float32)
    info = {
        f"{label}_embed_time_s": round(time.time() - t0, 2),
        f"num_{label}": len(texts),
    }
    return matrix, info


def embed_multimodal(inputs, mm_model, label=""):
    t0 = time.time()
    vecs = mm_model.embed_documents(inputs)
    matrix = np.array(vecs, dtype=np.float32)
    info = {
        f"{label}_embed_time_s": round(time.time() - t0, 2),
        f"num_{label}": len(vecs),
    }
    return matrix, info


def cosine_score_matrix(Q, C):
    Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12)
    C = C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-12)
    return Q @ C.T


def scores_to_results(scores, query_ids, corpus_ids):
    results = {}
    for qi, qid in enumerate(query_ids):
        results[qid] = {
            corpus_ids[ci]: float(scores[qi, ci]) for ci in range(len(corpus_ids))
        }
    return results


def compute_metrics(results, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut", "recall", "map_cut"})
    scores = evaluator.evaluate(results)
    k_values = [1, 3, 5, 10, 20]
    metrics = {}
    for k in k_values:
        ndcg_vals = [scores[q].get(f"ndcg_cut_{k}", 0) for q in scores]
        recall_vals = [scores[q].get(f"recall_{k}", 0) for q in scores]
        map_vals = [scores[q].get(f"map_cut_{k}", 0) for q in scores]
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_vals)) if ndcg_vals else None
        metrics[f"recall_at_{k}"] = float(np.mean(recall_vals)) if recall_vals else None
        metrics[f"map_at_{k}"] = float(np.mean(map_vals)) if map_vals else None
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


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


def evaluate_pipeline(dataset_name: str):
    print(f"\n{'=' * 70}")
    print(f"Pipeline benchmark: {dataset_name}")
    print(f"{'=' * 70}")

    # 1. Download & chunk PDFs
    print("\n  Downloading PDFs …")
    t0 = time.time()
    pdfs = download_pdfs(dataset_name)
    print(f"    {len(pdfs)} PDFs downloaded in {time.time() - t0:.1f}s")

    print("  Chunking PDFs …")
    t0 = time.time()
    chunks = chunk_pdfs(pdfs)
    print(f"    {len(chunks)} chunks created in {time.time() - t0:.1f}s")

    # 2. Build element corpus
    print("  Building element corpus …")
    elem_ids, elem_images, elem_texts = build_element_corpus(chunks)
    print(f"    {len(elem_ids)} corpus entries")

    # 3. Map qrels
    print("  Mapping qrels …")
    t0 = time.time()
    qrels = map_qrels_to_elements(dataset_name, chunks, elem_ids)
    print(f"    {len(qrels)} queries with qrels in {time.time() - t0:.1f}s")
    n_rel = sum(len(v) for v in qrels.values())
    print(f"    {n_rel} total relevance judgments")

    # 4. Load queries
    queries_ds = load_dataset(dataset_name, "queries", split="test")
    query_ids = [str(qid) for qid in queries_ds["query_id"]]
    query_texts = list(queries_ds["query"])
    print(f"    {len(query_ids)} queries")

    # 5. Embed with text-only (batched)
    print("\n  Embedding (text-only, batched) …")
    cfg, mm_model = _init_embedder()
    c_matrix, cinfo = embed_texts_batched(elem_texts, cfg, label="corpus")
    q_matrix, qinfo = embed_texts_batched(query_texts, cfg, label="query")
    scores = cosine_score_matrix(q_matrix, c_matrix)
    results_text = scores_to_results(scores, query_ids, elem_ids)
    metrics_text = compute_metrics(results_text, qrels)
    print(f"    Text-only NDCG@5: {metrics_text.get('ndcg_at_5', 0):.4f}")

    # 6. Embed multimodal (text + image for entries that have images)
    print("\n  Embedding (multimodal – text+image for entries with images) …")

    # Limit images per chunk to avoid VLLM processor overload
    # Count base64 PNG headers to estimate image count
    _PNG_HEADER = "iVBOR"
    _IMG_THRESHOLD = 4  # max images per multimodal input

    def _limit_images_in_input(inp):
        if isinstance(inp, dict) and "image" in inp:
            uri = inp["image"]
            n_imgs = uri.count(_PNG_HEADER)
            if n_imgs > _IMG_THRESHOLD:
                return inp["text"]
        return inp

    inputs_mm = [
        _limit_images_in_input({"text": txt, "image": img} if img else txt)
        for img, txt in zip(elem_images, elem_texts)
    ]
    cinfo_mm = {}
    try:
        c_matrix_mm, cinfo_mm = embed_multimodal(inputs_mm, mm_model, label="corpus")
        scores_mm = cosine_score_matrix(q_matrix, c_matrix_mm)
        results_mm = scores_to_results(scores_mm, query_ids, elem_ids)
        metrics_mm = compute_metrics(results_mm, qrels)
        print(f"    Multimodal NDCG@5: {metrics_mm.get('ndcg_at_5', 0):.4f}")
    except Exception as e:
        print(f"    Multimodal embedding failed: {e}")
        print("    Falling back to text-only results for multimodal metrics.")
        metrics_mm = metrics_text
        results_mm = results_text

    # 7. Reranker (optional, sample-based)
    reranker = _init_reranker()
    print("\n  Reranking (first 10 queries) …")
    results_rerank: Dict[str, Dict[str, float]] = {}
    reranker_top_k = 20
    top_k = 10
    for qi, qid in enumerate(tqdm(query_ids[:10], desc="Reranking", leave=False)):
        query = query_texts[qi]
        candidates = sorted(results_mm[qid].items(), key=lambda x: -x[1])[
            :reranker_top_k
        ]
        cids = [c[0] for c in candidates]

        docs = []
        for cid in cids:
            idx = elem_ids.index(cid)
            txt = elem_texts[idx]
            img = elem_images[idx]
            if img and len(img) <= 10000:
                docs.append({"text": txt, "image": img})
            else:
                docs.append(txt)

        try:
            scores_rr = reranker.score(query, docs)
            q_scores = scores_rr if isinstance(scores_rr, list) else scores_rr.tolist()
        except Exception as e:
            print(f"    Reranker error for query {qid}: {e}")
            q_scores = [0.0] * len(docs)
        sorted_pairs = sorted(zip(cids, q_scores), key=lambda x: -x[1])[:top_k]
        results_rerank[qid] = {cid: float(sc) for cid, sc in sorted_pairs}

    if results_rerank:
        metrics_rerank = compute_metrics(results_rerank, qrels)
        print(
            f"    Reranked (10 queries) NDCG@5: {metrics_rerank.get('ndcg_at_5', 0):.4f}"
        )

    # 8. Save results
    ds_short = dataset_name.replace("/", "_")
    out_dir = RESULTS_ROOT / ds_short
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "num_pdfs": len(pdfs),
        "num_chunks": len(chunks),
        "num_queries": len(query_ids),
        "num_qrels": n_rel,
        "metrics": {
            "text_only": metrics_text,
            "multimodal": metrics_mm,
        },
        "info": {
            "corpus_embed_text": cinfo,
            "query_embed": qinfo,
            "corpus_embed_multimodal": cinfo_mm,
        },
    }

    path = out_dir / "results.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to {path}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark on ViDoRe v3")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="Skip datasets with existing results"
    )
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]

    # Checkpoint: skip datasets that already have results
    if args.resume:
        existing = set()
        for p in RESULTS_ROOT.iterdir():
            if p.is_dir() and (p / "results.json").exists():
                existing.add(p.name.replace("vidore_vidore_v3_", "vidore/vidore_v3_"))
        datasets = [
            d
            for d in datasets
            if d.replace("/", "_")
            not in {
                p.name for p in RESULTS_ROOT.iterdir() if (p / "results.json").exists()
            }
        ]
        if not datasets:
            print("All datasets already processed.")
            return

    print("Pipeline Benchmark on ViDoRe v3")
    print(f"  Datasets: {len(datasets)}")
    print()

    total_start = time.time()
    for ds_name in datasets:
        evaluate_pipeline(ds_name)
    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
