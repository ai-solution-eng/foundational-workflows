"""Measure BM25 sparse-index footprint per chunk. Writes a JSON artifact."""

import json
import pathlib
import statistics
import sys

sys.path.insert(0, "src")
from multimodal_rag.utils import bm25

ROOT = pathlib.Path(".")
GROUPS = {
    "prose": ["README.md", "documentation/SCALE.md", "USAGE.md"],
    "code": [
        "src/multimodal_rag/vector_store.py",
        "src/multimodal_rag/rag_system.py",
        "src/multimodal_rag/api_server.py",
        "src/multimodal_rag/mcp_server.py",
    ],
}


def chunks(text, n):
    toks = bm25.tokenize(text)
    return [" ".join(toks[i : i + n]) for i in range(0, max(0, len(toks) - n // 4), n)]


res = {}
for label, files in GROUPS.items():
    per = []
    for f in files:
        p = ROOT / f
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for cs in (2048, 8192):
            for ch in chunks(txt, cs):
                per.append((cs, len(bm25.term_counts(ch))))
    for cs in (2048, 8192):
        vals = [v for c, v in per if c == cs]
        if vals:
            res[f"{label}_{cs}"] = {
                "chunks": len(vals),
                "mean": round(statistics.mean(vals), 1),
                "median": statistics.median(vals),
                "p90": sorted(vals)[int(len(vals) * 0.9)],
                "max": max(vals),
            }

# Posting-size model. Qdrant inverted-index posting = doc_id + value.
#   doc_id: u32 = 4 B (fixed).
#   value : depends on SparseIndexParams.datatype, which this repo does NOT set,
#           so the server default float32 (4 B) applies. NOT uint8.
#
# Cross-check: Qdrant's published BM42 figure is 13 MB for ~530k docs x 5.6
# elements => 4.59 B/posting. That measurement was taken WITH datatype: uint8
# (the BM42 article says so explicitly), i.e. u32 id + 1 B value + ~0 overhead.
# Scaling to our float32-valued postings: 4.59 - 1 + 4 = 7.59 B/posting.
BM42_BPP_UINT8 = 13 * 1024 * 1024 / (530_000 * 5.6)  # ~4.59
BPP = BM42_BPP_UINT8 - 1 + 4  # float32-valued postings
DENSE_INT8_BYTES = 4096

for k, v in res.items():
    sp = v["mean"] * BPP
    v.update(
        {
            "sparse_bytes_per_point": round(sp, 1),
            "dense_int8_bytes_per_point": DENSE_INT8_BYTES,
            "sparse_over_dense": round(sp / DENSE_INT8_BYTES, 2),
            "sparse_GB_per_1M": round(sp * 1e6 / 1024**3, 2),
            "dense_int8_GB_per_1M": round(DENSE_INT8_BYTES * 1e6 / 1024**3, 2),
        }
    )

out = {
    "bytes_per_posting_float32_values": round(BPP, 2),
    "calibration": "BM42 13MB/530k/5.6elem measured WITH datatype=uint8; "
    "scaled by -1B uint8 +4B float32 for our unset (default f32) datatype",
    "caveat": "Dev-checkout tokenizer is the regex fallback, NOT the bundled Qwen "
    "tokenizer used in production; subword tokenization yields MORE unique "
    "terms, so these ratios are a LOWER BOUND.",
    "results": res,
}
pathlib.Path("tests/measurements/sparse-footprint.json").write_text(json.dumps(out, indent=2))
print(f"BPP (float32 values) = {BPP:.2f}")
for k in sorted(res):
    v = res[k]
    print(
        f"{k:<14} mean={v['mean']:<7} "
        f"sparse={v['sparse_GB_per_1M']:>5.2f} GB/1M "
        f"dense={v['dense_int8_GB_per_1M']:>4.2f} GB/1M "
        f"ratio={v['sparse_over_dense']:.2f}x (lower bound)"
    )
