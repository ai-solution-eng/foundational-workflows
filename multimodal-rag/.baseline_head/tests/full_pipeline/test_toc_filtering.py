"""Regression tests for ToC/Index noise filtering in PDF extraction.

PyMuPDF frequently flattens a whole ToC/Index page into one or two run-on
lines (dot leaders collapse into '. . . .'), which defeated the original
line-oriented heuristic AND tripped its early return on few lines.  The user-
visible symptom: chunks like 'Contents 1 Introduction 4 2 Architecture 6
2.1 Designs . . . . 7 ...' polluting search results for every topic the
paper covers (they match BOTH the dense and the bm25 lanes).

Covers the pure heuristic (including the exact flattened sample from the G2
corpus) and an end-to-end extraction over a synthetic PDF with a ToC page and
a real content page.

Run:

    python tests/full_pipeline/test_toc_filtering.py    # standalone
    pytest tests/full_pipeline/test_toc_filtering.py    # under pytest
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.input_processing.pdf_processor import PDFProcessor, _is_toc_chunk

FLATTENED_TOC = (
    "Contents 1 Introduction 4 2 Architecture 6 2.1 Designs Inherited from "
    "DeepSeek-V3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7 "
    "2.2 Manifold-Constrained Hyper-Connections . . . . . . . . . . . . . . . 7 "
    "2.3 Hybrid Attention with CSA and HCA . . . . . . . . . . . . . . . . . 9 "
    "2.3.1 Compressed Sparse Attention . . . . . . . . . . . . . . . . . . . 9 "
    "2.3.2 Heavily Compressed Attention . . . . . . . . . . . . . . . . . . 11 "
    "3 General Infrastructures 15 3.1 Fine-Grained Communication-Computation "
    "Overlap in Expert Parallelism . . . . 15 4 Pre-Training 24 4.1 Data "
    "Construction . . . . . . . . . . . . . . . . . . . . . . . . . 24 2"
)


def test_flattened_toc_detected():
    assert _is_toc_chunk(FLATTENED_TOC) is True


def test_classic_line_toc_detected():
    assert _is_toc_chunk("Contents\n1 Introduction ........ 4\n2 Architecture ........ 6") is True


def test_index_page_detected():
    index_chunk = (
        "Index\nattention, 12, 45\nbenchmark, 7\nsparse, 33\ntransformer, 2\n"
        "tokenizer, 19\nembedding, 88\nquantization, 51"
    )
    assert _is_toc_chunk(index_chunk) is True


def test_real_prose_not_flagged():
    real = (
        "We inherit the architecture of DeepSeek-V3 (see section 2.3.1 for the "
        "compressed sparse attention design). The attention mechanism uses a "
        "gating factor of 0.15 and 3 groups. Table 2 summarizes the ablation: "
        "dense baselines reach 68.4 MMLU while the hybrid variant reaches 69.1. "
        "Training uses 4096 GPUs with 12.5 days of runtime."
    )
    assert _is_toc_chunk(real) is False


def test_empty_and_short_text_safe():
    assert _is_toc_chunk("") is False
    assert _is_toc_chunk("2.3.1") is False


def test_e2e_toc_page_filtered_from_extraction():
    import pymupdf

    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "paper.pdf"
        doc = pymupdf.open()
        toc_page = doc.new_page(width=612, height=792)
        y = 72
        for line in (
            "Contents",
            "1 Introduction ................ 4",
            "2 Architecture ................ 6",
            "2.1 Designs Inherited ......... 7",
            "2.3.1 Compressed Sparse Attention ......... 9",
            "3 General Infrastructures ..... 15",
            "4 Pre-Training ................ 24",
        ):
            toc_page.insert_text((72, y), line, fontsize=11)
            y += 18
        content_page = doc.new_page(width=612, height=792)
        content_page.insert_text(
            (72, 100),
            "The compressed sparse attention layer reduces KV cache size by a "
            "factor of four while preserving retrieval fidelity across long "
            "contexts in the DeepSeek architecture.",
            fontsize=11,
        )
        doc.save(str(pdf))
        doc.close()

        chunks = list(PDFProcessor().extract_chunks_iter(str(pdf)))
        texts = [str(c.get("text", "")) for c in chunks]
        assert not any("Introduction ...." in t or t.strip().startswith("Contents") for t in texts), (
            "ToC page must be filtered: " + str(texts)
        )
        assert any("compressed sparse attention" in t.lower() for t in texts), texts


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print("All tests passed!" if not failed else str(failed) + " test(s) failed")
    sys.exit(1 if failed else 0)
