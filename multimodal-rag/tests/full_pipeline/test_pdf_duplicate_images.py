"""Offline regression tests for the PDF image-duplication explosion (3.4.0).

``page.get_images(full=True)`` lists an image XObject once per *usage* in the
page's content stream, and grid/catalog layouts reference the same tile
hundreds of times.  The block-level extractor used to emit one standalone
image doc per (dict-block x region) bbox match and never marked standalone
yields as emitted, so a repeated tile fanned out into unbounded duplicate
``[Image on page N]`` documents (observed: 250k+ Qdrant points from a single
10-page PDF).  These tests pin the fix: one doc per distinct image.

Run::

    python tests/full_pipeline/test_pdf_duplicate_images.py    # standalone
    pytest tests/full_pipeline/test_pdf_duplicate_images.py    # under pytest
"""

import io
import os
import sys
from pathlib import Path

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pymupdf
from PIL import Image

from multimodal_rag.input_processing.pdf_processor import PDFProcessor


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    # 140x100 clears the PDF_MIN_IMG_SIDE=100 meaningfulness filter.
    buf = io.BytesIO()
    Image.new("RGB", (140, 100), color).save(buf, format="PNG")
    return buf.getvalue()


def _build_pdf(path: Path, images: list[bytes], placements_per_image: int = 2) -> None:
    """One page placing each image *placements_per_image* times.

    Inserting the same stream repeatedly makes PyMuPDF list one
    ``get_images(full=True)`` entry per usage, all sharing a single xref —
    the duplication pattern that triggered the explosion.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=600)
    for color in images:
        img = _png_bytes(color)
        for n in range(placements_per_image):
            x = 20 + n * 260
            page.insert_image(pymupdf.Rect(x, 20, x + 120, 90), stream=img)
    doc.save(str(path))
    doc.close()


def test_repeated_same_image_yields_one_doc():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "dup.pdf"
        _build_pdf(pdf, [(10, 200, 10)], placements_per_image=2)

        blocks = PDFProcessor().extract_text_blocks(str(pdf))
        image_blocks = [b for b in blocks if b["block_type"] == "image"]
        urls = {b["image_data_url"] for b in image_blocks}
        assert len(image_blocks) == 1, f"expected 1 image doc, got {len(image_blocks)}"
        assert len(urls) == 1

        docs = list(PDFProcessor().extract_chunks_iter(str(pdf), ocr=False))
        image_docs = [d for d in docs if str(d.get("text", "")).startswith("[Image on page")]
        assert len(image_docs) == 1, f"expected 1 standalone image doc, got {len(image_docs)}"


def test_distinct_images_are_not_over_deduped():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "distinct.pdf"
        _build_pdf(pdf, [(10, 200, 10), (200, 10, 10)], placements_per_image=2)

        blocks = PDFProcessor().extract_text_blocks(str(pdf))
        urls = {b["image_data_url"] for b in blocks if b["block_type"] == "image"}
        assert len(urls) == 2, f"expected 2 distinct images, got {len(urls)}"

        docs = list(PDFProcessor().extract_chunks_iter(str(pdf), ocr=False))
        image_docs = [d for d in docs if str(d.get("text", "")).startswith("[Image on page")]
        assert len(image_docs) == 2, f"expected 2 standalone image docs, got {len(image_docs)}"


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  {name} ... OK")
            except Exception:
                failed += 1
                print(f"  {name} ... FAIL")
                traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
