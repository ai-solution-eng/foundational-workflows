"""Offline tests for the OCR fallback for scanned PDFs (roadmap feature 3).

Builds a real "scanned" PDF — a page that is ONLY an image with rendered
text and no text layer — and exercises the real tesseract binary (required;
skipped when absent):

  * image-only page + ``ocr=True`` → chunk text carries the ``[OCR page N]``
    marker and recognised payload.
  * image-only page + ``ocr=False`` → no OCR (original image-only behaviour).
  * normal text page + ``ocr=True`` → untouched (no OCR marker).

Run::

    python tests/full_pipeline/test_pdf_ocr.py    # standalone
    pytest tests/full_pipeline/test_pdf_ocr.py    # under pytest
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from multimodal_rag.input_processing.pdf_processor import (
    PDFProcessor,
    _tesseract_available,
)

pytestmark = pytest.mark.skipif(not _tesseract_available(), reason="tesseract CLI not on PATH")


def _build_scanned_pdf(path: Path, text: str = "HELLO WORLD 12345") -> None:
    """One page that is only a rasterised image carrying *text* (no text layer)."""
    import io

    import pymupdf
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=72)  # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    draw.text((50, 150), text, fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)  # landscape A4 in points
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()


def _build_text_pdf(path: Path) -> None:
    """One page with a real text layer (no OCR should trigger)."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Ordinary machine-readable text layer.", fontsize=12)
    doc.save(str(path))
    doc.close()


def test_scanned_page_ocr_enabled():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "scan.pdf"
        _build_scanned_pdf(pdf)

        processor = PDFProcessor()
        chunks = list(processor.extract_chunks_iter(str(pdf), ocr=True))

        ocr_chunks = [c for c in chunks if str(c.get("text", "")).startswith("[OCR page 1]")]
        assert ocr_chunks, f"expected an OCR'd chunk, got: {[c.get('text', '')[:60] for c in chunks]}"
        payload = " ".join(ocr_chunks[0]["text"].split())
        assert len(payload) > len("[OCR page 1]"), "OCR must contribute recognised text beyond the marker"


def test_scanned_page_ocr_disabled_keeps_image_only_behaviour():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "scan.pdf"
        _build_scanned_pdf(pdf)

        processor = PDFProcessor()
        chunks = list(processor.extract_chunks_iter(str(pdf), ocr=False))
        assert chunks, "image-only pages still yield image chunks"
        assert not any("[OCR page" in str(c.get("text", "")) for c in chunks)


def test_text_layer_page_never_ocrd():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "text.pdf"
        _build_text_pdf(pdf)

        processor = PDFProcessor()
        chunks = list(processor.extract_chunks_iter(str(pdf), ocr=True))
        assert chunks
        assert not any("[OCR page" in str(c.get("text", "")) for c in chunks)
        assert any("Ordinary machine-readable" in str(c.get("text", "")) for c in chunks)


def test_ocr_default_resolves_from_availability():
    """ocr=None → enabled iff tesseract is present (here: it is, per skipif)."""
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "scan.pdf"
        _build_scanned_pdf(pdf)
        chunks = list(PDFProcessor().extract_chunks_iter(str(pdf), ocr=None))
        assert any("[OCR page" in str(c.get("text", "")) for c in chunks)


if __name__ == "__main__":
    import traceback

    try:
        import pytest
    except ImportError:
        print("pytest not available; skipping marker-based skip logic")

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            if not _tesseract_available():
                print(f"  {fn.__name__} ... SKIP (no tesseract)")
                continue
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
