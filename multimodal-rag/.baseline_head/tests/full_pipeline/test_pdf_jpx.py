"""Offline tests for browser-safe normalisation of PDF-extracted images.

PDFs may embed images in codecs that browsers and Open WebUI cannot render —
most commonly JPEG 2000 (the JPXDecode filter, extracted by PyMuPDF as raw
``.jpx`` streams).  Before the fix the processor emitted
``data:image/jpx;base64,...`` data URLs and persisted ``.jpx`` files, neither
of which render in a browser.  These tests pin the fix: every data URL that
leaves :class:`PDFProcessor` for a JPX/TIFF-embedded PDF must be JPEG or PNG.

The JPEG 2000 cases require a Pillow build with OpenJPEG support (skipped
otherwise; the TIFF cases exercise the same code path without it).

Run::

    python tests/full_pipeline/test_pdf_jpx.py    # standalone
    pytest tests/full_pipeline/test_pdf_jpx.py    # under pytest
"""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from PIL import Image, features

from multimodal_rag.input_processing.pdf_processor import PDFProcessor

_JPEG2000_OK = features.check("jpg_2000")

_JPX_SKIP = pytest.mark.skipif(not _JPEG2000_OK, reason="Pillow built without OpenJPEG (no JPEG 2000 support)")


def _save_jpx(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG2000")
    return buf.getvalue()


def _build_jpx_pdf(path: Path, jpx_bytes: bytes) -> None:
    """One page whose only content is a JPEG 2000 (JPXDecode) image."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=jpx_bytes)
    doc.save(str(path))
    doc.close()


def _build_tiff_pdf(path: Path) -> None:
    """One page whose only content is a TIFF image."""
    import pymupdf

    img = Image.new("RGB", (200, 150), (200, 30, 90))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()


def _decode(data_url: str) -> Image.Image:
    raw = io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))
    return Image.open(raw)


@_JPX_SKIP
def test_jpx_pdf_yields_jpeg_data_urls():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "jpx.pdf"
        _build_jpx_pdf(pdf, _save_jpx(Image.new("RGB", (200, 150), (30, 120, 200))))

        processor = PDFProcessor()
        pages = processor.extract_pages(str(pdf))
        urls = [im["data_url"] for p in pages for im in p["images"]]
        assert urls, "expected the JPX image to be extracted"
        for url in urls:
            assert url.startswith("data:image/jpeg;"), f"non-browser-safe data URL emitted: {url[:40]}"
            assert _decode(url).size == (200, 150)

        blocks = processor.extract_text_blocks(str(pdf))
        block_urls = [b["image_data_url"] for b in blocks if b["block_type"] == "image"]
        assert block_urls, "expected an image block"
        assert all(u.startswith("data:image/jpeg;") for u in block_urls), "non-browser-safe data URL in blocks"


@_JPX_SKIP
def test_jpx_chunks_never_emit_jpx():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "jpx.pdf"
        _build_jpx_pdf(pdf, _save_jpx(Image.new("RGB", (200, 150), (10, 200, 80))))

        chunks = list(PDFProcessor().extract_chunks_iter(str(pdf)))
        assert chunks
        for ch in chunks:
            for url in ch.get("image", []) if isinstance(ch.get("image"), list) else [ch.get("image", "")]:
                if url:
                    assert not url.startswith("data:image/jpx"), "image/jpx data URL leaked into a chunk"


def test_tiff_pdf_yields_browser_safe_data_urls():
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "tiff.pdf"
        _build_tiff_pdf(pdf)

        pages = PDFProcessor().extract_pages(str(pdf))
        urls = [im["data_url"] for p in pages for im in p["images"]]
        assert urls, "expected the TIFF image to be extracted"
        assert all(u.startswith(("data:image/jpeg;", "data:image/png;")) for u in urls)


@_JPX_SKIP
def test_normalize_jpx_converts_to_jpeg():
    raw = _save_jpx(Image.new("RGB", (64, 48), (255, 128, 0)))
    out, ext = PDFProcessor._normalize_to_browser_safe(raw, "jpx")
    assert ext == "jpg"
    assert Image.open(io.BytesIO(out)).size == (64, 48)


@_JPX_SKIP
def test_normalize_transparent_jpx_converts_to_png():
    raw = _save_jpx(Image.new("RGBA", (64, 48), (0, 0, 255, 128)))
    out, ext = PDFProcessor._normalize_to_browser_safe(raw, "jpx")
    assert ext == "png"
    img = Image.open(io.BytesIO(out))
    assert img.mode == "RGBA"
    assert img.size == (64, 48)


def test_normalize_passthrough_safe_formats():
    img = Image.new("RGB", (8, 8), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    assert PDFProcessor._normalize_to_browser_safe(png_bytes, "png") == (png_bytes, "png")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()
    assert PDFProcessor._normalize_to_browser_safe(jpg_bytes, "jpg") == (jpg_bytes, "jpg")


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
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
