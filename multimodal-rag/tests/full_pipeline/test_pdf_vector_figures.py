"""Regression tests for vector-figure capture (PDF_VECTOR_FIGURES).

Scientific PDFs frequently draw figures as vector paths (TikZ, draw.io,
matplotlib exports) instead of embedded rasters — the Mamba-2 paper's
Figure 6 ("Sequential / Parallel Mamba Block") is pure vector.  Such figures
produce no ``get_images()`` XObjects: the raster extraction path never sees
them, their tiny label text leaked into chunks as garbled spans, and
PyMuPDF's table detector misread the figure's boxes as ruled tables.

The feature detects dense vector-drawing clusters, renders each cluster's
page region, and injects it as an ordinary image region — the figure then
flows through the standard proximity-attach / joint-embedding / text-only-
twin path exactly like an embedded raster figure.

Run::

    python tests/full_pipeline/test_pdf_vector_figures.py    # standalone
    pytest tests/full_pipeline/test_pdf_vector_figures.py    # under pytest
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


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (140, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _box_diagram(page: pymupdf.Page, origin: tuple[float, float]) -> None:
    """Draw a realistic vector 'figure': two block stacks joined by arrows.

    Sized like real paper diagrams (~25% of a letter page, 10+ paths) so it
    clears the min-paths / min-area gates the way actual TikZ figures do.
    """
    x0, y0 = origin
    boxes = [
        pymupdf.Rect(x0, y0, x0 + 110, y0 + 36),  # input proj
        pymupdf.Rect(x0 + 25, y0 + 60, x0 + 135, y0 + 96),  # SSM
        pymupdf.Rect(x0 + 25, y0 + 120, x0 + 135, y0 + 156),  # conv
        pymupdf.Rect(x0 + 220, y0, x0 + 330, y0 + 36),
        pymupdf.Rect(x0 + 245, y0 + 60, x0 + 355, y0 + 96),
        pymupdf.Rect(x0 + 245, y0 + 120, x0 + 355, y0 + 156),
        pymupdf.Rect(x0 + 420, y0 + 30, x0 + 490, y0 + 60),  # legend swatch
        pymupdf.Rect(x0 + 420, y0 + 80, x0 + 490, y0 + 110),
    ]
    for r in boxes:
        page.draw_rect(r, color=(0, 0, 0), fill=(0.85, 0.92, 0.85), width=1.2)
    # connectors (arrows between stacked blocks, cross-links, legend rules)
    connectors = [
        ((x0 + 55, y0 + 36), (x0 + 55, y0 + 60)),
        ((x0 + 80, y0 + 96), (x0 + 80, y0 + 120)),
        ((x0 + 275, y0 + 36), (x0 + 275, y0 + 60)),
        ((x0 + 300, y0 + 96), (x0 + 300, y0 + 120)),
        ((x0 + 135, y0 + 78), (x0 + 245, y0 + 78)),
        ((x0 + 490, y0 + 45), (x0 + 520, y0 + 45)),
        ((x0 + 490, y0 + 95), (x0 + 520, y0 + 95)),
        ((x0 + 490, y0 + 145), (x0 + 520, y0 + 145)),
    ]
    for (ax, ay), (bx, by) in connectors:
        page.draw_line(pymupdf.Point(ax, ay), pymupdf.Point(bx, by), width=1.2)
    # tiny figure labels (the garbled-text failure mode when leaked raw)
    page.insert_text(pymupdf.Point(x0 + 50, y0 + 84), "SSM", fontsize=8)
    page.insert_text(pymupdf.Point(x0 + 55, y0 + 144), "Conv", fontsize=8)
    page.insert_text(pymupdf.Point(x0 + 270, y0 + 84), "SSM", fontsize=8)
    page.insert_text(pymupdf.Point(x0 + 275, y0 + 144), "Conv", fontsize=8)
    page.insert_text(pymupdf.Point(x0 + 524, y0 + 48), "Linear projection", fontsize=8)
    page.insert_text(pymupdf.Point(x0 + 524, y0 + 98), "Sequence transformation", fontsize=8)


def _dense_diagram(page: pymupdf.Page, origin: tuple[float, float]) -> None:
    """A caption-less, stroke-dense diagram (≥15 non-hairline paths).

    Mimics a real TikZ architecture figure: many filled boxes and
    connectors covering a large enough region for the density gate.
    """
    x0, y0 = origin
    n_cols, n_rows = 4, 4
    for c in range(n_cols):
        for r in range(n_rows):
            bx = x0 + c * 60
            by = y0 + r * 44
            page.draw_rect(pymupdf.Rect(bx, by, bx + 44, by + 30), color=(0, 0, 0), fill=(0.85, 0.92, 0.85), width=1.1)
            if c < n_cols - 1:
                page.draw_line(pymupdf.Point(bx + 44, by + 15), pymupdf.Point(bx + 60, by + 15), width=1.1)
            if r < n_rows - 1:
                page.draw_line(pymupdf.Point(bx + 22, by + 30), pymupdf.Point(bx + 22, by + 44), width=1.1)
    page.insert_text(pymupdf.Point(x0 + 8, y0 + 20), "SS", fontsize=7)
    page.insert_text(pymupdf.Point(x0 + 68, y0 + 20), "SS", fontsize=7)


def _build_pdf(path: Path, *, with_caption: bool = True, dense: bool = False) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    if dense:
        _dense_diagram(page, (60, 60))
    else:
        _box_diagram(page, (60, 60))
    if with_caption:
        page.insert_text(pymupdf.Point(60, 230), "Figure 6: (Architecture.) A toy block diagram.", fontsize=10)
    # body text well away from the figure
    page.insert_text(pymupdf.Point(60, 300), "Body text paragraph one. " * 12, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_captioned_vector_figure_is_captured():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "vec.pdf"
        _build_pdf(pdf, with_caption=True)

        blocks = PDFProcessor().extract_text_blocks(str(pdf))
        image_blocks = [b for b in blocks if b["block_type"] == "image"]
        assert image_blocks, "vector figure was not captured"
        # The captured region is the diagram cluster, not the whole page.
        bboxes = [b["bbox"] for b in image_blocks]
        assert all(bb[3] - bb[1] < 0.6 * 792 for bb in bboxes), f"captured region suspiciously tall: {bboxes}"

        docs = list(PDFProcessor().extract_chunks_iter(str(pdf), chunk_size=8192, chunk_overlap=64, ocr=False))
        with_img = [d for d in docs if d.get("image")]
        assert with_img, "no chunk carries the vector figure"
        # Real-text chunk jointly embedding the figure (not a bare [Image on page N] doc)
        joint = [d for d in with_img if d.get("text") and not str(d["text"]).startswith("[Image on page")]
        assert joint, "figure not attached to a real-text chunk"


def test_uncaptioned_dense_diagram_is_captured():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "vec_nocap.pdf"
        _build_pdf(pdf, with_caption=False, dense=True)

        docs = list(PDFProcessor().extract_chunks_iter(str(pdf), chunk_size=8192, chunk_overlap=64, ocr=False))
        assert any(d.get("image") for d in docs), "dense uncaptioned diagram not captured"


def test_feature_disabled_by_env():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "vec_off.pdf"
        _build_pdf(pdf, with_caption=True)

        os.environ["PDF_VECTOR_FIGURES"] = "0"
        try:
            docs = list(PDFProcessor().extract_chunks_iter(str(pdf), chunk_size=8192, chunk_overlap=64, ocr=False))
        finally:
            os.environ.pop("PDF_VECTOR_FIGURES", None)
        assert not any(d.get("image") for d in docs), "capture ran despite PDF_VECTOR_FIGURES=0"


def test_plain_table_is_not_captured_as_figure():
    with __import__("tempfile").TemporaryDirectory() as td:
        pdf = Path(td) / "table.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        # Full ruled grid (rows x 2 columns) — what find_tables needs to see.
        rows = [
            ("Model", "PPL"),
            ("Mamba", "11.49"),
            ("Mamba-2", "11.49"),
            ("Baseline", "11.76"),
            ("Transformer", "12.10"),
        ]
        x_l, x_m, x_r = 60.0, 280.0, 400.0
        y0, row_h = 60.0, 20.0
        for i in range(len(rows) + 1):
            y = y0 + i * row_h
            page.draw_line(pymupdf.Point(x_l, y), pymupdf.Point(x_r, y), width=0.7)
        page.draw_line(pymupdf.Point(x_l, y0), pymupdf.Point(x_l, y0 + len(rows) * row_h), width=0.7)
        page.draw_line(pymupdf.Point(x_m, y0), pymupdf.Point(x_m, y0 + len(rows) * row_h), width=0.7)
        page.draw_line(pymupdf.Point(x_r, y0), pymupdf.Point(x_r, y0 + len(rows) * row_h), width=0.7)
        for i, (model, ppl) in enumerate(rows):
            page.insert_text(pymupdf.Point(x_l + 10, y0 + i * row_h + 14), model, fontsize=9)
            page.insert_text(pymupdf.Point(x_m + 20, y0 + i * row_h + 14), ppl, fontsize=9)
        page.insert_text(pymupdf.Point(60, y0 + len(rows) * row_h + 30), "Table 1: results.", fontsize=10)
        doc.save(str(pdf))
        doc.close()

        docs = list(PDFProcessor().extract_chunks_iter(str(pdf), chunk_size=8192, chunk_overlap=64, ocr=False))
        assert not any(d.get("image") for d in docs), "ruled table wrongly captured as a figure"
        # The table still comes through as markdown text
        assert any("|" in d.get("text", "") for d in docs), "table markdown lost"


def test_real_mamba2_figure6_chunk_carries_image_and_no_table_junk():
    pdf = Path(__file__).resolve().parents[2] / "uploads_for_test_dataset" / "AndrewUploads" / "Mamba2.pdf"
    if not pdf.exists():
        print("  skip: Mamba2.pdf not present")
        return
    docs = list(PDFProcessor().extract_chunks_iter(str(pdf), chunk_size=8192, chunk_overlap=512, ocr=False))
    p23 = [d for d in docs if d.get("page") == 23]
    assert p23, "page 23 produced no chunks"
    figure_chunk = next((d for d in p23 if d.get("image")), None)
    assert figure_chunk, "Figure 6 (vector) not captured on page 23"
    text = figure_chunk.get("text", "")
    assert "|Y" not in text and "SS<br>" not in text, "garbled figure-label table markdown leaked into chunk"
    assert "Mamba" in text or "SSM" in text, "chunk text lost its semantic content"


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
