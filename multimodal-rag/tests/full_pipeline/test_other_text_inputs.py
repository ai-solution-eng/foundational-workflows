#!/usr/bin/env python3
"""Test the new input modality processors via DatasetManager.

Usage
-----
    # Test all processors independently (no endpoint required)
    python tests/full_pipeline/test_new_modalities.py

    # Full DatasetManager integration test (requires active embedding endpoint)
    python tests/full_pipeline/test_new_modalities.py --with-storage
"""

import argparse
import json
import os
import sys
import tempfile

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import DatasetManager, _classify_file
from multimodal_rag.input_processing import (
    EbookProcessor,
    LogProcessor,
    NotebookProcessor,
)
from multimodal_rag.utils.logging_utils import setup_logger

setup_logger(level="INFO")


# ---------------------------------------------------------------------------
# Sample data generators
# ---------------------------------------------------------------------------


def make_sample_notebook(path: str) -> str:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Introduction\n",
                    "\n",
                    "This is a **test** notebook for RAG ingestion.",
                ],
            },
            {
                "cell_type": "code",
                "source": ["import numpy as np\n", "print('hello')"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "text": ["hello\n"],
                    }
                ],
            },
            {
                "cell_type": "markdown",
                "source": ["## Results\n", "\n", "The output shows `hello`."],
            },
            {
                "cell_type": "code",
                "source": ["1 + 1"],
                "outputs": [
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": ["2"]},
                    }
                ],
            },
        ],
        "metadata": {"kernelspec": {"language": "python"}},
    }
    with open(path, "w") as f:
        json.dump(nb, f)
    return path


def make_sample_epub(path: str) -> str:
    """Create a minimal valid EPUB (ZIP of XHTML)."""
    import zipfile

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest>
    <item id="chapter1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter2" href="chap2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
  </spine>
</package>"""

    chap1 = "<html><body><h1>Chapter 1</h1><p>This is the first chapter of the test ebook.</p></body></html>"
    chap2 = (
        "<html><body><h1>Chapter 2</h1><p>This is the second chapter with more content for testing.</p></body></html>"
    )

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chap1.xhtml", chap1)
        zf.writestr("OEBPS/chap2.xhtml", chap2)
    return path


def make_sample_log(path: str) -> str:
    lines = [
        "2025-01-15 10:30:00 INFO  Starting data pipeline",
        "2025-01-15 10:30:01 DEBUG Loading configuration file",
        "2025-01-15 10:30:02 WARN  Deprecated config key 'old_param' used",
        "2025-01-15 10:30:05 ERROR Failed to connect to database: timeout",
        "2025-01-15 10:30:06 INFO  Retrying connection (attempt 1/3)",
        "2025-01-15 10:30:10 INFO  Connection established successfully",
        "2025-01-15 10:30:12 INFO  Processing batch 1 of 10",
        "2025-01-15 10:30:15 FATAL Out of memory during batch processing",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make_sample_json_log(path: str) -> str:
    entries = [
        {
            "timestamp": "2025-01-15T10:30:00Z",
            "level": "INFO",
            "message": "Service started",
            "service": "api",
        },
        {
            "timestamp": "2025-01-15T10:30:01Z",
            "level": "ERROR",
            "message": "DB connection failed",
            "db": "postgres",
        },
        {
            "timestamp": "2025-01-15T10:30:02Z",
            "level": "WARN",
            "message": "High memory usage",
            "usage_pct": 87,
        },
    ]
    with open(path, "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)
    return path


# ---------------------------------------------------------------------------
# Processor tests (always works — no endpoint needed)
# ---------------------------------------------------------------------------


def test_notebook_processor() -> None:
    print("  NotebookProcessor...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as f:
        nb_path = make_sample_notebook(f.name)

    try:
        proc = NotebookProcessor(chunk_size=200)
        docs = proc.process(nb_path)
        assert len(docs) >= 1, f"Expected at least 1 doc, got {len(docs)}"
        # Verify markdown content
        texts = " ".join(d.get("text", "") for d in docs)
        assert "Introduction" in texts, "Missing markdown content"
        assert "import numpy" in texts, "Missing code content"
        assert "hello" in texts, "Missing output content"
        print(f"OK ({len(docs)} chunks)")
    finally:
        os.unlink(nb_path)


def test_notebook_processor_text_only() -> None:
    print("  NotebookProcessor (text-only)...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as f:
        nb_path = make_sample_notebook(f.name)

    try:
        proc = NotebookProcessor(chunk_size=200, include_code=False, include_outputs=False)
        docs = proc.process(nb_path)
        texts = " ".join(d.get("text", "") for d in docs)
        assert "Introduction" in texts, "Missing markdown"
        assert "import numpy" not in texts, "Code should be excluded"
        print(f"OK ({len(docs)} chunks)")
    finally:
        os.unlink(nb_path)


def test_ebook_processor() -> None:
    print("  EbookProcessor...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        epub_path = make_sample_epub(f.name)

    try:
        proc = EbookProcessor(chunk_size=200)
        docs = proc.process(epub_path)
        assert len(docs) >= 1, f"Expected at least 1 doc, got {len(docs)}"
        texts = " ".join(d.get("text", "") for d in docs)
        assert "Chapter 1" in texts, "Missing chapter 1"
        assert "Chapter 2" in texts, "Missing chapter 2"
        print(f"OK ({len(docs)} chunks)")
    finally:
        os.unlink(epub_path)


def test_log_processor() -> None:
    print("  LogProcessor (syslog-style)...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
        log_path = make_sample_log(f.name)

    try:
        proc = LogProcessor(chunk_size=200)
        docs = proc.process(log_path)
        assert len(docs) >= 1, f"Expected at least 1 doc, got {len(docs)}"
        texts = " ".join(d.get("text", "") for d in docs)
        assert "Failed to connect" in texts, "Missing log content"
        # Check metadata
        found_severity = any("severities" in d for d in docs)
        assert found_severity, "Missing severity metadata"
        print(f"OK ({len(docs)} chunks)")
    finally:
        os.unlink(log_path)


def test_log_processor_json() -> None:
    print("  LogProcessor (JSON-lines)...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
        log_path = make_sample_json_log(f.name)

    try:
        proc = LogProcessor(chunk_size=200)
        docs = proc.process(log_path)
        assert len(docs) >= 1, f"Expected at least 1 doc, got {len(docs)}"
        texts = " ".join(d.get("text", "") for d in docs)
        assert "Service started" in texts, "Missing JSON log content"
        print(f"OK ({len(docs)} chunks)")
    finally:
        os.unlink(log_path)


def test_classify_file() -> None:
    print("  _classify_file...", end=" ")
    checks = [
        ("test.ipynb", "notebook"),
        ("test.epub", "ebook"),
        ("test.log", "log"),
        ("test.txt.log", "log"),
    ]
    for path, expected in checks:
        actual = _classify_file(path)
        assert actual == expected, f"{path}: got '{actual}', expected '{expected}'"
    print("OK")


# ---------------------------------------------------------------------------
# DatasetManager integration test (requires embedding endpoint)
# ---------------------------------------------------------------------------


def test_dataset_manager_integration(base_path: str, nb_path: str, epub_path: str, log_path: str) -> None:
    """Run files through DatasetManager.add_file().

    This exercises the full ingestion pipeline including file copying,
    processor dispatch, and vector storage.  Requires a running embedding
    endpoint (e.g. a Qwen3-VL service).
    """
    dm = DatasetManager(
        base_path=base_path,
        qdrant_host="",  # local in-memory Qdrant
    )
    dataset_name = "test_new_modalities"
    try:
        dm.create_dataset(dataset_name)

        print("\n  Ingesting notebook...", end=" ")
        result = dm.add_file(dataset_name, nb_path)
        print(f"{result['type']}, {result['chunks']} chunks")

        print("  Ingesting ebook...", end=" ")
        result = dm.add_file(dataset_name, epub_path)
        print(f"{result['type']}, {result['chunks']} chunks")

        print("  Ingesting log...", end=" ")
        result = dm.add_file(dataset_name, log_path)
        print(f"{result['type']}, {result['chunks']} chunks")

        # Search (requires working embeddings)
        print("  Searching...", end=" ")
        try:
            results = dm.search(dataset_name, "test", top_k=5)
            print(f"{len(results)} results")
        except Exception as e:
            print(f"Search skipped (no endpoint): {e}")

        dm.delete_dataset(dataset_name)
        print("  Cleanup: dataset deleted")

    except Exception as e:
        print(f"  Error: {e}")
        try:
            dm.delete_dataset(dataset_name)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Test new input modality processors")
    parser.add_argument(
        "--with-storage",
        action="store_true",
        help="Also run DatasetManager integration test (requires embedding endpoint)",
    )
    args = parser.parse_args()

    print("=== Processor unit tests ===\n")
    test_classify_file()
    test_notebook_processor()
    test_notebook_processor_text_only()
    test_ebook_processor()
    test_log_processor()
    test_log_processor_json()
    print("\nAll processor tests passed!")

    if args.with_storage:
        print("\n=== DatasetManager integration test ===")
        import shutil
        import uuid

        base_dir = f"/tmp/test_new_modalities_{uuid.uuid4().hex}"

        # Create sample files
        nb_path = os.path.join(base_dir, "sample.ipynb")
        epub_path = os.path.join(base_dir, "sample.epub")
        log_path = os.path.join(base_dir, "sample.log")
        os.makedirs(base_dir, exist_ok=True)
        make_sample_notebook(nb_path)
        make_sample_epub(epub_path)
        make_sample_log(log_path)

        try:
            test_dataset_manager_integration(base_dir, nb_path, epub_path, log_path)
            print("\nDatasetManager integration test passed!")
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)
    else:
        print("\nSkipping DatasetManager integration test (use --with-storage to enable)")


if __name__ == "__main__":
    main()
