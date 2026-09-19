"""Shared pytest fixtures for the offline test suite.

Everything here runs OFFLINE: no Qdrant server, no model endpoints.  The
house pattern (see ``tests/full_pipeline/test_hybrid_search.py``) is a
deterministic hash "embedder" wrapped in ``MultiModalEmbeddings`` plus
embedded local Qdrant collections.

Fixtures
--------
``base_path``   — a temp directory for a DatasetManager root.  The process
                  cwd is temporarily moved into it so the local-Qdrant
                  default ``./qdrant_storage`` lands inside the temp dir
                  instead of the repo tree.
``nb_path`` / ``epub_path`` / ``log_path`` — sample files for the
                  DatasetManager integration test
                  (``test_other_text_inputs.py``).
``embedder``    — the stub embedder that ``DatasetManager`` requires
                  (models are never built from the environment inside
                  tests; ``_verify_endpoint`` is neutralised because there
                  is no endpoint to verify against).

These fixtures only exist so ``pytest tests/`` can collect every test
module: ``test_other_text_inputs.py`` doubles as a standalone script whose
integration test previously expected them from a conftest that did not
exist (suite error: ``fixture 'base_path' not found``).
"""

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import pytest

# Ensure the source package shadows any installed version (same insertion
# every test module already performs for itself).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_DIM = 8


# ---------------------------------------------------------------------------
# Sample-file builders (self-contained — no imports from the test modules)
# ---------------------------------------------------------------------------


def _make_sample_notebook(path: str) -> str:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Introduction\n", "\n", "This is a **test** notebook for RAG ingestion."],
            },
            {
                "cell_type": "code",
                "source": ["import numpy as np\n", "print('hello')"],
                "outputs": [{"output_type": "stream", "text": ["hello\n"]}],
            },
            {"cell_type": "markdown", "source": ["## Results\n", "\n", "The output shows `hello`."]},
            {
                "cell_type": "code",
                "source": ["1 + 1"],
                "outputs": [{"output_type": "execute_result", "data": {"text/plain": ["2"]}}],
            },
        ],
        "metadata": {"kernelspec": {"language": "python"}},
    }
    with open(path, "w") as f:
        json.dump(nb, f)
    return path


def _make_sample_epub(path: str) -> str:
    """Create a minimal valid EPUB (ZIP of XHTML)."""
    container_xml = (
        '<?xml version="1.0"?>\n'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        "  <rootfiles>\n"
        '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
        "  </rootfiles>\n"
        "</container>"
    )
    content_opf = (
        '<?xml version="1.0"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">\n'
        "  <metadata>\n"
        '    <dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test Book</dc:title>\n'
        '    <dc:identifier xmlns:dc="http://purl.org/dc/elements/1.1/" id="uid">test-book</dc:identifier>\n'
        "  </metadata>\n"
        "  <manifest>\n"
        '    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>\n'
        "  </manifest>\n"
        "  <spine>\n"
        '    <itemref idref="ch1"/>\n'
        "  </spine>\n"
        "</package>"
    )
    ch1 = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        "  <head><title>Chapter 1</title></head>\n"
        "  <body>\n"
        "    <h1>Chapter 1</h1>\n"
        "    <p>This is a test ebook about retrieval augmented generation.</p>\n"
        "  </body>\n"
        "</html>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/ch1.xhtml", ch1)
    return path


def _make_sample_log(path: str) -> str:
    lines = [
        "2025-01-15 10:30:00 INFO  Starting data pipeline",
        "2025-01-15 10:30:01 DEBUG Loading configuration file",
        "2025-01-15 10:30:02 WARN  Deprecated config key 'old_param' used",
        "2025-01-15 10:30:05 ERROR Failed to connect to database: timeout",
        "2025-01-15 10:30:06 INFO  Retrying connection (attempt 1/3)",
        "2025-01-15 10:30:10 INFO  Connection established successfully",
        "2025-01-15 10:30:12 INFO  Processing batch 1 of 10",
        "2025-01-15 10:30:15 INFO  Batch complete",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_path(tmp_path, monkeypatch) -> str:
    """DatasetManager root in a temp dir.

    The cwd is moved into it so the local-Qdrant default path
    (``./qdrant_storage``, relative to the cwd) is created inside the temp
    directory instead of the repo tree, and each test gets an isolated
    store.
    """
    monkeypatch.chdir(tmp_path)
    return str(tmp_path)


@pytest.fixture
def nb_path(base_path: str) -> str:
    return _make_sample_notebook(str(Path(base_path) / "sample.ipynb"))


@pytest.fixture
def epub_path(base_path: str) -> str:
    return _make_sample_epub(str(Path(base_path) / "sample.epub"))


@pytest.fixture
def log_path(base_path: str) -> str:
    return _make_sample_log(str(Path(base_path) / "sample.log"))


class _StubEmbedder:
    """Minimal stand-in for the EmbeddingModel DatasetManager expects.

    Deterministic hash "embedding" (no endpoint, no lexical signal).  The
    ``MultiModalEmbeddings`` wrapper plus the chunker attributes mirror the
    real model class surface used by the ingest pipeline
    (``rag.embedder.chunk_size`` / ``text_splitter`` / ``code_chunk_*`` /
    ``mm_processor_kwargs`` / ``.model.embed_query`` for the fingerprint).
    """

    allowable_modalities: tuple[str, ...] = ("text", "image", "video")
    model_name = "stub-embedder"
    base_url = "http://stub/v1"
    url_remote = "http://stub"
    api_key = ""
    embedding_dim = None
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None
    code_chunk_size = 1200
    code_chunk_overlap = 0
    code_text_splitter = None

    def __init__(self) -> None:
        from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[assignment]
        self.model.aembed_query = self._aembed_query  # type: ignore[assignment]
        self.model.embed_query = self._embed_query  # type: ignore[assignment]

    def remote(self) -> None:
        """No-op stand-in for the real model's remote-mode switch."""

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:_DIM]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        return [self._hash_vector(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    async def _aembed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._hash_vector(text)

    def _embed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._hash_vector(text)


@pytest.fixture
def embedder(monkeypatch) -> _StubEmbedder:
    """Offline stub embedder for DatasetManager construction.

    ``DatasetManager._verify_endpoint`` performs a live ``/v1/models`` GET —
    neutralised here (there is no endpoint; the constructor's embedder
    requirement is still fully exercised).
    """
    from multimodal_rag.dataset_manager import DatasetManager

    monkeypatch.setattr(DatasetManager, "_verify_endpoint", staticmethod(lambda model, role: None), raising=True)
    return _StubEmbedder()
