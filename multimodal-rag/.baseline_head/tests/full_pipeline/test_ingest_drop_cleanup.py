"""Tests for the unified "skip entirely" drop rule and on-disk cleanup.

Unified rule (all media types): if a media modality is not supported by the
embedder AND no VLM/ASR can convert it to caption text, the sample is dropped —
the doc is omitted (or, if the doc has other usable content, just that media is
removed).  When every doc a file produced is dropped, the stored copy on disk
(and its ``*_preprocessed`` sibling) is deleted so it never takes up space.

Run::

    python tests/full_pipeline/test_ingest_drop_cleanup.py   # standalone
    pytest tests/full_pipeline/test_ingest_drop_cleanup.py   # under pytest
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.rag_system import MultimodalRAG, _has_embeddable_content
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

IMG = "data:image/jpeg;base64,AA=="
VID = "data:video/mp4;base64,AA=="
AUD = "data:audio/mpeg;base64,AA=="


# ---------------------------------------------------------------------------
# Stub embedders (no network)
# ---------------------------------------------------------------------------


class _StubModel:
    model_name = "stub"
    base_url = "http://stub/v1"
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None

    def __init__(self) -> None:
        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[assignment]

    async def _aembed_documents(self, docs):  # type: ignore[no-untyped-def]
        return [self._hash_vector(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:8]]


class _TextOnlyEmbedder(_StubModel):
    """An embedder that only supports text (image/video/audio all unconvertible)."""

    allowable_modalities: tuple[str, ...] = ("text",)


class _DefaultEmbedder(_StubModel):
    """The production embedder shape: text+image+video, audio unsupported."""

    allowable_modalities: tuple[str, ...] = ("text", "image", "video")


def _build_rag(embedder) -> MultimodalRAG:
    return MultimodalRAG(embedder=embedder, vector_store=None)  # in-memory store


def _stored_count(rag: MultimodalRAG) -> int:
    return len(rag.vector_store.store)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Unified drop — Preprocessor path (preprocess=True, default)
# ---------------------------------------------------------------------------


def test_pure_image_dropped_with_text_only_embedder():
    """Image not embeddable + no VLM → whole sample dropped."""
    rag = _build_rag(_TextOnlyEmbedder())
    doc = {"text": "[Image: photo.jpg]", "image": IMG, "source": "/tmp/photo.jpg"}
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    assert _stored_count(rag) == 0


def test_pure_video_dropped_with_text_only_embedder():
    """Video not embeddable + no VLM → whole sample dropped (no placeholder doc)."""
    rag = _build_rag(_TextOnlyEmbedder())
    doc = {"text": "[Video: game.mp4] [0s – 32s]", "video": VID, "source": "/tmp/game.mp4"}
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    assert _stored_count(rag) == 0


def test_pure_audio_dropped_no_asr():
    """Audio not embeddable + no ASR → whole sample dropped (existing rule)."""
    rag = _build_rag(_DefaultEmbedder())
    doc = {"text": "[Audio: x.mp3]", "audio": AUD, "source": "/tmp/x.mp3"}
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    assert _stored_count(rag) == 0


def test_mixed_doc_keeps_text_drops_unconvertible_media():
    """Real text + unconvertible image → doc kept (text), image removed."""
    rag = _build_rag(_TextOnlyEmbedder())
    doc = {
        "text": "A real paragraph about the project.\n[Image: photo.jpg]",
        "image": IMG,
        "source": "/tmp/photo.jpg",
    }
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    assert _stored_count(rag) == 1
    # The stored doc must NOT retain the dead image payload.
    entry = next(iter(rag.vector_store.store.values()))  # type: ignore[union-attr]
    stored_doc = entry["document"]
    assert "image" not in stored_doc.metadata
    assert "A real paragraph" in stored_doc.page_content


def test_embeddable_media_kept():
    """A supported image is never dropped (default stack)."""
    rag = _build_rag(_DefaultEmbedder())
    doc = {"text": "[Image: photo.jpg]\n[Image description]: a red car", "image": IMG, "source": "/tmp/photo.jpg"}
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    # base + caption twin
    assert _stored_count(rag) == 2


def test_has_embeddable_content():
    assert _has_embeddable_content({"text": "[Image: x.jpg]", "image": IMG}, {"text", "image"}) is True
    assert _has_embeddable_content({"text": "[Image: x.jpg]"}, {"text"}) is False
    assert _has_embeddable_content({"text": "[Audio: x.mp3]\n[Audio transcription]: hi"}, {"text"}) is True
    assert _has_embeddable_content({"text": "Real content"}, {"text"}) is True


# ---------------------------------------------------------------------------
# On-disk cleanup of fully-dropped files
# ---------------------------------------------------------------------------


def _make_manager(tmpdir: Path) -> DatasetManager:
    dm = DatasetManager.__new__(DatasetManager)
    dm.base_path = tmpdir
    dm.datasets_path = tmpdir / "datasets"
    dm.datasets_path.mkdir(parents=True, exist_ok=True)
    (dm.datasets_path / "ds").mkdir(parents=True, exist_ok=True)
    (dm.datasets_path / "ds" / "files").mkdir(parents=True, exist_ok=True)
    return dm


def _put_hash(dm: DatasetManager, ds: str, content_hash: str, path: str) -> None:
    hashes = {content_hash: path}
    (dm.datasets_path / ds / "files" / ".hashes.json").write_text(json.dumps(hashes), encoding="utf-8")


def test_delete_unreferenced_file_removes_copy_and_hash():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        stored = files / "abc123_song.mp3"
        stored.write_bytes(b"fake audio")
        _put_hash(dm, "ds", "deadbeef", str(stored))
        rag = _build_rag(_DefaultEmbedder())  # empty in-memory store
        dm._delete_unreferenced_file("ds", str(stored), "deadbeef", rag)
        assert not stored.exists(), "unreferenced stored file should be deleted"
        idx = json.loads((files / ".hashes.json").read_text())
        assert "deadbeef" not in idx, "hash entry should be removed"


def test_delete_unreferenced_file_removes_preprocessed_sibling():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        original = files / "abc123_vid.mp4"
        pre = files / "abc123_vid_preprocessed.mp4"
        original.write_bytes(b"x")
        pre.write_bytes(b"y")
        _put_hash(dm, "ds", "cafe", str(original))
        rag = _build_rag(_TextOnlyEmbedder())
        dm._delete_unreferenced_file("ds", str(original), "cafe", rag)
        assert not original.exists()
        assert not pre.exists(), "preprocessed sibling must also be deleted"


def test_delete_keeps_file_when_referenced():
    with tempfile.TemporaryDirectory() as td:
        dm = _make_manager(Path(td))
        files = dm.datasets_path / "ds" / "files"
        stored = files / "photo.jpg"
        stored.write_bytes(b"image")
        rag = _build_rag(_DefaultEmbedder())
        # A stored doc references this file as its source (supported image).
        asyncio.run(
            rag.aadd_to_vector_store(
                [{"text": "[Image: photo.jpg]", "image": IMG, "source": str(stored)}], deduplicate=False
            )
        )
        dm._delete_unreferenced_file("ds", str(stored), None, rag)
        assert stored.exists(), "file referenced by a stored doc must be kept"


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
