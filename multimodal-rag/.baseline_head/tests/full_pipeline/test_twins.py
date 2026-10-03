"""Tests for the multimodal "twin" embeddings feature.

Two twin kinds are created at ingest time in ``aadd_to_vector_store``:

1. **Text-only twins** — for docs that carry real extracted text (e.g. a PDF
   page): a pure-text embedding of the same text, so text queries match even
   when the multimodal embedding is dominated by visual content.
2. **Caption twins** — for pure media docs (image/video/audio) whose text is
   only a media placeholder plus an ingest-time VLM/ASR caption: an extra
   embedding of the SAME media WITH the caption text, so caption wording is
   searchable alongside the base media-only embedding.

This file covers the pure helpers (no network) plus an offline end-to-end
ingest through the in-memory vector store with a stub embedder (no model
endpoint required).

Run::

    python tests/full_pipeline/test_twins.py          # standalone
    pytest tests/full_pipeline/test_twins.py          # under pytest
"""

import asyncio
import hashlib
import os
import sys
from typing import Any, ClassVar

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.rag_system import (
    MultimodalRAG,
    _has_caption,
    _has_real_text,
    _media_caption_twin_needed,
    _strip_embed_caption,
)
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

SUPPORTED = {"text", "image", "video"}

# A tiny valid data-URL placeholder (content is irrelevant — embedding is stubbed).
IMG = "data:image/jpeg;base64,AA=="
VID = "data:video/mp4;base64,AA=="
AUD = "data:audio/mpeg;base64,AA=="


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_has_caption():
    assert _has_caption("[Image: a.jpg]\n[Image description]: a red car") is True
    assert _has_caption("[Image: a.jpg]\n[Video audio transcription]: hello") is True
    assert _has_caption("[Image: a.jpg]\n[Audio transcription]: hello") is True
    assert _has_caption("[Image: a.jpg]") is False
    assert _has_caption("") is False


def test_has_real_text():
    # Pure media placeholder + caption → no real text
    assert _has_real_text("[Image: a.jpg]\n[Image description]: red car") is False
    # Real extracted content wins even with captions around it
    assert _has_real_text("Some real paragraph\n[Image description]: red car") is True


def test_strip_embed_caption_pure_media():
    """Pure media docs get captions stripped (base embedding keys off media)."""
    doc = {"text": "[Image: a.jpg]\n[Image description]: red car", "image": IMG}
    out = _strip_embed_caption(doc, SUPPORTED)
    assert out is not doc
    assert "red car" not in out["text"]
    assert "red car" in doc["text"]  # original untouched


def test_media_caption_twin_needed():
    """Caption twins apply only to pure, embeddable media docs with captions."""
    base = {"text": "[Image: a.jpg]\n[Image description]: red car", "image": IMG, "source": "/tmp/a.jpg"}

    # Pure image + caption + image embedder → twin wanted
    assert _media_caption_twin_needed(dict(base), SUPPORTED) is True

    # No caption → nothing to add over the base embedding
    assert _media_caption_twin_needed({"text": "[Image: a.jpg]", "image": IMG}, SUPPORTED) is False

    # Embedder does NOT support the media (e.g. audio today) → caption-only
    # text embedding is handled by the Preprocessor, no twin
    aud = {"text": "[Audio: x.mp3]\n[Audio transcription]: hello", "audio": AUD}
    assert _media_caption_twin_needed(dict(aud), SUPPORTED) is False
    # …but a future embedder that DOES support audio gets the twin
    assert _media_caption_twin_needed(dict(aud), {"text", "audio"}) is True

    # Real text (PDF page) → handled by the text-only twin, not the caption twin
    page = {"text": "A real page of text\n[Image description]: red car", "image": IMG}
    assert _media_caption_twin_needed(page, SUPPORTED) is False

    # No media → never
    assert _media_caption_twin_needed({"text": "plain text"}, SUPPORTED) is False

    # Video + caption
    vid = {"text": "[Video: game.mp4] [0s – 32s]\n[Video description]: gameplay", "video": VID}
    assert _media_caption_twin_needed(vid, SUPPORTED) is True


# ---------------------------------------------------------------------------
# Offline ingest through the vector store
# ---------------------------------------------------------------------------


class _StubEmbedderModel:
    """Minimal stand-in for EmbeddingModel with a deterministic stub embedder."""

    allowable_modalities: tuple[str, ...] = ("text", "image", "video")
    model_name = "stub-embedder"
    base_url = "http://stub/v1"
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None

    def __init__(self) -> None:
        self.model = MultiModalEmbeddings(self)
        # Record every embedding input so tests can inspect exactly what the
        # embedder received (e.g. caption stripped for the base, kept for the
        # caption twin, media dropped for the text-only twin).
        self.embed_logs: list[list[Any]] = []
        # Override the network path with a deterministic text-hash embedding.
        self.model.aembed_documents = self._aembed_documents  # type: ignore[assignment]

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:8]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        self.embed_logs.append(list(docs))
        out = []
        for d in docs:
            text = d if isinstance(d, str) else (d.get("text") or "")
            out.append(self._hash_vector(text))
        return out


def _build_rag(*, audio: bool = False) -> tuple[MultimodalRAG, _StubEmbedderModel]:
    emb = _StubEmbedderModel()
    if audio:
        emb.allowable_modalities = ("text", "image", "video", "audio")
    # Stub stands in for EmbeddingModel; the in-memory store exercises the
    # full ingest path without a live model endpoint.
    return MultimodalRAG(embedder=emb, vector_store=None), emb  # type: ignore[arg-type]


def _stored(rag: MultimodalRAG) -> list[dict]:
    """Return [{_twin, text, has_media, source}, ...] for every stored point."""
    out = []
    for entry in rag.vector_store.store.values():  # type: ignore[union-attr]
        doc = entry["document"]
        meta = dict(doc.metadata)
        meta["text"] = doc.page_content
        out.append(meta)
    return out


def test_ingest_creates_caption_twin() -> None:
    rag, emb = _build_rag()
    doc = {
        "text": "[Image: photo.jpg]\n[Image description]: a red car on a mountain road",
        "image": IMG,
        "source": "/tmp/photo.jpg",
    }
    ids = asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))

    stored = _stored(rag)
    assert len(stored) == 2, f"expected base + caption twin, got {len(stored)}: {stored}"
    twins = [s for s in stored if s.get("_twin")]
    bases = [s for s in stored if not s.get("_twin")]
    assert len(twins) == 1 and len(bases) == 1
    # The caption twin retains the media and the caption text in its payload.
    twin = twins[0]
    assert twin.get("image") == IMG, "caption twin must keep the media reference"
    assert "red car" in twin["text"]
    # Both points share the same parent identity for retrieval dedup.
    assert bases[0]["source"] == twin["source"] == "/tmp/photo.jpg"
    assert len(ids) == 2

    # Embed inputs: base = media-only (caption STRIPPED), twin = media + caption.
    base_input, twin_input = emb.embed_logs[0][0], emb.embed_logs[1][0]
    assert "red car" not in base_input["text"]
    assert "red car" in twin_input["text"]
    assert base_input["image"] == twin_input["image"] == IMG


def test_ingest_no_twin_without_caption() -> None:
    rag, _ = _build_rag()
    doc = {"text": "[Image: photo.jpg]", "image": IMG, "source": "/tmp/photo.jpg"}
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    stored = _stored(rag)
    assert len(stored) == 1, f"expected no twin, got {len(stored)}: {stored}"
    assert not stored[0].get("_twin")


def test_ingest_text_only_twin_for_real_text() -> None:
    """Real-text docs (PDF pages) keep the text-only twin behaviour."""
    rag, emb = _build_rag()
    doc = {
        "text": "Attention is all you need. The transformer uses self-attention.\n[Image description]: architecture diagram",
        "image": IMG,
        "source": "/tmp/paper.pdf",
        "page": 3,
        "chunk_index": 0,
    }
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    stored = _stored(rag)
    assert len(stored) == 2, f"expected base + text-only twin, got {len(stored)}"
    twin = next(s for s in stored if s.get("_twin"))
    # Stored payload carries the media for retrieval…
    assert twin.get("image") == IMG
    # …but the EMBEDDING input for the text-only twin must NOT include media.
    twin_input = emb.embed_logs[1][0]
    assert "image" not in twin_input
    assert "Attention is all you need" in twin_input["text"]


def test_audio_unsupported_is_skipped() -> None:
    """Audio with no ASR and an embedder that doesn't support audio → skipped."""
    rag, _ = _build_rag()
    doc = {
        "text": "[Audio: x.mp3]\n[Audio transcription]: welcome to the podcast",
        "audio": AUD,
        "source": "/tmp/x.mp3",
    }
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    assert _stored(rag) == [], "expected the audio doc to be skipped entirely"


def test_audio_caption_twin_when_embedder_supports_audio() -> None:
    """A future audio-capable embedder gets base + caption-twin embeddings."""
    rag, emb = _build_rag(audio=True)
    doc = {
        "text": "[Audio: x.mp3]\n[Audio transcription]: welcome to the podcast",
        "audio": AUD,
        "source": "/tmp/x.mp3",
    }
    asyncio.run(rag.aadd_to_vector_store([doc], deduplicate=False))
    stored = _stored(rag)
    assert len(stored) == 2, f"expected base + caption twin, got {len(stored)}: {stored}"
    twins = [s for s in stored if s.get("_twin")]
    assert len(twins) == 1
    # Base embed strips the caption; twin keeps it.
    base_input, twin_input = emb.embed_logs[0][0], emb.embed_logs[1][0]
    assert "welcome to the podcast" not in base_input["text"]
    assert "welcome to the podcast" in twin_input["text"]


# ---------------------------------------------------------------------------
# Retrieval dedup of twins
# ---------------------------------------------------------------------------


def test_dedup_twins_collapses_parent_and_caption_twin() -> None:
    parent = {"text": "[Image: photo.jpg]", "image": IMG, "source": "/tmp/photo.jpg"}
    caption_twin = {
        "text": "[Image: photo.jpg]\n[Image description]: red car",
        "image": IMG,
        "source": "/tmp/photo.jpg",
        "_twin": True,
    }
    kept = MultimodalRAG._dedup_twins([(parent, 0.8), (caption_twin, 0.95)])
    # Parent and caption twin share (source, page=None, chunk=None) → twin dropped.
    assert len(kept) == 1
    assert kept[0][0] is parent


def test_dedup_twins_keeps_orphan_twin() -> None:
    # When ONLY the caption twin matches, it is kept (media still retrievable).
    twin = {
        "text": "[Image: photo.jpg]\n[Image description]: red car",
        "image": IMG,
        "source": "/tmp/photo.jpg",
        "_twin": True,
    }
    kept = MultimodalRAG._dedup_twins([(dict(twin), 0.9)])
    assert len(kept) == 1
    assert kept[0][0]["_twin"] is True


def test_dedup_twins_keeps_twin_across_video_segments() -> None:
    """Video segments share (source, None, None) — the time window must keep
    a twin of one segment from collapsing against a parent of another."""
    parent_seg0 = {
        "text": "[Video: game.mp4] [0s – 32s]",
        "video": VID,
        "source": "/tmp/game.mp4",
        "timestamp_start": 0.0,
        "timestamp_end": 32.0,
    }
    twin_seg1 = {
        "text": "[Video: game.mp4] [32s – 64s]\n[Video description]: level two gameplay",
        "video": VID,
        "source": "/tmp/game.mp4",
        "timestamp_start": 32.0,
        "timestamp_end": 64.0,
        "_twin": True,
    }
    kept = MultimodalRAG._dedup_twins([(parent_seg0, 0.8), (twin_seg1, 0.95)])
    assert len(kept) == 2, "segments of the same video must not collapse across time windows"
    # A twin and its OWN parent (same segment) still collapse.
    kept2 = MultimodalRAG._dedup_twins(
        [(dict(parent_seg0), 0.8), (dict(twin_seg1, timestamp_start=0.0, timestamp_end=32.0), 0.95)]
    )
    assert len(kept2) == 1


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
