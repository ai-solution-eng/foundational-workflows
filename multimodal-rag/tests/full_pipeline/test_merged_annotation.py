"""Tests for merged caption+context annotation (feature: merged annotation).

When a doc carries real text AND image media AND the VLM accepts images,
ONE VLM call produces both the [Image description] and [Document context]
artifacts, instead of two calls that ask the same model about the same chunk
(the second reading the first output as text, never seeing the pixels).

Every unmet condition or failure must fall back to the two-call path.
"""

import asyncio
import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.rag_system import (
    _MERGED_CONTEXT_KEY,
    Preprocessor,
    _vlm_supports_images,
)
from multimodal_rag.utils import contextualizer as C

IMG = "data:image/png;base64,AAA"


class _Resp:
    def __init__(self, content):
        self.choices = [type("Ch", (), {"message": type("M", (), {"content": content})()})()]


class _VLM:
    """Records (kind, has_image) per call; merged replies are configurable."""

    def __init__(self, modalities=("text", "image", "video"), merged=None, raise_merged=False):
        self.allowable_modalities = modalities
        self.calls = []
        self.merged = merged
        self.raise_merged = raise_merged

    async def llm_async_chat_function_call(self, messages, **kwargs):
        sysmsg = messages[0]["content"]
        user = messages[1]["content"]
        has_img = isinstance(user, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in user)
        is_merged = "CAPTION:" in sysmsg
        self.calls.append(("merged" if is_merged else "single") + ("+img" if has_img else "+txt"))
        if is_merged:
            if self.raise_merged:
                raise RuntimeError("activator request timeout")
            if self.merged is not None:
                return _Resp(self.merged)
            return _Resp("CAPTION: A benchmark bar chart.\nCONTEXT: Sits in the results section of the report.")
        return _Resp("A plain figure description.")


@contextmanager
def _merged_on():
    """Merged annotation is OFF by default; these tests opt in explicitly."""
    old = os.environ.get("RAG_MERGED_ANNOTATION")
    os.environ["RAG_MERGED_ANNOTATION"] = "true"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RAG_MERGED_ANNOTATION", None)
        else:
            os.environ["RAG_MERGED_ANNOTATION"] = old


def _run(doc, vlm, ctx=True, modalities=("text", "image", "video")):
    async def go():
        pre = Preprocessor(vlm=vlm, caption_with_vlm=True, chunk_size=8, contextualize=ctx)
        out = pre([dict(doc)], set(modalities))
        return await C.acontextualize_docs(out, vlm)

    with _merged_on():
        return asyncio.run(go())


TEXT_IMG = {"text": "GLM-5 reaches 92.1 on the suite.", "image": IMG, "source": "/d/GLM-5.pdf", "page": 3}


def test_merged_single_call_produces_both_artifacts():
    vlm = _VLM()
    res = _run(TEXT_IMG, vlm)
    assert len(vlm.calls) == 1, vlm.calls
    assert vlm.calls[0] == "merged+img"
    text = res[0]["text"]
    assert "[Image description]:" in text
    assert "[Document context]:" in text
    assert res[0][C.CONTEXTUALIZED_META_KEY] is True


def test_merged_stash_does_not_leak_into_payload():
    res = _run(TEXT_IMG, _VLM())
    assert _MERGED_CONTEXT_KEY not in res[0]


def test_context_off_keeps_two_call_path():
    vlm = _VLM()
    _run(TEXT_IMG, vlm, ctx=False)
    assert not any(c.startswith("merged") for c in vlm.calls), vlm.calls
    assert len(vlm.calls) == 2


def test_text_only_vlm_falls_back():
    vlm = _VLM(modalities=("text",))
    res = _run(TEXT_IMG, vlm)
    assert not any(c.startswith("merged") for c in vlm.calls), vlm.calls
    assert "[Document context]:" in res[0]["text"]
    assert "[Image description]:" in res[0]["text"]


def test_unparseable_merged_falls_back():
    vlm = _VLM(merged="sorry, I cannot do that")
    res = _run(TEXT_IMG, vlm)
    assert vlm.calls[0] == "merged+img"
    assert len(vlm.calls) == 3, vlm.calls
    assert "[Image description]:" in res[0]["text"]
    assert "[Document context]:" in res[0]["text"]


def test_merged_transport_error_falls_back():
    vlm = _VLM(raise_merged=True)
    res = _run(TEXT_IMG, vlm)
    assert len(vlm.calls) == 3, vlm.calls
    assert "[Document context]:" in res[0]["text"]


def test_no_image_never_merges():
    vlm = _VLM()
    doc = {"text": "Plain text page.", "source": "/d/x.pdf"}
    _run(doc, vlm)
    assert not any(c.startswith("merged") for c in vlm.calls), vlm.calls


def test_vlm_supports_images_reads_modalities():
    assert _vlm_supports_images(_VLM()) is True
    assert _vlm_supports_images(_VLM(modalities=("text",))) is False
    assert _vlm_supports_images(None) is False


def test_parse_merged_shapes():
    assert C._parse_merged("CAPTION: a cat\nCONTEXT: it is in a doc") == ("a cat", "it is in a doc")
    assert C._parse_merged("garbage") is None
    assert C._parse_merged("CAPTION: a cat") is None
    cap, ctx = C._parse_merged("CAPTION: NONE\nCONTEXT: context here")
    assert cap == "" and ctx == "context here"


def test_merged_context_used_without_second_call():
    """The stashed context must be consumed, not re-requested."""
    vlm = _VLM()
    with _merged_on():
        pre = Preprocessor(vlm=vlm, caption_with_vlm=True, chunk_size=8, contextualize=True)
        out = pre([dict(TEXT_IMG)], {"text", "image", "video"})
    assert _MERGED_CONTEXT_KEY in out[0]
    before = len(vlm.calls)
    res = asyncio.run(C.acontextualize_docs(out, vlm))
    assert len(vlm.calls) == before, "contextualize must not re-call the VLM"
    assert "[Document context]:" in res[0]["text"]


def test_merged_is_off_by_default():
    """The merged path must NOT run unless RAG_MERGED_ANNOTATION is set.

    Measured 2026-10-26: the merged prompt produced a 26-word caption where
    the dedicated captioning prompt produced 243 words on the same page, and
    the caption drives retrieval-time reuse plus the caption-twin embedding.
    """
    os.environ.pop("RAG_MERGED_ANNOTATION", None)
    assert C.merged_annotation_enabled() is False
    vlm = _VLM()
    pre = Preprocessor(vlm=vlm, caption_with_vlm=True, chunk_size=8, contextualize=True)
    out = pre([dict(TEXT_IMG)], {"text", "image", "video"})
    assert not any(c.startswith("merged") for c in vlm.calls), vlm.calls
    assert _MERGED_CONTEXT_KEY not in out[0]


def test_stash_never_leaks_on_skipped_docs():
    """A memory doc is SKIPPED by the contextualizer, so it never consumes the
    stash - the internal key must still not reach the stored payload."""
    vlm = _VLM()
    doc = {
        "text": "Durable fact: user prefers tabs.",
        "image": IMG,
        "source": "/d/m.txt",
        "memory_kind": "fact",
    }
    pre = Preprocessor(vlm=vlm, caption_with_vlm=True, chunk_size=8, contextualize=True)
    out = pre([dict(doc)], {"text", "image", "video"})
    res = asyncio.run(C.acontextualize_docs(out, vlm))
    assert _MERGED_CONTEXT_KEY not in res[0]


def test_to_documents_strips_stash():
    """Defence in depth: the doc->metadata converter drops the internal key."""
    from multimodal_rag.rag_system import MultimodalRAG

    doc = {"text": "body", _MERGED_CONTEXT_KEY: "should not persist", "source": "/d/x.pdf"}
    docs = MultimodalRAG._to_documents([doc])
    assert _MERGED_CONTEXT_KEY not in docs[0].metadata


def test_empty_merged_caption_falls_through_to_caption_call():
    """The merged prompt allows "CAPTION: NONE".  When that comes back for a
    doc that DOES have an image, the caption artifact must not be silently
    lost - the normal caption call has to run."""
    vlm = _VLM(merged="CAPTION: NONE\nCONTEXT: Context still produced.")
    res = _run(TEXT_IMG, vlm)
    # merged, then the fallback caption call, then contextualize consumes stash
    assert vlm.calls[0] == "merged+img"
    assert "single+img" in vlm.calls, vlm.calls
    text = res[0]["text"]
    assert "[Image description]:" in text, "caption artifact was lost"
    assert "[Document context]:" in text
