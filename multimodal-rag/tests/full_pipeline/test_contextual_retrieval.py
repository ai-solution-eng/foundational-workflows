"""Offline tests for ingest-time contextual retrieval (feature: contextual
retrieval, DECISIONS.md 2026-09).

Covers:

  * ``utils/contextualizer.py`` with a stub VLM (the rag.vlm ChatModel shape):
    context prepended as the ``[Document context]:`` line +
    ``contextualized: True`` stamp; fail-open on LLM errors (plain chunk +
    ingest warning); empty completion stored plain; concurrency bounded by
    RAG_CONTEXTUAL_CONCURRENCY (<= 0 disables the bound); the doc preamble is
    STABLE across chunks of one document (the prefix-caching contract);
    pure-media docs (no ``_has_real_text``) skipped; memory_kind-tagged docs
    skipped; ``vlm=None`` is an identity pass-through.
  * End-to-end ingest through ``MultimodalRAG.aadd_to_vector_store`` with
    ``contextualize=True``: real-text docs get the context line; the
    multimodal base embedding AND the text-only twin embed the
    contextualized text; pure-media caption twins are NOT contextualized.
  * The flag rail: create_dataset(contextual=) stamps meta; PATCH allowlist
    flips it and invalidates the cached RAG; disabled default = zero
    behaviour change (no meta key on the RAG, identity ingest).
  * BM25 indexability: the context text flows into ``_bm25_indexable_text``
    and therefore into the sparse lane for free.
  * The fingerprint-guard extension: ``meta["contextual"]`` is recorded next
    to the embedder fingerprint, and a mode mismatch WARNS (never raises).
  * The cost-preview math: ``contextualize_preview`` (labeled estimate, twin
    multiplier, model name, live-count fallback to meta).

No Qdrant server, no model endpoints — embedded local Qdrant + stubs.

Run::

    python tests/full_pipeline/test_contextual_retrieval.py    # standalone
    pytest tests/full_pipeline/test_contextual_retrieval.py    # under pytest
"""

import asyncio
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient

from multimodal_rag import api_server
from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.rag_system import MultimodalRAG, _bm25_indexable_text, _has_real_text, _ingest_warnings
from multimodal_rag.utils import contextualizer
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

IMG = "data:image/jpeg;base64,AA=="

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@contextmanager
def _env(**overrides: str):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _StubEmbedderModel:
    """Deterministic stub embedder (house pattern from test_twins.py)."""

    allowable_modalities: tuple[str, ...] = ("text", "image", "video")
    model_name = "stub-embedder"
    base_url = "http://stub/v1"
    url_remote = "http://stub"
    mm_processor_kwargs: ClassVar[dict[str, Any]] = {}
    chunk_size = 2048
    chunk_overlap = 0
    text_splitter = None

    def __init__(self) -> None:
        self.embed_logs: list[list[Any]] = []
        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[assignment]
        self.model.aembed_query = self._aembed_query  # type: ignore[assignment]
        self.model.embed_query = self._embed_query  # type: ignore[assignment]

    def remote(self) -> None:
        """No-op stand-in for the real model's remote-mode switch."""

    @staticmethod
    def _hash_vector(text: str) -> list[float]:
        import hashlib

        return [float(b) / 255.0 for b in hashlib.sha256(text.encode("utf-8")).digest()[:8]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        self.embed_logs.append(list(docs))
        return [self._hash_vector(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    async def _aembed_query(self, q: Any) -> list[float]:
        return self._hash_vector(q if isinstance(q, str) else (q.get("text") if isinstance(q, dict) else "") or "")

    def _embed_query(self, q: Any) -> list[float]:
        return self._hash_vector(q if isinstance(q, str) else (q.get("text") if isinstance(q, dict) else "") or "")


class _StubVLM:
    """Stand-in for the rag.vlm ChatModel surface the contextualizer uses.

    Modes (one active per instance):

    * ``replies``   — map a marker inside the prompt → canned context
    * ``fail_on``   — raise whenever the prompt contains the marker
    * ``empty_on``  — return an empty completion for those prompts
    * ``default``   — reply "ctx(<prompt head>)" for everything

    Records every prompt (order + concurrency trace) and exposes
    ``max_inflight`` so the concurrency-bound test can assert the semaphore.
    """

    model_name = "stub-vlm"
    url_remote = "http://stub"

    def __init__(
        self,
        replies: dict[str, str] | None = None,
        fail_on: str | None = None,
        empty_on: str | None = None,
    ) -> None:
        self.replies = replies or {}
        self.fail_on = fail_on
        self.empty_on = empty_on
        self.prompts: list[str] = []
        self.inflight = 0
        self.max_inflight = 0

    async def llm_async_chat_function_call(self, messages: Any, **kwargs: Any) -> Any:
        prompt = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for m in messages
            for part in (m["content"] if isinstance(m.get("content"), list) else [{"text": m.get("content", "")}])
        )
        self.prompts.append(prompt)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.01)
            if self.fail_on and self.fail_on in prompt:
                raise RuntimeError(f"stub VLM failure for {self.fail_on!r}")
            for marker, reply in self.replies.items():
                if marker in prompt:
                    return _resp(reply)
            if self.empty_on and self.empty_on in prompt:
                return _resp("")
            return _resp(f"ctx-{len(self.prompts)}")
        finally:
            self.inflight -= 1


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


def _resp(content: str) -> _Resp:
    return _Resp(content)


def _build_rag(
    vlm: Any = None, contextualize: bool = False, client: Any = None, coll: str = "ctx"
) -> tuple[MultimodalRAG, _StubEmbedderModel]:
    emb = _StubEmbedderModel()
    if client is not None:
        # A real stats sidecar in a temp dir — without it the BM25 lane
        # stays dark (same wiring as the _HybridRig pattern).
        files_dir = Path(tempfile.mkdtemp()) / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        store = MultimodalRAG._build_qdrant_vector_store(
            embedding=emb.model,
            client=client,
            collection_name=coll,
            bm25_stats_path=str(files_dir / "bm25_stats.json"),
        )
        rag = MultimodalRAG(embedder=emb, vlm=vlm, contextualize=contextualize, vector_store=store)
    else:
        rag = MultimodalRAG(embedder=emb, vlm=vlm, contextualize=contextualize, vector_store=None)
    return rag, emb  # type: ignore[return-value]


def _stored(rag: MultimodalRAG) -> list[dict[str, Any]]:
    out = []
    for entry in rag.vector_store.store.values():  # type: ignore[union-attr]
        doc = entry["document"]
        meta = dict(doc.metadata)
        meta["text"] = doc.page_content
        out.append(meta)
    return out


# ---------------------------------------------------------------------------
# contextualizer unit tests
# ---------------------------------------------------------------------------


def test_context_prepended_and_stamped() -> None:
    vlm = _StubVLM(replies={"Attention is all you need": "A page from the transformer paper."})
    docs = [
        {"text": "Attention is all you need.", "source": "/tmp/paper.pdf", "page": 1},
        {"text": "Attention is all you need.", "source": "/tmp/paper.pdf", "page": 2},
    ]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert len(out) == 2
    for o, orig in zip(out, docs):
        assert isinstance(o, dict)
        assert o["text"].startswith("[Document context]: A page from the transformer paper.\n")
        assert o["text"].endswith(orig["text"])
        assert o["contextualized"] is True
        # The original dicts are untouched (no caller-visible mutation).
        assert "[Document context]" not in str(orig.get("text", ""))
    # Two chunks → two LLM calls.
    assert len(vlm.prompts) == 2


def test_real_text_gate_on_marker_free_text() -> None:
    """The [Document context] marker must not confuse _has_real_text (it
    doesn't match the caption/placeholder regexes) and context text must be
    BM25-indexable."""
    line = "[Document context]: A page from the transformer paper."
    assert _has_real_text(f"{line}\nreal content") is True
    # BM25 lane keeps the context words (they are real lexical signal).
    assert "transformer" in _bm25_indexable_text(f"{line}\nreal content")


def test_fail_open_on_llm_error() -> None:
    warnings: list[str] = []
    token = _ingest_warnings.set(warnings)
    try:
        vlm = _StubVLM(fail_on="chunk-bad")
        docs = [
            {"text": "chunk-good text", "source": "/tmp/a.pdf"},
            {"text": "chunk-bad text", "source": "/tmp/a.pdf"},
        ]
        out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
        assert "ctx" in out[0]["text"] and out[0]["contextualized"] is True
        # Fail-open: the plain chunk is stored, stamped False, warning recorded.
        assert out[1]["text"] == "chunk-bad text"
        assert out[1]["contextualized"] is False
        assert any("chunk-bad" in w for w in warnings), warnings
    finally:
        _ingest_warnings.reset(token)


def test_empty_completion_stores_plain_chunk() -> None:
    vlm = _StubVLM(empty_on="dull")
    docs = [{"text": "dull chunk", "source": "/tmp/a.pdf"}]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert out[0]["text"] == "dull chunk"
    assert out[0]["contextualized"] is False


def test_preamble_stable_across_chunks_of_one_doc() -> None:
    """The prefix-caching contract: every chunk of a document shares the
    identical preamble prefix; different documents differ."""
    vlm = _StubVLM()
    docs = [{"text": f"doc-one chunk {i} " + "x" * 100, "source": "/tmp/one.pdf"} for i in range(3)] + [
        {"text": f"doc-two chunk {i}", "source": "/tmp/two.pdf"} for i in range(2)
    ]
    asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    heads = [p.split("Chunk")[0] for p in vlm.prompts]
    one = [h for h, p in zip(heads, vlm.prompts) if "one.pdf" in p]
    two = [h for h, p in zip(heads, vlm.prompts) if "two.pdf" in p]
    assert len(one) == 3 and len(set(one)) == 1, "preamble must be byte-identical across one doc's chunks"
    assert len(two) == 2 and len(set(two)) == 1
    assert one[0] != two[0]
    # Chunk position is labelled (helps the model situate the chunk).
    assert "Chunk 2 of 3." in vlm.prompts[1]


def test_pure_media_docs_skipped() -> None:
    """A context line would newly count as real text — pure-media docs must
    be skipped entirely (the verified _has_real_text interplay gotcha)."""
    vlm = _StubVLM()
    docs = [
        {
            "text": "[Image: photo.jpg]\n[Image description]: a red car on a mountain road",
            "image": IMG,
            "source": "/tmp/photo.jpg",
        },
        {"text": "real text chunk", "source": "/tmp/a.pdf"},
    ]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert "[Document context]" not in out[0]["text"], "pure-media doc must not be contextualized"
    assert out[0].get("contextualized") is False
    assert "[Document context]" in out[1]["text"]
    # Only ONE prompt: the media doc never calls the VLM.
    assert len(vlm.prompts) == 1


def test_memory_kind_docs_skipped() -> None:
    vlm = _StubVLM()
    docs = [
        {"text": "remember this fact", "source": "opencode:memory", "memory_kind": "note"},
        {"text": "normal chunk", "source": "/tmp/a.pdf"},
    ]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert "[Document context]" not in out[0]["text"]
    assert out[0].get("contextualized") is False
    assert "[Document context]" in out[1]["text"]
    assert len(vlm.prompts) == 1


def test_no_vlm_is_identity() -> None:
    docs: list[str | dict[str, Any]] = ["plain string", {"text": "chunk", "source": "/tmp/a.pdf"}]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, None))
    assert out == docs, "no VLM → the module is a no-op pass-through"


def test_concurrency_bound() -> None:
    """RAG_CONTEXTUAL_CONCURRENCY bounds in-flight calls within one
    document's fan-out; <= 0 disables the bound (the
    MODEL_EMBED_MAX_CONCURRENCY semaphore pattern)."""
    vlm = _StubVLM()
    # 9 chunks of ONE document → 1 serial first call + 8 gathered siblings.
    docs = [{"text": f"chunk {i}", "source": "/tmp/d.pdf"} for i in range(9)]
    with _env(RAG_CONTEXTUAL_CONCURRENCY="3"):
        # Fresh loop → fresh semaphore.
        asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert vlm.max_inflight == 3, f"semaphore must bound to 3, saw {vlm.max_inflight}"

    vlm2 = _StubVLM()
    with _env(RAG_CONTEXTUAL_CONCURRENCY="0"):
        asyncio.run(contextualizer.acontextualize_docs(docs, vlm2))
    assert vlm2.max_inflight > 3, "0 must disable the bound entirely"

    vlm3 = _StubVLM()
    with _env(RAG_CONTEXTUAL_CONCURRENCY="garbage"):
        assert contextualizer.concurrency() == 8, "unparseable values fall back to the default"
    asyncio.run(contextualizer.acontextualize_docs(docs[:2], vlm3))


def test_first_chunk_completes_before_siblings_fan_out() -> None:
    """The first chunk of each run is awaited BEFORE the gather — its
    response is what warms the serving engine's prefix cache."""
    order: list[str] = []

    class _OrderedVLM(_StubVLM):
        async def llm_async_chat_function_call(self, messages: Any, **kwargs: Any) -> Any:
            order.append("start")
            await asyncio.sleep(0.02)
            order.append("end")
            return _resp("ctx")

    docs = [{"text": f"chunk {i}", "source": "/tmp/d.pdf"} for i in range(4)]
    asyncio.run(contextualizer.acontextualize_docs(docs, _OrderedVLM()))
    # Strict alternation start/end would mean fully serial; the FIRST call
    # must complete (end) before the second start — that is the contract.
    assert order[0] == "start" and order[1] == "end", "first call must complete before the fan-out"


def test_long_raw_chunk_prompt_capped() -> None:
    """Raw-document drops are unbounded — the PROMPT must be capped even when
    the chunk is huge (stored text stays full)."""
    vlm = _StubVLM()
    huge = "y" * 100_000
    docs = [{"text": huge, "source": "/tmp/big.txt"}]
    out = asyncio.run(contextualizer.acontextualize_docs(docs, vlm))
    assert len(vlm.prompts[0]) < 10_000, "prompt must be capped"
    assert len(out[0]["text"]) > len(huge), "stored text is the full chunk + the context line"


# ---------------------------------------------------------------------------
# End-to-end ingest through aadd_to_vector_store (stage 0a½)
# ---------------------------------------------------------------------------


def test_ingest_contextualizes_real_text_and_twins() -> None:
    vlm = _StubVLM(replies={"transformer": "A page from the transformer paper."})
    rag, emb = _build_rag(vlm=vlm, contextualize=True)
    docs = [
        {
            "text": "Attention is all you need. The transformer uses self-attention.",
            "image": IMG,
            "source": "/tmp/paper.pdf",
            "page": 3,
        }
    ]
    asyncio.run(rag.aadd_to_vector_store(docs, deduplicate=False))
    stored = _stored(rag)
    assert len(stored) == 2, "base + text-only twin"
    for s in stored:
        assert "[Document context]: A page from the transformer paper." in s["text"]
        assert s["contextualized"] is True
    # BOTH embeddings (multimodal base and text-only twin) saw the
    # contextualized text.
    assert len(emb.embed_logs) == 2
    for batch in emb.embed_logs:
        assert "transformer paper" in batch[0]["text"]


def test_ingest_disabled_default_is_zero_behaviour_change() -> None:
    """contextualize=False (the default) → the VLM is never called and no
    stamp lands anywhere."""
    vlm = _StubVLM()
    rag, _ = _build_rag(vlm=vlm, contextualize=False)
    docs = [{"text": "plain chunk", "source": "/tmp/a.pdf"}]
    asyncio.run(rag.aadd_to_vector_store(docs, deduplicate=False))
    assert vlm.prompts == [], "no VLM call when the flag is off"
    stored = _stored(rag)
    assert stored[0]["text"] == "plain chunk"
    assert "contextualized" not in stored[0]


def test_ingest_contextualize_without_vlm_is_noop() -> None:
    rag, _ = _build_rag(vlm=None, contextualize=True)
    docs = [{"text": "plain chunk", "source": "/tmp/a.pdf"}]
    asyncio.run(rag.aadd_to_vector_store(docs, deduplicate=False))
    stored = _stored(rag)
    assert stored[0]["text"] == "plain chunk"
    assert "contextualized" not in stored[0]


def test_ingest_caption_twin_not_contextualized() -> None:
    """Pure media goes through the caption-twin path — untouched by the
    contextualizer (and the twin must NOT be contextualized either)."""
    vlm = _StubVLM()
    rag, _ = _build_rag(vlm=vlm, contextualize=True)
    docs = [
        {
            "text": "[Image: photo.jpg]\n[Image description]: a red car",
            "image": IMG,
            "source": "/tmp/photo.jpg",
        }
    ]
    asyncio.run(rag.aadd_to_vector_store(docs, deduplicate=False))
    stored = _stored(rag)
    assert len(stored) == 2, "base + caption twin"
    for s in stored:
        assert "[Document context]" not in s["text"]
        assert s.get("contextualized") is False


def test_ingest_hybrid_bm25_indexes_context_words() -> None:
    """The context line flows into the BM25 lane for free: a term that only
    exists in the context (not the chunk) is indexed in the sparse vector."""
    rag, _ = _build_rag(
        vlm=_StubVLM(replies={"surface code": "Notes on quantum error correction."}),
        contextualize=True,
        client=QdrantClient(":memory:"),
        coll="ctx-bm25",
    )
    docs = [{"text": "surface code stabilizer measurement cycle.", "source": "/tmp/q.pdf"}]
    asyncio.run(rag.aadd_to_vector_store(docs, deduplicate=False))
    vs = rag.vector_store
    client = vs._client  # type: ignore[attr-defined]
    records = client.scroll(vs.collection_name, limit=10, with_payload=True, with_vectors=True)[0]
    payload = records[0].payload or {}
    stored_text = payload.get("page_content", "")
    assert "quantum error correction" in stored_text
    # The BM25 sparse vector contains the context-only term.
    sparse = records[0].vector.get("bm25")
    assert sparse is not None and sparse.indices, "bm25 lane must be populated"
    from multimodal_rag.utils import bm25 as bm25_lane

    terms = set(bm25_lane.tokenize(stored_text))
    assert "quantum" in terms
    # And the term that ONLY exists in the context is what "quantum" is —
    # the chunk itself never mentions it.
    assert "quantum" not in docs[0]["text"]


# ---------------------------------------------------------------------------
# Flag rail (DatasetManager)
# ---------------------------------------------------------------------------


class _DM:
    """Real offline DatasetManager (webhook-test pattern)."""

    def __init__(self, base_path: str, embedder: Any) -> None:
        self.dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)


def test_create_dataset_stamps_contextual_false_by_default(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    meta = dm.create_dataset("ctx-default")
    assert meta["contextual"] is False
    assert dm._read_meta("ctx-default")["contextual"] is False


def test_create_dataset_contextual_true_flows_to_rag(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("ctx-on", contextual=True)
    assert dm._read_meta("ctx-on")["contextual"] is True
    rag = dm._get_rag("ctx-on")
    assert rag.contextualize is True


def _close_rag_client(rag: Any) -> None:
    """Close a cached RAG's local-Qdrant client (the LRU-eviction path).

    Local-mode Qdrant locks its storage dir per client, so a test that
    rebuilds the cached RAG must close the old client first — exactly what
    the cache's LRU eviction does in production (server mode never locks).
    """
    vs = getattr(rag, "vector_store", None)
    client = getattr(vs, "_client", None)
    if client is not None:
        client.close()


def test_patch_contextual_flips_and_invalidates_rag_cache(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("ctx-patch")
    rag_before = dm._get_rag("ctx-patch")
    assert rag_before.contextualize is False

    dm.update_dataset("ctx-patch", {"contextual": True})
    assert dm._read_meta("ctx-patch")["contextual"] is True
    # Local-Qdrant lock: close the evicted instance's client first (see
    # _close_rag_client).
    _close_rag_client(rag_before)
    rag_after = dm._get_rag("ctx-patch")
    assert rag_after is not rag_before, "flag flip must rebuild the cached RAG"
    assert rag_after.contextualize is True

    dm.update_dataset("ctx-patch", {"contextual": False})
    assert dm._read_meta("ctx-patch")["contextual"] is False
    _close_rag_client(rag_after)
    assert dm._get_rag("ctx-patch").contextualize is False


def test_rag_cache_invalidation_via_caption_path_covers_contextual(base_path: str, embedder: Any) -> None:
    """The cache-invalidation is the existing caption-changed path — a
    caption PATCH still invalidates (regression guard for the shared flag)."""
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("ctx-caps")
    rag1 = dm._get_rag("ctx-caps")
    dm.update_dataset("ctx-caps", {"caption_with_vlm": not rag1.caption_with_vlm})
    _close_rag_client(rag1)
    rag2 = dm._get_rag("ctx-caps")
    assert rag2 is not rag1


def test_disabled_default_zero_behaviour_change_on_rag(base_path: str, embedder: Any) -> None:
    """A default-created dataset's RAG carries contextualize=False and the
    ingest path never touches the contextualizer."""
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    dm.create_dataset("ctx-zero")
    rag = dm._get_rag("ctx-zero")
    assert rag.contextualize is False
    ids = dm.add_documents("ctx-zero", ["hello world"])
    assert len(ids) == 1
    vs = rag.vector_store
    client = vs._client  # type: ignore[attr-defined]
    records = client.scroll(vs.collection_name, limit=10, with_payload=True, with_vectors=False)[0]
    payload = records[0].payload or {}
    assert "contextualized" not in payload["metadata"]
    assert "[Document context]" not in payload["page_content"]


# ---------------------------------------------------------------------------
# Fingerprint guard extension (contextual mode)
# ---------------------------------------------------------------------------


def _guard_dm(embedder: Any, stored: dict[str, Any] | None, rag_contextualize: bool) -> DatasetManager:
    dm = DatasetManager.__new__(DatasetManager)
    dm.embedder = embedder  # type: ignore[method-assign]
    dm.contextualize = rag_contextualize  # type: ignore[method-assign]
    dm._embedder_verified = set()  # type: ignore[method-assign]
    dm._read_meta = lambda name: stored or {}  # type: ignore[method-assign]
    dm._write_meta = lambda name, meta: None  # type: ignore[method-assign]
    dm._get_meta_lock = lambda name: __import__("contextlib").nullcontext()  # type: ignore[method-assign]
    dm._embedder_dimension = lambda: 8  # type: ignore[method-assign]
    return dm


def _match(records: list[logging.LogRecord], needle: str) -> bool:
    return any(needle in r.getMessage() for r in records)


def _dm_fingerprint(base_path: str) -> tuple[DatasetManager, Any]:
    """A real DatasetManager with the endpoint verification neutralised
    (offline — the conftest embedder fixture pattern, local copy)."""
    from multimodal_rag.dataset_manager import DatasetManager as _DM

    saved = _DM._verify_endpoint
    _DM._verify_endpoint = staticmethod(lambda model, role: None)  # type: ignore[method-assign]
    try:
        embedder = _StubEmbedderModel()
        embedder.url_remote = "http://stub"
        dm = _DM(base_path=base_path, qdrant_host="", embedder=embedder)
    finally:
        _DM._verify_endpoint = saved  # type: ignore[method-assign]
    return dm, embedder


def test_fingerprint_records_contextual_mode() -> None:
    """_write_embedder_fingerprint persists contextual next to the embedder
    fingerprint."""
    dm, _ = _dm_fingerprint(tempfile.mkdtemp())
    dm.create_dataset("fp-ds", contextual=True)
    # create_dataset itself triggers _get_rag → the guard writes the
    # fingerprint (recording the DATASET's mode, not the manager default).
    meta = dm._read_meta("fp-ds")
    assert meta.get("embedder_model") == "stub-embedder"
    assert meta.get("embedder_dim") == 8
    assert meta.get("contextual") is True


def test_mode_mismatch_warns_never_raises(embedder: Any) -> None:
    dm = _guard_dm(embedder, {"embedder_model": "stub-embedder", "embedder_dim": 8, "contextual": True}, False)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logging.getLogger("multimodal_rag.dataset_manager").addHandler(handler)
    try:
        dm._assert_embedder_compatible("ds1")  # must NOT raise
    finally:
        logging.getLogger("multimodal_rag.dataset_manager").removeHandler(handler)
    assert _match(records, "contextual retrieval"), "mode mismatch must warn"
    assert any("ENABLED" in r.getMessage() for r in records)


def test_mode_match_is_silent(embedder: Any) -> None:
    dm = _guard_dm(embedder, {"embedder_model": "stub-embedder", "embedder_dim": 8, "contextual": False}, False)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logging.getLogger("multimodal_rag.dataset_manager").addHandler(handler)
    try:
        dm._assert_embedder_compatible("ds1")
    finally:
        logging.getLogger("multimodal_rag.dataset_manager").removeHandler(handler)
    assert not _match(records, "contextual retrieval")


def test_pre_context_datasets_warn_without_error(embedder: Any) -> None:
    """Datasets created before the feature carry no ``contextual`` key —
    the guard must treat that as "unknown", not a mismatch."""
    dm = _guard_dm(embedder, {"embedder_model": "stub-embedder", "embedder_dim": 8}, True)
    dm._assert_embedder_compatible("ds1")  # no raise, no mismatch warning


def test_embedder_mismatch_still_raises(embedder: Any) -> None:
    """The embedder guard's hard errors are untouched by the extension."""
    from multimodal_rag.dataset_manager import EmbedderMismatchError

    dm = _guard_dm(embedder, {"embedder_model": "stub-embedder", "embedder_dim": 8, "contextual": False}, False)
    dm._embedder_dimension = lambda: 16  # type: ignore[method-assign]
    try:
        dm._assert_embedder_compatible("ds1")
        raise AssertionError("dim mismatch must still raise")
    except EmbedderMismatchError:
        pass


# ---------------------------------------------------------------------------
# Cost preview
# ---------------------------------------------------------------------------


def _preview_dm(base_path: str, embedder: Any, vlm: Any, points: int, hybrid: bool) -> DatasetManager:
    """A DatasetManager shell whose collection reports *points* points."""
    dm = DatasetManager.__new__(DatasetManager)
    dm.vlm = vlm  # type: ignore[method-assign]
    dm.embedder = embedder  # type: ignore[method-assign]
    dm.contextualize = False  # type: ignore[method-assign]
    dm._read_meta = lambda name: {  # type: ignore[method-assign]
        "name": name,
        "document_count": points,
        "contextual": False,
    }

    class _Info:
        points_count = points

    class _VS:
        def __init__(self, name: str) -> None:
            self.collection_name = name

        def supports_hybrid(self) -> bool:
            return hybrid

    class _Client:
        def get_collection(self, coll: str) -> Any:
            return _Info()

    class _Rag:
        vector_store = _VS("ds")

    # Give the store the _client attribute the preview reads through.
    _Rag.vector_store._client = _Client()  # type: ignore[attr-defined]

    dm._get_rag = lambda name, check_embedder=True: _Rag()  # type: ignore[method-assign]
    return dm


def test_preview_math_and_labels(base_path: str, embedder: Any) -> None:
    vlm = _StubVLM()
    vlm.model_name = "stub-vlm-7b"
    dm = _preview_dm(base_path, embedder, vlm, points=100, hybrid=False)
    out = dm.contextualize_preview("ds")
    assert out["dataset"] == "ds"
    assert out["model"] == "stub-vlm-7b"
    assert out["estimated_chunk_count"] == 100
    assert out["hybrid_twin_multiplier"] == 1
    assert out["count_source"] == "qdrant"
    assert out["estimated_output_tokens_per_chunk"] == contextualizer.OUTPUT_TOKENS_PER_CHUNK
    # Token math: per-chunk input = preamble sample + chunk budget; total
    # = chunks × (input + output).
    per_chunk = out["estimated_input_tokens_per_chunk"] + out["estimated_output_tokens_per_chunk"]
    assert out["estimated_total_tokens"] == 100 * per_chunk
    # The estimate must be labeled as an estimate.
    assert "ESTIMATE" in out["estimate_basis"]
    assert out["estimated_input_tokens_per_chunk"] > contextualizer.sample_preamble_tokens()


def test_preview_doubles_for_hybrid_collections(base_path: str, embedder: Any) -> None:
    dm = _preview_dm(base_path, embedder, _StubVLM(), points=50, hybrid=True)
    out = dm.contextualize_preview("ds")
    assert out["estimated_chunk_count"] == 100, "real-text chunks get text-only twins — ×2"
    assert out["hybrid_twin_multiplier"] == 2


def test_preview_falls_back_to_meta_count(base_path: str, embedder: Any) -> None:
    dm = _preview_dm(base_path, embedder, _StubVLM(), points=0, hybrid=False)

    # Simulate a Qdrant failure: the count falls back to meta document_count.
    def _boom(name: str, check_embedder: bool = True) -> Any:
        raise RuntimeError("qdrant down")

    dm._get_rag = _boom  # type: ignore[method-assign]
    out = dm.contextualize_preview("ds")
    assert out["estimated_chunk_count"] == 0
    assert out["count_source"] == "meta_document_count"


def test_preview_missing_dataset_raises(base_path: str, embedder: Any) -> None:
    dm = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    try:
        dm.contextualize_preview("nope")
        raise AssertionError("missing dataset must raise FileNotFoundError")
    except FileNotFoundError:
        pass


def test_preview_endpoint_http_shape(base_path: str, embedder: Any) -> None:
    """The REST endpoint wires the manager call (404 passthrough + JSON)."""
    from fastapi import HTTPException

    class _PreviewDM:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def _noop_pw(self, *a: Any, **k: Any) -> None:
            return None

        def contextualize_preview(self, name: str) -> dict[str, Any]:
            self.calls.append(name)
            return {"dataset": name, "estimated_chunk_count": 7}

    dm = _PreviewDM()

    async def _fake_get_manager() -> Any:
        return dm

    async def _fake_pw(dm_: Any, name: str, pw: Any, request: Any) -> None:
        return None

    saved = (api_server.get_manager_async, api_server._require_dataset_password)
    api_server.get_manager_async = _fake_get_manager  # type: ignore[method-assign, assignment]
    api_server._require_dataset_password = _fake_pw  # type: ignore[method-assign, assignment]
    try:
        out = asyncio.run(api_server.api_contextual_preview("ds", request=None))
        assert out == {"dataset": "ds", "estimated_chunk_count": 7}
        assert dm.calls == ["ds"]

        def _missing(name: str) -> dict[str, Any]:
            raise FileNotFoundError(f"Dataset '{name}' not found")

        dm.contextualize_preview = _missing  # type: ignore[method-assign]
        try:
            asyncio.run(api_server.api_contextual_preview("ghost", request=None))
            raise AssertionError("missing dataset must be HTTP 404")
        except HTTPException as exc:
            assert exc.status_code == 404
    finally:
        api_server.get_manager_async, api_server._require_dataset_password = saved


# ---------------------------------------------------------------------------
# Estimate helpers
# ---------------------------------------------------------------------------


def test_estimate_tokens_and_sample_preamble() -> None:
    assert contextualizer.estimate_tokens("") == 0
    assert contextualizer.estimate_tokens("x" * 40) == 10
    # The digest knob bounds the sample (and is itself bounded).
    with _env(RAG_CONTEXTUAL_DIGEST_CHARS="128"):
        small = contextualizer.sample_preamble_tokens()
    with _env(RAG_CONTEXTUAL_DIGEST_CHARS="2048"):
        big = contextualizer.sample_preamble_tokens()
    assert small < big
    with _env(RAG_CONTEXTUAL_DIGEST_CHARS="bogus"):
        assert contextualizer.digest_chars() == 512


def test_env_defaults() -> None:
    with _env(RAG_CONTEXTUAL_CONCURRENCY=""):
        assert contextualizer.concurrency() == 8
    assert contextualizer.DOC_CONTEXT_MARKER == "[Document context]:"


# ---------------------------------------------------------------------------
# UI surface (template ships the controls)
# ---------------------------------------------------------------------------


def test_index_html_ships_contextual_controls() -> None:
    template = Path(__file__).resolve().parents[2] / "src" / "multimodal_rag" / "templates" / "index.html"
    html_text = template.read_text(encoding="utf-8")
    for required in (
        'id="ds-contextual"',
        'id="edit-contextual"',
        'id="ds-contextual-cost"',
        'meta[name="rag-contextual-default"]',
        "contextual-preview",
        "Recreate re-contextualizes",
    ):
        assert required in html_text, f"template must contain {required!r}"


def test_meta_injection_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gated rendering on the HTML surface: off → no INJECTED meta tag; on →
    present.  (The template's JS *consumer* mentions the meta name — the
    assertion targets the server-injected tag shape.)"""
    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", False)
    page = asyncio.run(api_server.index())
    assert '<meta name="rag-contextual-default"' not in page

    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", True)
    page = asyncio.run(api_server.index())
    assert '<meta name="rag-contextual-default" content="true">' in page


def test_chart_values_examples_carry_disabled_contextual() -> None:
    import yaml

    repo = Path(__file__).resolve().parents[2]
    example_files = [
        "helm/local/values.example.yaml",
        "helm/values-examples/values.g2.yaml",
        "helm/values-examples/values.hosted-trial.yaml",
        "helm-scale-medium/local/values.example.yaml",
        "helm-scale-medium/values-examples/values.g2.yaml",
        "helm-scale-medium/values-examples/values.hosted-trial.yaml",
        "helm-scale-large/local/values.example.yaml",
        "helm-scale-large/values-examples/values.g2.yaml",
        "helm-scale-large/values-examples/values.hosted-trial.yaml",
    ]
    for rel in example_files:
        data = yaml.safe_load((repo / rel).read_text())
        assert data["rag"]["contextual"] is False, f"{rel} must ship disabled"


def test_chart_configmap_gated_rendering() -> None:
    """The configmap renders RAG_CONTEXTUAL_DEFAULT only when
    rag.contextual is true (all three charts; the byte-identity of the
    disabled default render is covered by the watched-sources render test's
    pristine comparison and the manual baseline)."""
    import shutil
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    if shutil.which("helm") is None:
        print("  (helm not available — skipping the render check)")
        return
    for chart in ("helm", "helm-scale-medium", "helm-scale-large"):
        base = subprocess.run(["helm", "template", "t", str(repo / chart)], capture_output=True, text=True, timeout=120)
        assert base.returncode == 0
        assert "RAG_CONTEXTUAL_DEFAULT" not in base.stdout, f"{chart}: disabled default must render nothing"
        on = subprocess.run(
            ["helm", "template", "t", str(repo / chart), "--set", "rag.contextual=true"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert on.returncode == 0
        assert 'RAG_CONTEXTUAL_DEFAULT: "true"' in on.stdout, f"{chart}: enabled must render the env"


if __name__ == "__main__":
    import traceback

    failed = 0
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        import inspect

        try:
            sig = inspect.signature(fn)
            if sig.parameters:
                print(f"  {name} ... SKIP (needs fixtures)")
                continue
            fn()
            print(f"  {name} ... OK")
        except Exception:
            failed += 1
            print(f"  {name} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
