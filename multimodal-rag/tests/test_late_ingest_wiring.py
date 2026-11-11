"""Ingest-side wiring of the late-interaction sidecar (feature D hook in rag_system).

Feature D owns the sidecar (container, writes, fail-closed rules); the ONLY
main-collection upsert in the codebase is in MultimodalRAG.aadd_to_vector_store,
so the ingest hook has to live there.  These tests pin the wiring, not feature D's
internals (those are covered by tests/test_late_interaction.py):

  * late-ENABLED dataset: the helper is called exactly ONCE per sub-batch with the
    SAME point ids the main upsert just wrote (and the exact stored texts), and a
    real sidecar point is written per document, addressable by the main point id.
  * late-DISABLED dataset: the helper is still called (it is the helper's own
    state-file read that decides -- there is deliberately no second gate) but it
    is a NO-OP: zero return, no sidecar collection, and the main upsert is
    unchanged (same ids, same payload texts).
  * FAIL-CLOSED: an enabled dataset with no token embedder makes the INGEST raise
    LateInteractionUnavailable -- the hook is not wrapped in a broad try/except,
    so an enabled dataset can never end up with late-blind points.

Real ingest path on embedded local Qdrant with the house stub embedder; only the
token embedder is faked (there is no token endpoint in CI), exactly like
tests/test_late_interaction.py does.

Run::

    pytest tests/test_late_ingest_wiring.py -q
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multimodal_rag import late_interaction as late_lane
from multimodal_rag import rag_system
from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings

DIM = 4
COLL = "late_ingest_wiring_ds"


class _StubEmbedder:
    """Deterministic hash embedder (house pattern) -- no endpoint needed."""

    allowable_modalities = ("text", "image", "video")
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
        self.model = MultiModalEmbeddings(self)
        self.model.aembed_documents = self._aembed_documents  # type: ignore[method-assign, assignment]
        self.model.aembed_query = self._aembed_query  # type: ignore[method-assign, assignment]
        self.model.embed_query = self._embed_query  # type: ignore[method-assign, assignment]

    def remote(self) -> None:
        """No-op stand-in for the real model's remote-mode switch."""

    @staticmethod
    def _vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [float(b) / 255.0 for b in h[:DIM]]

    async def _aembed_documents(self, docs: Any) -> list[list[float]]:
        return [self._vec(d if isinstance(d, str) else (d.get("text") or "")) for d in docs]

    async def _aembed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)

    def _embed_query(self, query: Any) -> list[float]:
        text = query if isinstance(query, str) else (query.get("text") if isinstance(query, dict) else "") or ""
        return self._vec(text)


class _FakeTokenEmbedder:
    """Two unit token vectors per text, dim DIM -- the TokenEmbedder protocol."""

    def __init__(self, dim: int = DIM, tokens: int = 2) -> None:
        self.dim = dim
        self.tokens = tokens
        self.calls: list[list[str]] = []

    def embed_tokens(self, texts: list[str]) -> list[list[list[float]]]:
        self.calls.append(list(texts))
        out: list[list[list[float]]] = []
        for index, _text in enumerate(texts):
            rows = []
            for token in range(self.tokens):
                row = [0.0] * self.dim
                row[(index + token) % self.dim] = 1.0
                rows.append(row)
            out.append(rows)
        return out


class _Rig:
    """A real hybrid-schema Qdrant store + RAG with a real per-dataset state file."""

    def __init__(self, tmp_path: Path, late: dict[str, Any] | None) -> None:
        self.dataset_dir = tmp_path / "ds"
        files_dir = self.dataset_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        self.late_state_path = self.dataset_dir / late_lane.LATE_STATE_FILENAME
        if late is not None:
            late_lane.write_late_state(self.dataset_dir, late)
        self.emb = _StubEmbedder()
        self.client = QdrantClient(":memory:")
        self.store = MultimodalRAG._build_qdrant_vector_store(
            embedding=self.emb.model,
            client=self.client,
            collection_name=COLL,
            bm25_stats_path=str(files_dir / ".bm25_stats.json"),
            late_state_path=str(self.late_state_path),
        )
        self.rag = MultimodalRAG(embedder=self.emb, vector_store=self.store, preprocess=False)  # type: ignore[arg-type]

    def all_ids(self, collection: str) -> list[str]:
        ids: list[str] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(collection, limit=100, offset=offset, with_payload=False)
            ids.extend(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def collection_names(self) -> set[str]:
        return {c.name for c in self.client.get_collections().collections}


def _doc(text: str, source: str) -> dict[str, Any]:
    return {"text": text, "source": source, "page": 1, "chunk_index": 0}


DOCS = [_doc("alpha widgets text", "/docs/a.md"), _doc("beta sprockets text", "/docs/b.md")]
TEXTS = ["alpha widgets text", "beta sprockets text"]


def _spy_helper(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every ingest hook call while still running the REAL helper."""
    calls: list[dict[str, Any]] = []
    real = rag_system.write_late_sidecar_for_docs

    def _spy(vs: Any, point_ids: list[str], texts: list[str]) -> int:
        # Record BEFORE calling the real helper so a fail-closed raise is still
        # visible as "the hook was reached" (written stays None).
        entry: dict[str, Any] = {"vs": vs, "ids": list(point_ids), "texts": list(texts), "written": None}
        calls.append(entry)
        entry["written"] = real(vs, point_ids, texts)
        return int(entry["written"])

    monkeypatch.setattr(rag_system, "write_late_sidecar_for_docs", _spy)
    return calls


def test_late_enabled_ingest_writes_the_sidecar_for_the_upserted_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig(tmp_path, late={"enabled": True, "token_dim": DIM})
    token_embedder = _FakeTokenEmbedder()
    monkeypatch.setattr(late_lane, "resolve_token_embedder", lambda config=None: token_embedder)
    calls = _spy_helper(monkeypatch)

    ids = asyncio.run(rig.rag.aadd_to_vector_store([dict(d) for d in DOCS], deduplicate=False))

    assert len(ids) == len(DOCS)
    assert len(calls) == 1, "exactly one hook call per upserted sub-batch"
    assert calls[0]["ids"] == ids, "the hook gets the SAME point ids the main upsert wrote"
    assert calls[0]["texts"] == TEXTS, "and the texts those points were built from"
    assert calls[0]["written"] == len(ids), "one sidecar point per stored document"
    assert token_embedder.calls == [TEXTS], "token matrices are embedded once for the batch"

    sidecar = f"{COLL}__late"
    assert sidecar in rig.collection_names(), "the sidecar collection is created on ingest"
    assert rig.client.count(sidecar, exact=True).count == len(ids)
    assert set(rig.all_ids(sidecar)) == set(ids), "sidecar point ids match the main collection"
    stored = rig.client.retrieve(sidecar, ids=ids, with_vectors=["late"], with_payload=True)
    assert {str(p.id) for p in stored} == set(ids)
    matrices: list[list[Any]] = []
    for point in stored:
        vectors = point.vector if isinstance(point.vector, dict) else {}
        matrix = vectors.get("late")
        assert isinstance(matrix, list), "each sidecar point carries the named 'late' multivector"
        matrices.append(matrix)
    assert all(len(matrix) == 2 for matrix in matrices), "two token rows per document"


def test_late_disabled_ingest_is_unchanged_and_the_hook_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig(tmp_path, late=None)  # no state file at all -> late disabled
    token_embedder = _FakeTokenEmbedder()
    monkeypatch.setattr(late_lane, "resolve_token_embedder", lambda config=None: token_embedder)
    calls = _spy_helper(monkeypatch)

    ids = asyncio.run(rig.rag.aadd_to_vector_store([dict(d) for d in DOCS], deduplicate=False))

    # The hook is called unconditionally (the helper owns the gate) but does nothing.
    assert len(calls) == 1
    assert calls[0]["written"] == 0, "a disabled dataset must not write sidecar points"
    assert calls[0]["ids"] == ids
    assert token_embedder.calls == [], "no token embedding is spent on a disabled dataset"

    # Ingest itself is unchanged: one main point per doc, same ids, same payload text.
    assert rig.collection_names() == {COLL}, "no sidecar collection appears"
    assert sorted(rig.all_ids(COLL)) == sorted(ids)
    points = rig.client.retrieve(COLL, ids=ids, with_payload=True, with_vectors=False)
    assert {str(p.id): (p.payload or {})["page_content"] for p in points} == dict(zip(ids, TEXTS))


def test_late_enabled_ingest_without_a_token_embedder_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _Rig(tmp_path, late={"enabled": True, "token_dim": DIM})
    monkeypatch.setattr(late_lane, "resolve_token_embedder", lambda config=None: late_lane.UnconfiguredTokenEmbedder())
    calls = _spy_helper(monkeypatch)

    with pytest.raises(late_lane.LateInteractionUnavailable):
        asyncio.run(rig.rag.aadd_to_vector_store([dict(d) for d in DOCS], deduplicate=False))

    # The error is NOT swallowed by the ingest path -- that is the whole point.
    assert len(calls) == 1, "the hook was reached (and raised) instead of being skipped or caught"
    assert calls[0]["written"] is None, "the helper raised before returning a count"
    assert set(calls[0]["ids"]) == set(rig.all_ids(COLL)), "the main upsert had already written these points"
    sidecar = f"{COLL}__late"
    assert sidecar not in rig.collection_names() or rig.client.count(sidecar, exact=True).count == 0
