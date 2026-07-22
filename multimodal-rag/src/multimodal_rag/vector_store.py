"""Langchain-free vector store layer for MultimodalRAG.

Replaces:
  * ``langchain_core.documents.Document``
  * ``langchain_core.vectorstores.in_memory.InMemoryVectorStore``
  * ``langchain_qdrant.QdrantVectorStore``

with minimal local equivalents exposing only the surface used by ``rag_system.py``.
Qdrant operations that ``rag_system`` performs directly via ``qdrant_client``
(upsert / scroll / query_batch_points / delete) are unchanged — this module
only supplies the container object holding ``_client`` / ``collection_name`` /
``vector_name`` plus the similarity-search methods used at retrieval time.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from multimodal_rag.utils.general_tools import cosine_sim


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Minimal base class so ``isinstance(vs, VectorStore)`` keeps working.

    Concrete implementations are :class:`InMemoryVectorStore` and
    :class:`QdrantVectorStore`. The shared retrieval surface
    (``similarity_search_with_score_by_vector`` /
    ``asimilarity_search_with_relevance_scores``) is declared here; scores are
    raw cosine similarity (higher = more similar) for both backends.
    """

    def similarity_search_with_score_by_vector(self, embedding: list[float], k: int) -> list[tuple[Document, float]]:
        raise NotImplementedError

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        raise NotImplementedError

    async def asimilarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """In-process store mirroring the langchain ``InMemoryVectorStore`` surface.

    ``store`` maps ``doc_id -> {"document": Document, "vector": list[float] | None}``,
    matching the layout ``rag_system.py`` reads/writes directly.
    """

    def __init__(self, embedding: Any) -> None:
        self.embedding = embedding
        self.store: dict[str, dict[str, Any]] = {}

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids: list[str] = []
        for doc in documents:
            doc_id = uuid.uuid4().hex
            self.store[doc_id] = {"document": doc, "vector": None}
            ids.append(doc_id)
        return ids

    def similarity_search_with_score_by_vector(self, embedding: list[float], k: int) -> list[tuple[Document, float]]:
        entries = [(s["document"], s["vector"]) for s in self.store.values() if s.get("vector") is not None]
        if not entries:
            return []
        docs, vecs = zip(*entries)
        sims = cosine_sim(
            np.asarray(embedding, dtype=np.float32).reshape(1, -1),
            np.asarray(vecs, dtype=np.float32),
        )[0]
        k = min(k, len(docs))
        top = np.argsort(sims)[-k:][::-1]
        return [(docs[i], float(sims[i])) for i in top]

    async def asimilarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        emb = await self.embedding.aembed_query(query)
        return self.similarity_search_with_score_by_vector(emb, k)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store wrapping ``qdrant_client`` directly.

    Exposes ``_client`` / ``collection_name`` / ``vector_name`` for the direct
    upsert / scroll / query_batch_points / delete calls in ``rag_system.py``
    (and ``dataset_manager`` / ``api_server`` / ``mcp_server``), plus the
    similarity-search helpers used at retrieval time.

    The collection is expected to store payloads with ``page_content`` and
    ``metadata`` keys (the layout ``rag_system._build_qdrant_vector_store``
    writes), matching the previous langchain-qdrant defaults.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedding: Any,
        vector_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.vector_name = vector_name
        # Accept and ignore langchain-style kwargs (content_payload_key,
        # metadata_payload_key, distance_strategy, ...) for forward-compat.
        self._extra_kwargs = kwargs

    def similarity_search_with_score_by_vector(self, embedding: list[float], k: int) -> list[tuple[Document, float]]:
        from qdrant_client.models import QueryRequest

        responses = self._client.query_batch_points(
            collection_name=self.collection_name,
            requests=[
                QueryRequest(
                    query=embedding,
                    using=self.vector_name,
                    limit=k,
                    with_payload=True,
                    with_vector=False,
                )
            ],
        )
        out: list[tuple[Document, float]] = []
        for resp in responses:
            for pt in resp.points:
                payload = pt.payload or {}
                meta = dict(payload.get("metadata", {}))
                doc = Document(page_content=payload.get("page_content", ""), metadata=meta)
                out.append((doc, float(pt.score)))
        return out

    async def asimilarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple[Document, float]]:
        emb = await self.embedding.aembed_query(query)
        return self.similarity_search_with_score_by_vector(emb, k)
