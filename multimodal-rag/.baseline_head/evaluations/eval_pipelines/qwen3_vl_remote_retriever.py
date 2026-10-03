import base64
import io
import math
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vidore_benchmark.retrievers.base_vision_retriever import BaseVisionRetriever
from vidore_benchmark.retrievers.registry_utils import register_vision_retriever
from vidore_benchmark.utils.iter_utils import batched

from multimodal_rag.utils.pcai_models import qwen3_vl_8B


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _cosine_similarity(
    q_emb: Union[torch.Tensor, np.ndarray],
    p_emb: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(q_emb, torch.Tensor):
        q_norm = q_emb / (q_emb.norm(dim=-1, keepdim=True) + 1e-12)
        p_norm = p_emb / (p_emb.norm(dim=-1, keepdim=True) + 1e-12)
        return torch.mm(q_norm, p_norm.T)
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=-1, keepdims=True) + 1e-12)
    p_norm = p_emb / (np.linalg.norm(p_emb, axis=-1, keepdims=True) + 1e-12)
    return np.dot(q_norm, p_norm.T)


@register_vision_retriever("qwen3_vl_remote")
class Qwen3VLRemoteRetriever(BaseVisionRetriever):
    """
    Vision retriever that uses the remote VLLM-hosted Qwen3-VL-Embedding-8B
    via its OpenAI-compatible API.

    Supports three modalities (set via ``modality``):
    - ``"text"``    – embed only the markdown text of each passage
    - ``"image"``   – embed only the page image
    - ``"both"``    – embed the image and text jointly (default)

    When ``modality="text"``, set ``use_visual_embedding=False`` so the
    evaluator passes ``text_description`` instead of ``image``.
    For ``"image"`` and ``"both"`` the evaluator passes PIL images.
    """

    def __init__(
        self,
        modality: str = "both",
        embedding_batch_size: int = 8,
        top_k: int = 100,
        **kwargs,
    ):
        use_visual = modality in ("image", "both")
        super().__init__(use_visual_embedding=use_visual)
        self.modality = modality
        self.embedding_batch_size = embedding_batch_size
        self.top_k = top_k

        qwen3_vl_8B.remote()
        self._embedder = qwen3_vl_8B.model  # MultiModalEmbeddings
        self._model_name = qwen3_vl_8B.model_name

        # Optional corpus markdown (set externally for "both" mode)
        self.corpus_texts: Optional[List[str]] = None

    # ── passage embedding ────────────────────────────────────────────

    def forward_passages(
        self,
        passages: Any,
        batch_size: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.modality == "text":
            return self._embed_texts(cast(List[str], passages), batch_size)
        return self._embed_images(cast(List[Image.Image], passages), batch_size)

    def _embed_texts(self, texts: List[str], batch_size: int) -> List[torch.Tensor]:
        all_embs: List[torch.Tensor] = []
        for batch in tqdm(
            batched(texts, batch_size),
            total=math.ceil(len(texts) / batch_size),
            desc="Embedding text passages",
            leave=False,
        ):
            batch = cast(List[str], batch)
            vecs = self._embedder.embed_documents(list(batch))
            all_embs.extend(torch.tensor(v, dtype=torch.float32) for v in vecs)
        return all_embs

    def _embed_images(
        self, images: List[Image.Image], batch_size: int
    ) -> List[torch.Tensor]:
        all_embs: List[torch.Tensor] = []
        for batch in tqdm(
            batched(images, batch_size),
            total=math.ceil(len(images) / batch_size),
            desc="Embedding image passages",
            leave=False,
        ):
            batch = cast(List[Image.Image], batch)
            inputs: List[str | Dict[str, Any]] = []
            for i, img in enumerate(batch):
                uri = _pil_to_data_uri(img.convert("RGB"))
                if self.modality == "image":
                    inputs.append({"image": uri, "text": ""})
                else:  # both
                    txt = (
                        self.corpus_texts[i]
                        if self.corpus_texts is not None and i < len(self.corpus_texts)
                        else ""
                    )
                    inputs.append({"text": txt, "image": uri})
            vecs = self._embedder.embed_documents(inputs)
            all_embs.extend(torch.tensor(v, dtype=torch.float32) for v in vecs)
        return all_embs

    # ── query embedding ──────────────────────────────────────────────

    def forward_queries(
        self,
        queries: Any,
        batch_size: int,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        queries = cast(List[str], queries)
        all_embs: List[torch.Tensor] = []
        for batch in tqdm(
            batched(queries, batch_size),
            total=math.ceil(len(queries) / batch_size),
            desc="Embedding queries",
            leave=False,
        ):
            batch = cast(List[str], batch)
            vecs = self._embedder.embed_documents(list(batch))
            all_embs.extend(torch.tensor(v, dtype=torch.float32) for v in vecs)
        return all_embs

    # ── scoring ──────────────────────────────────────────────────────

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)
        return _cosine_similarity(query_embeddings, passage_embeddings)
