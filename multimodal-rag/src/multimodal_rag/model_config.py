"""Build model instances from environment variables.

Replaces the hardcoded model definitions in ``pcai_models.py`` for
production deployment.  URLs, API keys, and (optionally) model names
are injected at pod start via ConfigMap / Secret, keeping secrets out
of the Docker image; model names are auto-discovered from the endpoint
when not specified.

Environment variable convention (each model role has its own prefix)::

    MODEL_<ROLE>_NAME       — optional served model id, e.g.
                              "Qwen/Qwen3-VL-Embedding-8B".  When empty
                              the id is auto-discovered via GET /v1/models.
    MODEL_<ROLE>_URL        — full remote URL (e.g. "https://..."); required
                              to enable the role (empty disables it)
    MODEL_<ROLE>_API_KEY    — API key / service-account token
    MODEL_<ROLE>_CLASS      — Python class: "MultiModalEmbeddings" |
                              "MultiModalReranker"
    MODEL_<ROLE>_EXTRA      — JSON object of additional constructor kwargs
                              (e.g. ``{"chunk_size": 2048, "embedding_dim": 4096}``)

Roles:  EMBEDDER, RERANKER, VLM, ASR
"""

import json
import os
from typing import Any, Optional

from multimodal_rag.utils.model_adapters import (
    MultiModalEmbeddings,
)
from multimodal_rag.utils.pcai_model_classes import (
    ChatModel,
    EmbeddingModel,
    RerankerModel,
    VoiceModel,
)

# ---------------------------------------------------------------------------
# Defaults (sensible for Qwen3-VL models; override via EXTRA env var)
# ---------------------------------------------------------------------------

_DEFAULT_MM_KWARGS: dict[str, Any] = dict(
    fps=1.0,
    max_frames=64,
    min_pixels=4096,
    max_pixels=720 * 720,
    total_pixels=5 * 720 * 720,
)

_VLM_MODALITIES = ("text", "image", "video")

_EMBEDDER_INST_KWARGS: dict[str, Any] = dict(
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(prefix: str, key: str, default: str = "") -> str:
    return os.environ.get(f"{prefix}_{key}", default)


def _load_extra(prefix: str) -> dict[str, Any]:
    """Read ``<prefix>_EXTRA`` as a JSON object and return the dict."""
    raw = _get(prefix, "EXTRA")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Invalid JSON in %s_EXTRA: %s — ignoring",
            prefix,
            exc,
        )
        return {}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_embedder(prefix: str = "MODEL_EMBEDDER") -> Optional[EmbeddingModel]:
    # URL is required to enable a role; NAME is optional and, when empty,
    # auto-discovered from the endpoint's /v1/models listing.
    url = _get(prefix, "URL")
    if not url:
        return None
    name = _get(prefix, "NAME")
    api_key = _get(prefix, "API_KEY")

    extra = _load_extra(prefix)

    # Merge defaults with extra (extra wins)
    mm_kwargs = dict(_DEFAULT_MM_KWARGS)
    mm_kwargs.update(extra.pop("mm_processor_kwargs", {}))

    inst_kwargs = dict(_EMBEDDER_INST_KWARGS)
    inst_kwargs.update(extra.pop("model_instantiation_kwargs", {}))

    return EmbeddingModel(
        model_name=name,
        url_remote=url,
        api_key=api_key,
        model_instantiation_class=MultiModalEmbeddings,
        allowable_modalities=tuple(extra.pop("allowable_modalities", _VLM_MODALITIES)),
        mm_processor_kwargs=mm_kwargs,
        model_instantiation_kwargs=inst_kwargs,
        **extra,
    )


def build_reranker(prefix: str = "MODEL_RERANKER") -> Optional[RerankerModel]:
    # URL is required to enable a role; NAME is optional and, when empty,
    # auto-discovered from the endpoint's /v1/models listing.
    url = _get(prefix, "URL")
    if not url:
        return None
    name = _get(prefix, "NAME")
    api_key = _get(prefix, "API_KEY")

    extra = _load_extra(prefix)
    mm_kwargs = dict(_DEFAULT_MM_KWARGS)
    mm_kwargs.update(extra.pop("mm_processor_kwargs", {}))

    return RerankerModel(
        model_name=name,
        url_remote=url,
        api_key=api_key,
        allowable_modalities=tuple(extra.pop("allowable_modalities", _VLM_MODALITIES)),
        mm_processor_kwargs=mm_kwargs,
        **extra,
    )


def build_vlm(prefix: str = "MODEL_VLM") -> Optional[ChatModel]:
    # URL is required to enable a role; NAME is optional and, when empty,
    # auto-discovered from the endpoint's /v1/models listing.
    url = _get(prefix, "URL")
    if not url:
        return None
    name = _get(prefix, "NAME")
    api_key = _get(prefix, "API_KEY")
    extra = _load_extra(prefix)
    return ChatModel(
        model_name=name,
        url_remote=url,
        api_key=api_key,
        allowable_modalities=tuple(extra.pop("allowable_modalities", _VLM_MODALITIES)),
        **extra,
    )


def build_asr(prefix: str = "MODEL_ASR") -> Optional[VoiceModel]:
    # URL is required to enable a role; NAME is optional and, when empty,
    # auto-discovered from the endpoint's /v1/models listing.
    url = _get(prefix, "URL")
    if not url:
        return None
    name = _get(prefix, "NAME")
    api_key = _get(prefix, "API_KEY")
    extra = _load_extra(prefix)
    return VoiceModel(
        model_name=name,
        url_remote=url,
        api_key=api_key,
        **extra,
    )


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

ModelPack = tuple[
    Optional[EmbeddingModel],  # embedder
    Optional[RerankerModel],  # reranker
    Optional[ChatModel],  # vlm
    Optional[VoiceModel],  # asr
]


def build_all() -> ModelPack:
    """Build all four models from environment variables.

    Falls back to ``None`` for any model whose ``_URL`` variable is not
    set (``_NAME`` is optional and auto-discovered when omitted).
    """
    return (
        build_embedder("MODEL_EMBEDDER"),
        build_reranker("MODEL_RERANKER"),
        build_vlm("MODEL_VLM"),
        build_asr("MODEL_ASR"),
    )
