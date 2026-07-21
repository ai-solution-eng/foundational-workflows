# flake8: noqa: E501
"""Model definitions for the multimodal RAG pipeline.

This is the public, scrubbed template.  All ``api_key`` and
``url_remote`` values are intentionally empty.  Populate them locally
from ``pcai_models_full.py`` (gitignored) or, in production, via the
environment-driven factory in ``multimodal_rag.model_config`` which
reads ``MODEL_<ROLE>_URL`` / ``MODEL_<ROLE>_API_KEY`` from the pod
ConfigMap / Secret.
"""
from .langchain_overrides import MultiModalEmbeddings
from .pcai_model_classes import (
    ChatModel,
    EmbeddingModel,
    RerankerModel,
    VoiceModel,
    input_modalities,
)
from .preprocessors.qwen3_vl_8b import (
    prepare_vllm_inputs as qwen3_vl_8b_template,
)

__all__ = [
    "deepseek_v4_flash_280B",
    "gemma4_31B",
    "qwen3_vl_8B",
    "qwen3_vl_reranker_8B",
    "cohere_transcribe_3_2b",
]


vlm_modalities: tuple[input_modalities, input_modalities, input_modalities] = (
    "text",
    "image",
    "video",
)

# LLMs
gemma4_31B = ChatModel(
    model_name="RedHatAI/gemma-4-31B-it-FP8-block",
    url_remote="",
    api_key="",
    allowable_modalities=vlm_modalities,
)

deepseek_v4_flash_280B = ChatModel(
    model_name="deepseek-ai/DeepSeek-V4-Flash",
    url_remote="",
    api_key="",
)

_qwen3_vl_mm_proc_kwargs = dict(
    fps=1.0,
    max_frames=64,
    min_pixels=4096,
    max_pixels=720 * 720,
    total_pixels=5 * 720 * 720,
)

# Embeddings
qwen3_vl_8B = EmbeddingModel(
    model_name="Qwen/Qwen3-VL-Embedding-8B",
    model_instantiation_class=MultiModalEmbeddings,
    url_remote="",
    api_key="",
    embedding_dim=4096,
    model_instantiation_kwargs=dict(tiktoken_enabled=False, check_embedding_ctx_length=False),
    code_chunk_size=8192,
    code_chunk_overlap=512,
    tokenizer_name="Qwen/Qwen3-VL-Reranker-8B",
    tokenizer_type="HuggingFace",
    preprocessor=qwen3_vl_8b_template,
    allowable_modalities=vlm_modalities,
    mm_processor_kwargs=_qwen3_vl_mm_proc_kwargs,
)

# Reranker
qwen3_vl_reranker_8B = RerankerModel(
    model_name="Qwen/Qwen3-VL-Reranker-8B",
    url_remote="",
    api_key="",
    preprocessor=qwen3_vl_8b_template,
    allowable_modalities=vlm_modalities,
    mm_processor_kwargs=_qwen3_vl_mm_proc_kwargs,
)

# Voice Models
cohere_transcribe_3_2b = VoiceModel(
    model_name="CohereLabs/cohere-transcribe-03-2026",
    url_remote="",
    api_key="",
)
