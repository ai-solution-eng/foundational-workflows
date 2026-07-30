import asyncio
import base64
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from typing import Any

import numpy as np

from multimodal_rag.utils.general_tools import (
    cosine_sim,
    list_chunker,
    retry_call,
    sync_wrapper_safe,
)
from multimodal_rag.utils.logging_utils import logging
from multimodal_rag.utils.model_adapters import (
    MultiModalEmbeddings,
    MultiModalReranker,
)
from multimodal_rag.utils.pcai_model_classes import (
    ChatModel,
    EmbeddingModel,
    RerankerModel,
    VoiceModel,
)
from multimodal_rag.vector_store import (
    Document,
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

__all__ = ["MultiModalRAGSystem", "MultimodalRAG", "Postprocessor", "Preprocessor"]


def _qdrant_prefer_grpc() -> bool:
    """Whether Qdrant clients should prefer gRPC. Defaults to False (HTTP)
    to preserve the single-replica chart's behaviour; the scale chart sets
    ``QDRANT_PREFER_GRPC=true`` for higher throughput at high QPS."""
    return os.environ.get("QDRANT_PREFER_GRPC", "false").lower() in ("true", "1", "yes")


def _qdrant_client_timeout():
    """Hard timeout (seconds) for Qdrant client calls. Defaults to None
    (no timeout) to preserve the original behaviour; the scale chart sets
    ``QDRANT_CLIENT_TIMEOUT=30`` so a hung Qdrant cannot pin a sync_pool
    thread indefinitely."""
    raw = os.environ.get("QDRANT_CLIENT_TIMEOUT")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _qdrant_quantization_config():
    """Scalar quantization config for new Qdrant collections.

    Defaults to int8 (4x memory reduction, ~1-2% recall loss, faster
    search). Set ``QDRANT_QUANTIZATION=none`` to disable, or
    ``QDRANT_QUANTIZATION=int8`` to be explicit. No rescoring is used
    — the 1-2% recall loss is acceptable without the extra full-
    precision reranking pass.

    ``QDRANT_QUANTIZATION_ALWAYS_RAM`` (default ``true``) controls whether
    quantized vectors are pinned in RAM for fast search. Disable
    (``false``) for very large datasets (2M+ points) to avoid OOM —
    quantized vectors will be read from disk per-search instead.

    Only applies to *newly created* collections; existing collections
    keep their original config.
    """
    mode = os.environ.get("QDRANT_QUANTIZATION", "int8").lower().strip()
    if mode in ("none", "off", "false", "0"):
        return None

    always_ram = os.environ.get("QDRANT_QUANTIZATION_ALWAYS_RAM", "true").lower() in ("true", "1", "yes")

    kwargs = dict(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=always_ram,
    )

    if mode not in ("int8", "scalar", "true", "1", ""):
        logger.warning("Unknown QDRANT_QUANTIZATION='%s', falling back to int8", mode)

    from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

    return ScalarQuantization(
        scalar=ScalarQuantizationConfig(**kwargs),
    )


# ---------------------------------------------------------------------------
# Media helpers (shared by Preprocessor & Postprocessor)
# ---------------------------------------------------------------------------


def _as_url_list(val: Any) -> list[str]:
    if val is None:
        return []
    return [val] if isinstance(val, str) else list(val)


def _split_media(doc: dict[str, Any]) -> dict[str, list[str]]:
    """Return {audio: [...], image: [...], video: [...]} from a doc dict."""
    return {k: _as_url_list(doc.get(k)) for k in ("audio", "image", "video")}


def _preferred_media(doc: dict[str, Any]) -> dict[str, list[str]]:
    """Like :func:`_split_media` but prefer tier-2 ``preprocessed_*`` refs.

    Each media key (``image`` / ``video`` / ``audio``) holds the tier-3
    model-ready data URL used for embedding (e.g. a 1 fps, ≤720×720 video
    segment).  A parallel ``preprocessed_image`` / ``preprocessed_video`` /
    ``preprocessed_audio`` key (when present) points at the tier-2
    preprocessed file on the PVC (e.g. the full video at ≤720p @ 24 fps),
    which is what the base LLM and end user should see by default.

    Use this helper at *retrieval* time when surfacing media URLs to the LLM
    or end user.  VLM captioning and ASR should keep using
    :func:`_split_media` so they operate on the tier-3 data URL already
    stored in Qdrant — no file read or re-resize needed.
    """
    out: dict[str, list[str]] = {}
    for k in ("audio", "image", "video"):
        preproc = doc.get(f"preprocessed_{k}")
        if preproc:
            out[k] = _as_url_list(preproc)
        else:
            out[k] = _as_url_list(doc.get(k))
    return out


def _fetch_media_bytes(url: str, client=None) -> bytes:
    if url.startswith(("http://", "https://")):
        if client is not None:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        import httpx

        with httpx.Client() as c:
            resp = c.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
    path = url.removeprefix("file://")
    with open(path, "rb") as f:
        return f.read()


async def _afetch_media_bytes(url: str, async_client=None) -> bytes:
    if url.startswith(("http://", "https://")):
        if async_client is not None:
            resp = await async_client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        import httpx

        async with httpx.AsyncClient() as c:
            resp = await c.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
    if url.startswith("data:"):
        import re

        m = re.match(r"data:[^;]+;base64,(.+)", url)
        return base64.b64decode(m.group(1)) if m else base64.b64decode(url.split(",", 1)[1])
    path = url.removeprefix("file://")
    with open(path, "rb") as f:
        return f.read()


def _file_url_to_data_url(url: str) -> str:
    """Convert a ``file://`` path to an inline ``data:`` URL.

    HTTP/S and ``data:`` URLs are returned unchanged.  This lets the
    remote VLM consume files stored on the local PVC.

    If the local file does not exist the original URL is returned so
    callers can decide how to handle the missing file.
    """
    if url.startswith(("http://", "https://", "data:")):
        return url
    import mimetypes
    import os

    path = url.removeprefix("file://")
    if not os.path.exists(path):
        logger.warning("File not found, returning URL as-is: %s", path)
        return url
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _infer_mime(url: str) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(url)
    return mime or "application/octet-stream"


def _pvc_to_http_url(url: str) -> str:
    """Convert a ``file://`` or PVC path to a clickable HTTP URL.

    Uses the ``MEDIA_BASE_URL`` environment variable (set via Helm
    ConfigMap) to build an API-accessible URL.  When ``MEDIA_BASE_URL`` is
    not set the original *url* is returned unchanged.

    Examples::

        file:///data/datasets/my-ds/files/abc.jpg
        → https://rag.example.com/api/datasets/my-ds/files/abc.jpg

        /data/datasets/my-ds/files/abc.jpg
        → https://rag.example.com/api/datasets/my-ds/files/abc.jpg
    """
    media_base = os.environ.get("MEDIA_BASE_URL", "")
    if not media_base:
        return url

    data_path = os.environ.get("DATA_PATH", "/data")
    pvc_prefix = f"{data_path}/datasets/"
    file_prefix = f"file://{pvc_prefix}"

    path = url
    if url.startswith(file_prefix):
        path = url[len(file_prefix) :]
    elif url.startswith(pvc_prefix):
        path = url[len(pvc_prefix) :]
    else:
        return url  # not a PVC path

    # path is now "{dataset_name}/files/{filename}"
    return f"{media_base}/api/datasets/{path}"


def _media_url_to_displayable(url: str) -> str:
    """Convert any stored media URL to a displayable/clickable form.

    ``file://`` and raw PVC paths become HTTP URLs (via
    :func:`_pvc_to_http_url`).  ``data:``, ``http://``, ``https://`` and
    ``s3://`` URLs are returned unchanged.
    """
    if url.startswith(("data:", "http://", "https://", "s3://")):
        return url
    return _pvc_to_http_url(url)


# ---------------------------------------------------------------------------
# Shared media-conversion helpers (used by both Preprocessor & Postprocessor)
# ---------------------------------------------------------------------------


async def _transcribe_media(url: str, asr: VoiceModel) -> str:
    """Fetch audio from *url* and transcribe via *asr*.

    Handles ``data:`` URLs (inline base64), ``http(s)://`` (fetched),
    and ``file://`` / bare paths (read from disk).
    """
    if url.startswith("data:"):
        import re

        m = re.match(r"data:[^;]+;base64,(.+)", url)
        data = base64.b64decode(m.group(1)) if m else base64.b64decode(url.split(",", 1)[1])
        mime = url.split(";")[0].replace("data:", "") or "audio/mpeg"
    else:
        data = await _afetch_media_bytes(url)
        mime = _infer_mime(url)
    buf = BytesIO(data)
    buf.name = f"audio.{mime.split('/')[-1]}"
    result = await asr.asr_async_function_call(file=(buf.name, buf, mime))
    return result.text


_DESCRIBE_SYSTEM_PROMPT_SIMPLE = (
    "You are a detailed image captioning assistant. "
    "Describe the image(s) and video(s) in this document thoroughly: "
    "include visible text, objects, people, scene context, and any "
    "relationships between them. Be precise and factual."
)

_DESCRIBE_SYSTEM_PROMPT_WITH_SOURCE = (
    "You are a detailed image captioning assistant. "
    "Describe the image(s) and video(s) in this document thoroughly: "
    "include visible text, objects, people, scene context, and any "
    "relationships between them. Be precise and factual.\n\n"
    "If source or page information is provided above, include it in "
    "your response so the document can be properly referenced."
)


async def _describe_doc(
    doc_dict: dict[str, Any],
    query: str | dict[str, Any] | None,
    vlm: ChatModel,
    system_prompt: str,
    log_timing: bool = False,
) -> str:
    """Describe images/videos in *doc_dict* via *vlm*.

    Builds OpenAI-compatible content parts from source info, text, an
    optional user query, and any image/video URLs (converted from
    ``file://`` to data URLs).  Returns the VLM's text response.

    When *log_timing* is True, logs the elapsed time and media count at
    verbose level.
    """
    t0 = time.monotonic() if log_timing else 0.0

    source_info = ""
    for key in ("source", "page", "file"):
        val = doc_dict.get(key)
        if val is not None:
            source_info += f"[{key}]: {val}\n"

    content_parts: list[dict[str, Any]] = []
    if source_info:
        content_parts.append({"type": "text", "text": source_info.strip()})
    text = doc_dict.get("text", "")
    if text:
        content_parts.append({"type": "text", "text": text})
    q_text = query if isinstance(query, str) else (query.get("text", "") if isinstance(query, dict) else "")
    if q_text:
        content_parts.append({"type": "text", "text": f"User query: {q_text}"})

    n_media = 0
    for url in _as_url_list(doc_dict.get("image", [])):
        data_url = _file_url_to_data_url(url)
        if data_url and not data_url.startswith("file://"):
            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            n_media += 1
    for url in _as_url_list(doc_dict.get("video", [])):
        data_url = _file_url_to_data_url(url)
        if data_url and not data_url.startswith("file://"):
            content_parts.append({"type": "video_url", "video_url": {"url": data_url}})
            n_media += 1

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts or ""},
    ]
    response = await vlm.llm_async_chat_function_call(messages)

    if log_timing:
        logger.verbose(  # type: ignore[attr-defined]
            "  %.2fs vlm describe  — %d media items",
            time.monotonic() - t0,
            n_media,
        )  # type: ignore[attr-defined]
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Preprocessor  —  runs BEFORE embedding / storage
# ---------------------------------------------------------------------------


@dataclass
class Preprocessor:
    """Convert unsupported modalities to text before embedding.

    Behaviour is driven by *target_modalities* (e.g. ``{"text", "image"}``).
    Any modality present in a document but absent from *target_modalities* is
    converted to text via the supplied auxiliary models:

    * ``audio``  → ASR transcription appended to ``text``
    * ``image``  → VLM description appended to ``text``
    * ``video``  → VLM description + optional ASR audio transcription

    Parameters
    ----------
    chunk_size:
        Maximum number of documents to process concurrently.  Larger ingests
        are split into chunks to avoid overwhelming the ASR / VLM endpoints.
    """

    vlm: ChatModel | None = None
    asr: VoiceModel | None = None
    caption_video: bool = False
    chunk_size: int = 128

    def __call__(
        self,
        documents: list[str | dict[str, Any]],
        target_modalities: set[str],
        query: str | dict[str, Any] | None = None,
    ) -> list[str | dict[str, Any]]:
        return sync_wrapper_safe(
            self.acall,
            {"documents": documents, "target_modalities": target_modalities, "query": query},
        )

    @cached_property
    def caption(self) -> bool:
        return self.caption_video and self.asr is not None

    async def acall(
        self,
        documents: list[str | dict[str, Any]],
        target_modalities: set[str],
        query: str | dict[str, Any] | None = None,
    ) -> list[str | dict[str, Any]]:
        async def _process_one(doc) -> str | dict[str, Any] | None:
            if isinstance(doc, str):
                return doc

            d = dict(doc)
            media = _split_media(d)
            text_parts: list[str] = [d.get("text", "")] if d.get("text") else []

            asr_tasks: list = []
            vlm_tasks: list = []

            # -- audio not supported → ASR ----------------------------------
            if "audio" not in target_modalities and media["audio"]:
                if self.asr is None:
                    # No ASR to transcribe; drop the audio payload.  Omit the
                    # doc entirely if nothing meaningful remains.
                    d.pop("audio", None)
                    has_other_media = bool(media["image"]) or bool(media["video"])
                    text = (d.get("text") or "").strip()
                    if not has_other_media and (not text or text.startswith("[Audio:")):
                        logger.warning(
                            "Omitting audio document: no ASR model and embedder does not support audio. Source: %s",
                            d.get("source", "(unknown)"),
                        )
                        return None
                    logger.warning(
                        "Dropping audio from document: no ASR model and embedder "
                        "does not support audio. Keeping remaining content. Source: %s",
                        d.get("source", "(unknown)"),
                    )
                else:
                    asr_tasks.extend([self._transcribe(url) for url in media["audio"]])

            # -- video not supported → VLM description (+ optional ASR) -----
            video_needs_vlm = "video" not in target_modalities and media["video"]
            if video_needs_vlm:
                if self.vlm is not None:
                    vlm_tasks.append(self._describe(d, query=query))
                if self.caption:
                    asr_tasks.extend([self._transcribe(url) for url in media["video"]])
                d.pop("video", None)
            # -- video supported, but its audio track ignored by embedder ----
            elif "audio" not in target_modalities and self.caption:
                asr_tasks.extend([self._transcribe(url) for url in media["video"]])

            # -- image not supported → VLM description ----------------------
            if "image" not in target_modalities and media["image"] and self.vlm is not None:
                vlm_tasks.append(self._describe(d, query=query))

            asr_results = list(await asyncio.gather(*asr_tasks)) if asr_tasks else []
            vlm_results = list(await asyncio.gather(*vlm_tasks)) if vlm_tasks else []

            # -- audio ASR results ------------------------------------------
            if "audio" not in target_modalities and media["audio"] and self.asr is not None:
                n = len(media["audio"])
                for transcript in asr_results[:n]:
                    text_parts.append(f"[Audio transcription]: {transcript}")
                asr_results = asr_results[n:]

            # -- VLM results (in order: video describe, image describe) -----
            vlm_idx = 0
            if video_needs_vlm and self.vlm is not None:
                text_parts.append(f"[Video description]: {vlm_results[vlm_idx]}")
                vlm_idx += 1
            if "image" not in target_modalities and media["image"] and self.vlm is not None:
                text_parts.append(f"[Image description]: {vlm_results[vlm_idx]}")
                vlm_idx += 1

            # -- remaining ASR results (video caption) ----------------------
            for transcript in asr_results:
                text_parts.append(f"[Video audio]: {transcript}")

            d["text"] = "\n".join(p for p in text_parts if p)
            return d

        out: list = []
        for i in range(0, len(documents), self.chunk_size):
            chunk = documents[i : i + self.chunk_size]
            results = await asyncio.gather(*[_process_one(d) for d in chunk])
            out.extend(r for r in results if r is not None)
        return out

    # -- internal helpers ---------------------------------------------------

    async def _transcribe(self, url: str) -> str:
        assert self.asr is not None
        return await _transcribe_media(url, self.asr)

    _DESCRIBE_SYSTEM_PROMPT = _DESCRIBE_SYSTEM_PROMPT_SIMPLE

    async def _describe(self, doc_dict: dict[str, Any], query: str | dict[str, Any] | None = None) -> str:
        assert self.vlm is not None
        return await _describe_doc(doc_dict, query, self.vlm, self._DESCRIBE_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Postprocessor  —  runs AFTER retrieval, converts docs for LLM consumption
# ---------------------------------------------------------------------------


@dataclass
class Postprocessor:
    """Convert retrieved documents for the LLM, preserving supported modalities.

    Behaves as a *modality gate*: any modality the target LLM supports
    natively (e.g. ``image`` / ``video``) is passed through in the output
    dict.  Unsupported modalities are converted to text via the supplied
    auxiliary models:

    * ``audio``  → ASR transcription
    * ``image``  → VLM description
    * ``video``  → VLM description + optional ASR audio transcription
    """

    vlm: ChatModel | None = None
    asr: VoiceModel | None = None
    caption_video: bool = False

    def __call__(
        self,
        documents: list,
        llm_modalities: set[str],
        query: str | dict[str, Any] | None = None,
    ) -> list[str | dict[str, Any]]:
        return sync_wrapper_safe(
            self.acall,
            {"documents": documents, "llm_modalities": llm_modalities, "query": query},
        )

    async def acall(
        self,
        documents: list,
        llm_modalities: set[str],
        query: str | dict[str, Any] | None = None,
    ) -> list[str | dict[str, Any]]:
        async def _process_one(doc) -> str | dict[str, Any]:
            d = doc if isinstance(doc, dict) else {"text": str(doc)}
            text = d.get("text", "")
            media = _split_media(d)
            # Tier-2 preprocessed_* refs — used when surfacing URLs to the
            # base LLM / users so they link a user-viewable version.
            # VLM captioning and ASR keep using ``media`` (the tier-3 data
            # URL already in Qdrant) so descriptions stay precise and cheap
            # — no file read or re-resize needed.
            preferred = _preferred_media(d)

            result: dict[str, Any] = {}
            if text:
                result["text"] = text

            # -- pass through modalities the LLM supports natively ----------
            # Surface the tier-2 preprocessed_* refs (PVC files) rather than
            # the tier-3 embedding-grade data URLs stored in Qdrant.
            for modality in ("image", "video", "audio"):
                if modality in llm_modalities and preferred[modality]:
                    result[modality] = preferred[modality]

            asr_tasks: list = []
            vlm_tasks: list = []

            # -- audio: NEVER re-run ASR at retrieval ----------------------
            # Audio transcription is the ingest Preprocessor's job — it
            # runs once when the embedder doesn't support audio natively
            # (text is enriched with "[Audio transcription]: …").  Re-
            # running ASR at retrieval is wasteful (model call per result)
            # and unsafe for large files (e.g. a 38 MB MP3 hitting the ASR
            # endpoint's payload cap).  Whatever transcription was produced
            # at ingest is already in *text*; the link below lets the
            # LLM/user listen to the actual file.
            has_audio_transcript = "[Audio transcription]:" in text or "[Caption]:" in text

            # -- image / video not supported → VLM description --------------
            needs_vlm = ("image" not in llm_modalities and media["image"]) or (
                "video" not in llm_modalities and media["video"]
            )
            if needs_vlm and self.vlm is not None:
                vlm_tasks.append(self._describe(d, query=query))

            # -- optional video audio captioning ----------------------------
            # Only for video (audio is handled above via ingest-time text).
            # Skipped when the video's audio was already captioned at ingest
            # (marker "[Video audio transcription]:" present in text).
            has_video_caption = "[Video audio transcription]:" in text
            if (
                self.caption_video
                and "video" not in llm_modalities
                and media["video"]
                and self.asr is not None
                and not has_video_caption
            ):
                asr_tasks.extend([self._transcribe(url) for url in media["video"]])

            asr_results = list(await asyncio.gather(*asr_tasks)) if asr_tasks else []
            vlm_results = list(await asyncio.gather(*vlm_tasks)) if vlm_tasks else []

            # -- ASR results (video caption only — audio handled above) ----
            if (
                self.caption_video
                and "video" not in llm_modalities
                and media["video"]
                and self.asr is not None
                and not has_video_caption
            ):
                for transcript in asr_results:
                    caption = f"[Video audio transcription]: {transcript}"
                    result["text"] = result["text"] + "\n" + caption if result.get("text") else caption

            # -- VLM description --------------------------------------------
            if needs_vlm and self.vlm is not None:
                desc = vlm_results[0]
                desc_text = f"[Media description]: {desc}"
                result["text"] = result["text"] + "\n" + desc_text if result.get("text") else desc_text

            # -- Audio link (always — no ASR at retrieval) -----------------
            # Always emit a clickable link for audio results so the LLM
            # and user can access the file.  If ingest produced a
            # transcription, it's already in *text* — no need to reproduce.
            if "audio" not in llm_modalities and preferred["audio"]:
                links = [_media_url_to_displayable(u) for u in preferred["audio"]]
                marker = "[Audio file (transcribed)]" if has_audio_transcript else "[Audio file]"
                link_text = f"{marker}: {' '.join(links)}"
                result["text"] = result["text"] + "\n" + link_text if result.get("text") else link_text

            # -- Fallback: include clickable links for unconverted media ----
            # When the LLM doesn't support a modality AND the conversion
            # model (ASR/VLM) is unavailable, include a link so the LLM
            # can share the reference with the user.  Link the tier-2
            # preprocessed_* ref when available.
            if "image" not in llm_modalities and preferred["image"] and self.vlm is None:
                links = [_media_url_to_displayable(u) for u in preferred["image"]]
                link_text = f"[Image file]: {' '.join(links)}"
                result["text"] = result["text"] + "\n" + link_text if result.get("text") else link_text
                logger.warning("No VLM model — including image link(s) instead of description.")
            if "video" not in llm_modalities and preferred["video"] and self.vlm is None:
                links = [_media_url_to_displayable(u) for u in preferred["video"]]
                link_text = f"[Video file]: {' '.join(links)}"
                result["text"] = result["text"] + "\n" + link_text if result.get("text") else link_text
                logger.warning("No VLM model — including video link(s) instead of description.")

            return result if result else ""

        return list(await asyncio.gather(*[_process_one(d) for d in documents]))

    # -- internal helpers ---------------------------------------------------

    async def _transcribe(self, url: str) -> str:
        assert self.asr is not None
        return await _transcribe_media(url, self.asr)

    _DESCRIBE_SYSTEM_PROMPT = _DESCRIBE_SYSTEM_PROMPT_WITH_SOURCE

    async def _describe(self, doc_dict: dict[str, Any], query: str | dict[str, Any] | None = None) -> str:
        assert self.vlm is not None
        return await _describe_doc(doc_dict, query, self.vlm, self._DESCRIBE_SYSTEM_PROMPT, log_timing=True)


# ---------------------------------------------------------------------------
# Pure RAG Core  (embed, store, retrieve, pre/post-process, no LLM)
# ---------------------------------------------------------------------------


@dataclass
class MultimodalRAG:
    """RAG core without an LLM — embedding, storage, retrieval, modality conversion.

    Use this directly when you only need retrieval (e.g. building a custom
    generation pipeline).  For a complete RAG + LLM experience see
    :class:`MultiModalRAGSystem`.
    """

    embedder: EmbeddingModel
    reranker: RerankerModel | None = None
    vlm: ChatModel | None = None
    asr: VoiceModel | None = None
    caption_video: bool = False
    preprocess: bool = True
    preprocess_chunk_size: int = 128
    dedup_threshold: float = 0.995

    # VectorStore option — pass a ``VectorStore`` instance, a config dict for
    # auto-creation, or ``None`` (in-memory retrieval via ``documents`` param).
    vector_store: VectorStore | dict[str, Any] | None = None

    remote: bool = False

    _preprocessor: Preprocessor = field(init=False, repr=False)
    _postprocessor: Postprocessor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.remote:
            for m in (self.embedder, self.reranker, self.vlm, self.asr):
                if m is not None:
                    m.remote()

        if self.caption_video:
            assert self.asr is not None, "To caption video, an `asr` model must be provided"

        self._preprocessor = Preprocessor(
            vlm=self.vlm,
            asr=self.asr,
            caption_video=self.caption_video,
            chunk_size=self.preprocess_chunk_size,
        )
        self._postprocessor = Postprocessor(vlm=self.vlm, asr=self.asr, caption_video=False)

        vs = self.vector_store
        if isinstance(vs, dict):
            qdrant_path: str = vs.pop("qdrant_path", "./qdrant_storage")
            collection_name: str = vs.pop("collection_name", "documents")
            qdrant_host: str = vs.pop("qdrant_host", "")
            qdrant_port: int = int(vs.pop("qdrant_port", 6333))
            self.vector_store = self._build_qdrant_vector_store(
                embedding=self.embed,
                qdrant_path=qdrant_path,
                collection_name=collection_name,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                **vs,
            )
        elif isinstance(vs, VectorStore):
            logger.warning(
                "Embedding model must match the one used to build the vector store:\n"
                "  Embedder model : %s\n"
                "  Embedder URL   : %s\n"
                "Nothing is checked explicitly — ensure they match.",
                self.embedder.model_name,
                self.embedder.base_url,
            )
        else:
            self.vector_store = InMemoryVectorStore(embedding=self.embed)
            logger.info("No vector_store provided — using in-memory store")

    @staticmethod
    def _build_qdrant_vector_store(
        embedding: Any,
        qdrant_path: str = "./qdrant_storage",
        collection_name: str = "documents",
        qdrant_host: str = "",
        qdrant_port: int = 6333,
        **kwargs,
    ) -> VectorStore:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if qdrant_host:
            client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                prefer_grpc=_qdrant_prefer_grpc(),
                timeout=_qdrant_client_timeout(),
            )
        else:
            client = QdrantClient(path=qdrant_path)

        test_vec = embedding.embed_query("")
        vector_size = len(test_vec)

        collections = retry_call(
            lambda: client.get_collections().collections,
            max_attempts=3,
            base_delay=2.0,
            connection_delay=5.0,
        )
        if not any(c.name == collection_name for c in collections):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                quantization_config=_qdrant_quantization_config(),
            )

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            **kwargs,
        )

    def __repr__(self) -> str:
        def _mn(m: Any) -> str:
            return m.model_name if hasattr(m, "model_name") else str(m)

        def _vs(vs: Any) -> str:
            if vs is None:
                return "(none)"
            name = type(vs).__qualname__
            if name == "InMemoryVectorStore":
                return f"{name}({self.embedder.model_name})"
            try:
                return f"{name}(collection={vs.collection_name})"  # type: ignore[attr-defined]
            except Exception:
                return name

        fields = [
            ("embedder", _mn(self.embedder)),
            ("reranker", _mn(self.reranker) if self.reranker else "(none)"),
            ("vlm", _mn(self.vlm) if self.vlm else "(none)"),
            ("asr", _mn(self.asr) if self.asr else "(none)"),
            ("caption_video", str(self.caption_video)),
            ("preprocess", str(self.preprocess)),
            ("remote", str(self.remote)),
            ("vector_store", _vs(self.vector_store)),
        ]
        body = "\n".join(f"  {k}={v}" for k, v in fields)
        return f"MultimodalRAG(\n{body}\n)"

    # -- shortcuts -------------------------------------------------------------

    @property
    def _embed_modalities(self) -> set[str]:
        return set(self.embedder.allowable_modalities)

    @cached_property
    def embed(self):
        model = self.embedder.model
        type_name = type(model).__qualname__
        if type_name in ("MultiModalEmbeddings", "MultiModalReranker"):
            return model
        raise TypeError(
            f"Embedder model has unexpected type {type(model).__module__}.{type(model).__qualname__}.\n"
            f"  Embedder model name:            {self.embedder.model_name}\n"
            f"  model_instantiation_class:      {self.embedder.model_instantiation_class}\n"
            f"  model_usage:                    {self.embedder.model_usage}\n\n"
            f"Expected a {MultiModalEmbeddings.__module__}.{MultiModalEmbeddings.__qualname__} "
            f"instance."
        )

    @cached_property
    def rank(self) -> MultiModalReranker:
        if self.reranker is None:
            raise AttributeError("No reranker configured.")
        return self.reranker.model  # type: ignore[return-value]

    # -- 1:1 passthrough delegates -----------------------------------------------

    @property
    def embed_documents(self):
        return self.embed.embed_documents

    @property
    def aembed_documents(self):
        return self.embed.aembed_documents

    @property
    def embed_query(self):
        return self.embed.embed_query

    @property
    def aembed_query(self):
        return self.embed.aembed_query

    @property
    def arerank(self):
        return self.rank.arerank

    @property
    def ascore(self):
        return self.rank.ascore

    @property
    def rerank(self):
        return self.rank.rerank

    @property
    def score(self):
        return self.rank.score

    # -- vector store management ------------------------------------------------

    @staticmethod
    def _to_documents(inputs: Sequence[str | dict[str, Any]]) -> list[Document]:
        docs = []
        for inp in inputs:
            if isinstance(inp, str):
                docs.append(Document(page_content=inp, metadata={}))
            elif isinstance(inp, dict):
                text = inp.get("text", "")
                meta = {k: v for k, v in inp.items() if k != "text"}
                docs.append(Document(page_content=text, metadata=meta))
            else:
                docs.append(Document(page_content=str(inp), metadata={}))
        return docs

    @staticmethod
    def _from_documents(docs: list[Document]) -> list[str | dict[str, Any]]:
        result: list[str | dict[str, Any]] = []
        for doc in docs:
            if doc.metadata:
                entry = dict(doc.metadata)
                entry["text"] = doc.page_content
                result.append(entry)
            else:
                result.append(doc.page_content)
        return result

    @staticmethod
    def _extract_doc(doc: Any) -> str | dict[str, Any]:
        if isinstance(doc, Document):
            if doc.metadata:
                entry = dict(doc.metadata)
                entry["text"] = doc.page_content
                return entry
            return doc.page_content
        return doc

    async def _preprocess_docs(
        self,
        documents: Sequence[str | dict[str, Any]],
    ) -> list[str | dict[str, Any]]:
        """Run the preprocessor unless the embedder already supports all modalities."""
        if not self.preprocess:
            docs = list(documents)
            # Even without preprocessing, omit docs with unprocessable audio.
            if "audio" not in self._embed_modalities and self.asr is None:
                kept: list[str | dict[str, Any]] = []
                omitted = 0
                for d in docs:
                    if isinstance(d, dict) and d.get("audio"):
                        media = _split_media(d)
                        has_other_media = bool(media["image"]) or bool(media["video"])
                        text = (d.get("text") or "").strip()
                        if not has_other_media and (not text or text.startswith("[Audio:")):
                            omitted += 1
                            continue
                        d = dict(d)
                        d.pop("audio", None)
                    kept.append(d)
                if omitted:
                    logger.warning(
                        "Omitting %d audio document(s): no ASR model and embedder does not support audio.",
                        omitted,
                    )
                docs = kept
            return docs
        return await self._preprocessor.acall(
            list(documents),
            target_modalities=self._embed_modalities,
        )

    def list_documents(self, limit: int = 50) -> list[tuple[str, dict[str, Any]]]:
        """Return stored documents for debugging.

        Returns ``[(id, {"text": ..., "image": ..., ...}), ...]`` where the
        dict is the reconstructed multimodal entry.
        """
        return sync_wrapper_safe(self.alist_documents, {"limit": limit})

    async def alist_documents(self, limit: int = 50) -> list[tuple[str, dict[str, Any]]]:
        vs = self.vector_store
        assert vs is not None and not isinstance(vs, dict)
        if hasattr(vs, "store"):
            # InMemoryVectorStore
            out = []
            for doc_id, entry in list(vs.store.items())[:limit]:
                doc: Document = entry["document"]
                meta = dict(doc.metadata)
                meta["text"] = doc.page_content
                out.append((doc_id, meta))
            return out
        # QdrantVectorStore – use the underlying client
        try:
            client = vs._client  # type: ignore[attr-defined]
            coll = vs.collection_name  # type: ignore[attr-defined]
            records, _ = client.scroll(coll, limit=limit, with_payload=True, with_vectors=False)
            out = []
            for rec in records:
                payload = rec.payload or {}
                meta = dict(payload.get("metadata", {}))
                meta["text"] = payload.get("page_content", "")
                out.append((str(rec.id), meta))
            return out
        except Exception as e:
            return [(f"error: {e}", {})]

    def add_to_vector_store(
        self,
        documents: Sequence[str | dict[str, Any]],
        deduplicate: bool = True,
        dedup_threshold: float | None = None,
        **kwargs,
    ) -> list[str]:
        """Sync wrapper around :meth:`aadd_to_vector_store`."""
        return sync_wrapper_safe(
            self.aadd_to_vector_store,
            dict(
                documents=documents,
                deduplicate=deduplicate,
                dedup_threshold=dedup_threshold,
                **kwargs,
            ),
        )

    def _resize_media_in_docs(self, docs: list[str | dict[str, Any]]) -> list[str | dict[str, Any]]:
        """Downscale embedded images and transcode videos in document dicts before storage.

        * Images are resized so width × height ≤ *max_pixels*.
        * Videos are transcoded to *fps* (default 1) and scaled so each
          frame fits within *max_pixels*.
        * HTTP(S) URLs are left untouched (the server fetches them).
        """
        mpk = self.embedder.mm_processor_kwargs
        max_px = mpk.get("max_pixels", 0)
        fps = mpk.get("fps", 1.0)
        if max_px <= 0:
            return docs

        out: list[str | dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                out.append(doc)
                continue
            d = dict(doc)
            for key in ("image", "video"):
                val = d.get(key)
                if not val:
                    continue
                urls = val if isinstance(val, list) else [val]
                resized: list[str] = []
                for url in urls:
                    if url.startswith(("http://", "https://")):
                        resized.append(url)
                        continue
                    if url.startswith("data:") and key != "video":
                        # Data URL image — resize in-memory
                        import re

                        m = re.match(r"data:([^;]+);base64,(.+)", url)
                        if m:
                            mime = m.group(1)
                            raw = base64.b64decode(m.group(2))
                        else:
                            raw = base64.b64decode(url.split(",", 1)[1])
                            mime = "image/png"
                        resized_raw = self.embed.input_conversion._resize_image(raw, mime, max_px)
                        b64 = base64.b64encode(resized_raw).decode("utf-8")
                        resized.append(f"data:{mime};base64,{b64}")
                        continue
                    # Local file path — keep as-is (already resized by _save_doc_media)
                    if url.startswith("file://"):
                        resized.append(url)
                        continue
                    # video data URL — pass through
                    if url.startswith("data:"):
                        # video data URL — pass through
                        resized.append(url)
                        continue
                    path = url  # bare path, no scheme
                    if key == "image":
                        with open(path, "rb") as f:
                            raw = f.read()
                        mime = _infer_mime(url)
                        resized_raw = self.embed.input_conversion._resize_image(raw, mime, max_px)
                        b64 = base64.b64encode(resized_raw).decode("utf-8")
                        resized.append(f"data:{mime};base64,{b64}")
                    else:
                        # Video — transcode with ffmpeg: fps + dynamic aspect-ratio-aware resize
                        import json
                        import subprocess as sp

                        vw = vh = 0
                        try:
                            probe = sp.run(
                                [
                                    "ffprobe",
                                    "-v",
                                    "error",
                                    "-select_streams",
                                    "v:0",
                                    "-show_entries",
                                    "stream=width,height",
                                    "-of",
                                    "json",
                                    path,
                                ],
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                            pinfo = json.loads(probe.stdout)
                            for s in pinfo.get("streams", []):
                                vw = int(s.get("width", 0))
                                vh = int(s.get("height", 0))
                                break
                        except Exception:
                            logger.debug("Suppressed exception", exc_info=True)
                        if vw > 0 and vh > 0 and vw * vh > max_px:
                            scale = (max_px / (vw * vh)) ** 0.5
                            new_w = max(2, (int(vw * scale) // 2) * 2)
                            new_h = max(2, (int(vh * scale) // 2) * 2)
                            scale = f"scale={new_w}:{new_h}"
                        else:
                            scale = "scale='min(720,iw)':'min(720,ih)':force_original_aspect_ratio=decrease"
                        proc = sp.run(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-i",
                                path,
                                "-vf",
                                f"fps={fps},{scale}",
                                "-f",
                                "mp4",
                                "-movflags",
                                "frag_keyframe+empty_moov",
                                "-vcodec",
                                "libx264",
                                "-preset",
                                "fast",
                                "-crf",
                                "28",
                                "-",
                            ],
                            capture_output=True,
                            timeout=300,
                        )
                        mime = "video/mp4"
                        b64 = base64.b64encode(proc.stdout).decode("utf-8")
                        resized.append(f"data:{mime};base64,{b64}")
                d[key] = resized if isinstance(val, list) else resized[0]
            out.append(d)
        return out

    def _split_audio_chunks(self, docs: list[str | dict[str, Any]]) -> list[str | dict[str, Any]]:
        """Split long audio-transcription documents into multiple chunks.

        When a document has an ``audio`` key and its text exceeds the
        embedder's chunk budget, the text is split into smaller pieces so
        that each chunk fits within the embedding window.  Every chunk
        retains the audio reference.
        """
        chunk_size = self.embedder.chunk_size
        chunk_overlap = self.embedder.chunk_overlap
        text_splitter = self.embedder.text_splitter

        def _over_budget(text: str) -> bool:
            if text_splitter is not None:
                return text_splitter.count_tokens(text) > chunk_size
            return len(text) > chunk_size

        def _split(text: str) -> list[str]:
            if text_splitter is not None:
                return text_splitter.split_text(text)
            # Character-based fallback (mirrors TokenTextSplitter.split_text:
            # 10% net-new tail merge + chunk_size//4 minimum-tail backfill).
            min_new = max(chunk_size // 10, 1)
            min_tail = max(chunk_size // 4, 1)
            result: list[str] = []
            start = 0
            prev_end = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    next_space = text.find(" ", end)
                    if next_space != -1 and next_space - end < chunk_size // 2:
                        end = next_space
                if result and end >= len(text):
                    new_content = len(text) - prev_end
                    if new_content < min_new:
                        tail = text[prev_end:end].strip()
                        if tail:
                            result[-1] = result[-1] + " " + tail
                        break
                    if end - start < min_tail:
                        start = max(0, end - min_tail)
                result.append(text[start:end].strip())
                if end >= len(text):
                    break
                prev_end = end
                start = end - chunk_overlap
                start = max(start, 0)
            return [c for c in result if c]

        out: list[str | dict[str, Any]] = []
        for doc in docs:
            if not isinstance(doc, dict) or not doc.get("audio"):
                out.append(doc)
                continue
            text = doc.get("text", "")
            if not text or not _over_budget(text):
                out.append(doc)
                continue

            chunks = _split(text)
            for i, chunk in enumerate(chunks):
                entry = dict(doc)
                entry["text"] = chunk
                entry["chunk_index"] = i
                out.append(entry)

        return out

    def _batch_find_duplicates(
        self,
        vs: VectorStore,
        embs: list[list[float]],
        threshold: float,
    ) -> list[bool]:
        """Return a bool for each embedding: True if a duplicate exists above *threshold*.

        For Qdrant this uses a single batched ``query_batch_points`` call
        (1 HTTP request) instead of N individual queries.  For the
        InMemoryVectorStore path it uses vectorised numpy cosine similarity.
        """
        if not embs:
            return []

        if hasattr(vs, "store"):
            # ── InMemoryVectorStore — vectorised numpy ──────────────────
            stored_vecs = [
                s.get("vector")
                for s in vs.store.values()
                if s.get("vector") is not None  # type: ignore[attr-defined]
            ]
            if not stored_vecs:
                return [False] * len(embs)
            query_matrix = np.array(embs, dtype=np.float32)  # N x D
            stored_matrix = np.array(stored_vecs, dtype=np.float32)  # M x D
            sims = cosine_sim(query_matrix, stored_matrix)  # N x M
            return [bool(np.any(row >= threshold)) for row in sims]
        else:
            # ── QdrantVectorStore — single batched request ───────────────
            from qdrant_client.models import QueryRequest

            client = vs._client  # type: ignore[attr-defined]
            coll = vs.collection_name  # type: ignore[attr-defined]
            vector_name = vs.vector_name  # type: ignore[attr-defined]

            responses = client.query_batch_points(
                collection_name=coll,
                requests=[
                    QueryRequest(
                        query=emb,
                        using=vector_name,
                        limit=1,
                        with_payload=False,
                        with_vector=False,
                        score_threshold=threshold,
                    )
                    for emb in embs
                ],
            )
            return [bool(resp.points) for resp in responses]

    async def aadd_to_vector_store(
        self,
        documents: Sequence[str | dict[str, Any]],
        deduplicate: bool = True,
        dedup_threshold: float | None = None,
        **kwargs,
    ) -> list[str]:
        """Embed and add documents to the vector store.

        Multimodal content (``image``, ``video``, ``audio`` keys) is
        preserved in the dict and **embedded together with the text** so
        that Qwen3-VL-Embedding can use the full visual information for
        similarity search — not just the text caption.

        Images are resized client-side **before** storage so that the
        vector store never holds the original full-resolution payload.

        Parameters
        ----------
        deduplicate:
            When enabled, each document's embedding is checked against
            existing vectors in the store.  Documents whose cosine
            similarity to the nearest neighbour exceeds
            *dedup_threshold* are skipped and logged.
        dedup_threshold:
            Cosine similarity threshold above which a document is
            considered a duplicate (0.0 – 1.0).  When ``None`` (the
            default), falls back to ``self.dedup_threshold``.
        """
        if dedup_threshold is None:
            dedup_threshold = self.dedup_threshold
        t0 = time.monotonic()

        # ── 0. Preprocess ───────────────────────────────────────────────────
        processed = await self._preprocess_docs(documents)
        if not processed:
            logger.verbose("  %.2fs add_vs  — no docs to add", time.monotonic() - t0)  # type: ignore[attr-defined]
            return []
        logger.verbose(  # type: ignore[attr-defined]
            "  %.2fs add_vs  — preprocess (%d docs)",
            time.monotonic() - t0,
            len(processed),
        )  # type: ignore[attr-defined]

        # ── 0d helper (used inside sub-batch loop) ────────────────────────
        def _replace_audio(d: dict[str, Any]) -> dict[str, Any]:
            if "audio" not in d:
                return d
            src = d.get("source", "")
            ref = f"file://{src}" if src and not src.startswith(("file://", "http://", "https://", "s3://")) else src
            d = dict(d)
            d["audio"] = ref if ref else d["audio"]
            return d

        # ── 1-3. Sub-batched resize → embed → dedup → upsert ─────────────
        # Media resize, audio splitting, and audio payload replacement are
        # done per sub-batch so large doc dicts (with image/video data URLs)
        # are released after each upsert rather than accumulating for the
        # full set.  This bounds peak memory for large ingests.
        vs = self.vector_store
        assert vs is not None and not isinstance(vs, dict)

        embed_batch_size = getattr(self.embed, "chunk_size", 64) or 64
        all_ids: list[str] = []
        t_embed_total = 0.0
        t_store_total = 0.0
        total_skipped = 0

        for sub in list_chunker(processed, embed_batch_size):
            # ── 0b. Resize media before storage ─────────────────────────────
            # Run in a thread pool — ffmpeg/probe subprocess calls would
            # otherwise block the event loop for up to 300s per video.
            loop = asyncio.get_running_loop()
            sub = await loop.run_in_executor(None, self._resize_media_in_docs, sub)

            # ── 0c. Split long audio transcriptions into chunks ─────────────
            sub = self._split_audio_chunks(sub)

            # ── 0d. Replace audio payloads with file references ─────────────
            sub = [_replace_audio(d) if isinstance(d, dict) else d for d in sub]

            # ── 1. Embed this sub-batch ──────────────────────────────────────
            t1 = time.monotonic()
            sub_embs = await self.embed.aembed_documents(sub)
            t_embed_total += time.monotonic() - t1

            # Guard against silent data loss: if the embedding API returns
            # fewer/more vectors than documents, zip() would silently
            # truncate.  Log at error level and align to the shorter length.
            if len(sub_embs) != len(sub):
                logger.error(
                    "Embedding count mismatch: %d docs → %d embeddings. "
                    "Truncating to %d — %d document(s) will be DROPPED.",
                    len(sub),
                    len(sub_embs),
                    min(len(sub), len(sub_embs)),
                    abs(len(sub) - len(sub_embs)),
                )
                min_len = min(len(sub), len(sub_embs))
                sub = sub[:min_len]
                sub_embs = sub_embs[:min_len]

            # ── 2. Build Document payloads ──────────────────────────────────
            sub_docs = self._to_documents(sub)

            # ── 2b. Deduplicate ─────────────────────────────────────────────
            if deduplicate and sub_embs:
                results = await loop.run_in_executor(
                    None,
                    self._batch_find_duplicates,
                    vs,
                    sub_embs,
                    dedup_threshold,
                )
                skipped = sum(results)
                total_skipped += skipped
                sub_docs = [d for d, dup in zip(sub_docs, results) if not dup]
                sub_embs = [e for e, dup in zip(sub_embs, results) if not dup]
                if skipped:
                    logger.info("Dedup: skipped %d of %d document(s)", skipped, len(sub))

            if not sub_docs:
                continue

            # ── 3. Store vectors + payloads ─────────────────────────────────
            t2 = time.monotonic()

            if hasattr(vs, "store"):
                # ── InMemoryVectorStore ──────────────────────────────────────
                batch_ids = await vs.aadd_documents(sub_docs, **kwargs)
                for doc_id, emb in zip(batch_ids, sub_embs):
                    vs.store[doc_id]["vector"] = emb
                all_ids.extend(batch_ids)
            else:
                # ── QdrantVectorStore ────────────────────────────────────────
                import uuid

                from qdrant_client.models import PointStruct

                client = vs._client  # type: ignore[attr-defined]
                coll = vs.collection_name  # type: ignore[attr-defined]
                vector_name = vs.vector_name  # type: ignore[attr-defined]

                points = []
                for doc, emb in zip(sub_docs, sub_embs):
                    doc_id = uuid.uuid4().hex
                    all_ids.append(doc_id)
                    points.append(
                        PointStruct(
                            id=doc_id,
                            vector={vector_name: emb} if vector_name else emb,
                            payload={
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            },
                        )
                    )
                client.upsert(collection_name=coll, points=points, **kwargs)

            t_store_total += time.monotonic() - t2
            # sub_embs / sub_docs fall out of scope here — released before the
            # next sub-batch starts.

        if total_skipped:
            logger.info("Dedup total: skipped %d document(s)", total_skipped)

        logger.verbose(  # type: ignore[attr-defined]
            "  %.2fs add_vs  — embed (%d vectors), store (%d docs)  [total %.2fs]",
            t_embed_total,
            len(all_ids) + total_skipped,
            len(all_ids),
            time.monotonic() - t0,
        )  # type: ignore[attr-defined]
        return all_ids

    # -- retrieval --------------------------------------------------------------

    def _similarity_retrieve(
        self,
        query_emb: list[float],
        doc_embs: list[list[float]],
        documents: list,
        k: int,
    ) -> list[tuple[Any, float]]:
        scores = cosine_sim(
            np.array(query_emb, dtype=np.float32).reshape(1, -1),
            np.array(doc_embs, dtype=np.float32),
        )[0]
        top_indices = np.argsort(scores)[-k:][::-1]
        return [(documents[i], float(scores[i])) for i in top_indices]

    def _rerank_results(
        self,
        query: str | dict[str, Any],
        results: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        return sync_wrapper_safe(self._arerank_results, {"query": query, "results": results})

    async def _arerank_results(
        self,
        query: str | dict[str, Any],
        results: list[tuple[Any, float]],
        reranker_top_k: int = 3,
    ) -> list[tuple[Any, float]]:
        reranked = await self.rank.arerank(query, [self._extract_doc(d) for d, _ in results])
        # reranked[0] is a list of result dicts, e.g.
        # [{"index": 2, "relevance_score": 0.95, ...},
        #  {"index": 0, "relevance_score": 0.87, ...}]
        # Map each index back to its reranker score, then attach
        # both the embedding score and the reranker score to each doc.
        score_by_idx: dict[int, float] = {}
        for r in reranked[0] if reranked else []:
            score_by_idx[r.get("index", -1)] = r.get("relevance_score", r.get("score", 0.0))
        paired: list[tuple[Any, float]] = []
        for i, (d, emb_score) in enumerate(results):
            rerank_score = score_by_idx.get(i, 0.0)
            if isinstance(d, dict):
                d["_embedding_score"] = round(emb_score, 4)
                d["_reranker_score"] = round(rerank_score, 4)
            paired.append((d, rerank_score))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:reranker_top_k]

    def retrieve(
        self,
        query: str | dict[str, Any],
        documents: list[str | dict[str, Any]] | None = None,
        top_k: int = 10,
        use_reranker: bool = True,
        reranker_top_k: int = 3,
        query_vector: list[float] | None = None,
        need_media: bool | None = None,
    ) -> list[tuple[Any, float]]:
        """Sync wrapper around :meth:`aretrieve`."""
        return sync_wrapper_safe(
            self.aretrieve,
            {
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "reranker_top_k": reranker_top_k,
                "query_vector": query_vector,
                "need_media": need_media,
            },
        )

    async def aretrieve(
        self,
        query: str | dict[str, Any],
        documents: list[str | dict[str, Any]] | None = None,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
        query_vector: list[float] | None = None,
        need_media: bool | None = None,
    ) -> list[tuple[Any, float]]:

        # Auto-compute need_media: base64 media payloads are needed when the
        # reranker will consume them.  (VLM / base-LLM-vision cases are
        # handled by the caller — they pass need_media=True explicitly.)
        if need_media is None:
            need_media = use_reranker and self.reranker is not None

        if documents is not None:
            if query_vector is not None:
                query_emb = query_vector
            else:
                query_emb = await self.aembed_query(query)
            doc_embs = await self.aembed_documents(documents)
            results = self._similarity_retrieve(query_emb, doc_embs, documents, top_k)
        else:
            vs = self.vector_store
            assert vs is not None and not isinstance(vs, dict)

            if isinstance(query, dict):
                # Multimodal query (text + image/video/audio) — embed and
                # search by vector directly.
                if query_vector is not None:
                    query_emb = query_vector
                else:
                    query_emb = await self.aembed_query(query)
                docs_and_scores = await vs.asimilarity_search_with_score_by_vector(  # type: ignore[attr-defined]
                    query_emb,
                    top_k,
                    need_media=need_media,
                )
            else:
                docs_and_scores = await vs.asimilarity_search_with_relevance_scores(
                    query,
                    k=top_k,
                    need_media=need_media,
                )
            results = [(self._extract_doc(doc), score) for doc, score in docs_and_scores]

        if use_reranker and self.reranker is None:
            logger.warning(
                "Reranker requested (use_reranker=True) but no reranker model is "
                "configured — returning top embedding results."
            )
        elif self.reranker is not None and use_reranker and results:
            results = await self._arerank_results(query, results, reranker_top_k)
        elif use_reranker and results and len(results) > reranker_top_k:
            # Reranker unavailable — trim to reranker_top_k for consistency
            # with what the caller expected.
            results = results[:reranker_top_k]

        return results

    # -- context formatting ----------------------------------------------------

    def _format_context(self, documents: list) -> str:
        parts = []
        for doc in documents:
            if isinstance(doc, str):
                text = doc
                ref = ""
            elif isinstance(doc, dict):
                text = doc.get("text", "")
                src = doc.get("source")
                pg = doc.get("page")
                ref_parts = []
                if src:
                    ref_parts.append(f"Source: {_pvc_to_http_url(src)}")
                if pg:
                    ref_parts.append(f"Page: {pg}")
                ref = "[" + ", ".join(ref_parts) + "]\n" if ref_parts else ""
            elif isinstance(doc, Document):
                text = doc.page_content
                meta = doc.metadata or {}
                ref_parts = []
                if meta.get("source"):
                    ref_parts.append(f"Source: {_pvc_to_http_url(meta['source'])}")
                if meta.get("page"):
                    ref_parts.append(f"Page: {meta['page']}")
                ref = "[" + ", ".join(ref_parts) + "]\n" if ref_parts else ""
            else:
                parts.append(str(doc))
                continue
            parts.append(ref + text if ref else text)
        return "\n\n".join(parts)

    @staticmethod
    def _build_multimodal_content(
        postprocessed: list[str | dict[str, Any]],
        query: str | dict[str, Any],
    ) -> str | dict[str, Any] | list[dict[str, Any]]:
        """Build user message content from postprocessed docs + query.

        Returns a plain string when all content is text-only (backward
        compatible), otherwise returns an OpenAI-compatible list of content
        parts that can include ``image_url`` / ``video_url`` / ``audio_url``
        alongside text.
        """
        has_media = any(isinstance(d, dict) and any(k in d for k in ("image", "video", "audio")) for d in postprocessed)
        query_has_media = isinstance(query, dict) and any(k in query for k in ("image", "video", "audio"))

        if not has_media and not query_has_media:
            # All text — flatten to a single string (backward compatible)
            context = "\n\n".join(d if isinstance(d, str) else d.get("text", "") for d in postprocessed if d)
            if not context:
                return query
            if isinstance(query, str):
                return f"Context:\n{context}\n\nQuery: {query}"
            q = dict(query)
            q["text"] = f"Context:\n{context}\n\nQuery: {q.get('text', '')}"
            return q

        # -- multimodal: build content parts array ---------------------------
        content_parts: list[dict[str, Any]] = []

        if isinstance(query, dict) and query.get("text"):
            content_parts.append({"type": "text", "text": "Context documents:"})
        else:
            content_parts.append({"type": "text", "text": "Context documents:"})

        for doc in postprocessed:
            if isinstance(doc, str):
                if doc:
                    content_parts.append({"type": "text", "text": doc})
                continue

            text = doc.get("text", "")
            if text:
                src = doc.get("source", "")
                pg = doc.get("page", "")
                ref = ""
                if src:
                    ref = f"[Source: {_pvc_to_http_url(src)}"
                    if pg:
                        ref += f", Page: {pg}"
                    ref += "]\n"
                content_parts.append({"type": "text", "text": ref + text})

            for url in _as_url_list(doc.get("image", [])):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _file_url_to_data_url(url)},
                    }
                )
            for url in _as_url_list(doc.get("video", [])):
                content_parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": _file_url_to_data_url(url)},
                    }
                )
            for url in _as_url_list(doc.get("audio", [])):
                content_parts.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _file_url_to_data_url(url)},
                    }
                )

        # Append the query
        if isinstance(query, str):
            content_parts.append({"type": "text", "text": f"Query: {query}"})
        else:
            q_text = query.get("text", "")
            if q_text:
                content_parts.append({"type": "text", "text": f"Query: {q_text}"})
            for url in _as_url_list(query.get("image", [])):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _file_url_to_data_url(url)},
                    }
                )
            for url in _as_url_list(query.get("video", [])):
                content_parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": _file_url_to_data_url(url)},
                    }
                )
            for url in _as_url_list(query.get("audio", [])):
                content_parts.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": _file_url_to_data_url(url)},
                    }
                )

        return content_parts


# ---------------------------------------------------------------------------
# RAG + LLM  (generation on top of MultimodalRAG)
# ---------------------------------------------------------------------------


class MultiModalRAGSystem:
    """Complete RAG system with an LLM for generation.

    Holds a :class:`MultimodalRAG` instance internally and delegates all
    retrieval / storage / modality-conversion to it.  Generation methods
    (``generate`` / ``agenerate``) combine retrieval with the LLM.

    Parameters
    ----------
    llm:
        Chat model used for generation (and optional routing).
    embedder, reranker, vlm, asr, caption_video, preprocess,
    preprocess_chunk_size, vector_store, remote:
        Forwarded directly to :class:`MultimodalRAG`.
    """

    def __init__(
        self,
        llm: ChatModel,
        embedder: EmbeddingModel,
        reranker: RerankerModel | None = None,
        vlm: ChatModel | None = None,
        asr: VoiceModel | None = None,
        caption_video: bool = False,
        preprocess: bool = True,
        preprocess_chunk_size: int = 128,
        vector_store: VectorStore | dict[str, Any] | None = None,
        remote: bool = False,
    ):
        self.llm = llm
        if remote:
            self.llm.remote()
        if caption_video:
            assert asr is not None, "To caption video, an `asr` model must be provided"
        self._rag = MultimodalRAG(
            embedder=embedder,
            reranker=reranker,
            vlm=vlm,
            asr=asr,
            caption_video=caption_video,
            preprocess=preprocess,
            preprocess_chunk_size=preprocess_chunk_size,
            vector_store=vector_store,
            remote=remote,
        )

    def __repr__(self) -> str:
        def _mn(m: Any) -> str:
            return m.model_name if hasattr(m, "model_name") else str(m)

        fields = [
            ("llm", _mn(self.llm)),
            ("rag", repr(self._rag).replace("\n", "\n  ")),
        ]
        body = "\n".join(f"  {k}={v}" for k, v in fields)
        return f"MultiModalRAGSystem(\n{body}\n)"

    # -- model property delegates -----------------------------------------------

    @property
    def embedder(self):
        return self._rag.embedder

    @embedder.setter
    def embedder(self, val):
        self._rag.embedder = val

    @property
    def reranker(self):
        return self._rag.reranker

    @reranker.setter
    def reranker(self, val):
        self._rag.reranker = val

    @property
    def vlm(self):
        return self._rag.vlm

    @vlm.setter
    def vlm(self, val):
        self._rag.vlm = val

    @property
    def asr(self):
        return self._rag.asr

    @asr.setter
    def asr(self, val):
        self._rag.asr = val

    @property
    def caption_video(self):
        return self._rag.caption_video

    @caption_video.setter
    def caption_video(self, val):
        self._rag.caption_video = val

    @property
    def preprocess(self):
        return self._rag.preprocess

    @preprocess.setter
    def preprocess(self, val):
        self._rag.preprocess = val

    @property
    def preprocess_chunk_size(self):
        return self._rag.preprocess_chunk_size

    @preprocess_chunk_size.setter
    def preprocess_chunk_size(self, val):
        self._rag.preprocess_chunk_size = val

    @property
    def vector_store(self):
        return self._rag.vector_store

    @vector_store.setter
    def vector_store(self, val):
        self._rag.vector_store = val

    @property
    def remote(self):
        return self._rag.remote

    @remote.setter
    def remote(self, val):
        self._rag.remote = val

    @property
    def _preprocessor(self):
        return self._rag._preprocessor

    @property
    def _postprocessor(self):
        return self._rag._postprocessor

    @property
    def _embed_modalities(self):
        return self._rag._embed_modalities

    @property
    def embed(self):
        return self._rag.embed

    @property
    def rank(self):
        return self._rag.rank

    @property
    def embed_documents(self):
        return self._rag.embed_documents

    @property
    def aembed_documents(self):
        return self._rag.aembed_documents

    @property
    def embed_query(self):
        return self._rag.embed_query

    @property
    def aembed_query(self):
        return self._rag.aembed_query

    @property
    def arerank(self):
        return self._rag.arerank

    @property
    def ascore(self):
        return self._rag.ascore

    @property
    def rerank(self):
        return self._rag.rerank

    @property
    def score(self):
        return self._rag.score

    # -- LLM-specific -----------------------------------------------------------

    @property
    def _llm_modalities(self) -> set[str]:
        return set(self.llm.allowable_modalities) if hasattr(self.llm, "allowable_modalities") else {"text"}

    @property
    def chat(self):
        return self.llm.llm_chat_function_call

    @property
    def achat(self):
        return self.llm.llm_async_chat_function_call

    # -- method delegates -------------------------------------------------------

    def add_to_vector_store(self, documents, **kwargs):
        return self._rag.add_to_vector_store(documents, **kwargs)

    async def aadd_to_vector_store(self, documents, **kwargs):
        return await self._rag.aadd_to_vector_store(documents, **kwargs)

    def retrieve(self, query, documents=None, top_k=10, use_reranker=True, reranker_top_k=3, query_vector=None):
        return self._rag.retrieve(query, documents, top_k, use_reranker, reranker_top_k, query_vector=query_vector)

    async def aretrieve(self, query, documents=None, top_k=10, use_reranker=False, reranker_top_k=3, query_vector=None):
        return await self._rag.aretrieve(
            query, documents, top_k, use_reranker, reranker_top_k, query_vector=query_vector
        )

    def list_documents(self, limit=50):
        return self._rag.list_documents(limit)

    async def alist_documents(self, limit=50):
        return await self._rag.alist_documents(limit)

    # -- query routing (LLM decides if RAG is needed) -------------------------

    _ROUTE_PROMPT = (
        "You are a router. Determine if you need to retrieve external knowledge "
        "to answer the user's query accurately. Reply with exactly one word: "
        "YES or NO."
    )

    def _needs_rag(self, query: str | dict[str, Any]) -> bool:
        return sync_wrapper_safe(self._aneeds_rag, {"query": query})

    async def _aneeds_rag(self, query: str | dict[str, Any]) -> bool:
        query_text = query if isinstance(query, str) else query.get("text", str(query))
        messages = [
            {"role": "system", "content": self._ROUTE_PROMPT},
            {"role": "user", "content": query_text},
        ]
        reply = await self.llm.llm_async_chat_function_call(messages)
        answer = reply.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    # -- RAG generation ---------------------------------------------------------

    def generate(
        self,
        query: str | dict[str, Any],
        documents: list[str | dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        top_k: int = 10,
        use_vlm: bool = True,
        route: bool = False,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
        **llm_kwargs,
    ) -> str:
        return sync_wrapper_safe(
            self.agenerate,
            dict(
                query=query,
                documents=documents,
                system_prompt=system_prompt,
                top_k=top_k,
                use_vlm=use_vlm,
                route=route,
                use_reranker=use_reranker,
                reranker_top_k=reranker_top_k,
                **llm_kwargs,
            ),
        )

    async def agenerate(
        self,
        query: str | dict[str, Any],
        documents: list[str | dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        top_k: int = 10,
        use_vlm: bool = True,
        route: bool = False,
        use_reranker: bool = True,
        reranker_top_k: int = 3,
        **llm_kwargs,
    ) -> str:
        t0 = time.monotonic()

        if route:
            t_r = time.monotonic()
            needs_rag = await self._aneeds_rag(query)
            if not needs_rag:
                direct_messages: list[dict[str, Any]] = []
                if system_prompt:
                    direct_messages.append({"role": "system", "content": system_prompt})
                direct_messages.append({"role": "user", "content": query})
                response = await self.achat(direct_messages, **llm_kwargs)
                logger.verbose(  # type: ignore[attr-defined]
                    "%.2fs route(llm)  — direct answer (RAG skipped)",
                    time.monotonic() - t0,
                )  # type: ignore[attr-defined]
                return response.choices[0].message.content
            logger.verbose("%.2fs route(llm)  → RAG needed", time.monotonic() - t_r)  # type: ignore[attr-defined]

        t_r = time.monotonic()
        # Base64 media payloads are needed when the reranker, VLM, or a
        # vision-capable base LLM will consume them.
        llm_mods = self._llm_modalities
        need_media = (
            (use_reranker and self._rag.reranker is not None)
            or (use_vlm and self._rag.vlm is not None)
            or bool(llm_mods & {"image", "video"})
        )
        retrieved = await self._rag.aretrieve(
            query,
            documents,
            top_k,
            use_reranker=use_reranker,
            reranker_top_k=reranker_top_k,
            need_media=need_media,
        )
        retrieved_docs = [d for d, _ in retrieved]
        logger.verbose(  # type: ignore[attr-defined]
            "%.2fs retrieve  — %d docs (top_k=%s, reranker=%s%s)",
            time.monotonic() - t_r,
            len(retrieved_docs),
            top_k,
            "yes" if self._rag.reranker and use_reranker else "no",
            f", reranker_top_k={reranker_top_k}" if self._rag.reranker and use_reranker else "",
        )  # type: ignore[attr-defined]

        needs_llm_conversion = (
            use_vlm
            and (self._rag.vlm is not None or self._rag.asr is not None)
            and any(isinstance(d, dict) and any(k in d for k in ("image", "video", "audio")) for d in retrieved_docs)
        )

        t_p = time.monotonic()
        if needs_llm_conversion:
            postprocessed = await self._rag._postprocessor.acall(
                retrieved_docs,
                llm_modalities=self._llm_modalities,
                query=query,
            )
            user_content = self._rag._build_multimodal_content(postprocessed, query)
            logger.verbose(  # type: ignore[attr-defined]
                "%.2fs postproc  — VLM/ASR conversion (%d docs with media)",
                time.monotonic() - t_p,
                sum(
                    1
                    for d in retrieved_docs
                    if isinstance(d, dict) and any(k in d for k in ("image", "video", "audio"))
                ),
            )  # type: ignore[attr-defined]
        else:
            context = self._rag._format_context(retrieved_docs)
            if context:
                if isinstance(query, str):
                    user_content = f"Context:\n{context}\n\nQuery: {query}"
                else:
                    q = dict(query)
                    q["text"] = f"Context:\n{context}\n\nQuery: {q.get('text', '')}"
                    user_content = q
            else:
                user_content = query
            logger.verbose(  # type: ignore[attr-defined]
                "%.2fs postproc  — text-only (skipped VLM/ASR)", time.monotonic() - t_p
            )  # type: ignore[attr-defined]

        t_g = time.monotonic()
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif user_content:
            messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the retrieved context to answer the user's query. "
                    "When you reference specific documents, cite their source and page number if available.",
                }
            )
        messages.append({"role": "user", "content": user_content})
        response = await self.achat(messages, **llm_kwargs)
        logger.verbose(  # type: ignore[attr-defined]
            "%.2fs llm       — generation (%d tokens?)  [total %.2fs]",
            time.monotonic() - t_g,
            len(response.choices[0].message.content.split()),
            time.monotonic() - t0,
        )  # type: ignore[attr-defined]
        return response.choices[0].message.content
