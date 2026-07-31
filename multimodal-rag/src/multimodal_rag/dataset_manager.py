"""
Dataset manager for multimodal RAG.

Handles dataset creation, file uploads, text input, and Qdrant vector storage.
Each dataset is a Qdrant collection on a shared Qdrant instance, with
uploaded files stored on a PVC at ``/data/datasets/<name>/files/``.
"""

import base64
import contextlib
import hashlib
import json
import mimetypes
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

try:
    import fcntl  # Unix-only; k8s pods are Linux.

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows dev only
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False

from multimodal_rag.input_processing import (
    ArchiveProcessor,
    CodeProcessor,
    EbookProcessor,
    HTMLProcessor,
    ImageProcessor,
    JSONProcessor,
    LogProcessor,
    NotebookProcessor,
    OfficeProcessor,
    PDFProcessor,
    TableProcessor,
    TextProcessor,
    VideoProcessor,
    XMLProcessor,
    YAMLProcessor,
)
from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.utils.general_tools import retry_call
from multimodal_rag.utils.logging_utils import logging

logger = logging.getLogger(__name__)

# When True, get_dataset()/list_datasets() skip the per-call Qdrant count
# sync (which writes meta.json on every read). Counts are still maintained
# incrementally via _increment_count/_decrement_count. The scale chart sets
# this to avoid cross-replica meta.json write races and cut Qdrant load.
_DEFER_COUNT_SYNC = os.environ.get("RAG_DEFER_COUNT_SYNC", "false").lower() in (
    "true",
    "1",
    "yes",
)

# Optional Redis backend for cross-pod dataset existence caching.
# When a dataset is created on pod A, pod B's NFS client cache may not
# see meta.json for several seconds.  Redis provides an immediate
# existence signal so pod B can retry _read_meta instead of returning 404.
_REDIS_URL = os.environ.get("REDIS_URL", "")
_redis_client: Any = None
_redis_client_lock = threading.Lock()

# How many times to retry _read_meta when Redis says the dataset exists
# but NFS hasn't propagated yet.
_DATASET_EXIST_RETRY_COUNT = 4
_DATASET_EXIST_RETRY_DELAY = 0.5  # seconds


def _get_redis() -> Any:
    """Lazily build a Redis client, cached for the process lifetime."""
    global _redis_client
    if not _REDIS_URL:
        return None
    if _redis_client is not None:
        return _redis_client
    with _redis_client_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis  # type: ignore[import-untyped]

            _redis_client = redis.from_url(_REDIS_URL, socket_timeout=2.0, socket_connect_timeout=2.0)
            _redis_client.ping()
            logger.info("Dataset existence cache: Redis backend connected at %s", _REDIS_URL)
        except Exception as exc:
            logger.warning("Dataset existence cache: Redis unavailable (%s); falling back to NFS-only", exc)
            _redis_client = None
    return _redis_client


def _dataset_exist_key(name: str) -> str:
    return f"dataset_exists:{name}"


def _dataset_exist_set(name: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.set(_dataset_exist_key(name), "1", ex=86400)  # 24h TTL
        except Exception:
            pass


def _dataset_exist_delete(name: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.delete(_dataset_exist_key(name))
        except Exception:
            pass


def _dataset_exist_check(name: str) -> bool:
    """Return True if Redis says the dataset exists (NFS may lag)."""
    r = _get_redis()
    if r is None:
        return False
    try:
        return r.exists(_dataset_exist_key(name)) > 0
    except Exception:
        return False


# Fallback thread locks for platforms without fcntl (Windows dev only).
_fallback_locks: dict[str, threading.Lock] = {}
_fallback_locks_guard = threading.Lock()


@contextlib.contextmanager
def _cross_process_lock(lock_path: Path):
    """Cross-process + cross-thread advisory file lock.

    Uses ``fcntl.flock(LOCK_EX)`` on a sidecar ``.lock`` file so that
    multiple pods sharing the RWX PVC serialize read→modify→write cycles
    on dataset metadata and the dedup hash index. On non-unix platforms
    falls back to a per-path ``threading.Lock`` (in-process only).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_FCNTL:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    else:  # pragma: no cover - Windows dev only
        key = str(lock_path)
        with _fallback_locks_guard:
            lk = _fallback_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                _fallback_locks[key] = lk
        with lk:
            yield


# Supported file extensions mapped to a media type label
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"})
_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"})
_AUDIO_EXTS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"})
_TEXT_EXTS = frozenset({".txt", ".md"})
_TABLE_EXTS = frozenset({".csv", ".tsv", ".xlsx", ".xls", ".ods"})
_CODE_EXTS = frozenset(
    {
        ".py",
        ".pyw",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".java",
        ".cpp",
        ".cxx",
        ".cc",
        ".c",
        ".h",
        ".hpp",
        ".hxx",
        ".cs",
        ".rb",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".php",
        ".r",
        ".R",
        ".sh",
        ".bash",
        ".zsh",
    }
)
_OFFICE_EXTS = frozenset({".docx", ".pptx", ".odt", ".odp"})
_HTML_EXTS = frozenset({".html", ".htm"})
_XML_EXTS = frozenset({".xml"})
_YAML_EXTS = frozenset({".yaml", ".yml"})
_NOTEBOOK_EXTS = frozenset({".ipynb"})
_EBOOK_EXTS = frozenset({".epub"})
_LOG_EXTS = frozenset({".log", ".txt.log"})
_ARCHIVE_EXTS = frozenset({".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz2", ".txz", ".rar"})

# ── Preprocessing limits for files stored on PVC ─────────────────────────
_PVC_IMAGE_MAX_PIXELS: int = 1920 * 1080  # 2,073,600
_PVC_VIDEO_MAX_PIXELS: int = 1280 * 720  # 921,600
_PVC_VIDEO_MAX_FPS: float = 24.0

# Target frame count per video segment for *short* clips.  Short clips are
# sampled above the configured fps (toward this many frames per segment)
# instead of always falling back to ~1 frame/s; the configured fps remains a
# floor.  Set ``VIDEO_TARGET_FRAMES=0`` to restore the legacy fixed-fps
# behaviour.  Kept out of ``mm_processor_kwargs`` (which is forwarded to the
# embedding server) so the server never sees this client-side knob.
_VIDEO_TARGET_FRAMES: int = int(os.environ.get("VIDEO_TARGET_FRAMES", "30"))

# Audio files above this raw size (bytes) are truncated to avoid exceeding
# the embedding endpoint's JSON payload limit (~32 MiB). Base64 adds ~33%
# overhead, so ~20 MiB raw → ~27 MiB payload, leaving comfortable margin.
# Duration to truncate oversized audio files to (seconds).
_PVC_AUDIO_MAX_BYTES: int = 5 * 1024 * 1024

# Type-specific chunk size overrides for structured data (code, JSON, XML, YAML).
# These benefit from larger context windows to keep definitions intact.
# The actual chunk_size/chunk_overlap values come from the embedder model's
# ``code_chunk_size`` / ``code_chunk_overlap`` fields.
_STRUCTURED_TYPES: frozenset[str] = frozenset({"code", "json", "xml", "yaml"})


def _classify_file(path: str) -> str:
    """Return file type: ``'pdf'``, ``'image'``, ``'video'``, ``'audio'``,
    ``'text'``, ``'json'``, ``'table'``, ``'code'``, ``'office'``,
    ``'html'``, ``'xml'``, ``'yaml'``, ``'notebook'``, ``'ebook'``,
    ``'log'``, or ``'unknown'``."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _TEXT_EXTS:
        return "text"
    if ext == ".json":
        return "json"
    if ext in _TABLE_EXTS:
        return "table"
    if ext in _CODE_EXTS:
        return "code"
    if ext in _OFFICE_EXTS:
        return "office"
    if ext in _HTML_EXTS:
        return "html"
    if ext in _XML_EXTS:
        return "xml"
    if ext in _YAML_EXTS:
        return "yaml"
    if ext in _NOTEBOOK_EXTS:
        return "notebook"
    if ext in _EBOOK_EXTS:
        return "ebook"
    if ext in _LOG_EXTS:
        return "log"
    return "unknown"


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://", "s3://"))


def _get_s3_client() -> Any:
    """Return a configured boto3 S3 client.

    Reads connection details from environment variables so the same code
    works against AWS S3 or an on-cluster MinIO instance::

        S3_ENDPOINT_URL       — custom endpoint (e.g. http://minio.minio.svc.cluster.local:9000)
        S3_ACCESS_KEY_ID      — access key
        S3_SECRET_ACCESS_KEY  — secret key

    When ``S3_ENDPOINT_URL`` is unset the default boto3 credential chain
    (IAM roles, ~/.aws/credentials, etc.) is used.
    """
    try:
        import boto3
    except ImportError:
        raise ValueError("boto3 is required for S3 URLs. Install with: pip install boto3")

    endpoint = os.environ.get("S3_ENDPOINT_URL") or None
    access_key = os.environ.get("S3_ACCESS_KEY_ID") or None
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY") or None

    kwargs: dict[str, Any] = {}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key

    return boto3.client("s3", **kwargs)


def _download_s3(s3_url: str, timeout: int = 120) -> str:
    """Download a file from S3 to a temp location and return the local path."""
    import tempfile
    from urllib.parse import urlparse

    parsed = urlparse(s3_url)
    bucket = parsed.hostname  # s3://bucket/key → hostname = bucket
    key = parsed.path.lstrip("/")

    def _fetch() -> bytes:
        s3 = _get_s3_client()
        resp = s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    try:
        content = retry_call(_fetch, max_attempts=3, base_delay=2.0, connection_delay=10.0)
    except Exception as e:
        raise ValueError(f"Failed to download {s3_url}: {e}")

    basename = Path(key).name or "download"
    suffix = Path(basename).suffix or ""
    stem = Path(basename).stem or "file"

    tmp = tempfile.NamedTemporaryFile(prefix=f"{stem}_", suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp_path = tmp.name
    finally:
        tmp.close()

    return tmp_path


def _is_s3_directory_url(url: str) -> bool:
    """Return ``True`` if *url* is an S3 URL that should be treated as a
    prefix (directory) rather than a single file.

    The heuristic: ``s3://bucket/key/`` (trailing slash) or
    ``s3://bucket/prefix`` (no detectable file extension).
    """
    if not url.startswith("s3://"):
        return False
    from urllib.parse import urlparse

    parsed = urlparse(url)
    key = parsed.path.lstrip("/")
    if not key:
        return True  # just s3://bucket — list the whole bucket
    # Trailing slash → directory
    if key.endswith("/"):
        return True
    # No extension → likely a prefix
    return "." not in key.rsplit("/", 1)[-1]


def _list_s3_prefix(s3_url: str) -> list[str]:
    """List all supported-file-type objects under an S3 prefix.

    Returns full ``s3://bucket/key`` URLs for every object whose extension
    is recognised by :func:`_classify_file`.  Directories (common prefixes)
    are **not** recursed — only the immediate level is listed.
    """
    from urllib.parse import urlparse

    parsed = urlparse(s3_url)
    bucket = parsed.hostname
    prefix = parsed.path.lstrip("/")

    def _list() -> list[str]:
        s3 = _get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        result: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Skip "directory" markers
                if key.endswith("/"):
                    continue
                # Only include supported file types
                if _classify_file(key) != "unknown":
                    result.append(f"s3://{bucket}/{key}")
        return result

    try:
        urls = retry_call(_list, max_attempts=3, base_delay=2.0, connection_delay=10.0)
    except Exception as e:
        raise ValueError(f"Failed to list S3 prefix {s3_url}: {e}")

    return urls


def _expand_urls(raw_urls: list[str]) -> list[str]:
    """Expand a list of URLs, resolving S3 directory prefixes to
    individual file URLs.

    Non-S3 URLs and S3 file URLs are passed through unchanged.
    """
    expanded: list[str] = []
    for url in raw_urls:
        if url.startswith("s3://") and _is_s3_directory_url(url):
            expanded.extend(_list_s3_prefix(url))
        else:
            expanded.append(url)
    return expanded


def _download_url(url: str, timeout: int = 120) -> str:
    """Download a remote file to a temp location and return the local path.

    Supports HTTP(S) and S3 URLs.  The temp file retains the URL's basename
    and extension so that ``_classify_file`` can identify the type correctly.
    """
    if url.startswith("s3://"):
        return _download_s3(url, timeout=timeout)

    import tempfile
    from urllib.parse import urlparse

    import httpx

    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0), follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to download {url}: {e}")

    parsed = urlparse(url)
    basename = Path(parsed.path).name or "download"
    suffix = Path(basename).suffix or ""
    stem = Path(basename).stem or "file"

    tmp = tempfile.NamedTemporaryFile(prefix=f"{stem}_", suffix=suffix, delete=False)
    try:
        tmp.write(response.content)
        tmp_path = tmp.name
    finally:
        tmp.close()

    return tmp_path


# ---------------------------------------------------------------------------
# Password helpers (PBKDF2-SHA256 with random salt)
# ---------------------------------------------------------------------------


_PBKDF2_ITERATIONS = 600_000  # OWASP 2023 minimum for PBKDF2-SHA256


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"{salt}${h.hex()}"


def _check_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
        computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS).hex()
        return secrets.compare_digest(h, computed)
    except (ValueError, AttributeError):
        return False


def _fix_source(chunks: list, orig_name: str, stored_path: str) -> None:
    """Keep ``source`` as the stored path and add ``original_source`` with
    the original basename, for every dict chunk that has a ``source`` key."""
    for chunk in chunks:
        if isinstance(chunk, dict) and "source" in chunk:
            chunk.setdefault("original_source", orig_name)
            chunk["source"] = stored_path


# ---------------------------------------------------------------------------
# PVC preprocessing helpers
# ---------------------------------------------------------------------------


def _preprocess_image_file(path: Path, max_pixels: int = _PVC_IMAGE_MAX_PIXELS) -> Path:
    """Downscale an image on the PVC so width × height ≤ *max_pixels*.

    Aspect ratio is preserved.  Images already within the limit are
    returned unchanged.  A new ``*_preprocessed`` file is created
    alongside the original when resizing is needed.
    """
    from io import BytesIO

    from PIL import Image

    from multimodal_rag.input_processing.image_processor import _resize_image

    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"

    img = Image.open(BytesIO(raw))
    if img.width * img.height <= max_pixels:
        return path

    resized = _resize_image(raw, mime, max_pixels)
    preproc = path.with_stem(path.stem + "_preprocessed")
    preproc.write_bytes(resized)
    return preproc


def _preprocess_video_file(
    path: Path,
    max_pixels: int = _PVC_VIDEO_MAX_PIXELS,
    max_fps: float = _PVC_VIDEO_MAX_FPS,
) -> Path:
    """Transcode a video on the PVC to at most *max_pixels* per frame @ *max_fps*.

    Aspect ratio is preserved.  Videos already within both limits are
    returned unchanged.  A new ``*_preprocessed`` file is created
    alongside the original when any transcode is needed.
    """
    import json
    import subprocess as sp

    # Probe source dimensions and frame rate
    vw = vh = 0
    src_fps = 0.0
    try:
        probe = sp.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        pinfo = json.loads(probe.stdout)
        for s in pinfo.get("streams", []):
            vw = int(s.get("width", 0))
            vh = int(s.get("height", 0))
            r_frac = s.get("r_frame_rate", "0")
            if "/" in r_frac:
                num, den = r_frac.split("/", 1)
                src_fps = float(num) / float(den) if float(den) > 0 else 0.0
            else:
                src_fps = float(r_frac) if r_frac else 0.0
            break
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)

    if vw <= 0 or vh <= 0:
        return path

    needs_scale = vw * vh > max_pixels
    needs_fps = max_fps > 0 and src_fps > max_fps

    if not needs_scale and not needs_fps:
        return path

    filters: list[str] = []
    if needs_scale:
        scale = (max_pixels / (vw * vh)) ** 0.5
        new_w = max(2, (int(vw * scale) // 2) * 2)
        new_h = max(2, (int(vh * scale) // 2) * 2)
        filters.append(f"scale={new_w}:{new_h}")
    if needs_fps:
        filters.append(f"fps={max_fps}")

    preproc = path.with_stem(path.stem + "_preprocessed")
    cmd: list[str] = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(preproc),
    ]
    try:
        sp.run(cmd, capture_output=True, timeout=300)
        if preproc.exists() and preproc.stat().st_size > 0:
            return preproc
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)

    return path


def _truncate_audio_file(path: Path, max_seconds: float = 60.0) -> Path:
    """Truncate audio to at most *max_seconds* duration.

    Audio already within the limit is returned unchanged.  A new
    ``*_truncated`` file is created alongside the original when
    truncation is needed.  Used at staging time so audio queries don't
    exceed the ASR endpoint's payload cap.
    """
    import subprocess as sp

    try:
        probe = sp.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
    except Exception:
        logger.debug("Audio duration probe failed", exc_info=True)
        return path

    if duration <= max_seconds or duration <= 0:
        return path

    truncated = path.with_stem(path.stem + "_truncated")
    try:
        sp.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-t",
                str(max_seconds),
                "-i",
                str(path),
                "-c",
                "copy",
                "-y",
                str(truncated),
            ],
            capture_output=True,
            timeout=60,
        )
        if truncated.exists() and truncated.stat().st_size > 0:
            return truncated
        # Stream copy may fail for some containers — re-encode as fallback
        sp.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-t",
                str(max_seconds),
                "-i",
                str(path),
                "-y",
                str(truncated),
            ],
            capture_output=True,
            timeout=60,
        )
        if truncated.exists() and truncated.stat().st_size > 0:
            return truncated
    except Exception:
        logger.debug("Audio truncation failed", exc_info=True)

    return path


def _split_audio_segments(
    path: Path,
    max_bytes: int = _PVC_AUDIO_MAX_BYTES,
) -> list[Path]:
    """Split an audio file into segments each roughly *max_bytes* in size.

    Files already within *max_bytes* are returned as a single-element list.
    Oversized files are split using ffmpeg's segment muxer so that each
    segment can be transcribed individually without hitting API payload
    limits.  Segment files are created alongside the original with a
    ``_segment_NNN`` suffix.
    """
    import json
    import math
    import subprocess as sp

    if path.stat().st_size <= max_bytes:
        return [path]

    # Get total duration via ffprobe
    try:
        probe = sp.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        info = json.loads(probe.stdout)
        total_duration = float(info.get("format", {}).get("duration", 0))
    except Exception:
        return [path]

    if total_duration <= 0:
        return [path]

    # Number of segments so each is roughly max_bytes
    num_segments = max(1, math.ceil(path.stat().st_size / max_bytes))
    segment_duration = total_duration / num_segments

    stem = path.stem
    suffix = path.suffix
    output_pattern = str(path.parent / f"{stem}_segment_%03d{suffix}")

    cmd: list[str] = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "segment",
        "-segment_time",
        str(segment_duration),
        "-c",
        "copy",
        "-y",
        output_pattern,
    ]
    try:
        sp.run(cmd, capture_output=True, timeout=300)
    except Exception:
        return [path]

    # Collect segment files in order
    import glob

    segments = sorted(Path(p) for p in glob.glob(str(path.parent / f"{stem}_segment_*{suffix}")))
    return segments if segments else [path]


class DatasetManager:
    """Create, populate, query and delete multimodal RAG datasets.

    Parameters
    ----------
    base_path:
        Root directory for dataset storage on the PVC.
        Default ``/data`` → datasets live under ``/data/datasets/``.
    qdrant_host:
        Qdrant hostname (in-cluster service DNS).
        Empty string (default) uses local ``qdrant_path`` instead.
    qdrant_port:
        Qdrant gRPC port.
    embedder, reranker, vlm, asr:
        Model instances.  Uses defaults from :mod:`pcai_models` when
        ``None``.
    caption_with_asr:
        Whether to transcribe audio tracks from videos via ASR.
    remote:
        Whether to use remote (cluster) model endpoints.
    """

    def __init__(
        self,
        base_path: str = "/data",
        qdrant_host: str = "",
        qdrant_port: int = 6333,
        embedder=None,
        reranker=None,
        vlm=None,
        asr=None,
        caption_with_asr: bool = False,
        caption_with_vlm: bool = False,
        remote: bool = True,
        dedup_threshold: float = 0.995,
    ):
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.datasets_path.mkdir(parents=True, exist_ok=True)

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        if embedder is None:
            try:
                from multimodal_rag.utils.pcai_models import qwen3_vl_8B

                embedder = qwen3_vl_8B
            except ImportError:
                pass
        if reranker is None:
            try:
                from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B

                reranker = qwen3_vl_reranker_8B
            except ImportError:
                pass
        if vlm is None:
            try:
                from multimodal_rag.utils.pcai_models import gemma4_31B

                vlm = gemma4_31B
            except ImportError:
                pass
        if asr is None:
            try:
                from multimodal_rag.utils.pcai_models import cohere_transcribe_3_2b

                asr = cohere_transcribe_3_2b
            except ImportError:
                pass
        if embedder is None:
            raise RuntimeError(
                "Embedder model is required — set MODEL_EMBEDDER_NAME and "
                "MODEL_EMBEDDER_URL, or ensure the default can be imported."
            )

        self.embedder = embedder
        self.reranker = reranker
        self.vlm = vlm
        self.asr = asr
        self.caption_with_asr = caption_with_asr
        self.caption_with_vlm = caption_with_vlm
        self.remote = remote
        self.dedup_threshold = dedup_threshold

        self._verify_endpoints()

        # Cache: dataset_name → MultimodalRAG instance
        self._rag_cache: dict[str, MultimodalRAG] = {}
        self._rag_cache_lock = threading.Lock()

        # Note: per-dataset meta.json locking is handled by
        # _get_meta_lock() which returns a cross-process file lock
        # (fcntl.flock on a .meta.lock sidecar file).  No in-process
        # lock dict is needed.

    # ------------------------------------------------------------------
    # Model endpoint verification
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_endpoint(model: Any, role: str) -> None:
        """Verify a model endpoint is reachable via its OpenAI-compatible
        ``/v1/models`` endpoint."""
        if model is None:
            return
        url = model.url_remote.rstrip("/")
        headers: dict[str, str] = {}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"
        try:
            with httpx.Client(verify=False, timeout=5.0) as client:
                resp = client.get(url + "/v1/models", headers=headers)
                resp.raise_for_status()
                logger.info(
                    "✓ %s endpoint verified via /v1/models (%s)",
                    role,
                    url,
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Model '{role}' ({model.model_name}) at {url} is not reachable: {exc}")

    def _verify_endpoints(self) -> None:
        """Run endpoint verification for all configured models.

        The embedder is required — if it's unreachable, raise.  All other
        models (reranker, vlm, asr) are optional: a warning is logged but
        startup proceeds so the system can degrade gracefully without them.
        """
        self._verify_endpoint(self.embedder, "embedder")
        for role, model in (
            ("reranker", self.reranker),
            ("vlm", self.vlm),
            ("asr", self.asr),
        ):
            try:
                self._verify_endpoint(model, role)
            except RuntimeError as exc:
                logger.warning(
                    "Optional model '%s' endpoint not reachable — system will work without %s capabilities: %s",
                    role,
                    role,
                    exc,
                )

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dataset name validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject dataset names that could escape the datasets directory.

        Names must start with an alphanumeric character and contain only
        ``[A-Za-z0-9._-]``.  This prevents path-traversal via ``../`` in
        dataset names reaching ``mkdir`` / ``shutil.rmtree``.
        """
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", name):
            raise ValueError(
                f"Invalid dataset name {name!r}. Names must start with an "
                "alphanumeric character and contain only [A-Za-z0-9._-]."
            )

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        description: str = "",
        caption_with_asr: bool = False,
        caption_with_vlm: bool = False,
        keep_originals: bool = True,
        password: str | None = None,
    ) -> dict[str, Any]:
        """Create a new dataset.

        Parameters
        ----------
        caption_with_asr:
            Whether to transcribe audio tracks from videos during
            preprocessing.  Set per-dataset — applies to all files and
            documents added to this dataset.
        caption_with_vlm:
            Whether to generate VLM descriptions of images/videos during
            preprocessing (even when the embedder supports them natively).
            Descriptions enrich the text and are reused at retrieval time
            for generic queries, avoiding a VLM call.
        keep_originals:
            Whether to keep original (full-quality) files on disk after
            preprocessing.  When False, the original is deleted once a
            preprocessed copy exists, saving disk space.  The Qdrant
            ``original_image``/``original_video`` references are also
            removed.  Default True.
        password:
            Optional password to protect read access to the dataset.
            Stored as a PBKDF2-SHA256 hash.
        """
        self._validate_name(name)
        dataset_dir = self.datasets_path / name
        if dataset_dir.exists():
            raise FileExistsError(f"Dataset '{name}' already exists")
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "files").mkdir()

        meta: dict[str, Any] = {
            "name": name,
            "description": description,
            "caption_with_asr": caption_with_asr,
            "caption_with_vlm": caption_with_vlm,
            "keep_originals": keep_originals,
            "created": datetime.now().isoformat(),
            "document_count": 0,
        }
        if password:
            meta["password_hash"] = _hash_password(password)
        self._write_meta(name, meta)

        # Signal dataset existence to other pods via Redis (NFS cache bypass)
        _dataset_exist_set(name)

        # Pre-create the Qdrant collection by initialising the RAG instance
        self._get_rag(name)
        logger.info("Created dataset '%s' (caption_with_asr=%s)", name, caption_with_asr)
        return self._strip_password(meta)

    def has_password(self, name: str) -> bool:
        meta = self._read_meta(name)
        return bool(meta and meta.get("password_hash"))

    def verify_password(self, name: str, password: str) -> bool:
        meta = self._read_meta(name)
        if not meta:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        stored = meta.get("password_hash")
        if not stored:
            return True  # no password set → always passes
        return _check_password(password, stored)

    def set_password(self, name: str, password: str | None) -> None:
        meta = self._read_meta(name)
        if not meta:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        if password:
            meta["password_hash"] = _hash_password(password)
        else:
            meta.pop("password_hash", None)
        self._write_meta(name, meta)

    @staticmethod
    def _strip_password(meta: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *meta* with the password hash removed and a
        ``has_password`` boolean added."""
        m = dict(meta)
        m["has_password"] = "password_hash" in m
        m.pop("password_hash", None)
        return m

    def delete_dataset(self, name: str) -> None:
        """Delete a dataset and its Qdrant collection."""
        self._validate_name(name)
        rag = self._get_rag(name)
        try:
            vs = rag.vector_store
            assert vs is not None and not isinstance(vs, dict)
            client = vs._client  # type: ignore[attr-defined]
            client.delete_collection(vs.collection_name)  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
        if name in self._rag_cache:
            del self._rag_cache[name]

        dataset_dir = self.datasets_path / name
        if dataset_dir.exists():
            import shutil

            shutil.rmtree(dataset_dir)

        # Invalidate Redis existence cache so other pods don't retry
        _dataset_exist_delete(name)

        logger.info("Deleted dataset '%s'", name)

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return metadata for all existing datasets (password hash stripped)."""
        datasets: list[dict[str, Any]] = []
        if not self.datasets_path.exists():
            return datasets
        for child in sorted(self.datasets_path.iterdir()):
            if child.is_dir():
                meta = self._read_meta(child.name)
                if meta:
                    if not _DEFER_COUNT_SYNC:
                        self._sync_count_from_qdrant(child.name, meta)
                    datasets.append(self._strip_password(meta))
        return datasets

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        """Return metadata for a single dataset (password hash stripped).

        Raises ``FileNotFoundError`` if it does not exist.

        ``sync_count`` is ignored when ``RAG_DEFER_COUNT_SYNC`` is set, so
        reads never trigger a Qdrant round-trip + meta.json write under the
        scale deployment.

        When Redis is available and signals that the dataset exists, this
        method retries ``_read_meta`` a few times with short delays to
        bridge the NFS close-to-open cache propagation gap between pods.
        """
        meta = self._read_meta(name)
        if not meta:
            # NFS cache may not have propagated meta.json from another pod.
            # If Redis says the dataset exists, retry with short delays.
            if _dataset_exist_check(name):
                logger.debug(
                    "Dataset '%s' not visible on NFS yet but Redis confirms existence — retrying",
                    name,
                )
                for _ in range(_DATASET_EXIST_RETRY_COUNT):
                    time.sleep(_DATASET_EXIST_RETRY_DELAY)
                    meta = self._read_meta(name)
                    if meta:
                        break
            if not meta:
                raise FileNotFoundError(f"Dataset '{name}' not found")
        if sync_count and not _DEFER_COUNT_SYNC:
            self._sync_count_from_qdrant(name, meta)
        return self._strip_password(meta)

    def update_dataset(self, name: str, updates: dict[str, Any]) -> None:
        """Update metadata fields (e.g. description) for an existing dataset."""
        with self._get_meta_lock(name):
            meta = self._read_meta(name)
            if not meta:
                raise FileNotFoundError(f"Dataset '{name}' not found")
            for key in ("description", "caption_with_asr", "caption_with_vlm", "keep_originals"):
                if key in updates:
                    meta[key] = updates[key]
            self._write_meta(name, meta)

    # ------------------------------------------------------------------
    # Adding content
    # ------------------------------------------------------------------

    def add_documents(self, dataset_name: str, documents: list[str | dict[str, Any]]) -> list[str]:
        """Add raw documents (strings or multimodal dicts) to a dataset.

        Documents are embedded and stored in the dataset's Qdrant collection.
        Returns the list of stored point IDs.
        """
        rag = self._get_rag(dataset_name)
        ids = rag.add_to_vector_store(documents)
        self._increment_count(dataset_name, len(ids))
        return ids

    def add_file(self, dataset_name: str, file_path: str, original_name: str | None = None) -> dict[str, Any]:
        """Process and store a single file into a dataset.

        The file is first copied to the dataset's PVC-backed ``files/``
        directory, then processed according to its type and stored as one
        or more vector entries.

        After embedding, large media payloads (``image``, ``video``,
        ``audio`` data URLs) are stripped from the Qdrant stored payload
        and replaced with ``file://`` PVC paths.  The embedding step still
        sees the full media — only the persisted payload is kept lean.

        Returns
        -------
        dict
            ``{"type": …, "chunks": N, "stored_ids": …}``
        """
        rag = self._get_rag(dataset_name)
        mpk = rag.embedder.mm_processor_kwargs
        chunk_size = rag.embedder.chunk_size
        chunk_overlap = rag.embedder.chunk_overlap
        text_splitter = rag.embedder.text_splitter

        # Download remote URLs (HTTP or S3) to a temporary file — they are
        # processed directly without a separate PVC copy.
        _tmp_cleanup: list[str] = []
        source_url: str | None = None
        if _is_url(file_path):
            source_url = file_path
            file_path = _download_url(file_path)
            _tmp_cleanup.append(file_path)

        try:
            result = self._add_file_processed(
                dataset_name,
                file_path,
                rag,
                mpk,
                chunk_size,
                chunk_overlap,
                text_splitter=text_splitter,
                original_name=original_name,
                source_url=source_url,
            )
            return result
        finally:
            for tmp_path in _tmp_cleanup:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)

    def add_files_batch(
        self,
        dataset_name: str,
        file_entries: list[tuple[str, str]],
        progress_callback: Any | None = None,
        batch_score: float = 128.0,
    ) -> dict[str, Any]:
        """Process multiple files and store dedup-aware chunk counts.

        Producer–consumer pipeline:

        **Producer** (main thread): preprocesses files sequentially,
        accumulating chunks.  When *batch_score* is reached the batch is
        handed off to a background consumer thread and the producer
        immediately starts on the next file.

        **Consumer** (background thread): calls the embedding API and
        stores results in Qdrant.  This overlap hides the embedding
        latency behind preprocessing of subsequent files.

        Scoring (2.56 MB of data ≈ 1.0 score):

        * Standalone media files (image, video, audio) → ``file_bytes / 2.56 MB``
        * PDF chunks → ``1.0`` per chunk + ``image_data_url_bytes / 2.56 MB``
        * Other text-type files are processed individually inside
          :meth:`_add_file_processed` and are not batched.
        """
        import queue
        import threading

        rag = self._get_rag(dataset_name)
        mpk = rag.embedder.mm_processor_kwargs
        chunk_size = rag.embedder.chunk_size
        chunk_overlap = rag.embedder.chunk_overlap
        max_pixels = mpk.get("max_pixels", 720 * 720)
        fps = mpk.get("fps", 1.0)
        total_pixels = mpk.get("total_pixels", 0)
        target_frames = _VIDEO_TARGET_FRAMES

        file_results: list[dict[str, Any]] = []
        _MB = 2.56 * 1024 * 1024

        # -- Queues for producer-consumer handoff -------------------------------
        batch_queue: queue.Queue = queue.Queue()
        result_queue: queue.Queue = queue.Queue()
        consumer_busy = threading.Event()
        consumer_error: list[str | None] = [None]

        def consumer() -> None:
            """Background thread: embed + store batches from the producer."""
            try:
                while True:
                    batch = batch_queue.get()
                    if batch is None:
                        batch_queue.task_done()
                        break
                    consumer_busy.set()
                    docs, batch_files_list = batch
                    # Mark all files in this batch as embedding (with chunk count)
                    for fname, count, _ in batch_files_list:
                        if progress_callback:
                            progress_callback({"file": fname, "status": "embedding", "chunks": count})
                    try:

                        def _embed_and_store_batch():
                            """Embed + store all files in this batch. Raises on failure."""
                            results = []
                            offset = 0
                            for fname, count, file_type_ in batch_files_list:
                                file_docs = docs[offset : offset + count]
                                offset += count
                                ids = rag.add_to_vector_store(file_docs)
                                self._strip_media_payloads(rag, ids)
                                self._increment_count(dataset_name, len(ids), file_type=file_type_)
                                results.append((fname, len(ids)))
                            return results

                        batch_results = retry_call(
                            _embed_and_store_batch,
                            max_attempts=3,
                            base_delay=2.0,
                            connection_delay=10.0,
                        )
                        for fname, chunk_count in batch_results:
                            result_queue.put((fname, chunk_count, None))
                            if progress_callback:
                                progress_callback(
                                    {
                                        "file": fname,
                                        "status": "complete",
                                        "chunks": chunk_count,
                                    }
                                )
                    except Exception as exc:
                        logger.warning("Batch embedding failed after 3 attempts: %s", exc)
                        for fname, _, _ in batch_files_list:
                            result_queue.put((fname, 0, str(exc)))
                            if progress_callback:
                                progress_callback(
                                    {
                                        "file": fname,
                                        "status": "error",
                                        "error": str(exc),
                                    }
                                )
                    finally:
                        consumer_busy.clear()
                        batch_queue.task_done()
            except Exception as exc:
                logger.error("Consumer thread crashed unexpectedly: %s", exc)
                consumer_error[0] = str(exc)

        consumer_thread = threading.Thread(target=consumer, daemon=True)
        consumer_thread.start()

        # -- Producer: preprocess files, queue batches --------------------------
        batch_docs: list[str | dict[str, Any]] = []
        batch_files_list: list[tuple[str, int, str]] = []
        current_score = 0.0

        def _drain_results() -> None:
            """Pull completed results from the consumer into *file_results*.

            Progress callbacks are now sent directly from the consumer thread so
            the frontend gets live updates even after the producer has finished
            preprocessing all files.  This function only maintains the
            ``file_results`` accumulator used for the final return value.
            """
            while not result_queue.empty():
                fname, count, err = result_queue.get_nowait()
                if err:
                    file_results.append({"file": fname, "chunks": 0, "error": err})
                else:
                    file_results.append({"file": fname, "chunks": count})

        for tmp_path, orig_name in file_entries:
            fname = orig_name
            if progress_callback:
                progress_callback({"file": fname, "status": "preprocessing"})

            try:
                dst = self._store_file(dataset_name, tmp_path, original_name=fname)
                dst_str = str(dst)
                file_type = _classify_file(dst_str)
                file_bytes = dst.stat().st_size
                file_total = 0  # estimated total chunks (0 if unknown)

                if file_type == "pdf":
                    pdf_proc = PDFProcessor()
                    embed_batch = getattr(rag.embed, "chunk_size", 64) or 64
                    chunk_count = 0
                    pdf_batch: list[dict[str, Any]] = []
                    for chunk in pdf_proc.extract_chunks_iter(
                        dst_str,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        text_splitter=rag.embedder.text_splitter,
                    ):
                        pdf_batch.append(chunk)
                        chunk_count += 1

                        # Hand off to consumer mid-PDF for concurrent
                        # extraction + embedding.  This lets the embedder
                        # start processing the first pages while later
                        # pages are still being extracted.
                        if len(pdf_batch) >= embed_batch:
                            _fix_source(pdf_batch, fname, dst_str)
                            self._save_doc_media(dataset_name, pdf_batch, model_max_pixels=max_pixels)
                            batch_docs.extend(pdf_batch)
                            batch_files_list.append((fname, len(pdf_batch), file_type))
                            pdf_batch = []

                            batch_queue.put((batch_docs, batch_files_list))
                            batch_docs = []
                            batch_files_list = []
                            current_score = 0.0
                            _drain_results()

                    # Handle remaining chunks from the generator
                    if pdf_batch:
                        _fix_source(pdf_batch, fname, dst_str)
                        self._save_doc_media(dataset_name, pdf_batch, model_max_pixels=max_pixels)
                        batch_docs.extend(pdf_batch)
                        batch_files_list.append((fname, len(pdf_batch), file_type))
                    current_score = 0.0
                    file_total = chunk_count

                elif file_type == "image":
                    original_dst = dst_str
                    dst = _preprocess_image_file(dst)
                    dst_str = str(dst)
                    img_proc = ImageProcessor(max_pixels=max_pixels)
                    doc = img_proc.process(dst_str)
                    _fix_source([doc], fname, dst_str)
                    doc["preprocessed_image"] = f"file://{dst_str}"
                    if dst_str != original_dst:
                        doc["original_image"] = f"file://{original_dst}"
                    batch_docs.append(doc)
                    current_score += file_bytes / _MB
                    batch_files_list.append((fname, 1, file_type))
                    file_total = 1

                    # Delete original if keep_originals=False
                    if dst_str != original_dst and not self._get_keep_originals(dataset_name):
                        self._delete_original_file(dataset_name, original_dst, [], "original_image")
                        doc.pop("original_image", None)

                elif file_type == "video":
                    original_dst = dst_str
                    dst = _preprocess_video_file(dst)
                    dst_str = str(dst)
                    vid_proc = VideoProcessor(
                        fps=fps,
                        max_pixels=max_pixels,
                        total_pixels=total_pixels,
                        target_frames=target_frames,
                    )
                    embed_batch = getattr(rag.embed, "chunk_size", 64) or 64
                    vid_batch: list[dict[str, Any]] = []
                    vid_chunk_count = 0
                    store_url = f"file://{dst_str}"
                    original_url = f"file://{original_dst}" if dst_str != original_dst else None
                    for doc in vid_proc.process_iter(dst_str):
                        doc["preprocessed_video"] = store_url
                        if original_url:
                            doc["original_video"] = original_url
                        vid_batch.append(doc)
                        vid_chunk_count += 1

                        if len(vid_batch) >= embed_batch:
                            _fix_source(vid_batch, fname, dst_str)
                            self._save_doc_media(dataset_name, vid_batch)
                            batch_docs.extend(vid_batch)
                            batch_files_list.append((fname, len(vid_batch), file_type))
                            vid_batch = []

                            batch_queue.put((batch_docs, batch_files_list))
                            batch_docs = []
                            batch_files_list = []
                            current_score = 0.0
                            _drain_results()

                    if vid_batch:
                        _fix_source(vid_batch, fname, dst_str)
                        self._save_doc_media(dataset_name, vid_batch)
                        batch_docs.extend(vid_batch)
                        batch_files_list.append((fname, len(vid_batch), file_type))
                    current_score = 0.0
                    file_total = vid_chunk_count

                    # Delete original if keep_originals=False
                    if dst_str != original_dst and not self._get_keep_originals(dataset_name):
                        self._delete_original_file(dataset_name, original_dst, [], "original_video")
                        for vd in batch_docs:
                            if isinstance(vd, dict):
                                vd.pop("original_video", None)

                elif file_type == "audio":
                    segments = _split_audio_segments(dst)
                    audio_batch: list[dict[str, Any]] = []
                    for seg_idx, seg_path in enumerate(segments):
                        raw = seg_path.read_bytes()
                        mime = mimetypes.guess_type(str(seg_path))[0] or "audio/mpeg"
                        b64 = base64.b64encode(raw).decode("utf-8")
                        seg_name = (
                            f"{dst.name} — segment {seg_idx + 1}/{len(segments)}" if len(segments) > 1 else dst.name
                        )
                        doc = {
                            "text": f"[Audio: {seg_name}]",
                            "audio": f"data:{mime};base64,{b64}",
                            "source": dst_str,
                            "segment_index": seg_idx,
                        }
                        _fix_source([doc], fname, dst_str)
                        audio_batch.append(doc)
                    # Save each segment to disk so _strip_media_payloads
                    # doesn't replace the segment data URL with the full
                    # source file (which could be tens of MB and exceed
                    # the ASR endpoint's size cap at query time).
                    self._save_doc_media(dataset_name, audio_batch)
                    batch_docs.extend(audio_batch)
                    current_score += file_bytes / _MB
                    batch_files_list.append((fname, len(segments), file_type))
                    file_total = len(segments)

                else:
                    result = self._add_file_processed(
                        dataset_name,
                        dst_str,
                        rag,
                        mpk,
                        chunk_size,
                        chunk_overlap,
                        original_name=fname,
                    )
                    file_results.append({"file": fname, "chunks": result.get("chunks", 0)})
                    if progress_callback:
                        progress_callback(
                            {
                                "file": fname,
                                "chunks": result.get("chunks", 0),
                                "status": "complete",
                            }
                        )

                # Mark as preprocessed (waiting in batch buffer).
                # Skip for the else branch — those files are already fully
                # processed (embedded + stored) and have sent "complete".
                in_batch_buffer = file_type in ("pdf", "image", "video", "audio")
                if progress_callback and in_batch_buffer:
                    progress_callback({"file": fname, "status": "preprocessed", "total": file_total})

                # Hand off to consumer when the batch reaches the score
                # threshold.  Also send a reasonable chunk when the consumer
                # is idle so it never sits around waiting.
                if current_score >= batch_score:
                    batch_queue.put((batch_docs, batch_files_list))
                    batch_docs = []
                    batch_files_list = []
                    current_score = 0.0
                elif batch_docs and not consumer_busy.is_set():
                    # Consumer idle → send up to 10 files to keep it fed
                    # without flooding it with a huge batch.
                    n = min(len(batch_files_list), 10)
                    send_files = batch_files_list[:n]
                    keep_files = batch_files_list[n:]
                    send_doc_count = sum(c for _, c, _ in send_files)
                    batch_queue.put((batch_docs[:send_doc_count], send_files))
                    batch_docs = batch_docs[send_doc_count:]
                    batch_files_list = keep_files
                    current_score = 0.0

                # Drain any completed results so the frontend gets
                # "complete" callbacks promptly rather than all at the end.
                _drain_results()

            except Exception as exc:
                logger.warning("File '%s' failed: %s", fname, exc)
                if progress_callback:
                    progress_callback({"file": fname, "status": "error", "error": str(exc)})

        # -- Flush remaining batch and shut down consumer -----------------------
        if batch_docs:
            batch_queue.put((batch_docs, batch_files_list))
        batch_queue.put(None)  # sentinel
        consumer_thread.join()

        # If the consumer thread crashed, propagate the error
        if consumer_error[0] is not None:
            logger.error("Consumer thread crashed: %s", consumer_error[0])
            return {
                "status": "error",
                "error": f"Consumer thread crashed: {consumer_error[0]}",
                "file_count": len(file_entries),
                "files": file_results,
            }

        # -- Collect results from consumer --------------------------------------
        while not result_queue.empty():
            fname, count, err = result_queue.get_nowait()
            if err:
                file_results.append({"file": fname, "chunks": 0, "error": err})
            else:
                file_results.append({"file": fname, "chunks": count})

        return {"status": "ok", "file_count": len(file_entries), "files": file_results}

    def add_urls_batch(
        self,
        dataset_name: str,
        urls: list[str],
        progress_callback: Any | None = None,
        batch_score: float = 128.0,
    ) -> dict[str, Any]:
        """Download files from URLs (``s3://``, ``http://``, ``https://``)
        and process them as a batch.

        S3 URLs that look like directories (trailing slash or no file
        extension) are automatically expanded — each object under the
        prefix is listed and ingested individually.  Other URLs are
        processed as single files.

        Each file URL is downloaded to a temporary file, then delegated to
        :meth:`add_files_batch` for chunking, embedding and storage.
        Temporary files are cleaned up after ingestion.
        """
        # Expand S3 directory prefixes to individual file URLs
        expanded = _expand_urls(urls)
        if not expanded:
            return {"status": "ok", "file_count": 0, "files": []}

        file_entries: list[tuple[str, str]] = []
        tmp_paths: list[str] = []
        try:
            # Notify the frontend about each expanded file before downloading
            if progress_callback and len(expanded) > len(urls):
                for url in expanded:
                    clean = url.split("?")[0].rstrip("/")
                    orig_name = Path(clean).name or "download"
                    progress_callback({"file": orig_name, "status": "listed"})

            for url in expanded:
                # Derive a human-readable original name from the URL
                clean = url.split("?")[0].rstrip("/")
                orig_name = Path(clean).name or "download"

                tmp_path = _download_url(url)
                tmp_paths.append(tmp_path)
                file_entries.append((tmp_path, orig_name))

            result = self.add_files_batch(
                dataset_name,
                file_entries,
                progress_callback=progress_callback,
                batch_score=batch_score,
            )
            return result
        finally:
            for tmp_path in tmp_paths:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)

    def _add_file_processed(
        self,
        dataset_name: str,
        file_path: str,
        rag: MultimodalRAG,
        mpk: dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        text_splitter=None,
        original_name: str | None = None,
        source_url: str | None = None,
    ) -> dict[str, Any]:
        """Core file processing logic (shared by ``add_file`` and URL ingestion).

        If *source_url* is set (HTTP/S3 URL), the file is processed from the
        temp download and **not** copied to the PVC files directory.  The
        original URL is used as the canonical ``source``.
        """
        fname = original_name or Path(file_path).name
        # Archives: extract and process contained files
        if Path(file_path).suffix.lower() in _ARCHIVE_EXTS:
            if source_url:
                # URL archives extract directly from the temp file
                archive_proc = ArchiveProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    text_splitter=text_splitter,
                )
                chunks = archive_proc.process(file_path)
                _fix_source(chunks, fname, source_url)
                self._save_doc_media(dataset_name, chunks)
                ids = rag.add_to_vector_store(chunks) if chunks else []
                self._strip_media_payloads(rag, ids)
                self._increment_count(dataset_name, len(ids), file_type="archive")
            else:
                dst_path = self._store_file(dataset_name, file_path, original_name=fname)
                archive_proc = ArchiveProcessor(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    text_splitter=text_splitter,
                )
                chunks = archive_proc.process(str(dst_path))
                if chunks:
                    _fix_source(chunks, fname, str(dst_path))
                    self._save_doc_media(dataset_name, chunks)
                    ids = rag.add_to_vector_store(chunks)
                    self._strip_media_payloads(rag, ids)
                    self._increment_count(dataset_name, len(ids), file_type="archive")
                else:
                    ids = []
            return {"type": "archive", "chunks": len(ids), "stored_ids": ids}

        file_type = _classify_file(file_path)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        # Override chunk size for structured types (code, JSON, XML, YAML).
        # Uses the model's ``code_chunk_size`` / ``code_chunk_overlap``
        # (typically 8192/512) so that functions and document structures
        # stay intact.  A matching ``code_text_splitter`` is used when
        # available.
        if file_type in _STRUCTURED_TYPES:
            chunk_size = rag.embedder.code_chunk_size
            chunk_overlap = rag.embedder.code_chunk_overlap
            text_splitter = rag.embedder.code_text_splitter

        if source_url:
            # URL-based file: process from temp, use original URL as source, no PVC copy
            dst_path = Path(file_path)
            source_str = source_url
            store_url = source_url
        else:
            dst_path = self._store_file(dataset_name, file_path, original_name=fname)
            source_str = str(dst_path)
            store_url = f"file://{source_str}"

        if file_type == "pdf":
            pdf_proc = PDFProcessor()
            chunks = pdf_proc.extract_chunks(
                source_str,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            _fix_source(chunks, fname, source_str)
            self._save_doc_media(dataset_name, chunks, model_max_pixels=mpk.get("max_pixels", 720 * 720))
            ids = rag.add_to_vector_store(chunks)
            self._strip_media_payloads(rag, ids)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "pdf", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "image":
            original_path = str(dst_path)
            dst_path = _preprocess_image_file(dst_path)
            source_str = str(dst_path)
            store_url = f"file://{source_str}"
            img_proc = ImageProcessor(max_pixels=mpk.get("max_pixels", 720 * 720))
            doc = img_proc.process(source_str)
            _fix_source([doc], fname, source_str)
            doc["preprocessed_image"] = store_url
            if not source_url and source_str != original_path:
                doc["original_image"] = f"file://{original_path}"
            ids = rag.add_to_vector_store([doc])
            self._strip_media_payloads(rag, ids, store_url)
            self._increment_count(dataset_name, len(ids), file_type=file_type)

            # Delete original if keep_originals=False
            if not source_url and source_str != original_path and not self._get_keep_originals(dataset_name):
                self._delete_original_file(dataset_name, original_path, ids, "original_image")

            return {"type": "image", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "video":
            original_path = str(dst_path)
            dst_path = _preprocess_video_file(dst_path)
            source_str = str(dst_path)
            store_url = f"file://{source_str}"
            vid_proc = VideoProcessor(
                fps=mpk.get("fps", 1.0),
                max_pixels=mpk.get("max_pixels", 720 * 720),
                total_pixels=mpk.get("total_pixels", 0),
                target_frames=_VIDEO_TARGET_FRAMES,
            )
            docs = vid_proc.process(source_str)
            original_url = f"file://{original_path}" if not source_url and source_str != original_path else None
            for doc in docs:
                doc["preprocessed_video"] = store_url
                if original_url:
                    doc["original_video"] = original_url
            _fix_source(docs, fname, source_str)
            if docs:
                ids = rag.add_to_vector_store(docs)
                self._strip_media_payloads(rag, ids, store_url)
                self._increment_count(dataset_name, len(ids), file_type=file_type)
            else:
                ids = []

            # Delete original if keep_originals=False
            if original_url and not self._get_keep_originals(dataset_name):
                self._delete_original_file(dataset_name, original_path, ids, "original_video")

            return {"type": "video", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "audio":
            import base64

            segments = _split_audio_segments(dst_path)
            audio_docs: list[dict[str, Any]] = []
            for seg_idx, seg_path in enumerate(segments):
                raw = seg_path.read_bytes()
                mime = mimetypes.guess_type(str(seg_path))[0] or "audio/mpeg"
                b64 = base64.b64encode(raw).decode("utf-8")
                seg_name = (
                    f"{dst_path.name} — segment {seg_idx + 1}/{len(segments)}" if len(segments) > 1 else dst_path.name
                )
                doc = {
                    "text": f"[Audio: {seg_name}]",
                    "audio": f"data:{mime};base64,{b64}",
                    "source": source_str,
                    "segment_index": seg_idx,
                }
                _fix_source([doc], fname, source_str)
                audio_docs.append(doc)
            # Save each segment to disk so _strip_media_payloads
            # doesn't replace the segment data URL with the full source
            # file (which could be tens of MB and exceed the ASR endpoint's
            # size cap at query time).
            self._save_doc_media(dataset_name, audio_docs)
            ids = rag.add_to_vector_store(audio_docs)
            self._strip_media_payloads(rag, ids, store_url)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "audio", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "json":
            json_proc = JSONProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = json_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks) if chunks else []
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "json", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "table":
            table_proc = TableProcessor(chunk_size=chunk_size, text_splitter=text_splitter)
            chunks = table_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks) if chunks else []
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "table", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "code":
            code_proc = CodeProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = code_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "code", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "office":
            office_proc = OfficeProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = office_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            if chunks:
                self._save_doc_media(dataset_name, chunks)
                ids = rag.add_to_vector_store(chunks)
                self._strip_media_payloads(rag, ids, store_url)
                self._increment_count(dataset_name, len(ids), file_type=file_type)
            else:
                ids = []
            return {"type": "office", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "html":
            html_proc = HTMLProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = html_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "html", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "xml":
            xml_proc = XMLProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = xml_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks) if chunks else []
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "xml", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "yaml":
            yaml_proc = YAMLProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = yaml_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks) if chunks else []
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "yaml", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "notebook":
            nb_proc = NotebookProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = nb_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            if chunks:
                self._save_doc_media(dataset_name, chunks)
                ids = rag.add_to_vector_store(chunks)
                self._strip_media_payloads(rag, ids)
                self._increment_count(dataset_name, len(ids), file_type=file_type)
            else:
                ids = []
            return {"type": "notebook", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "ebook":
            epub_proc = EbookProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = epub_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            if chunks:
                self._save_doc_media(dataset_name, chunks)
                ids = rag.add_to_vector_store(chunks)
                self._strip_media_payloads(rag, ids)
                self._increment_count(dataset_name, len(ids), file_type=file_type)
            else:
                ids = []
            return {"type": "ebook", "chunks": len(ids), "stored_ids": ids}

        elif file_type == "log":
            log_proc = LogProcessor(chunk_size=chunk_size, text_splitter=text_splitter)
            chunks = log_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "log", "chunks": len(ids), "stored_ids": ids}

        else:  # text
            text_proc = TextProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
            chunks = text_proc.process(source_str)
            _fix_source(chunks, fname, source_str)
            ids = rag.add_to_vector_store(chunks)
            self._increment_count(dataset_name, len(ids), file_type=file_type)
            return {"type": "text", "chunks": len(ids), "stored_ids": ids}

    # ------------------------------------------------------------------
    # Media payload minimisation
    # ------------------------------------------------------------------

    def _strip_media_payloads(self, rag: MultimodalRAG, point_ids: list[str], source_url: str | None = None) -> None:
        """Replace heavy data-URL payloads in Qdrant with lightweight ``file://`` refs.

        **Tier 3 keys** (``image``, ``video``) are **kept as data URLs** — they
        hold model-ready base64 that the reranker and VLM consume directly at
        retrieval time, avoiding redundant file reads and re-resizing.

        **Tier 1/2 keys** (``preprocessed_*``, ``original_*``) and ``audio``
        are stripped if they still contain data URLs (they should normally be
        ``file://`` refs by this point, but this is a safety net).
        """
        if not point_ids:
            return
        vs = rag.vector_store
        assert vs is not None and not isinstance(vs, dict)
        client = vs._client  # type: ignore[attr-defined]
        coll = vs.collection_name  # type: ignore[attr-defined]

        # Batch retrieve all points in a single request
        try:
            points = client.retrieve(coll, ids=point_ids, with_payload=True, with_vectors=False)
        except Exception:
            return

        # Group modified point IDs by their new payload (JSON key for hashability)
        updates: dict[str, list[str]] = {}
        for pt in points:
            payload = pt.payload or {}
            meta = payload.get("metadata", {})
            if not isinstance(meta, dict):
                continue

            modified = False
            # NOTE: "image" and "video" are intentionally excluded — their
            # tier-3 data URLs are consumed directly by the reranker/VLM.
            for key in (
                "audio",
                "preprocessed_image",
                "preprocessed_video",
                "preprocessed_audio",
                "original_image",
                "original_video",
                "original_audio",
            ):
                val = meta.get(key)
                if val is None:
                    continue

                # Already a valid local file reference — nothing to strip
                def _is_valid_file_ref(v: Any) -> bool:
                    return isinstance(v, str) and v.startswith("file://") and os.path.exists(v[7:])

                if _is_valid_file_ref(val) or (
                    isinstance(val, list) and len(val) > 0 and all(_is_valid_file_ref(v) for v in val)
                ):
                    continue

                # Determine the replacement source reference
                src = source_url
                if src is None:
                    src = meta.get("source")

                if src is None:
                    # No source at all — remove the heavy payload
                    del meta[key]
                    modified = True
                elif src.startswith(("http://", "https://", "s3://")):
                    # Remote URL — keep as-is, no local copy to reference
                    pass
                elif os.path.exists(src if not src.startswith("file://") else src[7:]):
                    # Local file exists — replace heavy payload with file:// ref
                    ref = f"file://{src}" if not src.startswith("file://") else src
                    meta[key] = ref
                    modified = True
                else:
                    # No accessible source — remove the heavy payload key
                    del meta[key]
                    modified = True

            if modified:
                new_payload = {
                    "page_content": payload.get("page_content", ""),
                    "metadata": meta,
                }
                key = json.dumps(new_payload, sort_keys=True, default=str)
                updates.setdefault(key, []).append(str(pt.id))

        # Batch set_payload — one call per unique payload group
        for payload_json, ids in updates.items():
            try:
                client.set_payload(
                    coll,
                    payload=json.loads(payload_json),
                    points=ids,
                )
            except Exception:
                continue

    def _save_doc_media(
        self,
        dataset_name: str,
        docs: list[dict[str, Any]],
        model_max_pixels: int = 720 * 720,
    ) -> None:
        """Save inline data-URL media to files and swap the reference.

        **Tier 3 keys** (``image``, ``video``) are **kept as data URLs** so the
        reranker and VLM can consume them directly from Qdrant without file
        I/O or re-resizing.

        For ``image`` data URLs that arrive at a resolution above tier 3
        (e.g. PDF-extracted images), the data URL is resized down to
        *model_max_pixels* and a tier-2 ``preprocessed_image`` ``file://``
        ref is produced alongside.

        ``audio`` data URLs are saved to PVC files (existing behaviour).

        ``preprocessed_*`` / ``original_*`` data URLs, if any, are saved to
        PVC files as a safety net.
        """
        files_dir = self._dataset_dir(dataset_name) / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        for doc in docs:
            if not isinstance(doc, dict):
                continue

            # -- image: keep tier-3 data URL, produce tier-2 file ref ------
            val = doc.get("image")
            if val:
                urls = val if isinstance(val, list) else [val]
                new_urls: list[str] = []
                preproc_urls: list[str] = []
                for url in urls:
                    if not url.startswith("data:"):
                        new_urls.append(url)
                        continue
                    m = re.match(r"data:([^;]+);base64,(.+)", url)
                    if not m:
                        new_urls.append(url)
                        continue
                    mime = m.group(1)
                    raw = base64.b64decode(m.group(2))

                    from multimodal_rag.input_processing.image_processor import (
                        _resize_image,
                    )

                    # Tier 2: save to PVC file
                    tier2_raw = _resize_image(raw, mime, _PVC_IMAGE_MAX_PIXELS)
                    ext = mimetypes.guess_extension(mime) or ".bin"
                    fname = f"{uuid.uuid4().hex}_image{ext}"
                    dest = files_dir / fname
                    dest.write_bytes(tier2_raw)
                    preproc_urls.append(f"file://{dest}")

                    # Tier 3: keep as data URL (resized to model-ready)
                    tier3_raw = _resize_image(raw, mime, model_max_pixels)
                    b64 = base64.b64encode(tier3_raw).decode("utf-8")
                    new_urls.append(f"data:{mime};base64,{b64}")

                doc["image"] = new_urls if isinstance(val, list) else new_urls[0]
                if preproc_urls:
                    doc["preprocessed_image"] = preproc_urls if isinstance(val, list) else preproc_urls[0]

            # -- video: keep as tier-3 data URL (no file save) --------------
            # VideoProcessor already produces model-ready segments.
            # Don't save to file — the data URL stays in Qdrant for the
            # reranker/VLM to consume directly.
            # (No action needed for "video" key.)

            # -- audio: save to file (existing behaviour) ------------------
            for key in ("audio",):
                val = doc.get(key)
                if not val:
                    continue
                urls = val if isinstance(val, list) else [val]
                new_urls = []
                for url in urls:
                    if not url.startswith("data:"):
                        new_urls.append(url)
                        continue
                    m = re.match(r"data:([^;]+);base64,(.+)", url)
                    if not m:
                        new_urls.append(url)
                        continue
                    mime = m.group(1)
                    raw = base64.b64decode(m.group(2))
                    ext = mimetypes.guess_extension(mime) or ".bin"
                    fname = f"{uuid.uuid4().hex}_{key}{ext}"
                    dest = files_dir / fname
                    dest.write_bytes(raw)
                    new_urls.append(f"file://{dest}")
                doc[key] = new_urls if isinstance(val, list) else new_urls[0]

            # -- preprocessed_*/original_*: save data URLs to file ----------
            # Safety net — these should normally be file:// refs already.
            for key in (
                "preprocessed_image",
                "preprocessed_video",
                "preprocessed_audio",
                "original_image",
                "original_video",
                "original_audio",
            ):
                val = doc.get(key)
                if not val:
                    continue
                urls = val if isinstance(val, list) else [val]
                new_urls = []
                for url in urls:
                    if not url.startswith("data:"):
                        new_urls.append(url)
                        continue
                    m = re.match(r"data:([^;]+);base64,(.+)", url)
                    if not m:
                        new_urls.append(url)
                        continue
                    mime = m.group(1)
                    raw = base64.b64decode(m.group(2))
                    ext = mimetypes.guess_extension(mime) or ".bin"
                    fname = f"{uuid.uuid4().hex}_{key}{ext}"
                    dest = files_dir / fname
                    dest.write_bytes(raw)
                    new_urls.append(f"file://{dest}")
                doc[key] = new_urls if isinstance(val, list) else new_urls[0]

    # ------------------------------------------------------------------
    # Tier-schema migration
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_tier1_path(tier2_path: str) -> str | None:
        """Reverse-engineer the tier-1 (full-quality) path from a tier-2
        ``_preprocessed`` path.

        ``_preprocess_image_file`` / ``_preprocess_video_file`` create a
        sibling whose stem ends with ``_preprocessed``.  Stripping that
        suffix yields the original file.  Returns ``None`` when the tier-1
        file does not exist on disk (e.g. it was never preprocessed, or the
        original was deleted).
        """
        p = Path(tier2_path)
        if not p.stem.endswith("_preprocessed"):
            return None
        tier1 = p.with_name(p.stem[: -len("_preprocessed")] + p.suffix)
        if tier1.exists():
            return str(tier1)
        return None

    def _delete_original_file(
        self,
        dataset_name: str,
        original_path: str,
        doc_ids: list[str],
        tier1_key: str,
    ) -> None:
        """Delete a tier-1 original file and remove its Qdrant reference.

        Called after preprocessing succeeds when ``keep_originals=False``.
        Only deletes the file if it exists and is distinct from the
        preprocessed copy.  Also removes the ``original_image`` /
        ``original_video`` key from the Qdrant payloads and updates
        the dedup hash index.
        """
        p = Path(original_path)
        if not p.exists():
            return

        # Delete the file
        try:
            p.unlink()
            logger.info("Deleted original file (keep_originals=False): %s", original_path)
        except Exception:
            logger.debug("Failed to delete original file %s", original_path, exc_info=True)
            return

        # Remove from dedup hash index
        files_dir = self._dataset_dir(dataset_name) / "files"
        hashes_path = files_dir / ".hashes.json"
        if hashes_path.exists():
            with _cross_process_lock(files_dir / ".hashes.lock"):
                try:
                    with open(hashes_path) as f:
                        hash_index = json.load(f)
                    # Remove any entry pointing at this file
                    stale = [k for k, v in hash_index.items() if v == original_path]
                    for k in stale:
                        del hash_index[k]
                    with open(hashes_path, "w") as f:
                        json.dump(hash_index, f)
                except Exception:
                    logger.debug("Failed to update .hashes.json", exc_info=True)

        # Remove original_* key from Qdrant payloads
        try:
            rag = self._get_rag(dataset_name)
            vs = rag.vector_store
            assert vs is not None and not isinstance(vs, dict)
            client = vs._client  # type: ignore[attr-defined]
            coll = vs.collection_name  # type: ignore[attr-defined]

            for doc_id in doc_ids:
                client.set_payload(
                    collection_name=coll,
                    payload={tier1_key: None},
                    points=[doc_id],
                )
        except Exception:
            logger.debug("Failed to remove %s from Qdrant payloads", tier1_key, exc_info=True)

    def migrate_tier_schema(
        self,
        dataset_name: str,
        model_max_pixels: int = 720 * 720,
        batch_size: int = 200,
    ) -> dict[str, Any]:
        """Migrate existing Qdrant points to the three-tier media schema.

        * ``original_video`` (old tier-2 key) → renamed to ``preprocessed_video``
        * ``preprocessed_image`` / ``preprocessed_video`` derived from
          ``source`` when missing
        * ``original_image`` / ``original_video`` (tier 1) reverse-engineered
          from ``preprocessed_*`` paths
        * ``image`` / ``video`` ``file://`` refs converted to tier-3 base64
          data URLs (images are resized; videos are read as-is when small
          enough)

        The migration is **idempotent** — points that already have
        ``preprocessed_*`` keys and data-URL ``image``/``video`` are skipped.

        Returns a summary dict with ``migrated``, ``skipped``, and ``errors``
        counts.
        """
        import base64 as _b64

        from multimodal_rag.input_processing.image_processor import _resize_image

        rag = self._get_rag(dataset_name)
        vs = rag.vector_store
        if vs is None or isinstance(vs, dict):
            return {"error": "no vector store"}
        client = vs._client  # type: ignore[attr-defined]
        coll = vs.collection_name  # type: ignore[attr-defined]

        migrated = 0
        skipped = 0
        errors = 0

        scroll_offset: str | None = None
        while True:
            pts, scroll_offset = client.scroll(
                coll,
                limit=batch_size,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not pts:
                break

            updates: dict[str, list[str]] = {}
            for pt in pts:
                payload = pt.payload or {}
                meta = payload.get("metadata", {})
                if not isinstance(meta, dict):
                    continue

                changed = False
                pt_id = str(pt.id)

                # 1. Rename original_video → preprocessed_video (old tier 2)
                ov = meta.get("original_video")
                pv = meta.get("preprocessed_video")
                if ov and not pv:
                    meta["preprocessed_video"] = ov
                    del meta["original_video"]
                    changed = True
                elif ov and pv:
                    # Both exist — just drop the old key
                    del meta["original_video"]
                    changed = True

                # 2. Derive preprocessed_image from source if missing
                pi = meta.get("preprocessed_image")
                src = meta.get("source")
                if not pi and src and isinstance(src, str) and (src.startswith("file://") or os.path.exists(src)):
                    meta["preprocessed_image"] = f"file://{src}" if not src.startswith("file://") else src
                    changed = True

                # 3. Derive original_image / original_video (tier 1)
                for modality in ("image", "video"):
                    preproc_key = f"preprocessed_{modality}"
                    orig_key = f"original_{modality}"
                    if meta.get(orig_key):
                        continue  # already set
                    preproc_val = meta.get(preproc_key)
                    if not preproc_val:
                        # Fall back to source for images
                        if modality == "image" and src:
                            preproc_val = f"file://{src}" if not src.startswith("file://") else src
                        else:
                            continue
                    path_str = preproc_val.removeprefix("file://")
                    tier1 = self._derive_tier1_path(path_str)
                    if tier1:
                        meta[orig_key] = f"file://{tier1}"
                        changed = True

                # 4. Convert image file:// ref → tier-3 data URL
                img_val = meta.get("image")
                if img_val and isinstance(img_val, str) and img_val.startswith("file://"):
                    img_path = img_val[7:]
                    if os.path.exists(img_path):
                        try:
                            raw = Path(img_path).read_bytes()
                            mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                            tier3 = _resize_image(raw, mime, model_max_pixels)
                            b64 = _b64.b64encode(tier3).decode("utf-8")
                            meta["image"] = f"data:{mime};base64,{b64}"
                            changed = True
                        except Exception:
                            errors += 1
                    # If file doesn't exist, leave as-is

                # 5. Convert video file:// ref → tier-3 data URL
                #    Skip if the file is too large (> 8 MB) to avoid
                #    bloating Qdrant — those points will still work via
                #    file:// refs (reranker/VLM read the file).
                vid_val = meta.get("video")
                if vid_val and isinstance(vid_val, str) and vid_val.startswith("file://"):
                    vid_path = vid_val[7:]
                    if os.path.exists(vid_path):
                        try:
                            fsize = os.path.getsize(vid_path)
                            if fsize <= 8 * 1024 * 1024:
                                raw = Path(vid_path).read_bytes()
                                mime = mimetypes.guess_type(vid_path)[0] or "video/mp4"
                                b64 = _b64.b64encode(raw).decode("utf-8")
                                meta["video"] = f"data:{mime};base64,{b64}"
                                changed = True
                        except Exception:
                            errors += 1

                if not changed:
                    skipped += 1
                    continue

                new_payload = {
                    "page_content": payload.get("page_content", ""),
                    "metadata": meta,
                }
                key = json.dumps(new_payload, sort_keys=True, default=str)
                updates.setdefault(key, []).append(pt_id)
                migrated += 1

            # Batch set_payload
            for payload_json, ids in updates.items():
                try:
                    client.set_payload(
                        coll,
                        payload=json.loads(payload_json),
                        points=ids,
                    )
                except Exception:
                    errors += len(ids)

            if scroll_offset is None:
                break

        return {
            "migrated": migrated,
            "skipped": skipped,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        dataset_name: str,
        query: str | dict[str, Any],
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search a dataset and return ranked results.

        Returns a list of ``{"content": …, "score": …}`` dicts.
        """
        rag = self._get_rag(dataset_name)
        results = rag.retrieve(
            query,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_top_k=reranker_top_k,
            need_media=use_reranker and rag.reranker is not None,
        )
        output = []
        for doc, score in results:
            entry: dict[str, Any] = {"content": doc, "score": round(score, 4)}
            if isinstance(doc, dict):
                emb = doc.pop("_embedding_score", None)
                rerank = doc.pop("_reranker_score", None)
                if emb is not None:
                    entry["embedding_score"] = emb
                if rerank is not None:
                    entry["reranker_score"] = rerank
            output.append(entry)
        return output

    def list_documents(self, dataset_name: str, limit: int = 50) -> list[tuple[str, dict[str, Any]]]:
        """List stored document payloads for a dataset."""
        rag = self._get_rag(dataset_name)
        return rag.list_documents(limit=limit)

    def delete_document(self, dataset_name: str, doc_id: str) -> None:
        """Delete a single document from a dataset by its point ID."""
        rag = self._get_rag(dataset_name)
        vs = rag.vector_store
        assert vs is not None and not isinstance(vs, dict)
        client = vs._client  # type: ignore[attr-defined]
        from qdrant_client.models import PointIdsList

        client.delete(
            collection_name=vs.collection_name,  # type: ignore[attr-defined]
            points_selector=PointIdsList(points=[doc_id]),
            wait=True,
        )
        self._decrement_count(dataset_name, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_keep_originals(self, dataset_name: str) -> bool:
        """Read the keep_originals flag from dataset metadata (default True)."""
        meta = self._read_meta(dataset_name) or {}
        return meta.get("keep_originals", True)

    def _get_rag(self, dataset_name: str) -> MultimodalRAG:
        """Return (or create and cache) a MultimodalRAG for *dataset_name*."""
        # Fast path — no lock needed once cached
        rag = self._rag_cache.get(dataset_name)
        if rag is not None:
            return rag
        with self._rag_cache_lock:
            # Double-check after acquiring the lock
            rag = self._rag_cache.get(dataset_name)
            if rag is not None:
                return rag
            meta = self._read_meta(dataset_name) or {}
            ds_caption_with_asr = meta.get("caption_with_asr", self.caption_with_asr)
            ds_caption_with_vlm = meta.get("caption_with_vlm", self.caption_with_vlm)
            rag = MultimodalRAG(
                embedder=self.embedder,
                reranker=self.reranker,
                vlm=self.vlm,
                asr=self.asr,
                caption_with_asr=ds_caption_with_asr,
                caption_with_vlm=ds_caption_with_vlm,
                remote=self.remote,
                dedup_threshold=self.dedup_threshold,
                vector_store={
                    "qdrant_host": self.qdrant_host,
                    "qdrant_port": self.qdrant_port,
                    "collection_name": dataset_name,
                },
            )
            self._rag_cache[dataset_name] = rag
            return rag

    def _dataset_dir(self, name: str) -> Path:
        return self.datasets_path / name

    def _meta_path(self, name: str) -> Path:
        return self._dataset_dir(name) / "meta.json"

    def _read_meta(self, name: str) -> dict[str, Any] | None:
        p = self._meta_path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _write_meta(self, name: str, meta: dict[str, Any]) -> None:
        p = self._meta_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file then os.replace() so a crash
        # mid-write never leaves a truncated meta.json.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        os.replace(tmp, p)

    def _sync_count_from_qdrant(self, name: str, meta: dict[str, Any]) -> None:
        """Update ``document_count`` in *meta* to match the actual Qdrant point count."""
        try:
            rag = self._get_rag(name)
            vs = rag.vector_store
            if vs is not None and not isinstance(vs, dict):
                client = vs._client  # type: ignore[attr-defined]
                coll = vs.collection_name  # type: ignore[attr-defined]
                info = client.get_collection(coll)
                qdrant_count = info.points_count
                if meta.get("document_count", 0) != qdrant_count:
                    # Re-read meta inside the cross-process lock so we
                    # don't clobber a concurrent update (e.g. description
                    # change) that happened between the caller's read and
                    # our write.
                    with self._get_meta_lock(name):
                        fresh = self._read_meta(name)
                        if fresh:
                            fresh["document_count"] = qdrant_count
                            self._write_meta(name, fresh)
        except Exception:
            pass  # Best-effort: don't fail if Qdrant is unreachable

    def _get_meta_lock(self, name: str) -> contextlib.AbstractContextManager:
        """Return a cross-process lock for *meta.json* read→modify→write.

        Acquire this around any read-modify-write of a dataset's meta so
        that concurrent pods (sharing the RWX PVC) cannot clobber each
        other's updates.
        """
        return _cross_process_lock(self._dataset_dir(name) / ".meta.lock")

    def _increment_count(self, name: str, n: int, file_type: str | None = None) -> None:
        with self._get_meta_lock(name):
            meta = self._read_meta(name)
            if meta:
                meta["document_count"] = meta.get("document_count", 0) + n
                if file_type:
                    ft = meta.setdefault("file_type_counts", {})
                    ft[file_type] = ft.get(file_type, 0) + n
                self._write_meta(name, meta)

    def _decrement_count(self, name: str, n: int, file_type: str | None = None) -> None:
        """Decrement the document count (and optional file_type count) for a dataset."""
        with self._get_meta_lock(name):
            meta = self._read_meta(name)
            if meta:
                meta["document_count"] = max(0, meta.get("document_count", 0) - n)
                if file_type:
                    ft = meta.setdefault("file_type_counts", {})
                    ft[file_type] = max(0, ft.get(file_type, 0) - n)
                self._write_meta(name, meta)

    def _store_file(self, dataset_name: str, source_path: str, original_name: str | None = None) -> Path:
        """Copy *source_path* into the dataset's files directory and return the new path.

        Files are deduplicated by SHA-256 hash: if a file with the same
        content was already stored, the existing path is returned without
        copying.
        """
        self._validate_name(dataset_name)
        files_dir = self._dataset_dir(dataset_name) / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        # Compute hash of incoming file
        h = hashlib.sha256()
        with open(source_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        file_hash = h.hexdigest()

        # Load or create hash index.  The read→modify→write of .hashes.json
        # is guarded by a cross-process file lock so concurrent uploads
        # (possibly from different pods sharing the RWX PVC) cannot lose
        # hash entries and corrupt dedup.
        hash_index_path = files_dir / ".hashes.json"
        with _cross_process_lock(files_dir / ".hashes.lock"):
            if hash_index_path.exists():
                hash_index = json.loads(hash_index_path.read_text())
            else:
                hash_index = {}

            # Check for existing file with same hash
            if file_hash in hash_index:
                existing = Path(hash_index[file_hash])
                if existing.exists():
                    return existing
                # Stale entry — remove and re-store
                del hash_index[file_hash]

            # Copy file — use original_name for a readable stem if available
            if original_name:
                stem = Path(original_name).stem
                suffix = Path(original_name).suffix
            else:
                stem = Path(source_path).stem
                suffix = Path(source_path).suffix
            dest = files_dir / f"{uuid.uuid4().hex}_{stem}{suffix}"
            import shutil

            shutil.copy2(source_path, dest)
            hash_index[file_hash] = str(dest)
            # Atomic write so a crash never leaves a truncated index.
            tmp = hash_index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(hash_index, indent=2))
            os.replace(tmp, hash_index_path)
            return dest
