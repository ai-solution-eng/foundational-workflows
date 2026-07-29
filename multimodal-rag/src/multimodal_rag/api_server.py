"""
FastAPI application for the multimodal RAG dataset manager.

Run with::

    uvicorn multimodal_rag.api_server:app --host 0.0.0.0 --port 8000

Or via Python::

    python -m multimodal_rag.api_server
"""

import argparse
import asyncio
import mimetypes
import os
import random
import resource
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.utils.logging_utils import logging, setup_logger
from multimodal_rag.utils.general_tools import sync_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application config from environment
# ---------------------------------------------------------------------------

DATA_PATH = os.environ.get("DATA_PATH", "/data")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
# Path where the Qdrant PVC is mounted read-only (so we can report its disk
# usage). Empty when the mount is not configured (e.g. sharded Qdrant cluster).
QDRANT_STORAGE_PATH = os.environ.get("QDRANT_STORAGE_PATH", "")
RAG_REMOTE = os.environ.get("RAG_REMOTE", "true").lower() in ("true", "1", "yes")
RAG_CAPTION_VIDEO = os.environ.get("RAG_CAPTION_VIDEO", "false").lower() in (
    "true",
    "1",
    "yes",
)

# ---------------------------------------------------------------------------
# Lazy-initialised DatasetManager singleton
# ---------------------------------------------------------------------------

_dm: Optional[DatasetManager] = None
_dm_lock = threading.Lock()


def get_manager() -> DatasetManager:
    global _dm
    if _dm is not None:
        return _dm

    with _dm_lock:
        # Double-check after acquiring the lock — another thread may
        # have completed initialisation while we were waiting.
        if _dm is not None:
            return _dm

        # Read config from env (set by CLI args or K8s deployment) rather than
        # the module-level variables which are captured at import time.
        data_path = os.environ.get("DATA_PATH", "/data")
        qdrant_host = os.environ.get("QDRANT_HOST", "")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        rag_remote = os.environ.get("RAG_REMOTE", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        rag_caption_video = os.environ.get("RAG_CAPTION_VIDEO", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        rag_dedup_threshold = float(os.environ.get("RAG_DEDUP_THRESHOLD", "0.995"))

        from multimodal_rag.model_config import build_all

        embedder, reranker, vlm, asr = build_all()

        _dm = DatasetManager(
            base_path=data_path,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedder=embedder,
            reranker=reranker,
            vlm=vlm,
            asr=asr,
            caption_video=rag_caption_video,
            remote=rag_remote,
            dedup_threshold=rag_dedup_threshold,
        )
        logger.info(
            "DatasetManager initialised: data=%s qdrant=%s:%s remote=%s",
            data_path,
            qdrant_host or "(local)",
            qdrant_port,
            rag_remote,
        )
        return _dm


async def get_manager_async() -> DatasetManager:
    """Async-safe wrapper — offloads get_manager() to a thread pool.

    On the fast path (_dm already initialised) this is nearly free.
    On the slow path (startup init failed) it avoids blocking the event
    loop while model endpoints are probed under _dm_lock.
    """
    if _dm is not None:
        return _dm
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(sync_pool, get_manager)


# ---------------------------------------------------------------------------
# Unlock cache (password once, use-for-N-minutes)
# ---------------------------------------------------------------------------

# Identity is derived from the authenticated user (oauth2-proxy headers)
# rather than the client IP, which is unreliable behind Istio/oauth2-proxy
# (many users may share one proxy IP — one unlock would open the dataset
# for everyone).  Falls back to X-Forwarded-For / client host for
# non-authenticated deployments.
_UNLOCK_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
_UNLOCK_CACHE_LOCK = threading.Lock()
_UNLOCK_TTL = int(os.environ.get("UNLOCK_TTL", "1800"))  # seconds, default 30 min

# Optional Redis backend for cross-pod unlock sharing.  Enabled when
# REDIS_URL is set (the scale chart sets it); otherwise the per-process
# dict above is used (a user may be prompted to re-unlock if routed to a
# different pod).
_REDIS_URL = os.environ.get("REDIS_URL", "")
_redis_client: Any = None
_redis_client_lock = threading.Lock()


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
            logger.info("Unlock cache: Redis backend connected at %s", _REDIS_URL)
        except Exception as exc:
            logger.warning("Unlock cache: Redis unavailable (%s); falling back to in-memory", exc)
            _redis_client = None
    return _redis_client


def _unlock_client_id(request: Request) -> str:
    """Return a per-user identifier for the unlock cache.

    Prefers the authenticated user identity injected by oauth2-proxy
    (``X-Auth-Request-Email`` / ``X-Auth-Request-User``), then falls back
    to ``X-Forwarded-For`` and finally the socket peer — for deployments
    without an auth proxy.
    """
    for header in ("X-Auth-Request-Email", "X-Auth-Request-User", "X-Email", "X-User"):
        val = request.headers.get(header)
        if val:
            return val.strip()
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


def _unlock_cache_key(dataset: str, cid: str) -> str:
    return f"unlock:{dataset}:{cid}"


def _unlock_cache_get(dataset: str, cid: str) -> str | None:
    """Return a cached password if present and unexpired, else None."""
    r = _get_redis()
    if r is not None:
        try:
            val = r.get(_unlock_cache_key(dataset, cid))
            # Redis returns bytes; decode to str for downstream use.
            if val is not None:
                return val.decode() if isinstance(val, bytes) else val
            return None
        except Exception:
            return None
    with _UNLOCK_CACHE_LOCK:
        entry = _UNLOCK_CACHE.get((dataset, cid))
    if entry is None:
        return None
    expiry, cached_pw = entry
    if time.monotonic() < expiry:
        return cached_pw
    with _UNLOCK_CACHE_LOCK:
        _UNLOCK_CACHE.pop((dataset, cid), None)
    return None


def _unlock_cache_set(dataset: str, cid: str, password: str) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.set(_unlock_cache_key(dataset, cid), password, ex=_UNLOCK_TTL)
            return
        except Exception:
            pass
    with _UNLOCK_CACHE_LOCK:
        _UNLOCK_CACHE[(dataset, cid)] = (time.monotonic() + _UNLOCK_TTL, password)


def _require_dataset_password(
    dm: DatasetManager,
    name: str,
    password: str | None,
    request: Request | None = None,
) -> None:
    """Raise 401/403 if the dataset is password protected and the password
    is missing or incorrect.

    If *request* is provided, the unlock cache is also checked so that a
    previously-unlocked session can skip the password.
    """
    if not dm.has_password(name):
        return

    # 1. If a password was supplied, verify and cache it
    if password:
        if dm.verify_password(name, password):
            if request is not None:
                _unlock_cache_set(name, _unlock_client_id(request), password)
            return
        raise HTTPException(403, f"Incorrect password for dataset '{name}'")

    # 2. Check the unlock cache (trust the cached password without
    #    re-hashing — it was verified when first cached, and the TTL
    #    is short.  Re-verifying with PBKDF2-600k would add ~0.5s to
    #    every cached request.)
    if request is not None:
        cid = _unlock_client_id(request)
        cached = _unlock_cache_get(name, cid)
        if cached is not None:
            return

    raise HTTPException(401, f"Dataset '{name}' is password protected")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal RAG Dataset Manager",
    version="1.0.0",
)


@app.on_event("startup")
async def _eager_init() -> None:
    """Pre-initialise the DatasetManager before accepting traffic.

    ``get_manager()`` performs several synchronous HTTP calls (model
    endpoint verification) that can block for seconds.  By running this
    in a thread pool *and awaiting it* we ensure initialisation completes
    before uvicorn starts accepting connections — so /healthz probes and
    real requests never hit a partially-initialised manager or block on
    ``_dm_lock``.

    If a model endpoint is unreachable, the exception is logged but does
    not prevent startup — the first real request will retry.
    """
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(sync_pool, get_manager)
    except Exception as exc:
        logger.warning("DatasetManager startup init failed (will retry on first request): %s", exc)


# ---------------------------------------------------------------------------
# API routes — REST
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Liveness probe — returns 503 if the manager hasn't initialised yet."""
    if _dm is None:
        raise HTTPException(503, "DatasetManager not yet initialised")
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — checks that the manager is up and Qdrant is reachable.

    Uses a single lightweight GET to Qdrant's own ``/readyz`` endpoint
    instead of calling ``list_datasets()`` (which iterates every
    collection and can be slow under load).
    """
    if _dm is None:
        raise HTTPException(503, "DatasetManager not yet initialised")
    try:
        url = f"http://{_dm.qdrant_host}:{_dm.qdrant_port}/readyz"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(503, f"Qdrant not ready (HTTP {resp.status_code})")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Qdrant unreachable: {e}")
    return {"status": "ready"}


# -- Datasets -----------------------------------------------------------------


@app.post("/api/datasets")
async def api_create_dataset(body: dict[str, Any] = Body(...)):
    """Create a new named dataset.

    Request body::

        {"name": "my-dataset", "description": "...", "caption_video": false, "password": "secret"}

    ``caption_video`` (default ``false``) controls whether audio tracks
    from uploaded videos are transcribed during ingestion.
    ``password`` is optional — if set, all read operations on the dataset
    will require it.
    """
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Field 'name' is required")
    description = body.get("description", "")
    caption_video = body.get("caption_video", False)
    password = body.get("password") or None
    try:
        dm = await get_manager_async()
        loop = asyncio.get_running_loop()
        meta = await loop.run_in_executor(
            sync_pool,
            dm.create_dataset,
            name,
            description,
            bool(caption_video),
            password,
        )
        return {"status": "ok", "dataset": meta}
    except FileExistsError as e:
        raise HTTPException(409, str(e))


@app.post("/api/datasets/{name}/verify-password")
async def api_verify_dataset_password(name: str, body: dict[str, Any] = Body(...)):
    """Verify a dataset password.

    Request body::

        {"password": "secret"}

    Returns 200 on success, 401/403 on failure.
    """
    dm = await get_manager_async()
    try:
        dm.get_dataset(name, sync_count=False)  # ensure dataset exists
        _require_dataset_password(dm, name, body.get("password", ""))
        return {"status": "ok", "verified": True}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")
    except HTTPException:
        raise


@app.post("/api/datasets/{name}/unlock")
async def api_unlock_dataset(name: str, request: Request, body: dict[str, Any] = Body(...)):
    """Unlock a password-protected dataset for 30 minutes.

    Request body::

        {"password": "secret", "ttl": 1800}   # ttl in seconds (optional)

    Once unlocked, subsequent API calls to this dataset from the same
    client IP can omit the ``X-Dataset-Password`` header for the TTL
    duration.
    """
    dm = await get_manager_async()
    try:
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    if not dm.has_password(name):
        return {"status": "ok", "message": f"Dataset '{name}' is not password protected — nothing to unlock."}

    password = body.get("password", "")
    if not password:
        raise HTTPException(400, "Field 'password' is required")

    if not dm.verify_password(name, password):
        raise HTTPException(403, f"Incorrect password for dataset '{name}'")

    ttl = body.get("ttl", _UNLOCK_TTL)
    if not isinstance(ttl, int) or ttl < 60 or ttl > 86400:
        raise HTTPException(400, "TTL must be between 60 and 86400 seconds")

    cid = _unlock_client_id(request)
    with _UNLOCK_CACHE_LOCK:
        _UNLOCK_CACHE[(name, cid)] = (time.monotonic() + ttl, password)

    return {
        "status": "ok",
        "message": f"Dataset '{name}' unlocked for {ttl // 60} minutes.",
        "ttl_seconds": ttl,
    }


@app.post("/api/datasets/{name}/lock")
async def api_lock_dataset(name: str, request: Request):
    """Immediately revoke the unlock for a dataset (if any)."""
    dm = await get_manager_async()
    try:
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    cid = _unlock_client_id(request)
    with _UNLOCK_CACHE_LOCK:
        was = _UNLOCK_CACHE.pop((name, cid), None)

    if was:
        return {"status": "ok", "message": f"Dataset '{name}' locked."}
    return {"status": "ok", "message": f"Dataset '{name}' was not unlocked."}


@app.get("/api/datasets")
async def api_list_datasets(request: Request):
    """List all datasets with metadata."""
    dm = await get_manager_async()
    loop = asyncio.get_running_loop()
    datasets = await loop.run_in_executor(sync_pool, dm.list_datasets)
    cid = _unlock_client_id(request)
    with _UNLOCK_CACHE_LOCK:
        for ds in datasets:
            ds["unlocked"] = (ds["name"], cid) in _UNLOCK_CACHE
    return {"datasets": datasets}


@app.get("/api/datasets/{name}")
async def api_get_dataset(
    name: str,
    request: Request,
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Get metadata for a single dataset."""
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, x_dataset_password, request)
        loop = asyncio.get_running_loop()
        meta = await loop.run_in_executor(sync_pool, dm.get_dataset, name)
        return {"dataset": meta}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.patch("/api/datasets/{name}")
async def api_update_dataset(name: str, body: dict[str, Any] = Body(...)):
    """Update dataset metadata (e.g. description)."""
    dm = await get_manager_async()
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(sync_pool, dm.update_dataset, name, body)
        return {"status": "ok", "updated": name}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


@app.delete("/api/datasets/{name}")
async def api_delete_dataset(name: str):
    """Delete a dataset and its Qdrant collection."""
    try:
        dm = await get_manager_async()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(sync_pool, dm.delete_dataset, name)
        return {"status": "ok", "deleted": name}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


# -- Documents ----------------------------------------------------------------


@app.post("/api/datasets/{name}/documents")
async def api_add_documents(
    name: str,
    request: Request,
    body: Any = Body(None),
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Add documents (text strings or multimodal dicts) to a dataset.

    Accepts a JSON body that is either an array of documents or a single
    document dict/string.  Each document can be a plain string or a dict
    with optional ``text``, ``image``, ``video``, ``audio`` keys.
    """
    if body is None:
        raise HTTPException(400, "Request body is required (JSON array or object)")

    payload: list[Any]
    if isinstance(body, list):
        payload = body
    else:
        payload = [body]

    try:
        dm = await get_manager_async()
        _require_dataset_password(dm, name, x_dataset_password, request)
        loop = asyncio.get_running_loop()
        ids = await loop.run_in_executor(sync_pool, dm.add_documents, name, payload)
        return {"status": "ok", "stored_ids": ids, "count": len(ids)}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


# -- Files -------------------------------------------------------------------


@app.post("/api/datasets/{name}/files")
async def api_upload_file(
    name: str,
    request: Request,
    file: UploadFile = File(...),
    password: str = Form(""),
):
    """Upload a file to a dataset.

    Supported types: PDF, image (jpg/png/gif/bmp/webp), video (mp4/mkv/avi/mov),
    audio (mp3/wav/flac/ogg), and text files.  Files are processed and stored
    as vector entries in the dataset's Qdrant collection.
    """
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, password or None, request)
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    # Save uploaded file to a temporary location
    suffix = Path(file.filename or "upload").suffix if file.filename else ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(1 << 20):  # 1 MiB
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(sync_pool, dm.add_file, name, tmp_path, file.filename)
        return {"status": "ok", "file": file.filename, **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/api/datasets/{name}/batch-files")
async def api_upload_files_batch(
    name: str,
    request: Request,
    files: list[UploadFile] = File(...),
    password: str = Form(""),
):
    """Upload multiple files at once with SSE progress streaming."""
    import json
    from fastapi.responses import StreamingResponse

    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, password or None, request)
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    async def _progress_stream() -> AsyncGenerator[str, None]:
        import queue as thr_queue

        q: thr_queue.Queue = thr_queue.Queue()
        _KEEPALIVE_SECONDS = 10
        _event_id = 0  # incrementing SSE event ID for client reconnection

        # -- Write files to temp inside the SSE stream so the client gets
        #    "uploaded" events immediately rather than waiting in silence
        #    for all files to be copied first.
        file_entries: list[tuple[str, str]] = []
        try:
            for f in files:
                suffix = Path(f.filename or "upload").suffix if f.filename else ".bin"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                while chunk := await f.read(1 << 20):  # 1 MiB
                    tmp.write(chunk)
                tmp.close()
                file_entries.append((tmp.name, f.filename or "upload"))
                q.put({"file": f.filename or "upload", "status": "uploaded"})
        except Exception as exc:
            for p, _ in file_entries:
                try:
                    os.unlink(p)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)
            q.put({"error": str(exc)})
            q.put(None)
            # Drain queue and yield events
            while True:
                try:
                    event = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: q.get(timeout=_KEEPALIVE_SECONDS)
                    )
                except thr_queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                if "error" in event and "file" not in event:
                    _event_id += 1
                    yield f"id: {_event_id}\nevent: error\ndata: {json.dumps(event)}\n\n"
                    break
                _event_id += 1
                yield f"id: {_event_id}\nevent: progress\ndata: {json.dumps(event)}\n\n"
            return

        def _process() -> None:
            try:

                def cb(e):
                    q.put(e)

                r = dm.add_files_batch(name, file_entries, progress_callback=cb)
                q.put(r)
            except Exception as exc:
                q.put({"error": str(exc)})
            finally:
                q.put(None)

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _process)

        try:
            while True:
                try:
                    event = await loop.run_in_executor(None, lambda: q.get(timeout=_KEEPALIVE_SECONDS))
                except thr_queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                # Per-file progress errors have a "file" key — don't break
                # the stream for those; only fatal _process errors lack "file".
                if "error" in event and "file" not in event:
                    _event_id += 1
                    yield f"id: {_event_id}\nevent: error\ndata: {json.dumps(event)}\n\n"
                    break
                if "files" in event:
                    _event_id += 1
                    yield f"id: {_event_id}\nevent: complete\ndata: {json.dumps(event)}\n\n"
                    break
                _event_id += 1
                yield f"id: {_event_id}\nevent: progress\ndata: {json.dumps(event)}\n\n"
        finally:
            await task
            # Clean up temporary files (success path — error path already
            # cleans up inside the except block above).
            for p, _ in file_entries:
                try:
                    os.unlink(p)
                except Exception:
                    logger.debug("Suppressed exception", exc_info=True)

    return StreamingResponse(
        _progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/datasets/{name}/batch-urls")
async def api_upload_urls_batch(
    name: str,
    request: Request,
    body: dict[str, Any] = Body(...),
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Ingest files from URLs (S3, HTTP) into a dataset with SSE progress streaming.

    Request body::

        {"urls": ["s3://bucket/doc1.pdf", "https://example.com/doc2.pdf", ...]}

    Supports the same 17+ file formats as file upload (PDF, image, video,
    audio, code, Office docs, etc.).  Progress events are streamed via SSE.
    """
    import json
    from fastapi.responses import StreamingResponse

    urls = body.get("urls", [])
    if not urls or not isinstance(urls, list):
        raise HTTPException(400, "Field 'urls' must be a non-empty array of URL strings")

    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, x_dataset_password, request)
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    async def _progress_stream() -> AsyncGenerator[str, None]:
        import queue as thr_queue

        q: thr_queue.Queue = thr_queue.Queue()
        _KEEPALIVE_SECONDS = 10
        _event_id = 0  # incrementing SSE event ID for client reconnection

        def _process() -> None:
            try:

                def cb(e):
                    q.put(e)

                r = dm.add_urls_batch(name, urls, progress_callback=cb)
                q.put(r)
            except Exception as exc:
                q.put({"error": str(exc)})
            finally:
                q.put(None)

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _process)

        try:
            while True:
                try:
                    event = await loop.run_in_executor(None, lambda: q.get(timeout=_KEEPALIVE_SECONDS))
                except thr_queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                if "error" in event and "file" not in event:
                    _event_id += 1
                    yield f"id: {_event_id}\nevent: error\ndata: {json.dumps(event)}\n\n"
                    break
                if "files" in event:
                    _event_id += 1
                    yield f"id: {_event_id}\nevent: complete\ndata: {json.dumps(event)}\n\n"
                    break
                _event_id += 1
                yield f"id: {_event_id}\nevent: progress\ndata: {json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(
        _progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# -- Search ------------------------------------------------------------------


@app.get("/api/datasets/{name}/search")
async def api_search(
    name: str,
    request: Request,
    q: str = Query(..., description="Query text"),
    top_k: int = Query(10, ge=1, le=100),
    use_reranker: bool = Query(False),
    reranker_top_k: int = Query(3, ge=1, le=50),
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Search within a dataset using a text query.

    Returns ranked results with content and similarity scores.
    """
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, x_dataset_password, request)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            sync_pool,
            dm.search,
            name,
            q,
            top_k,
            use_reranker,
            reranker_top_k,
        )
        return {"query": q, "results": results}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


@app.post("/api/datasets/{name}/search")
async def api_search_multimodal(
    name: str,
    request: Request,
    body: dict[str, Any] = Body(...),
):
    """Search within a dataset using a multimodal query (text + image + video + audio).

    Request body::

        {
            "query": {
                "text": "description of what to find",
                "image": "data:image/png;base64,..."
            },
            "password": "my-dataset-password",
            "top_k": 10,
            "use_reranker": false,
            "reranker_top_k": 3
        }

    The ``query`` dict can contain any of the following keys:
      - ``text`` — plain text description
      - ``image`` — a data URL, remote URL, or list of URLs
      - ``video`` — a data URL, remote URL, or list of URLs
      - ``audio`` — a data URL, remote URL, or list of URLs

    Media values (image/video/audio) can be:
      - A single data URL string (``data:image/png;base64,...``)
      - A list of data URL strings
      - A remote HTTP(S) URL (``https://example.com/photo.jpg``)

    Returns ranked results with content and similarity scores.
    """
    query = body.get("query")
    if query is None:
        raise HTTPException(400, "Field 'query' is required (string or dict)")
    password = body.get("password") or None
    top_k = body.get("top_k", 10)
    use_reranker = body.get("use_reranker", False)
    reranker_top_k = body.get("reranker_top_k", 3)

    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, password, request)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            sync_pool,
            dm.search,
            name,
            query,
            top_k,
            use_reranker,
            reranker_top_k,
        )
        return {"query": query, "results": results}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


# -- Documents list ----------------------------------------------------------


@app.get("/api/datasets/{name}/documents")
async def api_list_documents(
    name: str,
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """List stored document payloads in a dataset."""
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, x_dataset_password, request)
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(sync_pool, dm.list_documents, name, limit)
        total = dm.get_dataset(name, sync_count=False).get("document_count", 0)
        entries = [{"id": doc_id, "payload": payload} for doc_id, payload in docs]
        return {"documents": entries, "count": total}
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")


@app.delete("/api/datasets/{name}/documents/{doc_id}")
async def api_delete_document(
    name: str,
    doc_id: str,
    request: Request,
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Delete a single document from a dataset by its point ID."""
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, x_dataset_password, request)
        dm.get_dataset(name, sync_count=False)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")
    try:
        dm = await get_manager_async()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(sync_pool, dm.delete_document, name, doc_id)
        return {"status": "ok", "deleted": doc_id}
    except Exception as e:
        raise HTTPException(500, str(e))


# -- File serving -----------------------------------------------------------


@app.get("/api/datasets/{name}/files/{filepath:path}")
async def api_serve_file(
    name: str,
    filepath: str,
    request: Request,
    password: str = Query(""),
    x_dataset_password: str | None = Header(None, alias="X-Dataset-Password"),
):
    """Serve a stored file from a dataset's files directory.

    Accepts the password via the ``X-Dataset-Password`` header or the
    ``password`` query parameter (the latter is used when the browser
    loads media in ``<img>`` / ``<video>`` / ``<audio>`` tags which
    cannot set custom headers).
    """
    dm = await get_manager_async()
    try:
        _require_dataset_password(dm, name, password or x_dataset_password or None, request)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{name}' not found")

    from fastapi.responses import FileResponse

    files_dir = (dm._dataset_dir(name) / "files").resolve()
    file_path = (files_dir / filepath).resolve()
    # Resolve to prevent directory traversal (e.g. "../../etc/passwd")
    try:
        file_path.relative_to(files_dir)
    except ValueError:
        raise HTTPException(403, "Invalid file path")
    if not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


# -- Staging (transient media handoff for MCP tools) -------------------------


_STAGING_TTL = int(os.environ.get("STAGING_TTL", "3600"))  # seconds, default 1h
# Fraction of staging requests that trigger a background sweep (0..1).
# Default 0.1 ≈ 1 in 10 uploads. 0 disables sweeping entirely.
_STAGING_SWEEP_RATE = float(os.environ.get("STAGING_SWEEP_RATE", "0.1"))
# Guards against overlapping sweeps; only one sweep runs at a time.
_sweep_lock = threading.Lock()


def _staging_root() -> Path:
    """Return (creating if needed) the staging directory under DATA_PATH."""
    d = Path(DATA_PATH) / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sweep_staging() -> None:
    """Remove staging subdirectories older than ``_STAGING_TTL`` seconds.

    Designed to run off the request path (see ``_maybe_sweep_staging``):
    the directory is flat (one subdir per staged file) so iteration is
    O(uploads), and every filesystem error is suppressed so a sweep never
    raises into the caller.
    """
    base = Path(DATA_PATH) / "staging"
    if not base.exists():
        return
    cutoff = time.time() - _STAGING_TTL
    for sub in base.iterdir():
        try:
            if not sub.is_dir():
                continue
            if sub.stat().st_mtime < cutoff:
                for f in sub.iterdir():
                    try:
                        f.unlink()
                    except Exception:
                        logger.debug("staging sweep: unlink failed", exc_info=True)
                try:
                    sub.rmdir()
                except Exception:
                    logger.debug("staging sweep: rmdir failed", exc_info=True)
        except Exception:
            logger.debug("staging sweep: failed on %s", sub, exc_info=True)


def _try_sweep_staging() -> None:
    """Run ``_sweep_staging`` only if no other sweep is in progress.

    Acquires ``_sweep_lock`` non-blocking: if a sweep is already running,
    returns immediately.  Any exception is logged and swallowed so the
    executor thread never propagates a failure to a fire-and-forget caller.
    """
    if not _sweep_lock.acquire(blocking=False):
        return
    try:
        _sweep_staging()
    except Exception:
        logger.debug("staging sweep: unexpected error", exc_info=True)
    finally:
        _sweep_lock.release()


async def _maybe_sweep_staging() -> None:
    """Probabilistically trigger a background sweep without blocking.

    With probability ``_STAGING_SWEEP_RATE`` (default 0.1), schedules
    ``_try_sweep_staging`` on ``sync_pool`` and returns immediately —
    fire-and-forget — so the staging request is not delayed by cleanup.
    Concurrent sweeps are skipped via ``_sweep_lock``.
    """
    if _STAGING_SWEEP_RATE <= 0:
        return
    if random.random() >= _STAGING_SWEEP_RATE:
        return
    loop = asyncio.get_running_loop()
    loop.run_in_executor(sync_pool, _try_sweep_staging)


def _is_valid_staging_id(staging_id: str) -> bool:
    """A staging id is a 32-char lowercase hex uuid4."""
    return len(staging_id) == 32 and all(c in "0123456789abcdef" for c in staging_id)


def _preprocess_staged_file(dest: Path) -> Path:
    """Downscale staged media to the embedder's native processing caps.

    Reads ``mm_processor_kwargs`` (``max_pixels``, ``fps``) from the
    embedder config so the staged query file is sized to what the embedder
    will actually process — no wasted bytes, no payload-size rejections.
    A 37 MB phone photo becomes a ≤720×720 JPEG (a few hundred KB); a
    high-bitrate video is capped at the embedder's ``fps`` and per-frame
    pixel budget before it ever reaches the embedding endpoint.

    Returns the path to the (possibly new) preprocessed file.  When a new
    ``*_preprocessed`` sibling is produced, the original is removed and
    the preprocessed file is moved into its place so the staged URL and
    filename stay clean.  Audio and non-media files are returned
    unchanged.
    """
    from multimodal_rag.dataset_manager import (
        _classify_file,
        _preprocess_image_file,
        _preprocess_video_file,
        _truncate_audio_file,
    )

    mpk = _get_embedder_mm_kwargs()
    max_pixels = mpk.get("max_pixels", 720 * 720)
    fps = mpk.get("fps", 1.0)

    file_type = _classify_file(dest.name)
    try:
        if file_type == "image":
            result = _preprocess_image_file(dest, max_pixels=max_pixels)
        elif file_type == "video":
            result = _preprocess_video_file(dest, max_pixels=max_pixels, max_fps=fps)
        elif file_type == "audio":
            result = _truncate_audio_file(dest, max_seconds=60.0)
        else:
            return dest
    except Exception:
        logger.warning("staging preprocess failed for %s", dest.name, exc_info=True)
        return dest

    if result == dest:
        return dest  # no resize needed

    # Replace the original with the preprocessed file so the staged URL
    # and filename are unchanged.
    try:
        dest.unlink(missing_ok=True)
        result.replace(dest)
        return dest
    except Exception:
        logger.warning("staging preprocess replace failed; using sibling", exc_info=True)
        return result


# Cached embedder mm_processor_kwargs — resolved once from env vars, no
# endpoint probing.  Used by _preprocess_staged_file to size staged media
# to the embedder's native processing caps.
_embedder_mm_kwargs: Optional[dict[str, Any]] = None


def _get_embedder_mm_kwargs() -> dict[str, Any]:
    """Return the embedder's ``mm_processor_kwargs`` (cached).

    Builds the embedder config from environment variables via
    :func:`build_embedder` — this reads ``MODEL_EMBEDDER_EXTRA`` /
    ``MODEL_EMBEDDER_NAME`` etc. and never invokes the embedding API,
    so it is safe to call at staging time before the DatasetManager is
    initialised.  If ``MODEL_EMBEDDER_NAME`` is omitted the model id is
    auto-discovered via a best-effort ``GET /v1/models`` probe (harmless
    to fail).  Falls back to the default Qwen3-VL kwargs when the
    embedder env vars are not set.
    """
    global _embedder_mm_kwargs
    if _embedder_mm_kwargs is not None:
        return _embedder_mm_kwargs

    from multimodal_rag.model_config import _DEFAULT_MM_KWARGS, build_embedder

    emb = build_embedder()
    if emb is not None and emb.mm_processor_kwargs:
        _embedder_mm_kwargs = dict(emb.mm_processor_kwargs)
    else:
        _embedder_mm_kwargs = dict(_DEFAULT_MM_KWARGS)
    logger.info("staging preprocess caps: %s", _embedder_mm_kwargs)
    return _embedder_mm_kwargs


@app.post("/api/staging")
async def api_staging_upload(
    request: Request,
    file: UploadFile = File(...),
):
    """Stage a transient media file so an MCP tool can consume it.

    Used by the Open WebUI filter to hand uploaded images / video / audio
    to the RAG MCP ``search_dataset`` tool **without** injecting base64
    data into the LLM context.  The filter uploads the raw bytes here and
    injects only the returned short URL as a hint; the LLM then calls
    ``search_dataset(image=...)`` (etc.) with that URL, and the MCP
    server — which shares this pod's PVC — reads the file directly from
    disk via its ``file://`` path.

    Files are written under ``DATA_PATH/staging/{uuid}/{filename}`` and
    swept after ``STAGING_TTL`` seconds (default 1h).  Sweeping runs
    probabilistically in the background (see ``_maybe_sweep_staging``)
    and never blocks an upload.
    """
    await _maybe_sweep_staging()

    # Sanitise the filename: keep only the basename (no path traversal).
    raw_name = file.filename or "upload"
    safe_name = Path(raw_name).name or "upload"
    if safe_name in (".", ".."):
        safe_name = "upload"

    staging_id = uuid.uuid4().hex
    sub = _staging_root() / staging_id
    try:
        sub.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # uuid4 collision — astronomically unlikely; retry once.
        staging_id = uuid.uuid4().hex
        sub = _staging_root() / staging_id
        sub.mkdir(parents=True, exist_ok=False)

    dest = sub / safe_name
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):  # 1 MiB
                out.write(chunk)
    except Exception as exc:
        try:
            dest.unlink(missing_ok=True)
            sub.rmdir()
        except Exception:
            logger.debug("staging upload: cleanup failed", exc_info=True)
        raise HTTPException(500, f"Failed to stage uploaded file: {exc}")

    # Downscale oversized staged media so the MCP search_dataset tool
    # doesn't hand a multi-megabyte file to the embedder (which rejects
    # payloads above its size cap — e.g. the 37 MB phone photo that
    # triggered "Maximum file size exceeded").  Reuses the same PVC
    # preprocessing helpers as ingest so the staged query is bounded by
    # the same pixel / fps caps the system was designed to embed.
    # Runs in the sync thread pool because PIL / ffmpeg are blocking.
    raw_size = dest.stat().st_size
    try:
        dest = await asyncio.get_running_loop().run_in_executor(sync_pool, _preprocess_staged_file, dest)
    except Exception:
        logger.warning("staging preprocess failed for %s", safe_name, exc_info=True)
    preprocessed_size = dest.stat().st_size
    if preprocessed_size != raw_size:
        logger.info(
            "staging preprocess: %s %.1f MB → %.1f MB",
            safe_name,
            raw_size / (1 << 20),
            preprocessed_size / (1 << 20),
        )
    elif raw_size > 5 * (1 << 20):  # > 5 MB and not shrunk — suspicious
        logger.warning(
            "staging preprocess NO-OP: %s is %.1f MB but was not shrunk "
            "(PIL/ffmpeg missing, file type unsupported, or already within caps)",
            safe_name,
            raw_size / (1 << 20),
        )

    # The MCP server is a sidecar in the same pod and shares this PVC,
    # so a file:// URL is directly resolvable by it (no HTTP hop, no auth).
    file_url = f"file://{dest.resolve()}"

    # Also provide an HTTP URL for clients that cannot read the PVC
    # (e.g. an MCP server deployed as a separate pod).  Uses localhost
    # because, in the standard helm chart, the API and MCP containers
    # share a network namespace.
    port = request.url.port or 8000
    http_url = f"http://localhost:{port}/api/staging/{staging_id}"

    return {
        "id": staging_id,
        "filename": safe_name,
        "file_url": file_url,
        "http_url": http_url,
        "size_bytes": dest.stat().st_size,
        "ttl_seconds": _STAGING_TTL,
    }


@app.get("/api/staging/{staging_id}")
async def api_staging_serve(staging_id: str):
    """Serve a previously staged file (one file per staging id).

    Primarily intended for MCP deployments where the server cannot read
    the shared PVC and must fetch over HTTP.  Staged files are swept in
    the background by uploads (see ``_maybe_sweep_staging``) and after
    ``STAGING_TTL`` seconds; this endpoint does not block on sweeping.
    """
    if not _is_valid_staging_id(staging_id):
        raise HTTPException(400, "Invalid staging id")
    sub = Path(DATA_PATH) / "staging" / staging_id
    if not sub.is_dir():
        raise HTTPException(404, "Staged file not found or expired")
    files = [f for f in sub.iterdir() if f.is_file()]
    if not files:
        raise HTTPException(404, "Staged file not found or expired")
    target = files[0]
    media_type, _ = mimetypes.guess_type(target.name)
    from fastapi.responses import FileResponse

    return FileResponse(str(target), media_type=media_type or "application/octet-stream")


# -- Admin / Management ------------------------------------------------------


# Module-level state for CPU% calculation (delta between calls)
_last_cpu_time: float = 0.0
_last_wall_time: float = 0.0


def _read_cgroup_memory_limit() -> int:
    """Read the cgroup v2/v1 memory limit (in bytes). Returns -1 if unknown."""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            val = int(Path(path).read_text().strip())
            if val > 0 and val < (1 << 62):  # "max" or huge value = no limit
                return val
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)
    return -1


def _read_cpu_limit() -> float:
    """Read the cgroup CPU limit in cores. Returns os.cpu_count() as fallback."""
    # cgroup v2: "400000 100000" → 4 cores
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] == "max":
            return float(os.cpu_count() or 1)
        return int(parts[0]) / int(parts[1])
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)
    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)
    return float(os.cpu_count() or 1)


@app.get("/api/admin/health")
async def api_health_stats() -> dict[str, Any]:
    """Lightweight health stats — polled every 10s by the management page.

    Returns CPU%, memory (RSS vs cgroup limit), file PVC disk usage,
    and Qdrant connection status + total points.
    """
    global _last_cpu_time, _last_wall_time

    import shutil

    # -- CPU (delta from last call) -------------------------------------------
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = rusage.ru_utime + rusage.ru_stime
    now = time.monotonic()
    cpu_limit = _read_cpu_limit()
    if _last_wall_time > 0:
        delta_cpu = cpu_seconds - _last_cpu_time
        delta_wall = now - _last_wall_time
        cpu_percent = round(min(delta_cpu / delta_wall / cpu_limit * 100, 100.0), 1) if delta_wall > 0 else 0.0
    else:
        cpu_percent = 0.0
    _last_cpu_time = cpu_seconds
    _last_wall_time = now

    # -- Memory ----------------------------------------------------------------
    rss_kb = rusage.ru_maxrss  # max RSS in KB (Linux)
    # Also read current VmRSS from /proc for live value
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break
    except Exception:
        logger.debug("Suppressed exception", exc_info=True)
    rss_bytes = rss_kb * 1024
    mem_limit = _read_cgroup_memory_limit()
    mem_percent = round(rss_bytes / mem_limit * 100, 1) if mem_limit > 0 else 0.0

    # -- File PVC disk ----------------------------------------------------------
    dm = await get_manager_async()
    usage = shutil.disk_usage(str(dm.base_path))

    # -- Qdrant status + total points ------------------------------------------
    qdrant_status = "unknown"
    qdrant_collections = 0
    qdrant_total_points = 0
    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            qhost = os.environ.get("QDRANT_HOST", "")
            qport = os.environ.get("QDRANT_PORT", "6333")
            if qhost:
                base = f"http://{qhost}:{qport}"
                resp = client.get(f"{base}/collections")
                if resp.status_code == 200:
                    qdrant_status = "ok"
                    cols = resp.json()["result"]["collections"]
                    qdrant_collections = len(cols)
                    for col in cols:
                        info = client.get(f"{base}/collections/{col['name']}").json()["result"]
                        qdrant_total_points += info.get("points_count", 0)
    except Exception:
        qdrant_status = "unreachable"

    # -- Qdrant PVC disk usage (only when mounted read-only) -------------------
    # The Qdrant PVC is a separate volume from the file PVC; Qdrant's HTTP API
    # does not expose on-disk usage, so we read it from the read-only mount
    # configured in the Helm chart (env QDRANT_STORAGE_PATH).
    qdrant_pvc: Optional[dict[str, Any]] = None
    if QDRANT_STORAGE_PATH and Path(QDRANT_STORAGE_PATH).exists():
        try:
            q_usage = shutil.disk_usage(QDRANT_STORAGE_PATH)
            qdrant_pvc = {
                "total_bytes": q_usage.total,
                "used_bytes": q_usage.used,
                "free_bytes": q_usage.free,
                "used_percent": round(q_usage.used / q_usage.total * 100, 1),
            }
        except Exception:
            logger.debug("Suppressed exception", exc_info=True)

    return {
        "cpu": {
            "percent": cpu_percent,
            "cores": cpu_limit,
            "limit_millicores": int(cpu_limit * 1000),
        },
        "memory": {
            "rss_bytes": rss_bytes,
            "limit_bytes": mem_limit,
            "percent": mem_percent,
        },
        "file_pvc": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round(usage.used / usage.total * 100, 1),
        },
        "qdrant": {
            "status": qdrant_status,
            "collections": qdrant_collections,
            "total_points": qdrant_total_points,
            "pvc": qdrant_pvc,
        },
    }


@app.get("/api/admin/storage")
async def api_storage_stats() -> dict[str, Any]:
    """Return PVC disk usage and dataset statistics."""
    dm = await get_manager_async()

    # Run the potentially slow stats collection in a thread pool so the
    # event loop (and health/readiness probes) are not blocked.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(sync_pool, _build_storage_stats, dm)


def _build_storage_stats(dm: DatasetManager) -> dict[str, Any]:
    """Synchronous helper that does the actual work for /api/admin/storage."""
    import shutil

    # PVC disk usage
    usage = shutil.disk_usage(str(dm.base_path))
    pvc = {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": round(usage.used / usage.total * 100, 1),
    }

    # File-type extension → label mapping
    _FT_MAP: dict[str, str] = {
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".gif": "image",
        ".bmp": "image",
        ".webp": "image",
        ".tiff": "image",
        ".mp4": "video",
        ".mkv": "video",
        ".avi": "video",
        ".mov": "video",
        ".webm": "video",
        ".m4v": "video",
        ".mp3": "audio",
        ".wav": "audio",
        ".flac": "audio",
        ".ogg": "audio",
        ".m4a": "audio",
        ".wma": "audio",
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".java": "code",
        ".cpp": "code",
        ".c": "code",
        ".go": "code",
        ".rs": "code",
        ".csv": "table",
        ".tsv": "table",
        ".xlsx": "table",
        ".xls": "table",
        ".docx": "office",
        ".pptx": "office",
        ".odt": "office",
        ".odp": "office",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".ipynb": "notebook",
        ".epub": "ebook",
        ".zip": "archive",
        ".tar": "archive",
        ".gz": "archive",
    }

    # Dataset-level stats
    datasets = dm.list_datasets()
    ds_stats = []
    total_docs = 0
    for ds in datasets:
        name = ds["name"]
        doc_count = ds.get("document_count", 0)
        total_docs += doc_count

        # File storage size + type breakdown
        files_dir = dm._dataset_dir(name) / "files"
        file_bytes = 0
        file_types: dict[str, int] = {}
        if files_dir.exists():
            for f in files_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    sz = f.stat().st_size
                    file_bytes += sz
                    ext = f.suffix.lower()
                    label = _FT_MAP.get(ext, "other")
                    file_types[label] = file_types.get(label, 0) + sz

        # Document counts per file type — backfill from Qdrant when missing.
        # Uses paginated scroll with a small batch size so large collections
        # (8k+ points) don't spike memory or block.
        ft_counts: dict[str, int] = ds.get("file_type_counts", {}) or {}
        if not ft_counts and doc_count > 0 and dm._rag_cache.get(name):
            try:
                rag_cached = dm._rag_cache[name]
                vs = rag_cached.vector_store
                if vs is not None and not isinstance(vs, dict):
                    client = vs._client  # type: ignore[attr-defined]
                    coll = vs.collection_name  # type: ignore[attr-defined]
                    from qdrant_client.models import PayloadSelectorInclude

                    _SCROLL_BATCH = 256
                    _MAX_SCAN = 50000
                    offset = None
                    scanned = 0
                    while scanned < _MAX_SCAN:
                        scroll = client.scroll(
                            coll,
                            limit=_SCROLL_BATCH,
                            offset=offset,
                            with_payload=PayloadSelectorInclude(include=["metadata.source"]),
                            with_vectors=False,
                        )
                        points = scroll[0]
                        if not points:
                            break
                        for pt in points:
                            md = (pt.payload or {}).get("metadata") or {}
                            src = str(md.get("source") or "") if isinstance(md, dict) else ""
                            ext = Path(src).suffix.lower()
                            label = _FT_MAP.get(ext, "other")
                            ft_counts[label] = ft_counts.get(label, 0) + 1
                        scanned += len(points)
                        offset = scroll[1]
                        if offset is None:
                            break
                    if ft_counts:
                        ds["file_type_counts"] = ft_counts
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)

        ds_stats.append(
            {
                "name": name,
                "documents": doc_count,
                "file_bytes": file_bytes,
                "file_types": file_types,
                "file_type_counts": ft_counts,
                "has_password": ds.get("has_password", False),
            }
        )

    return {
        "pvc": pvc,
        "datasets": ds_stats,
        "total_datasets": len(datasets),
        "total_documents": total_docs,
    }


@app.post("/api/admin/datasets/{name}/migrate-tier-schema")
async def api_migrate_tier_schema(name: str) -> dict[str, Any]:
    """Migrate a dataset's Qdrant points to the three-tier media schema.

    Renames ``original_video`` → ``preprocessed_video``, derives tier-1
    ``original_*`` paths, and converts ``image``/``video`` ``file://`` refs
    to tier-3 base64 data URLs.  Idempotent.
    """
    dm = await get_manager_async()

    def _do_migrate() -> dict[str, Any]:
        return dm.migrate_tier_schema(name)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(sync_pool, _do_migrate)


# ---------------------------------------------------------------------------
# HTML frontend
# ---------------------------------------------------------------------------

_HTML_INDEX: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    global _HTML_INDEX
    if _HTML_INDEX is None:
        html_path = Path(__file__).parent / "templates" / "index.html"
        if html_path.exists():
            _HTML_INDEX = html_path.read_text(encoding="utf-8")
        else:
            _HTML_INDEX = "<html><body><h1>Frontend not found</h1></body></html>"
    return _HTML_INDEX


@app.get("/favicon.png")
async def favicon():
    from fastapi.responses import FileResponse

    favicon_path = Path(__file__).parent / "templates" / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/png")
    return Response(status_code=404)


@app.get("/manage", response_class=HTMLResponse)
async def manage():
    return await index()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-path", default="/data")
    parser.add_argument("--qdrant-host", default="")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    os.environ["DATA_PATH"] = args.data_path
    os.environ["QDRANT_HOST"] = args.qdrant_host
    os.environ["QDRANT_PORT"] = str(args.qdrant_port)
    os.environ["RAG_REMOTE"] = os.environ.get("RAG_REMOTE", "true")

    setup_logger(level=args.log_level)

    import uvicorn

    uvicorn.run(
        "multimodal_rag.api_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
