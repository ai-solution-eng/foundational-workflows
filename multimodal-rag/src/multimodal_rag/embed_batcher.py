"""Shared (cross-process) embedding query batcher.

The embedding endpoint (vLLM pooling runner) costs ~160ms of server-side
overhead per HTTP call and only amortises it over the texts in a single
``input`` batch.  The app's per-process ``_QueryBatcher`` aggregates only
what arrives on *one* event loop, so with many workers/replicas the batches
shrink to ~1 text and throughput collapses to the model's unbatched rate.

This service is a singleton aggregator: every API/MCP pod POSTs its text-only
queries to ``/embed``, and this process collects them over the configured
window into a single ``/v1/embeddings`` call.  Because it is a single process,
the batch size is independent of the number of app processes, so app scaling
no longer degrades embedding throughput.

Run with a single worker (uvicorn workers=1); more would reintroduce the
batching split this service exists to remove.
"""

import argparse
import asyncio
import hashlib
import os
import threading
import time
from array import array
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from multimodal_rag.model_config import build_all
from multimodal_rag.utils.logging_utils import logging, setup_logger
from multimodal_rag.utils.model_adapters import _QueryBatcher

logger = logging.getLogger("embed-batcher")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown: (re)build the batcher and watch CONFIG_DIR for swaps.

    Replaces the deprecated ``@app.on_event("startup")`` hook (removed in
    FastAPI 0.99+); the lifespan context is the supported equivalent.
    """
    config_dir = os.environ.get("CONFIG_DIR", "")
    if config_dir:
        from multimodal_rag.model_config import apply_config_dirs, start_config_watcher

        apply_config_dirs(config_dir)
        start_config_watcher(config_dir, _reload)
    _reload()
    logger.info("Embed batcher initialised")
    yield


app = FastAPI(title="embed-batcher", version="1.0.0", lifespan=_lifespan)


@dataclass
class _BatcherState:
    """Mutable singleton state guarded by *_state_lock*."""

    batcher: _QueryBatcher | None = None
    error: str | None = None
    model_name: str = ""


_state_lock = threading.Lock()
_state = _BatcherState()

# Hard cap on a single flushed batch's model round-trip.  The batcher is
# single-flight: if the embedding endpoint stalls, every query queued in that
# batch would otherwise wait for it.  A healthy batch drains in ~1-4s, so this
# is a tripwire for a stalled model call — timing out lets the caller fall
# back to local batching instead of hanging the request for tens of seconds.
_FLUSH_TIMEOUT = float(os.environ.get("EMBED_BATCH_FLUSH_TIMEOUT", "15"))

# ---------------------------------------------------------------------------
# Query-vector cache
#
# The batcher's clients are agents: retries, repeated probes across turns,
# several sessions asking the same thing of one shared dataset.  Those are
# structurally repeat-heavy, and an identical (model, text) pair always
# produces an identical vector, so serving the repeat from memory skips the
# whole batch window + model round-trip.  Unique queries are unaffected —
# this is not a throughput lever for them, just a dedup of genuine repeats.
#
# Keyed on (model_name, text) via sha256: model-scoped so a hot-reload to a
# different embedder can never serve stale vectors (and _reload clears it
# anyway).  Values are packed array('f') — ~16KB per 4096-dim vector vs
# ~130KB as a list of boxed floats (same trick as mcp_server's media-query
# cache): 4096 entries cost ~65MB resident, not ~530MB.
_EMBED_BATCH_CACHE_MAX = max(0, int(os.environ.get("EMBED_BATCH_CACHE_MAX", "4096")))
_query_vec_cache: "dict[str, array]" = {}
_query_vec_cache_order: "list[str]" = []


def _cache_key(model_name: str, text: str) -> str:
    return hashlib.sha256((model_name + "\x00" + text).encode()).hexdigest()


def _cache_get(key: str) -> "array | None":
    if not _EMBED_BATCH_CACHE_MAX:
        return None
    with _state_lock:
        vec = _query_vec_cache.get(key)
        if vec is not None:
            # LRU touch: move the key to the back of the eviction order.
            try:
                _query_vec_cache_order.remove(key)
                _query_vec_cache_order.append(key)
            except ValueError:  # pragma: no cover - dict/order desync guard
                _query_vec_cache_order.append(key)
    return vec


def _cache_put(key: str, vector: "list[float]") -> None:
    if not _EMBED_BATCH_CACHE_MAX:
        return
    with _state_lock:
        if key not in _query_vec_cache and len(_query_vec_cache_order) >= _EMBED_BATCH_CACHE_MAX:
            for _ in range(len(_query_vec_cache_order) - _EMBED_BATCH_CACHE_MAX + 1):
                _query_vec_cache.pop(_query_vec_cache_order.pop(0), None)
        _query_vec_cache[key] = array("f", vector)
        if key not in _query_vec_cache_order:
            _query_vec_cache_order.append(key)


class EmbedRequest(BaseModel):
    text: str


def _embedder_model_name(embedder: Any) -> str:
    """The embedder's model identity for cache keying (wrapper or inner model)."""
    for obj in (embedder, getattr(embedder, "model", None)):
        name = getattr(obj, "model_name", None)
        if name:
            return str(name)
    return ""


def _build_batcher() -> _QueryBatcher:
    """Construct the aggregating batcher bound to the configured embedder."""
    embedder, _, _, _ = build_all()
    if embedder is None:
        raise RuntimeError("No embedder model configured (MODEL_EMBEDDER_URL not set)")
    model = embedder.model  # MultiModalEmbeddings
    max_size = int(os.environ.get("EMBEDDING_QUERY_BATCH_SIZE", "128"))
    max_wait_ms = float(os.environ.get("EMBEDDING_QUERY_BATCH_WAIT_MS", "200"))
    return _QueryBatcher(
        embed_fn=model._embed_text_batch_async,
        max_batch_size=max_size,
        max_wait_ms=max_wait_ms,
    )


def _reload() -> None:
    """(Re)build the batcher — also used by the hot-config watcher."""
    try:
        batcher = _build_batcher()
        embedder, _, _, _ = build_all()
        model_name = _embedder_model_name(embedder)
        with _state_lock:
            _state.batcher = batcher
            _state.error = None
            _state.model_name = model_name
            # A reload can change the embedder — cached vectors are model-scoped
            # (sha256 includes model_name), but a stale-while-clearing window is
            # not worth reasoning about: drop them all.
            _query_vec_cache.clear()
            _query_vec_cache_order.clear()
    except Exception as exc:
        logger.exception("Failed to build embedder for batcher")
        with _state_lock:
            _state.error = str(exc)


@app.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    with _state_lock:
        batcher = _state.batcher
        error = _state.error
        model_name = _state.model_name
    if batcher is None:
        raise HTTPException(503, f"Embedding not initialised: {error or 'building'}")
    key = _cache_key(model_name, req.text)
    cached = _cache_get(key)
    if cached is not None:
        return {"embedding": list(cached), "cache": "hit"}
    try:
        vec = await asyncio.wait_for(batcher.submit(req.text), timeout=_FLUSH_TIMEOUT)
    except TimeoutError:
        logger.warning("Embed batch flush timed out after %.0fs", _FLUSH_TIMEOUT)
        raise HTTPException(504, f"Embedding timed out after {_FLUSH_TIMEOUT:g}s")
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        raise HTTPException(502, f"Embedding failed: {exc}")
    _cache_put(key, vec)
    return {"embedding": vec, "cache": "miss"}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    with _state_lock:
        ready = _state.batcher is not None
    if not ready:
        raise HTTPException(503, f"Embed batcher not ready: {_state.error or 'building'}")
    return {"status": "ready"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared embedding query batcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logger(level=args.log_level)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
