"""Standalone vector store layer for MultimodalRAG.

Replaces the former langchain-based vector store (langchain_core,
langchain_qdrant) with minimal local equivalents exposing only the
surface used by ``rag_system.py``.
Qdrant operations that ``rag_system`` performs directly via ``qdrant_client``
(upsert / scroll / query_batch_points / delete) are unchanged — this module
only supplies the container object holding ``_client`` / ``collection_name`` /
``vector_name`` plus the similarity-search methods used at retrieval time.
"""

import asyncio
import logging
import os
import time
import uuid
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import multimodal_rag.utils.bm25 as bm25_lane
from multimodal_rag.utils.general_tools import cosine_sim

logger = logging.getLogger(__name__)

# Dedicated thread pool for Qdrant I/O (batched searches, upserts, scrolls).
# The sync Qdrant client performs network I/O that used to run either on the
# event loop (freezing every concurrent request for the duration of a
# multi-MB upsert) or on the default executor (where long-running batch
# ingest jobs and ffmpeg subprocesses could starve every search flush).
# Keeping Qdrant calls here gives them their own bounded lane; sized via
# QDRANT_POOL_SIZE.  rag_system imports this pool for the same reason.
_QDRANT_IO_POOL = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("QDRANT_POOL_SIZE", "4"))),
    thread_name_prefix="qdrant-io",
)


# Minimum per-lane candidate depth for hybrid (RRF) fusion.
#
# RRF is rank arithmetic over its candidate lists, so the prefetch window is
# part of the RANKING DEFINITION, not of the paging.  Coupling it to the
# display limit made a document's fused score depend on how many results the
# caller asked to see (see the prefetch block in _query_requests).  A fixed
# floor decouples the two: fusion always considers at least this many
# candidates per lane, and the fused list is truncated to k afterwards.
#
# Why 200: the floor must cover the LARGEST pool any caller can
# request, or the invariance this exists for is lost.  rag_system.aretrieve
# over-fetches for the reranker with fetch_k = max(top_k, 4*reranker_top_k),
# and reranker_top_k is allowed up to 50, so fetch_k can reach 200 while
# top_k tops out at 100.  A floor of 100 would fuse a reranked call over 200
# candidates and an unranked one over 100 — the same display-parameter
# coupling in a new costume.  At 200 every request fuses over the same pool.
# NOTE: lowering RAG_PREFETCH_FLOOR below 200 trades that invariance back.
_DEFAULT_PREFETCH_FLOOR = 200
# Hard ceiling on the configurable floor.  Every other limit on this path is
# bounded (top_k 1..100, reranker_top_k 1..50); without a cap a typo like
# RAG_PREFETCH_FLOOR=1000000 would be sent verbatim as a Prefetch limit on
# BOTH lanes and turn a config mistake into an unbounded Qdrant scan.
_MAX_PREFETCH_FLOOR = 1000


def _prefetch_floor() -> int:
    """Resolved per-lane prefetch floor (RAG_PREFETCH_FLOOR, default 200).

    Falls back to the default on a blank/unparseable/non-positive value and
    clamps an oversized one, rather than raising: a bad env var must not take
    retrieval down, and must not turn into an unbounded Qdrant scan either.
    """
    raw = (os.environ.get("RAG_PREFETCH_FLOOR") or "").strip()
    if not raw:
        return _DEFAULT_PREFETCH_FLOOR
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning("Unknown RAG_PREFETCH_FLOOR=%r, using %d", raw, _DEFAULT_PREFETCH_FLOOR)
        return _DEFAULT_PREFETCH_FLOOR
    if val <= 0:
        logger.warning("RAG_PREFETCH_FLOOR=%r is not positive, using %d", raw, _DEFAULT_PREFETCH_FLOOR)
        return _DEFAULT_PREFETCH_FLOOR
    if val > _MAX_PREFETCH_FLOOR:
        logger.warning("RAG_PREFETCH_FLOOR=%r exceeds the %d cap, clamping", raw, _MAX_PREFETCH_FLOOR)
        return _MAX_PREFETCH_FLOOR
    return val


def _prefetch_depth(k: int) -> int:
    """Per-lane candidate depth for a fused request of display limit k.

    Always at least k (a caller asking for more than the floor still gets a
    full page) and at least the configured floor, so ordinary queries are
    score-invariant in k.
    """
    return max(int(k), _prefetch_floor())


def _stable_tiebreak(items: "list[tuple[Document, float]]") -> "list[tuple[Document, float]]":
    """Order fused results deterministically: score desc, then a content key.

    Equal fused scores are structural under RRF, not a corner case: with the
    default weights (dense 1.5, sparse 1.0, k=2) the dense ladder yields
    0.6, 0.4286, 0.3333, ... and the sparse ladder 0.5, 0.3333, 0.25, ...,
    so a dense-position-3 hit ties a sparse-position-2 hit at exactly
    0.3333.  qdrant-client's fusion sorts on the score alone, leaving tied
    entries in Qdrant's return order, which is not guaranteed stable between
    identical queries.

    The key is content-derived ((source, page, chunk_index, text prefix))
    with the point id as the FINAL discriminator.  Content leads so the
    order is stable across a re-ingest of unchanged files (point ids are
    regenerated on re-ingest); the point id is needed because the content
    fields COLLIDE by construction for a twin and its parent, and for
    media-only points with no page/chunk_index.
    """

    def _key(pair: "tuple[Document, float]") -> tuple:
        doc = pair[0]
        meta = getattr(doc, 'metadata', None) or {}
        text = (getattr(doc, 'page_content', '') or '')[:200]
        # Numeric-aware coercion so page/chunk ordering is natural ('2' before
        # '10') and a missing value sorts consistently rather than as ''.
        def _num(v: Any) -> tuple:
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v or ''))

        return (
            str(meta.get('source') or ''),
            _num(meta.get('page')),
            _num(meta.get('chunk_index')),
            text,
            # FINAL total-order key.  The four content fields above COLLIDE by
            # construction for a twin and its parent (a text twin shares
            # source/page/chunk_index AND its page_content is identical), and
            # for media-only points (no page/chunk_index, empty text).  Without
            # a further key those ties fall back to Qdrant's return order —
            # exactly the non-determinism this function exists to remove.
            # _point_id is written by _points_to_docs on every point.
            str(meta.get('_point_id') or ''),
        )

    return sorted(items, key=lambda p: (-float(p[1]), _key(p)))


# ---------------------------------------------------------------------------
# Pre-fusion twin collapse + client-side RRF
# ---------------------------------------------------------------------------
#
# Twins are an intentional ingest feature: one document is represented by up
# to two points (a multimodal parent and a text-only/caption twin) so an image
# embedding and an image+text embedding can both represent the same content.
# Both points share (source, page, chunk_index, time-window).
#
# The defect this fixes.  Server-side RRF fuses BY POINT ID, so a parent and
# its twin enter fusion as two independent candidates.  Whether both land in
# the prefetch window depends on the requested depth, and _dedup_twins then
# PREFERS THE NON-TWIN REGARDLESS OF SCORE.  Net effect: a document's fused
# score DROPS as the caller asks for MORE results, because the weaker
# representation wins the tiebreak once the twin also makes the cut.  Observed
# on a real corpus: one page scored 0.1979 at top_k<=31 (twin only) and 0.0790
# at top_k>=32 (parent won the dedup).
#
# The fix.  Collapse the representations PER LANE first, keeping the HIGHER
# similarity of the two, then fuse the per-lane winners.  A document then
# occupies exactly one slot in each lane, and its score no longer depends on
# pool depth.  When dense and sparse pick DIFFERENT representations of the
# same document, both lane contributions still count toward one document
# (that is the point of twins: let each lane use its best representation).

_RRF_DEFAULT_K = 2


def _twin_collapse_enabled() -> bool:
    """Whether hybrid search collapses twins per lane BEFORE fusion.

    Default ON (product ruling 2026-10-22).  Server-side RRF fuses by point id,
    so a multimodal parent and its text-only twin arrived as two independent
    candidates: the same document could be returned TWICE, and its fused score
    depended on the requested depth (a doc at dense-11/sparse-1 scored 0.5000
    at top_k=10 but 0.5833 at top_k=20).  This switches hybrid queries to two
    lane requests fused client-side (see _query_requests_laned / _execute_laned),
    so a document occupies one slot per lane and its score stops depending on
    how many results the caller asked for.

    Escape hatch: RAG_TWIN_COLLAPSE=0 restores the legacy single-request
    server-side fusion path.
    """
    raw = (os.environ.get("RAG_TWIN_COLLAPSE") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _twin_identity(meta: dict) -> tuple | None:
    """Document identity shared by a parent and its twin, or None.

    Mirrors MultimodalRAG._twin_identity_key (rag_system) exactly:
    (source, page, chunk_index, time-window).  Kept as a local copy because
    vector_store must not import rag_system (rag_system imports this module).
    Video segments share (source, None, None) and are disambiguated by their
    time window.
    """
    src = meta.get("source") or ""
    if not src:
        return None
    return (
        str(src),
        meta.get("page"),
        meta.get("chunk_index"),
        (meta.get("timestamp_start"), meta.get("timestamp_end")),
    )


def _rrf_lane_term(pos: int, weight: float, k: int) -> float:
    """One lane RRF contribution at 0-indexed position pos.

    Matches qdrant-client weighted formula exactly
    (hybrid/fusion.py: 1 / ((pos + 1) / w + k - 1)) so client-side and
    server-side scores stay comparable.
    """
    if weight <= 0:
        return 0.0
    return 1.0 / ((pos + 1.0) / weight + k - 1.0)


def collapse_lane_by_twin(
    hits: "list[tuple[str, dict, float]]",
) -> "list[tuple[str, dict, float]]":
    """Collapse twin representations WITHIN one lane, keeping the better score.

    hits is [(point_id, metadata, similarity), ...] in the lane's own order.
    Returns the same shape, order preserved, with at most one entry per
    _twin_identity: the representative whose similarity is HIGHEST.

    This is the per-lane half of the fix.  It runs BEFORE fusion, so the lane
    contributes one candidate per document and its rank positions are not
    consumed by a sibling that will only be discarded afterwards.
    """
    best: dict[tuple, int] = {}
    keep = [True] * len(hits)
    for i, (pid, meta, sim) in enumerate(hits):
        ident = _twin_identity(meta or {}) or ("__point__", pid)
        j = best.get(ident)
        if j is None:
            best[ident] = i
        elif sim > hits[j][2]:
            keep[j] = False
            best[ident] = i
        else:
            keep[i] = False
    return [h for h, k in zip(hits, keep) if k]


def fuse_lanes_with_twin_collapse(
    dense_hits: "list[tuple[str, dict, float]]",
    sparse_hits: "list[tuple[str, dict, float]]",
    *,
    limit: int,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    k: int = _RRF_DEFAULT_K,
) -> "list[tuple[float, str, dict, int | None, int | None]]":
    """Per-lane twin collapse, then RRF over lane positions.

    dense_hits / sparse_hits are [(point_id, metadata, similarity), ...] in
    each lane's own rank order.  Each lane is collapsed first
    (:func:`collapse_lane_by_twin`), then positions are scored.

    Returns [(score, point_id, metadata, dense_pos, sparse_pos), ...] sorted
    best first, truncated to limit.  A document found by BOTH lanes merges
    into one entry carrying both positions -- including when the two lanes
    picked different representations of it, which is exactly what twins are
    for.
    """
    dense = collapse_lane_by_twin(dense_hits)
    sparse = collapse_lane_by_twin(sparse_hits)

    entries: dict[tuple, dict] = {}

    def _absorb(lane: str, pos: int, pid: str, meta: dict) -> None:
        ident = _twin_identity(meta or {}) or ("__point__", pid)
        cur = entries.get(ident)
        if cur is None:
            entries[ident] = {"pid": pid, "meta": meta, "d": None, "s": None}
            cur = entries[ident]
        key = "d" if lane == "dense" else "s"
        if cur[key] is None:
            cur[key] = pos

    for pos, (pid, meta, _sim) in enumerate(dense):
        _absorb("dense", pos, pid, meta)
    for pos, (pid, meta, _sim) in enumerate(sparse):
        _absorb("sparse", pos, pid, meta)

    scored: list[tuple[float, str, dict, int | None, int | None]] = []
    for cur in entries.values():
        score = 0.0
        if cur["d"] is not None:
            score += _rrf_lane_term(cur["d"], dense_weight, k)
        if cur["s"] is not None:
            score += _rrf_lane_term(cur["s"], sparse_weight, k)
        scored.append((score, cur["pid"], cur["meta"], cur["d"], cur["s"]))

    # Deterministic: score desc, then point id (stable across calls).
    scored.sort(key=lambda t: (-t[0], str(t[1])))
    return scored[:limit]

def _lightweight_payload_selector():
    """Payload selector that excludes heavy base64 ``image``/``video`` keys.

    Used on the fast search path (no reranker, no VLM, base LLM doesn't
    support image/video) to avoid transferring megabytes of base64 data
    from Qdrant that would be immediately discarded and replaced with
    ``preprocessed_*`` file refs.
    """
    from qdrant_client.models import PayloadSelectorExclude

    return PayloadSelectorExclude(exclude=["metadata.image", "metadata.video"])


# ---------------------------------------------------------------------------
# Metadata-filtered search (roadmap feature 1)
# ---------------------------------------------------------------------------

# The public filter dict accepted by the REST/MCP search surfaces.  Every key
# is optional; conditions are AND-combined:
#
#   file_types:     list[str] — metadata.file_type is one of these
#                   (the ``_classify_file`` labels: pdf, image, video, audio,
#                   text, json, table, code, office, html, xml, yaml,
#                   notebook, ebook, log, unknown)
#   severities:     list[str] — metadata.severities (log entries) contains ANY
#   source_prefix:  str       — metadata.source starts with this prefix
#   date_from:      str       — metadata.timestamp_start >= this (ISO datetime)
#   date_to:        str       — metadata.timestamp_start <= this (ISO datetime)
#
# Docs without the relevant metadata (e.g. video segments carry float
# seconds, not datetimes) simply never match a condition on that key.


@dataclass(frozen=True)
class RrfParams:
    """Caller-supplied weighted-RRF override for one query (feature: weighted RRF).

    Weights are positional by lane — ``dense_weight`` applies to the dense
    prefetch (S[0]) and ``sparse_weight`` to the BM25 prefetch (S[1]) — and
    are rank-space tilts, NOT score multipliers: qdrant computes
    ``1/((pos+1)/w + k − 1)`` per lane, so ``sparse_weight=3`` does not
    triple BM25 scores, it makes the sparse lane's rank positions count more
    ("trust the order, not the magnitude").

    When ``k`` is set here it pins the RRF ranking constant explicitly.  The
    default (``None``) is deliberate: today's unweighted
    ``FusionQuery(Fusion.RRF)`` inherits the client's implicit
    ``DEFAULT_RANKING_CONSTANT_K = 2``, and emitting an ``RrfQuery`` with a
    different k would silently change rank arithmetic on every hybrid query.

    Per-dataset defaults live OUT OF SCOPE for this slice (ruling, 2026-09):
    caller weights win uniformly, and only a caller-provided override may
    produce an ``RrfQuery`` — the default path stays byte-identical.
    """

    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    k: int | None = None


def build_payload_filter(filters: dict[str, Any] | None) -> Any:
    """Translate the public search-filter dict into a Qdrant ``Filter``.

    Returns ``None`` when *filters* is empty/None so callers can skip the
    filter entirely.  Unknown keys are ignored (forward compatibility).
    Raises ``ValueError`` on unparseable ``date_from``/``date_to`` values so
    API layers can surface a 400 instead of a silent no-op.
    """
    if not filters:
        return None
    from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny, MatchPrefix

    must: list[Any] = []

    file_types = [str(v) for v in (filters.get("file_types") or []) if str(v).strip()]
    if file_types:
        must.append(FieldCondition(key="metadata.file_type", match=MatchAny(any=file_types)))

    severities = [str(v) for v in (filters.get("severities") or []) if str(v).strip()]
    if severities:
        must.append(FieldCondition(key="metadata.severities", match=MatchAny(any=severities)))

    prefix = filters.get("source_prefix")
    if isinstance(prefix, str) and prefix.strip():
        must.append(FieldCondition(key="metadata.source", match=MatchPrefix(prefix=prefix.strip())))

    date_from, date_to = filters.get("date_from"), filters.get("date_to")
    if date_from or date_to:
        from datetime import datetime

        def _parse(value: Any, label: str) -> datetime:
            try:
                return datetime.fromisoformat(str(value))
            except ValueError as exc:
                raise ValueError(f"search filter '{label}' must be an ISO-8601 datetime, got {value!r}") from exc

        range_kwargs: dict[str, datetime] = {}
        if date_from:
            range_kwargs["gte"] = _parse(date_from, "date_from")
        if date_to:
            range_kwargs["lte"] = _parse(date_to, "date_to")
        must.append(FieldCondition(key="metadata.timestamp_start", range=DatetimeRange(**range_kwargs)))

    if not must:
        return None
    return Filter(must=must)


def filters_to_predicate(filters: dict[str, Any] | None) -> Any:
    """Python counterpart of :func:`build_payload_filter` for in-memory stores.

    Returns ``None`` when *filters* is empty/None, else a predicate over a
    metadata dict.  Dates are compared as parsed datetimes; metadata values
    that do not parse as datetimes (e.g. video segment seconds) never match a
    date condition — mirroring the Qdrant behaviour for non-datetime payload
    values.
    """
    if not filters:
        return None

    file_types = {str(v).strip() for v in (filters.get("file_types") or []) if str(v).strip()}
    severities = {str(v).strip() for v in (filters.get("severities") or []) if str(v).strip()}
    prefix = filters.get("source_prefix")
    prefix = prefix.strip() if isinstance(prefix, str) else ""
    date_from, date_to = filters.get("date_from"), filters.get("date_to")

    from datetime import datetime

    def _parse_dt(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return None  # media segment seconds are not datetimes
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    lo = _parse_dt(date_from) if date_from else None
    hi = _parse_dt(date_to) if date_to else None
    if (date_from and lo is None) or (date_to and hi is None):
        raise ValueError("search filter 'date_from'/'date_to' must be ISO-8601 datetimes")

    def predicate(meta: dict[str, Any]) -> bool:
        if file_types and str(meta.get("file_type")) not in file_types:
            return False
        if severities:
            values = meta.get("severities") or []
            if isinstance(values, str):
                values = [values]
            if not severities.intersection(str(v).strip() for v in values):
                return False
        if prefix and not str(meta.get("source") or "").startswith(prefix):
            return False
        if lo is not None or hi is not None:
            dt = _parse_dt(meta.get("timestamp_start"))
            if dt is None:
                return False
            if lo is not None and dt < lo:
                return False
            if hi is not None and dt > hi:
                return False
        return True

    return predicate


def ensure_search_payload_indexes(client: Any, collection_name: str) -> list[str]:
    """Create the payload indexes metadata-filtered search benefits from.

    Indexes only make filtering fast — absent indexes still filter correctly
    — so creation is best-effort: failures (older Qdrant without a schema
    type, transient unavailability) are logged and ignored.  Idempotent.
    """
    created: list[str] = []
    for field_name, schema in (
        ("metadata.file_type", "keyword"),
        ("metadata.severities", "keyword"),
        ("metadata.source", "keyword"),
        ("metadata.timestamp_start", "datetime"),
    ):
        try:
            client.create_payload_index(collection_name, field_name, field_schema=schema)
            created.append(field_name)
        except Exception as exc:  # pragma: no cover - depends on server version
            logger.debug("payload index %s on %s not created: %s", field_name, collection_name, exc)
    return created


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _points_to_docs(responses: Any) -> list[list[tuple[Document, float]]]:
    """Flatten ``query_batch_points`` responses into per-query result lists.

    Shared by every Qdrant query path so the payload → ``Document``
    reconstruction stays in one place.
    """
    all_results: list[list[tuple[Document, float]]] = []
    for resp in responses:
        results: list[tuple[Document, float]] = []
        for pt in resp.points:
            payload = pt.payload or {}
            meta = dict(payload.get("metadata", {}))
            # Private transport key (Wave-4): lets ``aretrieve`` back-fill the
            # tier-3 base64 media onto the surviving top_k docs after a
            # media-lite rerank.  Stripped again before results reach any
            # caller — never part of API/MCP responses.
            meta["_point_id"] = str(pt.id)
            doc = Document(page_content=payload.get("page_content", ""), metadata=meta)
            results.append((doc, float(pt.score)))
        all_results.append(results)
    return all_results


def _is_fusion_unsupported_error(exc: BaseException) -> bool:
    """True when *exc* unmistakably signals a backend without the fusion query API.

    Conservative by design: only a ``NotImplementedError``, or a
    request-shaped rejection (``ValueError`` from the embedded local mode,
    a 4xx ``UnexpectedResponse`` from a server) whose message mentions the
    query-API surface, counts as "fusion not supported".  Unrelated
    failures — network, auth, timeouts — must propagate, never silently
    downgrade every subsequent search to dense-only.
    """
    if isinstance(exc, NotImplementedError):
        return True
    from qdrant_client.http.exceptions import UnexpectedResponse

    if not isinstance(exc, (ValueError, UnexpectedResponse)):
        return False
    message = str(exc).lower()
    return any(
        marker in message for marker in ("fusion", "rrf", "prefetch", "not supported", "unsupported", "not implemented")
    )


class VectorStore:
    """Minimal base class so ``isinstance(vs, VectorStore)`` keeps working.

    Concrete implementations are :class:`InMemoryVectorStore` and
    :class:`QdrantVectorStore`. The shared retrieval surface
    (``similarity_search_with_score_by_vector`` /
    ``asimilarity_search_with_relevance_scores``) is declared here; scores are
    raw cosine similarity (higher = more similar) for both backends.
    """

    def _lane_requests(
        self,
        emb: list[float],
        sparse: Any,
        k: int,
        with_payload: Any,
        flt: Any,
        want_dense_vectors: bool,
    ) -> "list[tuple[Any, bool]]":
        """Per-lane requests for twin-collapsing hybrid search.

        Emits TWO flat requests (dense, then sparse) instead of one server-side
        RRF fusion request.  The caller fuses them client-side via
        fuse_lanes_with_twin_collapse, which is what makes per-lane twin
        collapse possible at all: server-side RRF fuses by point id, so a
        parent and its twin arrive as two independent candidates and cannot be
        collapsed before scoring.

        Both lanes return prefetch_k candidates (the same floor the fusion
        request used) and both carry the dense vector so the true cosine can
        be recomputed for sparse-lane-only hits.
        """
        from qdrant_client.models import QueryRequest

        prefetch_k = _prefetch_depth(k)
        dense_using = self._dense_using()
        vec: Any = ([dense_using] if dense_using is not None else True) if want_dense_vectors else False
        return [
            (
                QueryRequest(
                    query=emb,
                    using=dense_using,
                    limit=prefetch_k,
                    with_payload=with_payload,
                    with_vector=vec,
                    filter=flt,
                ),
                True,
            ),
            (
                QueryRequest(
                    query=sparse,
                    using=bm25_lane.BM25_VECTOR_NAME,
                    limit=prefetch_k,
                    with_payload=with_payload,
                    with_vector=vec,
                    filter=flt,
                ),
                True,
            ),
        ]
    def _query_requests_laned(
        self,
        queries: "list[tuple[list[float], int, bool, dict[str, Any] | None, str | None, RrfParams | None]]",
        hybrid: bool,
    ) -> "list[list[tuple[Any, bool]]]":
        """Like :meth:`_query_requests`, but one request LIST per query.

        Hybrid queries emit TWO requests (dense lane, sparse lane) instead of
        one server-side RRF fusion request; the caller fuses them client-side
        so twins can be collapsed per lane BEFORE scoring.  Non-hybrid queries
        emit the usual single flat dense request.

        Returning a list-of-lists (rather than a flat list) is what lets the
        executor map responses back to their query when the request count is
        no longer 1:1.
        """
        from qdrant_client.models import QueryRequest

        stats = self._bm25_stats() if hybrid else None
        lightweight = _lightweight_payload_selector()
        want_dense_vectors = self._embedding_scores_enabled()
        grouped: list[list[tuple[Any, bool]]] = []

        for emb, k, need_media, filters, query_text, rrf in queries:
            flt = build_payload_filter(filters)
            with_payload: Any = True if need_media else lightweight
            sparse = None
            if stats is not None and query_text:
                try:
                    sparse = bm25_lane.build_query_sparse_vector(query_text, stats)
                except Exception as exc:
                    logger.warning("BM25 query vector build failed - dense-only for this query: %s", exc)
            if sparse is not None:
                grouped.append(self._lane_requests(emb, sparse, k, with_payload, flt, want_dense_vectors))
            else:
                grouped.append(
                    [
                        (
                            QueryRequest(
                                query=emb,
                                using=self._dense_using(),
                                limit=k,
                                with_payload=with_payload,
                                with_vector=False,
                                filter=flt,
                            ),
                            False,
                        )
                    ]
                )
        return grouped
    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int, need_media: bool = True, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        raise NotImplementedError

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        need_media: bool = True,
        filters: dict[str, Any] | None = None,
        rrf: RrfParams | None = None,
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """In-process store mirroring the former langchain ``InMemoryVectorStore`` surface.

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

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int, need_media: bool = True, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        entries = [(s["document"], s["vector"]) for s in self.store.values() if s.get("vector") is not None]
        if entries and filters is not None:
            pred = filters_to_predicate(filters)
            if pred is not None:
                entries = [(d, v) for d, v in entries if pred(d.metadata)]
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

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        need_media: bool = True,
        filters: dict[str, Any] | None = None,
        rrf: RrfParams | None = None,
    ) -> list[tuple[Document, float]]:
        """In-memory fallback: no sparse lane exists, so a weighted-RRF
        override has nothing to weight — the dense result is returned
        unstamped (honesty rule: dense results never carry weights)."""
        emb = await self.embedding.aembed_query(query)
        return self.similarity_search_with_score_by_vector(emb, k, need_media=need_media, filters=filters)


class _QdrantBatcher:
    """Dynamic micro-batcher for concurrent Qdrant similarity searches.

    When multiple search requests call ``asimilarity_search_with_score_by_vector``
    concurrently, each normally fires its own gRPC ``query_batch_points`` call.
    While Qdrant is fast (~16ms/query), the per-call gRPC overhead and thread
    pool contention add up under high concurrency.

    This batcher collects ``(embedding, k, need_media, filters, query_text,
    rrf)`` requests arriving within a short window (default 5 ms) or until
    ``max_batch_size`` accumulate, then sends
    them as a single ``query_batch_points`` call with N ``QueryRequest``
    objects.  Qdrant processes the batch in one round-trip and returns
    results in the same order as the requests.

    Each caller awaits a ``Future`` and receives its individual result list
    — the batching is transparent.
    """

    def __init__(
        self,
        search_fn: "Callable[[list[tuple[list[float], int, bool, dict[str, Any] | None, str | None, RrfParams | None]]], list[list[tuple[Document, float]]]]",
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
    ) -> None:
        self._search_fn = search_fn
        self._max_batch_size = max(1, max_batch_size)
        self._max_wait = max(0.001, max_wait_ms / 1000.0)
        self._queue: list[
            tuple[
                list[float],
                int,
                bool,
                dict[str, Any] | None,
                str | None,
                RrfParams | None,
                asyncio.Future[list[tuple[Document, float]]],
            ]
        ] = []
        self._lock: asyncio.Lock | None = None
        self._flush_task: asyncio.Task[None] | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def submit(
        self,
        embedding: list[float],
        k: int,
        need_media: bool = True,
        filters: dict[str, Any] | None = None,
        query_text: str | None = None,
        rrf: RrfParams | None = None,
    ) -> list[tuple[Document, float]]:
        """Submit a single query and await its results.

        *query_text* is the raw text query when the caller has one (text
        retrieval) — it lets the store build the BM25 sparse query vector
        for hybrid fusion.  Vector-supplied queries (multimodal) pass
        ``None`` and stay dense-only.  *rrf* is the caller's optional
        weighted-RRF override (ignored on dense-only paths — a request with
        no query text can never fuse).
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[list[tuple[Document, float]]] = loop.create_future()
        flush_now = False

        async with self._get_lock():
            self._queue.append((embedding, k, need_media, filters, query_text, rrf, fut))
            if len(self._queue) >= self._max_batch_size:
                flush_now = True
            elif self._flush_task is None:
                self._flush_task = loop.create_task(self._timed_flush())

        if flush_now:
            await self._flush()

        return await fut

    async def _timed_flush(self) -> None:
        try:
            await asyncio.sleep(self._max_wait)
        except asyncio.CancelledError:
            return
        await self._flush(caller_task=asyncio.current_task())

    async def _flush(self, caller_task: "asyncio.Task[None] | None" = None) -> None:
        """Drain the queue and send one batched Qdrant query."""
        async with self._get_lock():
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()
            if self._flush_task is not None and not self._flush_task.done() and self._flush_task is not caller_task:
                self._flush_task.cancel()
            self._flush_task = None

        queries = [(emb, k, need_media, filters, text, rrf) for emb, k, need_media, filters, text, rrf, _ in batch]
        futs = [f for _, _, _, _, _, _, f in batch]

        loop = asyncio.get_running_loop()
        exec_fut = loop.run_in_executor(_QDRANT_IO_POOL, self._search_fn, queries)

        # Settle every caller's future from the executor result — even if the
        # task that triggered this flush is cancelled (e.g. a client disconnect),
        # so no queued caller is left hanging forever.
        def _settle(f: "asyncio.Future[list[list[tuple[Document, float]]]]") -> None:
            try:
                results = f.result()
                if len(results) != len(futs):
                    raise RuntimeError(f"Batched Qdrant query returned {len(results)} results for {len(futs)} queries")
                for fut, result in zip(futs, results):
                    if not fut.done():
                        fut.set_result(result)
            except BaseException as exc:  # settle callers regardless
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(exc)

        exec_fut.add_done_callback(_settle)
        await asyncio.shield(exec_fut)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store wrapping ``qdrant_client`` directly.

    Exposes ``_client`` / ``collection_name`` / ``vector_name`` for the direct
    upsert / scroll / query_batch_points / delete calls in ``rag_system.py``
    (and ``dataset_manager`` / ``api_server`` / ``mcp_server``), plus the
    similarity-search helpers used at retrieval time.

    The collection is expected to store payloads with ``page_content`` and
    ``metadata`` keys (the layout ``rag_system._build_qdrant_vector_store``
    writes), matching the former langchain-qdrant defaults.

    Hybrid retrieval (roadmap feature 2): when the collection carries the
    named ``bm25`` sparse vector, text queries are issued as RRF-fusion
    requests over a dense prefetch (``using=vector_name``) and a sparse
    prefetch (``using=bm25``).  ``bm25_stats_path`` points at the dataset's
    ``.bm25_stats.json`` sidecar (df counts the query vector is weighted
    with); ``None`` — e.g. stores built directly by tests — keeps the flat
    dense-only query, as do legacy unnamed-vector collections and
    vector-supplied (multimodal) queries.
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embedding: Any,
        vector_name: str | None = None,
        bm25_stats_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        import threading

        self._client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.vector_name = vector_name
        self.bm25_stats_path = bm25_stats_path
        # Serializes Qdrant calls on the in-process LOCAL backend, which has
        # no internal locking — concurrent writers (the Wave-4 parallel ingest
        # workers + the batch consumer) corrupt its state ("bad parameter or
        # other API misuse").  See :meth:`_local_write_guard`.
        self._write_lock = threading.Lock()
        # Feature-detection caches (None = not probed yet).  Collection
        # schema never changes short of a drop+recreate, and a fusion
        # failure against a given backend is permanent — both are probed
        # once per store instance.
        self._bm25_capable: bool | None = None
        self._fusion_supported: bool | None = None
        # Negative capability results are cached only briefly: a store built
        # before a Recreate (on this or any other worker/pod) must re-probe
        # and adopt the rebuilt collection's bm25 lane without a restart.
        self._bm25_probed_at: float = 0.0
        self._reprobe_s = max(1, int(os.environ.get("RAG_HYBRID_REPROBE_S", "60")))
        # The dense lane's detected vector NAME on bm25-capable collections -
        # lets stores constructed against the legacy unnamed vector target the
        # named dense lane correctly after a Recreate.
        self._dense_vector_name: str | None = None
        # True once a capability probe SUCCEEDED (whether or not it found a
        # named dense vector): the collection's shape is then KNOWN, so an
        # unnamed collection can safely target the server default vector
        # (``using=None``) instead of the "dense" fallback below.
        self._dense_name_probed: bool = False
        # Accept and ignore legacy kwargs (content_payload_key,
        # metadata_payload_key, distance_strategy, ...) for forward-compat.
        self._extra_kwargs = kwargs
        # Dynamic query batcher for concurrent similarity searches.
        # One batcher per event loop — same pattern as the embedding batcher
        # in model_adapters.py.
        self._batcher_max_size = int(os.environ.get("QDRANT_QUERY_BATCH_SIZE", "32"))
        self._batcher_max_wait_ms = float(os.environ.get("QDRANT_QUERY_BATCH_WAIT_MS", "5"))
        self._batchers: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _QdrantBatcher] = (
            weakref.WeakKeyDictionary()
        )

    def _local_write_guard(self) -> Any:
        """Context manager serializing Qdrant calls on the in-process LOCAL backend.

        qdrant-client's local mode has no internal locking: concurrent calls
        (the Wave-4 parallel ingest workers + the batch consumer, or any
        multi-threaded caller) corrupt its state — observed as
        ``sqlite3.InterfaceError: bad parameter or other API misuse`` under
        parallel ingests.  The remote HTTP/gRPC clients are thread-safe, so
        this is a transparent null context there.  Acquire per client call
        (never across an ``await``): the lock is taken inside the executor
        thread.

        Detection note: ``_client`` is qdrant_client's generic ``QdrantClient``
        wrapper; its ``__module__`` never says "local", so unwrap the inner
        backend (``_client._client`` → ``QdrantLocal``) before checking.
        """
        import contextlib

        backend = self._client
        inner = getattr(backend, "_client", None)
        if inner is not None and type(inner).__module__.startswith("qdrant_client.local"):
            return self._write_lock
        try:
            if type(backend).__module__.startswith("qdrant_client.local"):
                return self._write_lock
        except Exception:  # pragma: no cover - defensive
            pass
        return contextlib.nullcontext()

    # -- hybrid (dense + BM25) support ---------------------------------------

    def supports_hybrid(self) -> bool:
        """True when the collection has a `bm25` sparse vector to fuse against.
        Probed via `get_collection` (one round trip, then cached).  The positive
        result is cached permanently - collections only ever GAIN the bm25
        lane through Recreate, which rebuilds it with the lane present.  The
        NEGATIVE result is re-probed after `RAG_HYBRID_REPROBE_S` (default
        60s): stores built before a Recreate - on this or any other
        worker/pod, whose RAG caches the recreate POST cannot clear - then
        self-heal onto the hybrid lane without a restart.

        Requires a *named* dense vector as well - the unnamed default vector
        of legacy collections cannot host a second lane, so those keep the
        flat dense query.  When a named dense vector is detected its NAME
        is remembered (_dense_vector_name) so stores constructed against a
        legacy collection keep targeting the correct lane after a
        Recreate.
        """
        now = time.time()
        if self._bm25_capable is not None and (self._bm25_capable or now - self._bm25_probed_at < self._reprobe_s):
            return self._bm25_capable
        capable = False
        dense_name: str | None = None
        try:
            params = self._client.get_collection(self.collection_name).config.params
            sparse = getattr(params, "sparse_vectors", None) or {}
            vectors = getattr(params, "vectors", None)
            if isinstance(vectors, dict):
                for name in vectors:
                    if name != bm25_lane.BM25_VECTOR_NAME:
                        dense_name = name
                        break
            capable = bool(sparse.get(bm25_lane.BM25_VECTOR_NAME)) and dense_name is not None
        except Exception as exc:
            # Transient failure (observed live: probes fired during the
            # qdrant consensus outage of 2026-09-11 failed cluster-wide and
            # healed minutes later).  A failed probe must NOT be cached —
            # caching turned a 60-second outage into a permanent capability
            # regression on every worker that probed during the window.
            logger.warning(
                "Hybrid capability probe failed for %s (will retry on the next call): %s",
                self.collection_name,
                exc,
            )
            return False
        self._bm25_capable = capable
        self._dense_vector_name = dense_name
        # A successful probe means the collection shape is now known — the
        # unnamed-default case included (otherwise _dense_using() could not
        # tell "confirmed unnamed" apart from "probe never ran / failed").
        self._dense_name_probed = True
        self._bm25_probed_at = now
        return capable

    def _dense_using(self) -> str | None:
        """The dense lane using-target: the detected named vector on
        bm25-capable collections; the store's configured name; ``None``
        (= server default vector) on collections a successful probe
        confirmed to use the UNNAMED default vector; else the "dense"
        fallback.

        The trailing ``or "dense"`` is the v3.6.1 fix for the live failure
        where BOTH terms resolved falsy (probe failed during an outage +
        no configured name) and qdrant received ``using=""`` — failing every
        batched dataset search with "Not existing vector name" (2026-09-11:
        ~30 errors/10min while ALL collections were named-dense, so the
        fallback is safe fleet-wide).

        ``"dense"`` is only correct for NAMED-vector collections though —
        sending it for the legacy unnamed default vector fails with
        "Dense vector dense is not found in the collection" (local mode /
        "Not existing vector name" on the server), which broke every
        search against a legacy collection.  A successful capability probe
        tells the two shapes apart, so a confirmed-unnamed collection now
        targets ``None`` (the server default) and the "dense" fallback
        stays reserved for the genuinely-unknown case (probe failed or
        never ran — the outage scenario the fallback was added for).
        """
        if self._dense_vector_name:
            return self._dense_vector_name
        if self.vector_name:
            return self.vector_name
        if self._dense_name_probed:
            return None
        return "dense"

    @staticmethod
    def _embedding_scores_enabled() -> bool:
        """Whether hybrid results should carry the true dense cosine.

        Recomputed client-side from the dense vector the fusion response
        carries back (``with_vector`` — Qdrant fetches vectors by id after
        fusing, so this costs no extra query and no second HNSW traversal).
        Disable via ``RAG_HYBRID_EMBEDDING_SCORES=0`` to skip the vector
        transfer (very large k / very high-dimensional embedders).
        """
        return os.environ.get("RAG_HYBRID_EMBEDDING_SCORES", "1").strip().lower() not in ("0", "false", "no", "off")

    def _stamp_dense_cosines(
        self,
        query_emb: list[float],
        points: list[Any],
        per_query: list[tuple[Document, float]],
    ) -> None:
        """Recompute the dense cosine client-side and stamp it on each doc.

        The fusion response's points arrive in the same order as
        ``per_query`` (both are built from the same response), so pairing is
        positional — no point-id plumbing needed.  Best-effort: any shape or
        numeric surprise (dimension drift, unexpected vector type, zero
        vector) leaves the score unset and consumers show it as unknown.
        """
        q = np.asarray(query_emb, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if not np.isfinite(qnorm) or qnorm == 0.0:
            return
        dense_name = self._dense_using()
        for pt, (doc, _fused) in zip(points, per_query):
            vec = pt.vector
            if isinstance(vec, dict):
                vec = vec.get(dense_name)
            if not isinstance(vec, (list, tuple)) or len(vec) != len(q):
                continue
            try:
                v = np.asarray(vec, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            vnorm = float(np.linalg.norm(v))
            if not np.isfinite(vnorm) or vnorm == 0.0:
                continue
            cos = float(np.dot(q, v) / (qnorm * vnorm))
            if np.isfinite(cos):
                doc.metadata["_embedding_score"] = round(cos, 4)

    def _bm25_stats(self) -> dict[str, Any] | None:
        """Fresh df stats for the query weighting, or ``None`` when unusable.

        ``None`` for an empty index too: with no sparse vectors stored yet
        the sparse prefetch could only return nothing, so the request stays
        flat dense until the first ingest lands.
        """
        if not self.bm25_stats_path:
            return None
        try:
            stats = bm25_lane.load_stats(Path(self.bm25_stats_path))
        except Exception as exc:
            logger.debug("BM25 stats load failed for %s: %s", self.bm25_stats_path, exc)
            return None
        return stats if stats.get("n_docs", 0) > 0 else None

    def _query_requests(
        self,
        queries: list[tuple[list[float], int, bool, dict[str, Any] | None, str | None, RrfParams | None]],
        hybrid: bool,
    ) -> list[tuple[Any, bool]]:
        """Build one ``QueryRequest`` per queued query.

        Returns ``(request, is_hybrid)`` pairs.  Hybrid requests are the
        fusion form — ``prefetch=[dense, sparse], fusion=RRF`` — with the
        filter pushed into *both* prefetches so each lane searches only the
        filtered subset; anything that cannot go hybrid (no query text, no
        usable stats, env off, fusion known-unsupported) stays the flat
        dense request.  Fusion is per-request, so a batch mixes freely.

        Weighted RRF: when the query carries an :class:`RrfParams` override,
        the hybrid request emits ``RrfQuery(rrf=Rrf(k=…, weights=[dense,
        sparse]))`` instead of the plain ``FusionQuery`` — the weights are
        positional (dense=S[0], sparse=S[1]).  Without an override the
        request shape is byte-identical to the historical fusion request
        (the ``FusionQuery`` form carries the client's implicit
        ``DEFAULT_RANKING_CONSTANT_K = 2``; pinning a k here would silently
        change rank arithmetic on every default query — the k-default trap).
        """
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, QueryRequest, Rrf, RrfQuery

        stats = self._bm25_stats() if hybrid else None
        lightweight = _lightweight_payload_selector()
        out: list[tuple[Any, bool]] = []
        want_dense_vectors = self._embedding_scores_enabled()
        for emb, k, need_media, filters, query_text, rrf in queries:
            flt = build_payload_filter(filters)
            with_payload: Any = True if need_media else lightweight
            sparse = None
            if stats is not None and query_text:
                try:
                    sparse = bm25_lane.build_query_sparse_vector(query_text, stats)
                except Exception as exc:
                    # The lexical lane must never take retrieval down —
                    # degrade this one query to dense-only, audibly.
                    logger.warning("BM25 query vector build failed — dense-only for this query: %s", exc)
            if sparse is not None:
                # Hoisted so the None-check below narrows the same value mypy sees.
                dense_using = self._dense_using()
                # Weighted override → parameterized RrfQuery; anything else
                # keeps the default fusion form (byte-identical request).
                # A weights list of [1.0, 1.0] with no k pinned is rank-
                # arithmetic-identical to the default form, so it must not
                # switch the request shape either.
                if isinstance(rrf, RrfParams) and (
                    rrf.k is not None or rrf.dense_weight != 1.0 or rrf.sparse_weight != 1.0
                ):
                    fused_query: Any = RrfQuery(rrf=Rrf(k=rrf.k, weights=[rrf.dense_weight, rrf.sparse_weight]))
                else:
                    fused_query = FusionQuery(fusion=Fusion.RRF)
                # Prefetch DEPTH vs DISPLAY limit.  RRF is defined over its
                # candidate lists, so the prefetch window is part of the
                # RANKING DEFINITION, not of the paging.  Binding both
                # prefetches to `k` made a document's fused score depend on
                # how many results the caller asked to SEE.
                #
                # Worked example, unweighted (w=1.0, k=2), so a lane's
                # position p (1-indexed) contributes 1/(p+1): a doc that is
                # dense rank 11 and sparse rank 1 scores
                #   1/12 + 1/2 = 0.0833 + 0.5000 = 0.5833
                # when both lanes are prefetched deeply enough.  At top_k=10
                # the dense lane cannot return position 11 at all, so the
                # dense term vanishes and the doc scores 0.5000 — appearing
                # to be a pure sparse hit.  Same document, same query,
                # different score, and the shallow view misreports its lane.
                #
                # Decouple them: fuse over a fixed floor of candidates, then
                # truncate to k.  For every k <= the floor the fused score is
                # identical, and top-10 is a strict prefix of top-20.
                #
                # Honest limitations: (a) RRF scores remain relative to the
                # pool, so invariance holds only within the floor; (b) the
                # rerank path legitimately over-fetches (rag_system.aretrieve
                # uses fetch_k = max(top_k, 4*reranker_top_k)), so a reranked
                # call may fuse over a deeper pool than an unranked one.
                prefetch_k = _prefetch_depth(k)
                out.append(
                    (
                        QueryRequest(
                            query=fused_query,
                            prefetch=[
                                Prefetch(query=emb, using=self._dense_using(), limit=prefetch_k, filter=flt),
                                Prefetch(
                                    query=sparse, using=bm25_lane.BM25_VECTOR_NAME, limit=prefetch_k, filter=flt
                                ),
                            ],
                            limit=k,
                            with_payload=with_payload,
                            # Carrying the stored dense vectors back on the SAME
                            # fusion response (fetched by id post-fusion) lets the
                            # caller recompute the true dense cosine client-side —
                            # no extra query, no second HNSW traversal.
                            # Carry the stored dense vector back on the fusion
                            # response (see below).  On a confirmed-unnamed
                            # collection there is no name to ask for — True
                            # returns the single default vector; ``[None]``
                            # would fail qdrant-client request validation.
                            with_vector=([dense_using] if dense_using is not None else True)
                            if want_dense_vectors
                            else False,
                        ),
                        True,
                    )
                )
            else:
                out.append(
                    (
                        QueryRequest(
                            query=emb,
                            using=self._dense_using(),
                            limit=k,
                            with_payload=with_payload,
                            with_vector=False,
                            filter=flt,
                        ),
                        False,
                    )
                )
        return out

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int, need_media: bool = True, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Flat dense-only search (no query text available on this path).

        Text queries go through :meth:`asimilarity_search_with_relevance_scores`
        / the batcher, which is where hybrid fusion happens; this direct
        vector-supplied entry point (multimodal queries, dedup probes) keeps
        today's single-lane request shape.

        *need_media* is honoured here too (Wave-4): the default ``True``
        fetches the full payload exactly as before, while ``False`` applies
        the lightweight payload selector — dedup probes only need the text +
        non-media metadata, so megabytes of tier-3 base64 image/video stay in
        Qdrant instead of crossing the wire per probe.
        """
        from qdrant_client.models import QueryRequest

        from multimodal_rag.utils.metrics import observe_qdrant

        # Resolve the collection shape before naming the dense lane: on a
        # legacy unnamed-vector collection the using-target must be the
        # server default (None), not the "dense" fallback (which only
        # exists for named-dense collections whose probe failed).  Cached
        # by supports_hybrid() — one round trip per store, then free.
        self.supports_hybrid()
        with_payload: Any = True if need_media else _lightweight_payload_selector()
        with observe_qdrant("query_batch_points"):
            responses = self._client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    QueryRequest(
                        query=embedding,
                        using=self._dense_using(),
                        limit=k,
                        with_payload=with_payload,
                        with_vector=False,
                        filter=build_payload_filter(filters),
                    )
                ],
            )
        # Exactly one request was submitted — unwrap the single per-query
        # result list (_points_to_docs returns one list per response).
        return _points_to_docs(responses)[0]

    def similarity_search_with_score_by_vector_batch(
        self, queries: list[tuple[list[float], int, bool, dict[str, Any] | None, str | None, RrfParams | None]]
    ) -> list[list[tuple[Document, float]]]:
        """Batch-execute multiple queries in a single Qdrant gRPC call.

        *queries* is a list of ``(embedding, k, need_media, filters,
        query_text, rrf)`` tuples.  When *need_media* is False, heavy base64
        ``image``/``video`` payload keys are excluded from the Qdrant
        response to avoid transferring megabytes of data that would be
        discarded on the fast path.  Returns a list of result lists, one
        per query, in the same order.

        Hybrid: text queries on a bm25-capable collection (with
        ``RAG_HYBRID_SEARCH`` on and non-empty df stats) become RRF-fusion
        requests.  Backends without the fusion query API (older servers,
        older qdrant-client local mode) are detected ONCE — the first
        failure whose exception unmistakably signals the feature gap — and
        the store permanently degrades to dense-only instead of retrying
        per query; every other exception propagates.
        """
        from multimodal_rag.utils.metrics import SEARCH_HYBRID, observe_qdrant

        hybrid = self._fusion_supported is not False and self.supports_hybrid() and bm25_lane.hybrid_search_enabled()
        requests = self._query_requests(queries, hybrid)
        any_hybrid = any(flag for _, flag in requests)

        def _run(reqs: list[Any]) -> list[Any]:
            with observe_qdrant("query_batch_points"):
                return self._client.query_batch_points(collection_name=self.collection_name, requests=reqs)
        use_lanes = hybrid and _twin_collapse_enabled()
        if use_lanes:
            # Twin-collapsing path: two lane requests per hybrid query, fused
            # client-side so twins are collapsed PER LANE before scoring.
            grouped = self._query_requests_laned(queries, hybrid)
            try:
                results = self._execute_laned(queries, grouped)
            except Exception as exc:
                if not _is_fusion_unsupported_error(exc):
                    raise
                self._fusion_supported = False
                logger.warning(
                    "Qdrant backend rejected per-lane hybrid requests (%s: %s) - "
                    "collection '%s' falls back to dense-only retrieval",
                    type(exc).__name__,
                    exc,
                    self.collection_name,
                )
                fallback = self._query_requests(queries, False)
                for _rq, _h in fallback:
                    SEARCH_HYBRID.labels(mode="dense").inc()
                fb_results = _points_to_docs(_run([req for req, _ in fallback]))
                # Same deterministic tiebreak the legacy dense path applies, so
                # a degraded search is reproducible too.
                return [_stable_tiebreak(pq) for pq in fb_results]
            for group in grouped:
                for _rq, _h in group:
                    SEARCH_HYBRID.labels(mode="hybrid" if _h else "dense").inc()
            for q_idx, per_query in enumerate(results):
                rrf_requested = queries[q_idx][5]
                applied = bool(
                    isinstance(rrf_requested, RrfParams)
                    and (
                        rrf_requested.k is not None
                        or rrf_requested.dense_weight != 1.0
                        or rrf_requested.sparse_weight != 1.0
                    )
                )
                for doc, _ in per_query:
                    doc.metadata["_score_kind"] = "rrf"
                    if applied:
                        assert rrf_requested is not None
                        doc.metadata["_rrf"] = {
                            "dense": rrf_requested.dense_weight,
                            "sparse": rrf_requested.sparse_weight,
                            "k": rrf_requested.k,
                        }
                results[q_idx] = _stable_tiebreak(per_query)
            return results


        try:
            executed = requests
            responses = _run([req for req, _ in executed])
        except Exception as exc:
            if not any_hybrid or not _is_fusion_unsupported_error(exc):
                raise
            # Older backend without prefetch/fusion: mark the store
            # dense-only permanently (probe once, not per query) and retry.
            self._fusion_supported = False
            logger.warning(
                "Qdrant backend does not support prefetch/fusion queries (%s: %s) — "
                "collection '%s' falls back to dense-only retrieval",
                type(exc).__name__,
                exc,
                self.collection_name,
            )
            executed = self._query_requests(queries, False)
            responses = _run([req for req, _ in executed])

        # Count what actually ran — a hybrid-intent request that fell back is
        # a dense search (the fallback label would otherwise overstate fusion).
        for _, is_hybrid in executed:
            SEARCH_HYBRID.labels(mode="hybrid" if is_hybrid else "dense").inc()

        results = _points_to_docs(responses)
        # Annotate hybrid-lane results.  An RRF-fusion score is rank
        # arithmetic (Σ 1/(rank+2) over the lanes) — NOT a cosine similarity —
        # so every consumer that displays or thresholds these scores must be
        # able to tell them apart ("_score_kind").  Dense-only results keep
        # raw cosine and stay unstamped (consumers default to "cosine").
        #
        # When RAG_HYBRID_EMBEDDING_SCORES is on (default), the fusion request
        # also carries back each point's stored dense vector, and the TRUE
        # dense cosine is recomputed here against the query embedding we
        # already hold — including for points the dense lane never ranked
        # (sparse-only hits), which is exactly the score fusion hides.
        want_vectors = self._embedding_scores_enabled()
        for q_idx, ((_, is_hybrid), per_query) in enumerate(zip(executed, results)):
            if not is_hybrid:
                # Dense-degraded path: the request never fused, so any
                # weighted-RRF override was silently dropped — results stay
                # UNSTAMPED (an honest dense result carries no weights, and
                # the surfaces report ``rrf.applied=false`` from that).
                #
                # The deterministic tiebreak still applies: raw cosine ties
                # are common for duplicate/near-identical media, and this is
                # the most common path on a cold dataset (no BM25 stats yet,
                # hybrid disabled, or a backend that degraded to dense-only).
                results[q_idx] = _stable_tiebreak(per_query)
                continue
            rrf_requested = queries[q_idx][5]
            applied_override = bool(
                isinstance(rrf_requested, RrfParams)
                and (
                    rrf_requested.k is not None
                    or rrf_requested.dense_weight != 1.0
                    or rrf_requested.sparse_weight != 1.0
                )
            )
            for doc, _ in per_query:
                doc.metadata["_score_kind"] = "rrf"
                if applied_override:
                    # RRF is rank arithmetic — a rank-space tilt, not a score
                    # multiplier.  Stamp the exact weights/k the executed
                    # request used so consumers can attribute ordering.
                    assert rrf_requested is not None
                    doc.metadata["_rrf"] = {
                        "dense": rrf_requested.dense_weight,
                        "sparse": rrf_requested.sparse_weight,
                        "k": rrf_requested.k,
                    }
            if want_vectors:
                self._stamp_dense_cosines(queries[q_idx][0], responses[q_idx].points, per_query)
            # Deterministic tiebreak on EQUAL fused scores.
            #
            # qdrant-client's fusion sorts on the score alone
            # (hybrid/fusion.py: sorted(scores.items(), key=item[1], reverse=True)),
            # and equal fused scores are common and STRUCTURAL: the dense and
            # sparse ladders both produce exactly 0.3333, 0.2000, 0.1429 ...
            # so a dense-pos-2 hit and a sparse-pos-1 hit tie exactly.  With
            # no secondary key, the surviving order is whatever Qdrant
            # returned for the equal-scored points, which is not guaranteed —
            # observed as two documents swapping places between identical
            # calls.  Sorting here (score desc, then a stable content key)
            # makes repeated identical searches reproducible.
            results[q_idx] = _stable_tiebreak(per_query)
        return results

    def _execute_laned(
        self,
        queries: "list[tuple[list[float], int, bool, dict[str, Any] | None, str | None, RrfParams | None]]",
        grouped: "list[list[tuple[Any, bool]]]",
    ) -> "list[list[tuple[Document, float]]]":
        """Run per-lane requests and fuse them client-side with twin collapse.

        Each query contributes either ONE dense request (dense-only path, no
        fusion, raw cosine scores) or TWO lane requests (dense + sparse) that
        are collapsed per lane, then RRF-scored by
        :func:`fuse_lanes_with_twin_collapse`.

        Returns the same shape as the fused path: one list of
        ``(Document, fused_score)`` per query, best first.
        """
        from multimodal_rag.utils.metrics import observe_qdrant

        flat: list[Any] = [req for group in grouped for req, _ in group]
        with observe_qdrant("query_batch_points"):
            responses = self._client.query_batch_points(collection_name=self.collection_name, requests=flat)

        out: list[list[tuple[Document, float]]] = []
        cursor = 0
        for q_idx, group in enumerate(grouped):
            n = len(group)
            chunk = responses[cursor : cursor + n]
            cursor += n
            k = queries[q_idx][1]
            rrf_req = queries[q_idx][5]

            if n == 1 and not group[0][1]:
                # Dense-only: no fusion, scores are raw cosine.  Same shape and
                # semantics as the legacy flat path.
                out.append(_points_to_docs(chunk)[0])
                continue

            dense_docs = _points_to_docs([chunk[0]])[0]
            sparse_docs = _points_to_docs([chunk[1]])[0]

            dense_hits = [
                (doc.metadata.get("_point_id", ""), doc.metadata, float(score)) for doc, score in dense_docs
            ]
            sparse_hits = [
                (doc.metadata.get("_point_id", ""), doc.metadata, float(score)) for doc, score in sparse_docs
            ]

            dw = rrf_req.dense_weight if isinstance(rrf_req, RrfParams) else 1.0
            sw = rrf_req.sparse_weight if isinstance(rrf_req, RrfParams) else 1.0
            kk = rrf_req.k if isinstance(rrf_req, RrfParams) and rrf_req.k is not None else _RRF_DEFAULT_K

            fused = fuse_lanes_with_twin_collapse(
                dense_hits, sparse_hits, limit=k, dense_weight=dw, sparse_weight=sw, k=kk
            )

            by_id = {doc.metadata.get("_point_id", ""): doc for doc, _s in dense_docs + sparse_docs}
            per_query: list[tuple[Document, float]] = []
            for score, pid, _meta, _d, _s in fused:
                doc = by_id.get(pid)
                if doc is None:
                    continue
                per_query.append((doc, float(score)))

            # Stamp the TRUE dense cosine on every surviving doc, exactly as the
            # fused path does.  The stored vector arrives on BOTH lane responses
            # (each request asked for with_vector), so a sparse-only hit still
            # gets its cosine.  Pair positionally per lane; the doc objects are
            # shared with by_id, so the stamp lands on the fused result.
            if self._embedding_scores_enabled():
                q_emb = queries[q_idx][0]
                if dense_docs:
                    self._stamp_dense_cosines(q_emb, chunk[0].points, dense_docs)
                if len(chunk) > 1 and sparse_docs:
                    self._stamp_dense_cosines(q_emb, chunk[1].points, sparse_docs)

            out.append(per_query)
        return out
    def _batcher(self) -> _QdrantBatcher:
        """Per-event-loop batcher (created lazily, one per running loop)."""
        loop = asyncio.get_running_loop()
        batcher = self._batchers.get(loop)
        if batcher is None:
            batcher = _QdrantBatcher(
                search_fn=self.similarity_search_with_score_by_vector_batch,
                max_batch_size=self._batcher_max_size,
                max_wait_ms=self._batcher_max_wait_ms,
            )
            self._batchers[loop] = batcher
        return batcher

    async def asimilarity_search_with_score_by_vector(
        self, embedding: list[float], k: int, need_media: bool = True, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Async batched similarity search by vector.

        Routes through a per-event-loop ``_QdrantBatcher`` so that
        concurrent searches share a single gRPC call.

        When *need_media* is False, heavy base64 ``image``/``video`` payload
        keys are excluded from the Qdrant response (fast path: no reranker,
        no VLM, base LLM doesn't support image/video).  Vector-supplied
        queries carry no text, so they never take the hybrid lane.
        """
        return await self._batcher().submit(embedding, k, need_media, filters, None)

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
        need_media: bool = True,
        filters: dict[str, Any] | None = None,
        rrf: RrfParams | None = None,
    ) -> list[tuple[Document, float]]:
        """Async batched search for a *text* query — the hybrid entry point.

        The raw query text travels with the embedding through the batcher so
        the executed ``QueryRequest`` can become an RRF-fusion request over
        the dense + BM25 lanes (bm25-capable collections only; everything
        else keeps the flat dense request).  *rrf* carries an optional
        :class:`RrfParams` override — when set, the fusion request is
        ``RrfQuery(rrf=Rrf(k=…, weights=[dense, sparse]))`` and every fused
        result is stamped ``metadata["_rrf"]`` with the weights used;
        anything else keeps the default ``FusionQuery(RRF)`` request and no
        stamp.  Dense-degraded paths never apply (or stamp) the weights.
        """
        emb = await self.embedding.aembed_query(query)
        return await self._batcher().submit(emb, k, need_media, filters, query if isinstance(query, str) else None, rrf)
