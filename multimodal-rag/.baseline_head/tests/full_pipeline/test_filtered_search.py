"""Offline tests for metadata-filtered search (roadmap feature 1).

Covers:

  * ``build_payload_filter`` — the public filter dict → Qdrant ``Filter``
    translation (empty → None, unknown keys ignored, date validation).
  * ``filters_to_predicate`` — the Python counterpart used by the in-memory
    store.
  * ``QdrantVectorStore`` end-to-end on embedded local Qdrant: file-type,
    severity, source-prefix (``MatchPrefix``) and datetime-range filters,
    including the batched path with per-query filters.
  * ``MultimodalRAG._to_documents`` — ``metadata.file_type`` stamped from the
    document source at ingest.
  * ``DatasetManager.backfill_search_metadata`` — idempotent backfill that
    must preserve every existing metadata key (the ``set_payload`` call
    replaces the whole ``metadata`` object, so the write carries the full
    merged dict).

No model endpoint required — the embedding is stubbed.

Run::

    python tests/full_pipeline/test_filtered_search.py    # standalone
    pytest tests/full_pipeline/test_filtered_search.py    # under pytest
"""

import asyncio
import os
import sys
import uuid
from typing import Any

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    MatchAny,
    MatchPrefix,
    PointStruct,
    VectorParams,
)

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.vector_store import (
    QdrantVectorStore,
    build_payload_filter,
    ensure_search_payload_indexes,
    filters_to_predicate,
)

DIM = 8
COLL = "filtered_search_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubEmbedder:
    model_name = "stub"
    base_url = "http://stub/v1"

    async def aembed_query(self, query: Any) -> list[float]:
        return [0.1] * DIM

    def embed_query(self, query: Any) -> list[float]:
        return [0.1] * DIM


def _client() -> QdrantClient:
    client = QdrantClient(":memory:")
    try:
        client.delete_collection(COLL)
    except Exception:
        pass
    client.create_collection(
        COLL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    return client


def _dm_with(client: QdrantClient) -> DatasetManager:
    dm = DatasetManager.__new__(DatasetManager)
    vs = QdrantVectorStore(client, COLL, embedding=_StubEmbedder())
    rag = type("_RagStub", (), {"vector_store": vs})()
    dm._get_rag = lambda ds, check_embedder=True: rag  # type: ignore[method-assign]
    return dm


def _upsert(client: QdrantClient, text: str, **meta) -> str:
    pid = uuid.uuid4().hex
    client.upsert(
        COLL,
        points=[PointStruct(id=pid, vector=[0.1] * DIM, payload={"page_content": text, "metadata": meta})],
        wait=True,
    )
    return pid


def _seed(client: QdrantClient, stamp_file_type: bool = True) -> dict[str, str]:
    """Five docs across the filter axes: type, severity, source prefix, dates.

    ``stamp_file_type=False`` simulates a pre-feature dataset (no
    ``metadata.file_type``) for the backfill tests.
    """

    def _meta(source: str, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {"source": source, **extra}
        if stamp_file_type:
            from multimodal_rag.dataset_manager import _classify_file

            meta["file_type"] = _classify_file(source)
        return meta

    return {
        "pdf": _upsert(
            client,
            "quarterly report text",
            **_meta("/data/datasets/ds/files/reports/2025/q4.pdf", timestamp_start="2026-08-01T10:00:00"),
        ),
        "log_err": _upsert(
            client,
            "2026-08-15 ERROR server crash",
            **_meta(
                "/data/datasets/ds/files/logs/app.log",
                severities=["ERROR"],
                timestamp_start="2026-08-15T12:00:00",
            ),
        ),
        "log_info": _upsert(
            client,
            "2026-08-20 INFO started",
            **_meta(
                "/data/datasets/ds/files/logs/boot.log",
                severities=["INFO"],
                timestamp_start="2026-08-20T08:00:00",
            ),
        ),
        "image": _upsert(client, "[Image: c.jpg]", **_meta("/data/datasets/ds/files/img/c.jpg")),
        "video": _upsert(
            client,
            "[Video: v.mp4] [0s – 32s]",
            **_meta(
                "/data/datasets/ds/files/vid/v.mp4",
                timestamp_start=0.0,  # media segments carry float seconds, not datetimes
                timestamp_end=32.0,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# build_payload_filter (dict → Qdrant Filter)
# ---------------------------------------------------------------------------


def test_build_filter_empty_is_none():
    assert build_payload_filter(None) is None
    assert build_payload_filter({}) is None
    assert build_payload_filter({"unknown_key": 1}) is None  # unknown keys ignored


def test_build_filter_all_conditions():
    flt = build_payload_filter(
        {
            "file_types": ["pdf", "log"],
            "severities": ["ERROR"],
            "source_prefix": "reports/2025/",
            "date_from": "2026-08-01T00:00:00",
            "date_to": "2026-08-31T23:59:59",
        }
    )
    assert isinstance(flt, Filter)
    assert len(flt.must) == 4
    matches = [c.match for c in flt.must]
    assert MatchAny(any=["pdf", "log"]) in matches
    assert MatchAny(any=["ERROR"]) in matches
    assert MatchPrefix(prefix="reports/2025/") in matches
    assert any(getattr(m, "range", None) is not None for m in flt.must)


def test_build_filter_one_sided_date():
    flt = build_payload_filter({"date_from": "2026-08-01T00:00:00"})
    rng = next(c.range for c in flt.must if getattr(c, "range", None) is not None)
    assert rng.gte is not None and rng.lte is None


def test_build_filter_invalid_date_raises():
    for bad in ("not-a-date", "2026-13-45"):
        with pytest.raises(ValueError):
            build_payload_filter({"date_from": bad})


# ---------------------------------------------------------------------------
# filters_to_predicate (in-memory store)
# ---------------------------------------------------------------------------


def test_predicate_file_types_and_severities():
    pred = filters_to_predicate({"file_types": ["log"], "severities": ["ERROR"]})
    assert pred({"file_type": "log", "severities": ["ERROR", "INFO"]}) is True
    assert pred({"file_type": "log", "severities": ["INFO"]}) is False
    assert pred({"file_type": "pdf"}) is False
    assert pred({"file_type": "log", "severities": "ERROR"}) is True  # string form


def test_predicate_prefix_and_dates():
    pred = filters_to_predicate({"source_prefix": "reports/"})
    assert pred({"source": "reports/2025/q4.pdf"}) is True
    assert pred({"source": "logs/app.log"}) is False

    pred = filters_to_predicate({"date_from": "2026-08-10T00:00:00"})
    assert pred({"timestamp_start": "2026-08-15T12:00:00"}) is True
    assert pred({"timestamp_start": "2026-08-01T12:00:00"}) is False
    assert pred({"timestamp_start": 0.0}) is False, "float media timestamps never match date filters"


# ---------------------------------------------------------------------------
# QdrantVectorStore end-to-end (embedded local mode)
# ---------------------------------------------------------------------------


def test_store_filtered_search_all_axes():
    client = _client()
    _seed(client)
    vs = QdrantVectorStore(client, COLL, embedding=_StubEmbedder())

    async def _run():
        out = {}
        out["pdf"] = await vs.asimilarity_search_with_relevance_scores(
            "q", 10, filters={"file_types": ["pdf"]}
        )
        out["err"] = await vs.asimilarity_search_with_relevance_scores(
            "q", 10, filters={"severities": ["ERROR"]}
        )
        out["logs"] = await vs.asimilarity_search_with_relevance_scores(
            "q", 10, filters={"source_prefix": "/data/datasets/ds/files/logs/"}
        )
        out["aug_mid"] = await vs.asimilarity_search_with_relevance_scores(
            "q",
            10,
            filters={"date_from": "2026-08-10T00:00:00", "date_to": "2026-08-16T00:00:00"},
        )
        out["no_filter"] = await vs.asimilarity_search_with_relevance_scores("q", 10)
        return out

    res = asyncio.run(_run())
    assert len(res["pdf"]) == 1
    assert res["pdf"][0][0].metadata["source"].endswith("q4.pdf")
    assert len(res["err"]) == 1
    assert len(res["logs"]) == 2
    assert len(res["aug_mid"]) == 1, "datetime range matches only the log entry inside it"
    assert len(res["no_filter"]) == 5


def test_store_batched_search_with_per_query_filters():
    """Concurrent searches through the batcher each honour their own filter."""
    client = _client()
    _seed(client)
    vs = QdrantVectorStore(client, COLL, embedding=_StubEmbedder())

    async def _run():
        return await asyncio.gather(
            vs.asimilarity_search_with_relevance_scores("q", 10, filters={"file_types": ["pdf"]}),
            vs.asimilarity_search_with_relevance_scores("q", 10, filters={"severities": ["ERROR"]}),
            vs.asimilarity_search_with_relevance_scores("q", 10),
        )

    pdf_res, err_res, all_res = asyncio.run(_run())
    assert len(pdf_res) == 1
    assert len(err_res) == 1
    assert len(all_res) == 5


def test_ensure_indexes_idempotent():
    client = _client()
    first = ensure_search_payload_indexes(client, COLL)
    second = ensure_search_payload_indexes(client, COLL)
    assert first and second  # both runs succeed on local mode


# ---------------------------------------------------------------------------
# Ingest-side file_type stamping
# ---------------------------------------------------------------------------


def test_to_documents_stamps_file_type():
    docs = MultimodalRAG._to_documents(
        [
            {"text": "hello", "source": "/tmp/x/script.py"},
            {"text": "no source"},
            {"text": "explicit", "source": "/tmp/a.pdf", "file_type": "custom"},
        ]
    )
    assert docs[0].metadata["file_type"] == "code"
    assert "file_type" not in docs[1].metadata
    assert docs[2].metadata["file_type"] == "custom", "explicitly provided file_type wins"


# ---------------------------------------------------------------------------
# backfill_search_metadata
# ---------------------------------------------------------------------------


def test_backfill_adds_file_type_and_preserves_metadata():
    client = _client()
    _seed(client, stamp_file_type=False)  # none of the seeded docs carry file_type
    dm = _dm_with(client)

    result = dm.backfill_search_metadata("ds")
    assert result["scanned"] == 5
    assert result["updated"] == 5
    assert result["indexes"], "payload indexes should be created"

    rows = {pid: payload for pid, payload in dm.scroll_documents("ds", limit=100)}

    # Assert on metadata content by source.
    by_source = {payload["metadata"]["source"]: payload["metadata"] for payload in rows.values()}
    assert by_source["/data/datasets/ds/files/reports/2025/q4.pdf"]["file_type"] == "pdf"
    assert by_source["/data/datasets/ds/files/logs/app.log"]["file_type"] == "log"
    assert by_source["/data/datasets/ds/files/logs/boot.log"]["file_type"] == "log"
    assert by_source["/data/datasets/ds/files/img/c.jpg"]["file_type"] == "image"
    assert by_source["/data/datasets/ds/files/vid/v.mp4"]["file_type"] == "video"

    # Critical: the full-metadata write must preserve every pre-existing key.
    log_meta = by_source["/data/datasets/ds/files/logs/app.log"]
    assert log_meta["severities"] == ["ERROR"]
    assert log_meta["timestamp_start"] == "2026-08-15T12:00:00"
    video_meta = by_source["/data/datasets/ds/files/vid/v.mp4"]
    assert video_meta["timestamp_start"] == 0.0 and video_meta["timestamp_end"] == 32.0


def test_backfill_is_idempotent():
    client = _client()
    _seed(client, stamp_file_type=False)
    dm = _dm_with(client)
    dm.backfill_search_metadata("ds")
    again = dm.backfill_search_metadata("ds")
    assert again["updated"] == 0, "second run must find nothing to update"
    assert again["scanned"] == 5


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  {fn.__name__} ... OK")
        except Exception:
            failed += 1
            print(f"  {fn.__name__} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
