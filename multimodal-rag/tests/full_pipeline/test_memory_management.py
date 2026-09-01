"""Offline tests for the memory-management backend (roadmap feature 6).

Covers the :class:`~multimodal_rag.dataset_manager.DatasetManager` helpers
behind the new MCP tools ``delete_memory`` / ``list_memories`` /
``forget_session``:

  * ``delete_documents()`` — batched delete by **explicit** point IDs only
    (no similarity- or filter-directed deletion), with count bookkeeping.
  * ``scroll_documents()`` — server-side payload-filtered scroll, exercising
    the exact filter shapes ``list_memories`` builds: ``memory_kind``,
    ``memory_tags`` (MatchAny over a keyword array), the default
    ``session_history`` exclusion, and limit/pagination.

Qdrant runs in embedded local mode (``:memory:``) — no server, no models.

Run::

    python tests/full_pipeline/test_memory_management.py   # standalone
    pytest tests/full_pipeline/test_memory_management.py   # under pytest
"""

import os
import sys
import uuid

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.vector_store import QdrantVectorStore

DIM = 8
COLL = "memories_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _dm_with(client: QdrantClient) -> tuple[DatasetManager, list[int]]:
    """A bare DatasetManager wired to a fixed Qdrant collection."""
    dm = DatasetManager.__new__(DatasetManager)
    vs = QdrantVectorStore(client, COLL, embedding=None)
    rag = type("_RagStub", (), {"vector_store": vs})()
    dm._get_rag = lambda ds, check_embedder=True: rag  # type: ignore[method-assign]
    decrements: list[int] = []
    dm._decrement_count = lambda ds, n: decrements.append(n)  # type: ignore[method-assign]
    return dm, decrements


def _upsert(client: QdrantClient, text: str, **meta) -> str:
    pid = uuid.uuid4().hex
    client.upsert(
        COLL,
        points=[
            PointStruct(
                id=pid,
                vector=[0.1] * DIM,
                payload={"page_content": text, "metadata": meta},
            )
        ],
        wait=True,
    )
    return pid


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------


def test_delete_documents_removes_only_given_ids():
    client = _client()
    keep = _upsert(client, "keep me", memory_kind="note")
    kill1 = _upsert(client, "delete me 1", memory_kind="note")
    kill2 = _upsert(client, "delete me 2", memory_kind="decision")

    dm, decrements = _dm_with(client)
    deleted = dm.delete_documents("any", [kill1, kill2])
    assert deleted == 2
    assert decrements == [2], "document count must be decremented by the batch size"

    remaining = {pid for pid, _ in dm.scroll_documents("any", limit=100)}
    assert remaining == {keep}


def test_delete_documents_empty_is_noop():
    client = _client()
    pid = _upsert(client, "survivor")
    dm, decrements = _dm_with(client)
    assert dm.delete_documents("any", []) == 0
    assert dm.delete_documents("any", ["", "  "]) == 0, "blank IDs must be dropped"
    assert decrements == []
    assert {pid for pid, _ in dm.scroll_documents("any", limit=10)} == {pid}


def test_delete_documents_ignores_blank_entries():
    client = _client()
    pid = _upsert(client, "stays")
    junk = _upsert(client, "goes")
    dm, decrements = _dm_with(client)
    assert dm.delete_documents("any", ["", junk, " "]) == 1
    assert decrements == [1]
    assert {pid for pid, _ in dm.scroll_documents("any", limit=10)} == {pid}


# ---------------------------------------------------------------------------
# scroll_documents — the filter shapes list_memories builds
# ---------------------------------------------------------------------------


def _seed_store(client: QdrantClient) -> dict[str, str]:
    return {
        "decision": _upsert(
            client, "chose tabs", memory_kind="decision", memory_ts="2026-08-30T10:00:00", memory_tags=["style"]
        ),
        "preference": _upsert(
            client, "prefers uv", memory_kind="preference", memory_ts="2026-08-31T09:00:00", memory_tags=["tooling", "auth"]
        ),
        "note": _upsert(client, "plain note", memory_kind="note", memory_ts="2026-08-29T08:00:00"),
        "session": _upsert(
            client, "session log", memory_kind="session_history", session_id="s1", memory_ts="2026-08-31T12:00:00"
        ),
    }


def test_scroll_filter_by_kind():
    client = _client()
    ids = _seed_store(client)
    dm, _ = _dm_with(client)
    flt = Filter(must=[FieldCondition(key="metadata.memory_kind", match=MatchValue(value="decision"))])
    rows = dm.scroll_documents("any", flt, limit=100)
    assert {pid for pid, _ in rows} == {ids["decision"]}


def test_scroll_excludes_session_history_by_default():
    client = _client()
    ids = _seed_store(client)
    dm, _ = _dm_with(client)
    flt = Filter(must_not=[FieldCondition(key="metadata.memory_kind", match=MatchValue(value="session_history"))])
    rows = dm.scroll_documents("any", flt, limit=100)
    assert ids["session"] not in {pid for pid, _ in rows}
    assert len(rows) == 3


def test_scroll_tags_match_any():
    client = _client()
    ids = _seed_store(client)
    dm, _ = _dm_with(client)
    flt = Filter(must=[FieldCondition(key="metadata.memory_tags", match=MatchAny(any=["auth"]))])
    rows = dm.scroll_documents("any", flt, limit=100)
    assert {pid for pid, _ in rows} == {ids["preference"]}


def test_scroll_respects_limit_and_returns_payloads():
    client = _client()
    _seed_store(client)
    dm, _ = _dm_with(client)
    rows = dm.scroll_documents("any", None, limit=2)
    assert len(rows) == 2
    for pid, payload in rows:
        assert isinstance(pid, str) and pid
        assert "page_content" in payload and "metadata" in payload


def test_scroll_newest_first_sort_key():
    """list_memories sorts by metadata.memory_ts desc — verify the data supports it."""
    client = _client()
    ids = _seed_store(client)
    dm, _ = _dm_with(client)
    rows = dm.scroll_documents("any", None, limit=100)
    ts = {pid: dict(p["metadata"]).get("memory_ts", "") for pid, p in ((pid, payload) for pid, payload in rows)}
    ordered = sorted(rows, key=lambda r: ts[r[0]], reverse=True)
    assert [pid for pid, _ in ordered][:2] == [ids["session"], ids["preference"]]


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
