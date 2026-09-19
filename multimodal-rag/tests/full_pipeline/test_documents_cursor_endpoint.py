"""Regression: ``GET /api/datasets/{name}/documents?cursor=`` (empty string).

The endpoint docstring documents ``'' = start of listing``, but the handler
used to coerce the empty string to ``None`` (``cursor or None``), which flipped
``DatasetManager.list_documents`` to its no-cursor LIST return shape — the
handler then indexed that list like a page dict and raised TypeError → 500.

The fix passes ``cursor`` through verbatim in the non-``None`` branch so the
paginator (both the in-memory and Qdrant backends treat a falsy cursor as a
fresh listing) sees the documented start token and returns the PAGE shape.
"""

import asyncio
from typing import Any

import pytest
from starlette.requests import Request

from multimodal_rag import api_server


class _StubManager:
    """Just enough DatasetManager surface for the documents endpoint."""

    def __init__(self) -> None:
        self.seen_cursors: list[str | None] = []

    def has_password(self, dataset_name: str) -> bool:
        return False

    def get_dataset(self, dataset_name: str, with_stats: bool = False) -> dict[str, Any]:
        return {"document_count": 1}

    def list_documents(
        self,
        dataset_name: str,
        limit: int = 50,
        cursor: str | None = None,
        check_embedder: bool = True,
    ) -> list[tuple[str, dict[str, Any]]] | dict[str, Any]:
        self.seen_cursors.append(cursor)
        if cursor is None:
            return [("id-1", {"k": "v"})]  # legacy list shape (no cursor at all)
        return {"documents": [("id-1", {"k": "v"})], "next_cursor": None}  # page shape


def _request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""})


@pytest.fixture()
def stub_dm(monkeypatch: pytest.MonkeyPatch) -> _StubManager:
    dm = _StubManager()

    async def _fake_get_manager() -> _StubManager:
        return dm

    monkeypatch.setattr(api_server, "get_manager_async", _fake_get_manager)
    return dm


def _call(cursor: str | None) -> dict[str, Any]:
    return asyncio.run(
        api_server.api_list_documents(
            name="ds",
            request=_request(),
            limit=50,
            cursor=cursor,
            x_dataset_password=None,
        )
    )


def test_no_cursor_keeps_legacy_page_shape(stub_dm: _StubManager) -> None:
    body = _call(None)
    assert body == {"documents": [{"id": "id-1", "payload": {"k": "v"}}], "count": 1}
    assert stub_dm.seen_cursors == [None]


def test_empty_cursor_returns_first_page_not_500(stub_dm: _StubManager) -> None:
    """THE regression: ``?cursor=`` must return the page shape (HTTP 200 body),
    not crash indexing a list like a dict."""
    body = _call("")
    assert body == {"documents": [{"id": "id-1", "payload": {"k": "v"}}], "next_cursor": None}
    assert stub_dm.seen_cursors == [""]


def test_token_cursor_resumes_and_preserves_token(stub_dm: _StubManager) -> None:
    body = _call("tok-9")
    assert body["next_cursor"] is None
    assert stub_dm.seen_cursors == ["tok-9"]
