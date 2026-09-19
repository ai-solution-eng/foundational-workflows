"""Regression: ``POST /api/datasets`` with a space in the name used to 500.

``DatasetManager._validate_name`` raises ``ValueError`` for anything outside
``[A-Za-z0-9._-]``, and the create handler only caught ``FileExistsError`` —
so a hand-typed ``"my dataset"`` surfaced as an unhandled ValueError →
HTTP 500.  The endpoint now auto-converts whitespace runs to ``_`` (the
obvious intent: ``"my dataset"`` → ``"my_dataset"``) and maps remaining
validator rejections (e.g. ``"bad name!"``) to a proper HTTP 400.
"""

import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from multimodal_rag import api_server
from multimodal_rag.dataset_manager import DatasetManager


class _StubManager:
    """Just enough DatasetManager surface for the create endpoint."""

    caption_with_asr = False
    caption_with_vlm = False

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def create_dataset(
        self,
        name: str,
        description: str,
        caption_with_asr: bool,
        caption_with_vlm: bool,
        keep_originals: bool,
        password: str | None,
        ocr: bool,
    ) -> dict[str, Any]:
        DatasetManager._validate_name(name)  # the real validator
        self.seen.append((name, description, password, ocr))
        return {"name": name}


@pytest.fixture()
def stub_dm(monkeypatch: pytest.MonkeyPatch) -> _StubManager:
    dm = _StubManager()

    async def _fake_get_manager() -> _StubManager:
        return dm

    monkeypatch.setattr(api_server, "get_manager_async", _fake_get_manager)
    return dm


def _create(body: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(api_server.api_create_dataset(body))


def test_space_in_name_is_normalized(stub_dm: _StubManager) -> None:
    body = _create({"name": "my dataset"})
    assert body == {"status": "ok", "dataset": {"name": "my_dataset"}}
    assert stub_dm.seen == [("my_dataset", "", None, False)]


def test_whitespace_runs_collapse_to_one_underscore(stub_dm: _StubManager) -> None:
    body = _create({"name": "  my  data\tset\n"})
    assert body["dataset"]["name"] == "my_data_set"


def test_flags_and_password_flow_through_the_normalized_name(stub_dm: _StubManager) -> None:
    _create({"name": "team docs", "description": "d", "password": "pw", "ocr": True})
    assert stub_dm.seen == [("team_docs", "d", "pw", True)]


def test_invalid_name_is_400_not_500(stub_dm: _StubManager) -> None:
    with pytest.raises(HTTPException) as exc:
        _create({"name": "bad name!"})
    assert exc.value.status_code == 400
    assert "Invalid dataset name" in str(exc.value.detail)
    assert stub_dm.seen == [], "a rejected name must never reach create_dataset"
