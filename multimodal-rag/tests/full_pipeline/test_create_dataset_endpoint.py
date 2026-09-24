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
        self.seen: list[tuple[str, str, str | None, bool]] = []
        self.rrf_seen: list[Any] = []
        self.contextual_seen: list[bool] = []

    def create_dataset(
        self,
        name: str,
        description: str,
        caption_with_asr: bool,
        caption_with_vlm: bool,
        keep_originals: bool,
        password: str | None,
        ocr: bool,
        rrf: dict[str, Any] | None = None,
        contextual: bool = False,
    ) -> dict[str, Any]:
        DatasetManager._validate_name(name)  # the real validator
        self.seen.append((name, description, password, ocr))
        self.rrf_seen.append(rrf)
        self.contextual_seen.append(contextual)
        return {"name": name}

    def _sanitize_rrf_meta(self, payload: Any) -> Any:
        # The endpoint delegates validation to the real static discipline.
        return DatasetManager._sanitize_rrf_meta(payload)


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


# ---------------------------------------------------------------------------
# Weighted-RRF defaults on the create surface (dataset-defaults slice)
# ---------------------------------------------------------------------------


def test_no_rrf_body_and_disabled_env_stamps_nothing(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """The disabled-by-default rule: no body ``rrf`` + unset env → create is
    called with ``rrf=None`` (the global 1.0/1.0 applies at search time)."""
    monkeypatch.setattr(api_server, "RAG_RRF_DEFAULT", "")
    _create({"name": "plain"})
    assert stub_dm.rrf_seen == [None]


def test_env_default_stamps_new_datasets(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG_RRF_DEFAULT enabled → NEW datasets are create-time stamped with
    the parsed dense/sparse[/k]; the env never touches existing datasets
    (it only feeds the create call)."""
    monkeypatch.setattr(api_server, "RAG_RRF_DEFAULT", "1.0,3.0,2")
    _create({"name": "stamped"})
    assert stub_dm.rrf_seen == [{"dense_weight": 1.0, "sparse_weight": 3.0, "k": 2}]


def test_body_rrf_wins_over_env_default(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_server, "RAG_RRF_DEFAULT", "2.0,2.0,2")
    _create({"name": "explicit", "rrf": {"sparse_weight": 5.0}})
    assert stub_dm.rrf_seen == [{"sparse_weight": 5.0}]


def test_body_rrf_is_validated_not_stored_raw(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range clamps; a global-default payload stamps NOTHING; a
    non-numeric weight is HTTP 400 and never reaches create_dataset."""
    monkeypatch.setattr(api_server, "RAG_RRF_DEFAULT", "")
    _create({"name": "clamped", "rrf": {"dense_weight": 99.0, "k": 0}})
    assert stub_dm.rrf_seen == [{"dense_weight": 10.0, "k": 1}]

    _create({"name": "redundant", "rrf": {"dense_weight": 1.0, "sparse_weight": 1.0}})
    assert stub_dm.rrf_seen[-1] is None, "a pure-global-default payload stores nothing"

    with pytest.raises(HTTPException) as exc:
        _create({"name": "bad", "rrf": {"dense_weight": "abc"}})
    assert exc.value.status_code == 400
    assert stub_dm.seen[-1][0] != "bad", "a rejected rrf must not create the dataset"

    with pytest.raises(HTTPException) as exc2:
        _create({"name": "bad2", "rrf": 3})
    assert exc2.value.status_code == 400


# ---------------------------------------------------------------------------
# Contextual retrieval on the create surface (feature: contextual retrieval)
# ---------------------------------------------------------------------------


def test_no_contextual_body_and_disabled_env_stamps_false(
    stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disabled-by-default rule: no body ``contextual`` + env off →
    create is called with ``contextual=False`` (zero behaviour change)."""
    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", False)
    _create({"name": "plain"})
    assert stub_dm.contextual_seen == [False]


def test_contextual_env_default_stamps_new_datasets(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG_CONTEXTUAL_DEFAULT on → NEW datasets are create-time stamped;
    the env never touches existing datasets (it only feeds the create call)."""
    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", True)
    _create({"name": "stamped"})
    assert stub_dm.contextual_seen == [True]


def test_body_contextual_wins_over_env_default(stub_dm: _StubManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", True)
    _create({"name": "explicit", "contextual": False})
    assert stub_dm.contextual_seen == [False]

    monkeypatch.setattr(api_server, "RAG_CONTEXTUAL_DEFAULT", False)
    _create({"name": "explicit-on", "contextual": True})
    assert stub_dm.contextual_seen[-1] is True


def test_contextual_is_coerced_to_bool(stub_dm: _StubManager) -> None:
    """Any truthy/falsy body value is coerced (bool()), never stored raw."""
    _create({"name": "truthy", "contextual": "yes"})
    assert stub_dm.contextual_seen[-1] is True
    _create({"name": "falsy", "contextual": 0})
    assert stub_dm.contextual_seen[-1] is False
