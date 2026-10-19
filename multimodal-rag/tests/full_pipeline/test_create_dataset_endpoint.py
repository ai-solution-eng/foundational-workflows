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
        self.chunk_seen: list[Any] = []

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
        created_by: str | None = None,  # D23 ownership stamp (passed through)
        chunk: dict[str, Any] | None = None,  # per-dataset chunk overrides
    ) -> dict[str, Any]:
        DatasetManager._validate_name(name)  # the real validator
        self.seen.append((name, description, password, ocr))
        self.rrf_seen.append(rrf)
        self.contextual_seen.append(contextual)
        self.chunk_seen.append(chunk)
        return {"name": name}

    def _sanitize_chunk_meta(self, payload: Any) -> Any:
        # The endpoint delegates validation to the real static discipline.
        return DatasetManager._sanitize_chunk_meta(payload)

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


# -- Per-dataset chunk overrides (feature: per-dataset chunk size) ----------
# The whole point of the feature is that it is OPT-IN: a dataset that never
# sets a chunk value must inherit the embedder model config exactly as it did
# before the feature existed.  These tests pin that default plus the
# validation contract.


def test_no_chunk_body_stores_nothing(stub_dm: _StubManager) -> None:
    """Absent "chunk" -> None reaches create_dataset, so meta.json gets NO
    chunk key at all and the dataset inherits the model config."""
    _create({"name": "inherit"})
    assert stub_dm.chunk_seen == [None]


def test_empty_chunk_object_stores_nothing(stub_dm: _StubManager) -> None:
    """An empty (or all-null) chunk object is the "inherit" spelling too."""
    _create({"name": "empty", "chunk": {}})
    assert stub_dm.chunk_seen == [None]
    _create({"name": "nulls", "chunk": {"chunk_size": None}})
    assert stub_dm.chunk_seen[-1] is None


def test_chunk_overrides_flow_through(stub_dm: _StubManager) -> None:
    _create({"name": "big-chunks", "chunk": {"chunk_size": 8192, "chunk_overlap": 512}})
    assert stub_dm.chunk_seen[-1] == {"chunk_size": 8192, "chunk_overlap": 512}


def test_code_chunk_overrides_are_separate_from_base(stub_dm: _StubManager) -> None:
    """Base and code families are independent keys, not one shared number."""
    _create({"name": "code-only", "chunk": {"code_chunk_size": 4096, "code_chunk_overlap": 128}})
    assert stub_dm.chunk_seen[-1] == {"code_chunk_size": 4096, "code_chunk_overlap": 128}
    assert "chunk_size" not in stub_dm.chunk_seen[-1]


def test_chunk_overlap_zero_is_allowed(stub_dm: _StubManager) -> None:
    """Overlap 0 is meaningful (no overlap), unlike chunk_size 0."""
    _create({"name": "no-overlap", "chunk": {"chunk_overlap": 0}})
    assert stub_dm.chunk_seen[-1] == {"chunk_overlap": 0}


@pytest.mark.parametrize(
    "chunk,message",
    [
        ({"chunk_size": 0}, "chunk_size must be >= 1"),
        ({"chunk_overlap": -1}, "chunk_overlap must be >= 0"),
        ({"chunk_size": "big"}, "chunk_size must be an integer"),
        ({"chunk_size": True}, "must be an integer, not a boolean"),
    ],
)
def test_invalid_chunk_value_is_400(stub_dm: _StubManager, chunk: dict[str, Any], message: str) -> None:
    """A bad override is a 400 with the reason — never a silent store.

    The helper raises TypeError for a wrong TYPE (TRY004) and ValueError
    for an out-of-range VALUE; the endpoint catches BOTH and answers 400.
    This test is what stops a future refactor from turning either class of
    caller error into a 500.
    """
    with pytest.raises(HTTPException) as exc:
        _create({"name": "bad-chunk", "chunk": chunk})
    assert exc.value.status_code == 400
    assert message in exc.value.detail
    assert stub_dm.chunk_seen == [], "nothing may reach create_dataset on a 400"


def test_chunk_must_be_an_object(stub_dm: _StubManager) -> None:
    with pytest.raises(HTTPException) as exc:
        _create({"name": "bad-shape", "chunk": "8192"})
    assert exc.value.status_code == 400
    assert "must be an object" in exc.value.detail
