"""Offline tests for the embedder fingerprint guard: alias groups and the
payload-read bypass (``check_embedder=False``).

Covers:

  * ``_assert_embedder_compatible`` — same name passes; a different name
    raises :class:`EmbedderMismatchError`; a stored name listed in the
    configured embedder's ``alias`` passes (string and list alias forms,
    comma-separated strings included); a **dimension** mismatch still raises
    even under an alias (aliases never mask a real vector-space change).
  * ``stream_all_documents`` / ``list_documents`` forward ``check_embedder``
    to ``_get_rag`` — payload reads (listing / export / documents download)
    opt out of the guard so the documented export→import recovery works
    across an embedder swap; search/ingest paths keep the default (guarded).

No Qdrant server, no models — the guard only needs meta + the embedder
config object.

Run::

    python tests/full_pipeline/test_embedder_compat_guard.py   # standalone
    pytest tests/full_pipeline/test_embedder_compat_guard.py   # under pytest
"""

import os
import sys
from typing import Any

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import DatasetManager, EmbedderMismatchError
from multimodal_rag.utils.pcai_model_classes import EmbeddingModel

OLD_NAME = "Qwen/Qwen3-VL-Embedding-8B"
FP8_NAME = "RamManavalan/Qwen3-VL-Embedding-8B-FP8"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_guard_dm(embedder: EmbeddingModel, stored: dict[str, Any] | None) -> DatasetManager:
    """A bare DatasetManager exercising only ``_assert_embedder_compatible``."""
    dm = DatasetManager.__new__(DatasetManager)
    dm.embedder = embedder  # type: ignore[method-assign]
    dm._embedder_verified = set()  # type: ignore[method-assign]
    dm._read_meta = lambda name: stored or {}  # type: ignore[method-assign]
    dm._write_embedder_fingerprint = lambda dataset_name: None  # type: ignore[method-assign]
    dm._embedder_dimension = lambda: 4096  # type: ignore[method-assign]
    return dm


def _fp8_embedder(alias: Any = None) -> EmbeddingModel:
    kwargs: dict[str, Any] = {"model_name": FP8_NAME}
    if alias is not None:
        kwargs["alias"] = alias
    return EmbeddingModel(**kwargs)


def _expect_mismatch(dm: DatasetManager, dataset: str = "ds1") -> None:
    try:
        dm._assert_embedder_compatible(dataset)
        raise AssertionError("expected EmbedderMismatchError")
    except EmbedderMismatchError:
        pass


# ---------------------------------------------------------------------------
# Alias semantics
# ---------------------------------------------------------------------------


def test_same_name_passes():
    dm = _make_guard_dm(_fp8_embedder(), {"embedder_model": FP8_NAME, "embedder_dim": 4096})
    dm._assert_embedder_compatible("ds1")


def test_different_name_raises_without_alias():
    dm = _make_guard_dm(_fp8_embedder(), {"embedder_model": OLD_NAME, "embedder_dim": 4096})
    _expect_mismatch(dm)


def test_string_alias_passes():
    dm = _make_guard_dm(_fp8_embedder(alias=OLD_NAME), {"embedder_model": OLD_NAME, "embedder_dim": 4096})
    dm._assert_embedder_compatible("ds1")


def test_list_alias_passes():
    dm = _make_guard_dm(
        _fp8_embedder(alias=[OLD_NAME, "other/variant"]),
        {"embedder_model": OLD_NAME, "embedder_dim": 4096},
    )
    dm._assert_embedder_compatible("ds1")


def test_comma_separated_alias_passes():
    dm = _make_guard_dm(
        _fp8_embedder(alias=f"{OLD_NAME}, other/variant"),
        {"embedder_model": OLD_NAME, "embedder_dim": 4096},
    )
    dm._assert_embedder_compatible("ds1")


def test_alias_does_not_cover_unrelated_names():
    dm = _make_guard_dm(_fp8_embedder(alias=OLD_NAME), {"embedder_model": "some/other-model", "embedder_dim": 4096})
    _expect_mismatch(dm)


def test_dim_mismatch_still_raises_under_alias():
    """Aliases excuse the NAME only — a real vector-space change (dim) must
    keep failing loudly."""
    dm = _make_guard_dm(_fp8_embedder(alias=OLD_NAME), {"embedder_model": OLD_NAME, "embedder_dim": 2560})
    _expect_mismatch(dm)


def test_missing_fingerprint_records_and_passes():
    calls: list[str] = []
    dm = _make_guard_dm(_fp8_embedder(alias=OLD_NAME), None)
    dm._write_embedder_fingerprint = lambda name: calls.append(name)  # type: ignore[method-assign]
    dm._assert_embedder_compatible("fresh")
    assert calls == ["fresh"], "first touch records the fingerprint"


# ---------------------------------------------------------------------------
# Payload-read bypass (check_embedder=False)
# ---------------------------------------------------------------------------


class _FakeRAG:
    def __init__(self) -> None:
        self.vector_store = None

    def list_documents(self, limit: int = 50):
        return []

    def list_documents_page(self, limit: int = 50, cursor: str | None = None):
        return {"documents": [], "next_cursor": None}


def _make_bypass_dm() -> tuple[DatasetManager, dict[str, Any]]:
    dm = DatasetManager.__new__(DatasetManager)
    captured: dict[str, Any] = {}

    def _get_rag(dataset_name: str, check_embedder: bool = True):
        captured["check_embedder"] = check_embedder
        return _FakeRAG()

    dm._get_rag = _get_rag  # type: ignore[method-assign]
    return dm, captured


def test_stream_all_documents_forwards_flag():
    dm, captured = _make_bypass_dm()
    dm.stream_all_documents("ds", lambda doc: None, check_embedder=False)
    assert captured["check_embedder"] is False, "export path must bypass the guard"
    dm.stream_all_documents("ds", lambda doc: None)
    assert captured["check_embedder"] is True, "default stays guarded"


def test_list_documents_forwards_flag():
    dm, captured = _make_bypass_dm()
    dm.list_documents("ds", check_embedder=False)
    assert captured["check_embedder"] is False, "documents listing must bypass the guard"
    dm.list_documents("ds")
    assert captured["check_embedder"] is True, "default stays guarded"
    dm.list_documents("ds", cursor="next-page-token", check_embedder=False)
    assert captured["check_embedder"] is False, "paginated listing bypasses too"


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
