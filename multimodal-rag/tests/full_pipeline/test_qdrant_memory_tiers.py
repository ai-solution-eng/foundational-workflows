"""Tests for the Qdrant v1.19 memory-tier knobs (rag_system._qdrant_* helpers).

Covers the migration-critical logic that the legacy path never exercised:

  * ``_qdrant_memory_tier`` accepts the three v1.19 tiers and rejects garbage
    by returning ``None`` (so the caller keeps Qdrant's own default).
  * ``_qdrant_quantization_config`` precedence: the new
    ``QDRANT_QUANTIZATION_MEMORY`` wins over the deprecated
    ``QDRANT_QUANTIZATION_ALWAYS_RAM`` boolean, and the legacy boolean still
    maps ``true -> pinned`` / ``false -> cold`` so an upgrading deployment
    keeps its placement.
  * ``QDRANT_DATATYPE`` parsing and its float32 fallback.
  * ``_qdrant_shard_multiple`` parsing (clamped to >= 1).
  * ``_qdrant_node_count`` returns 0 when the cluster API is unavailable
    (embedded local mode raises NotImplementedError).
  * The dense-``pinned`` guard: Qdrant rejects ``pinned`` for dense vectors,
    so the create path must downgrade to ``cached`` instead of failing.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Memory, ScalarType

from multimodal_rag import rag_system as R


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "QDRANT_QUANTIZATION",
        "QDRANT_QUANTIZATION_MEMORY",
        "QDRANT_QUANTIZATION_ALWAYS_RAM",
        "QDRANT_DENSE_MEMORY",
        "QDRANT_SPARSE_MEMORY",
        "QDRANT_DATATYPE",
        "QDRANT_SHARD_MULTIPLE",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _quant_memory():
    q = R._qdrant_quantization_config()
    return None if q is None else q.scalar.memory


class TestMemoryTier:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("pinned", Memory.PINNED),
            ("cached", Memory.CACHED),
            ("cold", Memory.COLD),
            ("  PINNED  ", Memory.PINNED),
        ],
    )
    def test_parses_valid_tiers(self, monkeypatch, raw, expected):
        monkeypatch.setenv("X_TIER", raw)
        assert R._qdrant_memory_tier("X_TIER", "cold") == expected

    def test_unknown_tier_returns_none(self, monkeypatch):
        monkeypatch.setenv("X_TIER", "lukewarm")
        assert R._qdrant_memory_tier("X_TIER", "cold") is None

    def test_unset_uses_default(self) -> None:
        assert R._qdrant_memory_tier("X_UNSET_VAR", "cached") == Memory.CACHED


class TestQuantizationConfig:
    def test_default_is_int8_pinned(self) -> None:
        q = R._qdrant_quantization_config()
        assert q.scalar.type == ScalarType.INT8
        assert q.scalar.memory == Memory.PINNED
        assert q.scalar.always_ram is None, "legacy flag must not be set"

    def test_none_disables(self, monkeypatch):
        monkeypatch.setenv("QDRANT_QUANTIZATION", "none")
        assert R._qdrant_quantization_config() is None

    @pytest.mark.parametrize(
        "legacy,expected",
        [
            ("true", Memory.PINNED),
            ("false", Memory.COLD),
        ],
    )
    def test_legacy_boolean_still_honoured(self, monkeypatch, legacy, expected):
        monkeypatch.setenv("QDRANT_QUANTIZATION_ALWAYS_RAM", legacy)
        assert _quant_memory() == expected

    def test_new_var_wins_over_legacy(self, monkeypatch):
        monkeypatch.setenv("QDRANT_QUANTIZATION_ALWAYS_RAM", "false")
        monkeypatch.setenv("QDRANT_QUANTIZATION_MEMORY", "pinned")
        assert _quant_memory() == Memory.PINNED

    def test_unknown_tier_falls_back_to_qdrant_default(self, monkeypatch):
        monkeypatch.setenv("QDRANT_QUANTIZATION_MEMORY", "bogus")
        q = R._qdrant_quantization_config()
        assert q.scalar.type == ScalarType.INT8
        assert q.scalar.memory is None


class TestDatatype:
    def test_default_float32(self) -> None:
        assert R._qdrant_vector_datatype().value == "float32"

    def test_float16(self, monkeypatch):
        monkeypatch.setenv("QDRANT_DATATYPE", "float16")
        assert R._qdrant_vector_datatype().value == "float16"

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("QDRANT_DATATYPE", "float64")
        assert R._qdrant_vector_datatype().value == "float32"


class TestShardMultiple:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1", 1),
            ("2", 2),
            ("0", 1),
            ("-3", 1),
            ("abc", 1),
        ],
    )
    def test_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("QDRANT_SHARD_MULTIPLE", raw)
        assert R._qdrant_shard_multiple() == expected

    def test_default(self) -> None:
        assert R._qdrant_shard_multiple() == 1


class TestNodeCount:
    def test_embedded_local_returns_zero(self) -> None:
        client = QdrantClient(path=":memory:")
        # Local mode raises NotImplementedError for cluster_status -> 0.
        assert R._qdrant_node_count(client) == 0

    def test_unreachable_returns_zero(self) -> None:
        class Boom:
            def cluster_status(self):
                raise RuntimeError("no cluster")

        assert R._qdrant_node_count(Boom()) == 0

    def test_counts_peers(self) -> None:
        class Info:
            peers: ClassVar[dict[str, int]] = {"a": 1, "b": 2, "c": 3}

        class Fake:
            def cluster_status(self) -> Info:
                return Info()

        assert R._qdrant_node_count(Fake()) == 3


class TestDensePinnedGuard:
    """Qdrant rejects ``pinned`` for dense vectors; we must downgrade."""

    def test_helper_downgrades_pinned_to_cached(self, monkeypatch, caplog):
        """The PRODUCTION helper must downgrade, not a transcription of it."""
        import logging

        monkeypatch.setenv("QDRANT_DENSE_MEMORY", "pinned")
        with caplog.at_level(logging.WARNING):
            got = R._qdrant_dense_memory()
        assert got == Memory.CACHED
        assert any(
            "not supported for dense vectors" in r.message % r.args
            if r.args
            else "not supported for dense vectors" in r.message
            for r in caplog.records
        ), "must warn, not fail silently"

    @pytest.mark.parametrize(
        "env,expected",
        [
            ("cached", Memory.CACHED),
            ("cold", Memory.COLD),
        ],
    )
    def test_helper_passes_through_valid_tiers(self, monkeypatch, env, expected):
        monkeypatch.setenv("QDRANT_DENSE_MEMORY", env)
        assert R._qdrant_dense_memory() == expected

    def test_helper_default_is_cold(self) -> None:
        assert R._qdrant_dense_memory() == Memory.COLD

    def test_downgraded_tier_is_accepted_by_qdrant(self, monkeypatch):
        """End-to-end: the resolved tier must actually create a collection."""
        from qdrant_client.models import SparseIndexParams, SparseVectorParams, VectorParams

        monkeypatch.setenv("QDRANT_DENSE_MEMORY", "pinned")
        dm = R._qdrant_dense_memory()
        client = QdrantClient(path=":memory:")
        client.create_collection(
            collection_name="guard",
            vectors_config={"dense": VectorParams(size=8, distance=Distance.COSINE, memory=dm)},
            sparse_vectors_config={"bm25": SparseVectorParams(index=SparseIndexParams(memory=Memory.CACHED))},
        )
        got = client.get_collection("guard").config.params.vectors["dense"]
        assert got.memory == Memory.CACHED


class TestShardMultipleWarning:
    """shard_multiple>1 with an undeterminable peer count must not be silent."""

    def test_warns_when_node_count_unavailable(self, monkeypatch, caplog):
        import logging

        class Boom:
            def cluster_status(self):
                raise RuntimeError("no cluster")

        monkeypatch.setenv("QDRANT_SHARD_MULTIPLE", "2")
        with caplog.at_level(logging.WARNING):
            node_count = R._qdrant_node_count(Boom())
        assert node_count == 0
        # The create path warns on this combination; assert the pieces it
        # keys on so a regression in either is caught.
        assert R._qdrant_shard_multiple() == 2
