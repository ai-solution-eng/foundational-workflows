"""Feature D (multi-vector late interaction) — offline test suite.

Everything here runs OFFLINE: embedded local Qdrant (':memory:'), a
deterministic fake token embedder, and (for the wire-shape honesty tests)
a localhost http.server thread serving fixed responses — no external
network, no model endpoints.

The MaxSim tests assert HAND-COMPUTED values (wave contract, section 5):
each expected number is derived in the comment above its assertion.
"""

import asyncio
import gc
import json
import os
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    PointStruct,
    VectorParams,
)

from multimodal_rag import late_interaction as late_lane
from multimodal_rag import vector_store as vs_mod
from multimodal_rag.vector_store import (
    QdrantVectorStore,
    upsert_late_sidecar_entries,
    write_late_sidecar_for_docs,
)


@contextmanager
def _env(**overrides: str | None):
    """Temporarily set env vars; a None value REMOVES the key.

    Standalone-safe counterpart of monkeypatch.setenv/delenv.
    """
    saved = {k: os.environ.get(k) for k in overrides}

    def _apply() -> None:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    _apply()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenEmbedder:
    """Deterministic TokenEmbedder: each text maps to a FIXED token matrix.

    'tables' maps a text to its token matrix; unknown texts embed each
    whitespace-separated token through a vocab dict, falling back to a
    zero matrix.  Records every call for assertions.
    """

    def __init__(self, tables: dict[str, list[list[float]]] | None = None, dim: int = 2):
        self.tables = tables or {}
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_tokens(self, texts: list[str]) -> list[list[list[float]]]:
        self.calls.append(list(texts))
        return [self.tables.get(t, [[0.0] * self.dim]) for t in texts]


class _FakeDM:
    """Duck-typed DatasetManager for set_dataset_late_interaction tests.

    'persist_features' False reproduces TODAY's allowlist update_dataset
    (unknown keys silently ignored) to exercise the verify-and-warn path;
    True simulates the features passthrough impl-memory-lifecycle is adding.
    """

    def __init__(self, store: Any, doc_count: int = 3, persist_features: bool = False):
        self.store = store
        self.doc_count = doc_count
        self.persist_features = persist_features
        self.updates: list[dict[str, Any]] = []
        self.features: dict[str, Any] = {}

    def _get_rag(self, name: str) -> Any:
        return type("_Rag", (), {"vector_store": self.store})()

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        return {"document_count": self.doc_count, "features": dict(self.features)}

    def update_dataset(self, name: str, updates: dict[str, Any]) -> None:
        self.updates.append(updates)
        feats = updates.get("features")
        if isinstance(feats, dict) and self.persist_features:
            block = feats.get("late_interaction")
            if block is None or block == {}:
                self.features.pop("late_interaction", None)
            else:
                self.features["late_interaction"] = dict(block)


class _StubDenseEmbedding:
    """aembed_query returning a FIXED dense vector (async search path)."""

    def __init__(self, vec: list[float]):
        self.vec = vec

    async def aembed_query(self, query: Any) -> list[float]:
        return list(self.vec)


class _TokenEmbeddingsServer:
    """Local OpenAI-style /embeddings endpoint with a FIXED response table.

    mode 'token'   - data[i].embedding = list[list[float]] (token matrices)
    mode 'pooled'  - data[i].embedding = list[float] (single-vector endpoint:
                     the refusal case the honesty constraint demands)
    mode 'error'   - HTTP 500
    """

    def __init__(self, tables: dict[str, list[list[float]]], mode: str = "token"):
        self.tables = tables
        self.mode = mode
        self._server = HTTPServer(("127.0.0.1", 0), self._handler_factory())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def _handler_factory(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                inputs = body.get("input") or []
                if outer.mode == "error":
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{"error": "boom"}')
                    return
                data = []
                if outer.mode == "short":
                    # Fewer entries than inputs: the alignment must be
                    # refused, never guessed.
                    payload = json.dumps({"data": [], "model": "stub"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                for i, text in enumerate(inputs):
                    if outer.mode == "pooled":
                        embedding: Any = [0.1, 0.2, 0.3]
                    else:
                        embedding = outer.tables.get(str(text), [[0.0, 0.0]])
                    data.append({"index": i, "embedding": embedding, "object": "embedding"})
                payload = json.dumps({"data": data, "model": "stub", "object": "list"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:  # silence test output
                pass

        return Handler

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# MaxSim — hand-computed values (THE acceptance tests)
# ---------------------------------------------------------------------------


class TestMaxSimHandComputed:
    Q: ClassVar[list[list[float]]] = [[1.0, 0.0], [0.0, 1.0]]

    def test_identical_unit_matrices_score_one(self):
        # q1=[1,0]: max(dot with [1,0], dot with [0,1]) = max(1, 0) = 1
        # q2=[0,1]: max(0, 1) = 1;  score = (1 + 1) / 2 = 1.0
        assert late_lane.maxsim(self.Q, [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)

    def test_hand_computed_half(self):
        # q1=[1,0]: max(1, 0) = 1;  q2=[0,1]: max(0, 0) = 0;  (1 + 0) / 2 = 0.5
        assert late_lane.maxsim(self.Q, [[1.0, 0.0], [0.0, 0.0]]) == pytest.approx(0.5)

    def test_hand_computed_quarter_single_doc_token(self):
        # q1=[1,0]: dot([1,0],[0.5,0]) = 0.5;  q2=[0,1]: dot = 0.0
        # (0.5 + 0.0) / 2 = 0.25
        assert late_lane.maxsim(self.Q, [[0.5, 0.0]]) == pytest.approx(0.25)

    def test_hand_computed_three_dim_unnormalized(self):
        # q1=[2,0,0]: dots vs d-rows [0,1,0]/[0,0,1]/[1,0,0] -> 0/0/2 -> max 2
        # q2=[0,1,0]: dots -> 1/0/0 -> max 1;  score = (2 + 1) / 2 = 1.5
        q = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        d = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        assert late_lane.maxsim(q, d) == pytest.approx(1.5)

    def test_document_token_order_invariance(self):
        # MaxSim is invariant under permutation of the DOCUMENT's tokens:
        # the max over j does not depend on row order.
        q = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        d = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        assert late_lane.maxsim(q, d) == pytest.approx(late_lane.maxsim(q, d[::-1]))

    def test_query_token_order_changes_the_mean(self):
        # The MEAN is over query tokens, so swapping query rows swaps which
        # maxima are averaged: here both orderings coincidentally agree in
        # sum, so assert against the explicit hand values instead.
        q = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        d = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        # forward: q1 max = max(0, 2) = 2; q2 max = max(1, 0) = 1 -> 1.5
        assert late_lane.maxsim(q, d) == pytest.approx(1.5)
        # reversed query: q1'=[0,1,0] -> max(1,0)=1; q2'=[2,0,0] -> max(0,2)=2 -> 1.5
        assert late_lane.maxsim(q[::-1], d) == pytest.approx(1.5)

    def test_empty_document_scores_negative_inf(self):
        # A document with no tokens cannot satisfy any query term: defined
        # as -inf so it ranks last and never produces NaN.
        assert late_lane.maxsim(self.Q, []) == float("-inf")

    def test_empty_query_scores_zero(self):
        # Nothing to satisfy: 0.0 (mean over an empty set is defined as 0).
        assert late_lane.maxsim([], [[1.0, 0.0]]) == 0.0

    def test_dimension_mismatch_is_an_error_not_a_low_score(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            late_lane.maxsim([[1.0, 0.0]], [[1.0, 0.0, 0.0]])

    def test_score_documents_preserves_input_order(self):
        # [a -> 1.0, b -> 0.5, c -> 0.25] hand-computed above.
        docs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0]],
        ]
        assert late_lane.score_documents(self.Q, docs) == pytest.approx([1.0, 0.5, 0.25])

    def test_every_query_term_uses_its_own_best_document_token(self):
        # q = [e1, e2]; d has e1 and a 45-degree token.  Term 1's best is
        # dot(e1, e1) = 1; term 2's best is dot(e2, [0.6, 0.8]) = 0.8
        # (0.6*0 + 0.8*1) -> score (1 + 0.8) / 2 = 0.9.  A single shared
        # best token would give (1 + 1) / 2 = 1.0 - the distinction MaxSim
        # exists for.
        q = [[1.0, 0.0], [0.0, 1.0]]
        d = [[1.0, 0.0], [0.6, 0.8]]
        assert late_lane.maxsim(q, d) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Storage multiplier — COMPUTED from token counts
# ---------------------------------------------------------------------------


class TestStorageMultiplierComputed:
    def test_multiplier_is_the_token_count_at_equal_dtype(self):
        # 512 tokens x 4096 dim x 4 B = 8,388,608 B vs 4096 x 4 B = 16,384 B
        # -> ratio exactly 512.0.
        assert late_lane.storage_bytes_late(512, 4096) == 512 * 4096 * 4
        assert late_lane.storage_bytes_dense(4096) == 4096 * 4
        assert late_lane.storage_multiplier(512, 4096) == pytest.approx(512.0)

    def test_multiplier_independent_of_dim_and_dtype(self):
        # tokens*dim*dt / (dim*dt) = tokens — arithmetic, for ANY dim/dtype.
        assert late_lane.storage_multiplier(128, 768) == pytest.approx(128.0)
        assert late_lane.storage_multiplier(128, 768, dtype_bytes=2) == pytest.approx(128.0)
        assert late_lane.storage_multiplier(205, 1024) == pytest.approx(205.0)

    def test_zero_dimension_ratio_raises(self):
        with pytest.raises(ValueError):
            late_lane.storage_multiplier(10, 0)

    def test_note_is_generated_from_arithmetic(self):
        # The module-level note must EQUAL storage_note() for the documented
        # representative document (512 tokens, dim 4096) and must CONTAIN the
        # computed byte figures - a hand-asserted "205x" would fail here.
        note = late_lane.STORAGE_MULTIPLIER_NOTE
        assert note == late_lane.storage_note(512, 4096)
        assert "8,388,608" in note  # 512 * 4096 * 4
        assert "16,384" in note  # 4096 * 4
        assert "512.0x" in note
        # The contract's ~205x is a TOKEN-COUNT instance of the same formula.
        assert "205" in note

    def test_custom_document_note_recomputes(self):
        note = late_lane.storage_note(tokens_per_doc=128, dim=768)
        assert str(128 * 768 * 4) in note.replace(",", "") or f"{128 * 768 * 4:,}" in note
        assert "128.0x" in note


# ---------------------------------------------------------------------------
# Token embedder resolution — fail-closed honesty
# ---------------------------------------------------------------------------


class TestTokenEmbedderResolution:
    def test_unconfigured_by_default(self):
        with _env(
            **{
                k: None
                for k in (
                    late_lane.TOKEN_EMBEDDER_URL_ENV,
                    late_lane.TOKEN_EMBEDDER_MODEL_ENV,
                )
            }
        ):
            embedder = late_lane.resolve_token_embedder(None)
            assert isinstance(embedder, late_lane.UnconfiguredTokenEmbedder)
            with pytest.raises(late_lane.LateInteractionUnavailable) as excinfo:
                embedder.embed_tokens(["anything"])
            # The remediation names the env keys - the contract's "clear error".
            assert late_lane.TOKEN_EMBEDDER_URL_ENV in str(excinfo.value)

    def test_env_config_builds_http_embedder(self):
        with _env(
            **{
                late_lane.TOKEN_EMBEDDER_URL_ENV: "http://token-svc:8080/v1",
                late_lane.TOKEN_EMBEDDER_MODEL_ENV: "colbert-stub",
            }
        ):
            embedder = late_lane.resolve_token_embedder(None)
            assert isinstance(embedder, late_lane.HttpTokenEmbedder)
            assert embedder.base_url == "http://token-svc:8080/v1"
            assert embedder.model == "colbert-stub"
            assert late_lane.embedder_label(embedder) == "colbert-stub@http://token-svc:8080/v1"

    def test_config_overrides_env(self):
        with _env(**{late_lane.TOKEN_EMBEDDER_URL_ENV: "http://env-endpoint/v1"}):
            embedder = late_lane.resolve_token_embedder({"url": "http://cfg-endpoint/v1"})
            assert embedder.base_url == "http://cfg-endpoint/v1"

    def test_bad_scheme_rejected_at_construction(self):
        with pytest.raises(ValueError):
            late_lane.resolve_token_embedder({"url": "ftp://nope"})

    def test_bad_timeout_rejected(self):
        with pytest.raises(ValueError):
            late_lane.resolve_token_embedder({"url": "http://x", "timeout_s": "later"})

    def test_empty_texts_short_circuit_no_http(self):
        server = _TokenEmbeddingsServer({}, mode="error")  # would 500 if called
        try:
            embedder = late_lane.HttpTokenEmbedder(server.url)
            assert embedder.embed_tokens([]) == []
        finally:
            server.close()


class TestHttpTokenEmbedderWireShape:
    TABLES: ClassVar[dict[str, list[list[float]]]] = {
        "alpha": [[3.0, 4.0], [0.0, 0.0]],  # token 1 normalizes to [0.6, 0.8]
        "beta": [[0.0, 5.0]],
    }

    def test_token_matrices_accepted_and_l2_normalized(self):
        server = _TokenEmbeddingsServer(self.TABLES)
        try:
            out = late_lane.HttpTokenEmbedder(server.url).embed_tokens(["alpha", "beta"])
            assert out[0][0] == pytest.approx([0.6, 0.8], abs=1e-6)
            assert out[0][1] == [0.0, 0.0]  # zero vector stays zero (no NaN)
            assert out[1] == [[0.0, 1.0]]
        finally:
            server.close()

    def test_pooled_single_vector_response_is_refused(self):
        # THE honesty test: a single-vector endpoint (exactly what this
        # deployment's configured embedder is) must be REFUSED, never
        # repurposed into fake token vectors.
        server = _TokenEmbeddingsServer({}, mode="pooled")
        try:
            with pytest.raises(late_lane.LateInteractionUnavailable, match="pooled"):
                late_lane.HttpTokenEmbedder(server.url).embed_tokens(["alpha"])
        finally:
            server.close()

    def test_http_error_wrapped_as_unavailable(self):
        server = _TokenEmbeddingsServer({}, mode="error")
        try:
            with pytest.raises(late_lane.LateInteractionUnavailable, match="HTTP 500"):
                late_lane.HttpTokenEmbedder(server.url).embed_tokens(["alpha"])
        finally:
            server.close()

    def test_entry_count_mismatch_refused(self):
        server = _TokenEmbeddingsServer({}, mode="short")
        try:
            with pytest.raises(late_lane.LateInteractionUnavailable, match="entries"):
                late_lane.HttpTokenEmbedder(server.url).embed_tokens(["alpha"])
        finally:
            server.close()

    def test_non_list_embedding_refused(self):
        # A response entry whose embedding is not a list at all is refused.
        server = _TokenEmbeddingsServer(self.TABLES)
        try:
            embedder = late_lane.HttpTokenEmbedder(server.url)
            with pytest.raises(late_lane.LateInteractionUnavailable, match="token matrix"):
                embedder._token_matrix({"embedding": None})
        finally:
            server.close()

    def test_ragged_matrix_refused(self):
        server = _TokenEmbeddingsServer({"alpha": [[1.0, 0.0], [1.0]]})
        try:
            with pytest.raises(late_lane.LateInteractionUnavailable, match="ragged"):
                late_lane.HttpTokenEmbedder(server.url).embed_tokens(["alpha"])
        finally:
            server.close()


# ---------------------------------------------------------------------------
# Sidecar schema (embedded local Qdrant)
# ---------------------------------------------------------------------------


class TestSidecarSchema:
    def test_sidecar_name_idempotent(self):
        assert late_lane.sidecar_collection_name("ds") == "ds__late"
        assert late_lane.sidecar_collection_name("ds__late") == "ds__late"

    def test_ensure_creates_maxsim_multivector(self):
        client = QdrantClient(":memory:")
        name = late_lane.ensure_late_sidecar(client, "ds", 2)
        assert name == "ds__late"
        params = client.get_collection("ds__late").config.params.vectors["late"]
        assert int(params.size) == 2
        assert params.distance == Distance.COSINE
        assert params.multivector_config.comparator == MultiVectorComparator.MAX_SIM
        # Idempotent: a second ensure with the same dim succeeds.
        assert late_lane.ensure_late_sidecar(client, "ds", 2) == "ds__late"

    def test_ensure_conflicting_dim_fails_closed(self):
        client = QdrantClient(":memory:")
        late_lane.ensure_late_sidecar(client, "ds", 2)
        with pytest.raises(late_lane.LateInteractionUnavailable, match="token dim 2"):
            late_lane.ensure_late_sidecar(client, "ds", 3)

    def test_ensure_rejects_sidecar_without_late_vector(self):
        client = QdrantClient(":memory:")
        client.create_collection("ds__late", vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)})
        with pytest.raises(late_lane.LateInteractionUnavailable, match="no 'late' named"):
            late_lane.ensure_late_sidecar(client, "ds", 2)

    def test_dim_reader_and_delete(self):
        client = QdrantClient(":memory:")
        assert late_lane.late_sidecar_dim(client, "ds") is None
        late_lane.ensure_late_sidecar(client, "ds", 5)
        assert late_lane.late_sidecar_dim(client, "ds") == 5
        assert late_lane.delete_late_sidecar(client, "ds") is True
        assert late_lane.delete_late_sidecar(client, "ds") is False
        # Best-effort point delete: absent sidecar and absent ids never raise.
        late_lane.delete_late_sidecar_points(client, "ds", ["whatever"])
        late_lane.delete_late_sidecar_points(client, "ds", [])

    def test_invalid_dim_rejected(self):
        with pytest.raises(ValueError):
            late_lane.build_late_vector_params(0)


# ---------------------------------------------------------------------------
# Ingest-side sidecar writes — fail-closed
# ---------------------------------------------------------------------------


class TestSidecarIngest:
    def _mk_client(self) -> QdrantClient:
        return QdrantClient(":memory:")

    def test_writes_token_matrices_with_matching_ids(self):
        client = self._mk_client()
        late_lane.ensure_late_sidecar(client, "ds", 2)
        pid = uuid.uuid4().hex
        embedder = _FakeTokenEmbedder({"hello world": [[1.0, 0.0], [0.0, 1.0]]})
        written = upsert_late_sidecar_entries(client, "ds", [(pid, "hello world")], embedder)
        assert written == 1
        pts = client.retrieve("ds__late", ids=[pid], with_payload=True, with_vectors=["late"])
        assert len(pts) == 1
        assert pts[0].payload == {"doc_id": pid}
        matrix = pts[0].vector["late"]
        np.testing.assert_allclose(matrix, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)

    def test_unconfigured_embedder_fails_closed_at_ingest(self):
        client = self._mk_client()
        late_lane.ensure_late_sidecar(client, "ds", 2)
        with pytest.raises(late_lane.LateInteractionUnavailable, match="RAG_TOKEN_EMBEDDER_URL"):
            upsert_late_sidecar_entries(
                client, "ds", [(uuid.uuid4().hex, "text")], late_lane.UnconfiguredTokenEmbedder()
            )

    def test_no_sidecar_and_no_dim_fails_closed(self):
        client = self._mk_client()
        with pytest.raises(late_lane.LateInteractionUnavailable, match="set_late_interaction"):
            upsert_late_sidecar_entries(client, "ds", [(uuid.uuid4().hex, "text")], _FakeTokenEmbedder())

    def test_dimension_drift_fails_closed(self):
        client = self._mk_client()
        late_lane.ensure_late_sidecar(client, "ds", 2)
        drifting = _FakeTokenEmbedder({"text": [[1.0, 0.0, 0.0]]})
        with pytest.raises(late_lane.LateInteractionUnavailable, match="re-enable"):
            upsert_late_sidecar_entries(client, "ds", [(uuid.uuid4().hex, "text")], drifting)
        # The failed batch must not have written a poisoned point.
        assert client.count("ds__late", exact=True).count == 0

    def test_write_late_sidecar_for_docs_disabled_is_noop(self):
        client = self._mk_client()
        client.create_collection("main", vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)})
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / ".late_interaction.json"
            state.write_text(json.dumps({"enabled": False}))
            store = QdrantVectorStore(
                client=client,
                collection_name="main",
                embedding=object(),
                vector_name="dense",
                late_state_path=str(state),
            )
            assert write_late_sidecar_for_docs(store, [uuid.uuid4().hex], ["text"]) == 0
            assert not any(c.name == "main__late" for c in client.get_collections().collections)

    def test_write_late_sidecar_for_docs_enabled_writes(self):
        client = self._mk_client()
        client.create_collection("main", vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)})
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / ".late_interaction.json"
            state.write_text(json.dumps({"enabled": True, "token_dim": 2}))
            store = QdrantVectorStore(
                client=client,
                collection_name="main",
                embedding=object(),
                vector_name="dense",
                late_state_path=str(state),
            )
            pid = uuid.uuid4().hex
            fake = _FakeTokenEmbedder({"text": [[1.0, 0.0], [0.0, 1.0]]})
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                assert write_late_sidecar_for_docs(store, [pid], ["text"]) == 1
            finally:
                monkey.undo()
            assert client.count("main__late", exact=True).count == 1


# ---------------------------------------------------------------------------
# Search-side rerank (the wrapper around the batch core)
# ---------------------------------------------------------------------------

DIM = 4
COLL = "late_rerank_ds"

# Dense vectors chosen so the DENSE order (query [1,0,0,0]) is C > A > B,
# while the MaxSim order (query tokens e1/e2) is A > B > C - a real flip.
DOC_DENSE = {
    "A": [0.3, 0.2, 0.0, 0.0],  # cos 0.832
    "B": [0.5, 0.4, 0.0, 0.0],  # cos 0.781
    "C": [0.9, 0.1, 0.0, 0.0],  # cos 0.994
    "D": [0.0, 0.3, 0.0, 0.0],  # cos 0.0 (sidecar: absent -> missing_tokens)
}
DOC_LATE = {
    "A": [[1.0, 0.0], [0.0, 1.0]],  # maxsim 1.0  (hand-computed)
    "B": [[1.0, 0.0], [0.6, 0.8]],  # maxsim 0.9  (hand-computed)
    "C": [[0.0, 1.0], [0.0, 1.0]],  # maxsim 0.5  (hand-computed)
}
QUERY_TOKENS = [[1.0, 0.0], [0.0, 1.0]]


class _RerankRig:
    """Main collection + partial sidecar + state file, on local Qdrant."""

    def __init__(self, with_sidecar: bool = True, enabled: bool = True):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(COLL, vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
        self.ids: dict[str, str] = {}
        points = []
        for key, vec in DOC_DENSE.items():
            pid = uuid.uuid4().hex
            self.ids[key] = pid
            points.append(
                PointStruct(
                    id=pid,
                    vector={"dense": vec},
                    payload={"page_content": f"doc-{key}", "metadata": {"source": f"f{key}"}},
                )
            )
        self.client.upsert(COLL, points=points)
        if with_sidecar:
            late_lane.ensure_late_sidecar(self.client, COLL, 2)
            entries = [(self.ids[k], f"doc-{k}") for k in DOC_LATE]
            tables = {f"doc-{k}": matrix for k, matrix in DOC_LATE.items()}
            upsert_late_sidecar_entries(self.client, COLL, entries, _FakeTokenEmbedder(tables))
        self.state_path = Path(self.tmp.name) / ".late_interaction.json"
        self.state_path.write_text(json.dumps({"enabled": enabled, "token_dim": 2}))
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=COLL,
            embedding=object(),
            vector_name="dense",
            late_state_path=str(self.state_path),
        )

    def close(self) -> None:
        self.tmp.cleanup()

    def query(self, k: int = 4, text: str | None = "alpha beta"):
        return self.store.similarity_search_with_score_by_vector_batch(
            [([1.0, 0.0, 0.0, 0.0], k, True, None, text, None)]
        )[0]


def _labels(per_query) -> list[str]:
    return [doc.metadata.get("_late") for doc, _ in per_query]


def _keys(per_query) -> list[str]:
    return [str(doc.metadata.get("source")) for doc, _ in per_query]


class TestSearchRerank:
    def test_default_off_is_byte_identical(self):
        # NO state file -> the wrapper must return EXACTLY the core result:
        # same order, same scores, and NO _late stamps anywhere.
        with tempfile.TemporaryDirectory() as _tmp:
            rig = _RerankRig(with_sidecar=False, enabled=False)
            try:
                rig.state_path.unlink()
                queries = [([1.0, 0.0, 0.0, 0.0], 4, True, None, "alpha beta", None)]
                core = rig.store._similarity_search_batch_core(queries)[0]
                wrapped = rig.store.similarity_search_with_score_by_vector_batch(queries)[0]
                assert [(d.page_content, s) for d, s in core] == [(d.page_content, s) for d, s in wrapped]
                assert all("_late" not in d.metadata for d, _ in wrapped)
                # And the dense order is C before A before B (documented above).
                assert _keys(wrapped)[:3] == ["fC", "fA", "fB"]
            finally:
                rig.close()

    def test_late_enabled_flips_order_to_maxsim_with_hand_scores(self):
        rig = _RerankRig()
        try:
            fake = _FakeTokenEmbedder({"alpha beta": QUERY_TOKENS})
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                out = rig.query()
            finally:
                monkey.undo()
            # MaxSim order A (1.0) > B (0.9) > C (0.5); D has no sidecar
            # matrix -> last, stamped missing_tokens, dense score kept.
            assert _keys(out) == ["fA", "fB", "fC", "fD"]
            assert _labels(out) == ["maxsim", "maxsim", "maxsim", "missing_tokens"]
            assert out[0][1] == pytest.approx(1.0, abs=1e-3)
            assert out[1][1] == pytest.approx(0.9, abs=1e-3)
            assert out[2][1] == pytest.approx(0.5, abs=1e-3)
            assert out[0][0].metadata["_late_score"] == pytest.approx(1.0, abs=1e-3)
            # D keeps its ORIGINAL dense score (not a fabricated MaxSim).
            core = rig.store._similarity_search_batch_core([([1.0, 0.0, 0.0, 0.0], 4, True, None, "alpha beta", None)])[
                0
            ]
            dense_d = next(s for d, s in core if d.metadata.get("source") == "fD")
            assert out[3][1] == pytest.approx(dense_d)
        finally:
            rig.close()

    def test_late_enabled_without_embedder_stamps_unavailable(self):
        # Contract: search on a late-enabled dataset with no embedder
        # degrades to dense+bm25 with metadata["_late"]="unavailable".
        rig = _RerankRig()
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(
                late_lane, "resolve_token_embedder", lambda config=None: late_lane.UnconfiguredTokenEmbedder()
            )
            try:
                out = rig.query()
            finally:
                monkey.undo()
            core = rig.store._similarity_search_batch_core([([1.0, 0.0, 0.0, 0.0], 4, True, None, "alpha beta", None)])[
                0
            ]
            assert _labels(out) == ["unavailable"] * 4
            assert _keys(out) == _keys(core)  # dense+bm25 order untouched
        finally:
            rig.close()

    def test_missing_sidecar_stamps_unavailable(self):
        rig = _RerankRig()
        try:
            rig.client.delete_collection("late_rerank_ds__late")
            out = rig.query()
            assert _labels(out) == ["unavailable"] * 4
        finally:
            rig.close()

    def test_multimodal_query_is_skipped_not_faked(self):
        # A vector-supplied (multimodal) query has no text -> no query token
        # matrix can exist -> results keep core order, stamped "skipped".
        rig = _RerankRig()
        try:
            out = rig.query(text=None)
            core = rig.store._similarity_search_batch_core([([1.0, 0.0, 0.0, 0.0], 4, True, None, None, None)])[0]
            assert _labels(out) == ["skipped"] * 4
            assert _keys(out) == _keys(core)
        finally:
            rig.close()

    def test_embedder_crash_degrades_without_crashing_search(self):
        class _Boom:
            def embed_tokens(self, texts):
                raise RuntimeError("socket exploded")

        rig = _RerankRig()
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: _Boom())
            try:
                out = rig.query()
            finally:
                monkey.undo()
            assert _labels(out) == ["unavailable"] * 4
        finally:
            rig.close()

    def test_real_env_http_embedder_reranks(self):
        # End-to-end honesty: env-configured embedder -> localhost HTTP
        # endpoint -> normalized token matrices -> same hand-computed order.
        server = _TokenEmbeddingsServer({"alpha beta": QUERY_TOKENS})
        rig = _RerankRig()
        try:
            with _env(
                **{
                    late_lane.TOKEN_EMBEDDER_URL_ENV: server.url,
                    late_lane.TOKEN_EMBEDDER_MODEL_ENV: "stub",
                }
            ):
                out = rig.query()
            assert _keys(out) == ["fA", "fB", "fC", "fD"]
            assert out[0][1] == pytest.approx(1.0, abs=1e-3)
        finally:
            rig.close()
            server.close()

    def test_async_batcher_path_applies_rerank(self):
        # The batcher funnels text searches through the PUBLIC batch method;
        # prove the rerank fires on that path too (asimilarity entry).
        rig = _RerankRig()
        try:
            rig.store.embedding = _StubDenseEmbedding([1.0, 0.0, 0.0, 0.0])
            fake = _FakeTokenEmbedder({"alpha beta": QUERY_TOKENS})
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                out = asyncio.run(rig.store.asimilarity_search_with_relevance_scores("alpha beta", 4))
            finally:
                monkey.undo()
            assert _keys(out) == ["fA", "fB", "fC", "fD"]
            assert _labels(out)[:3] == ["maxsim"] * 3
        finally:
            rig.close()


# ---------------------------------------------------------------------------
# State file + dataset-dir derivation
# ---------------------------------------------------------------------------


class TestLateState:
    def test_roundtrip_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert late_lane.read_late_state(tmp) == {}
            late_lane.write_late_state(tmp, {"enabled": True, "token_dim": 7})
            assert late_lane.read_late_state(tmp) == {"enabled": True, "token_dim": 7}
            late_lane.write_late_state(tmp, None)
            assert late_lane.read_late_state(tmp) == {}

    def test_corrupt_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / late_lane.LATE_STATE_FILENAME).write_text("{not json")
            assert late_lane.read_late_state(tmp) == {}
            (Path(tmp) / late_lane.LATE_STATE_FILENAME).write_text("[1, 2]")
            assert late_lane.read_late_state(tmp) == {}

    def test_dataset_dir_derivation(self):
        client = QdrantClient(":memory:")
        explicit = QdrantVectorStore(
            client=client, collection_name="c", embedding=object(), late_state_path="/x/.late_interaction.json"
        )
        assert explicit.dataset_dir() == Path("/x")
        derived = QdrantVectorStore(
            client=client,
            collection_name="c",
            embedding=object(),
            bm25_stats_path="/data/datasets/ds/files/.bm25_stats.json",
        )
        assert derived.dataset_dir() == Path("/data/datasets/ds")
        bare = QdrantVectorStore(client=client, collection_name="c", embedding=object())
        assert bare.dataset_dir() is None
        assert bare.late_state() == {}
        assert bare.late_enabled() is False


# ---------------------------------------------------------------------------
# set_dataset_late_interaction (shared MCP/REST implementation)
# ---------------------------------------------------------------------------


class TestSetDatasetLateInteraction:
    def _rig(self, persist_features: bool = False, doc_count: int = 3):
        tmp = tempfile.TemporaryDirectory()
        client = QdrantClient(":memory:")
        client.create_collection("ds", vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)})
        if doc_count:
            # Seed the exact number of main points the count assertion reads.
            client.upsert(
                "ds",
                points=[
                    PointStruct(id=uuid.uuid4().hex, vector={"dense": [0.1] * DIM}, payload={})
                    for _ in range(doc_count)
                ],
            )
        state = Path(tmp.name) / ".late_interaction.json"
        store = QdrantVectorStore(
            client=client,
            collection_name="ds",
            embedding=object(),
            vector_name="dense",
            bm25_stats_path=str(Path(tmp.name) / "files" / ".bm25_stats.json"),
        )
        dm = _FakeDM(store, doc_count=doc_count, persist_features=persist_features)
        return tmp, client, store, dm, state

    def test_enable_without_embedder_fails_closed_and_changes_nothing(self):
        tmp, client, _store, dm, state = self._rig()
        try:
            with (
                _env(**{k: None for k in (late_lane.TOKEN_EMBEDDER_URL_ENV,)}),
                pytest.raises(late_lane.LateInteractionUnavailable, match="RAG_TOKEN_EMBEDDER_URL"),
            ):
                vs_mod.set_dataset_late_interaction(dm, "ds", enabled=True, token_dim=2)
            # Nothing was changed: no state file, no sidecar, no meta write.
            assert not state.exists()
            assert not any(c.name == "ds__late" for c in client.get_collections().collections)
            assert dm.updates == []
        finally:
            tmp.cleanup()

    def test_enable_probes_dim_creates_sidecar_reports_reingest(self):
        tmp, client, _store, dm, state = self._rig(doc_count=3)
        fake = _FakeTokenEmbedder({"late interaction dimension probe": QUERY_TOKENS})
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                result = vs_mod.set_dataset_late_interaction(dm, "ds", enabled=True, token_dim=None)
            finally:
                monkey.undo()
            assert result["enabled"] is True
            assert result["token_dim"] == 2  # measured from the probe
            assert result["document_count"] == 3
            assert result["reingest_required"] is True
            assert "3 existing document(s)" in result["message"]
            # Sidecar schema exists with the probed dim; state file enabled.
            params = client.get_collection("ds__late").config.params.vectors["late"]
            assert int(params.size) == 2
            assert state.exists() and json.loads(state.read_text())["enabled"] is True
            assert json.loads(state.read_text())["embedder"] == late_lane.embedder_label(fake)
            # The features passthrough is NOT in this build (allowlist dm) -
            # the verify step must WARN, not lie.
            assert len(dm.updates) == 1
            block = dm.updates[0]["features"]["late_interaction"]
            assert block["enabled"] is True
            assert block["token_dim"] == 2
            assert block["embedder"] == json.loads(state.read_text())["embedder"]
            assert any("did not persist" in w for w in result["warnings"])
        finally:
            tmp.cleanup()

    def test_enable_with_persisting_dm_has_no_warning(self):
        tmp, _client, _store, dm, _state = self._rig(persist_features=True, doc_count=0)
        fake = _FakeTokenEmbedder({"late interaction dimension probe": QUERY_TOKENS})
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                result = vs_mod.set_dataset_late_interaction(dm, "ds", enabled=True)
            finally:
                monkey.undo()
            assert result["reingest_required"] is False
            assert "warnings" not in result
            assert dm.features["late_interaction"]["enabled"] is True
        finally:
            tmp.cleanup()

    def test_enable_explicit_dim_skips_probe(self):
        tmp, client, _store, _dm, _state = self._rig()
        fake = _FakeTokenEmbedder()
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                result = vs_mod.set_dataset_late_interaction(_dm, "ds", enabled=True, token_dim=3)
            finally:
                monkey.undo()
            assert result["token_dim"] == 3
            # Only the empty sentinel-trigger call - never a probe embedding.
            assert [c for c in fake.calls if c] == []
            assert int(client.get_collection("ds__late").config.params.vectors["late"].size) == 3
        finally:
            tmp.cleanup()

    def test_enable_conflicting_existing_sidecar_fails_closed(self):
        tmp, client, _store, _dm, _state = self._rig()
        late_lane.ensure_late_sidecar(client, "ds", 2)
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: _FakeTokenEmbedder())
            try:
                with pytest.raises(late_lane.LateInteractionUnavailable, match="token dim 2"):
                    vs_mod.set_dataset_late_interaction(_dm, "ds", enabled=True, token_dim=3)
            finally:
                monkey.undo()
        finally:
            tmp.cleanup()

    def test_disable_retains_sidecar_and_stops_reading(self):
        tmp, client, store, dm, state = self._rig()
        fake = _FakeTokenEmbedder({"late interaction dimension probe": QUERY_TOKENS})
        try:
            monkey = pytest.MonkeyPatch()
            monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
            try:
                vs_mod.set_dataset_late_interaction(dm, "ds", enabled=True, token_dim=2)
                result = vs_mod.set_dataset_late_interaction(dm, "ds", enabled=False)
            finally:
                monkey.undo()
            assert result["enabled"] is False
            assert result["sidecar_retained"] is True
            assert "delete_collection" in result["message"]
            # Sidecar STILL exists (nothing destroyed) ...
            assert any(c.name == "ds__late" for c in client.get_collections().collections)
            # ... but the retrieval plane reads the flipped state file.
            assert store.late_enabled() is False
            assert json.loads(state.read_text())["enabled"] is False
            # And the meta contract: the disable writes the None-removal shape.
            assert dm.updates[-1] == {"features": {"late_interaction": None}}
        finally:
            tmp.cleanup()

    def test_disable_fails_closed_without_dataset_dir(self):
        client = QdrantClient(":memory:")
        store = QdrantVectorStore(client=client, collection_name="ds", embedding=object())
        dm = _FakeDM(store)
        with pytest.raises(late_lane.LateInteractionUnavailable, match="no dataset directory"):
            vs_mod.set_dataset_late_interaction(dm, "ds", enabled=False)


class TestRealDatasetManagerIntegration:
    """End-to-end against the LANDED features passthrough (impl-memory-lifecycle).

    Uses the house conftest fixtures (offline DatasetManager + stub embedder
    + local Qdrant storage under the tmp base_path).  This is the test that
    proves the set-late-interaction surfaces and the dataset_manager
    passthrough agree on the {"features": {"late_interaction": ...}} shape.
    """

    def test_enable_persists_features_block_real_dm(self, base_path, embedder):
        from multimodal_rag.dataset_manager import DatasetManager

        dm = DatasetManager(base_path=base_path, embedder=embedder)
        dm.create_dataset("late-e2e")
        fake = _FakeTokenEmbedder()
        monkey = pytest.MonkeyPatch()
        monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
        try:
            with _env(**{late_lane.TOKEN_EMBEDDER_URL_ENV: "http://token-svc:9/v1"}):
                result = vs_mod.set_dataset_late_interaction(dm, "late-e2e", enabled=True, token_dim=2)
        finally:
            monkey.undo()
        assert result["enabled"] is True
        assert result["reingest_required"] is False  # empty dataset
        # The LANDED passthrough really persisted the block into meta.json.
        features = dm.get_dataset("late-e2e", sync_count=False).get("features") or {}
        assert features["late_interaction"]["enabled"] is True
        assert features["late_interaction"]["token_dim"] == 2
        # Retrieval-plane enablement, probed WITHOUT re-opening the storage
        # folder: the features write invalidated the cached RAG, and the
        # enable call's local-mode client must not be re-opened while its
        # flock can linger (local-mode-only mechanics; server mode has no
        # folder locks).  late_state() never touches the client.
        probe_store = QdrantVectorStore(
            client=None,
            collection_name="late-e2e",
            embedding=object(),
            bm25_stats_path=str(Path(base_path) / "datasets" / "late-e2e" / "files" / ".bm25_stats.json"),
        )
        assert probe_store.late_enabled() is True
        # Disable: the passthrough invalidated the cached RAG, so this
        # _get_rag re-opens the storage folder; collect first so the previous
        # local-mode client's flock is definitely released.
        gc.collect()
        monkey = pytest.MonkeyPatch()
        monkey.setattr(late_lane, "resolve_token_embedder", lambda config=None: fake)
        try:
            result = vs_mod.set_dataset_late_interaction(dm, "late-e2e", enabled=False)
        finally:
            monkey.undo()
        assert result["enabled"] is False
        assert result["sidecar_retained"] is True
        features = dm.get_dataset("late-e2e", sync_count=False).get("features") or {}
        assert not features.get("late_interaction")
        probe_store._invalidate_late_state()
        assert probe_store.late_enabled() is False

    def test_enable_without_embedder_fails_before_any_change(self, base_path, embedder):
        from multimodal_rag.dataset_manager import DatasetManager

        dm = DatasetManager(base_path=base_path, embedder=embedder)
        dm.create_dataset("late-e2e-neg")
        with (
            _env(**{late_lane.TOKEN_EMBEDDER_URL_ENV: None}),
            pytest.raises(late_lane.LateInteractionUnavailable, match="RAG_TOKEN_EMBEDDER_URL"),
        ):
            vs_mod.set_dataset_late_interaction(dm, "late-e2e-neg", enabled=True, token_dim=2)
        # No meta block, no sidecar collection - fail closed meant NOTHING.
        features = dm.get_dataset("late-e2e-neg", sync_count=False).get("features") or {}
        assert not features
        rag = dm._get_rag("late-e2e-neg")
        names = [c.name for c in rag.vector_store._client.get_collections().collections]
        assert not any(name.endswith("__late") for name in names)
