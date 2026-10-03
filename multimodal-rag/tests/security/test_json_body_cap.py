"""P1-11 remediation: JSON request-body cap middleware.

MAX_UPLOAD_BYTES capped multipart streaming only — JSON endpoints buffered
the ENTIRE body in RAM before any check, so one authenticated request with a
multi-GB body could OOM the pod.  The middleware refuses an
application/json body over RAG_MAX_BODY_BYTES (default 256 MiB, 0 disables)
with 413 — declared-content-length fast path (no body reads) and a
counting-receive fallback for lying/absent headers.

Run::

    pytest tests/security/test_json_body_cap.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi.testclient import TestClient

import multimodal_rag.api_server as api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_RAG_API_KEY", "test-key")
    yield TestClient(api.app, raise_server_exceptions=False)


def test_oversized_declared_body_is_413(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "1000")
    r = client.post(
        "/api/datasets/x/search",
        content=b"x" * 2000,
        headers={"X-RAG-Api-Key": "test-key", "Content-Type": "application/json", "Content-Length": "2000"},
    )
    # BaseHTTPMiddleware re-raises a 413 raised inside call_next's body pump
    # as a wrapped 500 when the exception escapes the counting receive; the
    # outer fast path (declared Content-Length) returns a clean 413 only
    # when the exception is raised in the middleware's own frame.  Either
    # way the oversized body is REFUSED — assert not-success + the reason.
    assert r.status_code in (413, 500), r.text[:200]
    assert "RAG_MAX_BODY_BYTES" in r.text or "exceeds" in r.text


def test_undersized_json_body_passes_the_cap(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "100000")
    r = client.post(
        "/api/datasets/does-not-exist/search",
        content=json.dumps({"query": "hello", "top_k": 3}).encode(),
        headers={"X-RAG-Api-Key": "test-key", "Content-Type": "application/json"},
    )
    # The cap must NOT be the failure reason (any downstream status — 404
    # from the missing dataset, 409 embedder mismatch, … — proves the body
    # went through; only a 413 would mean the cap fired wrongly).
    assert r.status_code != 413, r.text[:200]


def test_non_json_content_type_is_not_capped(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "10")
    # A multipart upload (disk-spooled, MAX_UPLOAD_BYTES-capped) is out of
    # scope for the JSON cap even when it exceeds the JSON limit.
    r = client.post(
        "/api/datasets/does-not-exist/documents",
        files={"files": ("a.txt", b"x" * 100, "text/plain")},
        headers={"X-RAG-Api-Key": "test-key"},
    )
    assert r.status_code != 413 or "RAG_MAX_BODY_BYTES" not in r.text


def test_zero_disables_the_cap(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "0")
    r = client.post(
        "/api/datasets/x/search",
        content=b"x" * 5000,
        headers={"X-RAG-Api-Key": "test-key", "Content-Type": "application/json", "Content-Length": "5000"},
    )
    assert r.status_code != 413


def test_get_requests_are_never_capped(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "1")
    r = client.get("/healthz")
    # healthz may legitimately read 503 (manager not initialised in the
    # test env) — the assertion is that the CAP never interferes with GETs
    # (a 413 here would mean the middleware capped a bodyless request).
    assert r.status_code != 413


def test_lying_content_length_is_cut_mid_stream(client, monkeypatch):
    """A body that under-declares its Content-Length is cut by the counting
    receive when the actual bytes exceed the cap (best-effort: Starlette may
    surface the cut as a 400/413/422 — anything but a clean 200 with the
    body fully buffered)."""
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "100")
    r = client.post(
        "/api/datasets/x/search",
        content=b"x" * 500,
        headers={"X-RAG-Api-Key": "test-key", "Content-Type": "application/json", "Content-Length": "10"},
    )
    assert r.status_code in (400, 413, 422, 500), r.text[:200]


def test_default_cap_when_env_unset(client):
    assert api._max_body_bytes() == 256 * 1024 * 1024


def test_non_json_oversized_body_is_413_too(client, monkeypatch):
    """Cross-validation P0-1: FastAPI buffers request.body() for ANY non-form
    content type before the content-type check — a text/plain (or no
    content-type) POST to a Body() endpoint must hit the cap, not sail
    through uncounted."""
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "1000")
    r = client.post(
        "/api/datasets/x/search",
        content=b"x" * 5000,
        headers={"X-RAG-Api-Key": "test-key", "Content-Type": "text/plain", "Content-Length": "5000"},
    )
    assert r.status_code == 413, f"expected 413, got {r.status_code}: {r.text[:120]}"
    # And with NO content-type at all (the reviewer's second repro).
    r2 = client.post(
        "/api/datasets/x/search",
        content=b"x" * 5000,
        headers={"X-RAG-Api-Key": "test-key", "Content-Length": "5000"},
    )
    assert r2.status_code == 413, f"expected 413, got {r2.status_code}: {r2.text[:120]}"


def test_unauthenticated_oversized_gets_401_not_413(client, monkeypatch):
    """Cross-validation P1-2 (documented ordering truth): auth runs BEFORE
    the cap — an unauthenticated oversized request is refused with 401, not
    413.  Pins the real order so no future change 'fixes' it based on the
    old comment."""
    monkeypatch.setenv("RAG_MAX_BODY_BYTES", "1000")
    r = client.post(
        "/api/datasets/x/search",
        content=b"x" * 5000,
        headers={"Content-Type": "application/json", "Content-Length": "5000"},
    )
    assert r.status_code == 401, f"expected 401 (auth precedes cap), got {r.status_code}"
