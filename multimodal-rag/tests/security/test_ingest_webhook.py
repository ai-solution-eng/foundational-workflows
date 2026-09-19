"""Wave-5 F2: opt-in ingest webhooks (RAG_WEBHOOK_URL).

Pinned behaviour:

* DEFAULT (RAG_WEBHOOK_URL unset): zero behaviour — no request, no latency,
  ``notify_ingest`` returns False.
* When set: one small JSON event (dataset, doc_count, status, timestamp)
  after a completed ingest, with ``X-RAG-Webhook-Secret`` when
  ``RAG_WEBHOOK_SECRET`` is set.
* Failures are LOGGED-NOT-FATAL: an unreachable/misbehaving receiver never
  fails an ingest that already succeeded.
* Timeout-capped: ``RAG_WEBHOOK_TIMEOUT`` (default 5.0 s, clamped ≥ 0.1).
* Dataset restore replays are muted (one logical ingest, not N events).

Run::

    pytest tests/security/test_ingest_webhook.py -q
"""

import http.server
import json
import os
import sys
import threading
import urllib.error
from typing import ClassVar

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.dataset_manager import DatasetManager
from multimodal_rag.utils import webhook

# ---------------------------------------------------------------------------
# Local HTTP capture server (loopback only — offline-safe)
# ---------------------------------------------------------------------------


class _Capture(http.server.BaseHTTPRequestHandler):
    events: ClassVar[list] = []
    status: ClassVar[int] = 200
    lock: ClassVar[threading.Lock] = threading.Lock()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with _Capture.lock:
            _Capture.events.append(
                {
                    "path": self.path,
                    "body": json.loads(body.decode("utf-8")),
                    "secret": self.headers.get("X-RAG-Webhook-Secret"),
                    "content_type": self.headers.get("Content-Type"),
                }
            )
        self.send_response(_Capture.status)
        self.end_headers()

    def log_message(self, *a):  # silence the request log
        pass


@pytest.fixture
def capture_server():
    _Capture.events = []
    _Capture.status = 200
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/hook"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _clean_webhook_env(monkeypatch):
    monkeypatch.delenv(webhook.WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(webhook.WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.delenv(webhook.WEBHOOK_TIMEOUT_ENV, raising=False)
    yield


@pytest.fixture
def dm(base_path, embedder, monkeypatch):
    """Real offline DatasetManager with one dataset (webhook call sites)."""
    monkeypatch.setenv("DATA_PATH", base_path)
    manager = DatasetManager(base_path=base_path, qdrant_host="", embedder=embedder)
    manager.create_dataset("docs")
    return manager


# ---------------------------------------------------------------------------
# Default-off
# ---------------------------------------------------------------------------


def test_default_unset_is_zero_behavior():
    assert webhook.webhook_url() == ""
    assert webhook.notify_ingest("ds", 3) is False


def test_unset_fires_nothing_on_real_ingest(dm, capture_server):
    """The env decides — a URL configured nowhere means no request ever."""
    dm.add_documents("docs", ["one", "two"])
    assert _Capture.events == []


# ---------------------------------------------------------------------------
# Fired on ingest completion
# ---------------------------------------------------------------------------


def test_fired_on_completed_ingest(dm, capture_server, monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    dm.add_documents("docs", ["alpha", "beta", "gamma"])
    assert len(_Capture.events) == 1
    ev = _Capture.events[0]
    assert ev["path"] == "/hook"
    assert ev["content_type"] == "application/json"
    assert ev["body"]["dataset"] == "docs"
    assert ev["body"]["doc_count"] == 3
    assert ev["body"]["status"] == "ok"
    assert ev["body"]["timestamp"]  # ISO-8601 present


def test_secret_header_sent_when_configured(dm, capture_server, monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    monkeypatch.setenv(webhook.WEBHOOK_SECRET_ENV, "s3cr3t-value")
    dm.add_documents("docs", ["payload"])
    assert _Capture.events[0]["secret"] == "s3cr3t-value"


def test_no_secret_header_when_unset(dm, capture_server, monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    dm.add_documents("docs", ["payload"])
    assert _Capture.events[0]["secret"] is None


def test_batch_ingest_fires_one_event_with_stored_points(dm, capture_server, monkeypatch):
    """add_urls_batch delegates to add_files_batch — exactly one event per
    completed batch, counting the stored points (dedup-skipped add 0)."""
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    dm.add_documents("docs", ["first", "second"])
    assert _Capture.events[-1]["body"]["doc_count"] == 2


def test_no_event_for_noop_ingest(dm, capture_server, monkeypatch):
    """An empty add_documents (nothing stored) does not fire."""
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    dm.add_documents("docs", [])
    assert _Capture.events == []


# ---------------------------------------------------------------------------
# Failure is non-fatal
# ---------------------------------------------------------------------------


def test_receiver_error_is_logged_not_fatal(dm, capture_server, monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, capture_server)
    _Capture.status = 500
    ids = dm.add_documents("docs", ["survives"])
    assert ids, "the ingest itself must succeed"
    assert webhook.notify_ingest("docs", 1) is False


def test_unreachable_receiver_is_non_fatal(dm, monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://127.0.0.1:9/unreachable")
    ids = dm.add_documents("docs", ["survives too"])
    assert ids


def test_any_exception_is_swallowed(monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://example.invalid/hook")

    def _boom(*a, **k):
        raise RuntimeError("connection exploded")

    monkeypatch.setattr(webhook.urllib.request, "urlopen", _boom)
    assert webhook.notify_ingest("ds", 1, "ok") is False


def test_non_http_exception_in_notify_never_raises(monkeypatch):
    """notify_ingest must not raise on ANY receiver behaviour (HTTPError
    subclass is the common one)."""
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://127.0.0.1:1/x")

    def _http_error(*a, **k):
        raise urllib.error.HTTPError("http://127.0.0.1:1/x", 503, "down", {}, None)

    monkeypatch.setattr(webhook.urllib.request, "urlopen", _http_error)
    assert webhook.notify_ingest("ds", 1) is False


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------


def test_timeout_env_parsing_and_clamp(monkeypatch):
    assert webhook.webhook_timeout() == 5.0  # default
    monkeypatch.setenv(webhook.WEBHOOK_TIMEOUT_ENV, "2.5")
    assert webhook.webhook_timeout() == 2.5
    monkeypatch.setenv(webhook.WEBHOOK_TIMEOUT_ENV, "0")
    assert webhook.webhook_timeout() == 0.1  # clamped, never 0
    monkeypatch.setenv(webhook.WEBHOOK_TIMEOUT_ENV, "not-a-number")
    assert webhook.webhook_timeout() == 5.0  # falls back to the default


def test_timeout_is_passed_to_the_request(monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://example.invalid/hook")
    monkeypatch.setenv(webhook.WEBHOOK_TIMEOUT_ENV, "1.25")
    seen = {}

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        seen["headers"] = dict(req.header_items())
        seen["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr(webhook.urllib.request, "urlopen", _fake_urlopen)
    assert webhook.notify_ingest("ds", 2) is True
    assert seen["timeout"] == 1.25


def test_payload_shape_is_small_and_named(monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://example.invalid/hook")
    seen = {}

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        seen["data"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp()

    monkeypatch.setattr(webhook.urllib.request, "urlopen", _fake_urlopen)
    webhook.notify_ingest("reports", 7, "error")
    assert seen["data"]["dataset"] == "reports"
    assert seen["data"]["doc_count"] == 7
    assert seen["data"]["status"] == "error"


# ---------------------------------------------------------------------------
# Muting (restore replay)
# ---------------------------------------------------------------------------


def test_muted_suppresses_events(monkeypatch):
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://example.invalid/hook")
    fired = []

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        fired.append(json.loads(req.data.decode("utf-8")))
        return _FakeResp()

    monkeypatch.setattr(webhook.urllib.request, "urlopen", _fake_urlopen)
    with webhook.muted():
        assert webhook.notify_ingest("restored", 500) is False
    assert fired == []
    assert webhook.notify_ingest("normal", 1) is True
    assert len(fired) == 1


def test_dm_ingest_hook_helper_gates(monkeypatch):
    import multimodal_rag.dataset_manager as dm_mod

    # Phase 1 — URL unset: the real notify_ingest is a no-op and the hook
    # must never raise.
    dm_mod._notify_ingest_hook("ds", 5)

    # Phase 2 — URL set: the hook dispatches (dataset, doc_count, status).
    calls = []
    monkeypatch.setattr(webhook, "notify_ingest", lambda ds, n, s="ok": calls.append((ds, n, s)) or True)
    monkeypatch.setenv(webhook.WEBHOOK_URL_ENV, "http://example.invalid/hook")
    dm_mod._notify_ingest_hook("ds", 5)
    assert calls == [("ds", 5, "ok")]
    dm_mod._notify_ingest_hook("ds", 0, "error")
    assert calls[-1] == ("ds", 0, "error")
    # no-op gate: zero-count "ok" ingests stay silent
    dm_mod._notify_ingest_hook("ds", 0)
    assert len(calls) == 2


def test_stored_points_count_helper():
    from multimodal_rag.dataset_manager import _stored_points_count

    files = [
        {"file": "a", "chunks": 3, "stored_ids": ["1", "2", "3"]},
        {"file": "b", "chunks": 0, "deduplicated": True, "stored_ids": []},
        {"file": "c", "chunks": 2, "error": "boom"},
    ]
    assert _stored_points_count(files) == 3
    assert _stored_points_count(None) == 0
    assert _stored_points_count([{"file": "x", "chunks": 4}]) == 4
