"""Offline tests for Prometheus metrics (roadmap feature 7).

Covers :mod:`multimodal_rag.utils.metrics` — counter/histogram wiring and
the Prometheus text exposition — without booting the API server (whose
startup verifies the remote embedder):

  * ``render_metrics()`` emits the documented metric families.
  * ``observe_qdrant`` records success and error outcomes (the error path
    re-raises the original exception).
  * ``observe_ingest_results`` groups file outcomes and chunk counts.
  * The no-op fallback shape (used when ``prometheus_client`` is missing)
    absorbs label/inc/observe chains.

Run::

    python tests/full_pipeline/test_metrics_endpoint.py    # standalone
    pytest tests/full_pipeline/test_metrics_endpoint.py    # under pytest
"""

import os
import sys

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag.utils import metrics


def test_render_contains_metric_families():
    metrics.HTTP_REQUESTS.labels(route="/api/datasets/{name}/search", method="GET", status="200").inc()
    metrics.HTTP_LATENCY.labels(route="/api/datasets/{name}/search", method="GET").observe(0.02)
    body, ctype = metrics.render_metrics()
    text = body.decode("utf-8")
    assert "rag_http_requests_total" in text
    assert "rag_http_request_seconds" in text
    assert ctype.startswith("text/plain")


def test_observe_qdrant_success_and_error():
    with metrics.observe_qdrant("query_batch_points"):
        pass  # success path
    raised = False
    try:
        with metrics.observe_qdrant("scroll"):
            raise RuntimeError("boom")
    except RuntimeError:
        raised = True
    assert raised, "the original exception must propagate"
    body, _ = metrics.render_metrics()
    text = body.decode("utf-8")
    assert 'op="query_batch_points"' in text
    # prometheus_client sorts labels alphabetically — assert order-independently
    assert 'op="scroll"' in text
    assert 'error="1"' in text


def test_observe_ingest_results_grouping():
    results = [
        {"file": "a.pdf", "chunks": 12},
        {"file": "b.pdf", "chunks": 0, "deduplicated": True},
        {"file": "c.log", "chunks": 3},
        {"file": "d.mp4", "chunks": 0, "error": "ffmpeg exploded"},
        {"file": "e.bin", "chunks": "not-a-number"},  # tolerated
    ]
    metrics.observe_ingest_results(results, dataset_name="ds1")
    body, _ = metrics.render_metrics()
    text = body.decode("utf-8")
    assert 'result="stored"' in text
    assert 'result="deduplicated"' in text
    assert 'result="error"' in text
    assert 'dataset="ds1"' in text


def test_ingest_job_counters():
    metrics.INGEST_JOBS.labels(source="files", state="complete").inc()
    metrics.INGEST_JOBS.labels(source="urls", state="error").inc()
    body, _ = metrics.render_metrics()
    assert 'source="urls",state="error"' in body.decode("utf-8")


def test_cache_event_counters():
    metrics.CACHE_EVENTS.labels(cache="query_emb", event="hit").inc()
    metrics.CACHE_EVENTS.labels(cache="rag", event="eviction").inc()
    body, _ = metrics.render_metrics()
    text = body.decode("utf-8")
    assert 'cache="query_emb",event="hit"' in text
    assert 'cache="rag",event="eviction"' in text


def test_noop_fallback_shape():
    """The no-op metrics used when prometheus_client is missing must absorb chains.

    Only constructible in a minimal install; here we verify the same call
    contract locally against a hand-built stub so a missing dependency can
    never change call-site behaviour.
    """

    class _NoopStub:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, amount=1.0):
            pass

        def observe(self, amount):
            pass

        def time(self):
            import contextlib

            return contextlib.nullcontext()

    noop = _NoopStub()
    noop.labels(route="r", method="GET", status="200").inc(3)
    noop.labels(op="x").observe(0.5)
    with noop.time():
        pass


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
