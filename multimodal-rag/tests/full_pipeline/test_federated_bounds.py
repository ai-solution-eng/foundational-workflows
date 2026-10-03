"""Bounds + deadline for the federated fan-out (perf quick-win #6 / audit P1-3).

Both federated twins used to launch ONE retrieval per dataset, untimed::

    asyncio.gather(*(coro(name) for name in targets), return_exceptions=True)

so a caller naming 50 datasets put 50 retrievals in flight at once, and ONE
hung dataset (Qdrant has no client timeout in the base chart —
``rag_system`` returns ``None`` when ``QDRANT_CLIENT_TIMEOUT`` is unset) pinned
the whole gather.  The fan-out is now bounded and timed, identically on both
sides:

  * concurrency — ``RAG_FEDERATED_CONCURRENCY`` (default 8), an
    ``asyncio.Semaphore`` held for one dataset's retrieval at a time;
  * deadline — ``RAG_FEDERATED_TIMEOUT_SECONDS`` (default 60, ``0`` disables),
    applied per dataset with ``asyncio.wait_for``; on expiry the dataset
    contributes the SAME error note any other per-dataset failure produces
    (``{"dataset": …, "error": "TimeoutError: timed out after Ns"}``) and the
    other datasets still return.

Both knobs are read per call, so the tests flip them with
``monkeypatch.setenv`` and re-run without reloading a module.

No Qdrant and no models: the per-dataset coroutine is stubbed on the MCP side
(``_afederated_one_dataset``), and the REST side — which offloads the sync
``dm.search`` to ``sync_pool`` — runs against a fake pool that settles the
offloaded future on a timer (so "in flight" is observable) or never settles it
at all (the hang), without blocking a real thread.

Run::

    python tests/full_pipeline/test_federated_bounds.py    # standalone
    pytest tests/full_pipeline/test_federated_bounds.py    # under pytest
"""

import asyncio
import concurrent.futures
import os
import sys
import time
from typing import Any, cast

# Ensure the source package shadows any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag import api_server as _api
from multimodal_rag import mcp_server as _mcp

# ---------------------------------------------------------------------------
# Stubs (no Qdrant, no models, no real threads)
# ---------------------------------------------------------------------------


class _RagStub:
    """Smallest object the post-processing step accepts (no postprocessor)."""

    _postprocessor = None


class _DM:
    """Duck-typed DatasetManager — the surface ``resolve_federated_targets``
    and the REST fan-out touch (``get_dataset`` / ``list_datasets`` /
    ``has_password`` / ``search``)."""

    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def get_dataset(self, name: str, sync_count: bool = True) -> dict[str, Any]:
        if name not in self._names:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        return {"name": name, "document_count": 1, "has_password": False}

    def list_datasets(self) -> list[dict[str, Any]]:
        return [{"name": n, "document_count": 1, "has_password": False} for n in self._names]

    def has_password(self, name: str) -> bool:
        return False

    def search(self, name: str, q: str, **kwargs: Any) -> list[dict[str, Any]]:
        # Only reached through the (faked) pool in these tests.
        return [{"content": {"text": f"{name} doc", "source": f"/{name}/f.md"}, "score": 0.5}]


class _Recorder:
    """Async stand-in for ``mcp_server._afederated_one_dataset``.

    Counts how many dataset retrievals are running AT ONCE (the number the
    semaphore is supposed to cap) and honours a per-dataset behaviour:
    ``"ok"`` (returns one hit), ``"slow"`` (a finite sleep) or ``"hang"``
    (sleeps forever — only the per-dataset deadline ends it).
    """

    def __init__(self, behaviour: dict[str, str], slow_s: float = 0.15) -> None:
        self.behaviour = behaviour
        self.slow_s = slow_s
        self.live = 0
        self.max_live = 0
        self.started: list[str] = []
        self.finished: list[str] = []

    def reset(self) -> None:
        self.live = 0
        self.max_live = 0
        self.started = []
        self.finished = []

    async def __call__(self, _dm: Any, name: str, *_a: Any, **_kw: Any) -> tuple[Any, list[Any]]:
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        self.started.append(name)
        try:
            action = self.behaviour.get(name, "ok")
            if action == "hang":
                await asyncio.sleep(3600)  # the deadline (or an outer cancel) ends this
            elif action == "slow":
                await asyncio.sleep(self.slow_s)
            await asyncio.sleep(0)  # yield so overlap is observable
            self.finished.append(name)
            return _RagStub(), [(name, {"text": f"{name} doc", "source": f"/{name}/f.md"}, 0.5)]
        finally:
            self.live -= 1


class _FakePool:
    """``api_server.sync_pool`` stand-in for the REST twin.

    ONLY the per-dataset ``dm.search`` offloads are faked — recognised as the
    ``functools.partial`` whose wrapped callable is named ``search``.  Every
    other offload (notably the target resolution, which must really run to
    produce the ``(targets, skipped, errors)`` triple) is delegated to a local
    thread pool, exactly as in production.

    Instead of running the (trivial) stub search in that thread, the fake
    settles the returned future after *settle_after* seconds — or never, for a
    dataset in *hang* — so "searches in flight" is observable and a hang costs
    no thread.  The dataset name is read off the partial the REST twin submits
    (``partial(dm.search, name, q, ...)``) to build the canned hit list.
    """

    def __init__(self, settle_after: float = 0.05, hang: set[str] | None = None) -> None:
        self.settle_after = settle_after
        self.hang = hang or set()
        self.live = 0
        self.max_live = 0
        self._real = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="test-real-pool")

    def reset(self, *, settle_after: float | None = None, hang: set[str] | None = None) -> None:
        if settle_after is not None:
            self.settle_after = settle_after
        if hang is not None:
            self.hang = hang
        self.live = 0
        self.max_live = 0

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> "concurrent.futures.Future[Any]":
        if getattr(getattr(fn, "func", None), "__name__", "") != "search":
            return self._real.submit(fn, *args, **kwargs)
        loop = asyncio.get_running_loop()
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        name = getattr(fn, "args", ("",))[0]
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        if name in self.hang:
            return fut  # pending forever: only the fan-out deadline ends this
        loop.call_later(self.settle_after, self._settle, fut, name)
        return fut

    def _settle(self, fut: "concurrent.futures.Future[Any]", name: str) -> None:
        self.live -= 1
        fut.set_result([{"content": {"text": f"{name} doc", "source": f"/{name}/f.md"}, "score": 0.5}])


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _names(n: int) -> list[str]:
    return [f"ds{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# (a) the semaphore caps how many datasets are retrieved at once
# ---------------------------------------------------------------------------


def test_mcp_fan_out_concurrency_is_capped(monkeypatch):
    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "3")
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "60")  # every stub is fast
    rec = _Recorder({n: "ok" for n in _names(9)})
    monkeypatch.setattr(_mcp, "_afederated_one_dataset", rec)

    payload = _run(_mcp._afederated_search(_DM(_names(9)), _names(9), "q", top_k=1))

    assert rec.max_live == 3, "fan-out saturates the semaphore but never exceeds it"
    assert rec.max_live <= 3
    assert rec.live == 0, "every slot is released"
    assert payload["errors"] == []
    assert {r["dataset"] for r in payload["results"]} == set(_names(9))


def test_rest_fan_out_concurrency_is_capped(monkeypatch):
    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "3")
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "60")
    pool = _FakePool(settle_after=0.05)
    monkeypatch.setattr(_api, "sync_pool", pool)

    out = _run(_api._federated_rest_search(_DM(_names(9)), _names(9), "q", top_k=1))

    assert pool.max_live == 3, "fan-out saturates the semaphore but never exceeds it"
    assert pool.live == 0, "every slot is released"
    assert out["errors"] == []
    assert {r["dataset"] for r in out["results"]} == set(_names(9))


# ---------------------------------------------------------------------------
# (b) one hung dataset times out; the others still return
# ---------------------------------------------------------------------------


def test_mcp_hung_dataset_times_out_without_failing_the_call(monkeypatch):
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0.05")
    behaviour = {"ds0": "ok", "ds_hang": "hang", "ds2": "ok"}
    rec = _Recorder(behaviour)
    monkeypatch.setattr(_mcp, "_afederated_one_dataset", rec)

    payload = _run(_mcp._afederated_search(_DM(list(behaviour)), list(behaviour), "q", top_k=1))

    # The error note is the SAME shape a plain per-dataset failure produces.
    assert len(payload["errors"]) == 1
    assert set(payload["errors"][0]) == {"dataset", "error"}
    assert payload["errors"][0]["dataset"] == "ds_hang"
    assert payload["errors"][0]["error"] == "TimeoutError: timed out after 0.05s"
    assert "timed out after 0.05s" in payload["context"]

    assert {r["dataset"] for r in payload["results"]} == {"ds0", "ds2"}, "healthy datasets still return"
    assert rec.finished == ["ds0", "ds2"], "the hung retrieval was cancelled, not left running"
    assert rec.live == 0, "cancellation released the semaphore slot"


def test_rest_hung_dataset_times_out_without_failing_the_call(monkeypatch):
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0.05")
    pool = _FakePool(settle_after=0.0, hang={"ds_hang"})
    monkeypatch.setattr(_api, "sync_pool", pool)

    out = _run(_api._federated_rest_search(_DM(["ds0", "ds_hang", "ds2"]), ["ds0", "ds_hang", "ds2"], "q", top_k=1))

    assert len(out["errors"]) == 1
    assert set(out["errors"][0]) == {"dataset", "error"}
    assert out["errors"][0]["dataset"] == "ds_hang"
    assert out["errors"][0]["error"] == "TimeoutError: timed out after 0.05s"
    assert {r["dataset"] for r in out["results"]} == {"ds0", "ds2"}


def test_a_timed_out_slot_is_reusable_by_the_next_waiting_dataset(monkeypatch):
    """The deadline must not leak the semaphore: with concurrency 1, the
    datasets queued BEHIND the hung one still get their turn."""
    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "1")
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0.05")
    behaviour = {"ds_hang": "hang", "ds_a": "ok", "ds_b": "ok"}
    rec = _Recorder(behaviour)
    monkeypatch.setattr(_mcp, "_afederated_one_dataset", rec)

    payload = _run(_mcp._afederated_search(_DM(list(behaviour)), list(behaviour), "q", top_k=1))

    assert rec.max_live == 1, "one at a time"
    assert [e["dataset"] for e in payload["errors"]] == ["ds_hang"]
    assert {r["dataset"] for r in payload["results"]} == {"ds_a", "ds_b"}


# ---------------------------------------------------------------------------
# (c) timeout 0 disables the wrapper
# ---------------------------------------------------------------------------


def test_mcp_zero_timeout_disables_the_deadline(monkeypatch):
    """With ``0`` the retrieval is awaited UNWRAPPED: a slow-but-finite
    retrieval completes, where the same retrieval under a 0.05s deadline
    (the contrast below) is cut off and noted."""
    rec = _Recorder({n: "slow" for n in ("ds0", "ds1")}, slow_s=0.15)
    monkeypatch.setattr(_mcp, "_afederated_one_dataset", rec)

    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0")
    started = time.monotonic()
    payload = _run(_mcp._afederated_search(_DM(["ds0", "ds1"]), ["ds0", "ds1"], "q", top_k=1))
    assert payload["errors"] == [], "0 disables the deadline — no timeout note"
    assert {r["dataset"] for r in payload["results"]} == {"ds0", "ds1"}
    assert time.monotonic() - started >= 0.15, "the slow retrieval really was awaited, not cut short"

    # Contrast: the SAME slow retrieval is bounded once the knob is set.
    rec.reset()
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0.05")
    payload = _run(_mcp._afederated_search(_DM(["ds0", "ds1"]), ["ds0", "ds1"], "q", top_k=1))
    assert [e["error"] for e in payload["errors"]] == ["TimeoutError: timed out after 0.05s"] * 2
    assert payload["results"] == []


def test_rest_zero_timeout_disables_the_deadline(monkeypatch):
    pool = _FakePool(settle_after=0.15)
    monkeypatch.setattr(_api, "sync_pool", pool)

    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0")
    out = _run(_api._federated_rest_search(_DM(["ds0"]), ["ds0"], "q", top_k=1))
    assert out["errors"] == [], "0 disables the deadline — no timeout note"
    assert [r["dataset"] for r in out["results"]] == ["ds0"]

    # Contrast: the same 0.15s search is cut off once the knob is set.
    pool.reset(settle_after=0.15)
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "0.05")
    out = _run(_api._federated_rest_search(_DM(["ds0"]), ["ds0"], "q", top_k=1))
    assert [e["error"] for e in out["errors"]] == ["TimeoutError: timed out after 0.05s"]
    assert out["results"] == []


# ---------------------------------------------------------------------------
# (d) the env knobs are read per call (no restart, no module reload)
# ---------------------------------------------------------------------------


def test_concurrency_knob_is_read_per_call(monkeypatch):
    names = _names(6)
    rec = _Recorder({n: "ok" for n in names})
    monkeypatch.setattr(_mcp, "_afederated_one_dataset", rec)
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "60")
    dm = _DM(names)

    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "1")
    _run(_mcp._afederated_search(dm, names, "q", top_k=1))
    assert rec.max_live == 1

    rec.reset()
    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "6")
    _run(_mcp._afederated_search(dm, names, "q", top_k=1))
    assert rec.max_live == 6

    # The REST twin reads the same knob on its own side.
    pool = _FakePool(settle_after=0.05)
    monkeypatch.setattr(_api, "sync_pool", pool)
    monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "2")
    _run(_api._federated_rest_search(dm, names, "q", top_k=1))
    assert pool.max_live == 2


def test_knob_parsers_default_and_survive_garbage(monkeypatch):
    for mod in (_mcp, _api):
        monkeypatch.delenv("RAG_FEDERATED_CONCURRENCY", raising=False)
        monkeypatch.delenv("RAG_FEDERATED_TIMEOUT_SECONDS", raising=False)
        assert mod._federated_concurrency() == 8, "default concurrency is 8"
        assert mod._federated_timeout_seconds() == 60.0, "default per-dataset deadline is 60s"

        monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "not-a-number")
        monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "")
        assert mod._federated_concurrency() == 8
        assert mod._federated_timeout_seconds() == 60.0

        monkeypatch.setenv("RAG_FEDERATED_CONCURRENCY", "0")  # <= 0 clamps to 1
        monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "-5")  # negative = disabled
        assert mod._federated_concurrency() == 1
        assert mod._federated_timeout_seconds() == 0.0


# ---------------------------------------------------------------------------
# A Qdrant-client timeout inside one dataset stays ITS error, not ours
# ---------------------------------------------------------------------------


def test_inner_timeout_error_is_not_misreported_as_a_fan_out_timeout(monkeypatch):
    """A retrieval that raises ``TimeoutError`` itself (e.g. a Qdrant client
    deadline, once the chart sets one) must surface with its own message —
    the fan-out deadline would otherwise be blamed for it."""
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "60")

    class _InnerTimeout:
        async def __call__(self, _dm: Any, name: str, *_a: Any, **_kw: Any) -> Any:
            if name == "ds_inner":
                raise TimeoutError("qdrant client deadline exceeded")
            return _RagStub(), [(name, {"text": f"{name} doc", "source": f"/{name}/f.md"}, 0.5)]

    monkeypatch.setattr(_mcp, "_afederated_one_dataset", _InnerTimeout())
    payload = _run(_mcp._afederated_search(_DM(["ds0", "ds_inner"]), ["ds0", "ds_inner"], "q", top_k=1))

    assert [e["error"] for e in payload["errors"]] == ["TimeoutError: qdrant client deadline exceeded"]
    assert {r["dataset"] for r in payload["results"]} == {"ds0"}


def test_bounded_call_forwards_args_and_result(monkeypatch):
    """Sanity: the wrapper forwards positional/keyword args and returns the
    wrapped callable's result (both twins call it as ``(sem, fn, *args)``)."""
    monkeypatch.setenv("RAG_FEDERATED_TIMEOUT_SECONDS", "1")

    async def _fn(a: int, b: int = 0) -> int:
        return a + b

    async def _drive() -> int:
        return await _mcp._afederated_bounded_call(asyncio.Semaphore(1), cast(Any, _fn), 2, 3)

    assert _run(_drive()) == 5


if __name__ == "__main__":  # pragma: no cover - standalone convenience
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
