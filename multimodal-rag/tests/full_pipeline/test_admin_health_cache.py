"""Offline regression tests: /api/admin/health Qdrant fan-out TTL cache.

``_collect_health_stats`` used to issue one GET {qdrant}/collections plus
ONE GET {qdrant}/collections/{name} PER COLLECTION on every call, and the
management UI polls the endpoint every 10s - ~13 Qdrant round trips per poll
per open browser at 12 collections.  The fan-out is now served from
``_QDRANT_HEALTH_FANOUT_CACHE`` for ``_QDRANT_HEALTH_FANOUT_TTL`` (30s),
matching the neighbouring telemetry (60s) and cluster (30s) caches.  Failed
fan-outs are never cached, so a dead Qdrant still surfaces "unreachable" on
the very next poll and a cached "ok" masks an outage for at most one TTL.

The cluster fan-out (GET /cluster + GET /collections + one GET
/collections/{name}/cluster per collection) is cached the same way under
``_QDRANT_HEALTH_CLUSTER_CACHE`` with its own 30s TTL — deliberately NOT
routed through ``_qdrant_cluster_replicas``, whose readyz probes and
uri/status per replica belong to the stats endpoint's contract.

Offline throughout: ``httpx2`` is replaced in ``sys.modules`` by a scriptable
counting fake (the handler imports it lazily inside the call, so the fake is
picked up), the dataset manager is a stub, and nothing touches the network
or a real Qdrant.
"""

import asyncio
import os
import sys
import time
import types
from typing import Any, Self

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from multimodal_rag import api_server


class _FakeResponse:
    """Just enough of an httpx response: status code + JSON body."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        """httpx parity; the fake only serves scripted success payloads."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    """Context-manager client routing every GET through the fake Qdrant."""

    def __init__(self, fq: "_FakeQdrant") -> None:
        self._fq = fq

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def get(self, url: str) -> _FakeResponse:
        return self._fq.handle(url)


class _FakeQdrant:
    """Scriptable stand-in for the Qdrant HTTP surface; counts every GET."""

    def __init__(self, n_collections: int = 3) -> None:
        self.calls: list[str] = []
        self.names = [f"col{i}" for i in range(n_collections)]
        # Distinct per-collection point counts: the summed total doubles as a
        # checksum proving every per-collection response really flowed through.
        self.points = {name: (i + 1) * 100 for i, name in enumerate(self.names)}
        self.fail_with: Exception | None = None
        # When set, only URLs ending in this suffix fail (isolates the
        # cluster fan-out from the points fan-out); None fails everything.
        self.fail_suffix: str | None = None
        self.collections_status = 200
        # None -> cluster API reports "disabled" (the default fixture state);
        # a {"<peer id>": {"uri": ...}} map enables the cluster fan-out.
        self.cluster_peers: dict[str, Any] | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Publish self as the ``httpx2`` module the handler lazily imports."""
        fq = self

        def _client(timeout: float | None = None) -> _FakeClient:
            return _FakeClient(fq)

        fake = types.ModuleType("httpx2")
        fake.Client = _client  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx2", fake)

    def handle(self, url: str) -> _FakeResponse:
        self.calls.append(url)  # recorded BEFORE any failure: attempts count
        if self.fail_with is not None and (self.fail_suffix is None or url.endswith(self.fail_suffix)):
            raise self.fail_with
        if url.endswith("/collections"):
            return _FakeResponse(
                self.collections_status,
                {"result": {"collections": [{"name": n} for n in self.names]}},
            )
        if url.endswith("/cluster") and "/collections/" not in url:
            if self.cluster_peers is None:
                # Cluster API disabled -> the handler's cluster section stops
                # after this one call and issues no per-collection requests.
                return _FakeResponse(200, {"result": {"status": "disabled"}})
            return _FakeResponse(200, {"result": {"status": "enabled", "peers": self.cluster_peers}})
        if url.endswith("/cluster"):
            # /collections/{name}/cluster: every collection's local shards sit
            # on the first peer; the first collection also carries one remote
            # shard on the second peer (when one exists).
            ids = sorted(self.cluster_peers or {})
            name = url.split("/collections/")[-1].split("/cluster")[0]
            remote: list[dict[str, Any]] = []
            if name == self.names[0] and len(ids) > 1:
                remote = [{"peer_id": ids[1]}]
            return _FakeResponse(
                200,
                {
                    "result": {
                        "peer_id": ids[0] if ids else "1",
                        "local_shards": [{"shard_id": 1}],
                        "remote_shards": remote,
                    }
                },
            )
        name = url.rsplit("/", 1)[-1]
        return _FakeResponse(200, {"result": {"points_count": self.points.get(name, 0)}})

    def fanout_calls(self) -> list[str]:
        """The GETs the health fan-out issues: /collections + per-collection.

        Excludes the cluster section's /cluster and /collections/{name}/cluster
        requests so the assertions measure exactly the fan-out under test.
        """
        return [
            u for u in self.calls if u.endswith("/collections") or ("/collections/" in u and not u.endswith("/cluster"))
        ]

    def cluster_calls(self) -> list[str]:
        """The cluster fan-out's cluster GETs: the bare /cluster probe plus
        one /collections/{name}/cluster per collection (the shared /collections
        listing is counted by fanout_calls)."""
        return [u for u in self.calls if u.endswith("/cluster")]


class _StubManager:
    """Just ``base_path`` for the disk-usage section of the health helper."""

    def __init__(self, base_path: Any) -> None:
        self.base_path = base_path


@pytest.fixture()
def health_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> _FakeQdrant:
    """Fresh fan-out cache, fake httpx2, stubbed manager, QDRANT_HOST armed."""
    fq = _FakeQdrant()
    fq.install(monkeypatch)
    monkeypatch.setattr(
        api_server,
        "_QDRANT_HEALTH_FANOUT_CACHE",
        {"ts": 0.0, "status": "unknown", "collections": 0, "total_points": 0},
    )
    monkeypatch.setattr(api_server, "_QDRANT_HEALTH_CLUSTER_CACHE", {"ts": 0.0, "info": None})
    # The enabled-cluster path fills the SHARED telemetry cache as a side
    # effect; reset it too so tests stay independent of execution order.
    monkeypatch.setattr(api_server, "_QDRANT_TELEMETRY_CACHE", {"ts": 0.0, "usage": {}})
    monkeypatch.setattr(api_server, "get_manager", lambda: _StubManager(tmp_path))
    monkeypatch.setattr(api_server, "QDRANT_STORAGE_PATH", "")
    monkeypatch.setenv("QDRANT_HOST", "qdrant-fake")  # never contacted: httpx2 is faked
    return fq


def _health() -> dict[str, Any]:
    """Exercise the real endpoint handler end-to-end (offline)."""
    return asyncio.run(api_server.api_health_stats())


def test_two_calls_within_ttl_fan_out_once(health_env: _FakeQdrant) -> None:
    first = _health()
    assert first["qdrant"]["status"] == "ok"
    assert first["qdrant"]["collections"] == 3
    assert first["qdrant"]["total_points"] == 600  # 100 + 200 + 300
    assert len(health_env.fanout_calls()) == 4  # 1 /collections + 3 per-collection

    second = _health()
    assert second["qdrant"] == first["qdrant"]
    assert len(health_env.fanout_calls()) == 4, "second poll within the TTL must serve the cache"

    # And it really is the cached VALUES being served, not a re-read:
    # change the per-collection points under the fake; a poll inside the TTL
    # must still report the numbers captured by the first fan-out.
    health_env.points = {name: 1 for name in health_env.names}
    third = _health()
    assert third["qdrant"]["total_points"] == 600, "within-TTL poll must serve cached values"


def test_fan_out_reruns_after_ttl_expiry(health_env: _FakeQdrant, monkeypatch: pytest.MonkeyPatch) -> None:
    _health()
    assert len(health_env.fanout_calls()) == 4

    monkeypatch.setattr(api_server, "_QDRANT_HEALTH_FANOUT_TTL", 0.05)
    time.sleep(0.15)  # >> the shrunk TTL; ample margin, no clock precision needed
    after = _health()
    assert after["qdrant"]["status"] == "ok"
    assert after["qdrant"]["total_points"] == 600
    assert len(health_env.fanout_calls()) == 8, "expired cache must re-run the fan-out exactly once"


def test_failure_reports_unreachable_and_is_not_cached(health_env: _FakeQdrant) -> None:
    health_env.fail_with = RuntimeError("connection refused (simulated)")

    first = _health()
    assert first["qdrant"]["status"] == "unreachable"
    assert first["qdrant"]["collections"] == 0
    assert first["qdrant"]["total_points"] == 0
    assert len(health_env.fanout_calls()) == 1, "the failing GET /collections attempt must be observed"

    second = _health()
    assert second["qdrant"]["status"] == "unreachable"
    assert len(health_env.fanout_calls()) == 2, "a failed fan-out must NOT be cached: the next poll re-attempts"


def test_cached_ok_masks_outage_at_most_one_ttl(health_env: _FakeQdrant, monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = _health()
    assert healthy["qdrant"]["status"] == "ok"
    assert len(health_env.fanout_calls()) == 4

    # Qdrant dies right after a successful fan-out: while the cache is fresh
    # the poll still reports the cached "ok" (the accepted staleness bound).
    health_env.fail_with = RuntimeError("connection refused (simulated)")
    masked = _health()
    assert masked["qdrant"]["status"] == "ok"
    assert len(health_env.fanout_calls()) == 4

    # ...but never for longer than the TTL: the first poll after expiry
    # re-runs the fan-out, sees the dead Qdrant and reports "unreachable".
    monkeypatch.setattr(api_server, "_QDRANT_HEALTH_FANOUT_TTL", 0.05)
    time.sleep(0.15)
    dead = _health()
    assert dead["qdrant"]["status"] == "unreachable"
    assert len(health_env.fanout_calls()) == 5


def test_response_contract_keys_and_types_unchanged(health_env: _FakeQdrant) -> None:
    body = _health()
    assert set(body) == {"cpu", "memory", "file_pvc", "qdrant", "models"}
    assert set(body["cpu"]) == {"percent", "cores", "limit_millicores"}
    assert set(body["memory"]) == {"rss_bytes", "limit_bytes", "percent"}
    assert set(body["file_pvc"]) == {"total_bytes", "used_bytes", "free_bytes", "used_percent"}
    assert set(body["qdrant"]) == {"status", "collections", "total_points", "pvc", "cluster"}
    assert set(body["qdrant"]["cluster"]) == {
        "enabled",
        "replicas",
        "total_shards",
        "per_replica_capacity_bytes",
        "expected_replicas",
    }
    assert set(body["models"]) == {"embedder"}
    assert set(body["models"]["embedder"]) == {"status", "last_check", "error"}

    q = body["qdrant"]
    assert isinstance(q["status"], str)
    assert isinstance(q["collections"], int) and not isinstance(q["collections"], bool)
    assert isinstance(q["total_points"], int) and not isinstance(q["total_points"], bool)
    assert q["pvc"] is None  # QDRANT_STORAGE_PATH cleared: same "absent" shape as before
    assert isinstance(q["cluster"], dict)  # the cluster section still runs its disabled path
    assert q["cluster"]["enabled"] is False


def test_cluster_fan_out_cached_within_ttl_and_reruns_after_expiry(
    health_env: _FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    health_env.cluster_peers = {
        "1": {"uri": "http://qdrant-0:9200"},
        "2": {"uri": "http://qdrant-1:9200"},
    }
    body = _health()
    cluster = body["qdrant"]["cluster"]
    assert cluster["enabled"] is True
    assert cluster["total_shards"] == 4  # 3 local + 1 remote
    assert [r["host"] for r in cluster["replicas"]] == ["qdrant-0", "qdrant-1"]
    assert [r["index"] for r in cluster["replicas"]] == [0, 1]
    assert cluster["replicas"][0]["status"] == "unknown"  # default: no readyz on this path
    assert len(health_env.cluster_calls()) == 4  # 1 /cluster + 3 per-collection

    _health()  # second poll within the TTL
    assert len(health_env.cluster_calls()) == 4, "second poll within the TTL must serve the cached cluster info"

    monkeypatch.setattr(api_server, "_QDRANT_HEALTH_CLUSTER_TTL", 0.05)
    time.sleep(0.15)
    _health()
    assert len(health_env.cluster_calls()) == 8, "expired cache must re-run the cluster fan-out exactly once"


def test_cluster_failure_not_cached_and_surfaces_as_null(health_env: _FakeQdrant) -> None:
    # Fail ONLY the cluster calls: the points fan-out must keep working and
    # the cluster payload must surface as None exactly as the old inline
    # except-swallow did — while the failed pass itself is retried every poll.
    health_env.fail_with = RuntimeError("connection refused (simulated)")
    health_env.fail_suffix = "/cluster"

    first = _health()
    assert first["qdrant"]["status"] == "ok"  # points fan-out unaffected by the cluster outage
    assert first["qdrant"]["cluster"] is None
    assert len(health_env.cluster_calls()) == 1

    second = _health()
    assert second["qdrant"]["cluster"] is None
    assert len(health_env.cluster_calls()) == 2, "a failed cluster pass must NOT be cached: the next poll re-attempts"
