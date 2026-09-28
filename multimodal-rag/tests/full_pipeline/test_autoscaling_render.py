"""Offline render tests for the opt-in API HPA (scale charts) + request defaults.

Two contracts are pinned here:

  1. HPA (templates/hpa.yaml, scale charts only): with ``autoscaling.enabled``
     the API Deployment gets an autoscaling/v2 HorizontalPodAutoscaler and the
     Deployment renders NO ``spec.replicas`` — the HPA owns the count, so a
     subsequent ``helm upgrade`` cannot reset it (the classic helm/HPA fight).
     The default render ships no HPA at all and pins the historical
     ``spec.replicas`` values (base 1 / medium 2 / large 4), so today's
     behavior is byte-path unchanged. The Qdrant StatefulSet and the
     embed-batcher Deployment keep their own replica counts in both modes —
     only the API Deployment ever autoscales.
  2. HPA-calibrated requests (values defaults, scale charts): the HPA's CPU
     target reads utilization against the REQUEST, so the app request is the
     scaling sensitivity dial — app containers request 4Gi/2-cpu (large) /
     4Gi/1.5-cpu (medium), putting the 70% target at ~1.4 / ~1.05 cores of
     sustained per-container CPU; limits unchanged (8Gi and 3/4 cpu). The
     500m/1Gi burst-to-limit attempt read 474%+ utilization under the
     benchmarked load and slammed the HPA to maxReplicas. Qdrant replicas
     stay burst-to-limit (4Gi/1 cpu; limits 20/32Gi and 6/8 cpu unchanged) —
     deliberately NOT the HPA signal.

Skipped when the helm binary is unavailable (suite skip conventions).

Run::

    pytest tests/full_pipeline/test_autoscaling_render.py -q
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCALE_CHARTS = ["helm-scale-medium", "helm-scale-large"]
ALL_CHARTS = ["helm"] + SCALE_CHARTS


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render_ok(chart: str, *set_args: str) -> str:
    cmd = ["helm", "template", "render-test", str(REPO / chart)]
    cmd += list(set_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"helm template failed for {chart}:\\n{proc.stderr}"
    return proc.stdout


def _docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


def _doc(docs: list[dict], kind: str, name: str) -> dict:
    for d in docs:
        if d.get("kind") == kind and d.get("metadata", {}).get("name") == name:
            return d
    raise AssertionError(f"{kind}/{name} not found in render")


def _app_deployment(docs: list[dict]) -> dict:
    return _doc(docs, "Deployment", "rag-mcp-server")


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
class TestHpaDefaultRender:
    """Default (autoscaling off): no HPA, replicas rendered, pins preserved."""

    @pytest.mark.parametrize("chart", ALL_CHARTS)
    def test_no_hpa_by_default(self, chart: str) -> None:
        assert not [d for d in _docs(_render_ok(chart)) if d.get("kind") == "HorizontalPodAutoscaler"]

    @pytest.mark.parametrize(
        "chart,replicas", [("helm", 1), ("helm-scale-medium", 2), ("helm-scale-large", 4)]
    )
    def test_pinned_default_replica_count(self, chart: str, replicas: int) -> None:
        dep = _app_deployment(_docs(_render_ok(chart)))
        assert dep["spec"]["replicas"] == replicas

    # App requests are the HPA's sensitivity dial: utilization is computed
    # against the request, so the request sets where the 70% target fires.
    # cpu quantities render as YAML int or str depending on quoting in
    # values.yaml — normalize both sides through str() for the comparison.
    @pytest.mark.parametrize(
        "chart,app_mem,app_cpu",
        [("helm-scale-medium", "4Gi", "1.5"), ("helm-scale-large", "4Gi", "2")],
    )
    def test_default_app_requests_hpa_calibrated(
        self, chart: str, app_mem: str, app_cpu: str
    ) -> None:
        dep = _app_deployment(_docs(_render_ok(chart)))
        for c in dep["spec"]["template"]["spec"]["containers"]:
            res = c["resources"]
            assert {k: str(v) for k, v in res["requests"].items()} == {
                "memory": app_mem, "cpu": app_cpu
            }, res["requests"]
            assert res["limits"]["memory"] == "8Gi"

    # cpu quantities render as YAML int or str depending on quoting in
    # values.yaml — normalize both sides through str() for the comparison.
    @pytest.mark.parametrize("chart,qdrant_max_mem,qdrant_max_cpu", [
        ("helm-scale-medium", "20Gi", "6"),
        ("helm-scale-large", "32Gi", "8"),
    ])
    def test_default_qdrant_requests_burst_to_limit(
        self, chart: str, qdrant_max_mem: str, qdrant_max_cpu: str
    ) -> None:
        sts = _doc(_docs(_render_ok(chart)), "StatefulSet", "rag-mcp-server-qdrant")
        res = sts["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert {k: str(v) for k, v in res["requests"].items()} == {"memory": "4Gi", "cpu": "1"}, res["requests"]
        limits = {k: str(v) for k, v in res["limits"].items()}
        assert limits == {"memory": qdrant_max_mem, "cpu": qdrant_max_cpu}, res["limits"]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", SCALE_CHARTS)
class TestHpaEnabledRender:
    """autoscaling.enabled=true: the HPA owns the replica count."""

    def _render(self, chart: str, *extra: str) -> list[dict]:
        return _docs(_render_ok(chart, "--set", "autoscaling.enabled=true", *extra))

    def test_hpa_renders(self, chart: str) -> None:
        hpa = _doc(self._render(chart), "HorizontalPodAutoscaler", "rag-mcp-server")
        assert hpa["apiVersion"] == "autoscaling/v2"
        assert hpa["spec"]["scaleTargetRef"] == {
            "apiVersion": "apps/v1", "kind": "Deployment", "name": "rag-mcp-server"
        }

    def test_deployment_omits_replicas(self, chart: str) -> None:
        """HARD BAR: no spec.replicas while the HPA owns the count — a helm
        upgrade that re-rendered a fixed replica count would fight the HPA."""
        dep = _app_deployment(self._render(chart))
        assert "replicas" not in dep["spec"]
        # the rest of the Deployment must be intact (indentation regressions
        # in the guard would show up here)
        assert dep["spec"]["selector"]["matchLabels"]["app"] == "rag-mcp-server"
        assert dep["spec"]["template"]["spec"]["containers"]

    def test_hpa_defaults(self, chart: str) -> None:
        hpa = _doc(self._render(chart), "HorizontalPodAutoscaler", "rag-mcp-server")
        spec = hpa["spec"]
        assert spec["minReplicas"] == 2  # HA floor: 1 replica is not HA
        assert spec["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
        assert (
            spec["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == 300
        ), "bursty RAG traffic needs a scale-down stabilization window"

    def test_hpa_values_overrides(self, chart: str) -> None:
        hpa = _doc(
            self._render(
                chart,
                "--set", "autoscaling.minReplicas=3",
                "--set", "autoscaling.maxReplicas=10",
                "--set", "autoscaling.targetCPUUtilizationPercentage=80",
            ),
            "HorizontalPodAutoscaler", "rag-mcp-server",
        )
        spec = hpa["spec"]
        assert spec["minReplicas"] == 3
        assert spec["maxReplicas"] == 10
        assert spec["metrics"][0]["resource"]["target"]["averageUtilization"] == 80

    def test_only_the_api_deployment_scales(self, chart: str) -> None:
        """Qdrant StatefulSet + embed-batcher keep their own replica counts."""
        docs = self._render(chart)
        sts = _doc(docs, "StatefulSet", "rag-mcp-server-qdrant")
        assert sts["spec"]["replicas"] == (2 if chart.endswith("medium") else 3)
        if chart.endswith("large"):
            batcher = _doc(docs, "Deployment", "rag-mcp-server-embed-batcher")
            assert batcher["spec"]["replicas"] == 1  # singleton, never scaled
