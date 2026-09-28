"""Offline render tests for EzUA app-catalog pod labels (all three charts).

The EzUA app catalog derives a card's status from LABELED PODS: an app whose
pods carry no hpe-ezua labels shows "Unknown" in the catalog. This chart was
the only unlabeled-pod app on the G2 cluster and the only Unknown card
(2026-10) — its Service/Deployment objects were labeled (cluster policy + the
embed-batcher Service fix) but never the pod templates.

Contract pinned here:

  * every LONG-RUNNING workload's pod template (API Deployment, Qdrant
    StatefulSet, Redis + embed-batcher Deployments on the scale charts)
    carries hpe-ezua/type: vendor-service + hpe-ezua/app: <chart name>;
  * NO pod spec.selector gains the vendor keys (a selector is immutable —
    a change there would break every subsequent helm upgrade);
  * NO CronJob pod template carries vendor labels: the platform's
    assign-custom-scheduler-for-ezua-user-vendor-pods policy stamps
    schedulerName on labeled pods at EVERY admission, and pod updates
    (Job finalizer removal) then reject the immutable-field change — the
    G2 2026-09-24 undeletable-cron-pod incident (see
    test_kyverno_vendor_labels_render.py for the Deployment/Service-only
    policy-match pin).

Skipped when the helm binary is unavailable (suite skip conventions).

Run::

    pytest tests/full_pipeline/test_vendor_pod_labels_render.py -q
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]

# workload kind + name suffix per chart variant
def _workloads(chart: str) -> list[tuple[str, str]]:
    v = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    dep = v["deployment"]["name"]
    out = [("Deployment", dep), ("StatefulSet", f"{dep}-qdrant")]
    if chart != "helm":
        out.append(("Deployment", f"{dep}-redis"))
    if chart == "helm-scale-large":
        out.append(("Deployment", f"{dep}-embed-batcher"))
    return out


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render_ok(chart: str, *set_args: str) -> str:
    proc = subprocess.run(
        ["helm", "template", "render-test", str(REPO / chart), *set_args],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"helm template failed for {chart}:\n{proc.stderr}"
    return proc.stdout


def _docs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d]


def _chart_name(chart: str) -> str:
    return yaml.safe_load((REPO / chart / "Chart.yaml").read_text())["name"]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
class TestVendorPodLabels:
    def test_long_running_pods_carry_vendor_labels(self, chart: str) -> None:
        """Catalog status comes from labeled pods — every long-running
        workload's pod template must carry the vendor labels."""
        docs = _docs(_render_ok(chart))
        for kind, name in _workloads(chart):
            obj = next(
                (d for d in docs if d.get("kind") == kind and d["metadata"]["name"] == name),
                None,
            )
            assert obj is not None, f"{chart}: {kind}/{name} missing from render"
            labels = obj["spec"]["template"]["metadata"].get("labels", {})
            assert labels.get("hpe-ezua/type") == "vendor-service", (kind, name, labels)
            assert labels.get("hpe-ezua/app") == _chart_name(chart), (kind, name, labels)

    def test_selectors_never_carry_vendor_labels(self, chart: str) -> None:
        """HARD BAR: selector.matchLabels is immutable — adding the vendor
        keys there would break every subsequent helm upgrade."""
        docs = _docs(_render_ok(chart))
        for d in docs:
            if d.get("kind") in ("Deployment", "StatefulSet"):
                sel = d["spec"]["selector"]["matchLabels"]
                assert "hpe-ezua/type" not in sel and "hpe-ezua/app" not in sel, d["metadata"]["name"]

    def test_cron_pods_stay_unlabeled(self, chart: str) -> None:
        """HARD BAR: labeled cron pods become undeletable on pod update
        (platform scheduler-stamping vendor policy; G2 incident 2026-09-24)."""
        rendered = _render_ok(
            chart,
            "--set", "watchedSources.enabled=true",
            "--set", "watchedSources.sources[0].dataset=reports",
            "--set", "watchedSources.sources[0].prefixes[0]=s3://bucket/reports/",
            "--set", "backups.enabled=true",
            "--set", "backups.bucket=s3://backup-bucket",
        )
        for cron in [d for d in _docs(rendered) if d.get("kind") == "CronJob"]:
            labels = (
                cron["spec"]["jobTemplate"]["spec"]["template"]["metadata"].get("labels", {})
            )
            assert "hpe-ezua/type" not in labels and "hpe-ezua/app" not in labels, cron["metadata"]["name"]

    def test_pods_keep_their_app_label(self, chart: str) -> None:
        """The selector-keyed app label is untouched — Service routing and
        the anti-affinity rule keep working unchanged."""
        docs = _docs(_render_ok(chart))
        v = yaml.safe_load((REPO / chart / "values.yaml").read_text())
        dep = next(d for d in docs if d.get("kind") == "Deployment" and d["metadata"]["name"] == v["deployment"]["name"])
        labels = dep["spec"]["template"]["metadata"]["labels"]
        assert labels.get("app") == v["deployment"]["appName"]
        assert dep["spec"]["selector"]["matchLabels"]["app"] == v["deployment"]["appName"]
