"""Offline render tests for the vendor-label ClusterPolicy (templates/kyverno.yaml).

The policy stamps EzUA console-discovery labels (hpe-ezua/type: vendor-service,
hpe-ezua/app: <chart>) on resources in the release namespace.  It must match
**Deployment and Service only — never Pod**:

  * EzUA app discovery reads Deployment/Service objects; Pod objects don't
    need the labels.
  * The G2/EzAF platform ships `assign-custom-scheduler-for-ezua-user-vendor-pods`
    (Kyverno, admission webhooks, failurePolicy=Fail) which selects pods by
    hpe-ezua/type ∈ {app-service-user, vendor-service} and stamps
    spec.schedulerName = scheduler-plugins-scheduler on EVERY admission —
    CREATE and UPDATE.  A labelled pod created before that stamping (or with
    default-scheduler) becomes UNDELETABLE: every update — the Job
    controller's job-tracking-finalizer removal, `kubectl delete pod`'s
    deletionTimestamp write — gets schedulerName rewritten in the request and
    pod-update validation rejects the immutable-field change
    ("pod updates may not change fields other than ...").  Confirmed live on
    G2 2026-09-24: mm-rag watched-sources cron pods orphaned finalizers, the
    Job never reached Complete, and concurrencyPolicy: Forbid suppressed 145
    cron ticks until the stale Job was deleted by hand.
  * Therefore the policy matching Pod kind is a hard regression: these tests
    pin the match to Deployment/Service in all three chart variants, and pin
    that the labels ARE still applied to the Deployment (the console-discovery
    purpose is preserved).

Run::

    python tests/full_pipeline/test_kyverno_vendor_labels_render.py   # standalone
    pytest tests/full_pipeline/test_kyverno_vendor_labels_render.py   # under pytest
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render_policy(chart: str) -> str:
    proc = subprocess.run(
        ["helm", "template", "render-test", str(REPO / chart), "--show-only", "templates/kyverno.yaml"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout if proc.returncode == 0 else f"__ERROR__\n{proc.stderr}"


@pytest.mark.skipif(not _helm_available(), reason="helm binary unavailable")
@pytest.mark.parametrize("chart", CHARTS)
class TestVendorLabelPolicyMatchKinds:
    def test_policy_renders(self, chart: str) -> None:
        out = _render_policy(chart)
        assert not out.startswith("__ERROR__"), out
        docs = [d for d in yaml.safe_load_all(out) if d]
        assert len(docs) == 1
        assert docs[0]["kind"] == "ClusterPolicy"
        assert docs[0]["metadata"]["name"].startswith("add-vendor-app-labels-")

    def test_match_kinds_exclude_pod(self, chart: str) -> None:
        """HARD BAR: the policy must never match Pod kind again.

        Matching Pod feeds the platform's scheduler-stamping vendor policy and
        makes any unstamped pod undeletable (see module docstring for the full
        G2 incident chain).
        """
        out = _render_policy(chart)
        assert not out.startswith("__ERROR__"), out
        doc = next(yaml.safe_load_all(out))
        for rule in doc["spec"]["rules"]:
            for entry in rule["match"]["any"]:
                assert "Pod" not in entry["resources"]["kinds"], (
                    f"{chart}: kyverno vendor-label policy matches Pod kind — "
                    "this recreates the undeletable-pod incident (G2 2026-09-24)"
                )

    def test_match_kinds_cover_deployment_and_service(self, chart: str) -> None:
        """The EzUA console-discovery purpose is preserved: labels land on
        Deployment + Service (and nothing else)."""
        out = _render_policy(chart)
        assert not out.startswith("__ERROR__"), out
        doc = next(yaml.safe_load_all(out))
        for rule in doc["spec"]["rules"]:
            for entry in rule["match"]["any"]:
                assert sorted(entry["resources"]["kinds"]) == ["Deployment", "Service"]

    def test_labels_target_metadata_not_spec(self, chart: str) -> None:
        out = _render_policy(chart)
        assert not out.startswith("__ERROR__"), out
        doc = next(yaml.safe_load_all(out))
        for rule in doc["spec"]["rules"]:
            patch = rule["mutate"]["patchStrategicMerge"]
            assert patch["metadata"]["labels"]["hpe-ezua/type"] == "vendor-service"
            # the app label tracks the chart's own name (helm .Chart.Name)
            assert patch["metadata"]["labels"]["hpe-ezua/app"]  # non-empty


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
