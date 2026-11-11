"""Render-regression tests for the 2026-11-10 zone-W3 remediation (helm charts).

Runs the real `helm template` binary against the three chart variants and pins
every chart-side fix from documentation/AUDIT-2026-11-10.md zone W3:

  P2-8  the backup CronJob carries the wait-for-qdrant readiness gate (the
        737c686 fix previously reached only watched-sources + stm-reaper);
  P2-9  backup AND watched-sources ticks exit non-zero on ANY failure (the
        old "partial success is success" exit contract is gone);
  P2-10 backup + stm-reaper keep the failed pod readable (restartPolicy Never
        + ttlSecondsAfterFinished), matching watched-sources;
  P2-12 a PodDisruptionBudget per workload (API / qdrant / Redis) with
        maxUnavailable 1, opt-out via podDisruptionBudget.enabled=false;
  P2-13 security.requirePinnedKeys fails an unpinned OFFLINE render loudly
        (default false keeps CI renders working);
  P2-14 extraEnv reaches every CronJob, not only the API/MCP containers;
  P1-2c RAG_PREFETCH_FLOOR renders from rag.prefetchFloor (default 200);
        EMBED_BATCH_CACHE_MAX renders in the large chart (embed-batcher
        singleton is the only consumer);
  P3-7  optional image.digest / qdrant.image.digest / redis.image.digest
        render repository:tag@sha256:... when set;
  P3-8  qdrant containers run readOnlyRootFilesystem with a /tmp emptyDir.

Skipped when the helm binary is unavailable (offline/CI fallback), matching
the suite's skip conventions.

Run::

    pytest tests/full_pipeline/test_zone_c_render_remediation.py -q
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(str(REPO), "src"))

CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]
SCALE_CHARTS = ["helm-scale-medium", "helm-scale-large"]

WATCHED_ENABLE_ARGS = (
    "--set",
    "watchedSources.enabled=true",
    "--set",
    "watchedSources.sources[0].dataset=reports",
    "--set",
    "watchedSources.sources[0].prefixes[0]=s3://mm-rag-drop/reports/",
)
BACKUP_ENABLE_ARGS = ("--set", "backups.enabled=true", "--set", "backups.bucket=backup-bkt")
REAPER_ENABLE_ARGS = (
    "--set",
    "shortTermMemory.enabled=true",
    "--set",
    "shortTermMemory.nameSalt=test-salt",
)


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render(chart: str, *set_args: str) -> str:
    cmd = ["helm", "template", "render-test", str(REPO / chart)] + list(set_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode == 0:
        return proc.stdout
    return "__ERROR__: " + proc.stderr


def _render_ok(chart: str, *set_args: str) -> str:
    out = _render(chart, *set_args)
    assert not out.startswith("__ERROR__"), out[:2000]
    return out


def _render_fail(chart: str, *set_args: str) -> str:
    out = _render(chart, *set_args)
    assert out.startswith("__ERROR__"), "expected the render to FAIL, got: " + out[:800]
    return out


def _docs(rendered: str) -> Iterator[dict[str, Any]]:
    return (d for d in yaml.safe_load_all(rendered) if d)


def _cron(rendered: str, name_part: str) -> dict[str, Any]:
    crons = [d for d in _docs(rendered) if d.get("kind") == "CronJob" and name_part in d["metadata"]["name"]]
    assert len(crons) == 1, f"expected exactly one {name_part} CronJob, got {len(crons)}"
    return crons[0]


def _deployment(rendered: str, name: str) -> dict[str, Any]:
    deps = [d for d in _docs(rendered) if d.get("kind") == "Deployment" and d["metadata"]["name"] == name]
    assert len(deps) == 1, f"expected exactly one {name} Deployment, got {len(deps)}"
    return deps[0]


def _sts(rendered: str, name_part: str) -> dict[str, Any]:
    sts = [d for d in _docs(rendered) if d.get("kind") == "StatefulSet" and name_part in d["metadata"]["name"]]
    assert len(sts) == 1, f"expected exactly one {name_part} StatefulSet, got {len(sts)}"
    return sts[0]


def _pdbs(rendered: str) -> list[dict[str, Any]]:
    return [d for d in _docs(rendered) if d.get("kind") == "PodDisruptionBudget"]


def _pod_spec(workload: dict[str, Any]) -> dict[str, Any]:
    return (
        workload["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        if workload["kind"] == "CronJob"
        else workload["spec"]["template"]["spec"]
    )


def _container(workload: dict[str, Any]) -> dict[str, Any]:
    return _pod_spec(workload)["containers"][0]


def _env_map(container: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in container.get("env", []):
        if "value" in entry:
            out[entry["name"]] = entry["value"]
    return out


def _configmap(rendered: str) -> dict[str, Any]:
    cms = [d for d in _docs(rendered) if d.get("kind") == "ConfigMap"]
    assert len(cms) == 1, f"expected exactly one ConfigMap, got {len(cms)}"
    return cms[0]


# ---------------------------------------------------------------------------
# P2-8 — the backup cron must not touch a half-joined qdrant cluster
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_backup_cron_has_wait_for_qdrant_gate(chart: str) -> None:
    """P2-8: the gate 737c686 gave watched-sources/stm-reaper must exist on
    the backup cron too — its exports ARE cluster reads."""
    cron = _cron(_render_ok(chart, *BACKUP_ENABLE_ARGS), "backups")
    pod_spec = _pod_spec(cron)
    inits = [c["name"] for c in pod_spec.get("initContainers", [])]
    assert "wait-for-qdrant" in inits, f"missing readiness gate, initContainers={inits}"
    gate = next(c for c in pod_spec["initContainers"] if c["name"] == "wait-for-qdrant")
    script = gate["command"][2]
    assert "/readyz" in script, "the gate must poll /readyz, not /healthz"
    assert "deadline = time.time() + budget" in script, "the gate must be time-bounded"
    compile(script, "backup-gate.py", "exec")


# ---------------------------------------------------------------------------
# P2-9 — partial failure is failure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_backup_cron_exits_nonzero_on_any_failure(chart: str) -> None:
    cron = _cron(_render_ok(chart, *BACKUP_ENABLE_ARGS), "backups")
    script = _container(cron)["command"][2]
    assert "raise SystemExit(1 if failed else 0)" in script, script[-400:]
    assert "failed and not exported" not in script, "the old partial-success exit contract is back"
    compile(script, "backup-tick.py", "exec")


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_watched_cron_exits_nonzero_on_any_failure(chart: str) -> None:
    cron = _cron(_render_ok(chart, *WATCHED_ENABLE_ARGS), "watched-sources")
    script = _container(cron)["command"][2]
    assert "raise SystemExit(1 if failed else 0)" in script, script[-400:]
    assert "failed and not ok" not in script, "the old partial-success exit contract is back"
    compile(script, "watched-tick.py", "exec")


# ---------------------------------------------------------------------------
# P2-10 — a failed tick must leave its logs readable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_backup_cron_keeps_failed_pod_readable(chart: str) -> None:
    cron = _cron(_render_ok(chart, *BACKUP_ENABLE_ARGS), "backups")
    job_spec = cron["spec"]["jobTemplate"]["spec"]
    assert _pod_spec(cron)["restartPolicy"] == "Never", "OnFailure deletes the failed pod (log loss)"
    assert job_spec["ttlSecondsAfterFinished"] == 3600
    assert job_spec["backoffLimit"] == 1


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_stm_reaper_keeps_failed_pod_readable() -> None:
    cron = _cron(_render_ok("helm-scale-large", *REAPER_ENABLE_ARGS), "stm-reaper")
    job_spec = cron["spec"]["jobTemplate"]["spec"]
    assert _pod_spec(cron)["restartPolicy"] == "Never", "OnFailure deletes the failed pod (log loss)"
    assert job_spec["ttlSecondsAfterFinished"] == 3600


# ---------------------------------------------------------------------------
# P2-12 — PodDisruptionBudget trio
# ---------------------------------------------------------------------------


def _pdb_map(rendered: str) -> dict[str, dict[str, Any]]:
    return {d["metadata"]["name"]: d for d in _pdbs(rendered)}


def _effective_api_replicas(values: dict[str, Any]) -> int:
    if values.get("autoscaling", {}).get("enabled"):
        return int(values["autoscaling"]["minReplicas"])
    return int(values["replicaCount"])


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_pdb_follows_the_replica_topology(chart: str) -> None:
    """M2/M3 (wave-2 review): a PDB below 2 replicas is inert AND falsely
    reassuring (disruptionsAllowed = healthy - (replicas - maxUnavailable)
    >= 1 for a single pod), so none renders.  A 2-peer qdrant raft loses
    quorum on ANY eviction -> maxUnavailable FORCED to 0; at >= 3 peers the
    value renders (clamped to peers - 2).  Enforced against each chart's OWN
    values, not hard-coded names."""
    values = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    pdbs = _pdb_map(_render_ok(chart))
    api_replicas = _effective_api_replicas(values)
    qdrant_replicas = int(values["qdrant"].get("replicas", 1))
    if api_replicas >= 2:
        api = pdbs.get("rag-mcp-server-api")
        assert api is not None, f"{chart}: API has {api_replicas} replicas but no PDB"
        assert api["spec"]["maxUnavailable"] == min(
            int(values["podDisruptionBudget"]["api"]["maxUnavailable"]), api_replicas - 1
        )
    else:
        assert not [k for k in pdbs if k.endswith("-api")], (
            f"{chart}: API has {api_replicas} replica(s) — a PDB here is falsely reassuring"
        )
    if qdrant_replicas >= 2:
        qdr = pdbs.get("rag-mcp-server-qdrant")
        assert qdr is not None, f"{chart}: qdrant has {qdrant_replicas} peers but no PDB"
        expected_mu = (
            0
            if qdrant_replicas < 3
            else min(int(values["podDisruptionBudget"]["qdrant"]["maxUnavailable"]), qdrant_replicas - 2)
        )
        assert qdr["spec"]["maxUnavailable"] == expected_mu, (
            f"{chart}: {qdrant_replicas} peers must give maxUnavailable {expected_mu}"
        )
        # M5: a NOT-Ready peer (20-min cold-join budget) must never block a drain.
        assert qdr["spec"].get("unhealthyPodEvictionPolicy") == "AlwaysAllow"
    else:
        assert not [k for k in pdbs if k.endswith("-qdrant")], (
            f"{chart}: qdrant has {qdrant_replicas} node(s) — a PDB here is falsely reassuring"
        )


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_pdb_never_covers_the_single_replica_redis(chart: str) -> None:
    """M2: the Redis Deployment is hard-wired to 1 replica — a PDB there is
    inert and falsely reassuring, so none must render in any chart."""
    assert not [k for k in _pdb_map(_render_ok(chart)) if k.endswith("-redis")]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_api_pdb_selector_excludes_cron_pods(chart: str) -> None:
    """M4: the bare app=<appName> label also matches every CronJob pod; a
    cron waiting out its qdrant readiness gate (up to 30 min) would drop
    disruptionsAllowed to 0 and block kubectl drain.  The selector must carry
    the dedicated component label the cron pods do NOT have — proven by
    rendering both the selector and every workload's pod labels."""
    values = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    app_name = values["deployment"]["name"]
    rendered = _render_ok(chart, "--set", "backups.enabled=true", "--set", "backups.bucket=b")
    pdbs = _pdb_map(rendered)
    api = pdbs.get(f"{app_name}-api")
    if api is None:
        pytest.skip(f"{chart}: no API PDB at this replica count (M2 rule)")
    selector = api["spec"]["selector"]["matchLabels"]
    assert selector.get("app.kubernetes.io/component") == "api", selector
    for doc in _docs(rendered):
        if doc.get("kind") not in ("Deployment", "StatefulSet", "CronJob"):
            continue
        if doc["kind"] == "CronJob":
            pod_meta = doc["spec"]["jobTemplate"]["spec"]["template"]["metadata"]
        else:
            pod_meta = doc["spec"]["template"]["metadata"]
        labels = pod_meta.get("labels", {})
        if doc["metadata"]["name"] == app_name:
            assert all(labels.get(k) == v for k, v in selector.items()), "the API pod must match its own PDB selector"
        else:
            assert not all(labels.get(k) == v for k, v in selector.items()), (
                f"{doc['kind']} {doc['metadata']['name']} wrongly matches the API PDB selector "
                f"{selector}: labels={labels}"
            )


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_pdb_opt_out_renders_none(chart: str) -> None:
    assert _pdbs(_render_ok(chart, "--set", "podDisruptionBudget.enabled=false")) == []


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_pdb_survives_a_null_set(chart: str) -> None:
    """M7: `--reuse-values` on the first upgrade renders
    podDisruptionBudget=null (ROTATION.md) — a non-nil-guarded template would
    fail the render with a nil-pointer dereference."""
    rendered = _render_ok(chart, "--set", "podDisruptionBudget=null")
    assert _pdbs(rendered) == []


# ---------------------------------------------------------------------------
# P2-13 — requirePinnedKeys turns silent GitOps key rotation into a loud failure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_require_pinned_keys_fails_unpinned_offline_render(chart: str) -> None:
    err = _render_fail(chart, "--set", "security.requirePinnedKeys=true")
    assert "requirePinnedKeys" in err, err[:800]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_require_pinned_keys_passes_with_pinned_values(chart: str) -> None:
    _render_ok(
        chart,
        "--set",
        "security.requirePinnedKeys=true",
        "--set",
        "security.apiKey=test-key-material",
        "--set",
        "security.mediaTokenSecret=test-token-material",
    )


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_default_render_still_mints_keys_without_failing(chart: str) -> None:
    """Default (requirePinnedKeys: false) must stay offline-renderable — the
    render tests and CI depend on it."""
    out = _render_ok(chart)
    assert 'RAG_API_KEY: "' in out


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_pinned_rest_keys_render_byte_identical(chart: str) -> None:
    """M6 (wave-2 review): with BOTH REST keys pinned inline (and no other
    random source), two renders of the same values MUST be byte-identical —
    the GitOps contract.  (Pin only ONE key and MEDIA_TOKEN_SECRET is still
    minted per render; requirePinnedKeys catches that — next test.)"""
    args = (
        "--set",
        "security.apiKey=test-key",
        "--set",
        "security.mediaTokenSecret=test-token",
        "--set",
        "backups.enabled=false",
        "--set",
        "watchedSources.enabled=false",
    )
    a = _render_ok(chart, *args)
    b = _render_ok(chart, *args)
    assert a == b, "pinned values produced DIFFERENT renders — a key is minted per render"


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_require_pinned_keys_covers_the_media_token_alone(chart: str) -> None:
    """M6: the guard must refuse a render where ONLY the API key is pinned —
    a half-pinned deployment still mints MEDIA_TOKEN_SECRET per render."""
    err = _render_fail(
        chart,
        "--set",
        "security.requirePinnedKeys=true",
        "--set",
        "security.apiKey=only-the-api-key",
    )
    assert "MEDIA_TOKEN_SECRET" in err, err[:800]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_notes_warning_condition_covers_both_keys(chart: str) -> None:
    """M6: the NOTES warning fires unless BOTH keys are pinned — pinning only
    security.apiKey must still warn (helm template does not render NOTES, so
    this is the file-content contract)."""
    notes = (REPO / chart / "templates" / "NOTES.txt").read_text()
    assert "mediaTokenSecret" in notes.split("{{- if and")[-1], (
        "the NOTES warning condition does not test MEDIA_TOKEN_SECRET"
    )
    assert "$restKeysPinned" in notes, "the both-keys condition is missing"


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_busybox_init_image_digest_wiring(chart: str) -> None:
    """m9 (wave-2 review): the root init container joins the digest discipline."""
    dep = _deployment(_render_ok(chart), "rag-mcp-server")
    pod_spec = _pod_spec(dep)
    init = next(c for c in pod_spec["initContainers"] if c["name"] == "fix-pvc-permissions")
    assert init["image"] == "busybox:1.37.0"
    pinned = _deployment(_render_ok(chart, "--set", "initContainer.busybox.digest=sha256:bb11"), "rag-mcp-server")
    init_pinned = next(c for c in _pod_spec(pinned)["initContainers"] if c["name"] == "fix-pvc-permissions")
    assert init_pinned["image"] == "busybox:1.37.0@sha256:bb11"


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_notes_carries_the_gitops_key_warning(chart: str) -> None:
    """NOTES.txt renders only on install/upgrade (never in helm template),
    so this is a file-content contract — the warning must be conditional on
    the same generated-key path secret.yaml takes."""
    notes = (REPO / chart / "templates" / "NOTES.txt").read_text()
    assert "GITOPS / OFFLINE-RENDER WARNING" in notes, chart
    assert "requirePinnedKeys" in notes, chart
    assert 'lookup "v1" "Secret"' in notes, chart


# ---------------------------------------------------------------------------
# P2-14 — extraEnv reaches the cron workloads
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_extra_env_reaches_backup_and_watched_crons(chart: str) -> None:
    rendered = _render_ok(
        chart,
        "--set",
        "extraEnv.RAG_INGEST_CONCURRENCY=8",
        *BACKUP_ENABLE_ARGS,
        *WATCHED_ENABLE_ARGS,
    )
    for name_part in ("backups", "watched-sources"):
        env = _env_map(_container(_cron(rendered, name_part)))
        assert env.get("RAG_INGEST_CONCURRENCY") == "8", (
            f"extraEnv did not reach the {name_part} cron in {chart}: {env}"
        )


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_extra_env_reaches_stm_reaper_cron() -> None:
    rendered = _render_ok(
        "helm-scale-large",
        "--set",
        "extraEnv.RAG_INGEST_CONCURRENCY=8",
        *REAPER_ENABLE_ARGS,
    )
    env = _env_map(_container(_cron(rendered, "stm-reaper")))
    assert env.get("RAG_INGEST_CONCURRENCY") == "8", env


# ---------------------------------------------------------------------------
# P1-2c — operator knobs: prefetch floor + embed-batcher cache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_prefetch_floor_default_and_override(chart: str) -> None:
    data = _configmap(_render_ok(chart))["data"]
    assert data["RAG_PREFETCH_FLOOR"] == "200", data.get("RAG_PREFETCH_FLOOR")
    data2 = _configmap(_render_ok(chart, "--set", "rag.prefetchFloor=500"))["data"]
    assert data2["RAG_PREFETCH_FLOOR"] == "500"


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_embed_batch_cache_max_renders_in_large_chart() -> None:
    data = _configmap(_render_ok("helm-scale-large"))["data"]
    assert data["EMBED_BATCH_CACHE_MAX"] == "4096", data.get("EMBED_BATCH_CACHE_MAX")
    data2 = _configmap(_render_ok("helm-scale-large", "--set", "app.embedBatchCacheMax=64"))["data"]
    assert data2["EMBED_BATCH_CACHE_MAX"] == "64"


# ---------------------------------------------------------------------------
# P3-7 — optional image digest pinning
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_app_image_digest_wiring(chart: str) -> None:
    values = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    app_name = values["deployment"]["name"]
    repo = values["image"]["repository"]
    tag = values["image"]["tag"]
    dep = _deployment(_render_ok(chart), app_name)
    images = [c["image"] for c in _pod_spec(dep)["containers"]]
    assert all(img == f"{repo}:{tag}" for img in images), images  # default: no digest
    pinned = _render_ok(chart, "--set", "image.digest=sha256:abc123")
    dep_pinned = _deployment(pinned, app_name)
    images_pinned = [c["image"] for c in _pod_spec(dep_pinned)["containers"]]
    assert all(img == f"{repo}:{tag}@sha256:abc123" for img in images_pinned), images_pinned


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_qdrant_image_digest_wiring(chart: str) -> None:
    values = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    repo = values["qdrant"]["image"]["repository"]
    tag = values["qdrant"]["image"]["tag"]
    sts = _sts(_render_ok(chart), "qdrant")
    img = _container(sts)["image"]
    assert img == f"{repo}:{tag}", img
    pinned = _sts(_render_ok(chart, "--set", "qdrant.image.digest=sha256:qd42"), "qdrant")
    assert _container(pinned)["image"] == f"{repo}:{tag}@sha256:qd42"


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", SCALE_CHARTS)
def test_redis_image_digest_wiring(chart: str) -> None:
    values = yaml.safe_load((REPO / chart / "values.yaml").read_text())
    redis_name = values["deployment"]["name"] + "-redis"
    dep = _deployment(_render_ok(chart, "--set", "redis.image.digest=sha256:rd7"), redis_name)
    img = _container(dep)["image"]
    assert img.endswith("@sha256:rd7"), img


# ---------------------------------------------------------------------------
# P3-8 — qdrant runs with a read-only root filesystem
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_qdrant_readonly_root_filesystem(chart: str) -> None:
    sts = _sts(_render_ok(chart), "qdrant")
    container = _container(sts)
    sec = container["securityContext"]
    assert sec.get("readOnlyRootFilesystem") is True, sec
    assert sec.get("allowPrivilegeEscalation") is False
    mounts = {m["name"]: m["mountPath"] for m in container.get("volumeMounts", [])}
    assert mounts.get("tmp") == "/tmp", mounts
    # B1 (wave-2 review): qdrant writes snapshots to SNAPSHOTS_PATH (default
    # /qdrant/snapshots) — a read-only rootfs without a writable mount there
    # fails snapshot creation (and can fail at startup).
    assert mounts.get("snapshots") == "/qdrant/snapshots", mounts
    volumes = {v["name"]: v for v in _pod_spec(sts).get("volumes", [])}
    assert "tmp" in volumes and volumes["tmp"].get("emptyDir") == {}, volumes
    assert "snapshots" in volumes and volumes["snapshots"].get("emptyDir") == {}, volumes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
