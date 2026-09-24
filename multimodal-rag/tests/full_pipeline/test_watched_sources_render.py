"""Offline render tests for the watched-sources CronJob (feature: watched sources).

Runs the real `helm template` binary against the three chart variants and
asserts the gating contract from the implementation review / DECISIONS.md:

  * disabled (default) → the CronJob renders NOTHING and the whole default
    render is byte-identical to the render without the template file present
    (the "default render byte-identical" fleet bar);
  * enabled → exactly one CronJob carrying the app image, envFrom config +
    model-keys, `concurrencyPolicy: Forbid`, the source list as JSON env, and
    — the piece the backup cron historically missed — the deployment REST key
    (RAG_API_KEY from the model-keys Secret) the script sends as
    X-RAG-Api-Key;
  * config validation at values-parse time: non-s3 prefixes, single-object
    s3://file URLs, wildcards/query/whitespace, missing dataset/prefixes,
    duplicate datasets, unknown `type:` values, and an empty sources list all
    FAIL the render;
  * the backup cron carries the same X-RAG-Api-Key wiring (the verified
    latent-bug fix) — the script sends the header on both its calls.

Skipped when the helm binary is unavailable (offline/CI fallback), matching
the suite's skip conventions.

Run::

    python tests/full_pipeline/test_watched_sources_render.py    # standalone
    pytest tests/full_pipeline/test_watched_sources_render.py    # under pytest
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(str(REPO), "src"))  # keep the suite's import shape

CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render(chart: str, *set_args: str, values_file: str | None = None) -> str:
    cmd = ["helm", "template", "render-test", str(REPO / chart)]
    if values_file:
        cmd += ["-f", str(values_file)]
    cmd += list(set_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc.stdout if proc.returncode == 0 else f"__ERROR__\n{proc.stderr}"


def _render_ok(chart: str, *set_args: str) -> str:
    out = _render(chart, *set_args)
    assert not out.startswith("__ERROR__"), out[:2000]
    return out


def _render_err(chart: str, *set_args: str) -> str:
    out = _render(chart, *set_args)
    assert out.startswith("__ERROR__"), f"expected the render to FAIL, got:\n{out[:800]}"
    return out


def _cronjobs(rendered: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(rendered) if d and d.get("kind") == "CronJob"]


def _watched_cron(rendered: str) -> dict:
    crons = [c for c in _cronjobs(rendered) if "watched-sources" in c["metadata"]["name"]]
    assert len(crons) == 1, f"expected exactly one watched-sources CronJob, got {len(crons)}"
    return crons[0]


def _container(cron: dict) -> dict:
    return cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]


def _env(cron: dict) -> dict:
    return {e["name"]: e.get("value") for e in _container(cron).get("env", [])}


ENABLE_ARGS = (
    "--set",
    "watchedSources.enabled=true",
    "--set",
    "watchedSources.sources[0].dataset=reports",
    "--set",
    "watchedSources.sources[0].prefixes[0]=s3://mm-rag-drop/reports/",
)


# ---------------------------------------------------------------------------
# Gating: disabled renders nothing (the byte-identical default render)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_disabled_default_renders_no_cronjob(chart: str):
    rendered = _render_ok(chart)
    assert _cronjobs(rendered) == [], "no CronJob at all on the default render"
    assert "watched-sources" not in rendered
    assert "watchedSources" not in rendered


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_disabled_render_is_byte_identical_to_pristine(chart: str):
    """The fleet bar: with the feature off, the render equals the pre-feature
    render.  Compared against a render with the template file temporarily
    moved aside — the strongest form of the check (same inputs, same output).
    Randomly-generated Secret values are normalised (helm generates fresh
    ones per render); every other line must match byte-for-byte."""
    import re

    with tempfile.TemporaryDirectory() as td:
        tpl = REPO / chart / "templates" / "watched-sources-cronjob.yaml"
        parked = Path(td) / tpl.name
        shutil.move(str(tpl), parked)
        try:
            pristine = _render_ok(chart)
        finally:
            shutil.move(str(parked), tpl)
    rendered = _render_ok(chart)

    def norm(text: str) -> list[str]:
        out = []
        for line in text.splitlines():
            # First-install generated keys differ per render by design.
            line = re.sub(r'(RAG_API_KEY|MEDIA_TOKEN_SECRET): ".*"', r"\1: <generated>", line)
            out.append(line)
        return out

    assert norm(rendered) == norm(pristine), (
        "disabled watchedSources changed the default render — the gated template leaked"
    )


# ---------------------------------------------------------------------------
# Enabled: the CronJob and its wiring
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_enabled_renders_exactly_one_watched_cronjob(chart: str):
    rendered = _render_ok(chart, *ENABLE_ARGS)
    cron = _watched_cron(rendered)
    assert cron["metadata"]["name"].endswith("-watched-sources")
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"
    assert cron["spec"]["schedule"] == "*/30 * * * *"
    # Disabled backup cron must not appear just because watchedSources is on.
    assert not [c for c in _cronjobs(rendered) if "backup" in c["metadata"]["name"]]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_enabled_cronjob_wiring(chart: str):
    cron = _watched_cron(_render_ok(chart, *ENABLE_ARGS))
    spec = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    cont = _container(cron)

    # Same image as the app (values-driven tag).
    assert cont["image"].endswith(":v4.0.7") or ":v4." in cont["image"], cont["image"]

    # envFrom: config + model-keys (the model-keys Secret carries RAG_API_KEY).
    env_from = cont["envFrom"]
    assert {"configMapRef": {"name": "rag-mcp-server-config"}} in env_from
    assert {"secretRef": {"name": "rag-mcp-server-model-keys"}} in env_from

    # The script reads RAG_API_KEY and sends it as X-RAG-Api-Key — the piece
    # the backup cron was missing.
    script = cont["command"][2]
    assert 'os.environ.get("RAG_API_KEY")' in script
    assert '"X-RAG-Api-Key"' in script
    compile(script, "watched-cron.py", "exec")

    # Sources arrive as JSON env (no YAML parsing in the script).
    import json

    env = _env(cron)
    assert json.loads(env["WATCHED_SOURCES_JSON"]) == [
        {"dataset": "reports", "prefixes": ["s3://mm-rag-drop/reports/"]}
    ]
    # BACKUP_API_URL targets the -api Service of the same release.
    assert env["BACKUP_API_URL"].startswith("http://rag-mcp-server-api.default.svc.cluster.local")

    # Pod hardening mirrors the backup cron.
    assert spec["automountServiceAccountToken"] is False
    assert cont["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }


# ---------------------------------------------------------------------------
# Failure-debugging contract (2026-09 pcai-se incident): a failing tick must
# leave its logs readable.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_enabled_cronjob_debuggable_failure_retention(chart: str):
    """The pod must survive its own failure long enough for kubectl logs.

    Incident 2026-09 (pcai-se): restartPolicy OnFailure + backoffLimit 1
    made the job controller DELETE the failed pod ~5 s after start (the
    SuccessfulDelete-at-BackoffLimitExceeded event), so the cron's FAILED
    output was unreachable — the operator never saw the 404 that killed the
    tick.  The contract now:

      * restartPolicy Never — the failed pod stays in place as a terminated
        pod owned by the Job (no restart-within-pod, hence no deletion);
      * ttlSecondsAfterFinished: 3600 — the TTL controller garbage-collects
        the Job cascadingly (pods included) one hour after it finishes, so
        retention is bounded and failedJobsHistoryLimit (3 ticks) stays the
        outer bound;
      * backoffLimit stays 1 — a second pod attempt (fresh, since Never)
        runs before the Job fails.
    """
    cron = _watched_cron(_render_ok(chart, *ENABLE_ARGS))
    job_spec = cron["spec"]["jobTemplate"]["spec"]
    pod_spec = job_spec["template"]["spec"]
    assert pod_spec["restartPolicy"] == "Never", pod_spec["restartPolicy"]
    assert job_spec["ttlSecondsAfterFinished"] == 3600
    assert job_spec["backoffLimit"] == 1


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_enabled_cronjob_bootstraps_missing_datasets(chart: str):
    """The script creates each dataset before syncing it.

    Incident 2026-09 (pcai-se): the configured showcase-* datasets did not
    exist, so POST /api/datasets/{name}/batch-urls answered an instant 404
    (get_dataset raises FileNotFoundError before any job is created) — the
    cron raised_for_status()'d and the tick failed in ~2 s per source,
    forever.  The script now POSTs /api/datasets first (409 = already
    exists = the expected steady state) so a values-only rollout heals
    itself, and prints a MinIO listing hint when the sync still fails on a
    missing dataset/bucket so the next failure is actionable.
    """
    script = _container(_watched_cron(_render_ok(chart, *ENABLE_ARGS)))["command"][2]
    compile(script, "watched-cron.py", "exec")
    # Bootstrap: create-then-sync, 409 tolerated as success.
    assert 'f"{api}/api/datasets"' in script, "missing dataset bootstrap call"
    assert "409" in script and "already exists" in script
    assert "404" in script, "the 404 path must be explained, not just raised"
    # Actionable hints on the S3 failure paths.
    assert "HINT" in script
    assert "S3_ENDPOINT_URL" in script
    assert "NoSuchBucket" in script
    assert "InvalidAccessKeyId" in script and "SignatureDoesNotMatch" in script


# ---------------------------------------------------------------------------
# Values-parse-time config validation (the "helm template check")
# ---------------------------------------------------------------------------


def _src(prefix: str = "s3://bucket/pfx/", dataset: str = "reports") -> list[str]:
    return [
        "--set",
        "watchedSources.enabled=true",
        "--set",
        f"watchedSources.sources[0].dataset={dataset}",
        "--set",
        f"watchedSources.sources[0].prefixes[0]={prefix}",
    ]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_valid_configs_render(chart: str):
    # Plain prefix, whole-bucket prefix, extensionless (directory-heuristic) prefix.
    for prefix in ("s3://bucket/pfx/", "s3://bucket/", "s3://bucket/pfx"):
        _render_ok(chart, *_src(prefix))
    # Two sources, two prefixes on one source.
    _render_ok(
        chart,
        "--set",
        "watchedSources.enabled=true",
        "--set",
        "watchedSources.sources[0].dataset=reports",
        "--set",
        "watchedSources.sources[0].prefixes[0]=s3://b/r/",
        "--set",
        "watchedSources.sources[1].dataset=logs",
        "--set",
        "watchedSources.sources[1].prefixes[0]=s3://b/l/",
        "--set",
        "watchedSources.sources[1].prefixes[1]=s3://b/t/",
    )
    # The reserved type field, explicit "s3".
    _render_ok(chart, *_src(), "--set", "watchedSources.sources[0].type=s3")


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_invalid_configs_fail_the_render(chart: str):
    bad_sets = [
        # non-s3 prefix
        _src("https://example.com/x/"),
        # single-object s3 URL (sync refuses it server-side too)
        _src("s3://bucket/file.pdf"),
        # wildcard / query / whitespace
        _src("s3://bucket/*"),
        _src("s3://bucket/pfx?x=1"),
        _src("s3://bucket/pfx x/"),
        # missing dataset / empty prefixes
        ["--set", "watchedSources.enabled=true", "--set", "watchedSources.sources[0].prefixes[0]=s3://b/p/"],
        ["--set", "watchedSources.enabled=true", "--set", "watchedSources.sources[0].dataset=reports"],
        # enabled with no sources at all
        ["--set", "watchedSources.enabled=true"],
        # unknown reserved type
        [*_src(), "--set", "watchedSources.sources[0].type=sharepoint"],
        # duplicate dataset across sources
        [
            "--set",
            "watchedSources.enabled=true",
            "--set",
            "watchedSources.sources[0].dataset=reports",
            "--set",
            "watchedSources.sources[0].prefixes[0]=s3://b/p/",
            "--set",
            "watchedSources.sources[1].dataset=reports",
            "--set",
            "watchedSources.sources[1].prefixes[0]=s3://b/q/",
        ],
    ]
    for sets in bad_sets:
        err = _render_err(chart, *sets)
        assert "watchedSources" in err, err[:500]


# ---------------------------------------------------------------------------
# The backup-cron 401 fix rides here (verified latent bug)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_backup_cron_sends_the_api_key(chart: str):
    rendered = _render_ok(chart, "--set", "backups.enabled=true", "--set", "backups.bucket=test-bucket")
    crons = [c for c in _cronjobs(rendered) if "backup" in c["metadata"]["name"]]
    assert len(crons) == 1
    script = _container(crons[0])["command"][2]
    compile(script, "backup-cron.py", "exec")
    assert 'os.environ.get("RAG_API_KEY")' in script, "backup cron must present the deployment key"
    assert '"X-RAG-Api-Key"' in script
    # Both HTTP calls carry the header.
    assert script.count("headers=headers or None") == 2, "the datasets list AND the export stream must send the key"


# ---------------------------------------------------------------------------
# Example values files stay disabled (ship-what-we-document)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_example_values_files_carry_disabled_block():
    example_files = [
        "helm/local/values.example.yaml",
        "helm/values-examples/values.g2.yaml",
        "helm/values-examples/values.hosted-trial.yaml",
        "helm-scale-medium/values-examples/values.g2.yaml",
        "helm-scale-medium/values-examples/values.hosted-trial.yaml",
        "helm-scale-large/values-examples/values.g2.yaml",
        "helm-scale-large/values-examples/values.hosted-trial.yaml",
    ]
    for rel in example_files:
        data = yaml.safe_load((REPO / rel).read_text())
        assert "watchedSources" in data, rel
        ws = data["watchedSources"]
        assert ws["enabled"] is False, rel
        assert ws["sources"] == [], rel


if __name__ == "__main__":
    import traceback

    failed = 0
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        # Parametrised-over-chart tests expand once per chart standalone.
        charts = CHARTS if fn.__code__.co_varnames[:1] == ("chart",) else [None]
        try:
            for c in charts:
                fn(c) if c else fn()
            print(f"  {name} ... OK")
        except Exception:
            failed += 1
            print(f"  {name} ... FAIL")
            traceback.print_exc()
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) failed'}")
    sys.exit(1 if failed else 0)
