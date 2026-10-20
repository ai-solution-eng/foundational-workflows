"""Offline render tests for the backup CronJob's protected-dataset support.

Runs the real `helm template` binary against the three chart variants and
asserts:

  * default (existingSecret: '') → NO dataset-passwords volume/mount renders
    anywhere, and the whole default render stays byte-identical to a render
    with the key removed (the "default render byte-identical" fleet bar);
  * existingSecret set → the CronJob carries the secret volume + read-only
    mount at /etc/rag/dataset-passwords, and the inline script references it;
  * the template references .Values.backups.protectedPasswords.existingSecret
    (guards against the key being renamed without the template following).

Skipped when the helm binary is unavailable (offline/CI fallback), matching
the suite's skip conventions.

Run::

    python tests/full_pipeline/test_backup_protected_render.py   # standalone
    pytest tests/full_pipeline/test_backup_protected_render.py   # under pytest
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(str(REPO), "src"))

CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render(chart: str, *set_args: str) -> str:
    cmd = ["helm", "template", "render-test", str(REPO / chart)] + list(set_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc.stdout if proc.returncode == 0 else f"__ERROR__\n{proc.stderr}"


def _render_ok(chart: str, *set_args: str) -> str:
    out = _render(chart, *set_args)
    assert not out.startswith("__ERROR__"), out[:2000]
    return out


def _backup_cron(rendered: str) -> dict:
    crons = [d for d in yaml.safe_load_all(rendered) if d and d.get("kind") == "CronJob" and "backups" in d["metadata"]["name"]]
    assert len(crons) == 1, f"expected exactly one backup CronJob, got {len(crons)}"
    return crons[0]


def _has_passwords_artifacts(rendered: str) -> bool:
    return "dataset-passwords" in rendered


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_default_render_has_no_passwords_mount(chart):
    """Default values: the password Secret wiring must render nothing."""
    out = _render_ok(chart)
    assert not _has_passwords_artifacts(out), (
        f"{chart}: default render carries dataset-passwords artifacts — "
        "the feature must be fully opt-in"
    )


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_default_render_byte_identical_without_key(chart, tmp_path):
    """Removing the protectedPasswords key must not change the default render
    (randomly-generated Secret values normalised — they differ per render by
    design)."""
    import re

    values = (REPO / chart / "values.yaml").read_text()
    stripped = tmp_path / "values-stripped.yaml"
    lines = values.splitlines(keepends=True)
    out_lines, i = [], 0
    while i < len(lines):
        if lines[i].rstrip() == "  protectedPasswords:":
            i += 1  # drop the key line and its indented children
            while i < len(lines) and lines[i].startswith("    "):
                i += 1
        else:
            out_lines.append(lines[i]); i += 1
    stripped.write_text("".join(out_lines))
    out_with = _render_ok(chart)
    out_without = _render_ok(chart, "-f", str(stripped))

    def norm(text: str) -> list[str]:
        return [re.sub(r'(RAG_API_KEY|MEDIA_TOKEN_SECRET): ".*"', r"\1: <generated>", line)
                for line in text.splitlines()]

    assert norm(out_with) == norm(out_without)


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_enabled_secret_renders_volume_and_mount(chart):
    out = _render_ok(chart, "--set", "backups.enabled=true",
                     "--set", "backups.bucket=backup-bkt",
                     "--set", "backups.protectedPasswords.existingSecret=rag-dataset-passwords")
    cron = _backup_cron(out)
    container = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    mounts = {m["name"]: m for m in container.get("volumeMounts", [])}
    volumes = {v["name"]: v for v in cron["spec"]["jobTemplate"]["spec"]["template"]["spec"].get("volumes", [])}
    assert "dataset-passwords" in mounts, f"missing mount: {list(mounts)}"
    assert mounts["dataset-passwords"]["mountPath"] == "/etc/rag/dataset-passwords"
    assert mounts["dataset-passwords"].get("readOnly") is True
    assert "dataset-passwords" in volumes
    assert volumes["dataset-passwords"]["secret"]["secretName"] == "rag-dataset-passwords"
    assert "dataset-passwords" in container.get("command", [""])[2] or "PW_DIR" in container.get("command", [""])[2]


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_template_references_values_key():
    """The template must read the values key (rename-guard)."""
    for chart in CHARTS:
        template = (REPO / chart / "templates" / "backup-cronjob.yaml").read_text()
        assert ".Values.backups.protectedPasswords.existingSecret" in template, chart


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
def test_templates_identical_across_charts():
    blobs = [(REPO / c / "templates" / "backup-cronjob.yaml").read_text() for c in CHARTS]
    assert all(b == blobs[0] for b in blobs), "backup-cronjob.yaml drifted across chart variants"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
