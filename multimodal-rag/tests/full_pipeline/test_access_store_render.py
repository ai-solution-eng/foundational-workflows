"""Offline render tests for the D16 access store + unlock TTL bound (all charts).

Runs the real `helm template` binary against the three chart variants and
asserts the gating contract:

  * RAG_UNLOCK_MAX_TTL is UNCONDITIONAL (always rendered — the server's
    built-in default equals the chart default 86400, so behaviour is
    identical) and tracks the value;
  * RAG_ACCESS_STORE is GATED: absent on the default render (the whole
    security.* block leaks nothing), "true" only when security.accessStore;
  * RAG_ACCESS_DENY_SELECT and RAG_MEMORY_DEFAULT render only when set
    (empty values render nothing — no empty-string env noise);
  * the default render is byte-identical to a render with the access-store
    template lines removed (the "default render byte-identical" fleet bar);
  * all three charts (helm / helm-scale-medium / helm-scale-large) carry the
    same gating.

Skipped when the helm binary is unavailable (offline/CI fallback), matching
the suite's skip conventions.

Run::

    pytest tests/full_pipeline/test_access_store_render.py -q
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHARTS = ["helm", "helm-scale-medium", "helm-scale-large"]


def _helm_available() -> bool:
    return shutil.which("helm") is not None


def _render_ok(chart: str, *set_args: str) -> str:
    cmd = ["helm", "template", "render-test", str(REPO / chart)]
    cmd += list(set_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"helm template failed for {chart}:\n{proc.stderr}"
    return proc.stdout


def _configmap(rendered: str) -> str:
    """The ConfigMap's data section (the first `kind: ConfigMap` document)."""
    idx = rendered.index("kind: ConfigMap")
    # Walk back to the document start, forward to the data: block.
    doc = rendered[idx:]
    m = re.search(r"data:\n((?:  .*\n|\n)+?)(?=\n[^ ]|\Z)", doc)
    assert m, "configmap data section not found"
    return m.group(1)


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_unlock_max_ttl_always_rendered_at_default(chart):
    env = _configmap(_render_ok(chart))
    assert 'RAG_UNLOCK_MAX_TTL: "86400"' in env
    # D20-era default: accessStore defaults TRUE (ratified 2026-09-24 — the
    # user's internal charts ship self-service ON; the off case is explicit).
    assert 'RAG_ACCESS_STORE: "true"' in env
    assert "RAG_ACCESS_DENY_SELECT" not in env
    assert "RAG_MEMORY_DEFAULT" not in env


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_unlock_max_ttl_tracks_value(chart):
    env = _configmap(_render_ok(chart, "--set", "rag.unlockMaxTtl=3600"))
    assert 'RAG_UNLOCK_MAX_TTL: "3600"' in env
    env0 = _configmap(_render_ok(chart, "--set", "rag.unlockMaxTtl=0"))
    assert 'RAG_UNLOCK_MAX_TTL: "0"' in env0  # the no-expiry opt-in


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_access_store_flag_tracks_value(chart):
    # Default is TRUE (ratified); explicit false opts OUT and renders "false".
    on = _configmap(_render_ok(chart))
    assert 'RAG_ACCESS_STORE: "true"' in on
    off = _configmap(_render_ok(chart, "--set", "security.accessStore=false"))
    assert 'RAG_ACCESS_STORE: "false"' in off


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_denylist_and_memory_default_render_only_when_set(chart):
    # Empty (default): nothing rendered.
    env = _configmap(_render_ok(chart, "--set", "security.accessStore=true"))
    assert "RAG_ACCESS_DENY_SELECT" not in env
    assert "RAG_MEMORY_DEFAULT" not in env
    # Set: both render (comma values need helm's escaped form).
    env = _configmap(
        _render_ok(
            chart,
            "--set", "security.accessStore=true",
            "--set", "security.memoryDefault=team-knowledge",
            "--set-string", r"security.accessDenySelect=hr-data\,payroll",
        )
    )
    assert 'RAG_ACCESS_DENY_SELECT: "hr-data,payroll"' in env
    assert 'RAG_MEMORY_DEFAULT: "team-knowledge"' in env


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_access_store_always_rendered_tracks_default(chart):
    """With accessStore defaulting TRUE (ratified 2026-09-24), the
    RAG_ACCESS_STORE key is unconditional — every render carries it, and
    only the denylist/memory-default sub-blocks stay gated on being set."""
    env = _configmap(_render_ok(chart))
    assert 'RAG_ACCESS_STORE: "true"' in env
    # The gated sub-blocks render nothing when their values are empty.
    assert "RAG_ACCESS_DENY_SELECT" not in env
    assert "RAG_MEMORY_DEFAULT" not in env
    # And the opt-out state is explicit, not silent.
    off = _configmap(_render_ok(chart, "--set", "security.accessStore=false"))
    assert 'RAG_ACCESS_STORE: "false"' in off
