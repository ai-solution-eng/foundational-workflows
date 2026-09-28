"""Offline render tests for the SECRET-FIRST model/S3/Redis key sourcing.

The `<deployment.name>-model-keys` Secret carries four layers of material:

  * ``MODEL_*_API_KEY``   — outbound model-serving JWTs (values: modelSecrets.*)
  * ``S3_*``              — MinIO/S3 credentials (values: s3.*)
  * ``REDIS_PASSWORD``    — unlock-cache auth (values: redis.password)

Sourcing contract (2026-09, secret-first/sticky):

  * SEED: with no pre-existing Secret (offline `helm template` — `lookup`
    finds nothing), values seed the keys verbatim; unset values render
    `""` (no key material is ever invented — unlike the REST keys, which
    auto-generate on first install).
  * REST keys are UNCHANGED: seeded from `security.apiKey` only when set,
    never rendered at all when `security.existingSecret` is set.
  * STICKY (stored value wins on upgrade) requires a live cluster for the
    `lookup` — verified against the real release via `helm upgrade
    --dry-run`, not here; this file pins the seed path and determinism.

Skipped when the helm binary is unavailable (suite skip conventions).

Run::

    pytest tests/full_pipeline/test_model_keys_seed_render.py -q
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


def _model_keys_secret(rendered: str) -> str:
    """The `*-model-keys` Secret document's stringData block."""
    docs = rendered.split("\n---\n")
    for doc in docs:
        if "kind: Secret" in doc and "-model-keys" in doc:
            m = re.search(r"stringData:\n((?:  .*\n)+)", doc)
            assert m, "model-keys Secret has no stringData block"
            return m.group(1)
    raise AssertionError("model-keys Secret not found in render")


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_values_seed_model_keys(chart):
    """Values entries seed the keys verbatim on first render (no Secret yet)."""
    data = _model_keys_secret(
        _render_ok(
            chart,
            "--set", "modelSecrets.embedderApiKey=seed-embed",
            "--set", "modelSecrets.rerankerApiKey=seed-rerank",
            "--set", "modelSecrets.vlmApiKey=seed-vlm",
            "--set", "modelSecrets.asrApiKey=seed-asr",
            "--set", "s3.accessKeyId=seed-id",
            "--set", "s3.secretAccessKey=seed-key",
            "--set", "redis.password=seed-pw",
            "--set", "security.existingSecret=operator-owned",
        )
    )
    assert 'MODEL_EMBEDDER_API_KEY: "seed-embed"' in data
    assert 'MODEL_RERANKER_API_KEY: "seed-rerank"' in data
    assert 'MODEL_VLM_API_KEY: "seed-vlm"' in data
    assert 'MODEL_ASR_API_KEY: "seed-asr"' in data
    assert 'S3_ACCESS_KEY_ID: "seed-id"' in data
    assert 'S3_SECRET_ACCESS_KEY: "seed-key"' in data
    assert 'REDIS_PASSWORD: "seed-pw"' in data


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_unset_values_render_empty_never_generated(chart):
    """No material is invented for model/S3/Redis keys (unlike REST keys)."""
    data = _model_keys_secret(
        _render_ok(chart, "--set", "security.existingSecret=operator-owned")
    )
    for key in (
        "MODEL_EMBEDDER_API_KEY",
        "MODEL_RERANKER_API_KEY",
        "MODEL_VLM_API_KEY",
        "MODEL_ASR_API_KEY",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "REDIS_PASSWORD",
    ):
        assert f'{key}: ""' in data


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_rest_keys_still_hidden_under_existing_secret(chart):
    """existingSecret keeps suppressing the REST keys (unchanged behaviour)."""
    data = _model_keys_secret(
        _render_ok(
            chart,
            "--set", "modelSecrets.embedderApiKey=seed-embed",
            "--set", "security.existingSecret=operator-owned",
        )
    )
    assert "RAG_API_KEY" not in data
    assert "MEDIA_TOKEN_SECRET" not in data


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_seed_render_is_deterministic(chart):
    """Two identical seeded renders are byte-identical (no hidden randomness
    on the model-keys path — the only rand* calls left are the REST keys,
    which existingSecret mode suppresses)."""
    args = (
        "--set", "modelSecrets.embedderApiKey=seed-embed",
        "--set", "security.existingSecret=operator-owned",
    )
    assert _render_ok(chart, *args) == _render_ok(chart, *args)
