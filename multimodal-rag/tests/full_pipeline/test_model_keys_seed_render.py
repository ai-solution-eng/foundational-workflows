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
  * USER-SECRET MODE (2026-10): `modelSecrets.existingSecret` names a
    Secret you own; the chart wires it LAST in envFrom and CONFIG_DIR on
    every consuming workload (api + mcp [+ embed-batcher], backup +
    watched-sources cronjobs), so each key it carries overrides the
    chart-managed Secret and keys it omits keep the seeded/stored values.
    Hard-wired like security.existingSecret (missing Secret =
    CreateContainerConfigError).  The STICKY MODEL_*_URL keys are preserved
    from the live Secret the same way — offline renders never invent them.
  * UPGRADE GUARD (2026-10): in-cluster `helm upgrade` fails the render
    when no source provides an embedder URL or API key.  The guard is
    `.Release.IsUpgrade`-scoped BY DESIGN — every `helm template` render
    below doubles as the proof that offline/CI renders stay clean with no
    key material anywhere; the in-cluster failure path needs a live cluster
    (verify via `helm upgrade --dry-run` on a scratch namespace).

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
            "--set",
            "modelSecrets.embedderApiKey=seed-embed",
            "--set",
            "modelSecrets.rerankerApiKey=seed-rerank",
            "--set",
            "modelSecrets.vlmApiKey=seed-vlm",
            "--set",
            "modelSecrets.asrApiKey=seed-asr",
            "--set",
            "s3.accessKeyId=seed-id",
            "--set",
            "s3.secretAccessKey=seed-key",
            "--set",
            "redis.password=seed-pw",
            "--set",
            "security.existingSecret=operator-owned",
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
    data = _model_keys_secret(_render_ok(chart, "--set", "security.existingSecret=operator-owned"))
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
            "--set",
            "modelSecrets.embedderApiKey=seed-embed",
            "--set",
            "security.existingSecret=operator-owned",
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
        "--set",
        "modelSecrets.embedderApiKey=seed-embed",
        "--set",
        "security.existingSecret=operator-owned",
    )
    assert _render_ok(chart, *args) == _render_ok(chart, *args)


# ---------------------------------------------------------------------------
# User-owned Secret mode (modelSecrets.existingSecret, 2026-10)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_user_model_secret_wiring(chart):
    """modelSecrets.existingSecret wires the user Secret LAST on every
    consuming workload: envFrom (later source wins → per-key override),
    CONFIG_DIR (watcher merge order matches), and the volume mount.  The
    cronjobs chain it too (their S3_*/REDIS_PASSWORD view)."""
    rendered = _render_ok(
        chart,
        "--set",
        "modelSecrets.existingSecret=user-keys",
        "--set",
        "models.embedder.url=https://emb.example/v1",
        "--set",
        "modelSecrets.embedderApiKey=seed-embed",
        "--set",
        "security.existingSecret=operator-owned",
    )
    chart_refs = list(re.finditer(r"secretRef:\n\s+name: rag-mcp-server-model-keys\n", rendered))
    user_refs = list(re.finditer(r"secretRef:\n\s+name: user-keys\n", rendered))
    assert chart_refs, "chart-managed secretRefs not found"
    assert len(user_refs) == len(chart_refs), (
        f"user Secret must follow the chart Secret in EVERY envFrom chain "
        f"(got {len(chart_refs)} chart refs, {len(user_refs)} user refs)"
    )
    assert user_refs[0].start() > chart_refs[0].start()
    # CONFIG_DIR gains the user dir LAST — watcher precedence matches envFrom.
    assert 'value: "/etc/rag/config:/etc/rag/secrets:/etc/rag/model-keys-user"' in rendered
    # Volume wired to the user Secret (hot-reload path).
    assert "name: rag-model-keys-user" in rendered
    assert "secretName: user-keys" in rendered


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_user_model_secret_render_is_gated(chart):
    """Without modelSecrets.existingSecret the render carries no trace of the
    user-secret wiring (gated-feature rule: default render byte-identical)."""
    rendered = _render_ok(
        chart,
        "--set",
        "security.existingSecret=operator-owned",
        "--set",
        "models.embedder.url=https://emb.example/v1",
    )
    assert "model-keys-user" not in rendered


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", CHARTS)
def test_sticky_url_keys_absent_offline(chart):
    """MODEL_*_URL entries are sticky-only (re-rendered from the live Secret
    on upgrade so out-of-band edits survive); an offline render never
    invents them — the ConfigMap stays the values-borne URL source."""
    data = _model_keys_secret(
        _render_ok(
            chart,
            "--set",
            "security.existingSecret=operator-owned",
            "--set",
            "models.embedder.url=https://emb.example/v1",
        )
    )
    for key in ("MODEL_EMBEDDER_URL", "MODEL_RERANKER_URL", "MODEL_VLM_URL", "MODEL_ASR_URL"):
        assert key not in data


@pytest.mark.skipif(not _helm_available(), reason="helm binary not available")
@pytest.mark.parametrize("chart", ["helm-scale-medium", "helm-scale-large"])
def test_redis_server_follows_user_secret(chart):
    """The Redis SERVER (scale charts) must resolve REDIS_PASSWORD through the
    same per-key chain as the API clients: in user-secret mode it switches
    from the explicit env secretKeyRef (which would SHADOW the user Secret)
    to the chart→user envFrom chain, and --requirepass is passed whenever a
    password source exists (empty everywhere still means auth off)."""
    user_render = _render_ok(
        chart,
        "--set",
        "redis.enabled=true",
        "--set",
        "modelSecrets.existingSecret=user-keys",
        "--set",
        "models.embedder.url=https://emb.example/v1",
        "--set",
        "modelSecrets.embedderApiKey=seed-embed",
        "--set",
        "security.existingSecret=operator-owned",
    )
    # The redis workload is a Deployment (scale charts); identify its doc by
    # the unique redis-server command, not by kind.
    redis_doc = next(d for d in user_render.split("\n---\n") if "redis-server" in d)
    assert "name: user-keys" in redis_doc, "Redis server must envFrom the user Secret"
    # An explicit env valueFrom/secretKeyRef would shadow the user Secret —
    # user mode must use the envFrom chain only (check valueFrom:, not the
    # bare word, which also appears in the explanatory comment).
    assert "valueFrom:" not in redis_doc
    assert "--requirepass" in redis_doc
    # Values mode keeps the original shape: explicit secretKeyRef, no user Secret.
    values_render = _render_ok(
        chart,
        "--set",
        "redis.enabled=true",
        "--set",
        "redis.password=pw1",
        "--set",
        "models.embedder.url=https://emb.example/v1",
        "--set",
        "modelSecrets.embedderApiKey=seed-embed",
        "--set",
        "security.existingSecret=operator-owned",
    )
    redis_doc = next(d for d in values_render.split("\n---\n") if "redis-server" in d)
    assert "name: user-keys" not in redis_doc
    assert "secretKeyRef" in redis_doc
