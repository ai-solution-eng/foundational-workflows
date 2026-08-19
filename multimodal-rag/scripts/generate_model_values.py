#!/usr/bin/env python3
"""Generate the ``models:`` / ``modelSecrets:`` block for a chart ``values.yaml``
from the hardcoded model definitions in ``multimodal_rag/utils/pcai_models.py``.

Reads each selected model object and emits a YAML block in the same shape as
the charts expect::

    models:
      embedder:
        name: "Qwen/Qwen3-VL-Embedding-8B"
        url: "https://qwen3-vl-embedding-8b..../v1"
        className: "MultiModalEmbeddings"
        extra: {...}
      ...
    modelSecrets:
      embedderApiKey: "<jwt>"
      ...

The API keys are taken from ``pcai_models.py`` and written into the
``modelSecrets`` section by default.  Use ``--skip-secrets`` to leave them
blank so the values file stays secret-free (keys belong in the Kubernetes
Secret ``-model-keys`` anyway).

Usage::

    python scripts/generate_model_values.py                  # defaults (deployed set)
    python scripts/generate_model_values.py --list-models    # show candidates
    python scripts/generate_model_values.py --vlm deepseek_v4_flash_280B --skip-secrets
    python scripts/generate_model_values.py --out /tmp/models.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

import multimodal_rag.utils.pcai_models as pcai  # noqa: E402

# role -> (default model variable name, secret key in modelSecrets)
ROLES = {
    "embedder": ("qwen3_vl_8B", "embedderApiKey"),
    "reranker": ("qwen3_vl_reranker_8B", "rerankerApiKey"),
    "vlm": ("qwen38_27B", "vlmApiKey"),
    "asr": ("cohere_transcribe_3_2b", "asrApiKey"),
}

# Extra params surfaced for the embedding model (mirrors the values form).
_EMBEDDER_EXTRA_FIELDS = (
    "embedding_dim",
    "chunk_size",
    "chunk_overlap",
    "code_chunk_size",
    "code_chunk_overlap",
    "tokenizer_name",
    "tokenizer_type",
)


def _model_candidates() -> dict[str, object]:
    import inspect

    return {
        name: obj
        for name, obj in vars(pcai).items()
        if not name.startswith("_")
        and not inspect.isclass(obj)
        and hasattr(obj, "url_remote")
        and hasattr(obj, "api_key")
    }


def _cls_name(obj: object) -> str:
    return getattr(obj, "model_instantiation_class", None) and getattr(
        getattr(obj, "model_instantiation_class"), "__name__", ""
    ) or ""


def _resolve(role: str, choice: str | None) -> object:
    candidates = _model_candidates()
    default = ROLES[role][0]
    name = choice or default
    if name not in candidates:
        raise SystemExit(
            f"Unknown model '{name}' for role '{role}'. "
            f"Available: {', '.join(sorted(candidates))}"
        )
    obj = candidates[name]
    if not getattr(obj, "currently_deployed", True):
        print(f"# NOTE: {name} has currently_deployed=False — endpoint may reject requests.",
              file=sys.stderr)
    return obj


def _embedder_extra(obj: object) -> dict[str, object]:
    extra: dict[str, object] = {}
    for field in _EMBEDDER_EXTRA_FIELDS:
        value = getattr(obj, field, None)
        if value is None:
            continue
        extra[field] = value
    return extra


def build_values(opts: argparse.Namespace) -> dict[str, object]:
    models: dict[str, object] = {}
    secrets: dict[str, object] = {}

    for role, (default_var, secret_key) in ROLES.items():
        obj = _resolve(role, getattr(opts, role, None))
        entry: dict[str, object] = {}

        name = getattr(obj, "model_name", None)
        if name:
            entry["name"] = name
        entry["url"] = getattr(obj, "url_remote", "") or ""

        if role in ("embedder", "reranker"):
            entry["className"] = _cls_name(obj) or {
                "embedder": "MultiModalEmbeddings",
                "reranker": "MultiModalReranker",
            }[role]

        if role == "embedder":
            entry["extra"] = _embedder_extra(obj)
        else:
            entry["extra"] = {}

        models[role] = entry
        secrets[secret_key] = "" if opts.skip_secrets else (getattr(obj, "api_key", "") or "")

    return {"models": models, "modelSecrets": secrets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true",
                        help="print the available model variables from pcai_models.py and exit")
    parser.add_argument("--skip-secrets", action="store_true",
                        help="emit blank modelSecrets (secrets live in the Kubernetes Secret)")
    parser.add_argument("--out", type=str, default="",
                        help="write YAML to this file instead of stdout")
    for role in ROLES:
        parser.add_argument(f"--{role}", type=str, default=None,
                            help=f"model variable for role '{role}' (default: {ROLES[role][0]})")
    opts = parser.parse_args()

    if opts.list_models:
        print("\n".join(sorted(_model_candidates())))
        return

    values = build_values(opts)
    text = yaml.safe_dump(values, sort_keys=False, width=120, default_flow_style=False)
    # A bare safety comment so the block is self-explanatory when pasted.
    text = "# Generated from multimodal_rag/utils/pcai_models.py\n" + text

    if opts.out:
        Path(opts.out).write_text(text, encoding="utf-8")
        print(f"wrote {opts.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()