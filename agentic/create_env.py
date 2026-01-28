#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import yaml


def sanitize(value: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^A-Za-z0-9]", "_", value)


def upper(value: str) -> str:
    return value.upper()


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text())


def render_env(cfg: dict) -> dict:
    aie_domain = cfg["aie_domain"]
    nemo_ns = cfg["nemo"]["namespace"]

    main_model_id = cfg["nemo"]["main_model"]["id"]
    guardrail_model_id = cfg["nemo"]["guardrail_model"]["id"]

    main_model_name = sanitize(main_model_id)
    guardrail_model_name = sanitize(guardrail_model_id)

    return {
        # AIE
        "AIE_DOMAIN": aie_domain,

        # Phoenix
        "PHOENIX_ISTIO_ENDPOINT": f"https://phoenix.{aie_domain}",
        "PHOENIX_COLLECTOR_ENDPOINT": (
            "http://arize-phoenix-svc.phoenix.svc.cluster.local:4317"
        ),
        "PHOENIX_API_KEY": cfg["phoenix"].get("api_key", ""),

        # Guardrails
        "NEMO_NAMESPACE": nemo_ns,
        "GUARDRAIL_CONFIG_NAME": cfg["nemo"]["guardrail_config_name"],
        "GUARDRAILS_BASE_URL": f"https://nemo-guardrails.{aie_domain}/v1/guardrail",

        # Main model
        "NEMO_MAIN_MODEL_ENGINE": cfg["nemo"]["main_model"]["engine"],
        "NEMO_MAIN_MODEL_ID": main_model_id,
        "NEMO_MAIN_MODEL_URL": cfg["nemo"]["main_model"].get("url", ""),
        "NEMO_MAIN_MODEL_TOKEN": cfg["nemo"]["main_model"].get("token", ""),
        "NEMO_MAIN_MODEL_NAME": main_model_name,
        "DEPLOYMENT_CONFIG_MAIN_MODEL": upper(
            f"{nemo_ns}_{main_model_name}"
        ),

        # Guardrail model
        "NEMO_GUARDRAIL_MODEL_ENGINE": cfg["nemo"]["guardrail_model"]["engine"],
        "NEMO_GUARDRAIL_MODEL_ID": guardrail_model_id,
        "NEMO_GUARDRAIL_MODEL_URL": cfg["nemo"]["guardrail_model"].get("url", ""),
        "NEMO_GUARDRAIL_MODEL_TOKEN": cfg["nemo"]["guardrail_model"].get("token", ""),
        "NEMO_GUARDRAIL_MODEL_NAME": guardrail_model_name,
        "DEPLOYMENT_CONFIG_GUARDRAIL_MODEL": upper(
            f"{nemo_ns}_{guardrail_model_name}"
        ),

        # Endpoints
        "NIM_ENDPOINT_URL": f"https://nemo-nim-proxy.{aie_domain}",
        "DEPLOYMENT_BASE_URL": (
            f"https://nemo-deployment-management.{aie_domain}/v1/deployment"
        ),

        # MCP servers
        "MCP_EZPRESTO_SERVER": f"https://mcp-ezpresto-server.{aie_domain}/mcp",
        "MCP_FILESYSTEM_SERVER": f"https://mcp-filesystem-server.{aie_domain}/mcp",
        "MCP_S3_SERVER": f"https://mcp-s3-server.{aie_domain}/mcp",
    }


def write_env(env: dict, output: Path) -> None:
    lines = [f"export {key}={value}" for key, value in env.items()]
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a literal .env file from config.yaml"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "-o", "--output", default=".env", help="Output .env file"
    )

    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    env = render_env(cfg)
    write_env(env, Path(args.output))


if __name__ == "__main__":
    main()