# Agentic Core Workflow
## Walkthrough
### Setup
- Install the following frameworks:
    - [NeMo Microservices Guardrails `v25.12.0`](https://github.com/ai-solution-eng/frameworks/tree/main/nemo-microservices/guardrails-only)
    - [Arize Phoenix `v4.0.17`](https://github.com/ai-solution-eng/frameworks/tree/main/arize-phoenix)
    - [Minio `v5.4.0`](https://github.com/ai-solution-eng/frameworks/tree/main/minio)
- Install the following MCP Servers:
    - [EzPresto MCP Server](mcp-servers/ezpresto-server/)
    - [Filesystem MCP Server](mcp-servers/filesystem-server/)
    - [S3 MCP Server](mcp-servers/s3-server/)
- Install the following tools:
    - [MCP Inspector](tools/mcp-inspector/)
    - [Guardrail Model Controller](tools/controller/)
- Create an [`.env` file](../.env) by copying the provided [`sample.env`](../.sample_env):
    - `cp .sample_env .env`
    - Store the domain for the target AIE environment in the `AIE_DOMAIN` variable
- [Follow the guardrail configuration](guardrails/README.md)
- After setting up the models, finish the Arize Phoenix configuration:
    - Create and store the Arize Phoenix API Key in the `PHOENIX_API_KEY` variable:
        - Navigate to Arize Phoenix UI ([show in `PHOENIX_ISTIO_ENDPOINT`](./.env) or in the Arize Phoenix frameworks tab)
        - Settings -> General -> System Keys -> Create new System Key
        - Settings -> Models -> Add Model -> Add model name (`Llama-3.1-8B-Instruct` for example)