#!/usr/bin/env bash
# End-to-end release automation for Multimodal RAG:
#   build & push the MCP server image, bump the three helm charts,
#   package them, and prune stale chart archives.
# Usage: ./automation.sh <VS>   (e.g. ./automation.sh 2.3.0)
set -euo pipefail

VERSION="${1:?Usage: $0 <version> (e.g. 2.3.0)}"

# 'helm-p' is the user's standalone script at ~/.local/bin/helm-p: a bare
# call already packages every helm*/ chart directory, so the explicit
# scale lines below are redundant-but-harmless (prune_charts.py keeps the
# newest archive per chart).

echo "==> Building & pushing ghcr.io/ai-solution-eng/multimodal-rag-mcp:v${VERSION}"
docker buildx build -t "ghcr.io/ai-solution-eng/multimodal-rag-mcp:v${VERSION}" -f docker/Dockerfile . --push

echo "==> Bumping chart versions to ${VERSION}"
./bump_version.sh "${VERSION}"

echo "==> Packaging charts"
helm-p
helm-p helm-scale-medium
helm-p helm-scale-large

echo "==> Pruning old chart archives"
python prune_charts.py

echo "==> Done: release v${VERSION}"
