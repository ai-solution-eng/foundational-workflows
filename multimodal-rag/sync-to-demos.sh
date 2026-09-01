#!/usr/bin/env bash
#
# sync-to-demos.sh — publish multimodal-rag artifacts to the ai-solution-demos repo.
#
# Copies from this repo (foundational-workflows/multimodal-rag):
#   helm/                          -> <demos>/multimodal-rag/helm/          (full replace, drops stale chart)
#   rag-mcp-server-3.2.0.tar.gz    -> <demos>/multimodal-rag/                (old packaged chart removed)
#   src/multimodal_rag/            -> <demos>/multimodal-rag/docker/src/      (caches excluded)
#   docker/Dockerfile              -> <demos>/multimodal-rag/docker/Dockerfile.txt
#   docker/requirements.txt        -> <demos>/multimodal-rag/docker/requirements.txt
#   openwebui_extension/filter.py  -> <demos>/multimodal-rag/extensions/openwebui-filter/filter.py
#
# The demo repo's README.md is maintained separately there and is never touched.
#
# Usage: ./sync-to-demos.sh [-n|--dry-run]

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="${HOME}/Code/HPE/pcai-solutions/ai-solution-demos/multimodal-rag"

die() { echo "error: $*" >&2; exit 1; }

[[ -d "${SRC}/helm" ]]             || die "source chart dir not found: ${SRC}/helm"
[[ -f "${SRC}/rag-mcp-server-3.2.0.tar.gz" ]] || die "source packaged chart not found: ${SRC}/rag-mcp-server-3.2.0.tar.gz"
[[ -d "${DEMO}" ]]                 || die "demo dir missing: ${DEMO} (clone ai-solution-demos first)"
[[ -d "${DEMO}/helm" ]]            || die "demo chart dir missing: ${DEMO}/helm"
[[ -d "${DEMO}/docker" ]]          || die "demo docker dir missing: ${DEMO}/docker"
[[ -d "${DEMO}/extensions" ]]      || die "demo extensions dir missing: ${DEMO}/extensions"

run() {
  echo "+ $*"
  (( DRY_RUN )) || "$@"
}

echo "Syncing multimodal-rag -> ${DEMO}"
(( DRY_RUN )) && echo "(dry run — no changes will be made)"

# 1) Helm chart: full replace so stale templates/values (e.g. legacy helm/template/) cannot survive.
run rm -rf "${DEMO}/helm"
run mkdir -p "${DEMO}/helm"
run cp -R "${SRC}/helm/." "${DEMO}/helm/"

# 2) Packaged chart: drop every old tarball, copy in the current one.
run rm -f "${DEMO}"/rag-mcp-server-*.tar.gz
run cp "${SRC}/rag-mcp-server-3.2.0.tar.gz" "${DEMO}/"

# 3) Source code, without caches/venvs.
run rm -rf "${DEMO}/docker/src/multimodal_rag"
run mkdir -p "${DEMO}/docker/src"
run rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.mypy_cache/' --exclude '.ruff_cache/' --exclude '.pytest_cache/' \
  "${SRC}/src/multimodal_rag" "${DEMO}/docker/src/"

# 4) Docker build files (source Dockerfile ships as Dockerfile.txt in the demo repo).
run cp "${SRC}/docker/Dockerfile"       "${DEMO}/docker/Dockerfile.txt"
run cp "${SRC}/docker/requirements.txt" "${DEMO}/docker/requirements.txt"

# 5) Open WebUI extension: only the filter.
run mkdir -p "${DEMO}/extensions/openwebui-filter"
run cp "${SRC}/openwebui_extension/filter.py" "${DEMO}/extensions/openwebui-filter/filter.py"

(( DRY_RUN )) && echo "Dry run complete — nothing was modified." \
              || echo "Done. Review with: git -C \"${DEMO%/multimodal-rag}\" status"
