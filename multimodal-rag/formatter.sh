#!/usr/bin/env bash
# Unified formatter for PCAI repos.
#
# Runs ruff format / ruff check --fix / mypy on every src/* package and,
# when present, on tests/.
#
# Usage: ./formatter.sh [<target-repo-root>]
#   <target-repo-root>  defaults to this script's directory, so the script
#   can be hardlinked into a repo root and run as ./formatter.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 1 ]]; then
  ROOT="$(cd "$1" && pwd)"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: '$ROOT' is not a directory" >&2
  exit 1
fi

# Shared lint config is canonical (ruff.toml / mypy.ini), hardlinked into
# each repo root.  Require ruff.toml so we never reflow with ruff's defaults.
if [[ ! -f "$ROOT/ruff.toml" ]]; then
  echo "SKIP  $ROOT (no ruff.toml found; refusing to reformat shared utils with defaults)"
  exit 0
fi

shopt -s nullglob
packages=("$ROOT"/src/*/)
shopt -u nullglob

targets=()
add_target() { # add_target <dir>  — only if it contains Python files
  if find "$1" -maxdepth 3 -name "*.py" -print -quit 2>/dev/null | grep -q .; then
    targets+=("${1%/}")
  else
    echo "SKIP  $1 (no Python files)"
  fi
}
for pkg in "${packages[@]}"; do
  base="$(basename "$pkg")"
  case "$base" in
    .mypy_cache | __pycache__ | build) continue ;;
  esac
  add_target "${pkg%/}"
done
if [[ -d "$ROOT/tests" ]]; then
  add_target "$ROOT/tests"
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No src/*/ or tests/ found under $ROOT" >&2
  exit 1
fi

failed=0
for t in "${targets[@]}"; do
  echo "== $t =="
  ruff format "$t" || failed=1
  ruff check --fix "$t" || failed=1
  echo ''
  mypy --config-file "$ROOT/mypy.ini" "$t" || failed=1
  echo ''
done

if [[ $failed -ne 0 ]]; then
  echo "FORMATTER: errors found (see above)" >&2
  exit 1
fi
echo "FORMATTER: all clean"