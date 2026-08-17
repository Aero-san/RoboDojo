#!/usr/bin/env bash
# Create/update the Python environment declared by the WCM submodule.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WCM_ROOT="${ROOT_DIR}/external_dependencies/WCM"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
}
[[ -f "${WCM_ROOT}/pyproject.toml" ]] || {
  echo "WCM submodule is not initialized: ${WCM_ROOT}" >&2
  echo "Run: git submodule update --init --recursive external_dependencies/WCM" >&2
  exit 1
}

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"
uv sync --project "${WCM_ROOT}"
echo "WCM environment ready: ${WCM_ROOT}/.venv/bin/python"
