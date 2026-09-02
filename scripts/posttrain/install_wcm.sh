#!/usr/bin/env bash
# Create/update the Python environment declared by the vendored WCM source.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WCM_ROOT="${ROOT_DIR}/external_dependencies/WCM"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 1
}
[[ -f "${WCM_ROOT}/pyproject.toml" ]] || {
  echo "Vendored WCM source is missing: ${WCM_ROOT}" >&2
  echo "Re-clone RoboDojo so external_dependencies/WCM is present." >&2
  exit 1
}

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.cache/uv}"
mkdir -p "${UV_CACHE_DIR}"
uv sync --project "${WCM_ROOT}"
echo "WCM environment ready: ${WCM_ROOT}/.venv/bin/python"
