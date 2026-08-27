#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

exec bash scripts/posttrain/run_pi05_recap.sh \
  --config configs/posttrain/remote_6226.yaml \
  "$@"
