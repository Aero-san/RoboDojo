#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Config parsing belongs to RoboDojo and must not depend on a policy-specific
# environment. Policy/data/WCM interpreters are selected from the YAML later.
export POSTTRAIN_CONFIG_PYTHON="${POSTTRAIN_CONFIG_PYTHON:-${ROOT_DIR}/../miniconda3/envs/RoboDojo/bin/python}"

exec bash scripts/posttrain/run_recap.sh \
  --config "${RECAP_CONFIG:-configs/posttrain/remote_training.yaml}" \
  "$@"
