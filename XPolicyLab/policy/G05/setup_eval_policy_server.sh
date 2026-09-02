#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
task_name=${2:?task_name required}
ckpt_name=${3:?ckpt_name required}
env_cfg_type=${4:?env_cfg_type required}
action_type=${5:?action_type required}
seed=${6:?seed required}
policy_gpu_id=${7:?policy_gpu_id required}
policy_env=${8:-base}
policy_server_port=${9:?policy_server_port required}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

G05_ROOT="${G05_ROOT:-/share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA}"

resolve_python_bin() {
  local requested="${G05_PYTHON:-${policy_env}}"
  local candidate=""
  local conda_prefix=""
  local g05_venv="${G05_ROOT}/.venv/bin/python"

  # The vendored G0.5 checkout is a complete uv virtualenv.  The CLI value
  # "GalaxeaVLA" names this checkout in the documented command, so resolve it
  # to the interpreter that belongs to the checkout instead of passing the
  # literal string to env(1).
  if [[ "${requested}" == "GalaxeaVLA" || "${requested}" == "$(basename "${G05_ROOT}")" || "${requested}" == ".venv" ]]; then
    if [[ -x "${g05_venv}" ]]; then
      printf '%s\n' "${g05_venv}"
      return 0
    fi
  fi

  # Accept either an environment directory or an explicit Python executable.
  if [[ "${requested}" == /* && -x "${requested}/bin/python" ]]; then
    printf '%s\n' "${requested}/bin/python"
    return 0
  fi
  if [[ -x "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi
  if command -v "${requested}" >/dev/null 2>&1; then
    command -v "${requested}"
    return 0
  fi

  # Bare names are also accepted as conda environment names when conda is
  # available in the non-interactive shell used by robodojo.sh.
  if command -v conda >/dev/null 2>&1; then
    conda_prefix="$(conda env list 2>/dev/null | awk -v name="${requested}" '$1 == name {print $NF; exit}')"
    candidate="${conda_prefix}/bin/python"
    if [[ -n "${conda_prefix}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  echo "[SERVER][ERROR] cannot resolve G05 Python for --policy-env/G05_PYTHON='${requested}'" >&2
  echo "[SERVER][ERROR] pass GalaxeaVLA (uses ${g05_venv}), an environment directory, or an executable Python path" >&2
  return 1
}

if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G0.5 repo not found: ${G05_ROOT}" >&2
  exit 3
fi

PYTHON_BIN="$(resolve_python_bin)"

resolve_ckpt_path() {
  local raw="$1"
  local run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
  local candidates=()

  if [[ -n "${G05_CKPT_PATH:-}" ]]; then
    candidates+=("${G05_CKPT_PATH}")
  fi
  if [[ "${raw}" == /* ]]; then
    candidates+=("${raw}")
  elif [[ "${raw}" == */* ]]; then
    candidates+=("${SCRIPT_DIR}/${raw}")
  else
    candidates+=(
      "${SCRIPT_DIR}/checkpoints/${run_dir_name}"
      "${SCRIPT_DIR}/checkpoints/${raw}"
      "${G05_ROOT}/outputs/${run_dir_name}"
      "/efm-nas/efm-nas/group-jt/haoyu.zhang/experiments_compare/results/robodojo/g05/github_robodojo_arx_x5_joint/robodojo_arx_x5_joint/${run_dir_name}"
    )
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" || -d "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
  done

  echo "No checkpoint found. Set G05_CKPT_PATH to a G0.5 .pt file or run directory." >&2
  echo "Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  return 1
}

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 adapter currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi

ckpt_path="$(resolve_ckpt_path "${ckpt_name}")"

if ! action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}" 2>/dev/null); then
  if [[ "${env_cfg_type}" == "arx_x5" ]]; then
    action_dim=14
  else
    echo "Could not resolve action_dim for ${env_cfg_type}; check ${BENCH_ROOT}/env_cfg." >&2
    exit 4
  fi
fi

echo "[SERVER] policy=${policy_name} task=${task_name} host=${policy_server_host} port=${policy_server_port}"
echo "[SERVER] policy_env=${policy_env}"
echo "[SERVER] python=${PYTHON_BIN}"
echo "[SERVER] g05_root=${G05_ROOT}"
echo "[SERVER] ckpt_path=${ckpt_path}"
echo "[SERVER] action_dim=${action_dim}"

cd "${G05_ROOT}"

exec env \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${G05_ROOT}/src:${G05_ROOT}:${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      host="${policy_server_host}" \
      port="${policy_server_port}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      action_dim="${action_dim}" \
      ckpt_path="${ckpt_path}" \
      g05_root="${G05_ROOT}"
