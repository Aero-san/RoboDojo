#!/usr/bin/env bash
# Train or evaluate the upstream WCM on RoboDojo's LeRobot-v2.1 video export.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WCM_ROOT="${ROOT_DIR}/external_dependencies/WCM"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${WCM_ROOT}/.venv/bin/python}"
MODE="${MODE:-train}"
CONFIG="${WCM_CONFIG:-${ROOT_DIR}/configs/wcm/robodojo_pi05.yaml}"
DATASET_ROOT="${WCM_DATASET_ROOT:-${ROOT_DIR}/data/RoboDojo_lerobot_v21_video}"
OUTPUT_DIR="${WCM_OUTPUT_DIR:-${ROOT_DIR}/outputs/wcm/robodojo_pi05}"
CHECKPOINT="${WCM_CHECKPOINT:-${OUTPUT_DIR}/deploy.pt}"
EVAL_OUTPUT_DIR="${WCM_EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WCM_EPOCHS="${WCM_EPOCHS:-1}"
TASK_NAME="${TASK_NAME:-${WCM_TASK_NAME:-}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires a value" >&2; exit 2; }
      TASK_NAME="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires a value" >&2; exit 2; }
      MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${TASK_NAME}" && -z "${WCM_OUTPUT_DIR:-}" ]]; then
  TASK_SLUG=$(printf '%s' "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//' | cut -c1-80)
  OUTPUT_DIR="${OUTPUT_DIR}/${TASK_SLUG}"
  if [[ -z "${WCM_CHECKPOINT:-}" ]]; then CHECKPOINT="${OUTPUT_DIR}/deploy.pt"; fi
  if [[ -z "${WCM_EVAL_OUTPUT_DIR:-}" ]]; then EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"; fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -n "${PYTHON_BIN_FALLBACK:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK}"
  else
    echo "WCM Python environment not found: ${PYTHON_BIN}" >&2
    echo "Run scripts/posttrain/install_wcm.sh or set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi
[[ -d "${WCM_ROOT}/world_critic" ]] || {
  echo "WCM submodule is missing: ${WCM_ROOT}" >&2
  echo "Run: git submodule update --init --recursive external_dependencies/WCM" >&2
  exit 1
}
[[ -f "${CONFIG}" ]] || { echo "WCM config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || {
  echo "RoboDojo dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must be a comma-separated list of numeric GPU ids" >&2
  exit 2
fi
IFS=',' read -r -a WCM_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
declare -A WCM_SEEN_GPU_IDS=()
for gpu_id in "${WCM_GPU_IDS[@]}"; do
  if [[ -n "${WCM_SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES contains duplicate GPU id: ${gpu_id}" >&2
    exit 2
  fi
  WCM_SEEN_GPU_IDS[${gpu_id}]=1
done
WCM_WORLD_SIZE="${#WCM_GPU_IDS[@]}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT_DIR}:${WCM_ROOT}:${PYTHONPATH:-}"
export WCM_DATASET_ROOT="${DATASET_ROOT}"
export WCM_OUTPUT_DIR="${OUTPUT_DIR}"
if [[ -n "${TASK_NAME}" ]]; then
  export WCM_TASK_NAME="${TASK_NAME}"
else
  unset WCM_TASK_NAME 2>/dev/null || true
fi
export WCM_EXPECTED_WORLD_SIZE="${WCM_WORLD_SIZE}"
export WCM_PRECISION="${WCM_PRECISION:-bf16}"
export WCM_NUM_WORKERS="${WCM_NUM_WORKERS:-4}"
export WCM_PER_DEVICE_BATCH_SIZE="${WCM_PER_DEVICE_BATCH_SIZE:-16}"
export WCM_EVAL_BATCH_SIZE="${WCM_EVAL_BATCH_SIZE:-32}"
export WCM_FAILURE_PENALTY="${WCM_FAILURE_PENALTY:-300}"
export WCM_GAMMA="${WCM_GAMMA:-1.0}"
export WCM_ASSUME_SUCCESS="${WCM_ASSUME_SUCCESS:-1}"
export WCM_VIDEO_DECODER="${WCM_VIDEO_DECODER:-pyav}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${WCM_SUCCESS_LABELS:-}" ]]; then
  export WCM_SUCCESS_LABELS
else
  unset WCM_SUCCESS_LABELS 2>/dev/null || true
fi
if [[ -n "${WCM_EPOCHS:-}" ]]; then export WCM_EPOCHS; fi
if [[ -n "${WCM_RESUME:-}" ]]; then export WCM_RESUME; fi
if [[ -n "${WCM_INIT_CHECKPOINT:-}" ]]; then export WCM_INIT_CHECKPOINT; fi

run_wcm() {
  local args
  if [[ "${MODE}" == "train" ]]; then
    args=(train --config "${CONFIG}")
  else
    [[ -f "${CHECKPOINT}" ]] || {
      echo "WCM checkpoint not found: ${CHECKPOINT}" >&2
      echo "Set WCM_CHECKPOINT or run MODE=train first." >&2
      exit 1
    }
    args=(
      eval
      --checkpoint "${CHECKPOINT}"
      --output-dir "${EVAL_OUTPUT_DIR}"
      --split "${WCM_SPLIT:-val}"
      --batch-size "${WCM_EVAL_BATCH_SIZE:-32}"
      --num-workers "${WCM_NUM_WORKERS:-4}"
      --expected-world-size "${WCM_WORLD_SIZE}"
      --episode-curves
    )
  fi
  if [[ "${WCM_WORLD_SIZE}" == "1" ]]; then
    echo "[WCM] mode=${MODE} GPUs=${CUDA_VISIBLE_DEVICES} world_size=1"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_wcm.py" "${args[@]}"
  else
    echo "[WCM] mode=${MODE} GPUs=${CUDA_VISIBLE_DEVICES} world_size=${WCM_WORLD_SIZE} (DDP)"
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${WCM_WORLD_SIZE}" "${SCRIPT_DIR}/run_wcm.py" "${args[@]}"
  fi
}

case "${MODE}" in
  train|eval) run_wcm ;;
  all) MODE=train run_wcm; MODE=eval run_wcm ;;
  *) echo "MODE must be train, eval, or all (got ${MODE})" >&2; exit 2 ;;
esac
