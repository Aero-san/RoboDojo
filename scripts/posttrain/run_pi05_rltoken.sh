#!/usr/bin/env bash
# Launch offline Pi0.5 RL Token/WCM-actor training in the WCM environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
ACTOR_TRAIN_GPUS="${ACTOR_TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "WCM Python environment not found: ${PYTHON_BIN}" >&2
  echo "Run scripts/posttrain/install_wcm.sh or set PYTHON_BIN explicitly." >&2
  exit 1
fi

ACTOR_TRAIN_GPUS="${ACTOR_TRAIN_GPUS//[[:space:]]/}"
if [[ ! "${ACTOR_TRAIN_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ACTOR_TRAIN_GPUS must be a comma-separated list of numeric GPU ids" >&2
  exit 2
fi
IFS=',' read -r -a ACTOR_GPU_IDS <<< "${ACTOR_TRAIN_GPUS}"
declare -A SEEN_ACTOR_GPU_IDS=()
for gpu_id in "${ACTOR_GPU_IDS[@]}"; do
  if [[ -n "${SEEN_ACTOR_GPU_IDS[${gpu_id}]:-}" ]]; then
    echo "ACTOR_TRAIN_GPUS contains duplicate GPU id: ${gpu_id}" >&2
    exit 2
  fi
  SEEN_ACTOR_GPU_IDS[${gpu_id}]=1
done
ACTOR_WORLD_SIZE="${#ACTOR_GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="${ACTOR_TRAIN_GPUS}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TASK_NAME="${TASK_NAME:-${WCM_TASK_NAME:-}}"
ARGS=("$@")
if [[ -n "${TASK_NAME}" ]]; then
  has_task_arg=0
  for arg in "${ARGS[@]}"; do
    if [[ "${arg}" == "--task" || "${arg}" == --task=* ]]; then
      has_task_arg=1
      break
    fi
  done
  if [[ "${has_task_arg}" == "0" ]]; then ARGS+=(--task "${TASK_NAME}"); fi
fi
ARGS+=(--expected-world-size "${ACTOR_WORLD_SIZE}")
if (( ACTOR_WORLD_SIZE == 1 )); then
  echo "[Pi0.5 actor] GPUs=${ACTOR_TRAIN_GPUS} world_size=1"
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_pi05_rltoken.py" "${ARGS[@]}"
fi
echo "[Pi0.5 actor] GPUs=${ACTOR_TRAIN_GPUS} world_size=${ACTOR_WORLD_SIZE} (DDP)"
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${ACTOR_WORLD_SIZE}" \
  "${SCRIPT_DIR}/train_pi05_rltoken.py" "${ARGS[@]}"
