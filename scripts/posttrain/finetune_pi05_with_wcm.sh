#!/usr/bin/env bash
# Prepare a WCM-selected RoboDojo LeRobot dataset and launch OpenPI Pi0.5 FT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/gpu_reservation.sh"
install_gpu_reservation_exit_trap
PI_DIR="${ROOT_DIR}/XPolicyLab/policy/Pi_05"
PI_PYTHON_BIN="${PYTHON_BIN:-${PI_DIR}/openpi/.venv/bin/python}"
WCM_PYTHON_BIN="${WCM_PYTHON_BIN:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
DATASET_ROOT="${WCM_DATASET_ROOT:-${ROOT_DIR}/data/RoboDojo_lerobot_v21_video}"
BENCH_NAME="${BENCH_NAME:-RoboDojo}"
RUN_NAME="${RUN_NAME:-wcm_pi05}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0,1,2,3}"
TASK_NAME="${TASK_NAME:-${WCM_TASK_NAME:-}}"
PI05_FINETUNE_MODE="${PI05_FINETUNE_MODE:-action_expert}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      [[ $# -ge 2 ]] || { echo "--task requires a value" >&2; exit 2; }
      TASK_NAME="$2"
      shift 2
      ;;
    --finetune-mode)
      [[ $# -ge 2 ]] || { echo "--finetune-mode requires a value" >&2; exit 2; }
      PI05_FINETUNE_MODE="$2"
      shift 2
      ;;
    --wcm-checkpoint)
      [[ $# -ge 2 ]] || { echo "--wcm-checkpoint requires a value" >&2; exit 2; }
      WCM_CHECKPOINT="$2"
      shift 2
      ;;
    --repo-id)
      [[ $# -ge 2 ]] || { echo "--repo-id requires a value" >&2; exit 2; }
      PI05_REPO_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done
TASK_SLUG=""
if [[ -n "${TASK_NAME}" ]]; then
  TASK_SLUG=$(printf '%s' "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//' | cut -c1-80)
fi
if [[ -n "${PI05_REPO_ID:-}" ]]; then
  REPO_ID="${PI05_REPO_ID}"
elif [[ -n "${TASK_SLUG}" ]]; then
  REPO_ID="${BENCH_NAME}-${RUN_NAME}-${ENV_CFG_TYPE}-${ACTION_TYPE}-${TASK_SLUG}"
else
  REPO_ID="${BENCH_NAME}-${RUN_NAME}-${ENV_CFG_TYPE}-${ACTION_TYPE}"
fi
LABELS="${WCM_SUCCESS_LABELS:-}"

if [[ -n "${TASK_SLUG}" && -z "${WCM_CHECKPOINT:-}" ]]; then
  WCM_CHECKPOINT="${ROOT_DIR}/outputs/wcm/robodojo_pi05/${TASK_SLUG}/deploy.pt"
else
  WCM_CHECKPOINT="${WCM_CHECKPOINT:-${ROOT_DIR}/outputs/wcm/robodojo_pi05/deploy.pt}"
fi

[[ -x "${PI_PYTHON_BIN}" ]] || { echo "Pi0.5 Python not found: ${PI_PYTHON_BIN}" >&2; exit 1; }
if [[ ! -x "${WCM_PYTHON_BIN}" ]]; then
  WCM_PYTHON_BIN="${PI_PYTHON_BIN}"
fi
[[ -f "${WCM_CHECKPOINT}" ]] || {
  echo "WCM checkpoint not found: ${WCM_CHECKPOINT}" >&2
  echo "Train it with scripts/posttrain/run_wcm.sh first." >&2
  exit 1
}
[[ -f "${DATASET_ROOT}/meta/info.json" ]] || { echo "Dataset not found: ${DATASET_ROOT}" >&2; exit 1; }
export WCM_VIDEO_DECODER="${WCM_VIDEO_DECODER:-pyav}"

if [[ -z "${LABELS}" ]]; then
  LABELS="${ROOT_DIR}/outputs/posttrain/${RUN_NAME}${TASK_SLUG:+_${TASK_SLUG}}_wcm_labels.json"
  echo "[Pi_05/WCM] ranking episodes with WCM (fraction=${WCM_SELECT_FRACTION:-1.0})"
  SELECT_ARGS=(
    --wcm-checkpoint "${WCM_CHECKPOINT}"
    --dataset-root "${DATASET_ROOT}"
    --output "${LABELS}"
    --fraction "${WCM_SELECT_FRACTION:-1.0}"
    --device "${WCM_DEVICE:-cuda}"
    --batch-size "${WCM_SELECT_BATCH_SIZE:-8}"
    --num-workers "${WCM_SELECT_NUM_WORKERS:-2}"
  )
  if [[ -n "${TASK_NAME}" ]]; then SELECT_ARGS+=(--task "${TASK_NAME}"); fi
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/select_wcm_episodes.py" \
    "${SELECT_ARGS[@]}"
fi

CONVERT_ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --repo-id "${REPO_ID}"
  --mode "${OPENPI_DATA_MODE:-image}"
)
CONVERT_ARGS+=(--episode-labels "${LABELS}")
if [[ -n "${TASK_NAME}" ]]; then CONVERT_ARGS+=(--task "${TASK_NAME}"); fi
echo "[Pi_05/WCM] preparing ${REPO_ID}"
cd "${ROOT_DIR}"
start_gpu_reservation "${GPU_ID}" "${WCM_PYTHON_BIN}" "WCM-selected Pi0.5 dataset conversion"
"${PI_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pi05_dataset.py" "${CONVERT_ARGS[@]}"

echo "[Pi_05/WCM] fine-tuning with OpenPI"
GPU_COUNT=$(awk -F',' '{print NF}' <<<"${GPU_ID}")
FSDP_DEVICES="${OPENPI_FSDP_DEVICES:-$(( GPU_COUNT < 2 ? 1 : 2 ))}"
CKPT_SETTING="${BENCH_NAME}-${RUN_NAME}-${ENV_CFG_TYPE}-${ACTION_TYPE}-${SEED}"
CKPT_DIR="${PI_DIR}/checkpoints/${CKPT_SETTING}"
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_ARGS=(
  --openpi-root "${PI_DIR}/openpi"
  --train-config-name "${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
  --repo-id "${REPO_ID}"
  --exp-name "${CKPT_SETTING}"
  --checkpoint-dir "${CKPT_DIR}"
  --finetune-mode "${PI05_FINETUNE_MODE}"
  --seed "${SEED}"
  --fsdp-devices "${FSDP_DEVICES}"
)
for option in batch-size num-workers num-train-steps save-interval log-interval; do
  variable="OPENPI_${option^^}"
  variable="${variable//-/_}"
  if [[ -n "${!variable:-}" ]]; then TRAIN_ARGS+=("--${option}" "${!variable}"); fi
done
if [[ -n "${OPENPI_LEARNING_RATE:-}" ]]; then TRAIN_ARGS+=(--learning-rate "${OPENPI_LEARNING_RATE}"); fi
if [[ -n "${OPENPI_WARMUP_STEPS:-}" ]]; then TRAIN_ARGS+=(--warmup-steps "${OPENPI_WARMUP_STEPS}"); fi
if [[ -n "${OPENPI_DECAY_LR:-}" ]]; then TRAIN_ARGS+=(--decay-lr "${OPENPI_DECAY_LR}"); fi
if [[ -n "${OPENPI_WEIGHT_DECAY:-}" ]]; then TRAIN_ARGS+=(--weight-decay "${OPENPI_WEIGHT_DECAY}"); fi
if [[ -n "${OPENPI_CLIP_GRADIENT_NORM:-}" ]]; then TRAIN_ARGS+=(--clip-gradient-norm "${OPENPI_CLIP_GRADIENT_NORM}"); fi
if [[ -n "${OPENPI_ACTION_EXPERT_VARIANT:-}" ]]; then TRAIN_ARGS+=(--action-expert-variant "${OPENPI_ACTION_EXPERT_VARIANT}"); fi
if [[ -n "${OPENPI_PALIGEMMA_VARIANT:-}" ]]; then TRAIN_ARGS+=(--paligemma-variant "${OPENPI_PALIGEMMA_VARIANT}"); fi
if [[ "${OPENPI_WANDB_ENABLED:-1}" == "0" ]]; then TRAIN_ARGS+=(--disable-wandb); fi
if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then TRAIN_ARGS+=(--resume); fi
cd "${PI_DIR}/openpi"
stop_gpu_reservation
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/train_pi05.py" "${TRAIN_ARGS[@]}"
