#!/usr/bin/env bash
# Iterated off-policy RoboDojo post-training with WCM and RECAP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTIVE_ROLLOUT_PID=""
ACTIVE_ROLLOUT_MONITOR_PID=""

kill_process_tree() {
  local parent_pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    kill_process_tree "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
  kill -TERM "${parent_pid}" 2>/dev/null || true
}

interrupt_rollout() {
  trap - INT TERM
  if [[ -n "${ACTIVE_ROLLOUT_PID}" ]]; then
    echo "[RECAP rollout] stopping active worker" >&2
    kill_process_tree "${ACTIVE_ROLLOUT_PID}"
    wait "${ACTIVE_ROLLOUT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ACTIVE_ROLLOUT_MONITOR_PID}" ]]; then
    kill -TERM "${ACTIVE_ROLLOUT_MONITOR_PID}" 2>/dev/null || true
    wait "${ACTIVE_ROLLOUT_MONITOR_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap interrupt_rollout INT TERM

PI_DIR="${ROOT_DIR}/XPolicyLab/policy/Pi_05"
PI_PYTHON_BIN="${PI_PYTHON_BIN:-${PI_DIR}/openpi/.venv/bin/python}"
WCM_PYTHON_BIN="${WCM_PYTHON_BIN:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
POLICY_DIR="${POLICY_DIR:-${PI_DIR}}"
POLICY_ENV="${POLICY_ENV:-openpi}"
EVAL_ENV="${EVAL_ENV:-RoboDojo}"
TASK_NAME="${TASK_NAME:-}"
DEMO_ROOT="${DEMO_ROOT:-${ROOT_DIR}/data/RoboDojo_lerobot_v21_video}"
INITIAL_POLICY_CHECKPOINT="${INITIAL_POLICY_CHECKPOINT:-}"
INITIAL_WCM_CHECKPOINT="${INITIAL_WCM_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/recap}"
ITERATIONS="${RECAP_ITERATIONS:-1}"
ROLLOUT_EPISODES="${RECAP_ROLLOUT_EPISODES:-40}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
WCM_TRAIN_GPUS="${WCM_TRAIN_GPUS:-}"
POLICY_GPU="${POLICY_GPU:-}"
ENV_GPU="${ENV_GPU:-}"
SEED="${SEED:-0}"
TRAIN_CONFIG="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
FINETUNE_MODE="${PI05_FINETUNE_MODE:-action_expert_lora}"
GAMMA="${RECAP_GAMMA:-1.0}"
BETA="${RECAP_BETA:-1.0}"
FAILURE_PENALTY="${WCM_FAILURE_PENALTY:-300}"

usage() {
  cat <<'EOF'
Usage: bash scripts/posttrain/run_pi05_recap.sh [options]

Required:
  --task TASK                         RoboDojo task slug or instruction
  --initial-policy-checkpoint PATH    Initial Pi0.5 SFT checkpoint

Options:
  --demo-root PATH                    Successful SFT LeRobot-v2.1 dataset
  --initial-wcm-checkpoint PATH       Warm-start WCM model weights
  --output-root PATH                  Run output root (default: outputs/recap)
  --iterations N                      Policy-improvement iterations (default: 3)
  --rollout-episodes N                Simulator episodes per iteration (default: 50)
  --env-cfg NAME                      RoboDojo robot/environment config
  --action-type joint|ee              Policy action representation
  --finetune-mode MODE                full/action_expert/*_lora mode
  --train-gpus IDS                    Pi0.5 training GPUs, e.g. 0,1,2,3
  --wcm-train-gpus IDS                WCM DDP GPUs (default: --train-gpus)
  --policy-gpu ID                     Rollout policy-server GPU
  --env-gpu ID                        Rollout Isaac Sim GPU

Additional optimizer, device, WCM, and RECAP controls are documented in
configs/posttrain/pi05_recap.env.example.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK_NAME="$2"; shift 2 ;;
    --demo-root) DEMO_ROOT="$2"; shift 2 ;;
    --initial-policy-checkpoint) INITIAL_POLICY_CHECKPOINT="$2"; shift 2 ;;
    --initial-wcm-checkpoint) INITIAL_WCM_CHECKPOINT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --rollout-episodes) ROLLOUT_EPISODES="$2"; shift 2 ;;
    --env-cfg) ENV_CFG_TYPE="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --finetune-mode) FINETUNE_MODE="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --wcm-train-gpus) WCM_TRAIN_GPUS="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${TASK_NAME}" ]] || { echo "--task is required" >&2; exit 2; }
[[ -n "${INITIAL_POLICY_CHECKPOINT}" ]] || { echo "--initial-policy-checkpoint is required" >&2; exit 2; }
[[ -x "${PI_PYTHON_BIN}" ]] || { echo "Pi0.5 Python not found: ${PI_PYTHON_BIN}" >&2; exit 1; }
[[ -x "${WCM_PYTHON_BIN}" ]] || { echo "WCM Python not found: ${WCM_PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DEMO_ROOT}/meta/info.json" ]] || { echo "Demo dataset not found: ${DEMO_ROOT}" >&2; exit 1; }
[[ -d "${INITIAL_POLICY_CHECKPOINT}" ]] || { echo "Initial Pi0.5 checkpoint not found: ${INITIAL_POLICY_CHECKPOINT}" >&2; exit 1; }
if [[ -n "${INITIAL_WCM_CHECKPOINT}" && ! -f "${INITIAL_WCM_CHECKPOINT}" ]]; then
  echo "Initial WCM checkpoint not found: ${INITIAL_WCM_CHECKPOINT}" >&2
  exit 1
fi
[[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || { echo "--iterations must be positive" >&2; exit 2; }
[[ "${ROLLOUT_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "--rollout-episodes must be positive" >&2; exit 2; }

parse_gpu_ids() {
  local raw_ids="${1//[[:space:]]/}"
  local output_name="$2"
  [[ "${raw_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "${output_name} must be a comma-separated list of numeric GPU ids" >&2
    exit 2
  }
  local -a parsed_ids
  local -A seen_ids=()
  local gpu_id
  IFS=',' read -r -a parsed_ids <<< "${raw_ids}"
  for gpu_id in "${parsed_ids[@]}"; do
    [[ -z "${seen_ids[${gpu_id}]:-}" ]] || {
      echo "${output_name} contains duplicate GPU id: ${gpu_id}" >&2
      exit 2
    }
    seen_ids[${gpu_id}]=1
  done
  local -n output_ids="${output_name}"
  output_ids=("${parsed_ids[@]}")
}

declare -a POLICY_TRAIN_GPU_IDS WCM_TRAIN_GPU_IDS
parse_gpu_ids "${TRAIN_GPUS}" POLICY_TRAIN_GPU_IDS
WCM_TRAIN_GPUS="${WCM_TRAIN_GPUS:-${TRAIN_GPUS}}"
parse_gpu_ids "${WCM_TRAIN_GPUS}" WCM_TRAIN_GPU_IDS
TRAIN_GPUS=$(IFS=','; echo "${POLICY_TRAIN_GPU_IDS[*]}")
WCM_TRAIN_GPUS=$(IFS=','; echo "${WCM_TRAIN_GPU_IDS[*]}")
GPU_COUNT="${#POLICY_TRAIN_GPU_IDS[@]}"
WCM_GPU_COUNT="${#WCM_TRAIN_GPU_IDS[@]}"
POLICY_GPU="${POLICY_GPU:-${POLICY_TRAIN_GPU_IDS[0]}}"
if [[ -z "${ENV_GPU}" ]]; then
  if (( GPU_COUNT > 1 )); then
    ENV_GPU="${POLICY_TRAIN_GPU_IDS[1]}"
  else
    ENV_GPU="${POLICY_TRAIN_GPU_IDS[0]}"
  fi
fi
[[ "${POLICY_GPU}" =~ ^[0-9]+$ ]] || { echo "--policy-gpu must be one numeric GPU id" >&2; exit 2; }
[[ "${ENV_GPU}" =~ ^[0-9]+$ ]] || { echo "--env-gpu must be one numeric GPU id" >&2; exit 2; }
FSDP_DEVICES="${OPENPI_FSDP_DEVICES:-$(( GPU_COUNT < 2 ? 1 : 2 ))}"
[[ "${FSDP_DEVICES}" =~ ^[1-9][0-9]*$ ]] || { echo "OPENPI_FSDP_DEVICES must be positive" >&2; exit 2; }
(( FSDP_DEVICES <= GPU_COUNT && GPU_COUNT % FSDP_DEVICES == 0 )) || {
  echo "OPENPI_FSDP_DEVICES=${FSDP_DEVICES} must divide the ${GPU_COUNT} Pi0.5 training GPUs" >&2
  exit 2
}
if [[ -n "${OPENPI_BATCH_SIZE:-}" ]]; then
  [[ "${OPENPI_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "OPENPI_BATCH_SIZE must be positive" >&2; exit 2; }
  (( OPENPI_BATCH_SIZE % GPU_COUNT == 0 )) || {
    echo "OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE} must be divisible by ${GPU_COUNT} Pi0.5 training GPUs" >&2
    exit 2
  }
fi

echo "[RECAP devices] WCM DDP=${WCM_TRAIN_GPUS} (${WCM_GPU_COUNT} processes)"
echo "[RECAP devices] Pi0.5=${TRAIN_GPUS} (${GPU_COUNT} devices, FSDP=${FSDP_DEVICES}, data_parallel=$((GPU_COUNT / FSDP_DEVICES)))"
echo "[RECAP devices] rollout policy=${POLICY_GPU}, Isaac Sim=${ENV_GPU} (one stateful episode is sequential)"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TASK_SLUG=$(printf '%s' "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//' | cut -c1-80)
RUN_ROOT="${OUTPUT_ROOT}/${TASK_SLUG}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "RECAP run already exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
LEROBOT_HOME="${RUN_ROOT}/lerobot"
mkdir -p "${LEROBOT_HOME}"

CURRENT_POLICY="${INITIAL_POLICY_CHECKPOINT}"
PREVIOUS_WCM="${INITIAL_WCM_CHECKPOINT}"
ROLLOUT_ROOTS=()

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  ITER_DIR=$(printf '%s/iteration_%02d' "${RUN_ROOT}" "${iteration}")
  RAW_ROLLOUTS="${ITER_DIR}/rollouts"
  BUFFER_ROOT="${ITER_DIR}/replay_buffer"
  WCM_OUTPUT="${ITER_DIR}/wcm"
  ADVANTAGES="${ITER_DIR}/recap_advantages.jsonl"
  REPO_ID="RoboDojo-recap-${TASK_SLUG}-iter-${iteration}"
  POLICY_OUTPUT="${ITER_DIR}/pi05"
  NORM_STATS="${ITER_DIR}/norm_stats"
  mkdir -p "${ITER_DIR}"

  echo "[RECAP ${iteration}/${ITERATIONS}] aggregating SFT plus ${#ROLLOUT_ROOTS[@]} completed rollout rounds"
  BUFFER_ARGS=(
    --demo-root "${DEMO_ROOT}"
    --output "${BUFFER_ROOT}"
    --task "${TASK_NAME}"
    --seed "$((SEED + iteration))"
  )
  for source in "${ROLLOUT_ROOTS[@]}"; do BUFFER_ARGS+=(--rollout-root "${source}"); done
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer.py" "${BUFFER_ARGS[@]}"

  echo "[RECAP ${iteration}/${ITERATIONS}] updating WCM on successes and failures"
  WCM_ENV=(
    PYTHON_BIN="${WCM_PYTHON_BIN}"
    WCM_DATASET_ROOT="${BUFFER_ROOT}"
    WCM_SUCCESS_LABELS="${BUFFER_ROOT}/meta/success_labels.json"
    WCM_ASSUME_SUCCESS=0
    WCM_FAILURE_PENALTY="${FAILURE_PENALTY}"
    WCM_GAMMA="${GAMMA}"
    WCM_OUTPUT_DIR="${WCM_OUTPUT}"
    WCM_EPOCHS="${RECAP_WCM_EPOCHS:-1}"
    CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}"
    WCM_RESUME=
    WCM_INIT_CHECKPOINT=
  )
  if [[ -n "${PREVIOUS_WCM}" ]]; then WCM_ENV+=(WCM_INIT_CHECKPOINT="${PREVIOUS_WCM}"); fi
  env "${WCM_ENV[@]}" bash "${SCRIPT_DIR}/run_wcm.sh" --task "${TASK_NAME}"
  PREVIOUS_WCM="${WCM_OUTPUT}/deploy.pt"
  [[ -f "${PREVIOUS_WCM}" ]] || { echo "WCM deploy checkpoint missing: ${PREVIOUS_WCM}" >&2; exit 1; }

  echo "[RECAP ${iteration}/${ITERATIONS}] computing N-step advantage labels"
  ADVANTAGE_ARGS=(
    --wcm-checkpoint "${PREVIOUS_WCM}"
    --dataset-root "${BUFFER_ROOT}"
    --output "${ADVANTAGES}"
    --task "${TASK_NAME}"
    --lookahead "${RECAP_LOOKAHEAD:-50}"
    --gamma "${GAMMA}"
    --failure-penalty "${FAILURE_PENALTY}"
    --positive-fraction "${RECAP_POSITIVE_FRACTION:-0.4}"
    --batch-size "${RECAP_WCM_INFER_BATCH_SIZE:-8}"
    --num-workers "${RECAP_WCM_NUM_WORKERS:-2}"
    --device "${RECAP_WCM_DEVICE:-cuda}"
    --expected-world-size "${WCM_GPU_COUNT}"
  )
  if (( WCM_GPU_COUNT == 1 )); then
    CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}" \
      "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/annotate_recap_advantages.py" "${ADVANTAGE_ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}" \
      "${WCM_PYTHON_BIN}" -m torch.distributed.run --standalone \
        --nproc_per_node="${WCM_GPU_COUNT}" \
        "${SCRIPT_DIR}/annotate_recap_advantages.py" "${ADVANTAGE_ARGS[@]}"
  fi

  echo "[RECAP ${iteration}/${ITERATIONS}] creating advantage-conditioned Pi0.5 dataset"
  HF_LEROBOT_HOME="${LEROBOT_HOME}" "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pi05_dataset.py" \
    --dataset-root "${BUFFER_ROOT}" \
    --repo-id "${REPO_ID}" \
    --advantage-labels "${ADVANTAGES}" \
    --task "${TASK_NAME}" \
    --mode "${OPENPI_DATA_MODE:-video}"

  echo "[RECAP ${iteration}/${ITERATIONS}] computing robot-specific normalization"
  HF_LEROBOT_HOME="${LEROBOT_HOME}" "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/compute_pi05_norm_stats.py" \
    --openpi-root "${PI_DIR}/openpi" \
    --train-config-name "${TRAIN_CONFIG}" \
    --repo-id "${REPO_ID}" \
    --env-cfg-type "${ENV_CFG_TYPE}" \
    --action-type "${ACTION_TYPE}" \
    --output "${NORM_STATS}" \
    --batch-size "${OPENPI_NORM_BATCH_SIZE:-64}" \
    --num-workers "${OPENPI_NUM_WORKERS:-2}" \
    --max-frames "${OPENPI_NORM_MAX_FRAMES:-0}"

  TRAIN_INIT="${CURRENT_POLICY}"
  TRAIN_ARGS=(
    --openpi-root "${PI_DIR}/openpi"
    --train-config-name "${TRAIN_CONFIG}"
    --repo-id "${REPO_ID}"
    --exp-name "recap-${TASK_SLUG}-iter-${iteration}"
    --checkpoint-dir "${POLICY_OUTPUT}"
    --finetune-mode "${FINETUNE_MODE}"
    --init-checkpoint "${TRAIN_INIT}"
    --env-cfg-type "${ENV_CFG_TYPE}"
    --action-type "${ACTION_TYPE}"
    --norm-stats-dir "${NORM_STATS}"
    --recap
    --recap-beta "${BETA}"
    --seed "$((SEED + iteration))"
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
  if [[ "${OPENPI_WANDB_ENABLED:-1}" == "0" ]]; then TRAIN_ARGS+=(--disable-wandb); fi

  echo "[RECAP ${iteration}/${ITERATIONS}] updating Pi0.5 with advantage-conditioned flow matching"
  HF_LEROBOT_HOME="${LEROBOT_HOME}" \
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/train_pi05.py" "${TRAIN_ARGS[@]}"
  CURRENT_POLICY="${POLICY_OUTPUT}"
  printf '%s\n' "${CURRENT_POLICY}" > "${RUN_ROOT}/latest_policy.txt"
  printf '%s\n' "${PREVIOUS_WCM}" > "${RUN_ROOT}/latest_wcm.txt"

  echo "[RECAP ${iteration}/${ITERATIONS}] collecting ${ROLLOUT_EPISODES} episodes with the updated policy"
  ROLLOUT_LOG="${RAW_ROLLOUTS}/rollout.log"
  mkdir -p "${RAW_ROLLOUTS}/episodes"
  (
    ROBODOJO_DISABLE_PROGRESS=1 \
      bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
        --policy-dir "${POLICY_DIR}" \
        --task "${TASK_NAME}" \
        --ckpt "${CURRENT_POLICY}" \
        --env-cfg "${ENV_CFG_TYPE}" \
        --action-type "${ACTION_TYPE}" \
        --seed "$((SEED + iteration - 1))" \
        --policy-gpu "${POLICY_GPU}" \
        --env-gpu "${ENV_GPU}" \
        --policy-env "${POLICY_ENV}" \
        --eval-env "${EVAL_ENV}" \
        --eval-num "${ROLLOUT_EPISODES}" \
        --rollout-dir "${RAW_ROLLOUTS}" \
        --no-video
  ) >"${ROLLOUT_LOG}" 2>&1 &
  ACTIVE_ROLLOUT_PID=$!
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/monitor_rollout_progress.py" \
    --root "${RAW_ROLLOUTS}" \
    --total "${ROLLOUT_EPISODES}" \
    --worker-pid "${ACTIVE_ROLLOUT_PID}" \
    --desc "RECAP rollout ${iteration}/${ITERATIONS}" &
  ACTIVE_ROLLOUT_MONITOR_PID=$!
  rollout_status=0
  wait "${ACTIVE_ROLLOUT_PID}" || rollout_status=$?
  ACTIVE_ROLLOUT_PID=""
  rollout_monitor_status=0
  wait "${ACTIVE_ROLLOUT_MONITOR_PID}" || rollout_monitor_status=$?
  ACTIVE_ROLLOUT_MONITOR_PID=""
  if (( rollout_status != 0 || rollout_monitor_status != 0 )); then
    echo "RECAP rollout failed; tail of ${ROLLOUT_LOG}:" >&2
    tail -n 80 "${ROLLOUT_LOG}" >&2 || true
    exit 1
  fi
  recorded_episodes=$(find "${RAW_ROLLOUTS}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
  recorded_episodes="${recorded_episodes//[[:space:]]/}"
  if [[ "${recorded_episodes}" != "${ROLLOUT_EPISODES}" ]]; then
    echo "Expected ${ROLLOUT_EPISODES} rollout episodes, recorded ${recorded_episodes}; see ${ROLLOUT_LOG}" >&2
    exit 1
  fi
  ROLLOUT_ROOTS+=("${RAW_ROLLOUTS}")
done

echo "RECAP complete"
echo "policy=${CURRENT_POLICY}"
echo "wcm=${PREVIOUS_WCM}"
