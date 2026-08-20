#!/usr/bin/env bash
# Iterated off-policy RoboDojo post-training with WCM and RECAP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/gpu_reservation.sh"
source "${SCRIPT_DIR}/posttrain_config.sh"
install_gpu_reservation_exit_trap
find_posttrain_config "$@"
load_posttrain_config "${POSTTRAIN_CONFIG_FILE}"

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
ITERATIONS="${RECAP_ITERATIONS:-3}"
ROLLOUT_EPISODES="${RECAP_ROLLOUT_EPISODES:-50}"
MAX_DEMO_EPISODES="${RECAP_MAX_DEMO_EPISODES:-0}"
VALUE_VIDEO_EPISODES="${RECAP_VALUE_VIDEO_EPISODES:-}"
VALUE_VIDEO_GPU="${RECAP_VALUE_VIDEO_GPU:-}"
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
UNCONDITIONAL_PROB="${RECAP_UNCONDITIONAL_PROB:-0.1}"
GUIDANCE_SCALE="${RECAP_GUIDANCE_SCALE:-1.0}"
FAILURE_PENALTY="${WCM_FAILURE_PENALTY:-300}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-${NUM_TRAIN_STEPS:-1000}}"
RESUME_RUN="${RECAP_RESUME:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/posttrain/run_pi05_recap.sh [options]

Required:
  --task TASK                         RoboDojo task slug or instruction
  --initial-policy-checkpoint PATH    Initial Pi0.5 SFT checkpoint

Options:
  --config PATH                       Flat YAML hyperparameter config
  --demo-root PATH                    Successful SFT LeRobot-v2.1 dataset
  --initial-wcm-checkpoint PATH       Warm-start WCM model weights
  --output-root PATH                  Run output root (default: outputs/recap)
  --iterations N                      Policy-improvement iterations (default: 3)
  --rollout-episodes N                Simulator episodes per iteration (default: 50)
  --max-demo-episodes N               Use first N task demonstrations (0: all)
  --value-video-episodes N            Render N WCM-value rollout videos per iteration (0: disable)
  --env-cfg NAME                      RoboDojo robot/environment config
  --action-type joint|ee              Policy action representation
  --finetune-mode MODE                full/action_expert/*_lora mode
  --train-gpus IDS                    Pi0.5 training GPUs, e.g. 0,1,2,3
  --wcm-train-gpus IDS                WCM DDP GPUs (default: --train-gpus)
  --policy-gpu ID                     Rollout policy-server GPU
  --env-gpu ID                        Rollout Isaac Sim GPU
  --num-train-steps N                 Pi0.5 training steps per iteration
  --resume                            Continue an existing run from its first incomplete stage

Additional optimizer, device, WCM, and RECAP controls are documented in
configs/posttrain/pi05_recap.yaml.example.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift 2 ;;
    --task) TASK_NAME="$2"; shift 2 ;;
    --demo-root) DEMO_ROOT="$2"; shift 2 ;;
    --initial-policy-checkpoint) INITIAL_POLICY_CHECKPOINT="$2"; shift 2 ;;
    --initial-wcm-checkpoint) INITIAL_WCM_CHECKPOINT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --rollout-episodes) ROLLOUT_EPISODES="$2"; shift 2 ;;
    --max-demo-episodes) MAX_DEMO_EPISODES="$2"; shift 2 ;;
    --value-video-episodes) VALUE_VIDEO_EPISODES="$2"; shift 2 ;;
    --env-cfg) ENV_CFG_TYPE="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --finetune-mode) FINETUNE_MODE="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --wcm-train-gpus) WCM_TRAIN_GPUS="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --resume) RESUME_RUN=1; shift ;;
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
[[ "${NUM_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num-train-steps must be positive" >&2; exit 2; }
[[ "${RESUME_RUN}" == "0" || "${RESUME_RUN}" == "1" ]] || { echo "RECAP_RESUME must be 0 or 1" >&2; exit 2; }
VALUE_VIDEO_EPISODES="${VALUE_VIDEO_EPISODES:-$((ROLLOUT_EPISODES < 3 ? ROLLOUT_EPISODES : 3))}"
[[ "${MAX_DEMO_EPISODES}" =~ ^[0-9]+$ ]] || { echo "--max-demo-episodes must be non-negative" >&2; exit 2; }
[[ "${VALUE_VIDEO_EPISODES}" =~ ^[0-9]+$ ]] || { echo "--value-video-episodes must be non-negative" >&2; exit 2; }
(( VALUE_VIDEO_EPISODES <= ROLLOUT_EPISODES )) || {
  echo "--value-video-episodes cannot exceed --rollout-episodes" >&2
  exit 2
}

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
VALUE_VIDEO_GPU="${VALUE_VIDEO_GPU:-${WCM_TRAIN_GPU_IDS[0]}}"
if [[ -z "${ENV_GPU}" ]]; then
  if (( GPU_COUNT > 1 )); then
    ENV_GPU="${POLICY_TRAIN_GPU_IDS[1]}"
  else
    ENV_GPU="${POLICY_TRAIN_GPU_IDS[0]}"
  fi
fi
[[ "${POLICY_GPU}" =~ ^[0-9]+$ ]] || { echo "--policy-gpu must be one numeric GPU id" >&2; exit 2; }
[[ "${ENV_GPU}" =~ ^[0-9]+$ ]] || { echo "--env-gpu must be one numeric GPU id" >&2; exit 2; }
[[ "${VALUE_VIDEO_GPU}" =~ ^[0-9]+$ ]] || { echo "RECAP_VALUE_VIDEO_GPU must be one numeric GPU id" >&2; exit 2; }
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
if (( VALUE_VIDEO_EPISODES > 0 )); then
  echo "[RECAP devices] WCM value-video inference=${VALUE_VIDEO_GPU}"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

TASK_SLUG=$(printf '%s' "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//' | cut -c1-80)
RUN_ROOT="${OUTPUT_ROOT}/${TASK_SLUG}"
if [[ "${RESUME_RUN}" == "1" ]]; then
  [[ -d "${RUN_ROOT}" ]] || { echo "RECAP resume root does not exist: ${RUN_ROOT}" >&2; exit 1; }
  echo "[RECAP resume] inspecting ${RUN_ROOT}"
else
  [[ ! -e "${RUN_ROOT}" ]] || { echo "RECAP run already exists: ${RUN_ROOT}; pass --resume to continue" >&2; exit 1; }
  mkdir -p "${RUN_ROOT}"
fi
LEROBOT_HOME="${RUN_ROOT}/lerobot"
mkdir -p "${LEROBOT_HOME}"

artifact_complete() {
  local stage="$1"
  local path="$2"
  local expected="${3:-0}"
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" check \
    --stage "${stage}" --path "${path}" --expected "${expected}" >/dev/null 2>&1
}

archive_incomplete() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" archive --path "${path}"
  fi
}

reuse_stage() {
  echo "[RECAP resume] reusing iteration ${iteration} stage=$1"
}

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
  PI_DATASET="${LEROBOT_HOME}/${REPO_ID}"
  REBUILD_DOWNSTREAM=0
  mkdir -p "${ITER_DIR}"

  # RECAP is offline with respect to each policy update: collect with the
  # current policy first, then use those trajectories in this iteration's
  # critic and CFG update.  Collecting after training drops the final round
  # from optimization and leaves iteration 1 with demonstrations only.
  if [[ "${RESUME_RUN}" == "1" ]] && artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}"; then
    reuse_stage rollout
  else
    mkdir -p "${RAW_ROLLOUTS}/episodes"
    if [[ -e "${RAW_ROLLOUTS}/_in_progress" ]]; then
      archive_incomplete "${RAW_ROLLOUTS}/_in_progress"
    fi
    recorded_episodes=$(find "${RAW_ROLLOUTS}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    recorded_episodes="${recorded_episodes//[[:space:]]/}"
    (( recorded_episodes <= ROLLOUT_EPISODES )) || {
      echo "Rollout directory has ${recorded_episodes} episodes, expected at most ${ROLLOUT_EPISODES}: ${RAW_ROLLOUTS}" >&2
      exit 1
    }
    remaining_episodes=$((ROLLOUT_EPISODES - recorded_episodes))
    if (( remaining_episodes > 0 )); then
      echo "[RECAP ${iteration}/${ITERATIONS}] collecting ${remaining_episodes} remaining episodes (${recorded_episodes}/${ROLLOUT_EPISODES} already complete)"
      if [[ -f "${RAW_ROLLOUTS}/rollout.log" ]]; then
        ROLLOUT_LOG="${RAW_ROLLOUTS}/rollout_resume_$(date +%Y%m%dT%H%M%S).log"
      else
        ROLLOUT_LOG="${RAW_ROLLOUTS}/rollout.log"
      fi
      (
        ROBODOJO_DISABLE_PROGRESS=1 \
          bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
            --policy-dir "${POLICY_DIR}" \
            --task "${TASK_NAME}" \
            --ckpt "${CURRENT_POLICY}" \
            --env-cfg "${ENV_CFG_TYPE}" \
            --action-type "${ACTION_TYPE}" \
            --seed "$((SEED + iteration - 1 + recorded_episodes))" \
            --policy-gpu "${POLICY_GPU}" \
            --env-gpu "${ENV_GPU}" \
            --policy-env "${POLICY_ENV}" \
            --eval-env "${EVAL_ENV}" \
            --eval-num "${remaining_episodes}" \
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
    fi
    artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}" || {
      echo "Expected ${ROLLOUT_EPISODES} complete rollout episodes below ${RAW_ROLLOUTS}" >&2
      exit 1
    }
    REBUILD_DOWNSTREAM=1
  fi
  ROLLOUT_ROOTS+=("${RAW_ROLLOUTS}")

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete buffer "${BUFFER_ROOT}"; then
    reuse_stage replay_buffer
  else
    archive_incomplete "${BUFFER_ROOT}"
    echo "[RECAP ${iteration}/${ITERATIONS}] aggregating SFT plus ${#ROLLOUT_ROOTS[@]} completed rollout rounds"
    start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP replay-buffer and WCM dataset preparation"
    BUFFER_ARGS=(
      --demo-root "${DEMO_ROOT}"
      --output "${BUFFER_ROOT}"
      --task "${TASK_NAME}"
      --max-demo-episodes "${MAX_DEMO_EPISODES}"
      --seed "$((SEED + iteration))"
    )
    for source in "${ROLLOUT_ROOTS[@]}"; do BUFFER_ARGS+=(--rollout-root "${source}"); done
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer.py" "${BUFFER_ARGS[@]}"
    REBUILD_DOWNSTREAM=1
  fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete wcm "${WCM_OUTPUT}"; then
    reuse_stage wcm
    stop_gpu_reservation
  else
    WCM_STAGE_RESUME=""
    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && [[ -f "${WCM_OUTPUT}/checkpoints/last.pt" ]]; then
      WCM_STAGE_RESUME="${WCM_OUTPUT}/checkpoints/last.pt"
      echo "[RECAP resume] resuming iteration ${iteration} WCM checkpoint=${WCM_STAGE_RESUME}"
    else
      archive_incomplete "${WCM_OUTPUT}"
    fi
    echo "[RECAP ${iteration}/${ITERATIONS}] updating WCM on successes and failures"
    start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "WCM dataset preparation"
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
      WCM_RESUME="${WCM_STAGE_RESUME}"
      WCM_INIT_CHECKPOINT=
    )
    if [[ -z "${WCM_STAGE_RESUME}" && -n "${PREVIOUS_WCM}" ]]; then
      WCM_ENV+=(WCM_INIT_CHECKPOINT="${PREVIOUS_WCM}")
    fi
    stop_gpu_reservation
    env "${WCM_ENV[@]}" bash "${SCRIPT_DIR}/run_wcm.sh" --task "${TASK_NAME}"
    REBUILD_DOWNSTREAM=1
  fi
  PREVIOUS_WCM="${WCM_OUTPUT}/deploy.pt"
  [[ -f "${PREVIOUS_WCM}" ]] || { echo "WCM deploy checkpoint missing: ${PREVIOUS_WCM}" >&2; exit 1; }

  ADVANTAGE_ARGS=(
    --wcm-checkpoint "${PREVIOUS_WCM}"
    --dataset-root "${BUFFER_ROOT}"
    --output "${ADVANTAGES}"
    --task "${TASK_NAME}"
    --lookahead "${RECAP_LOOKAHEAD:-10}"
    --gamma "${GAMMA}"
    --failure-penalty "${FAILURE_PENALTY}"
    --positive-fraction "${RECAP_POSITIVE_FRACTION:-0.3}"
    --batch-size "${RECAP_WCM_INFER_BATCH_SIZE:-8}"
    --num-workers "${RECAP_WCM_NUM_WORKERS:-2}"
    --device "${RECAP_WCM_DEVICE:-cuda}"
    --expected-world-size "${WCM_GPU_COUNT}"
  )
  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete advantages "${ADVANTAGES}"; then
    reuse_stage advantages
  else
    archive_incomplete "${ADVANTAGES}"
    echo "[RECAP ${iteration}/${ITERATIONS}] computing N-step advantage labels"
    if (( WCM_GPU_COUNT == 1 )); then
      CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}" \
        "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/annotate_recap_advantages.py" "${ADVANTAGE_ARGS[@]}"
    else
      CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}" \
        "${WCM_PYTHON_BIN}" -m torch.distributed.run --standalone \
          --nproc_per_node="${WCM_GPU_COUNT}" \
          "${SCRIPT_DIR}/annotate_recap_advantages.py" "${ADVANTAGE_ARGS[@]}"
    fi
    REBUILD_DOWNSTREAM=1
  fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete pi_dataset "${PI_DATASET}"; then
    reuse_stage pi_dataset
  else
    archive_incomplete "${PI_DATASET}"
    echo "[RECAP ${iteration}/${ITERATIONS}] creating advantage-conditioned Pi0.5 dataset"
    start_gpu_reservation "${TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP Pi0.5 dataset preparation"
    HF_LEROBOT_HOME="${LEROBOT_HOME}" "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pi05_dataset.py" \
      --dataset-root "${BUFFER_ROOT}" \
      --repo-id "${REPO_ID}" \
      --advantage-labels "${ADVANTAGES}" \
      --task "${TASK_NAME}" \
      --mode "${OPENPI_DATA_MODE:-video}"
    REBUILD_DOWNSTREAM=1
  fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete norm "${NORM_STATS}"; then
    reuse_stage norm_stats
  else
    archive_incomplete "${NORM_STATS}"
    echo "[RECAP ${iteration}/${ITERATIONS}] computing robot-specific normalization"
    start_gpu_reservation "${TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP Pi0.5 dataset preparation"
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
    REBUILD_DOWNSTREAM=1
  fi

  TRAIN_INIT="${CURRENT_POLICY}"
  TRAIN_ARGS=(
    --openpi-root "${PI_DIR}/openpi"
    --train-config-name "${TRAIN_CONFIG}"
    --repo-id "${REPO_ID}"
    --exp-name "recap-${TASK_SLUG}-iter-${iteration}"
    --checkpoint-dir "${POLICY_OUTPUT}"
    --finetune-mode "${FINETUNE_MODE}"
    --env-cfg-type "${ENV_CFG_TYPE}"
    --action-type "${ACTION_TYPE}"
    --norm-stats-dir "${NORM_STATS}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --recap
    --recap-unconditional-prob "${UNCONDITIONAL_PROB}"
    --recap-guidance-scale "${GUIDANCE_SCALE}"
    --seed "$((SEED + iteration))"
    --fsdp-devices "${FSDP_DEVICES}"
  )
  for option in batch-size num-workers save-interval log-interval; do
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

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete policy "${POLICY_OUTPUT}"; then
    reuse_stage policy
    stop_gpu_reservation
  else
    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete policy_resume "${POLICY_OUTPUT}"; then
      echo "[RECAP resume] resuming iteration ${iteration} Pi0.5 optimizer checkpoint"
      TRAIN_ARGS+=(--resume)
    else
      archive_incomplete "${POLICY_OUTPUT}"
      TRAIN_ARGS+=(--init-checkpoint "${TRAIN_INIT}")
    fi
    echo "[RECAP ${iteration}/${ITERATIONS}] updating Pi0.5 with advantage-conditioned flow matching"
    echo "using finetune mode ${FINETUNE_MODE}"
    stop_gpu_reservation
    HF_LEROBOT_HOME="${LEROBOT_HOME}" \
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/train_pi05.py" "${TRAIN_ARGS[@]}"
    REBUILD_DOWNSTREAM=1
  fi
  CURRENT_POLICY="${POLICY_OUTPUT}"
  printf '%s\n' "${CURRENT_POLICY}" > "${RUN_ROOT}/latest_policy.txt"
  printf '%s\n' "${PREVIOUS_WCM}" > "${RUN_ROOT}/latest_wcm.txt"

  if (( VALUE_VIDEO_EPISODES > 0 )); then
    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete value_videos "${ITER_DIR}/value_videos" "${VALUE_VIDEO_EPISODES}"; then
      reuse_stage value_videos
    else
      archive_incomplete "${ITER_DIR}/value_videos"
      echo "[RECAP ${iteration}/${ITERATIONS}] rendering ${VALUE_VIDEO_EPISODES} rollout videos with WCM value overlays"
      CUDA_VISIBLE_DEVICES="${VALUE_VIDEO_GPU}" \
        "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/render_rollout_value_videos.py" \
          --wcm-checkpoint "${PREVIOUS_WCM}" \
          --rollout-root "${RAW_ROLLOUTS}" \
          --output-dir "${ITER_DIR}/value_videos" \
          --max-episodes "${VALUE_VIDEO_EPISODES}" \
          --batch-size "${RECAP_VALUE_VIDEO_BATCH_SIZE:-8}" \
          --device "${RECAP_VALUE_VIDEO_DEVICE:-cuda}" \
          --precision "${RECAP_VALUE_VIDEO_PRECISION:-bf16}" \
          --backend "${RECAP_VALUE_VIDEO_BACKEND:-auto}" \
          --speed "${RECAP_VALUE_VIDEO_SPEED:-1.0}" \
          --y-min "${RECAP_VALUE_VIDEO_Y_MIN:--1.0}" \
          --y-max "${RECAP_VALUE_VIDEO_Y_MAX:-1.0}" \
          --title "WCM RECAP ITER ${iteration}"
      REBUILD_DOWNSTREAM=1
    fi
  fi
done

echo "RECAP complete"
echo "policy=${CURRENT_POLICY}"
echo "wcm=${PREVIOUS_WCM}"
stop_gpu_reservation