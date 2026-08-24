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
ROLLOUT_EPISODES="${RECAP_ROLLOUT_EPISODES:-100}"
MIN_ROLLOUT_EPISODES="${RECAP_MIN_ROLLOUT_EPISODES:-50}"
MIN_SUCCESS_EPISODES="${RECAP_MIN_SUCCESS_EPISODES:-5}"
MIN_FAILURE_EPISODES="${RECAP_MIN_FAILURE_EPISODES:-5}"
MAX_DEMO_EPISODES="${RECAP_MAX_DEMO_EPISODES:-100}"
WCM_REPLAY_EPISODES="${RECAP_WCM_REPLAY_EPISODES:-20}"
VALUE_VIDEO_EPISODES="${RECAP_VALUE_VIDEO_EPISODES:-}"
VALUE_VIDEO_GPU="${RECAP_VALUE_VIDEO_GPU:-}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
WCM_TRAIN_GPUS="${WCM_TRAIN_GPUS:-}"
POLICY_GPU="${POLICY_GPU:-}"
ENV_GPU="${ENV_GPU:-}"
SEED="${SEED:-0}"
ROLLOUT_LAYOUT_SEED="${RECAP_ROLLOUT_LAYOUT_SEED:-0}"
TRAIN_CONFIG="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
FINETUNE_MODE="${PI05_FINETUNE_MODE:-action_expert_lora}"
GAMMA="${RECAP_GAMMA:-1.0}"
UNCONDITIONAL_PROB="${RECAP_UNCONDITIONAL_PROB:-0.1}"
GUIDANCE_SCALE="${RECAP_GUIDANCE_SCALE:-1.0}"
DEMO_SAMPLING_WEIGHT="${RECAP_DEMO_SAMPLING_WEIGHT:-1.0}"
ROLLOUT_SAMPLING_WEIGHT="${RECAP_ROLLOUT_SAMPLING_WEIGHT:-1.0}"
FAILURE_PENALTY="${WCM_FAILURE_PENALTY:-300}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-3000}"
POLICY_WARMUP_STEPS="${OPENPI_WARMUP_STEPS:-}"
POLICY_EVAL_INTERVAL="${RECAP_POLICY_EVAL_INTERVAL:-1000}"
POLICY_EVAL_EPISODES="${RECAP_POLICY_EVAL_EPISODES:-20}"
POLICY_EVAL_LAYOUT_SEED="${RECAP_POLICY_EVAL_LAYOUT_SEED:-1}"
POLICY_EVAL_LAYOUT_OFFSET="${RECAP_POLICY_EVAL_LAYOUT_OFFSET:-0}"
PROMOTION_MAX_SUCCESS_DROP="${RECAP_PROMOTION_MAX_SUCCESS_DROP:-0.1}"
PROMOTION_MIN_SUCCESS_RATE="${RECAP_PROMOTION_MIN_SUCCESS_RATE:-0.0}"
STOP_ON_REJECTION="${RECAP_STOP_ON_REJECTION:-1}"
NORM_ASSET_ID="${OPENPI_NORM_ASSET_ID:-}"
RESUME_RUN="${RECAP_RESUME:-0}"
REMOTE_HOST="${RECAP_REMOTE_ROLLOUT_HOST:-}"
REMOTE_REPO_ROOT="${RECAP_REMOTE_REPO_ROOT:-}"
REMOTE_WORK_ROOT="${RECAP_REMOTE_WORK_ROOT:-}"
REMOTE_POLICY_GPU="${RECAP_REMOTE_POLICY_GPU:-0}"
REMOTE_ENV_GPU="${RECAP_REMOTE_ENV_GPU:-0}"
REMOTE_VALUE_VIDEO_GPU="${RECAP_REMOTE_VALUE_VIDEO_GPU:-${REMOTE_ENV_GPU}}"
REMOTE_POLICY_EVAL="${RECAP_REMOTE_POLICY_EVAL:-1}"
REMOTE_ENABLED=0
[[ -z "${REMOTE_HOST}" ]] || REMOTE_ENABLED=1

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
  --rollout-episodes N                Simulator episodes per iteration (default: 100)
  --max-demo-episodes N               Use first N task demonstrations (0: all)
  --wcm-replay-episodes K             Old episodes sampled for each later WCM update
  --value-video-episodes N            Render N WCM-value rollout videos per iteration (0: disable)
  --env-cfg NAME                      RoboDojo robot/environment config
  --action-type joint|ee              Policy action representation
  --finetune-mode MODE                full/action_expert/*_lora mode
  --train-gpus IDS                    Pi0.5 training GPUs, e.g. 0,1,2,3
  --wcm-train-gpus IDS                WCM DDP GPUs (default: --train-gpus)
  --policy-gpu ID                     Rollout policy-server GPU
  --env-gpu ID                        Rollout Isaac Sim GPU
  --num-train-steps N                 Pi0.5 training steps per iteration
  --policy-eval-interval N            Save/evaluate every N policy steps
  --policy-eval-episodes N            Held-out episodes per candidate checkpoint
  --remote-rollout-host HOST          SSH alias/user@host for pipelined rollout
  --remote-repo-root PATH             RoboDojo checkout on the rollout host
  --remote-work-root PATH             Persistent remote job/cache directory
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
    --wcm-replay-episodes) WCM_REPLAY_EPISODES="$2"; shift 2 ;;
    --value-video-episodes) VALUE_VIDEO_EPISODES="$2"; shift 2 ;;
    --env-cfg) ENV_CFG_TYPE="$2"; shift 2 ;;
    --action-type) ACTION_TYPE="$2"; shift 2 ;;
    --finetune-mode) FINETUNE_MODE="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --wcm-train-gpus) WCM_TRAIN_GPUS="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --env-gpu) ENV_GPU="$2"; shift 2 ;;
    --num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2 ;;
    --policy-eval-interval) POLICY_EVAL_INTERVAL="$2"; shift 2 ;;
    --policy-eval-episodes) POLICY_EVAL_EPISODES="$2"; shift 2 ;;
    --remote-rollout-host) REMOTE_HOST="$2"; REMOTE_ENABLED=1; shift 2 ;;
    --remote-repo-root) REMOTE_REPO_ROOT="$2"; shift 2 ;;
    --remote-work-root) REMOTE_WORK_ROOT="$2"; shift 2 ;;
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
[[ "${MIN_ROLLOUT_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "RECAP_MIN_ROLLOUT_EPISODES must be positive" >&2; exit 2; }
[[ "${MIN_SUCCESS_EPISODES}" =~ ^[0-9]+$ ]] || { echo "RECAP_MIN_SUCCESS_EPISODES must be non-negative" >&2; exit 2; }
[[ "${MIN_FAILURE_EPISODES}" =~ ^[0-9]+$ ]] || { echo "RECAP_MIN_FAILURE_EPISODES must be non-negative" >&2; exit 2; }
(( ROLLOUT_EPISODES >= MIN_ROLLOUT_EPISODES )) || {
  echo "RECAP requires at least ${MIN_ROLLOUT_EPISODES} rollout episodes per iteration; got ${ROLLOUT_EPISODES}" >&2
  exit 2
}
(( MIN_SUCCESS_EPISODES + MIN_FAILURE_EPISODES <= ROLLOUT_EPISODES )) || {
  echo "RECAP success/failure minimums exceed rollout episode count" >&2
  exit 2
}
if [[ -z "${POLICY_WARMUP_STEPS}" ]]; then
  POLICY_WARMUP_STEPS=$(( (NUM_TRAIN_STEPS + 5) / 6 ))
fi
[[ "${POLICY_WARMUP_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "OPENPI_WARMUP_STEPS must be positive" >&2; exit 2; }
(( POLICY_WARMUP_STEPS < NUM_TRAIN_STEPS )) || {
  echo "OPENPI_WARMUP_STEPS=${POLICY_WARMUP_STEPS} must be smaller than training steps=${NUM_TRAIN_STEPS}" >&2
  exit 2
}
[[ "${POLICY_EVAL_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "RECAP_POLICY_EVAL_INTERVAL must be positive" >&2; exit 2; }
(( POLICY_EVAL_INTERVAL < NUM_TRAIN_STEPS )) || {
  echo "RECAP_POLICY_EVAL_INTERVAL=${POLICY_EVAL_INTERVAL} must be smaller than training steps=${NUM_TRAIN_STEPS}" >&2
  exit 2
}
[[ "${POLICY_EVAL_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "RECAP_POLICY_EVAL_EPISODES must be positive" >&2; exit 2; }
[[ "${POLICY_EVAL_LAYOUT_SEED}" =~ ^[0-9]+$ ]] || { echo "RECAP_POLICY_EVAL_LAYOUT_SEED must be non-negative" >&2; exit 2; }
[[ "${POLICY_EVAL_LAYOUT_OFFSET}" =~ ^[0-9]+$ ]] || { echo "RECAP_POLICY_EVAL_LAYOUT_OFFSET must be non-negative" >&2; exit 2; }
[[ "${STOP_ON_REJECTION}" == "0" || "${STOP_ON_REJECTION}" == "1" ]] || { echo "RECAP_STOP_ON_REJECTION must be 0 or 1" >&2; exit 2; }
[[ "${REMOTE_POLICY_EVAL}" == "0" || "${REMOTE_POLICY_EVAL}" == "1" ]] || { echo "RECAP_REMOTE_POLICY_EVAL must be 0 or 1" >&2; exit 2; }
if (( REMOTE_ENABLED )); then
  [[ -n "${REMOTE_REPO_ROOT}" ]] || { echo "RECAP_REMOTE_REPO_ROOT is required with remote rollout" >&2; exit 2; }
  [[ -n "${REMOTE_WORK_ROOT}" ]] || { echo "RECAP_REMOTE_WORK_ROOT is required with remote rollout" >&2; exit 2; }
  for remote_gpu in REMOTE_POLICY_GPU REMOTE_ENV_GPU REMOTE_VALUE_VIDEO_GPU; do
    [[ "${!remote_gpu}" =~ ^[0-9]+$ ]] || { echo "${remote_gpu} must be one numeric GPU id" >&2; exit 2; }
  done
fi
"${WCM_PYTHON_BIN}" -c '
import sys
for offset in range(1, len(sys.argv), 4):
    name, raw, lower, upper = sys.argv[offset:offset + 4]
    value = float(raw)
    if not float(lower) <= value <= float(upper):
        raise SystemExit(f"{name}={raw} must be in [{lower}, {upper}]")
' \
  RECAP_DEMO_SAMPLING_WEIGHT "${DEMO_SAMPLING_WEIGHT}" 0 1000000 \
  RECAP_ROLLOUT_SAMPLING_WEIGHT "${ROLLOUT_SAMPLING_WEIGHT}" 0 1000000 \
  RECAP_PROMOTION_MAX_SUCCESS_DROP "${PROMOTION_MAX_SUCCESS_DROP}" 0 1 \
  RECAP_PROMOTION_MIN_SUCCESS_RATE "${PROMOTION_MIN_SUCCESS_RATE}" 0 1
"${WCM_PYTHON_BIN}" -c 'import sys; assert float(sys.argv[1]) > 0 and float(sys.argv[2]) > 0, "RECAP sampling weights must be positive"' \
  "${DEMO_SAMPLING_WEIGHT}" "${ROLLOUT_SAMPLING_WEIGHT}"
[[ "${RESUME_RUN}" == "0" || "${RESUME_RUN}" == "1" ]] || { echo "RECAP_RESUME must be 0 or 1" >&2; exit 2; }
[[ "${ROLLOUT_LAYOUT_SEED}" =~ ^[0-9]+$ ]] || { echo "RECAP_ROLLOUT_LAYOUT_SEED must be non-negative" >&2; exit 2; }
VALUE_VIDEO_EPISODES="${VALUE_VIDEO_EPISODES:-$((ROLLOUT_EPISODES < 3 ? ROLLOUT_EPISODES : 3))}"
[[ "${MAX_DEMO_EPISODES}" =~ ^[0-9]+$ ]] || { echo "--max-demo-episodes must be non-negative" >&2; exit 2; }
[[ "${WCM_REPLAY_EPISODES}" =~ ^[0-9]+$ ]] || { echo "--wcm-replay-episodes must be non-negative" >&2; exit 2; }
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
if (( REMOTE_ENABLED )); then
  echo "[RECAP devices] remote rollout=${REMOTE_HOST} policy=${REMOTE_POLICY_GPU}, Isaac Sim=${REMOTE_ENV_GPU}"
fi
if (( VALUE_VIDEO_EPISODES > 0 )); then
  echo "[RECAP devices] WCM value-video inference=${VALUE_VIDEO_GPU}"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if (( REMOTE_ENABLED )); then
  echo "[RECAP remote] checking passwordless SSH, zstd, and remote RoboDojo checkout"
  REMOTE_PREFLIGHT_ARGS=(
    preflight
    --host "${REMOTE_HOST}" --remote-repo-root "${REMOTE_REPO_ROOT}"
    --remote-work-root "${REMOTE_WORK_ROOT}"
    --gpu "${REMOTE_POLICY_GPU}" --gpu "${REMOTE_ENV_GPU}"
    --gpu "${REMOTE_VALUE_VIDEO_GPU}"
  )
  if (( VALUE_VIDEO_EPISODES > 0 )); then REMOTE_PREFLIGHT_ARGS+=(--require-wcm); fi
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/remote_recap.py" "${REMOTE_PREFLIGHT_ARGS[@]}"
fi

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
  local fingerprint="${4-${ACTIVE_STAGE_FP:-}}"
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" check \
    --stage "${stage}" --path "${path}" --expected "${expected}" \
    --fingerprint "${fingerprint}" >/dev/null 2>&1
}

mark_artifact() {
  local stage="$1"
  local path="$2"
  local fingerprint="${3:-${ACTIVE_STAGE_FP:-}}"
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" mark \
    --stage "${stage}" --path "${path}" --fingerprint "${fingerprint}"
}

artifact_fingerprint_matches() {
  local stage="$1"
  local path="$2"
  local fingerprint="${3:-${ACTIVE_STAGE_FP:-}}"
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" matches \
    --stage "${stage}" --path "${path}" --fingerprint "${fingerprint}" >/dev/null 2>&1
}

stage_fingerprint() {
  local stage="$1"
  shift
  local -a args=(fingerprint --stage "${stage}")
  local entry
  for entry in "$@"; do args+=(--entry "${entry}"); done
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/recap_artifacts.py" "${args[@]}"
}

evaluate_policy_checkpoint() {
  local checkpoint="$1"
  local output="$2"
  local label="$3"
  local eval_fp
  eval_fp=$(stage_fingerprint policy_eval \
    "iteration=${ACTIVE_STAGE_FP}" "checkpoint=${checkpoint}" \
    "episodes=${POLICY_EVAL_EPISODES}" "seed=${POLICY_EVAL_LAYOUT_SEED}" \
    "offset=${POLICY_EVAL_LAYOUT_OFFSET}")
  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && \
     artifact_complete rollout "${output}" "${POLICY_EVAL_EPISODES}" "${eval_fp}"; then
    echo "[RECAP promotion] reusing ${label} evaluation"
    return
  fi
  archive_incomplete "${output}"
  echo "[RECAP promotion] evaluating ${label}: ${checkpoint}"
  if (( REMOTE_ENABLED && REMOTE_POLICY_EVAL )); then
    local eval_job_id="${TASK_SLUG}-${RUN_CONFIG_FP:0:12}-eval-${eval_fp:0:16}"
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/remote_recap.py" rollout \
      --host "${REMOTE_HOST}" --remote-repo-root "${REMOTE_REPO_ROOT}" \
      --remote-work-root "${REMOTE_WORK_ROOT}" --job-id "${eval_job_id}" \
      --checkpoint "${checkpoint}" --output "${output}" --task "${TASK_NAME}" \
      --episodes "${POLICY_EVAL_EPISODES}" --layout-seed "${POLICY_EVAL_LAYOUT_SEED}" \
      --layout-offset "${POLICY_EVAL_LAYOUT_OFFSET}" \
      --policy-gpu "${REMOTE_POLICY_GPU}" --env-gpu "${REMOTE_ENV_GPU}" \
      --env-cfg "${ENV_CFG_TYPE}" --action-type "${ACTION_TYPE}" \
      --policy-env "${POLICY_ENV}" --eval-env "${EVAL_ENV}"
  else
    mkdir -p "${output}"
    local log="${output}/eval.log"
    ROBODOJO_DISABLE_PROGRESS=1 \
      bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
        --policy-dir "${POLICY_DIR}" --task "${TASK_NAME}" --ckpt "${checkpoint}" \
        --env-cfg "${ENV_CFG_TYPE}" --action-type "${ACTION_TYPE}" \
        --seed "${POLICY_EVAL_LAYOUT_SEED}" --layout-offset "${POLICY_EVAL_LAYOUT_OFFSET}" \
        --policy-gpu "${POLICY_GPU}" --env-gpu "${ENV_GPU}" \
        --policy-env "${POLICY_ENV}" --eval-env "${EVAL_ENV}" \
        --eval-num "${POLICY_EVAL_EPISODES}" --rollout-dir "${output}" --no-video \
        >"${log}" 2>&1 || {
          echo "Policy evaluation failed; tail of ${log}:" >&2
          tail -n 80 "${log}" >&2 || true
          return 1
        }
  fi
  artifact_complete rollout "${output}" "${POLICY_EVAL_EPISODES}" "" || {
    echo "Incomplete policy evaluation artifact: ${output}" >&2
    return 1
  }
  mark_artifact rollout "${output}" "${eval_fp}"
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

RUN_CONFIG_FP=$(stage_fingerprint run \
  "task=${TASK_NAME}" "demo_root=$(realpath "${DEMO_ROOT}")" \
  "initial_policy=$(realpath "${INITIAL_POLICY_CHECKPOINT}")" \
  "initial_wcm=${INITIAL_WCM_CHECKPOINT}" "iterations=${ITERATIONS}" \
  "rollout_episodes=${ROLLOUT_EPISODES}" "min_rollouts=${MIN_ROLLOUT_EPISODES}" \
  "min_successes=${MIN_SUCCESS_EPISODES}" "min_failures=${MIN_FAILURE_EPISODES}" \
  "max_demos=${MAX_DEMO_EPISODES}" "wcm_replay=${WCM_REPLAY_EPISODES}" \
  "wcm_epochs=${RECAP_WCM_EPOCHS:-5}" "lookahead=${RECAP_LOOKAHEAD:-10}" \
  "wcm_batch=${WCM_PER_DEVICE_BATCH_SIZE:-}" "wcm_lr=${WCM_LR:-}" \
  "wcm_warmup=${WCM_WARMUP_STEPS:-}" "wcm_infer_batch=${RECAP_WCM_INFER_BATCH_SIZE:-8}" \
  "gamma=${GAMMA}" "failure_penalty=${FAILURE_PENALTY}" \
  "positive_fraction=${RECAP_POSITIVE_FRACTION:-0.3}" \
  "unconditional_prob=${UNCONDITIONAL_PROB}" "guidance_scale=${GUIDANCE_SCALE}" \
  "demo_weight=${DEMO_SAMPLING_WEIGHT}" "rollout_weight=${ROLLOUT_SAMPLING_WEIGHT}" \
  "train_config=${TRAIN_CONFIG}" "finetune_mode=${FINETUNE_MODE}" \
  "train_steps=${NUM_TRAIN_STEPS}" "warmup_steps=${POLICY_WARMUP_STEPS}" \
  "batch_size=${OPENPI_BATCH_SIZE:-}" "learning_rate=${OPENPI_LEARNING_RATE:-5e-6}" \
  "num_workers=${OPENPI_NUM_WORKERS:-}" "decay_lr=${OPENPI_DECAY_LR:-}" \
  "weight_decay=${OPENPI_WEIGHT_DECAY:-}" "clip_grad=${OPENPI_CLIP_GRADIENT_NORM:-}" \
  "eval_interval=${POLICY_EVAL_INTERVAL}" "eval_episodes=${POLICY_EVAL_EPISODES}" \
  "eval_seed=${POLICY_EVAL_LAYOUT_SEED}" "eval_offset=${POLICY_EVAL_LAYOUT_OFFSET}" \
  "promotion_drop=${PROMOTION_MAX_SUCCESS_DROP}" "promotion_min=${PROMOTION_MIN_SUCCESS_RATE}" \
  "env_cfg=${ENV_CFG_TYPE}" "action_type=${ACTION_TYPE}" "seed=${SEED}" \
  "remote_host=${REMOTE_HOST}" "remote_repo=${REMOTE_REPO_ROOT}" \
  "remote_work=${REMOTE_WORK_ROOT}" "remote_policy_gpu=${REMOTE_POLICY_GPU}" \
  "remote_env_gpu=${REMOTE_ENV_GPU}" "remote_policy_eval=${REMOTE_POLICY_EVAL}")

FIXED_NORM_STATS="${RUN_ROOT}/fixed_norm_stats"
NORM_FP=$(stage_fingerprint fixed_norm "run=${RUN_CONFIG_FP}" "asset_id=${NORM_ASSET_ID}")
ACTIVE_STAGE_FP="${NORM_FP}"
if [[ "${RESUME_RUN}" == "1" ]] && artifact_complete norm "${FIXED_NORM_STATS}" 0; then
  echo "[RECAP resume] reusing fixed initial-checkpoint normalization"
else
  archive_incomplete "${FIXED_NORM_STATS}"
  "${PI_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_fixed_pi05_norm_stats.py" \
    --checkpoint "${INITIAL_POLICY_CHECKPOINT}" --output "${FIXED_NORM_STATS}" \
    --asset-id "${NORM_ASSET_ID}"
  mark_artifact norm "${FIXED_NORM_STATS}"
fi

CURRENT_POLICY="${INITIAL_POLICY_CHECKPOINT}"
PREVIOUS_WCM="${INITIAL_WCM_CHECKPOINT}"
ROLLOUT_ROOTS=()
PREVIOUS_PI_DATASET=""
PREVIOUS_BUFFER=""
PREVIOUS_WCM_INPUT_EPISODES=0

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  ACTIVE_STAGE_FP=$(stage_fingerprint "iteration_${iteration}" "run=${RUN_CONFIG_FP}" "iteration=${iteration}")
  ITER_DIR=$(printf '%s/iteration_%02d' "${RUN_ROOT}" "${iteration}")
  RAW_ROLLOUTS="${ITER_DIR}/rollouts"
  BUFFER_ROOT="${ITER_DIR}/replay_buffer"
  WCM_BUFFER="${ITER_DIR}/wcm_training_buffer"
  WCM_OUTPUT="${ITER_DIR}/wcm"
  ADVANTAGES="${ITER_DIR}/recap_advantages.jsonl"
  REPO_ID="RoboDojo-recap-${TASK_SLUG}-iter-${iteration}"
  POLICY_OUTPUT="${ITER_DIR}/pi05"
  NORM_STATS="${FIXED_NORM_STATS}"
  PI_DATASET="${LEROBOT_HOME}/${REPO_ID}"
  REBUILD_DOWNSTREAM=0
  mkdir -p "${ITER_DIR}"

  if (( ! REMOTE_ENABLED )); then
  # Every update first gathers experience from the policy it is about to
  # improve. This guarantees that iteration 1 already contains rollout
  # failures instead of degenerating into positive-only conditional SFT.
  if [[ "${RESUME_RUN}" == "1" ]] && artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}"; then
    reuse_stage rollout
  else
    if [[ -e "${RAW_ROLLOUTS}" ]] && \
       ! artifact_fingerprint_matches rollout "${RAW_ROLLOUTS}"; then
      archive_incomplete "${RAW_ROLLOUTS}"
    fi
    mkdir -p "${RAW_ROLLOUTS}/episodes"
    # Mark the request before collection so an interrupted rollout can resume
    # only when the full configuration fingerprint still matches.
    mark_artifact rollout "${RAW_ROLLOUTS}"
    if [[ -e "${RAW_ROLLOUTS}/_in_progress" ]]; then
      archive_incomplete "${RAW_ROLLOUTS}/_in_progress"
    fi
    recorded_episodes=$(find "${RAW_ROLLOUTS}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    recorded_episodes="${recorded_episodes//[[:space:]]/}"
    layout_offset=$("${WCM_PYTHON_BIN}" -c \
      'import json,sys; from pathlib import Path; ids=[int(json.loads(p.read_text())["layout_id"]) for p in Path(sys.argv[1]).glob("*/manifest.json")]; print(max(ids, default=-1) + 1)' \
      "${RAW_ROLLOUTS}/episodes")
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
      # RoboDojo's eval seed selects Assets/Eval_Layout/.../<seed>. It is not
      # an unconstrained RNG seed and must not track the training iteration.
      (
        ROBODOJO_DISABLE_PROGRESS=1 \
          bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
            --policy-dir "${POLICY_DIR}" \
            --task "${TASK_NAME}" \
            --ckpt "${CURRENT_POLICY}" \
            --env-cfg "${ENV_CFG_TYPE}" \
            --action-type "${ACTION_TYPE}" \
            --seed "${ROLLOUT_LAYOUT_SEED}" \
            --policy-gpu "${POLICY_GPU}" \
            --env-gpu "${ENV_GPU}" \
            --policy-env "${POLICY_ENV}" \
            --eval-env "${EVAL_ENV}" \
            --eval-num "${remaining_episodes}" \
            --layout-offset "${layout_offset}" \
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
    artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}" "" || {
      echo "Expected ${ROLLOUT_EPISODES} complete rollout episodes below ${RAW_ROLLOUTS}" >&2
      exit 1
    }
    mark_artifact rollout "${RAW_ROLLOUTS}"
    REBUILD_DOWNSTREAM=1
  fi
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/check_recap_rollouts.py" \
    --root "${RAW_ROLLOUTS}" --expected "${ROLLOUT_EPISODES}" \
    --min-successes "${MIN_SUCCESS_EPISODES}" --min-failures "${MIN_FAILURE_EPISODES}" \
    --output "${RAW_ROLLOUTS}/quality.json"
  ROLLOUT_ROOTS+=("${RAW_ROLLOUTS}")

  OLD_BUFFER_EPISODES=0
  if [[ -n "${PREVIOUS_BUFFER}" ]]; then
    OLD_BUFFER_EPISODES=$("${WCM_PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
      "${PREVIOUS_BUFFER}/meta/info.json")
  fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete buffer "${BUFFER_ROOT}"; then
    reuse_stage replay_buffer
  else
    archive_incomplete "${BUFFER_ROOT}"
    echo "[RECAP ${iteration}/${ITERATIONS}] aggregating SFT plus ${#ROLLOUT_ROOTS[@]} completed rollout rounds"
    start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP replay-buffer and WCM dataset preparation"
    if [[ -n "${PREVIOUS_BUFFER}" ]]; then
      "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer_incremental.py" \
        --previous-buffer "${PREVIOUS_BUFFER}" \
        --rollout-root "${RAW_ROLLOUTS}" \
        --output "${BUFFER_ROOT}" \
        --task "${TASK_NAME}" \
        --seed "$((SEED + iteration))"
    else
      BUFFER_ARGS=(
        --demo-root "${DEMO_ROOT}"
        --output "${BUFFER_ROOT}"
        --task "${TASK_NAME}"
        --max-demo-episodes "${MAX_DEMO_EPISODES}"
        --seed "$((SEED + iteration))"
      )
      for source in "${ROLLOUT_ROOTS[@]}"; do BUFFER_ARGS+=(--rollout-root "${source}"); done
      "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer.py" "${BUFFER_ARGS[@]}"
    fi
    mark_artifact buffer "${BUFFER_ROOT}"
    REBUILD_DOWNSTREAM=1
  fi
  PREVIOUS_BUFFER="${BUFFER_ROOT}"
  BUFFER_EPISODES=$("${WCM_PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
    "${BUFFER_ROOT}/meta/info.json")

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete wcm "${WCM_OUTPUT}"; then
    reuse_stage wcm_training_buffer_not_needed
  elif [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete buffer "${WCM_BUFFER}"; then
    reuse_stage wcm_training_buffer
  else
    archive_incomplete "${WCM_BUFFER}"
    echo "[RECAP ${iteration}/${ITERATIONS}] WCM subset: replay up to ${WCM_REPLAY_EPISODES} old + all $((BUFFER_EPISODES - OLD_BUFFER_EPISODES)) new episodes"
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_wcm_training_subset.py" \
      --buffer "${BUFFER_ROOT}" \
      --output "${WCM_BUFFER}" \
      --old-episode-count "${OLD_BUFFER_EPISODES}" \
      --replay-episodes "${WCM_REPLAY_EPISODES}" \
      --seed "$((SEED + iteration))"
    mark_artifact buffer "${WCM_BUFFER}"
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
      WCM_DATASET_ROOT="${WCM_BUFFER}"
      WCM_SUCCESS_LABELS="${WCM_BUFFER}/meta/success_labels.json"
      WCM_ASSUME_SUCCESS=0
      WCM_FAILURE_PENALTY="${FAILURE_PENALTY}"
      WCM_GAMMA="${GAMMA}"
      WCM_OUTPUT_DIR="${WCM_OUTPUT}"
      WCM_EPOCHS="${RECAP_WCM_EPOCHS:-5}"
      CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}"
      WCM_RESUME="${WCM_STAGE_RESUME}"
      WCM_INIT_CHECKPOINT=
    )
    if [[ -z "${WCM_STAGE_RESUME}" && -n "${PREVIOUS_WCM}" ]]; then
      WCM_ENV+=(WCM_INIT_CHECKPOINT="${PREVIOUS_WCM}")
    fi
    stop_gpu_reservation
    env "${WCM_ENV[@]}" bash "${SCRIPT_DIR}/run_wcm.sh" --task "${TASK_NAME}"
    mark_artifact wcm "${WCM_OUTPUT}"
    REBUILD_DOWNSTREAM=1
  fi
  PREVIOUS_WCM="${WCM_OUTPUT}/deploy.pt"
  [[ -f "${PREVIOUS_WCM}" ]] || { echo "WCM deploy checkpoint missing: ${PREVIOUS_WCM}" >&2; exit 1; }

  else
    # Pipeline the simulator one round ahead of the critic. WCM iteration 1
    # sees demonstrations only; later WCM updates see rollouts through i-1.
    ROLLOUT_PENDING=0
    if [[ "${RESUME_RUN}" == "1" ]] && artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}"; then
      reuse_stage rollout
    else
      archive_incomplete "${RAW_ROLLOUTS}"
      ROLLOUT_LOG="${ITER_DIR}/remote_rollout.log"
      REMOTE_JOB_ID="${TASK_SLUG}-${RUN_CONFIG_FP:0:12}-iter-$(printf '%02d' "${iteration}")"
      echo "[RECAP ${iteration}/${ITERATIONS}] launching remote rollout job=${REMOTE_JOB_ID}"
      (
        "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/remote_recap.py" rollout \
          --host "${REMOTE_HOST}" --remote-repo-root "${REMOTE_REPO_ROOT}" \
          --remote-work-root "${REMOTE_WORK_ROOT}" --job-id "${REMOTE_JOB_ID}" \
          --checkpoint "${CURRENT_POLICY}" --output "${RAW_ROLLOUTS}" \
          --task "${TASK_NAME}" --episodes "${ROLLOUT_EPISODES}" \
          --layout-seed "${ROLLOUT_LAYOUT_SEED}" \
          --policy-gpu "${REMOTE_POLICY_GPU}" --env-gpu "${REMOTE_ENV_GPU}" \
          --env-cfg "${ENV_CFG_TYPE}" --action-type "${ACTION_TYPE}" \
          --policy-env "${POLICY_ENV}" --eval-env "${EVAL_ENV}"
      ) >"${ROLLOUT_LOG}" 2>&1 &
      ACTIVE_ROLLOUT_PID=$!
      ROLLOUT_PENDING=1
    fi

    if [[ -n "${PREVIOUS_BUFFER}" ]]; then
      WCM_INPUT_BUFFER="${PREVIOUS_BUFFER}"
    else
      WCM_INPUT_BUFFER="${ITER_DIR}/demo_replay_buffer"
      if [[ "${RESUME_RUN}" == "1" ]] && artifact_complete buffer "${WCM_INPUT_BUFFER}"; then
        reuse_stage demo_replay_buffer
      else
        archive_incomplete "${WCM_INPUT_BUFFER}"
        echo "[RECAP ${iteration}/${ITERATIONS}] building demonstration-only first WCM buffer"
        start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP demonstration buffer preparation"
        "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer.py" \
          --demo-root "${DEMO_ROOT}" --output "${WCM_INPUT_BUFFER}" \
          --task "${TASK_NAME}" --max-demo-episodes "${MAX_DEMO_EPISODES}" \
          --seed "$((SEED + iteration))"
        mark_artifact buffer "${WCM_INPUT_BUFFER}"
        REBUILD_DOWNSTREAM=1
      fi
    fi
    WCM_INPUT_EPISODES=$("${WCM_PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
      "${WCM_INPUT_BUFFER}/meta/info.json")

    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete wcm "${WCM_OUTPUT}"; then
      reuse_stage wcm_training_buffer_not_needed
    elif [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete buffer "${WCM_BUFFER}"; then
      reuse_stage wcm_training_buffer
    else
      archive_incomplete "${WCM_BUFFER}"
      echo "[RECAP ${iteration}/${ITERATIONS}] lagged WCM subset: replay up to ${WCM_REPLAY_EPISODES} old + all $((WCM_INPUT_EPISODES - PREVIOUS_WCM_INPUT_EPISODES)) newly available episodes"
      "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_wcm_training_subset.py" \
        --buffer "${WCM_INPUT_BUFFER}" --output "${WCM_BUFFER}" \
        --old-episode-count "${PREVIOUS_WCM_INPUT_EPISODES}" \
        --replay-episodes "${WCM_REPLAY_EPISODES}" --seed "$((SEED + iteration))"
      mark_artifact buffer "${WCM_BUFFER}"
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
      echo "[RECAP ${iteration}/${ITERATIONS}] training lagged WCM while remote rollout runs"
      start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "WCM dataset preparation"
      WCM_ENV=(
        PYTHON_BIN="${WCM_PYTHON_BIN}"
        WCM_DATASET_ROOT="${WCM_BUFFER}"
        WCM_SUCCESS_LABELS="${WCM_BUFFER}/meta/success_labels.json"
        WCM_ASSUME_SUCCESS=0
        WCM_FAILURE_PENALTY="${FAILURE_PENALTY}"
        WCM_GAMMA="${GAMMA}"
        WCM_OUTPUT_DIR="${WCM_OUTPUT}"
        WCM_EPOCHS="${RECAP_WCM_EPOCHS:-5}"
        CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}"
        WCM_RESUME="${WCM_STAGE_RESUME}"
        WCM_INIT_CHECKPOINT=
      )
      if [[ -z "${WCM_STAGE_RESUME}" && -n "${PREVIOUS_WCM}" ]]; then
        WCM_ENV+=(WCM_INIT_CHECKPOINT="${PREVIOUS_WCM}")
      fi
      stop_gpu_reservation
      env "${WCM_ENV[@]}" bash "${SCRIPT_DIR}/run_wcm.sh" --task "${TASK_NAME}"
      mark_artifact wcm "${WCM_OUTPUT}"
      REBUILD_DOWNSTREAM=1
    fi
    PREVIOUS_WCM="${WCM_OUTPUT}/deploy.pt"
    [[ -f "${PREVIOUS_WCM}" ]] || { echo "WCM deploy checkpoint missing: ${PREVIOUS_WCM}" >&2; exit 1; }

    if (( ROLLOUT_PENDING )); then
      echo "[RECAP ${iteration}/${ITERATIONS}] WCM finished; waiting for remote rollout transfer"
      rollout_status=0
      wait "${ACTIVE_ROLLOUT_PID}" || rollout_status=$?
      ACTIVE_ROLLOUT_PID=""
      if (( rollout_status != 0 )); then
        echo "Remote RECAP rollout failed; log follows:" >&2
        tail -n 120 "${ROLLOUT_LOG}" >&2 || true
        exit "${rollout_status}"
      fi
      artifact_complete rollout "${RAW_ROLLOUTS}" "${ROLLOUT_EPISODES}" "" || {
        echo "Remote transfer did not produce ${ROLLOUT_EPISODES} complete episodes: ${RAW_ROLLOUTS}" >&2
        exit 1
      }
      mark_artifact rollout "${RAW_ROLLOUTS}"
      REBUILD_DOWNSTREAM=1
    fi
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/check_recap_rollouts.py" \
      --root "${RAW_ROLLOUTS}" --expected "${ROLLOUT_EPISODES}" \
      --min-successes "${MIN_SUCCESS_EPISODES}" --min-failures "${MIN_FAILURE_EPISODES}" \
      --output "${RAW_ROLLOUTS}/quality.json"

    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete buffer "${BUFFER_ROOT}"; then
      reuse_stage replay_buffer
    else
      archive_incomplete "${BUFFER_ROOT}"
      echo "[RECAP ${iteration}/${ITERATIONS}] appending returned rollout to the replay buffer"
      start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP replay-buffer preparation"
      "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer_incremental.py" \
        --previous-buffer "${WCM_INPUT_BUFFER}" --rollout-root "${RAW_ROLLOUTS}" \
        --output "${BUFFER_ROOT}" --task "${TASK_NAME}" --seed "$((SEED + iteration))"
      mark_artifact buffer "${BUFFER_ROOT}"
      REBUILD_DOWNSTREAM=1
    fi
    PREVIOUS_WCM_INPUT_EPISODES="${WCM_INPUT_EPISODES}"
    PREVIOUS_BUFFER="${BUFFER_ROOT}"
    BUFFER_EPISODES=$("${WCM_PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
      "${BUFFER_ROOT}/meta/info.json")
  fi

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
    mark_artifact advantages "${ADVANTAGES}"
    REBUILD_DOWNSTREAM=1
  fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && \
     artifact_complete pi_dataset "${PI_DATASET}" "${BUFFER_EPISODES}"; then
    reuse_stage pi_dataset
  else
    if [[ -e "${PI_DATASET}" ]] && \
       ! artifact_fingerprint_matches pi_dataset "${PI_DATASET}"; then
      archive_incomplete "${PI_DATASET}"
    fi
    echo "[RECAP ${iteration}/${ITERATIONS}] incrementally updating advantage-conditioned Pi0.5 dataset"
    start_gpu_reservation "${TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RECAP Pi0.5 dataset preparation"
    PI_DATASET_ARGS=(
      --dataset-root "${BUFFER_ROOT}"
      --repo-id "${REPO_ID}"
      --advantage-labels "${ADVANTAGES}"
      --task "${TASK_NAME}"
      --mode "${OPENPI_DATA_MODE:-video}"
    )
    if [[ ! -e "${PI_DATASET}" && -n "${PREVIOUS_PI_DATASET}" ]]; then
      PI_DATASET_ARGS+=(--previous-dataset "${PREVIOUS_PI_DATASET}")
    fi
    HF_LEROBOT_HOME="${LEROBOT_HOME}" "${PI_PYTHON_BIN}" \
      "${SCRIPT_DIR}/prepare_pi05_recap_dataset.py" "${PI_DATASET_ARGS[@]}"
    mark_artifact pi_dataset "${PI_DATASET}"
    REBUILD_DOWNSTREAM=1
  fi
  PREVIOUS_PI_DATASET="${PI_DATASET}"

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
    --recap-demo-weight "${DEMO_SAMPLING_WEIGHT}"
    --recap-rollout-weight "${ROLLOUT_SAMPLING_WEIGHT}"
    --seed "$((SEED + iteration))"
    --fsdp-devices "${FSDP_DEVICES}"
    --warmup-steps "${POLICY_WARMUP_STEPS}"
    --save-interval "${POLICY_EVAL_INTERVAL}"
  )
  for option in batch-size num-workers log-interval; do
    variable="OPENPI_${option^^}"
    variable="${variable//-/_}"
    if [[ -n "${!variable:-}" ]]; then TRAIN_ARGS+=("--${option}" "${!variable}"); fi
  done
  if [[ -n "${OPENPI_LEARNING_RATE:-}" ]]; then TRAIN_ARGS+=(--learning-rate "${OPENPI_LEARNING_RATE}"); fi
  if [[ -n "${OPENPI_DECAY_LR:-}" ]]; then TRAIN_ARGS+=(--decay-lr "${OPENPI_DECAY_LR}"); fi
  if [[ -n "${OPENPI_WEIGHT_DECAY:-}" ]]; then TRAIN_ARGS+=(--weight-decay "${OPENPI_WEIGHT_DECAY}"); fi
  if [[ -n "${OPENPI_CLIP_GRADIENT_NORM:-}" ]]; then TRAIN_ARGS+=(--clip-gradient-norm "${OPENPI_CLIP_GRADIENT_NORM}"); fi
  if [[ "${OPENPI_WANDB_ENABLED:-1}" == "0" ]]; then TRAIN_ARGS+=(--disable-wandb); fi

  if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && \
     artifact_complete policy "${POLICY_OUTPUT}" "${NUM_TRAIN_STEPS}"; then
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
    artifact_complete policy "${POLICY_OUTPUT}" "${NUM_TRAIN_STEPS}" "" || {
      echo "Pi0.5 did not produce required final checkpoint step $((NUM_TRAIN_STEPS - 1))" >&2
      exit 1
    }
    mark_artifact policy "${POLICY_OUTPUT}"
    REBUILD_DOWNSTREAM=1
  fi

  EVAL_ROOT="${ITER_DIR}/policy_evaluations"
  BASELINE_EVAL="${EVAL_ROOT}/baseline"
  evaluate_policy_checkpoint "${TRAIN_INIT}" "${BASELINE_EVAL}" "baseline"
  SELECT_ARGS=(
    --baseline-checkpoint "${TRAIN_INIT}"
    --baseline-rollouts "${BASELINE_EVAL}"
    --max-success-drop "${PROMOTION_MAX_SUCCESS_DROP}"
    --min-success-rate "${PROMOTION_MIN_SUCCESS_RATE}"
    --output "${ITER_DIR}/promotion.json"
  )
  mapfile -t POLICY_CANDIDATES < <(
    find "${POLICY_OUTPUT}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%f\n' | sort -n
  )
  (( ${#POLICY_CANDIDATES[@]} >= 2 )) || {
    echo "Expected at least one intermediate and one final policy checkpoint; " \
         "reduce RECAP_POLICY_EVAL_INTERVAL=${POLICY_EVAL_INTERVAL}." >&2
    exit 1
  }
  for candidate_step in "${POLICY_CANDIDATES[@]}"; do
    candidate_checkpoint="${POLICY_OUTPUT}/${candidate_step}"
    candidate_eval="${EVAL_ROOT}/step_${candidate_step}"
    evaluate_policy_checkpoint "${candidate_checkpoint}" "${candidate_eval}" "step ${candidate_step}"
    SELECT_ARGS+=(--candidate "${candidate_step}::${candidate_checkpoint}::${candidate_eval}")
  done
  promotion_status=0
  if CURRENT_POLICY=$("${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/select_recap_policy.py" "${SELECT_ARGS[@]}"); then
    echo "[RECAP promotion] selected ${CURRENT_POLICY}"
  else
    promotion_status=$?
    CURRENT_POLICY="${TRAIN_INIT}"
    echo "[RECAP promotion] rejected iteration ${iteration}; retaining ${CURRENT_POLICY}" >&2
    if (( promotion_status != 3 )); then exit "${promotion_status}"; fi
  fi
  printf '%s\n' "${CURRENT_POLICY}" > "${RUN_ROOT}/latest_policy.txt"
  printf '%s\n' "${PREVIOUS_WCM}" > "${RUN_ROOT}/latest_wcm.txt"

  if (( VALUE_VIDEO_EPISODES > 0 && (iteration == ITERATIONS || promotion_status == 3) )); then
    if [[ "${RESUME_RUN}" == "1" ]] && (( REBUILD_DOWNSTREAM == 0 )) && artifact_complete value_videos "${ITER_DIR}/value_videos" "${VALUE_VIDEO_EPISODES}"; then
      reuse_stage value_videos
    else
      archive_incomplete "${ITER_DIR}/value_videos"
      echo "[RECAP ${iteration}/${ITERATIONS}] rendering ${VALUE_VIDEO_EPISODES} rollout videos with WCM value overlays"
      if (( REMOTE_ENABLED )); then
        "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/remote_recap.py" value-video \
          --host "${REMOTE_HOST}" --remote-repo-root "${REMOTE_REPO_ROOT}" \
          --remote-work-root "${REMOTE_WORK_ROOT}" \
          --job-id "${TASK_SLUG}-${RUN_CONFIG_FP:0:12}-iter-$(printf '%02d' "${iteration}")" \
          --wcm-checkpoint "${PREVIOUS_WCM}" --output "${ITER_DIR}/value_videos" \
          --episodes "${VALUE_VIDEO_EPISODES}" --gpu "${REMOTE_VALUE_VIDEO_GPU}" \
          --batch-size "${RECAP_VALUE_VIDEO_BATCH_SIZE:-16}" \
          --device "${RECAP_VALUE_VIDEO_DEVICE:-cuda}" \
          --precision "${RECAP_VALUE_VIDEO_PRECISION:-bf16}" \
          --backend "${RECAP_VALUE_VIDEO_BACKEND:-auto}" \
          --speed "${RECAP_VALUE_VIDEO_SPEED:-1.0}" \
          --y-min "${RECAP_VALUE_VIDEO_Y_MIN:--1.0}" \
          --y-max "${RECAP_VALUE_VIDEO_Y_MAX:-1.0}" \
          --title "WCM RECAP ITER ${iteration}"
      else
        CUDA_VISIBLE_DEVICES="${VALUE_VIDEO_GPU}" \
          "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/render_rollout_value_videos.py" \
            --wcm-checkpoint "${PREVIOUS_WCM}" \
            --rollout-root "${RAW_ROLLOUTS}" \
            --output-dir "${ITER_DIR}/value_videos" \
            --max-episodes "${VALUE_VIDEO_EPISODES}" \
            --batch-size "${RECAP_VALUE_VIDEO_BATCH_SIZE:-16}" \
            --device "${RECAP_VALUE_VIDEO_DEVICE:-cuda}" \
            --precision "${RECAP_VALUE_VIDEO_PRECISION:-bf16}" \
            --backend "${RECAP_VALUE_VIDEO_BACKEND:-auto}" \
            --speed "${RECAP_VALUE_VIDEO_SPEED:-1.0}" \
            --y-min "${RECAP_VALUE_VIDEO_Y_MIN:--1.0}" \
            --y-max "${RECAP_VALUE_VIDEO_Y_MAX:-1.0}" \
            --title "WCM RECAP ITER ${iteration}"
      fi
      mark_artifact value_videos "${ITER_DIR}/value_videos"
      REBUILD_DOWNSTREAM=1
    fi
  fi
  if (( promotion_status == 3 && STOP_ON_REJECTION == 1 )); then
    echo "RECAP stopped safely because no candidate passed the promotion gate." >&2
    exit 3
  fi
done

echo "RECAP complete"
echo "policy=${CURRENT_POLICY}"
echo "wcm=${PREVIOUS_WCM}"
stop_gpu_reservation
