#!/usr/bin/env bash
# Iterated off-policy WCM + Pi0.5 RL Token actor training on RoboDojo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/gpu_reservation.sh"
source "${SCRIPT_DIR}/posttrain_config.sh"
install_gpu_reservation_exit_trap
find_posttrain_config "$@"
load_posttrain_config "${POSTTRAIN_CONFIG_FILE}"

detect_gpu_ids() {
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  visible="${visible//[[:space:]]/}"
  if [[ -n "${visible}" && "${visible}" != "NoDevFiles" ]]; then
    printf '%s\n' "${visible}"
    return
  fi
  local -a detected=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t detected < <(
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null
    )
  fi
  if (( ${#detected[@]} == 0 )); then
    printf '0\n'
  else
    local IFS=','
    printf '%s\n' "${detected[*]}"
  fi
}

DETECTED_GPUS="$(detect_gpu_ids)"
ACTIVE_ROLLOUT_PIDS=()

kill_process_tree() {
  local parent_pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    kill_process_tree "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
  kill -TERM "${parent_pid}" 2>/dev/null || true
}

interrupt_rollout_workers() {
  trap - INT TERM
  if (( ${#ACTIVE_ROLLOUT_PIDS[@]} > 0 )); then
    echo "[RLToken RECAP rollout] stopping active workers" >&2
    local worker_pid
    for worker_pid in "${ACTIVE_ROLLOUT_PIDS[@]}"; do
      kill_process_tree "${worker_pid}"
    done
    wait "${ACTIVE_ROLLOUT_PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap interrupt_rollout_workers INT TERM

PI_DIR="${ROOT_DIR}/XPolicyLab/policy/Pi_05"
WCM_PYTHON_BIN="${WCM_PYTHON_BIN:-${ROOT_DIR}/external_dependencies/WCM/.venv/bin/python}"
POLICY_DIR="${POLICY_DIR:-${PI_DIR}}"
POLICY_ENV="${POLICY_ENV:-openpi}"
EVAL_ENV="${EVAL_ENV:-RoboDojo}"
TASK_NAME="${TASK_NAME:-}"
DEMO_ROOT="${DEMO_ROOT:-${ROOT_DIR}/data/RoboDojo_lerobot_v21_video}"
BASE_POLICY_CHECKPOINT="${BASE_POLICY_CHECKPOINT:-${INITIAL_POLICY_CHECKPOINT:-}}"
INITIAL_WCM_CHECKPOINT="${INITIAL_WCM_CHECKPOINT:-}"
INITIAL_ACTOR_CHECKPOINT="${INITIAL_ACTOR_CHECKPOINT:-}"
ENCODER_RESUME="${RLTOKEN_ENCODER_RESUME:-}"
BC_RESUME="${RLTOKEN_BC_RESUME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/rltoken_recap}"
ITERATIONS="${RLTOKEN_RECAP_ITERATIONS:-1}"
ROLLOUT_EPISODES="${RLTOKEN_RECAP_ROLLOUT_EPISODES:-40}"
ENV_CFG_TYPE="${ENV_CFG_TYPE:-arx_x5}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
ACTOR_OBJECTIVE="${RLTOKEN_OBJECTIVE:-wcm_actor}"
ACTOR_MODE="${RLTOKEN_ACTOR_MODE:-direct}"
WCM_TRAIN_GPUS="${WCM_TRAIN_GPUS:-${DETECTED_GPUS}}"
ACTOR_TRAIN_GPUS="${ACTOR_TRAIN_GPUS:-${WCM_TRAIN_GPUS}}"
ROLLOUT_GPUS="${RLTOKEN_ROLLOUT_GPUS:-${ACTOR_TRAIN_GPUS}}"
ROLLOUT_ENVS_PER_WORKER="${RLTOKEN_ROLLOUT_ENVS_PER_WORKER:-4}"
MAX_DEMO_EPISODES="${RECAP_MAX_DEMO_EPISODES:-100}"
WCM_REPLAY_EPISODES="${RLTOKEN_RECAP_WCM_REPLAY_EPISODES:-${RECAP_WCM_REPLAY_EPISODES:-20}}"
SEED="${SEED:-0}"
ROLLOUT_LAYOUT_SEED="${RLTOKEN_ROLLOUT_LAYOUT_SEED:-0}"
GAMMA="${RECAP_GAMMA:-1.0}"
FAILURE_PENALTY="${WCM_FAILURE_PENALTY:-300}"

usage() {
  cat <<'EOF'
Usage: bash scripts/posttrain/run_pi05_rltoken_recap.sh [options]

Required:
  --task TASK                       RoboDojo task slug or instruction
  --base-policy-checkpoint CKPT     Frozen Pi0.5 SFT checkpoint/path

Options:
  --config PATH                     Flat YAML hyperparameter config
  --demo-root PATH                  Successful SFT LeRobot-v2.1 dataset
  --initial-wcm-checkpoint PATH     Warm-start WCM weights and action statistics
  --initial-actor-checkpoint PATH   Full actor/encoder training resume
  --encoder-resume PATH             Standalone encoder initialization/resume
  --bc-resume PATH                  Standalone behavior-cloned actor resume
  --output-root PATH                Output root (default: outputs/rltoken_recap)
  --iterations N                    Off-policy iterations
  --rollout-episodes N              Simulator episodes after each actor update
  --objective wcm_actor|rltoken     Actor update objective
  --actor-mode direct|residual      Fresh actor parameterization
  --wcm-train-gpus IDS              WCM DDP GPU list
  --actor-train-gpus IDS            Actor DDP GPU list
  --rollout-gpus IDS                GPUs paired as policy,Isaac workers
  --rollout-envs-per-worker N       Vectorized Isaac envs in each worker
  --wcm-replay-episodes K           Old episodes sampled for each later WCM update

Additional controls are documented in
configs/posttrain/pi05_rltoken_recap.yaml.example.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift 2 ;;
    --task) TASK_NAME="$2"; shift 2 ;;
    --base-policy-checkpoint) BASE_POLICY_CHECKPOINT="$2"; shift 2 ;;
    --demo-root) DEMO_ROOT="$2"; shift 2 ;;
    --initial-wcm-checkpoint) INITIAL_WCM_CHECKPOINT="$2"; shift 2 ;;
    --initial-actor-checkpoint) INITIAL_ACTOR_CHECKPOINT="$2"; shift 2 ;;
    --encoder-resume) ENCODER_RESUME="$2"; shift 2 ;;
    --bc-resume) BC_RESUME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --rollout-episodes) ROLLOUT_EPISODES="$2"; shift 2 ;;
    --objective) ACTOR_OBJECTIVE="$2"; shift 2 ;;
    --actor-mode) ACTOR_MODE="$2"; shift 2 ;;
    --wcm-train-gpus) WCM_TRAIN_GPUS="$2"; shift 2 ;;
    --actor-train-gpus) ACTOR_TRAIN_GPUS="$2"; shift 2 ;;
    --rollout-gpus) ROLLOUT_GPUS="$2"; shift 2 ;;
    --rollout-envs-per-worker) ROLLOUT_ENVS_PER_WORKER="$2"; shift 2 ;;
    --wcm-replay-episodes) WCM_REPLAY_EPISODES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${TASK_NAME}" ]] || { echo "--task is required" >&2; exit 2; }
[[ -n "${BASE_POLICY_CHECKPOINT}" ]] || { echo "--base-policy-checkpoint is required" >&2; exit 2; }
[[ -x "${WCM_PYTHON_BIN}" ]] || { echo "WCM Python not found: ${WCM_PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DEMO_ROOT}/meta/info.json" ]] || { echo "Demo dataset not found: ${DEMO_ROOT}" >&2; exit 1; }
for checkpoint in "${INITIAL_WCM_CHECKPOINT}" "${INITIAL_ACTOR_CHECKPOINT}" "${ENCODER_RESUME}" "${BC_RESUME}"; do
  [[ -z "${checkpoint}" || -f "${checkpoint}" ]] || { echo "Checkpoint not found: ${checkpoint}" >&2; exit 1; }
done
if [[ -n "${INITIAL_ACTOR_CHECKPOINT}" && ( -n "${ENCODER_RESUME}" || -n "${BC_RESUME}" ) ]]; then
  echo "Use --initial-actor-checkpoint or standalone encoder/BC resumes, not both" >&2
  exit 2
fi
[[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || { echo "--iterations must be positive" >&2; exit 2; }
[[ "${ROLLOUT_EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "--rollout-episodes must be positive" >&2; exit 2; }
[[ "${ROLLOUT_ENVS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--rollout-envs-per-worker must be positive" >&2
  exit 2
}
[[ "${MAX_DEMO_EPISODES}" =~ ^[0-9]+$ ]] || { echo "RECAP_MAX_DEMO_EPISODES must be non-negative" >&2; exit 2; }
[[ "${WCM_REPLAY_EPISODES}" =~ ^[0-9]+$ ]] || { echo "--wcm-replay-episodes must be non-negative" >&2; exit 2; }
[[ "${ROLLOUT_LAYOUT_SEED}" =~ ^[0-9]+$ ]] || {
  echo "RLTOKEN_ROLLOUT_LAYOUT_SEED must be non-negative" >&2
  exit 2
}
[[ "${ACTOR_OBJECTIVE}" == "wcm_actor" || "${ACTOR_OBJECTIVE}" == "rltoken" ]] || {
  echo "--objective must be wcm_actor or rltoken" >&2
  exit 2
}
[[ "${ACTOR_MODE}" == "direct" || "${ACTOR_MODE}" == "residual" ]] || {
  echo "--actor-mode must be direct or residual" >&2
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

declare -a WCM_GPU_IDS ACTOR_GPU_IDS ROLLOUT_GPU_IDS
parse_gpu_ids "${WCM_TRAIN_GPUS}" WCM_GPU_IDS
parse_gpu_ids "${ACTOR_TRAIN_GPUS}" ACTOR_GPU_IDS
parse_gpu_ids "${ROLLOUT_GPUS}" ROLLOUT_GPU_IDS
WCM_TRAIN_GPUS=$(IFS=','; echo "${WCM_GPU_IDS[*]}")
ACTOR_TRAIN_GPUS=$(IFS=','; echo "${ACTOR_GPU_IDS[*]}")
ROLLOUT_GPUS=$(IFS=','; echo "${ROLLOUT_GPU_IDS[*]}")

declare -a ROLLOUT_POLICY_GPU_IDS ROLLOUT_ENV_GPU_IDS
if (( ${#ROLLOUT_GPU_IDS[@]} == 1 )); then
  ROLLOUT_POLICY_GPU_IDS=("${ROLLOUT_GPU_IDS[0]}")
  ROLLOUT_ENV_GPU_IDS=("${ROLLOUT_GPU_IDS[0]}")
else
  if (( ${#ROLLOUT_GPU_IDS[@]} % 2 == 1 )); then
    echo "[RLToken RECAP devices] odd rollout GPU count; leaving GPU ${ROLLOUT_GPU_IDS[-1]} unused"
  fi
  for ((gpu_index = 0; gpu_index + 1 < ${#ROLLOUT_GPU_IDS[@]}; gpu_index += 2)); do
    ROLLOUT_POLICY_GPU_IDS+=("${ROLLOUT_GPU_IDS[gpu_index]}")
    ROLLOUT_ENV_GPU_IDS+=("${ROLLOUT_GPU_IDS[gpu_index + 1]}")
  done
fi
ROLLOUT_WORKER_COUNT="${#ROLLOUT_POLICY_GPU_IDS[@]}"
if (( ROLLOUT_WORKER_COUNT > ROLLOUT_EPISODES )); then
  ROLLOUT_WORKER_COUNT="${ROLLOUT_EPISODES}"
  ROLLOUT_POLICY_GPU_IDS=("${ROLLOUT_POLICY_GPU_IDS[@]:0:ROLLOUT_WORKER_COUNT}")
  ROLLOUT_ENV_GPU_IDS=("${ROLLOUT_ENV_GPU_IDS[@]:0:ROLLOUT_WORKER_COUNT}")
fi

TASK_SLUG=$(printf '%s' "${TASK_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_*//; s/_*$//' | cut -c1-80)
RUN_ROOT="${OUTPUT_ROOT}/${TASK_SLUG}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "RL Token RECAP run already exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

echo "[RLToken RECAP devices] WCM=${WCM_TRAIN_GPUS} (${#WCM_GPU_IDS[@]} processes)"
echo "[RLToken RECAP devices] actor=${ACTOR_TRAIN_GPUS} (${#ACTOR_GPU_IDS[@]} processes)"
echo "[RLToken RECAP devices] rollout=${ROLLOUT_GPUS} (${ROLLOUT_WORKER_COUNT} policy/Isaac workers, ${ROLLOUT_ENVS_PER_WORKER} envs each)"
for ((worker = 0; worker < ROLLOUT_WORKER_COUNT; worker++)); do
  echo "[RLToken RECAP devices] worker ${worker}: policy=${ROLLOUT_POLICY_GPU_IDS[worker]}, Isaac=${ROLLOUT_ENV_GPU_IDS[worker]}"
done

CURRENT_ACTOR="${INITIAL_ACTOR_CHECKPOINT}"
PREVIOUS_WCM="${INITIAL_WCM_CHECKPOINT}"
ROLLOUT_ROOTS=()
PREVIOUS_BUFFER=""

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  ITER_DIR=$(printf '%s/iteration_%02d' "${RUN_ROOT}" "${iteration}")
  BUFFER_ROOT="${ITER_DIR}/replay_buffer"
  WCM_BUFFER="${ITER_DIR}/wcm_training_buffer"
  WCM_OUTPUT="${ITER_DIR}/wcm"
  ACTOR_OUTPUT="${ITER_DIR}/actor.pt"
  RAW_ROLLOUTS="${ITER_DIR}/rollouts"
  mkdir -p "${ITER_DIR}"

  OLD_BUFFER_EPISODES=0
  if [[ -n "${PREVIOUS_BUFFER}" ]]; then
    OLD_BUFFER_EPISODES=$("${WCM_PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
      "${PREVIOUS_BUFFER}/meta/info.json")
  fi
  echo "[RLToken RECAP ${iteration}/${ITERATIONS}] building SFT + rollout replay buffer"
  start_gpu_reservation "${WCM_TRAIN_GPUS}" "${WCM_PYTHON_BIN}" "RLToken replay-buffer and WCM dataset preparation"
  if [[ -n "${PREVIOUS_BUFFER}" ]]; then
    newest_rollout="${ROLLOUT_ROOTS[$((${#ROLLOUT_ROOTS[@]} - 1))]}"
    INCREMENTAL_BUFFER_ARGS=(
      --previous-buffer "${PREVIOUS_BUFFER}"
      --rollout-root "${newest_rollout}"
      --output "${BUFFER_ROOT}"
      --task "${TASK_NAME}"
      --seed "$((SEED + iteration))"
    )
    if [[ -n "${RECAP_MAX_ROLLOUT_EPISODES:-}" ]]; then
      INCREMENTAL_BUFFER_ARGS+=(--max-rollout-episodes "${RECAP_MAX_ROLLOUT_EPISODES}")
    fi
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer_incremental.py" \
      "${INCREMENTAL_BUFFER_ARGS[@]}"
  else
    BUFFER_ARGS=(
      --demo-root "${DEMO_ROOT}"
      --output "${BUFFER_ROOT}"
      --task "${TASK_NAME}"
      --seed "$((SEED + iteration))"
    )
    BUFFER_ARGS+=(--max-demo-episodes "${MAX_DEMO_EPISODES}")
    if [[ -n "${RECAP_MAX_ROLLOUT_EPISODES:-}" ]]; then
      BUFFER_ARGS+=(--max-rollout-episodes "${RECAP_MAX_ROLLOUT_EPISODES}")
    fi
    for source in "${ROLLOUT_ROOTS[@]}"; do BUFFER_ARGS+=(--rollout-root "${source}"); done
    "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_replay_buffer.py" "${BUFFER_ARGS[@]}"
  fi
  PREVIOUS_BUFFER="${BUFFER_ROOT}"

  BUFFER_EPISODES=$("${WCM_PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' \
    "${BUFFER_ROOT}/meta/info.json")
  echo "[RLToken RECAP ${iteration}/${ITERATIONS}] WCM subset: replay up to ${WCM_REPLAY_EPISODES} old + all $((BUFFER_EPISODES - OLD_BUFFER_EPISODES)) new episodes"
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/build_wcm_training_subset.py" \
    --buffer "${BUFFER_ROOT}" \
    --output "${WCM_BUFFER}" \
    --old-episode-count "${OLD_BUFFER_EPISODES}" \
    --replay-episodes "${WCM_REPLAY_EPISODES}" \
    --seed "$((SEED + iteration))"

  echo "[RLToken RECAP ${iteration}/${ITERATIONS}] updating WCM"
  WCM_ENV=(
    PYTHON_BIN="${WCM_PYTHON_BIN}"
    CUDA_VISIBLE_DEVICES="${WCM_TRAIN_GPUS}"
    WCM_DATASET_ROOT="${WCM_BUFFER}"
    WCM_SUCCESS_LABELS="${WCM_BUFFER}/meta/success_labels.json"
    WCM_ASSUME_SUCCESS=0
    WCM_FAILURE_PENALTY="${FAILURE_PENALTY}"
    WCM_GAMMA="${GAMMA}"
    WCM_OUTPUT_DIR="${WCM_OUTPUT}"
    WCM_EPOCHS="${RLTOKEN_RECAP_WCM_EPOCHS:-1}"
    WCM_RESUME=
    WCM_INIT_CHECKPOINT=
  )
  if [[ -n "${PREVIOUS_WCM}" ]]; then WCM_ENV+=(WCM_INIT_CHECKPOINT="${PREVIOUS_WCM}"); fi
  stop_gpu_reservation
  env "${WCM_ENV[@]}" bash "${SCRIPT_DIR}/run_wcm.sh" --task "${TASK_NAME}"
  PREVIOUS_WCM="${WCM_OUTPUT}/deploy.pt"
  [[ -f "${PREVIOUS_WCM}" ]] || { echo "WCM deploy checkpoint missing: ${PREVIOUS_WCM}" >&2; exit 1; }

  echo "[RLToken RECAP ${iteration}/${ITERATIONS}] updating ${ACTOR_OBJECTIVE} actor"
  ACTOR_ARGS=(
    --wcm-checkpoint "${PREVIOUS_WCM}"
    --dataset-root "${BUFFER_ROOT}"
    --task "${TASK_NAME}"
    --output "${ACTOR_OUTPUT}"
    --objective "${ACTOR_OBJECTIVE}"
    --chunk-steps "${RLTOKEN_CHUNK_STEPS:-3}"
    --token-dim "${RLTOKEN_TOKEN_DIM:-256}"
    --actor-hidden-dim "${RLTOKEN_ACTOR_HIDDEN_DIM:-256}"
    --actor-layers "${RLTOKEN_ACTOR_LAYERS:-2}"
    --actor-mode "${ACTOR_MODE}"
    --fixed-std "${RLTOKEN_FIXED_STD:-0.04}"
    --action-low "${RLTOKEN_ACTION_LOW:--5.0}"
    --action-high "${RLTOKEN_ACTION_HIGH:-5.0}"
    --encoder-warmup-steps "${RLTOKEN_ENCODER_WARMUP_STEPS:-1000}"
    --bc-init-steps "${RLTOKEN_BC_INIT_STEPS:-1000}"
    --encoder-checkpoint "${ITER_DIR}/encoder.pt"
    --bc-checkpoint "${ITER_DIR}/bc_actor.pt"
    --epochs "${RLTOKEN_ACTOR_EPOCHS:-10}"
    --batch-size "${RLTOKEN_BATCH_SIZE:-8}"
    --num-workers "${RLTOKEN_NUM_WORKERS:-2}"
    --lr "${RLTOKEN_ACTOR_LR:-1.0e-4}"
    --encoder-lr "${RLTOKEN_ENCODER_LR:-3.0e-4}"
    --weight-decay "${RLTOKEN_WEIGHT_DECAY:-1.0e-5}"
    --bc-weight "${RLTOKEN_BC_WEIGHT:-0.25}"
    --baseline-loss-penalty "${RLTOKEN_BASELINE_LOSS_PENALTY:-1.0}"
    --wcm-value-weight "${RLTOKEN_WCM_VALUE_WEIGHT:-1.0}"
    --reconstruction-weight "${RLTOKEN_RECONSTRUCTION_WEIGHT:-0.01}"
    --seed "$((SEED + iteration))"
  )
  if [[ -n "${RLTOKEN_MAX_STEPS:-}" ]]; then ACTOR_ARGS+=(--max-steps "${RLTOKEN_MAX_STEPS}"); fi
  if [[ "${RLTOKEN_TRAIN_ENCODER_WITH_ACTOR:-0}" == "1" ]]; then
    ACTOR_ARGS+=(--train-encoder-with-actor)
  fi
  if [[ -n "${CURRENT_ACTOR}" ]]; then
    ACTOR_ARGS+=(--resume "${CURRENT_ACTOR}")
  else
    if [[ -n "${ENCODER_RESUME}" ]]; then ACTOR_ARGS+=(--encoder-resume "${ENCODER_RESUME}"); fi
    if [[ -n "${BC_RESUME}" ]]; then ACTOR_ARGS+=(--bc-resume "${BC_RESUME}"); fi
  fi
  WCM_SUCCESS_LABELS="${BUFFER_ROOT}/meta/success_labels.json" \
  WCM_ASSUME_SUCCESS=0 \
  WCM_FAILURE_PENALTY="${FAILURE_PENALTY}" \
  WCM_GAMMA="${GAMMA}" \
  ACTOR_TRAIN_GPUS="${ACTOR_TRAIN_GPUS}" \
  PYTHON_BIN="${WCM_PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/run_pi05_rltoken.sh" "${ACTOR_ARGS[@]}"
  CURRENT_ACTOR="${ACTOR_OUTPUT}"
  [[ -f "${CURRENT_ACTOR}" ]] || { echo "Actor checkpoint missing: ${CURRENT_ACTOR}" >&2; exit 1; }
  printf '%s\n' "${CURRENT_ACTOR}" > "${RUN_ROOT}/latest_actor.txt"
  printf '%s\n' "${PREVIOUS_WCM}" > "${RUN_ROOT}/latest_wcm.txt"

  echo "[RLToken RECAP ${iteration}/${ITERATIONS}] collecting ${ROLLOUT_EPISODES} actor episodes with ${ROLLOUT_WORKER_COUNT} workers"
  mkdir -p "${RAW_ROLLOUTS}/logs" "${RAW_ROLLOUTS}/episodes"
  declare -a rollout_pids=() rollout_worker_episodes=() rollout_worker_logs=()
  declare -a rollout_worker_statuses=()
  episodes_per_worker=$((ROLLOUT_EPISODES / ROLLOUT_WORKER_COUNT))
  extra_episodes=$((ROLLOUT_EPISODES % ROLLOUT_WORKER_COUNT))
  # RoboDojo's eval seed selects a concrete Eval_Layout directory. Keep it
  # fixed across actor iterations; worker layout_shard provides disjoint work.
  rollout_seed="${ROLLOUT_LAYOUT_SEED}"
  rollout_run_prefix="$(date +%Y-%m-%d_%H-%M-%S)-$$-iter$(printf '%02d' "${iteration}")"
  for ((worker = 0; worker < ROLLOUT_WORKER_COUNT; worker++)); do
    worker_episodes="${episodes_per_worker}"
    if (( worker < extra_episodes )); then worker_episodes=$((worker_episodes + 1)); fi
    worker_num_envs="${ROLLOUT_ENVS_PER_WORKER}"
    if (( worker_num_envs > worker_episodes )); then worker_num_envs="${worker_episodes}"; fi
    worker_run_id="${rollout_run_prefix}-worker$(printf '%02d' "${worker}")"
    worker_log="${RAW_ROLLOUTS}/logs/worker_$(printf '%02d' "${worker}").log"
    echo "[RLToken RECAP rollout] worker=${worker} episodes=${worker_episodes} envs=${worker_num_envs} shard=${worker}/${ROLLOUT_WORKER_COUNT} policy_gpu=${ROLLOUT_POLICY_GPU_IDS[worker]} env_gpu=${ROLLOUT_ENV_GPU_IDS[worker]} log=${worker_log}"
    (
      POSTTRAIN_MODE="${ACTOR_OBJECTIVE}" \
      POSTTRAIN_CHECKPOINT="${CURRENT_ACTOR}" \
      POSTTRAIN_DETERMINISTIC="${RLTOKEN_ROLLOUT_DETERMINISTIC:-0}" \
      ROBODOJO_DISABLE_PROGRESS=1 \
      ROBODOJO_RUN_ID="${worker_run_id}" \
        bash "${ROOT_DIR}/scripts/robodojo.sh" eval \
          --policy-dir "${POLICY_DIR}" \
          --task "${TASK_NAME}" \
          --ckpt "${BASE_POLICY_CHECKPOINT}" \
          --env-cfg "${ENV_CFG_TYPE}" \
          --action-type "${ACTION_TYPE}" \
          --seed "${rollout_seed}" \
          --policy-gpu "${ROLLOUT_POLICY_GPU_IDS[worker]}" \
          --env-gpu "${ROLLOUT_ENV_GPU_IDS[worker]}" \
          --policy-env "${POLICY_ENV}" \
          --eval-env "${EVAL_ENV}" \
          --eval-num "${worker_episodes}" \
          --num-envs "${worker_num_envs}" \
          --layout-shard "${worker}/${ROLLOUT_WORKER_COUNT}" \
          --rollout-dir "${RAW_ROLLOUTS}" \
          --no-video
    ) >"${worker_log}" 2>&1 &
    rollout_pids+=("$!")
    ACTIVE_ROLLOUT_PIDS+=("$!")
    rollout_worker_episodes+=("${worker_episodes}")
    rollout_worker_logs+=("${worker_log}")
  done

  progress_args=(
    --root "${RAW_ROLLOUTS}"
    --total "${ROLLOUT_EPISODES}"
    --desc "RLToken rollout ${iteration}/${ITERATIONS}"
  )
  for worker_pid in "${rollout_pids[@]}"; do
    progress_args+=(--worker-pid "${worker_pid}")
  done
  "${WCM_PYTHON_BIN}" "${SCRIPT_DIR}/monitor_rollout_progress.py" "${progress_args[@]}" &
  rollout_progress_pid=$!
  ACTIVE_ROLLOUT_PIDS+=("${rollout_progress_pid}")

  rollout_failed=0
  for ((worker = 0; worker < ROLLOUT_WORKER_COUNT; worker++)); do
    if wait "${rollout_pids[worker]}"; then
      rollout_worker_statuses+=("0")
    else
      worker_status=$?
      rollout_worker_statuses+=("${worker_status}")
      rollout_failed=1
    fi
  done
  if ! wait "${rollout_progress_pid}"; then
    rollout_failed=1
  fi
  ACTIVE_ROLLOUT_PIDS=()
  for ((worker = 0; worker < ROLLOUT_WORKER_COUNT; worker++)); do
    if [[ "${rollout_worker_statuses[worker]}" == "0" ]]; then
      echo "[RLToken RECAP rollout] worker=${worker} completed (${rollout_worker_episodes[worker]} episodes)"
    else
      echo "[RLToken RECAP rollout] worker=${worker} failed with status ${rollout_worker_statuses[worker]}; tail follows" >&2
      tail -n 80 "${rollout_worker_logs[worker]}" >&2 || true
    fi
  done
  if (( rollout_failed != 0 )); then
    echo "One or more rollout workers failed; logs are under ${RAW_ROLLOUTS}/logs" >&2
    exit 1
  fi

  recorded_episodes=$(find "${RAW_ROLLOUTS}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
  recorded_episodes="${recorded_episodes//[[:space:]]/}"
  if [[ "${recorded_episodes}" != "${ROLLOUT_EPISODES}" ]]; then
    echo "Expected ${ROLLOUT_EPISODES} rollout episodes, recorded ${recorded_episodes}; check worker logs and available layout shards" >&2
    exit 1
  fi
  echo "[RLToken RECAP rollout] recorded ${recorded_episodes}/${ROLLOUT_EPISODES} episodes"
  ROLLOUT_ROOTS+=("${RAW_ROLLOUTS}")
done

echo "RL Token RECAP complete"
echo "actor=${CURRENT_ACTOR}"
echo "wcm=${PREVIOUS_WCM}"
echo "base_policy=${BASE_POLICY_CHECKPOINT}"
