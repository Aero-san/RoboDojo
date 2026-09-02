#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [openpi_train_arg ...]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
extra_train_args=("$@")

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
finetune_mode="${OPENPI_FINETUNE_MODE:-full}"
parameter_dtype="${OPENPI_PARAMETER_DTYPE:-bfloat16}"
sharding_strategy="${OPENPI_SHARDING_STRATEGY:-full_shard}"
cpu_offload="${OPENPI_CPU_OFFLOAD:-0}"
lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"

case "${finetune_mode}" in
  full|action_expert|action_expert_lora|paligemma_lora|all_lora) ;;
  *)
    echo "OPENPI_FINETUNE_MODE must be one of: full, action_expert, action_expert_lora, paligemma_lora, all_lora" >&2
    exit 2
    ;;
esac

case "${parameter_dtype}" in
  float32|bfloat16) ;;
  *)
    echo "OPENPI_PARAMETER_DTYPE must be float32 or bfloat16, got: ${parameter_dtype}" >&2
    exit 2
    ;;
esac

case "${sharding_strategy}" in
  full_shard|shard_grad_op|no_shard) ;;
  *)
    echo "OPENPI_SHARDING_STRATEGY must be one of: full_shard, shard_grad_op, no_shard" >&2
    exit 2
    ;;
esac

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_non_negative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

require_boolean() {
  local name=$1
  local value=$2
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "${name} must be 0 or 1, got: ${value}" >&2
    exit 2
  fi
}

resolve_checkpoint_step() {
  local input=$1
  local resolved
  local candidate
  local candidate_step
  local latest=""
  local latest_step=-1

  [[ -e "${input}" ]] || {
    echo "OPENPI_INIT_CHECKPOINT does not exist: ${input}" >&2
    exit 2
  }
  resolved=$(realpath "${input}")
  if [[ "$(basename "${resolved}")" == "params" ]]; then
    resolved=$(dirname "${resolved}")
  fi
  if [[ -d "${resolved}/params" ]]; then
    printf '%s\n' "${resolved}"
    return
  fi
  shopt -s nullglob
  for candidate in "${resolved}"/*; do
    [[ -d "${candidate}/params" ]] || continue
    candidate_step=$(basename "${candidate}")
    [[ "${candidate_step}" =~ ^[0-9]+$ ]] || continue
    if (( candidate_step > latest_step )); then
      latest_step=${candidate_step}
      latest=${candidate}
    fi
  done
  shopt -u nullglob
  [[ -n "${latest}" ]] || {
    echo "No OpenPI step directory containing params found under: ${resolved}" >&2
    exit 2
  }
  printf '%s\n' "${latest}"
}

read_checkpoint_finetune_mode() {
  local step_dir=$1
  local metadata_path="${step_dir}/robodojo_pi05_model.json"
  if [[ ! -f "${metadata_path}" ]]; then
    metadata_path="$(dirname "${step_dir}")/robodojo_pi05_model.json"
  fi
  [[ -f "${metadata_path}" ]] || return 0
  sed -nE 's/^[[:space:]]*"finetune_mode":[[:space:]]*"([^"]+)".*/\1/p' "${metadata_path}"
}

validate_checkpoint_finetune_mode() {
  local step_dir=$1
  local checkpoint_mode
  checkpoint_mode=$(read_checkpoint_finetune_mode "${step_dir}")
  if [[ -n "${checkpoint_mode}" && "${checkpoint_mode}" != "${finetune_mode}" ]]; then
    echo "Checkpoint ${step_dir} uses finetune_mode=${checkpoint_mode}, but OPENPI_FINETUNE_MODE=${finetune_mode}" >&2
    exit 2
  fi
}

is_positive_integer "${gpu_count}" || { echo "gpu_id must be a comma-separated GPU list" >&2; exit 2; }
is_positive_integer "${fsdp_devices}" || { echo "OPENPI_FSDP_DEVICES must be a positive integer" >&2; exit 2; }
require_boolean OPENPI_CPU_OFFLOAD "${cpu_offload}"
if (( fsdp_devices > gpu_count || gpu_count % fsdp_devices != 0 )); then
  echo "OPENPI_FSDP_DEVICES=${fsdp_devices} must divide the ${gpu_count} visible GPUs" >&2
  exit 2
fi

resume="${OPENPI_RESUME:-0}"
overwrite="${OPENPI_OVERWRITE:-0}"
wandb_enabled="${OPENPI_WANDB_ENABLED:-1}"
dry_run="${OPENPI_DRY_RUN:-0}"
require_boolean OPENPI_RESUME "${resume}"
require_boolean OPENPI_OVERWRITE "${overwrite}"
require_boolean OPENPI_WANDB_ENABLED "${wandb_enabled}"
require_boolean OPENPI_DRY_RUN "${dry_run}"
if [[ "${resume}" == "1" && "${overwrite}" == "1" ]]; then
  echo "OPENPI_RESUME and OPENPI_OVERWRITE cannot both be 1" >&2
  exit 2
fi
if [[ "${resume}" == "1" && -n "${OPENPI_INIT_CHECKPOINT:-}" ]]; then
  echo "OPENPI_RESUME and OPENPI_INIT_CHECKPOINT are mutually exclusive" >&2
  exit 2
fi

mkdir -p "${POLICY_DIR}/checkpoints"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# LeRobot loads parquet via HuggingFace datasets, which builds pyarrow mmap cache
# under HF_DATASETS_CACHE. Keep dataset on shared storage, but use per-host local
# cache to avoid NFS lock contention when multiple nodes train concurrently.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] finetune_mode=${finetune_mode}"
echo "[Pi_05] parameter_dtype=${parameter_dtype}"
echo "[Pi_05] sharding_strategy=${sharding_strategy}"
echo "[Pi_05] cpu_offload=${cpu_offload}"
echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
echo "[Pi_05] fsdp_devices=${fsdp_devices}"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir}"

train_args=(
  "${train_config_name}"
  "--exp-name=${ckpt_setting}"
  "--data.repo-id=${lerobot_repo_id}"
  "--finetune-mode=${finetune_mode}"
  "--parameter-dtype=${parameter_dtype}"
  "--sharding-strategy=${sharding_strategy}"
  "--fsdp-devices=${fsdp_devices}"
  "--checkpoint-dir-override=${ckpt_dir}"
  "--seed=${seed}"
)
if [[ "${cpu_offload}" == "1" ]]; then
  train_args+=(--cpu-offload)
else
  train_args+=(--no-cpu-offload)
fi

for option in batch-size num-workers num-train-steps log-interval save-interval; do
  variable="OPENPI_${option^^}"
  variable="${variable//-/_}"
  if [[ -n "${!variable:-}" ]]; then
    is_positive_integer "${!variable}" || { echo "${variable} must be a positive integer" >&2; exit 2; }
    train_args+=("--${option}=${!variable}")
  fi
done
if [[ -n "${OPENPI_BATCH_SIZE:-}" ]] && (( OPENPI_BATCH_SIZE % gpu_count != 0 )); then
  echo "OPENPI_BATCH_SIZE=${OPENPI_BATCH_SIZE} must be divisible by ${gpu_count} visible GPUs" >&2
  exit 2
fi
if [[ -n "${OPENPI_KEEP_PERIOD:-}" ]]; then train_args+=("--keep-period=${OPENPI_KEEP_PERIOD}"); fi
if [[ -n "${OPENPI_EMA_DECAY:-}" ]]; then train_args+=("--ema-decay=${OPENPI_EMA_DECAY}"); fi
if [[ -n "${OPENPI_LEARNING_RATE:-}" ]]; then train_args+=("--lr-schedule.peak-lr=${OPENPI_LEARNING_RATE}"); fi
if [[ -n "${OPENPI_WARMUP_STEPS:-}" ]]; then
  is_non_negative_integer "${OPENPI_WARMUP_STEPS}" || { echo "OPENPI_WARMUP_STEPS must be a non-negative integer" >&2; exit 2; }
  train_args+=("--lr-schedule.warmup-steps=${OPENPI_WARMUP_STEPS}")
fi
if [[ -n "${OPENPI_DECAY_STEPS:-}" ]]; then
  is_positive_integer "${OPENPI_DECAY_STEPS}" || { echo "OPENPI_DECAY_STEPS must be a positive integer" >&2; exit 2; }
  train_args+=("--lr-schedule.decay-steps=${OPENPI_DECAY_STEPS}")
fi
if [[ -n "${OPENPI_DECAY_LR:-}" ]]; then train_args+=("--lr-schedule.decay-lr=${OPENPI_DECAY_LR}"); fi
if [[ -n "${OPENPI_WEIGHT_DECAY:-}" ]]; then train_args+=("--optimizer.weight-decay=${OPENPI_WEIGHT_DECAY}"); fi
if [[ -n "${OPENPI_CLIP_GRADIENT_NORM:-}" ]]; then
  train_args+=("--optimizer.clip-gradient-norm=${OPENPI_CLIP_GRADIENT_NORM}")
fi
if [[ "${wandb_enabled}" == "1" ]]; then
  train_args+=(--wandb-enabled)
else
  train_args+=(--no-wandb-enabled)
fi
if [[ "${resume}" == "1" ]]; then train_args+=(--resume); fi
if [[ "${overwrite}" == "1" ]]; then train_args+=(--overwrite); fi

if [[ -n "${OPENPI_INIT_CHECKPOINT:-}" ]]; then
  init_step=$(resolve_checkpoint_step "${OPENPI_INIT_CHECKPOINT}")
  validate_checkpoint_finetune_mode "${init_step}"
  output_path=$(realpath -m "${ckpt_dir}")
  if [[ "${init_step}" == "${output_path}" || "${init_step}" == "${output_path}/"* ]]; then
    echo "OPENPI_INIT_CHECKPOINT is inside the output run. Use OPENPI_RESUME=1 instead." >&2
    exit 2
  fi
  train_args+=("--weight-loader.params-path=${init_step}/params")
  if [[ -z "${OPENPI_ASSETS_DIR:-}" && -d "${init_step}/assets" ]]; then
    OPENPI_ASSETS_DIR="${init_step}/assets"
  fi
  echo "[Pi_05] init_checkpoint=${init_step} (model weights only)"
fi
if [[ "${resume}" == "1" ]]; then
  resume_step=$(resolve_checkpoint_step "${ckpt_dir}")
  validate_checkpoint_finetune_mode "${resume_step}"
  [[ -d "${resume_step}/train_state" ]] || {
    echo "Cannot resume: ${resume_step} has params but no train_state. Use it as OPENPI_INIT_CHECKPOINT for a new run." >&2
    exit 2
  }
  if [[ -z "${OPENPI_ASSETS_DIR:-}" && -d "${resume_step}/assets" ]]; then
    OPENPI_ASSETS_DIR="${resume_step}/assets"
  fi
  echo "[Pi_05] resume_checkpoint=${resume_step} (full training state)"
fi
if [[ -n "${OPENPI_ASSETS_DIR:-}" ]]; then
  [[ -d "${OPENPI_ASSETS_DIR}" ]] || { echo "OPENPI_ASSETS_DIR is not a directory: ${OPENPI_ASSETS_DIR}" >&2; exit 2; }
  train_args+=("--data.assets.assets-dir=$(realpath "${OPENPI_ASSETS_DIR}")")
fi
if [[ -n "${OPENPI_ASSET_ID:-}" ]]; then train_args+=("--data.assets.asset-id=${OPENPI_ASSET_ID}"); fi
train_args+=("${extra_train_args[@]}")

cd "${POLICY_DIR}/openpi/"
if [[ "${dry_run}" == "1" ]]; then
  printf '[Pi_05] command:'
  printf ' %q' uv run scripts/train.py "${train_args[@]}"
  printf '\n'
  exit 0
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run scripts/train.py "${train_args[@]}"
