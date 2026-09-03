#!/usr/bin/env bash
# Productized RoboDojo benchmark entry point.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/robodojo.sh <command> [options]

Commands:
  doctor      Check RoboDojo assets/configs/env before launching evaluation
  eval        Run one RoboDojo task through an XPolicyLab policy eval.sh (server + client on localhost)
  server      Start only the policy server (for split / multi-machine eval)
  client      Run only the sim client against an already-running policy server
  smoke       Run selected/all tasks; default sequential, or balanced multi-GPU with --gpu-ids
  benchmark   Run selected/all tasks; default sequential, or balanced multi-GPU with --gpu-ids
  dimensions  List capability dimensions and their runnable tasks
  summarize   Aggregate eval_result into a markdown summary table

Maintainer:
  tasks       Inspect canonical runnable tasks (not needed for normal eval)

Run `bash scripts/robodojo.sh <command> --help` for command options.
EOF
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[robodojo] Missing value for $1" >&2
    exit 2
  fi
}

policy_eval_uses_expert_num() {
  local eval_script="$1"
  grep -Eq 'expert_data_num|expert_num' "${eval_script}"
}

resolve_policy_name() {
  local policy_dir="$1"
  basename "${policy_dir}"
}

validate_policy_dir() {
  local policy_dir="$1"
  local label="${2:-policy}"
  if [[ ! -f "${policy_dir}/setup_eval_policy_server.sh" ]]; then
    echo "[robodojo ${label}] setup_eval_policy_server.sh not found: ${policy_dir}/setup_eval_policy_server.sh" >&2
    exit 1
  fi
}

# probe_tcp HOST PORT TIMEOUT_SECONDS -> exit 0 if a TCP connect succeeds.
# Uses bash /dev/tcp so it needs no extra tools; `timeout` bounds the wait.
probe_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-5}"
  timeout "${timeout_s}" bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null
}

run_doctor() {
  bash "${ROOT_DIR}/scripts/internal/verify_install.sh" "$@"
}

run_tasks() {
  python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" "$@"
}

run_dimensions() {
  python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --list-dimensions "$@"
}

plot_action_noise() {
  local input_dir="$1"
  local method="$2"
  local highlight_k="$3"
  local python_env="${4:-}"
  local -a plot_cmd=(
    python3 "${ROOT_DIR}/scripts/internal/plot_action_noise.py"
    --input-dir "${input_dir}"
    --method "${method}"
    --highlight-k "${highlight_k}"
  )
  if [[ -n "${python_env}" ]] && command -v conda >/dev/null 2>&1; then
    conda run -n "${python_env}" "${plot_cmd[@]}"
  else
    "${plot_cmd[@]}"
  fi
}

run_eval() {
  local dataset="RoboDojo"
  local task=""
  local ckpt=""
  local env_cfg="arx_x5"
  local expert_num="100"
  local action_type="ee"
  local seed="0"
  local policy_gpu="0"
  local env_gpu="0"
  local policy_gpu_explicit="false"
  local env_gpu_explicit="false"
  local policy_env=""
  local eval_env="RoboDojo"
  local policy_dir=""
  local eval_num="${EVAL_NUM:-}"
  local num_envs="${ROBODOJO_NUM_ENVS:-}"
  local max_steps="${ROBODOJO_MAX_STEPS:-}"
  local fixed_horizon="${ROBODOJO_FIXED_HORIZON:-0}"
  local layout_shard="${ROBODOJO_LAYOUT_SHARD:-}"
  local layout_offset="${ROBODOJO_LAYOUT_OFFSET:-0}"
  local dry_run="false"
  local save_video="true"
  local rollout_dir="${ROBODOJO_ROLLOUT_DIR:-}"
  local action_noise_viz="false"
  local action_noise_dir="${ROBODOJO_ACTION_NOISE_DIR:-}"
  local noise_viz_method="${ROBODOJO_NOISE_VIZ_METHOD:-umap}"
  local noise_viz_k="${ROBODOJO_NOISE_VIZ_K:-5}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --expert-num) need_value "$@"; expert_num="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --policy-gpu) need_value "$@"; policy_gpu="$2"; policy_gpu_explicit="true"; shift 2 ;;
      --env-gpu) need_value "$@"; env_gpu="$2"; env_gpu_explicit="true"; shift 2 ;;
      --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
      --eval-env) need_value "$@"; eval_env="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
      --num-envs) need_value "$@"; num_envs="$2"; shift 2 ;;
      --max-steps) need_value "$@"; max_steps="$2"; shift 2 ;;
      --fixed-horizon) fixed_horizon="1"; shift ;;
      --layout-shard) need_value "$@"; layout_shard="$2"; shift 2 ;;
      --layout-offset) need_value "$@"; layout_offset="$2"; shift 2 ;;
      --save-video) save_video="true"; shift ;;
      --no-video) save_video="false"; shift ;;
      --rollout-dir) need_value "$@"; rollout_dir="$(abs_path "$2")"; shift 2 ;;
      --action-noise-viz) action_noise_viz="true"; shift ;;
      --noise-viz-dir) need_value "$@"; action_noise_dir="$(abs_path "$2")"; action_noise_viz="true"; shift 2 ;;
      --noise-viz-method) need_value "$@"; noise_viz_method="$2"; action_noise_viz="true"; shift 2 ;;
      --noise-viz-k) need_value "$@"; noise_viz_k="$2"; action_noise_viz="true"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/robodojo.sh eval --policy-dir PATH --task TASK --ckpt CKPT --policy-env ENV [options]

Required:
  --policy-dir PATH     XPolicyLab policy directory containing eval.sh
  --task TASK           RoboDojo task name
  --ckpt CKPT           Policy checkpoint name
  --policy-env ENV      Policy conda env, uv, or env path

Common options:
  --eval-num NUM|native  Override EVAL_NUM for this eval; use `native` for per-task counts from _task.yml
  --num-envs NUM          Vectorized Isaac environments in this client process
  --max-steps NUM         Override the task's maximum deployed actions per episode
  --fixed-horizon         Continue simulation until max-steps after success or failure
  --layout-shard I/N      Use non-overlapping layout shard I out of N (zero-based)
  --layout-offset N       Skip the first N layouts inside the selected shard
  --save-video           Save one MP4 per camera and rollout (default)
  --no-video             Disable MP4 encoding; camera observations remain enabled for vision policies
  --rollout-dir PATH      Record state/action/camera trajectories with simulator success labels
  --action-noise-viz      Record policy initial noise and generate two low-dimensional plots
  --noise-viz-dir PATH    Raw/plot output root (default: eval_result/action_noise/<run-id>)
  --noise-viz-method NAME Reduction method: umap (default) or tsne
  --noise-viz-k NUM       Evenly spaced highlighted points per selected rollout (default: 5)
  --env-cfg NAME        env_cfg stem (default: arx_x5)
  --expert-num NUM      Expert-data count for policy eval.sh files that accept it (default: 100)
  --action-type NAME    Policy action type (default: ee)
  --seed NUM            Eval seed / layout seed (default: 0)
  --policy-gpu ID       Policy server GPU (default: 0; auto-split with multiple visible GPUs)
  --env-gpu ID          Isaac Sim GPU (default: 0; auto-split to the second visible GPU)
  --eval-env ENV        Simulator conda env (default: RoboDojo)
  --dry-run             Print command without running it

Split / multi-machine: use `robodojo.sh server` + `robodojo.sh client` (see docs/SPLIT_EVAL.md).
EOF
        return 0
        ;;
      *)
        echo "[robodojo eval] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -z "${policy_dir}" || -z "${task}" || -z "${ckpt}" || -z "${policy_env}" ]]; then
    echo "[robodojo eval] --policy-dir, --task, --ckpt, and --policy-env are required" >&2
    exit 2
  fi
  if [[ -n "${num_envs}" && ! "${num_envs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[robodojo eval] --num-envs must be a positive integer" >&2
    exit 2
  fi
  if [[ -n "${max_steps}" && ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[robodojo eval] --max-steps must be a positive integer" >&2
    exit 2
  fi
  if [[ "${fixed_horizon}" != "0" && "${fixed_horizon}" != "1" ]]; then
    echo "[robodojo eval] ROBODOJO_FIXED_HORIZON must be 0 or 1" >&2
    exit 2
  fi
  if [[ "${fixed_horizon}" == "1" && -z "${max_steps}" ]]; then
    echo "[robodojo eval] --fixed-horizon requires --max-steps" >&2
    exit 2
  fi
  if [[ -n "${layout_shard}" ]]; then
    if [[ ! "${layout_shard}" =~ ^([0-9]+)/([1-9][0-9]*)$ ]]; then
      echo "[robodojo eval] --layout-shard must use zero-based I/N syntax" >&2
      exit 2
    fi
    if (( BASH_REMATCH[1] >= BASH_REMATCH[2] )); then
      echo "[robodojo eval] layout shard index must be smaller than shard count" >&2
      exit 2
    fi
  fi
  if [[ ! "${layout_offset}" =~ ^[0-9]+$ ]]; then
    echo "[robodojo eval] --layout-offset must be a non-negative integer" >&2
    exit 2
  fi
  if [[ "${noise_viz_method}" != "umap" && "${noise_viz_method}" != "tsne" ]]; then
    echo "[robodojo eval] --noise-viz-method must be umap or tsne" >&2
    exit 2
  fi
  if [[ ! "${noise_viz_k}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[robodojo eval] --noise-viz-k must be a positive integer" >&2
    exit 2
  fi
  local visible_gpu_list="${CUDA_VISIBLE_DEVICES:-}"
  local -a visible_gpu_ids=()
  if [[ -n "${visible_gpu_list}" ]]; then
    IFS=',' read -r -a visible_gpu_ids <<< "${visible_gpu_list}"
  fi
  if [[ ${#visible_gpu_ids[@]} -ge 2 && "${policy_gpu_explicit}" == "false" && "${env_gpu_explicit}" == "false" ]]; then
    policy_gpu="${visible_gpu_ids[0]}"
    env_gpu="${visible_gpu_ids[1]}"
    echo "[robodojo eval] multiple GPUs detected: auto-split policy_gpu=${policy_gpu}, env_gpu=${env_gpu}"
  elif [[ "${policy_gpu}" == "${env_gpu}" && ${#visible_gpu_ids[@]} -ge 2 ]]; then
    echo "[robodojo eval] policy GPU and env GPU are both ${policy_gpu}; use different IDs to avoid shared-GPU OOM" >&2
    exit 2
  fi
  if [[ ! -f "${policy_dir}/eval.sh" ]]; then
    echo "[robodojo eval] policy eval.sh not found: ${policy_dir}/eval.sh" >&2
    exit 1
  fi

  # Policy deploy.yml is the source of truth for the action representation.
  # Keep the generic CLI default for legacy policies, but make an accidental
  # mismatch visible before starting Isaac Sim (Pi_05 uses joint actions).
  local policy_action_type=""
  if [[ -f "${policy_dir}/deploy.yml" ]]; then
    policy_action_type="$(sed -n 's/^[[:space:]]*action_type:[[:space:]]*//p' "${policy_dir}/deploy.yml" | head -n 1 | tr -d '\"' | tr -d "'")"
  fi
  if [[ -n "${policy_action_type}" && "${policy_action_type}" != "${action_type}" ]]; then
    echo "[robodojo eval][WARN] --action-type=${action_type} differs from ${policy_dir}/deploy.yml action_type=${policy_action_type}; pass --action-type ${policy_action_type} unless this override is intentional." >&2
  fi

  local eval_args=()
  if policy_eval_uses_expert_num "${policy_dir}/eval.sh"; then
    eval_args=(
      "${dataset}"
      "${task}"
      "${ckpt}"
      "${env_cfg}"
      "${expert_num}"
      "${action_type}"
      "${seed}"
      "${policy_gpu}"
      "${env_gpu}"
      "${policy_env}"
      "${eval_env}"
    )
  else
    eval_args=(
      "${dataset}"
      "${task}"
      "${ckpt}"
      "${env_cfg}"
      "${action_type}"
      "${seed}"
      "${policy_gpu}"
      "${env_gpu}"
      "${policy_env}"
      "${eval_env}"
    )
  fi

  if [[ -n "${eval_num}" && "${eval_num}" != "native" ]]; then
    export EVAL_NUM="${eval_num}"
  fi
  if [[ -n "${num_envs}" ]]; then
    export ROBODOJO_NUM_ENVS="${num_envs}"
  else
    unset ROBODOJO_NUM_ENVS 2>/dev/null || true
  fi
  if [[ -n "${max_steps}" ]]; then
    export ROBODOJO_MAX_STEPS="${max_steps}"
  else
    unset ROBODOJO_MAX_STEPS 2>/dev/null || true
  fi
  export ROBODOJO_FIXED_HORIZON="${fixed_horizon}"
  if [[ -n "${layout_shard}" ]]; then
    export ROBODOJO_LAYOUT_SHARD="${layout_shard}"
  else
    unset ROBODOJO_LAYOUT_SHARD 2>/dev/null || true
  fi
  export ROBODOJO_LAYOUT_OFFSET="${layout_offset}"
  export ROBODOJO_SAVE_VIDEO="$([[ "${save_video}" == "true" ]] && echo 1 || echo 0)"
  if [[ -n "${rollout_dir}" ]]; then
    export ROBODOJO_ROLLOUT_DIR="${rollout_dir}"
  else
    unset ROBODOJO_ROLLOUT_DIR 2>/dev/null || true
  fi
  if [[ "${action_noise_viz}" == "true" ]]; then
    if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
      export ROBODOJO_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)-$$"
    fi
    if [[ -z "${action_noise_dir}" ]]; then
      action_noise_dir="${ROOT_DIR}/eval_result/action_noise/${ROBODOJO_RUN_ID}"
    fi
    export ROBODOJO_ACTION_NOISE_DIR="${action_noise_dir}"
    export ROBODOJO_NOISE_VIZ_METHOD="${noise_viz_method}"
    export ROBODOJO_NOISE_VIZ_K="${noise_viz_k}"
  fi

  echo "[robodojo eval] policy_dir=${policy_dir}"
  echo "[robodojo eval] task=${task} env_cfg=${env_cfg} eval_num=${EVAL_NUM:-default} num_envs=${num_envs:-config} max_steps=${max_steps:-task-default} fixed_horizon=${fixed_horizon} layout_shard=${layout_shard:-all} layout_offset=${layout_offset} save_video=${save_video} rollout_dir=${rollout_dir:-disabled} action_noise_dir=${action_noise_dir:-disabled}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[robodojo eval] dry-run: bash %q' "${ROOT_DIR}/scripts/internal/run_policy_eval.sh"
    printf ' %q' "${policy_dir}"
    printf ' %q' "${eval_args[@]}"
    printf '\n'
    return 0
  fi

  local start_sec end_sec elapsed_sec
  start_sec="$(date +%s)"
  (
    bash "${ROOT_DIR}/scripts/internal/run_policy_eval.sh" "${policy_dir}" "${eval_args[@]}"
  )
  end_sec="$(date +%s)"
  elapsed_sec=$((end_sec - start_sec))
  echo "[robodojo eval] wall_clock=${elapsed_sec}s"
  if [[ "${action_noise_viz}" == "true" && "${ROBODOJO_ACTION_NOISE_DEFER_PLOT:-0}" != "1" ]]; then
    plot_action_noise "${action_noise_dir}" "${noise_viz_method}" "${noise_viz_k}" "${eval_env}"
  fi
}

run_server() {
  local dataset="RoboDojo"
  local task=""
  local ckpt=""
  local env_cfg="arx_x5"
  local action_type="ee"
  local seed="0"
  local policy_gpu="0"
  local gpu_ids=""
  local policy_gpu_ids=""
  local policy_env=""
  local policy_dir=""
  local policy_port=""
  local bind_host="0.0.0.0"
  local dry_run="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
      --gpu-ids) need_value "$@"; gpu_ids="$2"; shift 2 ;;
      --policy-gpu-ids) need_value "$@"; policy_gpu_ids="$2"; shift 2 ;;
      --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --policy-port) need_value "$@"; policy_port="$2"; shift 2 ;;
      --bind-host) need_value "$@"; bind_host="$2"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/robodojo.sh server --policy-dir PATH --task TASK --ckpt CKPT --policy-env ENV [options]

Starts only the XPolicyLab policy WebSocket server (foreground). Use with
`robodojo.sh client` on another host or container. See docs/SPLIT_EVAL.md.

Required:
  --policy-dir PATH     XPolicyLab policy directory (XPolicyLab/policy/<NAME>)
  --task TASK           RoboDojo task name passed to the policy server
  --ckpt CKPT           Policy checkpoint name
  --policy-env ENV      Policy conda env, uv, or env path

Common options:
  --policy-port PORT    TCP port (default: auto-selected free port)
  --bind-host HOST      Bind address (default: 0.0.0.0 for remote clients; use localhost for local-only)
  --env-cfg NAME        env_cfg stem (default: arx_x5)
  --action-type NAME    Policy action type (default: ee)
  --seed NUM            Eval seed (default: 0)
  --policy-gpu ID       Policy server GPU (default: 0)
  --gpu-ids ID          Optional alias for --policy-gpu; multiple ids are rejected
  --policy-gpu-ids ID   Optional alias for --policy-gpu; multiple ids are rejected
  --dry-run             Print command without running it
EOF
        return 0
        ;;
      *)
        echo "[robodojo server] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -n "${gpu_ids}" ]]; then
    if [[ "${gpu_ids}" == *","* ]]; then
      echo "[robodojo server] multiple --gpu-ids are not supported" >&2
      exit 2
    fi
    policy_gpu="${gpu_ids}"
  fi
  if [[ -n "${policy_gpu_ids}" ]]; then
    if [[ "${policy_gpu_ids}" == *","* ]]; then
      echo "[robodojo server] multiple --policy-gpu-ids are not supported" >&2
      exit 2
    fi
    policy_gpu="${policy_gpu_ids}"
  fi

  if [[ -z "${policy_dir}" || -z "${task}" || -z "${ckpt}" || -z "${policy_env}" ]]; then
    echo "[robodojo server] --policy-dir, --task, --ckpt, and --policy-env are required" >&2
    exit 2
  fi
  validate_policy_dir "${policy_dir}" "server"
  if [[ "${policy_port}" == *","* || "${bind_host}" == *","* ]]; then
    echo "[robodojo server] comma-separated --policy-port/--bind-host are not supported" >&2
    exit 2
  fi

  local server_args=(
    "${policy_dir}"
    "${dataset}"
    "${task}"
    "${ckpt}"
    "${env_cfg}"
    "${action_type}"
    "${seed}"
    "${policy_gpu}"
    "${policy_env}"
    "${policy_port}"
    "${bind_host}"
  )

  echo "[robodojo server] policy_dir=${policy_dir} task=${task} bind_host=${bind_host} port=${policy_port:-auto}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[robodojo server] dry-run: bash %q' "${ROOT_DIR}/scripts/internal/run_policy_server.sh"
    printf ' %q' "${server_args[@]}"
    printf '\n'
    return 0
  fi

  bash "${ROOT_DIR}/scripts/internal/run_policy_server.sh" "${server_args[@]}"
}

run_client() {
  local dataset="RoboDojo"
  local task=""
  local env_cfg="arx_x5"
  local policy_name=""
  local policy_dir=""
  local policy_host="127.0.0.1"
  local policy_port=""
  local seed="0"
  local env_gpu="0"
  local gpu_ids=""
  local env_gpu_ids=""
  local ckpt="external"
  local action_type="ee"
  local eval_num="${EVAL_NUM:-}"
  local max_steps="${ROBODOJO_MAX_STEPS:-}"
  local fixed_horizon="${ROBODOJO_FIXED_HORIZON:-0}"
  local connect_timeout="5"
  local only_tasks=""
  local tasks_file=""
  local dimensions=""
  local limit=""
  local dry_run="false"
  local action_noise_viz="false"
  local action_noise_dir="${ROBODOJO_ACTION_NOISE_DIR:-}"
  local noise_viz_method="${ROBODOJO_NOISE_VIZ_METHOD:-umap}"
  local noise_viz_k="${ROBODOJO_NOISE_VIZ_K:-5}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
      --task) need_value "$@"; task="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --policy-name) need_value "$@"; policy_name="$2"; shift 2 ;;
      --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
      --policy-host) need_value "$@"; policy_host="$2"; shift 2 ;;
      --policy-port) need_value "$@"; policy_port="$2"; shift 2 ;;
      --seed) need_value "$@"; seed="$2"; shift 2 ;;
      --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
      --gpu-ids) need_value "$@"; gpu_ids="$2"; shift 2 ;;
      --env-gpu-ids) need_value "$@"; env_gpu_ids="$2"; shift 2 ;;
      --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
      --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
      --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
      --max-steps) need_value "$@"; max_steps="$2"; shift 2 ;;
      --fixed-horizon) fixed_horizon="1"; shift ;;
      --connect-timeout) need_value "$@"; connect_timeout="$2"; shift 2 ;;
      --only) need_value "$@"; only_tasks="$2"; shift 2 ;;
      --tasks-file) need_value "$@"; tasks_file="$2"; shift 2 ;;
      --dimension)
        need_value "$@"
        dimensions="${dimensions:+${dimensions},}$2"
        shift 2
        ;;
      --all) shift ;;
      --limit) need_value "$@"; limit="$2"; shift 2 ;;
      --action-noise-viz) action_noise_viz="true"; shift ;;
      --noise-viz-dir) need_value "$@"; action_noise_dir="$(abs_path "$2")"; action_noise_viz="true"; shift 2 ;;
      --noise-viz-method) need_value "$@"; noise_viz_method="$2"; action_noise_viz="true"; shift 2 ;;
      --noise-viz-k) need_value "$@"; noise_viz_k="$2"; action_noise_viz="true"; shift 2 ;;
      --dry-run) dry_run="true"; shift ;;
      -h|--help)
        cat <<'EOF'
Usage: bash scripts/robodojo.sh client --task TASK (--policy-name NAME | --policy-dir PATH) --policy-host HOST --policy-port PORT [options]

Runs the RoboDojo simulator client against an already-running external policy
server. Pair with `robodojo.sh server` on the policy machine. See docs/SPLIT_EVAL.md.

Batch mode:
  Omit `--task` and pass task filters such as `--only`, `--tasks-file`, or
  `--dimension`, together with `--gpu-ids` or `--env-gpu-ids`. Tasks will be
  grouped by embedded runtime weights and launched across client workers.
  If `--policy-host` contains multiple hosts, its count and `--policy-port`
  count must both equal the group count. If `--policy-host` contains one host,
  `--policy-port` count must equal the group count and the host is reused.

Required:
  --task TASK            RoboDojo task name
  --policy-name NAME     XPolicyLab deploy module (XPolicyLab/policy/<NAME>/deploy.py)
                         Omit when --policy-dir is set (name inferred from directory)
  --policy-dir PATH      Same as eval's --policy-dir; sets --policy-name from basename
  --policy-host HOST     Policy server IP / hostname reachable from this client
  --policy-port PORT     Policy server TCP port

Common options:
  --eval-num NUM|native  Override EVAL_NUM for this run; `native` uses per-task counts
  --max-steps NUM        Override the task's maximum deployed actions per episode
  --fixed-horizon        Continue simulation until max-steps after success or failure
  --env-cfg NAME         env_cfg stem (default: arx_x5)
  --seed NUM             Eval seed / layout seed (default: 0)
  --env-gpu ID           Isaac Sim GPU (default: 0)
  --gpu-ids IDS          Batch mode only: comma-separated client GPU ids
  --env-gpu-ids IDS      Batch mode only: comma-separated client GPU ids
  --ckpt NAME            Checkpoint label recorded in result paths (default: external)
  --action-type NAME     Action type label recorded in result paths (default: ee)
  --connect-timeout SEC  Pre-flight policy-server reachability probe timeout (default: 5)
  --only a,b,c           Batch mode task subset
  --tasks-file PATH      Batch mode task subset file
  --dimension NAMES      Batch mode capability dimensions
  --limit NUM            Batch mode task limit after filtering
  --action-noise-viz     Record policy initial noise and generate low-dimensional plots
  --noise-viz-dir PATH   Shared raw/plot output root
  --noise-viz-method NAME  Reduction method: umap (default) or tsne
  --noise-viz-k NUM      Highlighted points per selected rollout (default: 5)
  --dry-run              Print the resolved eval_policy.sh command without running it
EOF
        return 0
        ;;
      *)
        echo "[robodojo client] Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done

  if [[ -n "${policy_dir}" ]]; then
    validate_policy_dir "${policy_dir}" "client"
    if [[ -z "${policy_name}" ]]; then
      policy_name="$(resolve_policy_name "${policy_dir}")"
    fi
  fi

  if [[ -n "${max_steps}" && ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[robodojo client] --max-steps must be a positive integer" >&2
    exit 2
  fi
  if [[ "${fixed_horizon}" != "0" && "${fixed_horizon}" != "1" ]]; then
    echo "[robodojo client] ROBODOJO_FIXED_HORIZON must be 0 or 1" >&2
    exit 2
  fi
  if [[ "${fixed_horizon}" == "1" && -z "${max_steps}" ]]; then
    echo "[robodojo client] --fixed-horizon requires --max-steps" >&2
    exit 2
  fi
  if [[ "${noise_viz_method}" != "umap" && "${noise_viz_method}" != "tsne" ]]; then
    echo "[robodojo client] --noise-viz-method must be umap or tsne" >&2
    exit 2
  fi
  if [[ ! "${noise_viz_k}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[robodojo client] --noise-viz-k must be a positive integer" >&2
    exit 2
  fi

  if [[ -z "${policy_name}" ]]; then
    echo "[robodojo client] --policy-name or --policy-dir is required" >&2
    exit 2
  fi

  local deploy_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.py"
  if [[ ! -f "${deploy_file}" ]]; then
    echo "[robodojo client] policy deploy adapter not found: ${deploy_file}" >&2
    echo "[robodojo client] --policy-name must match a directory under XPolicyLab/policy/" >&2
    exit 1
  fi

  local batch_mode="false"
  if [[ -z "${task}" ]]; then
    batch_mode="true"
  fi

  if [[ "${batch_mode}" == "true" ]]; then
    if [[ -z "${policy_host}" || -z "${policy_port}" ]]; then
      echo "[robodojo client] batch mode requires --policy-host and --policy-port" >&2
      exit 2
    fi
    local batch_args=(
      --mode client
      --dataset "${dataset}"
      --env-cfg "${env_cfg}"
      --policy-name "${policy_name}"
      --policy-host "${policy_host}"
      --policy-port "${policy_port}"
      --seed "${seed}"
      --env-gpu "${env_gpu}"
      --ckpt "${ckpt}"
      --action-type "${action_type}"
      --connect-timeout "${connect_timeout}"
    )
    if [[ -n "${policy_dir}" ]]; then
      batch_args+=(--policy-dir "${policy_dir}")
    fi
    if [[ -n "${gpu_ids}" ]]; then
      batch_args+=(--gpu-ids "${gpu_ids}")
    fi
    if [[ -n "${env_gpu_ids}" ]]; then
      batch_args+=(--env-gpu-ids "${env_gpu_ids}")
    fi
    if [[ -n "${only_tasks}" ]]; then
      batch_args+=(--only "${only_tasks}")
    fi
    if [[ -n "${tasks_file}" ]]; then
      batch_args+=(--tasks-file "${tasks_file}")
    fi
    if [[ -n "${dimensions}" ]]; then
      batch_args+=(--dimension "${dimensions}")
    fi
    if [[ -n "${limit}" ]]; then
      batch_args+=(--limit "${limit}")
    fi
    if [[ -n "${eval_num}" ]]; then
      batch_args+=(--eval-num "${eval_num}")
    fi
    if [[ -n "${max_steps}" ]]; then
      batch_args+=(--max-steps "${max_steps}")
    fi
    if [[ "${fixed_horizon}" == "1" ]]; then
      batch_args+=(--fixed-horizon)
    fi
    if [[ "${action_noise_viz}" == "true" ]]; then
      batch_args+=(
        --action-noise-viz
        --noise-viz-method "${noise_viz_method}"
        --noise-viz-k "${noise_viz_k}"
      )
      if [[ -n "${action_noise_dir}" ]]; then
        batch_args+=(--noise-viz-dir "${action_noise_dir}")
      fi
    fi
    if [[ "${dry_run}" == "true" ]]; then
      batch_args+=(--dry-run)
    fi
    bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" "${batch_args[@]}"
    return $?
  fi

  if [[ -z "${policy_host}" || -z "${policy_port}" ]]; then
    echo "[robodojo client] single-task mode requires --task, --policy-host, and --policy-port" >&2
    exit 2
  fi
  if [[ "${policy_host}" == *","* || "${policy_port}" == *","* ]]; then
    echo "[robodojo client] comma-separated --policy-host/--policy-port are only supported in batch mode" >&2
    exit 2
  fi

  local additional_info="ckpt_name=${ckpt},action_type=${action_type}"

  if [[ -n "${eval_num}" && "${eval_num}" != "native" ]]; then
    export EVAL_NUM="${eval_num}"
  fi
  if [[ -n "${max_steps}" ]]; then
    export ROBODOJO_MAX_STEPS="${max_steps}"
  else
    unset ROBODOJO_MAX_STEPS 2>/dev/null || true
  fi
  export ROBODOJO_FIXED_HORIZON="${fixed_horizon}"
  if [[ "${action_noise_viz}" == "true" ]]; then
    if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
      export ROBODOJO_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)-$$"
    fi
    if [[ -z "${action_noise_dir}" ]]; then
      action_noise_dir="${ROOT_DIR}/eval_result/action_noise/${ROBODOJO_RUN_ID}"
    fi
    export ROBODOJO_ACTION_NOISE_DIR="${action_noise_dir}"
  fi

  local client_args=(
    --dataset_name "${dataset}"
    --task_name "${task}"
    --env_cfg_type "${env_cfg}"
    --policy_name "${policy_name}"
    --host "${policy_host}"
    --port "${policy_port}"
    --protocol ws
    --root_dir "${ROOT_DIR}"
    --device_id "${env_gpu}"
    --additional_info "${additional_info}"
    --seed "${seed}"
  )
  if [[ -n "${max_steps}" ]]; then
    client_args+=(--max_steps "${max_steps}")
  fi
  if [[ "${fixed_horizon}" == "1" ]]; then
    client_args+=(--fixed_horizon)
  fi

  echo "[robodojo client] task=${task} policy=${policy_name} server=${policy_host}:${policy_port} eval_num=${EVAL_NUM:-default} max_steps=${max_steps:-task-default} fixed_horizon=${fixed_horizon}"

  if [[ "${dry_run}" == "true" ]]; then
    printf '[robodojo client] dry-run: bash %q' "${ROOT_DIR}/scripts/eval_policy.sh"
    printf ' %q' "${client_args[@]}"
    printf '\n'
    return 0
  fi

  if probe_tcp "${policy_host}" "${policy_port}" "${connect_timeout}"; then
    echo "[robodojo client] policy server reachable at ${policy_host}:${policy_port}"
  else
    echo "[robodojo client] WARNING: could not reach ${policy_host}:${policy_port} within ${connect_timeout}s" >&2
    echo "[robodojo client] The client will keep retrying; verify that:" >&2
    echo "[robodojo client]   - the policy server is running and bound to 0.0.0.0 (not its own localhost)" >&2
    echo "[robodojo client]   - in Docker use --network host, or --policy-host host.docker.internal on a bridge network" >&2
    echo "[robodojo client]   - the port is correct and not blocked by a firewall" >&2
  fi

  bash "${ROOT_DIR}/scripts/eval_policy.sh" "${client_args[@]}"
  if [[ "${action_noise_viz}" == "true" && "${ROBODOJO_ACTION_NOISE_DEFER_PLOT:-0}" != "1" ]]; then
    plot_action_noise "${action_noise_dir}" "${noise_viz_method}" "${noise_viz_k}"
  fi
}

run_sweep() {
  local mode="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
      bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" "$@"
      return
    fi
  done
  if [[ "${mode}" == "smoke" ]]; then
    bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" --eval-num "${EVAL_NUM:-1}" "$@"
  else
    local has_eval_num="false"
    for arg in "$@"; do
      if [[ "${arg}" == "--eval-num" ]]; then
        has_eval_num="true"
        break
      fi
    done
    if [[ "${has_eval_num}" != "true" && -z "${EVAL_NUM:-}" ]]; then
      echo "[robodojo benchmark] pass --eval-num NUM|native or set EVAL_NUM" >&2
      exit 2
    fi
    bash "${ROOT_DIR}/scripts/internal/smoke_all_tasks.sh" "$@"
  fi
}

run_summarize() {
  python3 "${ROOT_DIR}/scripts/internal/summarize_result.py" "$@"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command="$1"
shift

case "${command}" in
  doctor) run_doctor "$@" ;;
  tasks) run_tasks "$@" ;;
  dimensions) run_dimensions "$@" ;;
  eval) run_eval "$@" ;;
  server) run_server "$@" ;;
  client) run_client "$@" ;;
  smoke) run_sweep smoke "$@" ;;
  benchmark) run_sweep benchmark "$@" ;;
  summarize) run_summarize "$@" ;;
  -h|--help) usage ;;
  *)
    echo "[robodojo] Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
