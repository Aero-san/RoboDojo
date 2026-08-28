#!/usr/bin/env bash
# Execute one resumable RECAP simulator or value-video job on a remote GPU host.
set -euo pipefail

required=(RECAP_REMOTE_ACTION RECAP_REMOTE_REPO_ROOT RECAP_REMOTE_WORK_ROOT RECAP_REMOTE_JOB_ROOT)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing remote worker variable: ${name}" >&2; exit 2; }
done
zstd_bin=""
if [[ "${RECAP_REMOTE_ACTION}" != "cancel" ]]; then
  zstd_command="${RECAP_REMOTE_ZSTD_BIN:-zstd}"
  zstd_bin="${zstd_command}"
  if [[ "${zstd_bin}" == */* ]]; then
    [[ -x "${zstd_bin}" ]] || {
      echo "Remote zstd is not executable: ${zstd_bin}" >&2
      echo "Set RECAP_REMOTE_ZSTD_BIN to the remote zstd executable." >&2
      exit 1
    }
  else
    zstd_bin=$(command -v "${zstd_bin}") || {
      echo "Remote zstd is not available in the non-login worker PATH: ${zstd_command}" >&2
      echo "Set RECAP_REMOTE_ZSTD_BIN to the absolute remote zstd path." >&2
      exit 1
    }
  fi
  conda_command="${RECAP_REMOTE_CONDA_BIN:-conda}"
  conda_bin="${conda_command}"
  if [[ "${conda_bin}" == */* ]]; then
    [[ -x "${conda_bin}" ]] || {
      echo "Remote conda is not executable: ${conda_bin}" >&2
      echo "Set RECAP_REMOTE_CONDA_BIN to the remote conda executable." >&2
      exit 1
    }
  else
    conda_bin=$(command -v "${conda_bin}") || {
      echo "Remote conda is not available in the non-login worker PATH: ${conda_command}" >&2
      echo "Set RECAP_REMOTE_CONDA_BIN to the absolute remote conda path." >&2
      exit 1
    }
  fi
  python_command="${RECAP_REMOTE_PYTHON_BIN:-python}"
  python_bin="${python_command}"
  if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || {
      echo "Remote bootstrap Python is not executable: ${python_bin}" >&2
      echo "Set RECAP_REMOTE_PYTHON_BIN to a remote Python containing PyYAML." >&2
      exit 1
    }
  else
    python_bin=$(command -v "${python_bin}") || {
      echo "Remote bootstrap Python is not in the non-login PATH: ${python_command}" >&2
      echo "Set RECAP_REMOTE_PYTHON_BIN to its absolute remote path." >&2
      exit 1
    }
  fi
  # Policy bootstrap scripts invoke both conda and python by name. Keep the
  # bootstrap Python first so bare python has PyYAML; keep conda available too.
  # GNU tar still finds the explicitly checked zstd through its bin directory.
  zstd_dir="${zstd_bin%/*}"
  conda_dir="${conda_bin%/*}"
  python_dir="${python_bin%/*}"
  export PATH="${python_dir:-/}:${conda_dir:-/}:${zstd_dir:-/}:${PATH:-/usr/bin:/bin}"
fi

control_dir="${RECAP_REMOTE_JOB_ROOT}/control"
reservation_pid_file="${control_dir}/reservation.pid"
reservation_ready_file="${control_dir}/reservation.ready"
reservation_log="${control_dir}/reservation.log"
worker_pid_file="${control_dir}/worker.pid"
reservation_gpus_file="${control_dir}/reservation.gpus"
gpu_lock_root="${RECAP_REMOTE_WORK_ROOT}/gpu_locks"
mkdir -p "${control_dir}"
mkdir -p "${gpu_lock_root}"

read_live_pid() {
  local pid_file="$1"
  local pid=""
  if [[ -f "${pid_file}" ]]; then
    read -r pid <"${pid_file}" || true
  fi
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
    printf '%s\n' "${pid}"
    return 0
  fi
  return 1
}

stop_pid_file() {
  local pid_file="$1"
  local process_group="$2"
  local pid=""
  if ! pid=$(read_live_pid "${pid_file}"); then
    rm -f "${pid_file}"
    return 0
  fi
  if [[ "${process_group}" == "1" ]]; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  for _ in {1..100}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    if [[ "${process_group}" == "1" ]]; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    else
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

stop_remote_reservation() {
  stop_pid_file "${reservation_pid_file}" 0
  rm -f "${reservation_ready_file}"
}

stop_reservation_controller() {
  local change_lock="${control_dir}/reservation.lock"
  local controller=""
  if [[ -f "${change_lock}/controller.pid" ]]; then
    read -r controller <"${change_lock}/controller.pid" || true
  fi
  if [[ -n "${controller}" && "${controller}" != "$$" ]]; then
    stop_pid_file "${change_lock}/controller.pid" 0
  fi
  rm -f "${change_lock}/controller.pid"
  rmdir "${change_lock}" 2>/dev/null || true
}

release_gpu_locks() {
  local gpu="" lock_dir="" owner=""
  if [[ ! -f "${reservation_gpus_file}" ]]; then
    return
  fi
  while IFS= read -r gpu; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || continue
    lock_dir="${gpu_lock_root}/gpu-${gpu}"
    if [[ -f "${lock_dir}/owner" ]]; then
      read -r owner <"${lock_dir}/owner" || true
    fi
    if [[ "${owner}" == "${RECAP_REMOTE_JOB_ROOT}" ]]; then
      rm -f "${lock_dir}/owner"
      rmdir "${lock_dir}" 2>/dev/null || true
    fi
  done <"${reservation_gpus_file}"
  rm -f "${reservation_gpus_file}"
}

owner_job_is_alive() {
  local owner_root="$1"
  local owner_pid=""
  for owner_pid_file in "${owner_root}/control/reservation.pid" "${owner_root}/control/worker.pid"; do
    if [[ -f "${owner_pid_file}" ]]; then
      read -r owner_pid <"${owner_pid_file}" || true
      if [[ "${owner_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${owner_pid}" 2>/dev/null; then
        return 0
      fi
    fi
  done
  return 1
}

release_change_lock() {
  local lock_dir="$1"
  local controller=""
  if [[ -f "${lock_dir}/controller.pid" ]]; then
    read -r controller <"${lock_dir}/controller.pid" || true
  fi
  if [[ -z "${controller}" || "${controller}" == "$$" ]]; then
    rm -f "${lock_dir}/controller.pid"
    rmdir "${lock_dir}" 2>/dev/null || true
  fi
}

acquire_gpu_locks() {
  local gpu_ids="$1"
  local -a gpu_array
  IFS=',' read -r -a gpu_array <<<"${gpu_ids}"
  printf '%s\n' "${gpu_array[@]}" >"${reservation_gpus_file}"
  local gpu lock_dir owner
  for gpu in "${gpu_array[@]}"; do
    lock_dir="${gpu_lock_root}/gpu-${gpu}"
    if ! mkdir "${lock_dir}" 2>/dev/null; then
      owner=""
      if [[ -f "${lock_dir}/owner" ]]; then
        read -r owner <"${lock_dir}/owner" || true
      fi
      if [[ "${owner}" == "${RECAP_REMOTE_JOB_ROOT}" ]]; then
        continue
      fi
      if [[ -n "${owner}" ]] && ! owner_job_is_alive "${owner}"; then
        rm -f "${lock_dir}/owner"
        rmdir "${lock_dir}" 2>/dev/null || true
      fi
      if ! mkdir "${lock_dir}" 2>/dev/null; then
        echo "Remote GPU ${gpu} is reserved by another RECAP job: ${owner:-unknown}" >&2
        release_gpu_locks
        return 1
      fi
    fi
    printf '%s\n' "${RECAP_REMOTE_JOB_ROOT}" >"${lock_dir}/owner"
  done
}

cancel_remote_job() {
  stop_pid_file "${worker_pid_file}" 1
  stop_reservation_controller
  stop_remote_reservation
  release_gpu_locks
  echo "[RECAP remote] cancelled worker and released GPU reservation for ${RECAP_REMOTE_JOB_ROOT}"
}

start_remote_reservation() {
  local gpu_ids="${RECAP_REMOTE_RESERVATION_GPUS:-}"
  local leave_free_mib="${RECAP_REMOTE_RESERVATION_LEAVE_FREE_MIB:-2048}"
  local idle_used_max_mib="${RECAP_REMOTE_RESERVATION_IDLE_USED_MAX_MIB:-64}"
  local max_hold_seconds="${RECAP_REMOTE_RESERVATION_MAX_HOLD_SECONDS:-1800}"
  [[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "Remote reservation requires comma-separated numeric GPU ids: ${gpu_ids}" >&2
    return 2
  }
  [[ "${leave_free_mib}" =~ ^[0-9]+$ ]] && (( leave_free_mib >= 256 )) || {
    echo "Remote reservation leave-free margin must be at least 256 MiB" >&2
    return 2
  }
  [[ "${idle_used_max_mib}" =~ ^[0-9]+$ ]] || {
    echo "Remote reservation idle-used threshold must be non-negative MiB" >&2
    return 2
  }
  [[ "${max_hold_seconds}" =~ ^[0-9]+$ ]] && (( max_hold_seconds >= 60 )) || {
    echo "Remote reservation maximum hold time must be at least 60 seconds" >&2
    return 2
  }

  local existing_pid=""
  if existing_pid=$(read_live_pid "${reservation_pid_file}"); then
    [[ -f "${reservation_ready_file}" ]] || {
      echo "Remote reservation process ${existing_pid} is alive but not ready" >&2
      return 1
    }
    echo "[RECAP remote] reusing GPU reservation pid=${existing_pid} GPUs=${gpu_ids}"
    return 0
  fi

  local lock_dir="${control_dir}/reservation.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    local controller=""
    if [[ -f "${lock_dir}/controller.pid" ]]; then
      read -r controller <"${lock_dir}/controller.pid" || true
    fi
    if [[ ! "${controller}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${controller}" 2>/dev/null; then
      rm -f "${lock_dir}/controller.pid"
      rmdir "${lock_dir}" 2>/dev/null || true
    fi
    if ! mkdir "${lock_dir}" 2>/dev/null; then
      echo "Another client is changing the remote reservation for ${RECAP_REMOTE_JOB_ROOT}" >&2
      return 1
    fi
  fi
  printf '%s\n' "$$" >"${lock_dir}/controller.pid"
  stop_remote_reservation

  local -a gpu_array
  IFS=',' read -r -a gpu_array <<<"${gpu_ids}"
  if ! acquire_gpu_locks "${gpu_ids}"; then
    release_change_lock "${lock_dir}"
    return 1
  fi
  local gpu active_pids used_mib
  for gpu in "${gpu_array[@]}"; do
    used_mib=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    used_mib="${used_mib//[[:space:]]/}"
    if [[ ! "${used_mib}" =~ ^[0-9]+$ ]]; then
      echo "Could not determine existing memory use for remote GPU ${gpu}" >&2
      release_gpu_locks
      release_change_lock "${lock_dir}"
      return 1
    fi
    if (( used_mib > idle_used_max_mib )); then
      echo "Remote GPU ${gpu} is not idle: ${used_mib} MiB already used (limit ${idle_used_max_mib} MiB)" >&2
      release_gpu_locks
      release_change_lock "${lock_dir}"
      return 1
    fi
    active_pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ "${active_pids}" =~ [0-9] ]]; then
      echo "Remote GPU ${gpu} is not idle; active compute PID(s): ${active_pids//$'\n'/,}" >&2
      release_gpu_locks
      release_change_lock "${lock_dir}"
      return 1
    fi
  done

  local reservation_python="${RECAP_REMOTE_REPO_ROOT}/external_dependencies/WCM/.venv/bin/python"
  local reservation_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reserve_gpu_memory.py"
  [[ -x "${reservation_python}" ]] || {
    echo "Remote reservation Python is missing: ${reservation_python}" >&2
    release_gpu_locks
    release_change_lock "${lock_dir}"
    return 1
  }
  [[ -r "${reservation_script}" ]] || {
    echo "Remote reservation helper is missing: ${reservation_script}" >&2
    release_gpu_locks
    release_change_lock "${lock_dir}"
    return 1
  }
  rm -f "${reservation_ready_file}" "${reservation_log}"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" nohup "${reservation_python}" \
    "${reservation_script}" \
      --device-count "${#gpu_array[@]}" \
      --ready-file "${reservation_ready_file}" \
      --leave-free-mib "${leave_free_mib}" \
      --max-hold-seconds "${max_hold_seconds}" \
      --label "remote RECAP ${RECAP_REMOTE_ACTION}" \
      >"${reservation_log}" 2>&1 </dev/null &
  local reservation_pid=$!
  printf '%s\n' "${reservation_pid}" >"${reservation_pid_file}"
  for _ in {1..600}; do
    if [[ -f "${reservation_ready_file}" ]]; then
      echo "[RECAP remote] reserved idle GPU(s) ${gpu_ids}; pid=${reservation_pid}"
      release_change_lock "${lock_dir}"
      return 0
    fi
    if ! kill -0 "${reservation_pid}" 2>/dev/null; then
      wait "${reservation_pid}" 2>/dev/null || true
      sed -n '1,120p' "${reservation_log}" >&2 || true
      stop_remote_reservation
      release_gpu_locks
      release_change_lock "${lock_dir}"
      return 1
    fi
    sleep 0.1
  done
  echo "Remote GPU reservation did not become ready within 60 seconds" >&2
  stop_remote_reservation
  release_gpu_locks
  release_change_lock "${lock_dir}"
  return 1
}

case "${RECAP_REMOTE_ACTION}" in
  cancel)
    cancel_remote_job
    exit 0
    ;;
  reserve)
    start_remote_reservation
    exit $?
    ;;
esac

existing_worker=""
if existing_worker=$(read_live_pid "${worker_pid_file}") && [[ "${existing_worker}" != "$$" ]]; then
  echo "Remote job already has an active worker pid=${existing_worker}" >&2
  exit 1
fi
printf '%s\n' "$$" >"${worker_pid_file}.tmp"
mv "${worker_pid_file}.tmp" "${worker_pid_file}"
launcher_parent="${PPID}"
launcher_parent_start=$(awk '{print $22}' "/proc/${launcher_parent}/stat" 2>/dev/null || true)
launcher_watchdog_pid=""
if [[ -n "${launcher_parent_start}" ]]; then
  (
    while [[ "$(awk '{print $22}' "/proc/${launcher_parent}/stat" 2>/dev/null || true)" == "${launcher_parent_start}" ]]; do
      sleep 2
    done
    echo "Remote SSH launcher disappeared; terminating worker process group" >&2
    kill -TERM -- "-$$" 2>/dev/null || kill -TERM "$$" 2>/dev/null || true
  ) &
  launcher_watchdog_pid=$!
fi
cleanup_worker() {
  if [[ -n "${launcher_watchdog_pid}" ]]; then
    kill -TERM "${launcher_watchdog_pid}" 2>/dev/null || true
    wait "${launcher_watchdog_pid}" 2>/dev/null || true
  fi
  stop_remote_reservation
  release_gpu_locks
  if [[ -f "${worker_pid_file}" ]] && [[ "$(<"${worker_pid_file}")" == "$$" ]]; then
    rm -f "${worker_pid_file}"
  fi
}
trap cleanup_worker EXIT
trap 'exit 130' INT TERM HUP

case "${RECAP_REMOTE_ACTION}" in
  rollout)
    required=(
      RECAP_REMOTE_CHECKPOINT_ARCHIVE RECAP_REMOTE_RESULT_ARCHIVE
      RECAP_REMOTE_TASK RECAP_REMOTE_EPISODES RECAP_REMOTE_LAYOUT_SEED
      RECAP_REMOTE_POLICY_GPU RECAP_REMOTE_ENV_GPU RECAP_REMOTE_ENV_CFG
      RECAP_REMOTE_ACTION_TYPE RECAP_REMOTE_POLICY_ENV RECAP_REMOTE_EVAL_ENV
      RECAP_REMOTE_POLICY
    )
    for name in "${required[@]}"; do
      [[ -n "${!name:-}" ]] || { echo "Missing rollout variable: ${name}" >&2; exit 2; }
    done
    checkpoint_dir="${RECAP_REMOTE_JOB_ROOT}/policy"
    rollout_dir="${RECAP_REMOTE_JOB_ROOT}/rollouts"
    if [[ ! -f "${checkpoint_dir}/.extracted" ]]; then
      rm -rf "${checkpoint_dir}"
      mkdir -p "${checkpoint_dir}"
      tar --zstd -xf "${RECAP_REMOTE_CHECKPOINT_ARCHIVE}" -C "${checkpoint_dir}"
      touch "${checkpoint_dir}/.extracted"
    fi
    mkdir -p "${rollout_dir}/episodes"
    case "${RECAP_REMOTE_POLICY}" in
      pi05) policy_dir="${RECAP_REMOTE_REPO_ROOT}/XPolicyLab/policy/Pi_05" ;;
      g05)
        policy_dir="${RECAP_REMOTE_REPO_ROOT}/XPolicyLab/policy/G05"
        [[ -n "${RECAP_REMOTE_G05_ROOT:-}" ]] || { echo "Missing RECAP_REMOTE_G05_ROOT" >&2; exit 2; }
        export G05_ROOT="${RECAP_REMOTE_G05_ROOT}"
        export G05_ACTION_TOKENIZER_PATH="${checkpoint_dir}/action_tokenizer.pt"
        export ROBODOJO_G05_DATA_STATS="${checkpoint_dir}/dataset_stats.json"
        export ROBODOJO_RECAP_INFERENCE_CONDITION=positive
        export ROBODOJO_G05_ACTION_SOURCE="${RECAP_REMOTE_G05_ACTION_SOURCE:-fm}"
        G05_CKPT_PATH=$(find "${checkpoint_dir}/checkpoints" -maxdepth 1 -type f \( -name 'step_*.pt' -o -name checkpoint \) | sort -V | tail -n 1)
        [[ -n "${G05_CKPT_PATH}" ]] || { echo "Transferred G05 checkpoint is missing" >&2; exit 1; }
        export G05_CKPT_PATH
        ;;
      *) echo "Unsupported RECAP policy: ${RECAP_REMOTE_POLICY}" >&2; exit 2 ;;
    esac
    recorded=$(find "${rollout_dir}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    recorded="${recorded//[[:space:]]/}"
    (( recorded <= RECAP_REMOTE_EPISODES )) || {
      echo "Remote rollout has ${recorded} episodes, expected at most ${RECAP_REMOTE_EPISODES}" >&2
      exit 1
    }
    remaining=$((RECAP_REMOTE_EPISODES - recorded))
    if (( remaining > 0 )); then
      layout_offset=$(( ${RECAP_REMOTE_LAYOUT_OFFSET:-0} + recorded ))
      log="${rollout_dir}/rollout_$(date +%Y%m%dT%H%M%S).log"
      stop_remote_reservation
      ROBODOJO_DISABLE_PROGRESS=1 \
        bash "${RECAP_REMOTE_REPO_ROOT}/scripts/robodojo.sh" eval \
          --policy-dir "${policy_dir}" \
          --task "${RECAP_REMOTE_TASK}" \
          --ckpt "${checkpoint_dir}" \
          --env-cfg "${RECAP_REMOTE_ENV_CFG}" \
          --action-type "${RECAP_REMOTE_ACTION_TYPE}" \
          --seed "${RECAP_REMOTE_LAYOUT_SEED}" \
          --layout-offset "${layout_offset}" \
          --policy-gpu "${RECAP_REMOTE_POLICY_GPU}" \
          --env-gpu "${RECAP_REMOTE_ENV_GPU}" \
          --policy-env "${RECAP_REMOTE_POLICY_ENV}" \
          --eval-env "${RECAP_REMOTE_EVAL_ENV}" \
          --eval-num "${remaining}" \
          --rollout-dir "${rollout_dir}" \
          --no-video >"${log}" 2>&1 || {
            tail -n 100 "${log}" >&2 || true
            exit 1
          }
    fi
    recorded=$(find "${rollout_dir}/episodes" -mindepth 2 -maxdepth 2 -name manifest.json -type f | wc -l)
    recorded="${recorded//[[:space:]]/}"
    (( recorded == RECAP_REMOTE_EPISODES )) || {
      echo "Remote rollout is incomplete: ${recorded}/${RECAP_REMOTE_EPISODES}" >&2
      exit 1
    }
    result_tmp="${RECAP_REMOTE_RESULT_ARCHIVE}.tmp"
    rm -f "${result_tmp}"
    # Isaac/eval writes temporary episodes below rollouts/_in_progress. They
    # are not completed artifacts and may still change during packaging.
    tar --zstd --exclude='rollouts/_in_progress' \
      -cf "${result_tmp}" -C "${RECAP_REMOTE_JOB_ROOT}" rollouts
    mv "${result_tmp}" "${RECAP_REMOTE_RESULT_ARCHIVE}"
    ;;
  run)
    required=(RECAP_REMOTE_COMMAND RECAP_REMOTE_RESULT_ARCHIVE RECAP_REMOTE_OUTPUT_PATH RECAP_REMOTE_OUTPUT_KIND)
    for name in "${required[@]}"; do
      [[ -n "${!name:-}" ]] || { echo "Missing remote run variable: ${name}" >&2; exit 2; }
    done
    [[ "${RECAP_REMOTE_OUTPUT_KIND}" == "directory" || "${RECAP_REMOTE_OUTPUT_KIND}" == "file" ]] || {
      echo "RECAP_REMOTE_OUTPUT_KIND must be directory or file" >&2
      exit 2
    }
    run_log="${RECAP_REMOTE_JOB_ROOT}/run.log"
    stop_remote_reservation
    bash -c "${RECAP_REMOTE_COMMAND}" >"${run_log}" 2>&1 || {
      tail -n 160 "${run_log}" >&2 || true
      exit 1
    }
    result_dir="${RECAP_REMOTE_JOB_ROOT}/result"
    rm -rf "${result_dir}"
    mkdir -p "${result_dir}"
    if [[ "${RECAP_REMOTE_OUTPUT_KIND}" == "directory" ]]; then
      [[ -d "${RECAP_REMOTE_OUTPUT_PATH}" ]] || {
        echo "Remote run output directory is missing: ${RECAP_REMOTE_OUTPUT_PATH}" >&2
        exit 1
      }
      cp -a "${RECAP_REMOTE_OUTPUT_PATH}/." "${result_dir}/"
    else
      [[ -f "${RECAP_REMOTE_OUTPUT_PATH}" ]] || {
        echo "Remote run output file is missing: ${RECAP_REMOTE_OUTPUT_PATH}" >&2
        exit 1
      }
      cp -a "${RECAP_REMOTE_OUTPUT_PATH}" "${result_dir}/"
    fi
    result_tmp="${RECAP_REMOTE_RESULT_ARCHIVE}.tmp"
    rm -f "${result_tmp}"
    tar --zstd -cf "${result_tmp}" -C "${RECAP_REMOTE_JOB_ROOT}" result
    mv "${result_tmp}" "${RECAP_REMOTE_RESULT_ARCHIVE}"
    ;;
  value-video)
    required=(
      RECAP_REMOTE_WCM_ARCHIVE RECAP_REMOTE_RESULT_ARCHIVE
      RECAP_REMOTE_VALUE_EPISODES RECAP_REMOTE_VALUE_GPU
    )
    for name in "${required[@]}"; do
      [[ -n "${!name:-}" ]] || { echo "Missing value-video variable: ${name}" >&2; exit 2; }
    done
    rollout_dir="${RECAP_REMOTE_JOB_ROOT}/rollouts"
    if [[ ! -d "${rollout_dir}/episodes" ]]; then
      rollout_archive="${RECAP_REMOTE_ROLLOUT_ARCHIVE:-}"
      [[ -f "${rollout_archive}" ]] || {
        echo "Local rollout transfer is missing; cannot rebuild ${rollout_dir}" >&2
        exit 1
      }
      rm -rf "${rollout_dir}"
      mkdir -p "${rollout_dir}"
      tar --zstd -xf "${rollout_archive}" -C "${rollout_dir}"
      [[ -d "${rollout_dir}/episodes" ]] || {
        echo "Rebuilt rollout cache has no episodes: ${rollout_dir}" >&2
        exit 1
      }
      echo "[RECAP remote] rebuilt rollout cache from uploaded local artifacts"
    fi
    wcm_checkpoint="${RECAP_REMOTE_JOB_ROOT}/wcm/deploy.pt"
    mkdir -p "$(dirname "${wcm_checkpoint}")"
    "${zstd_bin}" -q -d -f "${RECAP_REMOTE_WCM_ARCHIVE}" -o "${wcm_checkpoint}"
    value_dir="${RECAP_REMOTE_JOB_ROOT}/value_videos"
    rm -rf "${value_dir}"
    stop_remote_reservation
    value_log="${RECAP_REMOTE_JOB_ROOT}/value_video.log"
    CUDA_VISIBLE_DEVICES="${RECAP_REMOTE_VALUE_GPU}" \
      "${RECAP_REMOTE_REPO_ROOT}/external_dependencies/WCM/.venv/bin/python" \
        "${RECAP_REMOTE_REPO_ROOT}/scripts/posttrain/render_rollout_value_videos.py" \
          --wcm-checkpoint "${wcm_checkpoint}" \
          --rollout-root "${rollout_dir}" \
          --output-dir "${value_dir}" \
          --max-episodes "${RECAP_REMOTE_VALUE_EPISODES}" \
          --batch-size "${RECAP_REMOTE_VALUE_BATCH_SIZE:-16}" \
          --device "${RECAP_REMOTE_VALUE_DEVICE:-cuda}" \
          --precision "${RECAP_REMOTE_VALUE_PRECISION:-bf16}" \
          --backend "${RECAP_REMOTE_VALUE_BACKEND:-auto}" \
          --speed "${RECAP_REMOTE_VALUE_SPEED:-1.0}" \
          --y-min "${RECAP_REMOTE_VALUE_Y_MIN:--1.0}" \
          --y-max "${RECAP_REMOTE_VALUE_Y_MAX:-1.0}" \
          --title "${RECAP_REMOTE_VALUE_TITLE:-WCM RECAP}" \
          >"${value_log}" 2>&1 || {
            tail -n 100 "${value_log}" >&2 || true
            exit 1
          }
    result_tmp="${RECAP_REMOTE_RESULT_ARCHIVE}.tmp"
    rm -f "${result_tmp}"
    tar --zstd -cf "${result_tmp}" -C "${RECAP_REMOTE_JOB_ROOT}" value_videos
    mv "${result_tmp}" "${RECAP_REMOTE_RESULT_ARCHIVE}"
    ;;
  *)
    echo "Unknown remote action: ${RECAP_REMOTE_ACTION}" >&2
    exit 2
    ;;
esac
