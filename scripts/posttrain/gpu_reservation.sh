#!/usr/bin/env bash
# Shared lifecycle helpers for reserving otherwise-idle GPUs throughout a run.

GPU_RESERVATION_PID="${ROBODOJO_GPU_RESERVATION_PID:-}"
GPU_RESERVATION_READY_FILE="${ROBODOJO_GPU_RESERVATION_READY_FILE:-}"
GPU_RESERVATION_GPU_IDS="${ROBODOJO_GPU_RESERVATION_GPU_IDS:-}"
GPU_RESERVATION_TEMP_DIR=""

start_gpu_reservation() {
  local gpu_ids="${1//[[:space:]]/}"
  local python_bin="$2"
  local label="$3"
  if [[ "${GPU_RESERVATION_ENABLED:-1}" == "0" ]]; then return; fi
  if [[ -n "${GPU_RESERVATION_PID}" ]]; then
    if kill -0 "${GPU_RESERVATION_PID}" 2>/dev/null && \
        [[ "${GPU_RESERVATION_GPU_IDS}" == "${gpu_ids}" ]]; then
      return
    fi
    stop_gpu_reservation
  fi
  [[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "GPU reservation requires comma-separated numeric GPU ids, got: ${gpu_ids}" >&2
    return 2
  }
  local -a gpu_id_array
  IFS=',' read -r -a gpu_id_array <<< "${gpu_ids}"
  GPU_RESERVATION_TEMP_DIR=$(mktemp -d /tmp/robodojo-gpu-reservation.XXXXXX)
  GPU_RESERVATION_READY_FILE="${GPU_RESERVATION_TEMP_DIR}/ready"
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "${python_bin}" \
    "${SCRIPT_DIR}/reserve_gpu_memory.py" \
      --device-count "${#gpu_id_array[@]}" \
      --ready-file "${GPU_RESERVATION_READY_FILE}" \
      --leave-free-mib "${GPU_RESERVATION_FREE_MIB:-2048}" \
      --max-hold-seconds "${GPU_RESERVATION_LOCAL_MAX_HOLD_SECONDS:-1800}" \
      --label "${label}" &
  GPU_RESERVATION_PID=$!
  GPU_RESERVATION_GPU_IDS="${gpu_ids}"
  export ROBODOJO_GPU_RESERVATION_PID="${GPU_RESERVATION_PID}"
  export ROBODOJO_GPU_RESERVATION_READY_FILE="${GPU_RESERVATION_READY_FILE}"
  export ROBODOJO_GPU_RESERVATION_GPU_IDS="${GPU_RESERVATION_GPU_IDS}"
  while [[ ! -f "${GPU_RESERVATION_READY_FILE}" ]]; do
    if ! kill -0 "${GPU_RESERVATION_PID}" 2>/dev/null; then
      wait "${GPU_RESERVATION_PID}" || true
      echo "GPU reservation failed during ${label}" >&2
      return 1
    fi
    sleep 0.1
  done
}

stop_gpu_reservation() {
  if [[ -n "${GPU_RESERVATION_PID}" ]]; then
    kill -TERM "${GPU_RESERVATION_PID}" 2>/dev/null || true
    wait "${GPU_RESERVATION_PID}" 2>/dev/null || true
  fi
  if [[ -n "${GPU_RESERVATION_READY_FILE}" && -f "${GPU_RESERVATION_READY_FILE}" ]]; then
    unlink "${GPU_RESERVATION_READY_FILE}"
  fi
  if [[ -n "${GPU_RESERVATION_TEMP_DIR}" && -d "${GPU_RESERVATION_TEMP_DIR}" ]]; then
    rmdir "${GPU_RESERVATION_TEMP_DIR}" 2>/dev/null || true
  fi
  GPU_RESERVATION_PID=""
  GPU_RESERVATION_READY_FILE=""
  GPU_RESERVATION_GPU_IDS=""
  GPU_RESERVATION_TEMP_DIR=""
  unset ROBODOJO_GPU_RESERVATION_PID ROBODOJO_GPU_RESERVATION_READY_FILE ROBODOJO_GPU_RESERVATION_GPU_IDS
}

install_gpu_reservation_exit_trap() {
  trap stop_gpu_reservation EXIT
}
