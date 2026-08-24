#!/usr/bin/env bash
# Execute one resumable RECAP simulator or value-video job on a remote GPU host.
set -euo pipefail

required=(RECAP_REMOTE_ACTION RECAP_REMOTE_REPO_ROOT RECAP_REMOTE_JOB_ROOT)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing remote worker variable: ${name}" >&2; exit 2; }
done
command -v zstd >/dev/null 2>&1 || { echo "zstd is required on the remote host" >&2; exit 1; }

case "${RECAP_REMOTE_ACTION}" in
  rollout)
    required=(
      RECAP_REMOTE_CHECKPOINT_ARCHIVE RECAP_REMOTE_RESULT_ARCHIVE
      RECAP_REMOTE_TASK RECAP_REMOTE_EPISODES RECAP_REMOTE_LAYOUT_SEED
      RECAP_REMOTE_POLICY_GPU RECAP_REMOTE_ENV_GPU RECAP_REMOTE_ENV_CFG
      RECAP_REMOTE_ACTION_TYPE RECAP_REMOTE_POLICY_ENV RECAP_REMOTE_EVAL_ENV
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
      ROBODOJO_DISABLE_PROGRESS=1 \
        bash "${RECAP_REMOTE_REPO_ROOT}/scripts/robodojo.sh" eval \
          --policy-dir "${RECAP_REMOTE_REPO_ROOT}/XPolicyLab/policy/Pi_05" \
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
    tar --zstd -cf "${result_tmp}" -C "${RECAP_REMOTE_JOB_ROOT}" rollouts
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
    [[ -d "${rollout_dir}/episodes" ]] || {
      echo "Remote rollout cache is missing: ${rollout_dir}" >&2
      exit 1
    }
    wcm_checkpoint="${RECAP_REMOTE_JOB_ROOT}/wcm/deploy.pt"
    mkdir -p "$(dirname "${wcm_checkpoint}")"
    zstd -q -d -f "${RECAP_REMOTE_WCM_ARCHIVE}" -o "${wcm_checkpoint}"
    value_dir="${RECAP_REMOTE_JOB_ROOT}/value_videos"
    rm -rf "${value_dir}"
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
          --title "${RECAP_REMOTE_VALUE_TITLE:-WCM RECAP}"
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
