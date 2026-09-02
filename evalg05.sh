
set -euo pipefail

cd /share/mingyang/RoboDojo

CHECKPOINT_ROOT="/share/mingyang/recap_remote_jobs/jobs/<job-id>/policy"
ROLLOUT_DIR="/share/mingyang/recap_remote_jobs/manual_tests/g05_$(date +%Y%m%dT%H%M%S)"
PREPARER="/share/mingyang/recap_remote_jobs/bin/prepare_g05_inference_checkpoint.py"

test -f "${CHECKPOINT_ROOT}/.hydra/config.yaml"
test -f "${CHECKPOINT_ROOT}/action_tokenizer.pt"
test -f "${CHECKPOINT_ROOT}/dataset_stats.json"
test -f "${PREPARER}"

# 确认远程是包含本次 vqvae_type 修复的新脚本
grep -q "_materialize_action_tokenizer_config" "${PREPARER}"

# 修正远程 checkpoint 副本中的推理配置
/share/mingyang/miniconda3/envs/RoboDojo/bin/python \
"${PREPARER}" \
--checkpoint-root "${CHECKPOINT_ROOT}"

export G05_ROOT="/share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA"
export G05_HF_PROCESSOR_PATH="/share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA/checkpoints/
qwen3_5_2b_base_processor"
export G05_ACTION_TOKENIZER_PATH="${CHECKPOINT_ROOT}/action_tokenizer.pt"
export ROBODOJO_G05_DATA_STATS="${CHECKPOINT_ROOT}/dataset_stats.json"
export ROBODOJO_RECAP_INFERENCE_CONDITION="positive"
export ROBODOJO_G05_ACTION_SOURCE="fm"

export G05_CKPT_PATH="$(
find "${CHECKPOINT_ROOT}/checkpoints" \
-maxdepth 1 -type f \
\( -name 'step_*.pt' -o -name checkpoint \) |
sort -V |
tail -n 1
)"

test -n "${G05_CKPT_PATH}"
test -f "${G05_CKPT_PATH}"
test -f "${G05_HF_PROCESSOR_PATH}/tokenizer.json"

unset ROBODOJO_LEROBOT_V30_ROOT
unset G05_OUTPUT_DIR
unset WANDB_PROJECT
unset WANDB_ENTITY


ROBODOJO_DISABLE_PROGRESS=1 \
bash /share/mingyang/RoboDojo/scripts/robodojo.sh eval \
--policy-dir /share/mingyang/RoboDojo/XPolicyLab/policy/G05 \
--task general_pickup \
--ckpt "${CHECKPOINT_ROOT}" \
--env-cfg arx_x5 \
--action-type joint \
--seed 0 \
--layout-offset 0 \
--policy-gpu 0 \
--env-gpu 1 \
--policy-env /share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA/.venv \
--eval-env /share/mingyang/miniconda3/envs/RoboDojo \
--eval-num 1 \
--rollout-dir "${ROLLOUT_DIR}" \
--no-video