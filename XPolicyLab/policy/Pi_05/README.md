# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
OPENPI_INSTRUCTION="Stack the bowls." \
  bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls

# If using RoboDojo's existing v2.1 LeRobot video export, filter it directly.
cd ../../../
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python scripts/posttrain/prepare_pi05_dataset.py \
  --dataset-root data/RoboDojo_lerobot_v21_video \
  --repo-id RoboDojo-stack_bowls-arx_x5-joint \
  --task stack_bowls --mode video
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: fine-tune one task on four GPUs from the latest step in an existing run
OPENPI_INIT_CHECKPOINT="$PWD/checkpoints/RoboDojo-sim-arx_x5-joint-0" \
OPENPI_FINETUNE_MODE=action_expert_lora \
OPENPI_PARAMETER_DTYPE=bfloat16 \
OPENPI_BATCH_SIZE=64 \
OPENPI_NUM_TRAIN_STEPS=10000 \
OPENPI_LEARNING_RATE=1e-5 \
OPENPI_WANDB_ENABLED=0 \
  bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0,1,2,3
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset.

`OPENPI_INIT_CHECKPOINT` accepts a run directory, a numeric step directory, or its `params/` directory. It initializes a new run from model weights and automatically reuses that step's normalization assets. `OPENPI_RESUME=1` has different semantics: it restores the latest full training state from the output run, including optimizer, step, EMA, and data-loader state. Do not set both. Existing output is never deleted unless `OPENPI_OVERWRITE=1` is explicit.

All comma-separated GPUs participate in the JAX mesh. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default. `OPENPI_FSDP_DEVICES` must divide the visible GPU count; the remaining mesh dimension is data parallel. The global batch size must also be divisible by the visible GPU count.

Memory sharding and host offload are configurable:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `OPENPI_SHARDING_STRATEGY` | `full_shard`, `shard_grad_op`, `no_shard` | `full_shard` | `full_shard` minimizes per-GPU memory; `shard_grad_op` keeps parameters replicated and is usually faster when they fit; `no_shard` replicates the complete train state. |
| `OPENPI_CPU_OFFLOAD` | `0`, `1` | `0` | Keep Adam/optimizer state in pinned host memory between steps. Parameters retain their selected GPU sharding. This saves GPU memory but is slower, so it is disabled by default. |

The existing multi-GPU path already used full parameter/optimizer-state sharding; it did not use CPU offload, so `full_shard` remains the safe default. Select `shard_grad_op` only when the replicated model and EMA copies fit comfortably. Enable `OPENPI_CPU_OFFLOAD=1` only if `full_shard` still runs out of memory.

During training, the progress bar shows the latest `loss`, `lr`, `grad_norm`, and `param_norm`. Every `log_interval` steps, the averaged `loss`, `lr`, `grad_norm`, `param_norm`, and timestamp are appended to `training_metrics.json` in the run checkpoint directory. After checkpoint finalization, each step directory receives the metric history up to that checkpoint.

Fine-tuning mode is selected with `OPENPI_FINETUNE_MODE`:

| Mode | Trainable parameters |
|---|---|
| `full` | Every Pi0.5 parameter. |
| `action_expert` | The complete action Gemma stream plus action and timestep input/output projections. |
| `action_expert_lora` | Action-expert LoRA parameters plus action and timestep input/output projections. |
| `paligemma_lora` | PaliGemma LoRA parameters only. |
| `all_lora` | LoRA parameters in both PaliGemma and the action expert. |

LoRA variants are selected automatically. The run metadata is written immediately and copied into completed checkpoint steps, so XPolicyLab reconstructs the matching LoRA architecture during evaluation. When resuming, use the same fine-tuning mode as the original run.

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_INIT_CHECKPOINT` | Initialize a new run from an existing OpenPI run/step/`params` path; also reuses its assets by default. |
| `OPENPI_FINETUNE_MODE` | `full`, `action_expert`, `action_expert_lora`, `paligemma_lora`, or `all_lora`; defaults to `full`. |
| `OPENPI_PARAMETER_DTYPE` | Train-state parameter/optimizer dtype: `bfloat16` or `float32`; defaults to `bfloat16`. |
| `OPENPI_RESUME` | Set to `1` to restore the full state of the same output run. |
| `OPENPI_OVERWRITE` | Set to `1` to delete and recreate an existing output run. |
| `OPENPI_BATCH_SIZE` | Global batch size; must be divisible by the visible GPU count. |
| `OPENPI_NUM_TRAIN_STEPS` | Final training step for a new or resumed run. |
| `OPENPI_NUM_WORKERS` | Data-loader worker count. |
| `OPENPI_LEARNING_RATE` | Peak learning rate. |
| `OPENPI_WARMUP_STEPS`, `OPENPI_DECAY_STEPS`, `OPENPI_DECAY_LR` | Cosine schedule controls. |
| `OPENPI_WEIGHT_DECAY`, `OPENPI_CLIP_GRADIENT_NORM` | AdamW controls. |
| `OPENPI_LOG_INTERVAL`, `OPENPI_SAVE_INTERVAL`, `OPENPI_KEEP_PERIOD` | Logging and checkpoint cadence. |
| `OPENPI_EMA_DECAY` | EMA decay passed to OpenPI. Set it to `None` to disable the extra EMA parameter copy and reduce GPU memory. |
| `OPENPI_WANDB_ENABLED` | `1` enables Weights & Biases, `0` disables it. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_SHARDING_STRATEGY` | `full_shard` (memory-safe default), `shard_grad_op` (faster when replicated parameters fit), or `no_shard` (replicated state). |
| `OPENPI_CPU_OFFLOAD` | `1` keeps optimizer state in pinned host memory between steps while parameters retain their GPU sharding; `0` (default) avoids host-device transfer overhead. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_INSTRUCTION` | Default natural-language task prompt for demonstrations without embedded instructions. |
| `OPENPI_ASSETS_DIR`, `OPENPI_ASSET_ID` | Explicit normalization assets override. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

Any additional arguments after `gpu_id` are forwarded to OpenPI's Tyro CLI, so less common fields in `training/config.py` remain configurable without editing source. Use `OPENPI_DRY_RUN=1` to print the resolved command without starting training. `OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

### Bernoulli-Continuation Policy (BCP)

BCP implements Xu et al., *Continue or Replan? Bernoulli-Continuation Policy
Learning for Adaptive Horizon Execution* (arXiv:2608.03483). The frozen JAX
Pi0.5 policy exposes its visual-language token representations, normalized
denoised actions, and final flow velocity to a two-layer Transformer head. The
head produces `M-1` Bernoulli continuation logits for the ordered horizon set;
training uses the paper's fixed-horizon-anchored GRPO objective and
Replanning-Efficiency Reward.

The paper specifies horizons `(15, 20, ..., 50)`, two Transformer layers,
`delta_positive=0.7`, `delta_negative=0.3`, group size 8, 300 GRPO steps, and
two update epochs. It does not publish the head width, attention-head count,
optimizer learning rate, or PPO clipping bounds. These are therefore explicit
deployment/training parameters; this adapter defaults to width 512, 8 heads,
learning rate `1e-4`, and symmetric clipping `0.2`. Pi0.5 BCP feature export
currently supports JAX checkpoints; PyTorch OpenPI checkpoints fail clearly
instead of silently using different features.

BCP and AAC are mutually exclusive horizon controllers. `BCP_ENABLED=1`
always disables AAC internally, including when `AAC_ENABLED=1` or the AAC YAML
default is enabled.

Collect each GRPO group from the same task layouts: one fixed-50 reference and
seven stochastic adaptive rollouts. Use the same `BCP_GROUP_ID`, checkpoint,
task, and eval layout set for all eight runs, and vary `BCP_SEED` across the
seven adaptive runs. The first adaptive run can infer feature dimensions and
write the initial seeded checkpoint:

```bash
COMMON="--policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi --task stack_bowls --ckpt <PI05_CKPT> --action-type joint --eval-num 10"

BCP_ENABLED=1 BCP_DETERMINISTIC=0 BCP_SEED=1 \
BCP_INITIALIZE_CHECKPOINT=/tmp/bcp_round0.pt \
BCP_ROLLOUT_DIR=/tmp/bcp_rollouts BCP_GROUP_ID=round0 \
  bash scripts/robodojo.sh eval ${COMMON}

# Repeat with seeds 2..7 and the initialized checkpoint.
BCP_ENABLED=1 BCP_DETERMINISTIC=0 BCP_SEED=2 \
BCP_CHECKPOINT=/tmp/bcp_round0.pt \
BCP_ROLLOUT_DIR=/tmp/bcp_rollouts BCP_GROUP_ID=round0 \
  bash scripts/robodojo.sh eval ${COMMON}

# One fixed-horizon reference over the identical layouts.
BCP_ENABLED=1 BCP_REFERENCE=1 \
BCP_ROLLOUT_DIR=/tmp/bcp_rollouts BCP_GROUP_ID=round0 \
  bash scripts/robodojo.sh eval ${COMMON}

XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  XPolicyLab/policy/Pi_05/train_bcp.py \
  --rollout-dir /tmp/bcp_rollouts \
  --checkpoint /tmp/bcp_round0.pt \
  --output /tmp/bcp_round1.pt --group-size 8 --epochs 2
```

All-reference-and-adaptive-failure groups are discarded as in the paper. A
reference reward is binary success; adaptive successful trajectories receive
the relative VLA-call efficiency adjustment. Rollouts store frozen features,
the sampled horizon index, and its old log probability, so Pi0.5 remains
frozen throughout training.

Evaluate deterministically with:

```bash
BCP_ENABLED=1 BCP_CHECKPOINT=/tmp/bcp_round1.pt BCP_DETERMINISTIC=1 \
  bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task stack_bowls --ckpt <PI05_CKPT> --action-type joint --eval-num 10
```

Environment overrides are `BCP_ENABLED`, `BCP_CHECKPOINT`,
`BCP_INITIALIZE_CHECKPOINT`, `BCP_DETERMINISTIC`, `BCP_REFERENCE`,
`BCP_ROLLOUT_DIR`, `BCP_GROUP_ID`, and `BCP_SEED`.

### Adaptive Action Chunking

`deploy.yml` enables inference-time Adaptive Action Chunking (AAC) by default.
It draws 20 flow samples in one policy call, reuses the observation KV cache,
computes Gaussian entropy for each continuous arm group and Bernoulli entropy
for each gripper dimension, and executes the paper's selected prefix of sample
0. No retraining or AAC-specific checkpoint is required.

The complete AAC configuration is under `adaptive_action_chunking`:

| Key | Paper/default value | Meaning |
|---|---:|---|
| `enabled` | `true` | Enable adaptive rather than full fixed-horizon execution. |
| `num_samples` | `20` | Candidate chunks used to estimate the action distribution. |
| `min_chunk_size` | `2` | Absolute lower bound used by the reference implementation. |
| `max_chunk_size` | `null` | Optional cap; `null` uses the model action horizon. |
| `movement_threshold` | `3.0` | Appendix A minimum movement energy, alpha; retune when moving to a differently scaled action representation. |
| `covariance_regularization` | `1e-6` | Diagonal jitter for a stable covariance log determinant. |
| `discrete_threshold` | `0.5` | Threshold that maps executable RoboDojo gripper predictions to open/close for entropy. |
| `magnitude_discrete_threshold` | `0.5` | Threshold for executable RoboDojo gripper actions. |
| `*_entropy_weight` | `1.0` | Weights in the sum of per-component entropies. |
| `*_magnitude_weight` | `1.0` | Weights in the minimum-movement constraint. |
| `candidate_index` | `0` | Candidate whose selected prefix is executed. |

Equation 7 in the paper is an unweighted sum, so the deployment defaults all
magnitude weights to `1.0`. The released LIBERO helper instead hard-codes a
`0.2` gripper-magnitude weight; set `discrete_magnitude_weight: 0.2` to
reproduce that implementation detail. Its gripper range is `[-1, 1]`, whereas
RoboDojo uses `[0, 1]`, which is why both RoboDojo thresholds default to `0.5`.

For one-off fixed/AAC comparisons, environment variables override the most
common settings without editing YAML:

```bash
# Paper-default AAC (also the deploy.yml default)
AAC_ENABLED=1 bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task stack_bowls --ckpt <CKPT> --action-type joint --eval-num 1

# Original full fixed-horizon behavior
AAC_ENABLED=0 bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task stack_bowls --ckpt <CKPT> --action-type joint --eval-num 1
```

Supported overrides are `AAC_ENABLED`, `AAC_NUM_SAMPLES`,
`AAC_MIN_CHUNK_SIZE`, `AAC_MAX_CHUNK_SIZE`, and
`AAC_MOVEMENT_THRESHOLD`. Joint control treats each arm as one continuous
group; end-effector control separates translation and rotation. Gripper
dimensions are discrete in both layouts.

## RoboDojo WCM / RL Token post-training

The RoboDojo repository adds the official WCM checkout at
`external_dependencies/WCM`.  From the RoboDojo root, initialize it and train
WCM directly on the distributed v2.1 video export:

```bash
git submodule update --init --recursive external_dependencies/WCM
./scripts/posttrain/install_wcm.sh
WCM_DATASET_ROOT=$PWD/data/RoboDojo_lerobot_v21_video \
  bash scripts/posttrain/run_wcm.sh
```

The released expert export has no reward column.  It defaults to treating
every episode as successful; for mixed success/failure rollouts pass a JSON
map of `episode_index` to `true`/`false` with `WCM_SUCCESS_LABELS=/path/labels.json`.
The WCM artifact is written to `outputs/wcm/robodojo_pi05/deploy.pt`.

For WCM-only offline evaluation, set `MODE=eval` and `WCM_CHECKPOINT` to the
resulting `deploy.pt` (or `best.pt`).

RL Token and WCM-actor artifacts use the same Pi0.5 model-server contract:

```bash
scripts/posttrain/run_pi05_rltoken.sh \
  --wcm-checkpoint outputs/wcm/robodojo_pi05/deploy.pt \
  --dataset-root data/RoboDojo_lerobot_v21_video \
  --output outputs/posttrain/pi05_wcm_actor.pt --objective wcm_actor

POSTTRAIN_MODE=wcm_actor \
POSTTRAIN_CHECKPOINT=$PWD/outputs/posttrain/pi05_wcm_actor.pt \
  bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
    --task stack_bowls --ckpt RoboDojo-sim-arx_x5-joint-2 \
    --action-type joint --eval-num 1 --no-video
```

Set `posttrain_mode: rltoken` or `wcm_actor` in `deploy.yml`, or export
`POSTTRAIN_MODE=rltoken|wcm_actor` alongside `POSTTRAIN_CHECKPOINT`, to enable
the reference-conditioned actor. The actor
returns physical RoboDojo actions after restoring the WCM action statistics;
the adapter uses the robot/action dimensions from `env_cfg_type`, supports
single- and dual-arm layouts (including `arx_x5` and
`dual_x5_and_franka_competition`), and keeps the standard `Model`/WebSocket/
`scripts/robodojo.sh` loading path.  A direct WCM-weighted Pi0.5 fine-tune is
available via `scripts/posttrain/finetune_pi05_with_wcm.sh`, which prepares a
LeRobot dataset from the RoboDojo v2.1 export and invokes the existing OpenPI
`train.sh` entry point.

For iterative actor improvement, run
`scripts/posttrain/run_pi05_rltoken_recap.sh` from the RoboDojo root. It starts
from successful SFT data, updates WCM, performs encoder warmup and actor BC
initialization when needed, updates the actor, and appends labelled
actor rollouts to the next replay buffer. `INITIAL_ACTOR_CHECKPOINT` restores
the full actor/encoder/optimizer state; `RLTOKEN_ENCODER_RESUME` and
`RLTOKEN_BC_RESUME` load standalone initialization artifacts. During actor
rollouts the WebSocket response also carries the frozen Pi0.5 action chunk so
the recorder can store it as `reference_action`; this metadata does not change
the normal action response seen by RoboDojo evaluation code.
`RLTOKEN_ROLLOUT_GPUS` is paired in order into independent policy-server and
Isaac workers, while `RLTOKEN_ROLLOUT_ENVS_PER_WORKER` controls vectorized
environments inside each Isaac process. Every worker receives a disjoint
layout shard, so concurrent trajectories can be merged without duplicates.

The learning-from-experience path is `scripts/posttrain/run_pi05_recap.sh` in
the RoboDojo root. Its first update uses SFT data only; each updated policy
then collects simulator-labelled attempts for the next round. It retains both
failures and successes, trains WCM with globally min-max-normalized returns and
`C_fail=300`, and optimizes the explicit RECAP objective
`L_unconditional + beta * L_conditioned`. RECAP
checkpoints remain standard OpenPI checkpoints. Their
`robodojo_pi05_model.json` enables the positive advantage condition
automatically in this `Model` loader, so the checkpoint path can be passed
directly to `scripts/robodojo.sh --ckpt` without a separate policy class.
