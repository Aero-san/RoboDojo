# RoboDojo scripts

## Public entry points

| Script | Purpose |
| --- | --- |
| [robodojo.sh](robodojo.sh) | Main CLI: `doctor`, `eval`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download robot/object assets |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `robodojo.sh client` and XPolicyLab) |
| [posttrain/run_wcm.sh](posttrain/run_wcm.sh) | Official WCM train/eval on RoboDojo LeRobot-v2.1 data |
| [posttrain/install_wcm.sh](posttrain/install_wcm.sh) | Create the Python environment declared by WCM |
| [posttrain/run_pi05_rltoken.sh](posttrain/run_pi05_rltoken.sh) | Launch Pi0.5 WCM-actor / RL Token training |
| [posttrain/run_pi05_rltoken_recap.sh](posttrain/run_pi05_rltoken_recap.sh) | Iterated replay-buffer + WCM + RL Token actor training |
| [posttrain/train_pi05_rltoken.py](posttrain/train_pi05_rltoken.py) | WCM actor / RL Token Pi0.5 post-training |
| [posttrain/finetune_pi05_with_wcm.sh](posttrain/finetune_pi05_with_wcm.sh) | WCM-selected OpenPI Pi0.5 fine-tuning |
| [posttrain/train_pi05.py](posttrain/train_pi05.py) | OpenPI Pi0.5 fine-tuning with explicit freeze modes |
| [posttrain/select_wcm_episodes.py](posttrain/select_wcm_episodes.py) | Rank/filter demonstrations with a trained WCM |
| [posttrain/run_recap.sh](posttrain/run_recap.sh) | Model-selectable simulator rollout + WCM + RECAP training |
| [posttrain/render_rollout_value_videos.py](posttrain/render_rollout_value_videos.py) | Score rollout frames with WCM and render official value overlays |
| [posttrain/remote_training.py](posttrain/remote_training.py) | Transfer and execute GPU-bound RECAP stages on a local or SSH host |
| [posttrain/build_replay_buffer.py](posttrain/build_replay_buffer.py) | Aggregate SFT demonstrations and labelled policy rollouts |
| [posttrain/build_replay_buffer_incremental.py](posttrain/build_replay_buffer_incremental.py) | Reuse a preceding RECAP buffer and append one rollout round |
| [posttrain/build_wcm_training_subset.py](posttrain/build_wcm_training_subset.py) | Sample old replay episodes and combine them with every newly appended episode for one WCM update |
| [posttrain/annotate_recap_advantages.py](posttrain/annotate_recap_advantages.py) | Compute WCM N-step advantages and RECAP conditions |
| [posttrain/prepare_recap_dataset.py](posttrain/prepare_recap_dataset.py) | Incrementally append rollouts and refresh model-specific RECAP conditions |
| [posttrain/compute_pi05_norm_stats_incremental.py](posttrain/compute_pi05_norm_stats_incremental.py) | Continue Pi0.5 normalization from a serialized running accumulator |

## Pi0.5 post-training options

All three post-training paths accept `--task` (or `TASK_NAME` for the shell
launchers). The selector accepts a complete episode instruction or a unique
RoboDojo-style slug such as `stack_bowls`; it refuses ambiguous matches.

```bash
# One-task WCM. The output is automatically placed in a task-specific folder.
TASK_NAME=stack_bowls CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/posttrain/run_wcm.sh

# The same task-specific WCM checkpoint can train either objective.
bash scripts/posttrain/run_pi05_rltoken.sh \
  --task stack_bowls \
  --wcm-checkpoint outputs/wcm/robodojo_pi05/stack_bowls/deploy.pt \
  --dataset-root data/RoboDojo_lerobot_v21_video \
  --output outputs/posttrain/stack_bowls_rltoken.pt \
  --objective rltoken

# Direct Pi0.5 fine-tuning, defaulting to the action expert and its heads.
TASK_NAME=stack_bowls \
PI05_FINETUNE_MODE=action_expert \
bash scripts/posttrain/finetune_pi05_with_wcm.sh

# Equivalent flag form for the direct fine-tune wrapper.
bash scripts/posttrain/finetune_pi05_with_wcm.sh \
  --task stack_bowls --finetune-mode action_expert_lora
```

`PI05_FINETUNE_MODE` is one of `full`, `action_expert`,
`action_expert_lora`, `paligemma_lora`, and `all_lora`. `action_expert`
trains the Pi0.5 second Gemma stream together with `action_in_proj`, both
timestep MLPs, and `action_out_proj`; it freezes vision and PaliGemma and is
the recommended single-task default. The LoRA modes use OpenPI's native
Gemma LoRA variants. Other useful controls are `OPENPI_BATCH_SIZE`,
`OPENPI_NUM_TRAIN_STEPS`, `OPENPI_LEARNING_RATE`, `OPENPI_WARMUP_STEPS`,
`OPENPI_WEIGHT_DECAY`, `OPENPI_FSDP_DEVICES`, `OPENPI_WANDB_ENABLED=0`,
`OPENPI_ACTION_EXPERT_VARIANT`, and `OPENPI_PALIGEMMA_VARIANT`.

## Off-policy WCM + RL Token actor

`run_pi05_rltoken_recap.sh` applies the same off-policy data loop to the small
Pi0.5 reference-conditioned actor instead of changing Pi0.5 itself. Iteration 1 builds a
buffer from successful SFT demonstrations, updates WCM, optionally warms up
the RL-token encoder, behavior-clones the actor on successful windows, and
then performs the WCM-guided actor update. The updated actor collects labelled
success and failure rollouts, which are added to every later buffer.

```bash
TASK_NAME=stack_bowls \
BASE_POLICY_CHECKPOINT=$PWD/XPolicyLab/policy/Pi_05/checkpoints/my_sft/59999 \
WCM_TRAIN_GPUS=0,1,2,3,4,5,6,7 \
ACTOR_TRAIN_GPUS=0,1,2,3,4,5,6,7 \
RLTOKEN_ROLLOUT_GPUS=0,1,2,3,4,5,6,7 \
bash scripts/posttrain/run_pi05_rltoken_recap.sh
```

Rollouts store both the action actually executed by the actor and the frozen
Pi0.5 `reference_action`. WCM learns from the executed action, while actor BC
and policy updates continue to condition on the correct base-policy action.
WCM weights and action-normalization statistics are carried between rounds;
its optimizer and episode split are rebuilt for each expanded buffer.

The first actor may be initialized in three mutually exclusive ways:

- With no resume, run `RLTOKEN_ENCODER_WARMUP_STEPS` reconstruction updates
  followed by `RLTOKEN_BC_INIT_STEPS` successful-data BC updates.
- Set `RLTOKEN_ENCODER_RESUME` and/or `RLTOKEN_BC_RESUME` to reuse standalone
  encoder and BC artifacts while completing any remaining initialization.
- Set `INITIAL_ACTOR_CHECKPOINT` to restore encoder, actor, optimizer and step
  state from a complete actor artifact. Later iterations always use this full
  resume behavior automatically.

Fresh actors use `RLTOKEN_ACTOR_MODE=direct`: the actor must learn to reproduce
successful actions during BC, instead of starting as an exact residual
pass-through with zero loss. SFT demonstrations intentionally have
`reference_action == action`, so `baseline=0` is expected. The trainer reports
`target_reference_mse` before BC and rejects `RLTOKEN_ACTOR_MODE=residual` when
that equality would make every BC gradient zero. Residual mode is therefore
only appropriate when successful replay samples already contain executed
actions that differ from their Pi0.5 references. Full and BC resumes retain
the mode stored in their checkpoint.
An encoder-only resume restores only encoder architecture, weights, optimizer,
and encoder-step progress. Actor fields embedded by older encoder artifacts
(including `actor_residual` and `bc_steps`) are ignored; a fresh actor still
uses `RLTOKEN_ACTOR_MODE` and runs the requested BC initialization. A BC or
full actor resume continues to restore its saved actor mode and BC progress.

`RLTOKEN_OBJECTIVE=wcm_actor` differentiates the final imagined WCM value
through the whole candidate action chunk and regularizes it toward the frozen
Pi0.5 reference action, including on failed rollout samples.
`RLTOKEN_OBJECTIVE=rltoken` retains the return-weighted BC objective. Actor
training uses DDP with one process per `ACTOR_TRAIN_GPUS` entry; batch size and
worker count are per process. Actor rollout collection samples its configured
Gaussian by default; set `RLTOKEN_ROLLOUT_DETERMINISTIC=1` to collect means
only. Normal XPolicyLab evaluation remains deterministic unless
`POSTTRAIN_DETERMINISTIC=0` is explicitly exported. See
[pi05_rltoken_recap.yaml.example](../configs/posttrain/pi05_rltoken_recap.yaml.example)
for all common controls.

Rollout cards are paired in list order. With
`RLTOKEN_ROLLOUT_GPUS=0,1,2,3,4,5,6,7`, workers use `(policy=0, Isaac=1)`,
`(2,3)`, `(4,5)`, and `(6,7)`; four cards create two workers. The requested
episode total is divided evenly, and each worker runs
`RLTOKEN_ROLLOUT_ENVS_PER_WORKER` vectorized environments. Workers use
disjoint layout shards and unique run IDs while writing to one replay source;
the launcher verifies the exact episode count before continuing. Worker logs
are stored under each iteration's `rollouts/logs/` directory. Because Pi0.5
inference is currently issued once per environment, start with four envs per
worker on 24 GB cards and increase only after checking memory and throughput.
When the GPU variables are unset or empty, the launcher uses
`CUDA_VISIBLE_DEVICES` and then `nvidia-smi`, so the worker count adapts to the
machine automatically.

Training, dataset preparation, WCM scoring, and rollout collection show one
rank-zero `tqdm` progress bar. Parallel rollout workers write their detailed
output to `rollouts/logs/`, while the terminal shows one aggregate episode
bar, so worker logs cannot leave duplicate progress lines. Set
`ROBODOJO_DISABLE_PROGRESS=1` to disable all of these bars; the legacy
`WCM_DISABLE_PROGRESS=1` switch also disables WCM-side bars.

The final actor remains an XPolicyLab artifact. Score it with the unchanged
Pi0.5 loader:

```bash
POSTTRAIN_MODE=wcm_actor \
POSTTRAIN_CHECKPOINT="$(cat outputs/rltoken_recap/stack_bowls/latest_actor.txt)" \
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task stack_bowls --ckpt /path/to/pi05_sft/59999 \
  --env-cfg arx_x5 --action-type joint --eval-num 1
```

## Off-policy WCM + RECAP

`run_recap.sh` is the model-selectable learning-from-experience pipeline:

1. collect labelled RoboDojo rollouts from the current policy;
2. combine them with task-filtered demonstrations in the replay buffer;
3. update WCM and annotate frame-level RECAP advantages;
4. materialize the policy dataset with model-specific language conditioning;
5. train/evaluate candidate checkpoints and continue from the latest step.

Every iteration consumes its own rollout round. Replay metadata and videos are
reused incrementally, and interrupted rollout collection resumes from complete
episodes. Pi0.5 and G05 share orchestration, WCM, advantage annotation, remote
staging, artifact validation, and reporting; their conditioning, fixed assets,
trainer launch, checkpoint packaging, and candidate discovery are adapters.

### Data formats

`data.format` selects exactly one source reader; readers never fall back across
formats. External LeRobot v2.1/v3.0 sources must match their `meta/info.json`.
`hdf5` reads RoboDojo trajectory files named `episode_*.hdf5` or
`episode_*.h5`, including their embedded camera streams. Sources are normalized
into an explicitly marked internal v2.1 replay layout for WCM. The policy
materializer then writes LeRobot v3.0, which is consumed by G05 through
`BaseLerobotDatasetV3`. Consequently, a remote G05 trainer receives the
normalized v3.0 policy dataset and does not need access to the original HDF5
tree.

The policy dataset is incremental: existing packed videos are hard-linked,
only new rollout episodes are encoded, and task indices are rewritten from the
latest advantage labels. G05 failures use the base instruction; successful
frames use `Advantage: positive` with deterministic unconditional dropout.
Pi0.5 retains explicit positive and negative suffixes.

### Configuration and launch

The launcher requires schema-v2 nested YAML and rejects unknown fields, invalid
enums, incompatible rollout limits, and model/format mismatches before running
a stage. Resolved configuration is saved at
`<output_root>/<task>/resolved_config.yaml`.

```bash
# Pi0.5 template (replace placeholder paths first)
bash scripts/posttrain/run_recap.sh \
  --config configs/posttrain/pi05_recap.yaml.example

# G05 template (replace host paths first)
bash scripts/posttrain/run_recap.sh \
  --config configs/posttrain/g05_recap.yaml.example

# G05 with RoboDojo HDF5 demonstrations
RECAP_CONFIG=configs/posttrain/g05_hdf5_remote.yaml bash remote_training.sh

# Active remote G05 configuration
bash remote_training.sh
```

See [pi05_recap.yaml.example](../configs/posttrain/pi05_recap.yaml.example),
[g05_recap.yaml.example](../configs/posttrain/g05_recap.yaml.example), and
[g05_hdf5_remote.yaml](../configs/posttrain/g05_hdf5_remote.yaml). The
repository default [remote_training.yaml](../configs/posttrain/remote_training.yaml)
selects G05, `data/pickup_video`, LeRobot v3.0, remote rollout, and remote G05
training. `remote_training.sh` uses the RoboDojo conda environment only for
configuration/bootstrap work; the YAML selects separate data, WCM, evaluation,
and policy interpreters.

G05 uses the upstream `scripts/finetune.py` entrypoint with the RoboDojo-owned
Hydra overlays under `configs/g05/`. A G05 bundle contains `.hydra/config.yaml`,
`dataset_stats.json`, `action_tokenizer.pt`, and a checkpoint file. Fixed stats
and tokenizer assets are frozen from the initial bundle for the complete RECAP
run. G05 currently requires joint actions, FM output, equal demonstration and
rollout sampling weights, and `recap.guidance_scale: 1.0`.

GalaxeaVLA currently wraps the model with DDP, so every rank keeps a complete
model and optimizer replica; the upstream use_fsdp task keys are not consumed
by its trainer. RECAP therefore exposes only the native, working controls below
g05.memory: BF16 model weights, bitsandbytes 8-bit AdamW, and independent
vision/VLM/action-expert activation checkpointing. Keep use_8bit_optimizer
unchanged when resuming a partial optimizer checkpoint.

### Remote execution and resume

`rollout.remote` runs the XPolicyLab policy server plus Isaac Sim.
`training.remote` independently stages WCM and policy training on a second
host. For G05, `rollout.remote.g05_root`, `g05.root`, and
`training.remote.policy_python` refer to paths on their respective hosts.

While the launcher is alive, local GPUs listed by `devices.policy_train` or
`devices.wcm_train` are kept occupied whenever they are not performing real
RECAP compute. Full local reservation covers startup/preflight, fixed-asset and
normalization work, replay/WCM/policy dataset generation, artifact checks,
reports, and every remote rollout, evaluation, training, inference, rendering,
upload, download, or result wait. During local rollout/evaluation, WCM,
advantage inference, policy training, or value-video rendering, the launcher
releases only the GPUs used by that workload and keeps the remaining configured
GPUs reserved. It restores the full reservation immediately after the local
workload exits.

Reservation allocates otherwise-free memory rather than running artificial
compute. `devices.reservation.leave_free_mib` remains free on each held card,
`devices.reservation.enabled: false` disables the behavior, and the `EXIT`,
`INT`, and `TERM` cleanup paths terminate the holder so all local memory is
released when the main program ends.

Staging hosts named `local` or `localhost` are treated as real local GPU work,
so their configured cards are excluded from the holder rather than overcommitted.
Remote preflight checks SSH, executables, repository files, upstream G05, tar,
zstd, `setsid`, and requested GPUs.

Repeat the launch with `--resume` to continue. Complete artifacts are reused
only after structural and fingerprint validation; incomplete non-rollout
artifacts are moved to recoverable `.incomplete-*` siblings. WCM and policy
optimizer checkpoints resume when available. `run.reuse_completed_artifacts`
can explicitly trust structurally complete stages after a checkout move.

The continuation checkpoint is written to `latest_policy.txt`, WCM to
`latest_wcm.txt`, and evaluation/report artifacts stay under each iteration.
Policy evaluation is independent of WCM and never gates training.

## Typical eval flow

```text
robodojo.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine (see docs/SPLIT_EVAL.md):

robodojo.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server (bind 0.0.0.0)
robodojo.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

## Headless evaluation

The simulator client is launched with `--headless`, so an X server or desktop
session is not required. Isaac Sim still needs a working NVIDIA driver and
offscreen EGL/Vulkan rendering when the policy consumes camera observations.
`Pi_05` is a vision policy, so `--no-video` disables MP4 encoding but does not
disable cameras or rendering.

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --policy-env openpi \
  --task stack_bowls \
  --ckpt RoboDojo-sim-arx_x5-joint-2 \
  --action-type joint \
  --eval-num 1 \
  --no-video
```

Use `--save-video` (the default) to write one MP4 per camera and rollout under
`eval_result/RoboDojo/...`. The terminal prints the cumulative success rate,
and each run writes `_result.json` containing `success_rate`, `score`,
`eval_time`, `max_steps`, per-rollout `steps`/`termination_reason`, and
`video_enabled`. Videos record the initial observation and every deployed
action, including actions executed from inside an adaptive chunk. The `steps`
field remains the authoritative episode action count.

Every task defines its own episode action limit. Override it for `eval`,
`client`, `smoke`, or `benchmark` with `--max-steps`; omitting the option keeps
the task default:

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task push_T --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --action-type joint --eval-num 1 --max-steps 1200
```

If the machine has no working GPU driver or no offscreen graphics support, a
vision policy cannot be evaluated: disabling video is not equivalent to
disabling camera rendering. Use a GPU machine for the simulator, or use the
split setup with the policy server and simulator client on separate machines.

When multiple physical GPUs are listed in `CUDA_VISIBLE_DEVICES`, `eval`
automatically assigns the first GPU to the policy server and the second GPU to
Isaac Sim. You can also choose them explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --policy-env openpi \
  --ckpt RoboDojo-sim-arx_x5-joint-0 \
  --task pour_by_language \
  --env-cfg arx_x5 \
  --action-type joint \
  --eval-num 10 \
  --policy-gpu 0 \
  --env-gpu 1
```

The `wine_bottle` assets are imported as one RoboDojo rigid object. The
runtime removes an embedded child `RigidBodyAPI` from the asset so the wrapper
and its mesh do not form an invalid nested rigid-body hierarchy.

Run one or more official capability dimensions in a benchmark sweep:

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh benchmark \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further.

## Auto multi-GPU grouping

`robodojo.sh smoke` and `robodojo.sh benchmark` now support balanced
multi-GPU execution driven by the runtime table in `../optimal_8group.txt`.
Pass a concrete GPU id list and RoboDojo will partition the selected tasks
online instead of relying on a hard-coded task group.

Dry-run example:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Launch only a subset of tasks:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num native \
  --only imitate_sorting_sequence,pour_by_language,play_tic_tac_toe \
  --gpu-ids 0,1,3
```

## Internal (`internal/`)

Not intended for direct daily use. Called by `robodojo.sh` or policy utilities.

| File | Called by |
| --- | --- |
| [verify_install.sh](internal/verify_install.sh) | `robodojo.sh doctor` |
| [task_inventory.py](internal/task_inventory.py) | `robodojo.sh tasks` |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | `robodojo.sh smoke` / `benchmark` |
| [summarize_result.py](internal/summarize_result.py) | `robodojo.sh summarize` |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

## Docker

Container install and smoke tests live under [../docker/](../docker/), not here.

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).
