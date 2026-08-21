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
| [posttrain/run_pi05_recap.sh](posttrain/run_pi05_recap.sh) | Iterated simulator rollout + WCM + RECAP Pi0.5 training |
| [posttrain/render_rollout_value_videos.py](posttrain/render_rollout_value_videos.py) | Score rollout frames with WCM and render official value overlays |
| [posttrain/build_replay_buffer.py](posttrain/build_replay_buffer.py) | Aggregate SFT demonstrations and labelled policy rollouts |
| [posttrain/build_replay_buffer_incremental.py](posttrain/build_replay_buffer_incremental.py) | Reuse a preceding RECAP buffer and append one rollout round |
| [posttrain/build_wcm_training_subset.py](posttrain/build_wcm_training_subset.py) | Sample old replay episodes and combine them with every newly appended episode for one WCM update |
| [posttrain/annotate_recap_advantages.py](posttrain/annotate_recap_advantages.py) | Compute WCM N-step advantages and RECAP conditions |
| [posttrain/prepare_pi05_recap_dataset.py](posttrain/prepare_pi05_recap_dataset.py) | Incrementally append RECAP rollouts and refresh Pi0.5 conditions |
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

`run_pi05_recap.sh` is the end-to-end learning-from-experience path. Iteration
1 contains no rollout: it initializes WCM and Pi0.5 from up to
`RECAP_MAX_DEMO_EPISODES` successful SFT demonstrations (100 by default).
Starting at iteration 2, the preceding policy collects successes and failures,
the complete new round is appended to the replay buffer, WCM and Pi0.5 are
updated, and the final iteration's collected round is consumed as well. Each
round exposes the complete replay buffer, but materializes it incrementally by
copying the small parquet/metadata prefix, hard-linking existing videos, and
appending only the newest rollout round. RLToken RECAP uses the same
incremental buffer path and already follows train-then-rollout ordering.
This implements the advantage-conditioned objective from
[RECAP / $\pi^*_{0.6}$](https://arxiv.org/abs/2511.14759) using Pi0.5's native
continuous-action flow-matching loss.

Pi0.5 dataset conversion is incremental. The first iteration performs the
one-time LeRobot-v3 materialization. Each later iteration copies only the
small parquet/metadata tree, hard-links the preceding packed videos, decodes
and encodes only newly appended rollout episodes, and then rewrites the
lightweight `task_index`/task metadata for all frames using the latest global
advantage threshold. If a run is resumed after advantage annotation, an
already materialized dataset with the same episode prefix is updated in place,
so no videos are processed again. The source prefix and episode lengths are
validated before reuse; a provenance manifest is retained under
`meta/recap_incremental.json`.

Pi0.5 normalization is incremental when `OPENPI_NORM_MAX_FRAMES=0` (the
default). Alongside `norm_stats.json`, each iteration stores count, first and
second moments, min/max, and OpenPI's 5000-bin quantile histograms. The next
iteration processes only newly appended frames and continues those exact
running statistics. A legacy normalization directory without an accumulator
triggers one final full scan, after which later rounds are incremental. A
nonzero `OPENPI_NORM_MAX_FRAMES` intentionally keeps full randomized sampling
each round because that sample is not a stable dataset prefix.

WCM update cost is bounded after iteration 1. Its per-iteration dataset is all
new rollout episodes plus a deterministic sample of at most
`RECAP_WCM_REPLAY_EPISODES` old episodes (20 by default). The sampled dataset
is stored under `iteration_XX/wcm_training_buffer`; the complete
`replay_buffer` remains unchanged. Advantage inference must still rescore every
state because WCM parameters change each round. Return targets in the sampled
WCM dataset retain the complete replay buffer's global min-max coordinates, so
critic values remain comparable with full-buffer advantage inference. Pi0.5
and RLToken actor updates likewise train from the complete replay distribution,
although they warm-start from the prior policy/actor.

```bash
TASK_NAME=stack_bowls \
DEMO_ROOT=$PWD/data/RoboDojo_lerobot_v21_video \
INITIAL_POLICY_CHECKPOINT=$PWD/XPolicyLab/policy/Pi_05/checkpoints/my_sft/59999 \
RECAP_ITERATIONS=3 RECAP_ROLLOUT_EPISODES=50 \
bash scripts/posttrain/run_pi05_recap.sh
```

To continue an interrupted run, repeat the same command with `--resume` (or
set `RECAP_RESUME=1`). The launcher validates replay-buffer metadata, WCM and
Pi0.5 checkpoints, advantage labels, normalization assets, rollout manifests,
and value-video outputs in order. Complete stages are reused. Incomplete
non-rollout artifacts are moved to a timestamped `.incomplete-*` sibling
before rebuilding. WCM `last.pt` and OpenPI optimizer checkpoints resume
inside their respective training stages when available. Completed rollout
episodes are preserved and only the remaining episode count is collected.
Iteration 1 has no rollout; later interrupted rollout rounds preserve completed
episodes and continue from the next layout.

After each rollout-bearing iteration, the launcher scores the first three newly
collected rollouts by default (or all of them when fewer than three were requested) and
writes head-view monitoring videos to
`iteration_XX/value_videos/videos/`. The translucent lower-half chart is the
official WCM `episode_value_video` renderer. WCM inference still consumes all
camera views listed in its checkpoint; only the presentation layer uses the
head camera. Set `RECAP_VALUE_VIDEO_EPISODES=N` or pass
`--value-video-episodes N` to choose the count, and use `0` to disable it.
`RECAP_VALUE_VIDEO_GPU` selects the inference card. The default fixed y-axis
of `[-1, 1]` makes values comparable between iterations; it can be changed
with `RECAP_VALUE_VIDEO_Y_MIN` and `RECAP_VALUE_VIDEO_Y_MAX`. Raw curves,
success labels, alignment reports, previews, and a summary JSON are retained
beside the videos.

Set `RECAP_MAX_DEMO_EPISODES=20`, or pass `--max-demo-episodes 20`, to use
only the first 20 demonstrations after filtering to the requested task and
sorting by original `episode_index`. The same fixed subset is included in
every replay-buffer iteration. The default is 100; setting it to `0`
explicitly keeps all available demonstrations.

The default RECAP settings use 10-step value lookahead, one global threshold
selecting the top 30% of advantages, `gamma=1`, and 10% positive-sample CFG
dropout. Demonstrations are always positive; rollout labels use the inclusive
global threshold. Negative samples always use the base prompt, while positive
samples use `Advantage: positive` unless dropped to the base prompt. Return
labels and N-step rewards share the global `[-1,0]` normalization used by the
critic. Set `RECAP_LOOKAHEAD`, `RECAP_POSITIVE_FRACTION`,
`RECAP_UNCONDITIONAL_PROB`, `RECAP_GAMMA`, and `WCM_FAILURE_PENALTY` to tune
these values. The current JAX integration intentionally accepts only
`RECAP_GUIDANCE_SCALE=1.0`, where guided inference reduces to the positive
conditional branch.
Every iteration starts from the preceding policy checkpoint. WCM can be warm
started with `INITIAL_WCM_CHECKPOINT`; model weights and action-normalization
statistics are reused while the optimizer and episode split are rebuilt after
the buffer changes.

Robot layout is selected by `ENV_CFG_TYPE` and `ACTION_TYPE`. The trainer
derives the joint/gripper delta mask from `env_cfg/robot/_robot_info.json` and
recomputes normalization statistics for that buffer, covering both single-
and dual-arm RoboDojo configurations. Fine-tuning still supports `full`,
`action_expert`, `action_expert_lora`, `paligemma_lora`, and `all_lora` via
`PI05_FINETUNE_MODE`.

Pass `--config configs/posttrain/pi05_recap.yaml.example` to load a flat YAML
configuration. Command-line flags override environment variables, which
override YAML, which overrides launcher defaults. See
[pi05_recap.yaml.example](../configs/posttrain/pi05_recap.yaml.example) for all
common controls. Set `WCM_TRAIN_GPUS=0,1,2,3` to launch one official WCM
DDP process per listed GPU; its world size is derived from the list, so there
is no separate process-count setting. Advantage-label inference uses the same
GPU list and shards windows without padding; both WCM batch-size settings are
per GPU. `TRAIN_GPUS` exposes all listed devices
to OpenPI. `OPENPI_FSDP_DEVICES` controls the FSDP axis and must divide the
number of training GPUs; the remaining mesh axis is data parallel. For
example, 8 training GPUs with `OPENPI_FSDP_DEVICES=2` use 4-way data parallel
and 2-way FSDP, so all 8 GPUs participate.

WCM, RECAP, RL Token RECAP, and WCM-selected Pi0.5 launchers reserve their
training GPUs during long CPU-only dataset construction phases. The holder is
released at the model-allocation boundary, including inside the official WCM
and RL Token trainers, and is also cleaned up on errors or interrupts. It
leaves 2048 MiB per GPU available for CUDA/NCCL initialization by default.
Set `GPU_RESERVATION_FREE_MIB` to change that margin or
`GPU_RESERVATION_ENABLED=0` to disable reservation.

A rollout episode itself is a sequential simulator-policy interaction and is
not split across GPUs. `POLICY_GPU` and `ENV_GPU` do place the XPolicyLab
policy server and Isaac Sim on separate cards (by default the first two
`TRAIN_GPUS` entries when available). During RECAP collection, simulator and
policy-server output is written to the iteration's `rollouts/rollout.log`, and
the terminal keeps one aggregate episode progress bar. A rollout-only
collection can also be launched with `robodojo.sh eval --rollout-dir PATH`;
this does not require LeRobot inside the Isaac environment.

The final path is written to `outputs/recap/<task>/latest_policy.txt`. It is a
normal OpenPI checkpoint with RoboDojo model metadata, so it can be scored via:

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env openpi \
  --task stack_bowls \
  --ckpt "$(cat outputs/recap/stack_bowls/latest_policy.txt)" \
  --env-cfg arx_x5 --action-type joint --eval-num 1
```

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
