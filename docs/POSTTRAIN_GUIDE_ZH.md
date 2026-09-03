# RoboDojo posttrain 使用指南

本文介绍 `scripts/posttrain/` 和 `configs/` 下与 WCM、Pi0.5、G05、RL Token、RECAP 相关的脚本和配置。目标是让刚接手项目的工程师能够：

- 选择正确的训练入口；
- 从配置文件启动本地或远程训练；
- 理解数据在 WCM 与策略训练之间的转换；
- 找到 checkpoint、rollout、评估和报告；
- 在中断后安全恢复，而不是重复或覆盖已有实验。

本文只描述当前仓库中的实现。运行前请先确认工作目录是 RoboDojo 根目录：

```bash
cd /path/to/RoboDojo
```

## 1. 先看结论

| 目标 | 主入口 | 配置方式 | 主要输出 |
| --- | --- | --- | --- |
| 单独训练/评估 WCM | [`run_wcm.sh`](../scripts/posttrain/run_wcm.sh) | 环境变量 + [`configs/wcm/robodojo_pi05.yaml`](../configs/wcm/robodojo_pi05.yaml) | `outputs/wcm/robodojo_pi05/` |
| 用 WCM 筛选数据后微调 Pi0.5 | [`finetune_pi05_with_wcm.sh`](../scripts/posttrain/finetune_pi05_with_wcm.sh) | 环境变量 | `XPolicyLab/policy/Pi_05/checkpoints/` |
| 单轮 Pi0.5 WCM actor / RL Token | [`run_pi05_rltoken.sh`](../scripts/posttrain/run_pi05_rltoken.sh) | CLI + 环境变量 | `--output` 指定的 `.pt` |
| 多轮 Pi0.5 RL Token off-policy 训练 | [`run_pi05_rltoken_recap.sh`](../scripts/posttrain/run_pi05_rltoken_recap.sh) | flat YAML 或 `.env` | `outputs/rltoken_recap/<task>/` |
| 多轮 Pi0.5 / G05 RECAP | [`run_recap.sh`](../scripts/posttrain/run_recap.sh) | schema v2 nested YAML | `<output_root>/<task>/` |
| 远程 rollout 或远程训练 | [`remote_training.sh`](../remote_training.sh)、RECAP 配置 | `rollout.remote` / `training.remote` | 本地 run 目录 + 远端临时 job 目录 |

推荐的完整闭环是 `run_recap.sh`：它会收集 rollout、建立 replay buffer、训练 WCM、生成 RECAP advantage、准备策略数据、训练策略、评估 checkpoint，并选择下一轮继续使用的 checkpoint。

## 2. 三条训练路径

### 2.1 独立 WCM 训练或评估

安装 WCM 环境：

```bash
bash scripts/posttrain/install_wcm.sh
```

使用默认配置训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/posttrain/run_wcm.sh
```

常用覆盖项：

```bash
WCM_CONFIG=configs/wcm/robodojo_pi05.yaml \
WCM_DATASET_ROOT=data/RoboDojo_lerobot_v21_video \
WCM_OUTPUT_DIR=outputs/wcm/robodojo_pi05 \
WCM_EPOCHS=5 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/posttrain/run_wcm.sh
```

默认使用 WCM 的 `train` 模式。评估已有 checkpoint：

```bash
MODE=eval \
WCM_CHECKPOINT=outputs/wcm/robodojo_pi05/deploy.pt \
WCM_EVAL_OUTPUT_DIR=outputs/wcm/robodojo_pi05/eval \
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/posttrain/run_wcm.sh
```

如果数据包含多个任务，可以通过 `TASK_NAME` 或 `--task` 只读取一个任务：

```bash
TASK_NAME=stack_bowls \
  bash scripts/posttrain/run_wcm.sh
```

WCM 对混合的成功/失败 rollout 需要知道每个 episode 是否成功。可通过以下文件提供标签：

```bash
WCM_SUCCESS_LABELS=/path/to/episode_labels.json \
  bash scripts/posttrain/run_wcm.sh
```

该 JSON 是 `episode_index -> bool` 映射。只有纯 expert 数据时，脚本默认将其视为成功数据；RECAP 内部会显式设置 `WCM_ASSUME_SUCCESS=0` 并使用 replay buffer 生成的标签。

### 2.2 WCM 筛选后直接微调 Pi0.5

该路径依次执行：WCM episode 打分/筛选 → 转换为策略数据 → 调用 OpenPI Pi0.5 微调器。

```bash
TASK_NAME=stack_bowls \
WCM_DATASET_ROOT=data/RoboDojo_lerobot_v21_video \
WCM_CHECKPOINT=outputs/wcm/robodojo_pi05/stack_bowls/deploy.pt \
PI05_FINETUNE_MODE=action_expert \
  bash scripts/posttrain/finetune_pi05_with_wcm.sh
```

也可以用参数传入任务和微调模式：

```bash
bash scripts/posttrain/finetune_pi05_with_wcm.sh \
  --task stack_bowls \
  --finetune-mode action_expert_lora \
  --wcm-checkpoint outputs/wcm/robodojo_pi05/stack_bowls/deploy.pt
```

`PI05_FINETUNE_MODE` 支持 `full`、`action_expert`、`action_expert_lora`、`paligemma_lora`、`all_lora`。单任务微调通常从 `action_expert` 或 `action_expert_lora` 开始。训练参数主要由 `OPENPI_*` 环境变量控制，例如 `OPENPI_BATCH_SIZE`、`OPENPI_NUM_TRAIN_STEPS`、`OPENPI_LEARNING_RATE`、`OPENPI_FSDP_DEVICES`、`OPENPI_WANDB_ENABLED=0`。

### 2.3 多轮 RECAP

`run_recap.sh` 支持 `policy.name: pi05` 和 `policy.name: g05`。每轮的大致数据流如下：

```text
初始 demonstrations + 当前策略 rollout
             │
             ▼
       replay_buffer
             │
             ├── wcm_training_buffer ──> WCM 更新 ──> deploy.pt
             │                                      │
             │                                      ▼
             └────────────────────────> N-step advantage labels
                                                    │
                                                    ▼
                                   advantage-conditioned policy dataset
                                                    │
                              ┌─────────────────────┴─────────────────────┐
                              ▼                                           ▼
                         Pi0.5/OpenPI                              G05/GalaxeaVLA
                              │                                           │
                              └──────── checkpoint 评估、选择、下一轮 ────────┘
```

启动 Pi0.5 RECAP：

```bash
bash scripts/posttrain/run_recap.sh \
  --config configs/posttrain/pi05_recap.yaml.example
```

启动 G05 RECAP：

```bash
bash scripts/posttrain/run_recap.sh \
  --config configs/posttrain/g05_recap.yaml.example
```

模板中的绝对路径、初始 checkpoint、任务名和远程主机信息必须先替换。启动前建议只做配置解析：

```bash
python3 scripts/posttrain/recap_config.py \
  configs/posttrain/pi05_recap.yaml.example \
  --format yaml \
  --output /tmp/pi05_recap_resolved.yaml
```

RECAP 的 schema v2 会拒绝未知字段，并检查任务、GPU 数量、FSDP、训练步数、rollout 下限、数据格式和远程路径之间的组合关系。

## 3. 配置文件如何工作

### 3.1 RECAP schema v2

`run_recap.sh` 的唯一主配置是嵌套 YAML，必须包含 `schema_version: 2`。[`recap_config.py`](../scripts/posttrain/recap_config.py) 将嵌套字段解析为环境变量；解析后的完整配置会保存为：

```text
<output_root>/<task_slug>/resolved_config.yaml
```

字段优先级与普通 shell launcher 不同：对 `run_recap.sh` 而言，YAML 是事实来源，`--resume` 可以在命令行显式打开恢复模式。不要只在 shell 中设置一个同名变量后期待它覆盖 YAML。

主要 section：

| Section | 作用 | 接手项目时首先检查 |
| --- | --- | --- |
| `policy` | 选择 `pi05` 或 `g05` | G05 当前要求 joint action、v3.0/HDF5、FM |
| `run` | 任务、输出根目录、迭代次数、恢复策略 | `task`、`output_root`、`resume` |
| `environment` | 机器人、动作格式、策略和 Isaac 环境 | `env_cfg`、`action_type`、两个环境是否属于正确主机 |
| `checkpoints` | 初始策略和可选初始 WCM | checkpoint 是否包含所需 sidecar |
| `data` | demonstrations 来源和格式 | `v2.1`、`v3.0`、`hdf5` 必须与实际数据一致 |
| `devices` | 本地训练、WCM、rollout 和显存保留 GPU | GPU 不重复、不与别的任务冲突 |
| `rollout` | 每轮采集数量、步数、质量门槛和远程 rollout | `minimum` 不要超过 `episodes` |
| `training.remote` | WCM/策略训练是否放到另一台机器 | `enabled`、远端 checkout、Python 和 GPU |
| `wcm` | WCM 训练、推理和 advantage 参数 | `config`、epoch、batch、precision |
| `recap` | gamma、失败惩罚、正例比例和采样权重 | 正负样本是否足够、权重必须为正 |
| `pi05` / `g05` | 对应策略的训练器参数 | steps、warmup、学习率、显存选项 |
| `evaluation` | 中间 checkpoint 的评估和选择 | `interval < steps` |
| `value_video` | WCM value overlay 视频 | 视频数量、设备、解码 backend |
| `runtime` | 本地数据、策略、WCM Python 路径 | `runtime.data_python` 必须能导入数据依赖 |

重要约束：

- `data.format` 是严格选择，不会在 v2.1、v3.0、HDF5 之间静默 fallback。
- HDF5 源通常配置为 `demo_root: data/RoboDojo`，脚本会按任务查找 `episode_*.hdf5` 或 `episode_*.h5`，并读取 state、action 和三路相机数据。
- Pi0.5 的 `pi05.batch_size` 必须能被有效 GPU 数量整除，`fsdp_devices` 必须能整除有效 Pi0.5 GPU 数量。
- G05 当前只支持 `environment.action_type: joint`、`g05.action_source: fm`，并要求 `recap.guidance_scale: 1.0`。
- `run.reuse_completed_artifacts: true` 必须同时设置 `run.resume: true`。它会信任结构完整的旧 stage，即使 fingerprint 与当前配置不同；只有确认旧产物确实可以复用时才打开。
- 同一个 `<output_root>/<task_slug>` 默认不能重复创建。需要继续已有 run 时使用 `run.resume: true` 或 `--resume`。

### 3.2 RL Token 的 flat 配置

`run_pi05_rltoken_recap.sh` 使用另一套较早的 flat 配置，不要把它和 schema v2 混用：

- [`pi05_rltoken_recap.yaml.example`](../configs/posttrain/pi05_rltoken_recap.yaml.example) 是 `KEY: value` 形式的 YAML；
- [`pi05_rltoken_recap.env.example`](../configs/posttrain/pi05_rltoken_recap.env.example) 是可 `source` 的 `KEY=value` 环境变量模板；
- 该 launcher 的优先级是：命令行参数 > 已有环境变量 > flat YAML > launcher 默认值。

YAML 用法：

```bash
bash scripts/posttrain/run_pi05_rltoken_recap.sh \
  --config configs/posttrain/pi05_rltoken_recap.yaml.example
```

`.env` 用法：

```bash
source configs/posttrain/pi05_rltoken_recap.env.example
bash scripts/posttrain/run_pi05_rltoken_recap.sh
```

RL Token rollout GPU 按顺序配对：`0,1,2,3` 表示 `(policy=0, Isaac=1)` 和 `(policy=2, Isaac=3)`。奇数张卡的最后一张会被舍弃；单卡时同一张卡同时运行策略和 Isaac。`RLTOKEN_ROLLOUT_ENVS_PER_WORKER` 是每个 worker 的向量化环境数量。

## 4. 启用远程 rollout 或远程训练

这里有两个相互独立的开关，最容易混淆：

| 配置 | 远程内容 | 本地仍负责 |
| --- | --- | --- |
| `rollout.remote.enabled` | 策略推理 + Isaac Sim rollout/evaluation | RECAP 状态机、数据处理、训练（除非另开 `training.remote`） |
| `training.remote.enabled` | WCM、advantage inference、Pi0.5/G05 training、可选 value-video | 本地配置解析、状态机、上传下载和最终输出 |

### 4.1 只启用远程 rollout

适合策略训练在当前机器、Isaac Sim 在另一台有 GPU 的机器：

```yaml
rollout:
  remote:
    enabled: true
    host: SIM_HOST
    repo_root: /abs/path/to/RoboDojo
    work_root: /abs/path/to/recap_rollout_jobs
    zstd: /usr/bin/zstd
    conda: /abs/path/to/conda
    python: /abs/path/to/robodojo-python
    policy_env: /abs/path/to/policy/runtime
    eval_env: /abs/path/to/robodojo-conda-env
    policy_gpu: 0
    environment_gpu: 1
    value_video_gpu: 0
    policy_evaluation: true
```

G05 远程 rollout 还需要：

```yaml
    g05_root: /abs/path/to/GalaxeaVLA
    g05_processor_path: /abs/path/to/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor
```

远端必须有自己的 RoboDojo checkout、策略运行环境、Isaac/RoboDojo 环境和 G05 processor。checkpoint 会被打包为 zstd archive 传过去，rollout 完成后再传回本地 run 目录。

### 4.2 启用远程 WCM 和策略训练

在同一个 schema v2 配置中增加或修改：

```yaml
training:
  remote:
    enabled: true
    host: TRAIN_HOST
    repo_root: /abs/path/to/RoboDojo
    work_root: /abs/path/to/recap_training_jobs
    zstd: /usr/bin/zstd
    conda: /abs/path/to/conda
    python: /abs/path/to/python-with-yaml
    policy_python: /abs/path/to/policy-python
    wcm_python: /abs/path/to/RoboDojo/external_dependencies/WCM/.venv/bin/python
    policy_gpus: [0, 1, 2, 3]
    wcm_gpus: [0, 1, 2, 3]
    value_video_gpu: 0
    render_value_video: true
```

Pi0.5 远程训练时，`policy_python` 应指向远端 OpenPI Python；G05 远程训练时，`policy_python` 应指向远端 GalaxeaVLA `.venv/bin/python`。G05 本地不训练时，本地 `runtime.policy_python` 可以为 `null`，但 `runtime.data_python` 仍要能运行数据物化脚本。

远程训练前需要满足：

1. 本机到 `host` 的 SSH 免密登录可用；
2. 远端 `repo_root` 是绝对路径，且包含同步后的 RoboDojo；
3. 远端非 login shell 能找到 `zstd`、`conda` 和带 PyYAML 的 bootstrap Python；
4. 远端存在 WCM `.venv`；
5. 策略 Python、G05 source、processor、Isaac 环境和指定 GPU 可用；
6. GNU tar 支持 `--zstd`，并有 `setsid --wait`。

RECAP 启动时会自动执行远程 preflight。也可以直接使用远程入口做检查：

```bash
python3 scripts/posttrain/remote_training.py preflight \
  --host TRAIN_HOST \
  --remote-repo-root /abs/path/to/RoboDojo \
  --remote-work-root /abs/path/to/recap_training_jobs \
  --policy g05 \
  --remote-policy-python /abs/path/to/GalaxeaVLA/.venv/bin/python \
  --remote-wcm-python /abs/path/to/RoboDojo/external_dependencies/WCM/.venv/bin/python \
  --g05-root /abs/path/to/GalaxeaVLA \
  --gpu 0 --gpu 1 --gpu 2 --gpu 3
```

仓库根目录的 [`remote_training.sh`](../remote_training.sh) 是 `run_recap.sh` 的薄封装，默认读取 `configs/posttrain/remote_training.yaml`，也可通过 `RECAP_CONFIG` 指定配置：

```bash
RECAP_CONFIG=configs/posttrain/g05_recap.yaml.example \
  bash remote_training.sh
```

注意：当前 `remote_training.yaml` 的 `rollout.remote.enabled` 为 `true`，但 `training.remote.enabled` 仍为 `false`；它表示“远程 rollout + 本地训练”的现成配置。如果需要训练也放到远端，必须明确改为 `training.remote.enabled: true` 并补齐远端路径。

### 4.3 远程 job 和清理

远程训练和 rollout 使用 `<remote_work_root>/jobs/<job_id>/`。每个 job 会保存输入 archive、控制文件、日志、GPU reservation 和结果 archive。成功传输后本地 stage 才会被标记完成；收到 `INT`/`TERM` 或 SSH launcher 消失时，worker 会尝试终止进程组并释放远端 GPU。

远端训练脚本会把本地跟踪的 WCM/G05 adapter 临时同步到远端 checkout，训练输出完成后压缩传回本地。不要手工删除正在运行的 job 目录；需要取消时让主进程退出，或显式使用 `remote_recap.py cancel`。

## 5. 输出目录和 checkpoint

### 5.1 独立入口输出

| 入口 | 输出规则 |
| --- | --- |
| `run_wcm.sh` | 默认 `outputs/wcm/robodojo_pi05/`；设置 `TASK_NAME` 且未显式设置 `WCM_OUTPUT_DIR` 时追加 `<task_slug>/`；训练 checkpoint 通常是 `deploy.pt`，中间恢复点在 `checkpoints/`；评估输出在 `eval/` |
| `finetune_pi05_with_wcm.sh` | 数据集位于 OpenPI 的 `HF_LEROBOT_HOME/<repo-id>`；策略 checkpoint 位于 `XPolicyLab/policy/Pi_05/checkpoints/<experiment>/` |
| `run_pi05_rltoken.sh` | 完全由 `--output` 指定，通常是单个 actor `.pt`；可另传 encoder/BC checkpoint 路径 |
| `train_g05.py` | 完全由 `--output` 指定；输出是包含 checkpoint、`.hydra/config.yaml`、`dataset_stats.json`、`action_tokenizer.pt` 的 G05 bundle |

### 5.2 RECAP 输出树

对于 `run_recap.sh`，任务名会转为小写 slug，最终根目录是 `<output_root>/<task_slug>`：

```text
<output_root>/<task_slug>/
├── resolved_config.yaml
├── fixed_norm_stats/
│   ├── norm_stats.json
│   ├── fixed_norm_stats.json             # Pi0.5 固定资产的 manifest
│   ├── dataset_stats.json                # G05
│   └── action_tokenizer.pt               # G05
├── lerobot/
│   └── RoboDojo-recap-<task>-iter-<N>/   # 策略训练用增量数据集
├── iteration_01/
│   ├── rollouts/
│   │   ├── episodes/<episode>/
│   │   │   ├── manifest.json
│   │   │   ├── trajectory.npz
│   │   │   └── cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4
│   │   ├── quality.json
│   │   └── rollout*.log
│   ├── replay_buffer/
│   ├── wcm_training_buffer/
│   ├── wcm/
│   │   ├── deploy.pt
│   │   └── checkpoints/
│   ├── recap_advantages.jsonl
│   ├── pi05/ 或 g05/
│   │   ├── checkpoints/
│   │   └── robodojo_*_model.json
│   ├── policy_evaluations/
│   │   ├── baseline/evaluation.json
│   │   └── step_<step>/evaluation.json
│   ├── selection.json
│   └── value_videos/
│       ├── videos/episode-*.mp4
│       ├── summary.json
│       └── episode_curves.json
├── latest_policy.txt
├── latest_wcm.txt
├── report.json
├── report.md
├── best_checkpoint.json / .md / .txt
└── incomplete_cleanup.json              # 有不完整产物被清理时生成
```

其中：

- `replay_buffer` 是给 WCM 和 RECAP 使用的内部 RoboDojo LeRobot-v2.1 布局，包含 demo、成功/失败标签、provenance 和视频引用；
- `wcm_training_buffer` 是当前轮实际用于 WCM 更新的子集，后续轮次会保留一定数量旧 episode，再加入新 episode；
- `recap_advantages.jsonl` 第一行是 schema/header，后续是每个 episode 的 frame-level advantage 与正负条件；
- `pi05/` 的 checkpoint 通常是数字目录（如 `0`、`1000`），每个完整 checkpoint 应有 `params/`、`assets/` 和 `_CHECKPOINT_METADATA`；
- `g05/` 的 checkpoint 通常是 `checkpoints/step_*.pt`，并带 `dataset_stats.json`、`action_tokenizer.pt`、`.hydra/config.yaml`；
- `selection.json` 同时记录 baseline、候选 checkpoint、评估指标和 continuation checkpoint；训练会继续使用每轮最后一个 checkpoint，不会因为评估排名最好就跳转到更早的 checkpoint；
- `report.md` 适合人工查看，`report.json` 适合 agent 或脚本读取；`best_checkpoint.txt` 可直接被 shell 命令读取。

恢复时，完整 stage 会检查结构和 fingerprint；不完整的非 rollout 产物会移动为可恢复的 `.incomplete-<timestamp>` 目录。不要把这些目录误认为新的正式实验输出。

### 5.3 RL Token RECAP 输出树

```text
outputs/rltoken_recap/<task_slug>/
├── iteration_01/
│   ├── replay_buffer/
│   ├── wcm_training_buffer/
│   ├── wcm/deploy.pt
│   ├── encoder.pt
│   ├── bc_actor.pt
│   ├── actor.pt
│   └── rollouts/
│       ├── episodes/
│       └── logs/worker_*.log
├── latest_actor.txt
└── latest_wcm.txt
```

部署 RL Token actor 时，使用 `latest_actor.txt` 指向的 checkpoint，并设置 XPolicyLab 的 `POSTTRAIN_MODE` 与 `POSTTRAIN_CHECKPOINT`。正常 XPolicyLab evaluation 默认仍是确定性的；采集 rollout 时可用 `RLTOKEN_ROLLOUT_DETERMINISTIC=1` 关闭 actor 采样。

## 6. 配置文件目录索引

### 6.1 `configs/posttrain/`

以下文件是当前目录中的全部 posttrain 配置。带 `.example` 的文件是模板；带具体主机名、绝对路径或任务名的文件是历史/运行样例，使用前应检查路径和 `resume` 状态。

| 文件 | 类型 | 简要说明 |
| --- | --- | --- |
| [`pi05_recap.yaml.example`](../configs/posttrain/pi05_recap.yaml.example) | schema v2 | Pi0.5 RECAP 模板；v2.1 video demonstrations，本地 rollout、本地 WCM/策略训练默认关闭远程。 |
| [`g05_recap.yaml.example`](../configs/posttrain/g05_recap.yaml.example) | schema v2 | G05 RECAP 完整模板；v3.0 数据，示范了远程 rollout/训练字段，需替换所有 host/path。 |
| [`pi05_rltoken_recap.yaml.example`](../configs/posttrain/pi05_rltoken_recap.yaml.example) | flat YAML | RL Token off-policy 迭代训练模板；字段名直接对应环境变量，不经过 schema v2 校验。 |
| [`pi05_rltoken_recap.env.example`](../configs/posttrain/pi05_rltoken_recap.env.example) | shell env | RL Token flat 配置的 `KEY=value` 版本，可 `source` 后启动。 |
| [`g05_remote.yaml`](../configs/posttrain/g05_remote.yaml) | schema v2 | `general_pickup` 的 G05 v3.0 样例：远程 rollout，本地 G05/WCM 训练，启用 resume/reuse。 |
| [`g05_6226.yaml`](../configs/posttrain/g05_6226.yaml) | schema v2 | `general_pickup` 的另一套 GPU/路径布局；远程 rollout，本地训练，适合已有 run 的恢复样例。 |
| [`remote_training.yaml`](../configs/posttrain/remote_training.yaml) | schema v2 | 综合远程环境样例；当前实际是远程 rollout + 本地训练，若要远程训练需改 `training.remote.enabled`。 |
| [`remote_6226.yaml`](../configs/posttrain/remote_6226.yaml) | schema v2 | Pi0.5 `put_bottles_into_dustbin` 的恢复样例；v2.1 数据、远程 rollout、本地训练。 |
| [`g05_hdf5_remote.yaml`](../configs/posttrain/g05_hdf5_remote.yaml) | schema v2 | G05 HDF5 demonstrations + 远程 rollout 样例；任务为 `pour_by_language`。 |
| [`g05_6226_hdf5.yaml`](../configs/posttrain/g05_6226_hdf5.yaml) | schema v2 | G05 HDF5 `align_blocks` 样例；与上一个 HDF5 配置的任务、GPU 和训练迭代设置不同。 |
| [`ab2.yaml`](../configs/posttrain/ab2.yaml) | schema v2 | G05 HDF5 `align_block` 实验样例；3 轮、较大的 WCM replay/batch 配置，使用前重点确认路径。 |

### 6.2 `configs/g05/`

| 文件 | 简要说明 |
| --- | --- |
| [`data/robodojo_recap.yaml`](../configs/g05/data/robodojo_recap.yaml) | G05 Hydra 数据配置；声明 v3.0 LeRobot、三路相机、20 维 action/state 结构、transforms 和通过 `ROBODOJO_RECAP_DATASET` 注入的数据目录。 |
| [`task/robodojo_recap.yaml`](../configs/g05/task/robodojo_recap.yaml) | G05 Hydra task overlay；选择 G05 model、actioncodec 和上面的 data 配置，并从环境变量读取 `ROBODOJO_G05_DATA_STATS`。 |

这两个文件由 [`train_g05.py`](../scripts/posttrain/train_g05.py) 通过 `--task-config robodojo_recap` 使用；不要把它们当成独立的 RoboDojo launcher。G05 上游训练仍由 GalaxeaVLA 的 `scripts/finetune.py` 执行，RoboDojo 只负责 overlay、数据适配、source balancing、checkpoint sidecar 和运行时保护。

### 6.3 `configs/wcm/`

| 文件 | 简要说明 |
| --- | --- |
| [`robodojo_pi05.yaml`](../configs/wcm/robodojo_pi05.yaml) | WCM 模型/数据/优化器默认配置；`run_wcm.sh` 会通过环境变量替换数据根目录、输出目录、task、epoch、batch、precision、失败惩罚和 gamma。 |

该配置的 `data.root` 默认是 `data/RoboDojo_lerobot_v21_video`。RECAP 每轮会把它替换为当前 `replay_buffer` 或 `wcm_training_buffer`，所以不能只看 YAML 中的默认 root 判断实际训练数据。

## 7. `scripts/posttrain/` 文件索引

下面列出当前目录中的全部 40 个 Python 文件和 9 个 shell 文件。表中的“被谁调用”是定位代码的建议，不代表只能从该入口调用。

### 7.1 Shell 文件

| 文件 | 功能与用法 |
| --- | --- |
| [`install_wcm.sh`](../scripts/posttrain/install_wcm.sh) | 检查 vendored WCM 和 `uv`，在 `external_dependencies/WCM/.venv` 创建/同步 WCM 环境；先于 WCM/RECAP 运行。 |
| [`run_wcm.sh`](../scripts/posttrain/run_wcm.sh) | WCM train/eval wrapper；处理 GPU、DDP、task filter、环境变量覆盖和输出路径。支持 `MODE=train|eval|all`、`--task`。 |
| [`run_recap.sh`](../scripts/posttrain/run_recap.sh) | schema v2 的 Pi0.5/G05 多轮 RECAP 主状态机；串起 rollout、buffer、WCM、advantage、策略训练、评估、selection 和报告。 |
| [`run_pi05_rltoken.sh`](../scripts/posttrain/run_pi05_rltoken.sh) | 单轮 RL Token/WCM actor 的 DDP launcher；按 `ACTOR_TRAIN_GPUS` 启动一进程一卡。 |
| [`run_pi05_rltoken_recap.sh`](../scripts/posttrain/run_pi05_rltoken_recap.sh) | flat 配置驱动的 RL Token 多轮 loop；每轮更新 WCM、actor，再并行收集 policy/Isaac rollout。 |
| [`finetune_pi05_with_wcm.sh`](../scripts/posttrain/finetune_pi05_with_wcm.sh) | WCM episode 选择、策略数据转换和 OpenPI Pi0.5 直接微调的一键 wrapper。 |
| [`posttrain_config.sh`](../scripts/posttrain/posttrain_config.sh) | 公共 shell 配置函数：提取 `--config`、加载 flat YAML、加载 schema v2 RECAP YAML、写出 resolved YAML。 |
| [`gpu_reservation.sh`](../scripts/posttrain/gpu_reservation.sh) | 公共显存保留生命周期函数；在 CPU 准备、远程等待和数据处理期间占住闲置 GPU，退出时释放。 |
| [`remote_recap_worker.sh`](../scripts/posttrain/remote_recap_worker.sh) | 被 `remote_recap.py`/`remote_training.py` 安装到远端执行的 worker；处理 reservation、rollout、通用 stage、value-video、归档和取消。通常不直接手工运行。 |

常用 shell 入口的最小命令：

```bash
bash scripts/posttrain/run_recap.sh --help
bash scripts/posttrain/run_pi05_rltoken_recap.sh --help
bash scripts/posttrain/install_wcm.sh
```

### 7.2 数据读取、转换与基础设施 Python 文件

| 文件 | 简要总结 |
| --- | --- |
| [`lerobot_io.py`](../scripts/posttrain/lerobot_io.py) | 严格读取 LeRobot v2.1/v3.0 的 metadata、episode、task、parquet 和 video 路径；拒绝格式不匹配。 |
| [`hdf5_io.py`](../scripts/posttrain/hdf5_io.py) | 流式读取 RoboDojo HDF5 demonstration；组装 state/action、解析 instruction/fps，并把内嵌相机流物化为视频。 |
| [`robodojo_dataset.py`](../scripts/posttrain/robodojo_dataset.py) | WCM 使用的 RoboDojo 内部 v2.1 dataset adapter；过滤 task、读取 success labels、计算 returns，并提供 video cache。 |
| [`prepare_policy_dataset.py`](../scripts/posttrain/prepare_policy_dataset.py) | 将内部 v2.1 video 数据转换为策略使用的 LeRobot 数据集；可按 task/episode labels 筛选，并写入三路相机。 |
| [`prepare_recap_dataset.py`](../scripts/posttrain/prepare_recap_dataset.py) | 增量物化 advantage-conditioned 策略数据；复用旧视频、追加新 episode、重写 task 条件，供 Pi0.5/G05 使用。 |
| [`build_replay_buffer.py`](../scripts/posttrain/build_replay_buffer.py) | 首轮合并 task-filtered demonstrations 与 rollout，生成内部 replay buffer；尽量 hard-link 已有视频。 |
| [`build_replay_buffer_incremental.py`](../scripts/posttrain/build_replay_buffer_incremental.py) | 在已有 replay buffer 上追加一轮 rollout，保留旧 episode 和 metadata。 |
| [`build_wcm_training_subset.py`](../scripts/posttrain/build_wcm_training_subset.py) | 从旧 replay episode 中抽样，再加上全部新 episode，构造当前轮 WCM 训练子集。 |
| [`g05_source_sampling.py`](../scripts/posttrain/g05_source_sampling.py) | 根据 manifest 区分 demonstration/rollout，构造虚拟 frame index，让 G05 按 source 权重采样而不复制视频。 |
| [`progress.py`](../scripts/posttrain/progress.py) | 统一 WCM、数据转换和并行 rollout 的 tqdm 输出，避免多进程重复进度条。 |
| [`monitor_rollout_progress.py`](../scripts/posttrain/monitor_rollout_progress.py) | 监视一个或多个 rollout worker，显示 aggregate episode 进度。 |

### 7.3 WCM、RECAP label 与诊断 Python 文件

| 文件 | 简要总结 |
| --- | --- |
| [`run_wcm.py`](../scripts/posttrain/run_wcm.py) | 加载官方 WCM trainer/evaluator，并替换 RoboDojo dataset adapter；处理 DDP、旧 ViT key、初始 checkpoint、resume 和 optimizer override。 |
| [`wcm_checkpoint.py`](../scripts/posttrain/wcm_checkpoint.py) | 在 Transformers ViT key rename 之间做严格的 WCM state-dict 兼容转换。 |
| [`select_wcm_episodes.py`](../scripts/posttrain/select_wcm_episodes.py) | 用 WCM 为 demonstration/episode 打分，输出 `episode_index -> bool` 标签，供直接 Pi0.5 微调使用。 |
| [`annotate_recap_advantages.py`](../scripts/posttrain/annotate_recap_advantages.py) | 用 WCM 计算 N-step advantage，并输出 frame-level positive 条件；支持单卡或 torchrun 多卡。 |
| [`recap_advantage_metadata.py`](../scripts/posttrain/recap_advantage_metadata.py) | 只更新 advantage 文件的 metadata 统计；恢复旧 run 时补齐，不重新做 GPU inference。 |
| [`render_rollout_value_videos.py`](../scripts/posttrain/render_rollout_value_videos.py) | 对 rollout frame 做 WCM value inference，生成带 value 曲线/颜色 overlay 的视频和汇总。 |
| [`value_video_metadata.py`](../scripts/posttrain/value_video_metadata.py) | 从 rollout manifest 回填 value-video instruction，并验证 summary/curve 的 episode 对齐。 |
| [`check_recap_rollouts.py`](../scripts/posttrain/check_recap_rollouts.py) | 汇总 rollout 数量、成功率、失败率，执行 RECAP 的 episode/质量门槛。 |
| [`policy_evaluation.py`](../scripts/posttrain/policy_evaluation.py) | 记录 policy evaluation JSON；可复用已有 rollout episode，或把远程 evaluation 结果登记到本地。 |
| [`select_recap_policy.py`](../scripts/posttrain/select_recap_policy.py) | 比较 baseline 和中间 checkpoint 的 evaluation，记录 best evaluated，同时始终返回最后 checkpoint 作为下一轮 continuation。 |
| [`write_recap_report.py`](../scripts/posttrain/write_recap_report.py) | 汇总所有 iteration 的 rollout、evaluation、selection、value video，生成 `report.json`、`report.md` 和 best checkpoint 文件。 |
| [`recap_artifacts.py`](../scripts/posttrain/recap_artifacts.py) | 检查各 stage 的结构、数量和 fingerprint；写 `.recap_stage.json`，可 recoverably archive 不完整产物并 finalize iteration。 |

### 7.4 策略训练与 checkpoint Python 文件

| 文件 | 简要总结 |
| --- | --- |
| [`train_pi05.py`](../scripts/posttrain/train_pi05.py) | 从 OpenPI 动态加载训练实现，按 `full`/action expert/LoRA 模式构造模型和 freeze filter；支持 RECAP conditioning、FSDP、norm stats 和 checkpoint metadata。 |
| [`train_pi05_rltoken.py`](../scripts/posttrain/train_pi05_rltoken.py) | 从 WCM replay buffer 训练 reference-conditioned Pi0.5 actor；实现 actor window、BC 初始化、WCM actor/RL Token objective、resume 和 checkpoint 保存。 |
| [`train_g05.py`](../scripts/posttrain/train_g05.py) | 将 RoboDojo 参数转换成 G05 Hydra override，调用 upstream GalaxeaVLA finetune，并把 stats/tokenizer/RECAP metadata 复制进 G05 bundle。 |
| [`g05_finetune_entry.py`](../scripts/posttrain/g05_finetune_entry.py) | G05 upstream finetune 的 guarded entrypoint；安装 source-balanced dataset 和有限 getitem retry，再运行上游脚本。 |
| [`prepare_fixed_pi05_norm_stats.py`](../scripts/posttrain/prepare_fixed_pi05_norm_stats.py) | 从初始 Pi0.5 checkpoint 复制唯一 norm stats，并写 hash manifest；整个 RECAP run 固定这份坐标系。 |
| [`compute_pi05_norm_stats.py`](../scripts/posttrain/compute_pi05_norm_stats.py) | 针对 RECAP buffer 计算 Pi0.5 robot-specific normalization 统计。 |
| [`compute_pi05_norm_stats_incremental.py`](../scripts/posttrain/compute_pi05_norm_stats_incremental.py) | 在已有 running accumulator 上增量更新 Pi0.5 normalization；当前主 RECAP 路径固定初始 stats 时通常不直接调用。 |
| [`prepare_g05_assets.py`](../scripts/posttrain/prepare_g05_assets.py) | 从初始 G05 bundle 固定 `dataset_stats.json` 和 `action_tokenizer.pt`，同时生成 RECAP 需要的 `norm_stats.json`。 |
| [`prepare_g05_inference_checkpoint.py`](../scripts/posttrain/prepare_g05_inference_checkpoint.py) | 清理/重写复制后的 G05 checkpoint 配置，使其能在没有原始训练数据路径的远程 rollout 主机上推理。 |
| [`g05_remote.py`](../scripts/posttrain/g05_remote.py) | `remote_training.py g05` 的远程 G05 adapter；同步训练支持文件和 Hydra configs，打包 bundle 并在远端调用 `train_g05.py`。 |
| [`remote_training.py`](../scripts/posttrain/remote_training.py) | 远程 stage 编排器；把目录/文件压缩上传，展开安全 marker，调用远端 WCM、advantage、Pi0.5、G05 或 render，再下载结果。 |
| [`remote_recap.py`](../scripts/posttrain/remote_recap.py) | 远程 rollout/value-video 执行器；支持 SSH 或 `local` 主机、preflight、checkpoint 打包、远端缓存、取消和 GPU reservation。 |
| [`reserve_gpu_memory.py`](../scripts/posttrain/reserve_gpu_memory.py) | 用少量 CUDA allocation 保留闲置 GPU 的显存，防止同机其他任务抢占 RECAP 计划使用的卡。 |

### 7.5 配置解析与辅助 Python 文件

| 文件 | 简要总结 |
| --- | --- |
| [`load_posttrain_config.py`](../scripts/posttrain/load_posttrain_config.py) | 将 flat uppercase YAML 转成 NUL 分隔的 shell `NAME/value` 流；由 `posttrain_config.sh` 读取。 |
| [`recap_config_base.py`](../scripts/posttrain/recap_config_base.py) | Pi0.5 RECAP 字段 registry、类型归一化和通用解析基础；当前 schema v2 的 `recap_config.py` 复用其字段/工具。不要直接把它当作主入口。 |
| [`recap_config.py`](../scripts/posttrain/recap_config.py) | schema v2 的 model-selectable 配置解析器；扩展 Pi0.5 registry 支持 G05，并执行模型/数据/远程配置校验。 |
| [`recap_conditioning.py`](../scripts/posttrain/recap_conditioning.py) | 生成策略训练用的 advantage prompt；Pi0.5 显式使用 positive/negative，G05 对正例执行确定性 unconditional dropout。 |

## 8. 常见问题排查

### 配置解析失败

先运行：

```bash
python3 scripts/posttrain/recap_config.py CONFIG \
  --format yaml --output /tmp/resolved.yaml
```

重点检查 `schema_version: 2`、字段拼写、`policy.name`、`data.format`、GPU 列表是否重复、`warmup_steps < steps`，以及 remote enabled 时的绝对路径。

### 找不到数据

- LeRobot v2.1/v3.0：检查 `data.demo_root/meta/info.json` 以及 `codebase_version`；
- HDF5：检查 `demo_root` 下是否有对应任务的 `episode_*.hdf5`/`.h5`；
- G05：策略训练输入最终必须是 v3.0，HDF5 会先被归一化并转换，不会直接交给 G05；
- WCM 混合数据：检查 `WCM_SUCCESS_LABELS` 或 `meta/success_labels.json`。

### 找不到 checkpoint sidecar

- Pi0.5 初始 checkpoint 必须能解析出 `params/`、`assets/` 和 `norm_stats.json`；
- G05 初始 checkpoint bundle 必须包含 checkpoint、`.hydra/config.yaml`、`dataset_stats.json`、`action_tokenizer.pt`；
- 远程 G05 rollout 还必须有远端 processor 目录和 `tokenizer.json`。

### RECAP 被质量门槛阻止

查看 `<iteration>/rollouts/quality.json`、`rollout*.log` 和 episode manifest。`rollout.minimum.total` 只允许在收集完成数量不少于该值时继续；`minimum.successes`/`failures` 用于避免 WCM 只看到单一标签。

### 想恢复中断的 run

```bash
bash scripts/posttrain/run_recap.sh \
  --config configs/posttrain/my_run.yaml \
  --resume
```

先确认 `<output_root>/<task_slug>/` 仍存在、远程 host 可访问、配置没有误改。普通恢复会检查 fingerprint；只有明确知道修改只是路径搬迁或诊断设置时，才使用 `run.reuse_completed_artifacts: true`。

### 想让 agent 自动判断是否完成

优先读取：

1. `resolved_config.yaml`：实际生效配置；
2. `latest_policy.txt` / `latest_actor.txt` / `latest_wcm.txt`：继续训练或部署的 checkpoint；
3. `report.json`：机器可读的所有 iteration 评估；
4. `selection.json`：某一轮的候选、指标和 continuation；
5. 各 stage 的 `.recap_stage.json`：结构完成标记和 fingerprint。

不要只依据进程退出码判断成功；还要确认 checkpoint、episode manifest、`quality.json`、evaluation JSON 或 value-video summary 实际存在。

## 9. 接手新实验的最小 checklist

```text
[ ] 在 RoboDojo 根目录运行
[ ] install_wcm.sh 已完成，WCM .venv 可执行
[ ] 任务 slug 与 demonstrations 中的 task instruction 唯一匹配
[ ] data.format 与 meta/info.json 或 HDF5 实际格式一致
[ ] 初始 policy checkpoint 可读，G05 sidecar / Pi0.5 norm stats 齐全
[ ] policy / WCM / rollout GPU 列表没有重复或冲突
[ ] 先运行 recap_config.py 做 dry validation
[ ] 远程模式先确认 SSH、conda、Python、zstd、tar、setsid 和 GPU
[ ] 小规模 run 先减少 iterations、rollout episodes 和 train steps
[ ] 运行后保存 resolved_config.yaml、report.json 和 latest_* 文件
[ ] 恢复前区分普通 --resume 与 reuse_completed_artifacts
```
