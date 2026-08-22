# 使用 Codex 从零迁移和恢复 RoboDojo

本文档用于把当前 RoboDojo 工作区（包括本地 Git 历史、XPolicyLab 修改、
Assets、checkpoints、训练数据和结果）迁移到另一台 Linux 机器。官方安装页仍是
环境和硬件要求的上游参考：
<https://robodojo-benchmark.com/doc/usage/install-and-download/>。

## 迁移包包含什么

`scripts/migration/create_bundle.sh` 默认在 `migration_bundle/` 中生成：

- RoboDojo 和全部已初始化 submodule 的 Git bundle；
- 每个仓库的未提交二进制 patch，以及未被 Git 忽略的 untracked 文件；
- `Assets/`；
- `XPolicyLab/policy/Pi_05/checkpoints/`；
- `data/`、`outputs/`、`eval_result/`、`wandb/` 和 `smoke_results/`；
- 环境元数据和覆盖全部文件的 `SHA256SUMS`。

Python 虚拟环境和缓存不会被打包：根目录 `.venv/`、OpenPI `.venv/`、WCM
`.venv/` 和 `.cache/` 都与机器路径、CUDA 和系统库绑定，应在新机器重新安装。

当前数据的未压缩体积约为：Assets 39 GiB、Pi_05 checkpoints 189 GiB、data
60 GiB、outputs 111 GiB、eval_result 28 GiB。请为源机器的归档和目标机器的
解压分别预留足够空间。

## 1. 在源机器创建迁移包

确保没有训练或评测进程仍在写 checkpoints、data、outputs 或 eval_result，避免
得到跨时间点的不一致快照，然后执行：

```bash
cd /share/mingyang/RoboDojo
bash scripts/migration/create_bundle.sh
cd migration_bundle
sha256sum -c SHA256SUMS
```

脚本不移动、不删除原文件。若中途中断，再次运行同一命令会保留已完成的归档并
继续缺失部分。只想先测试 Git/源码恢复流程时可使用：

```bash
bash scripts/migration/create_bundle.sh --skip-large-data
```

把整个 `migration_bundle/` 复制到目标机。推荐使用支持断点续传的工具，例如：

```bash
rsync -aH --info=progress2 migration_bundle/ user@new-host:/data/robodojo-migration/
```

传输后必须在目标机再次运行 `sha256sum -c SHA256SUMS`。

## 2. 在目标机器离线恢复源码和大文件

目标机需要先安装 `git`、`git-lfs`、`tar`、`zstd`、`rsync` 和 NVIDIA 驱动。
在迁移包目录执行：

```bash
cd /data/robodojo-migration
bash restore.sh /data/work/RoboDojo
cd /data/work/RoboDojo
```

`restore.sh` 会从离线 Git bundle 克隆主仓库和 submodule，恢复未提交修改，然后
解压全部大文件。它要求目标目录不存在，以避免覆盖已有工作区。

## 3. 让 Codex 完成机器相关安装

在目标机启动 Codex，打开恢复后的仓库并发送下面的提示词：

```text
请阅读 AGENTS.md 和 docs/MIGRATION_WITH_CODEX.md。这个仓库是从迁移包恢复的。
请先只做只读检查：核对 GPU/驱动、磁盘、Git/submodule、Assets、checkpoints 和
SHA-256 结果；然后运行官方本地安装流程，重建 RoboDojo conda 环境、Pi_05 的
OpenPI uv 环境和需要的 WCM 环境。不要下载或覆盖迁移包中已经恢复的大文件。
安装后重写 Assets 中与旧机器绑定的绝对路径，运行 doctor、任务 inventory、
ruff、git diff --check 和 dry-run eval/smoke，并汇报所有未通过项。任何需要 sudo、
联网下载或可能覆盖数据的步骤先请求我的批准。
```

Codex 获得批准后，标准安装顺序是：

```bash
# 主仿真环境：Python 3.11、Isaac Sim 5.1、IsaacLab、CuRobo
bash scripts/install.sh -i
conda activate RoboDojo

# 重新生成目标机器的 Assets 绝对路径
python utils/update_embodiment_config_path.py

# Pi_05 策略环境必须独立安装；不要把它装进 RoboDojo conda 环境
bash XPolicyLab/policy/Pi_05/install.sh

# 只在使用 WCM / RL-token post-training 时安装
bash scripts/posttrain/install_wcm.sh
```

`scripts/requirements.txt` 是 RoboDojo 直接 Python 依赖清单；Isaac Sim、IsaacLab、
CuRobo 由 `scripts/install.sh` 分阶段安装；Pi_05 使用
`XPolicyLab/policy/Pi_05/openpi/pyproject.toml` 和 uv；WCM 使用它自己的
`pyproject.toml`。不要用当前机器的 `pip freeze` 覆盖这些分层依赖，因为其中会
混入 editable 路径、GPU wheel 和机器相关包。

## 4. XPolicyLab submodule 的正确工作流

XPolicyLab 已经是 submodule。父仓库索引只保存一个 gitlink，因此内部文件修改在
父仓库里只显示为 `m XPolicyLab`，这是正常行为。当前迁移分支是
`mingyang/robodojo-migration`；迁移包同时保存该分支、本地提交、未提交 patch 和
untracked 源文件。

恢复后先检查：

```bash
git ls-files --stage XPolicyLab       # 第一列应为 160000
git -C XPolicyLab status --short --branch
git submodule status
```

确认未提交文件内容后，在子模块内部提交并推送，然后在父仓库更新 gitlink：

```bash
git -C XPolicyLab add <确认要保存的文件>
git -C XPolicyLab commit -m "[Pi_05] feat: describe local changes"
git -C XPolicyLab push -u origin mingyang/robodojo-migration

git add XPolicyLab
git commit -m "[scripts] update: pin XPolicyLab migration changes"
git push
```

不要在父仓库直接 `git add XPolicyLab/<file>`；父仓库不会逐文件追踪 submodule。
安装脚本也不再使用 `git submodule update --remote`，而是严格恢复父仓库记录的提交，
避免本地/固定版本被远端 `main` 意外替换。

## 5. 验证恢复结果

先运行不需要 Isaac/policy server 的检查：

```bash
git submodule status
bash -n scripts/install.sh scripts/init_assets.sh scripts/robodojo.sh \
  scripts/eval_policy.sh scripts/migration/create_bundle.sh \
  scripts/migration/restore_bundle.sh
python scripts/internal/task_inventory.py --format json --check
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
ruff check .
git diff --check
```

然后用真实 checkpoint 名称和恢复后的策略环境运行 dry-run：

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task stack_bowls \
  --ckpt <CKPT_NAME> \
  --policy-env openpi \
  --dry-run

bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt <CKPT_NAME> \
  --policy-env openpi \
  --only stack_bowls,push_T \
  --dry-run
```

最后再执行 `--eval-num 1` 的真实评测。验收不仅要看退出码，还要确认生成的
`_result.json` 中 `eval_time >= 1`。

## 6. 常见恢复问题

- `XPolicyLab` 显示 detached HEAD：切回
  `git -C XPolicyLab switch mingyang/robodojo-migration`。
- submodule 提交在 GitHub 上找不到：先从迁移包的 Git bundle 恢复，再把本地分支
  推送到有权限的远端；不要先执行 `submodule update --remote`。
- CuRobo 报旧机器上的绝对路径不存在：在仓库根目录重新运行
  `python utils/update_embodiment_config_path.py`。
- OpenPI/Isaac 依赖冲突：确认 Pi_05 使用 `openpi/.venv`，仿真客户端使用
  `RoboDojo` conda 环境，二者不要合并。
- 归档损坏或传输不完整：先运行 `sha256sum -c SHA256SUMS`，不要尝试带错误解压。
