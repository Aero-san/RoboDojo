# 使用 Codex 迁移和恢复 RoboDojo

本文档用于把 RoboDojo 工作区迁移到另一台 Linux 机器。RoboDojo 现在是单一
Git 仓库：`XPolicyLab`、IsaacLab、CuRobo、WCM 和 G05 GalaxeaVLA 的源码都已经
作为普通文件纳入主仓库。源码迁移只需要 Git；Assets、checkpoints、训练数据和
运行结果仍然按机器单独迁移。

官方安装页仍是环境和硬件要求的上游参考：
<https://robodojo-benchmark.com/doc/usage/install-and-download/>。

## 迁移包包含什么

升级后的 `scripts/migration/create_bundle.sh` 默认在 `migration_bundle/` 中生成：

- RoboDojo 单一 Git bundle，以及当前工作区的未提交二进制 patch；
- 当前工作区未被 Git 忽略的 untracked 文件；
- `Assets/`；
- `XPolicyLab/policy/Pi_05/checkpoints/`；
- `data/`、`outputs/`、`eval_result/`、`wandb/` 和 `smoke_results/`；
- 环境元数据和覆盖归档、Git bundle、patch 的 `SHA256SUMS`。

不会打包 Python 虚拟环境和缓存：根目录 `.venv/`、OpenPI `.venv/`、WCM
`.venv/` 和 `.cache/` 都与机器路径、CUDA 和系统库绑定，应在新机器重新安装。

## 1. 优先使用 Git 同步源码

完成这次转换后，在源机器提交并推送 RoboDojo 改动：

```bash
git add -A
git commit -m "[scripts] refactor: vendor project dependencies"
git push origin main
```

目标机器可以直接获得完整源码：

```bash
git clone <RoboDojo-repository-url> /data/work/RoboDojo
cd /data/work/RoboDojo
git pull --ff-only
```

不需要 `git submodule init`、`git submodule update` 或额外的 XPolicyLab/WCM
仓库。后续对这些目录的代码修改也直接在 RoboDojo 中提交和同步。各源码快照的
来源和 commit 记录在 [`VENDORED_SOURCES.md`](VENDORED_SOURCES.md)。

如果目标机器已有旧版 submodule 工作区，推荐在备份本地修改、checkpoint、数据
和 Assets 后新建一个 clone，再把这些机器本地目录挂载或复制过去。若必须复用旧
目录，先备份所有非 Git 文件和本地改动，再执行旧版本的
`git submodule deinit -f --all`，最后 `git pull --ff-only`；不要在未备份数据时
强制清理旧 submodule 工作树。

## 2. 在源机器创建离线迁移包

确保没有训练或评测进程仍在写 checkpoints、data、outputs 或 eval_result，避免
得到跨时间点的不一致快照，然后执行：

```bash
cd /share/mingyang/RoboDojo
bash scripts/migration/create_bundle.sh
cd migration_bundle
sha256sum -c SHA256SUMS
```

脚本不移动、不删除原文件。若中途中断，再次运行同一命令会保留已完成的归档并
继续缺失部分。只想先测试单一 Git 仓库恢复流程时可使用：

```bash
bash scripts/migration/create_bundle.sh --skip-large-data
```

把整个 `migration_bundle/` 复制到目标机。推荐使用支持断点续传的工具，例如：

```bash
rsync -aH --info=progress2 migration_bundle/ user@new-host:/data/robodojo-migration/
```

传输后必须在目标机再次运行 `sha256sum -c SHA256SUMS`。

## 3. 在目标机器离线恢复源码和大文件

目标机需要先安装 `git`、`git-lfs`、`tar`、`zstd`、`rsync` 和 NVIDIA 驱动。在
迁移包目录执行：

```bash
cd /data/robodojo-migration
bash restore.sh /data/work/RoboDojo
cd /data/work/RoboDojo
```

`restore.sh` 只从 RoboDojo 的离线 Git bundle 克隆一个仓库，恢复主仓库工作区
patch，然后解压大文件归档；不会访问或初始化任何 submodule。它要求目标目录不
存在，以避免覆盖已有工作区。

## 4. 让 Codex 完成机器相关安装

在目标机启动 Codex，打开恢复后的仓库并发送下面的提示词：

```text
请阅读 AGENTS.md 和 docs/MIGRATION_WITH_CODEX.md。这个仓库是 RoboDojo 单一仓库。
请先只做只读检查：核对 GPU/驱动、磁盘、Git、Assets、checkpoints 和 SHA-256
结果；然后运行官方本地安装流程，重建 RoboDojo conda 环境、Pi_05 的 OpenPI
uv 环境和需要的 WCM 环境。不要下载或覆盖迁移包中已经恢复的大文件。安装后
重写 Assets 中与旧机器绑定的绝对路径，运行 doctor、任务 inventory、ruff、
Git 差异检查和 dry-run eval/smoke，并汇报所有未通过项。vendored 上游源码的
差异检查请排除；任何需要 sudo、
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

## 5. 验证单仓库恢复结果

先运行不需要 Isaac/policy server 的检查：

```bash
git ls-files --stage | awk '$1 == "160000" {print}'   # 应无输出
find XPolicyLab third_party external_dependencies \
  \( -name .git -o -name .gitmodules \) -print       # 应无输出
bash -n scripts/install.sh scripts/init_assets.sh scripts/robodojo.sh \
  scripts/eval_policy.sh scripts/migration/create_bundle.sh \
  scripts/migration/restore_bundle.sh
python scripts/internal/task_inventory.py --format json --check
bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
ruff check .
git diff --check -- . ':!XPolicyLab' ':!third_party' ':!external_dependencies'
git diff --cached --check -- . ':!XPolicyLab' ':!third_party' ':!external_dependencies'
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

- `XPolicyLab`、`third_party/IsaacLab`、`third_party/curobo` 或
  `external_dependencies/WCM` 为空：确认 clone/pull 使用的是完成单仓库转换的
  RoboDojo commit，不要执行 submodule 命令；必要时重新 clone。
- CuRobo 报旧机器上的绝对路径不存在：在仓库根目录重新运行
  `python utils/update_embodiment_config_path.py`。
- OpenPI/Isaac 依赖冲突：确认 Pi_05 使用 `openpi/.venv`，仿真客户端使用
  `RoboDojo` conda 环境，二者不要合并。
- 归档损坏或传输不完整：先运行 `sha256sum -c SHA256SUMS`，不要尝试带错误解压。
