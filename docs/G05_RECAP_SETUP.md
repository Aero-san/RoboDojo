# G05 RECAP source and environment setup

This document prepares the vendored G05 source and its Python environment for
RoboDojo RECAP. G05 uses a project-local `uv` virtual environment; it must not
be installed into the RoboDojo/Isaac Conda environment.

## Version anchor

- Upstream: `https://github.com/OpenGalaxea/GalaxeaVLA.git`
- Tested commit: `89f2322b4ad016e192437adc1a2c253b05bab246`
- Python requirement: `>=3.10.16,<3.11`
- Tested Python: `3.10.21`, managed by `uv`
- Tested PyTorch: `2.7.1+cu128`
- Expected environment location: `XPolicyLab/policy/G05/GalaxeaVLA/.venv`

The complete environment is approximately 11 GB. Keep at least another 12 GB
available while `uv` downloads and unpacks wheels. `ffmpeg` must be available in
`PATH` for video-backed datasets.

## 1. Use the vendored source

Run this from a RoboDojo checkout. Change `ROBODOJO_ROOT` on each server.

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export G05_DIR="${ROBODOJO_ROOT}/XPolicyLab/policy/G05/GalaxeaVLA"

test -f "${G05_DIR}/pyproject.toml" || {
  echo "Vendored G05 source is missing: ${G05_DIR}" >&2
  echo "Clone or pull a RoboDojo revision containing the vendored source." >&2
  exit 1
}
```

The source is intentionally nested below the XPolicyLab G05 adapter. Do not
copy a `.venv` from another server: virtual environments contain absolute paths.

## 2. Install or resume the environment

Install `uv` outside Conda, then run the following command. Clearing the Conda
variables is important on machines whose login shell automatically activates an
inaccessible or incompatible Conda environment.

```bash
cd "${G05_DIR}"

env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  -u CONDA_EXE \
  -u CONDA_PYTHON_EXE \
  -u CONDA_SHLVL \
  UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  UV_CACHE_DIR=/tmp/robodojo_g05_uv_cache \
  UV_HTTP_TIMEOUT=300 \
  UV_HTTP_RETRIES=10 \
  uv sync \
    --locked \
    --managed-python \
    --python 3.10 \
    --index-strategy unsafe-best-match
```

This command is idempotent. If a download is interrupted, run the same command
again. Reusing `UV_CACHE_DIR` avoids downloading completed wheels again. The
Aliyun index matches the registry recorded by the pinned `uv.lock`; CUDA wheels
still come from the PyTorch CUDA 12.8 index configured by the upstream project.

If `uv` itself is not installed, follow the official uv installation method and
then rerun the command above. Do not activate RoboDojo's Conda environment for
this step.

## 3. Verify the environment

```bash
cd "${G05_DIR}"

.venv/bin/python - <<'PY'
import accelerate
import fla
import g05
import torch
import transformers
from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
from g05.models.g05.inferencer import PolicyInferencer

print("torch", torch.__version__)
print("torch CUDA build", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("CUDA devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
PY

ffmpeg -version | head -n 1
```

An optional non-mutating consistency check is:

```bash
cd "${G05_DIR}"

env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  -u CONDA_EXE \
  -u CONDA_PYTHON_EXE \
  -u CONDA_SHLVL \
  UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  UV_CACHE_DIR=/tmp/robodojo_g05_uv_cache \
  uv sync \
    --locked \
    --dry-run \
    --managed-python \
    --python 3.10 \
    --index-strategy unsafe-best-match
```

The expected final line is `Would make no changes`.

## 4. Download the processor asset

RECAP uses the initial checkpoint's action tokenizer and dataset statistics, but
the vendored G05 source tree must also contain the Qwen 3.5 processor directory.
Download only that directory instead of the complete model repository:

```bash
cd "${G05_DIR}"

.venv/bin/hf download OpenGalaxea/G05 \
  --include 'qwen3_5_2b_base_processor/**' \
  --local-dir checkpoints

test -d checkpoints/qwen3_5_2b_base_processor
```

The resulting path is:

```text
XPolicyLab/policy/G05/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor
```

The RoboDojo FM-only checkpoint remains at
`XPolicyLab/policy/G05/checkpoints/hf_g05_robodojo_fm_only_checkpoint`.
RECAP packages that checkpoint and its sidecars automatically; it does not need
to be copied below the vendored source tree.

## 5. Configure local training with remote rollout

For the current topology, G05 training runs on the current server and rollout
runs on `XYZ4090`. Set these fields in `configs/posttrain/g05_remote.yaml`:

```yaml
environment:
  # Directory containing bin/python on XYZ4090.
  policy_env: /share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA/.venv

rollout:
  # Maximum deployed policy actions in every rollout/evaluation episode.
  max_steps: 40
  remote:
    enabled: true
    host: XYZ4090
    g05_root: /share/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA

training:
  remote:
    enabled: false

g05:
  root: /mnt/cpfs-E/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA
  processor_path: checkpoints/qwen3_5_2b_base_processor

runtime:
  policy_python: /mnt/cpfs-E/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA/.venv/bin/python
```

`rollout.max_steps` overrides the task's built-in episode limit for both local
and remote RECAP rollout/evaluation. It defaults to `40` and must be positive.
`environment.eval_env` remains the RoboDojo Conda environment. Do not point it
to the G05 `.venv`.

## 6. Configure remote training on XYZ6226 instead

Install stages 1 through 4 on `XYZ6226`, using its RoboDojo root
`/mnt/data-cpfs/mingyang/RoboDojo`. Then use:

```yaml
training:
  remote:
    enabled: true
    host: XYZ6226
    policy_python: /mnt/data-cpfs/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA/.venv/bin/python

g05:
  root: /mnt/data-cpfs/mingyang/RoboDojo/XPolicyLab/policy/G05/GalaxeaVLA

runtime:
  policy_python: null
```

The rollout settings from stage 5 still refer to `XYZ4090`; rollout and training
hosts each need their own RoboDojo checkout, `.venv`, and processor directory.

## 7. Validate and launch

From the RoboDojo root:

```bash
../miniconda3/envs/RoboDojo/bin/python \
  scripts/posttrain/recap_config.py \
  configs/posttrain/g05_remote.yaml \
  --format yaml \
  --output /tmp/g05_remote_resolved.yaml

RECAP_CONFIG=configs/posttrain/g05_remote.yaml bash remote_training.sh
```

The second command starts the actual RECAP run. Run it only after the config
validator succeeds and the corresponding RoboDojo/environment paths exist on
every enabled host.

## Current-server recovery point (2026-08-28)

Current server root: `/mnt/cpfs-E/mingyang/RoboDojo`.

- Stage 1 complete: vendored G05 source is available at
  `XPolicyLab/policy/G05/GalaxeaVLA`, imported at commit `89f2322`.
- Stage 2 complete: `.venv` is installed with Python 3.10.21 and occupies about
  11 GB. The reusable temporary uv cache is `/tmp/robodojo_g05_uv_cache`, also
  about 11 GB.
- Stage 3 complete: `g05`, PyTorch 2.7.1+cu128, Transformers 4.57.1,
  Accelerate 1.8.1, FLA, the G05 policy/inferencer modules, eight CUDA devices,
  and `ffmpeg` were verified. A locked dry-run reports `Would make no changes`.
- Stage 4 pending: `checkpoints/qwen3_5_2b_base_processor` is absent.
- Stage 5 pending: `configs/posttrain/g05_remote.yaml` still contains obsolete
  `/mlplatform/...` and `/efm-nas/...` G05 paths and has
  `runtime.policy_python: null` while local training is enabled.
- Remote setup pending: `XYZ4090` and `XYZ6226` do not currently contain the
  RoboDojo checkout or `.venv` at the paths shown above.

Therefore, continue from **stage 4** on the current server. If environment
integrity is ever uncertain, rerun stage 2 first; it is safe and will reuse the
existing `.venv` and cache.
