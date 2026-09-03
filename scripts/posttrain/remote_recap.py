"""Transfer compressed RECAP jobs to a local or SSH-addressable host."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile

SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _is_local_host(host: str) -> bool:
    return host in {
        "local",
        "localhost",
        "127.0.0.1",
        "::1",
        socket.gethostname(),
        socket.getfqdn(),
    }


def _remote(args: argparse.Namespace, command: list[str]) -> None:
    if _is_local_host(args.host):
        _run(["bash", "-lc", shlex.join(command)])
    else:
        _run(["ssh", *SSH_OPTIONS, args.host, shlex.join(command)])


def _remote_result(args: argparse.Namespace, command: list[str]) -> subprocess.CompletedProcess[str]:
    if _is_local_host(args.host):
        return subprocess.run(
            ["bash", "-lc", shlex.join(command)],
            check=False,
            capture_output=True,
            text=True,
        )
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, args.host, shlex.join(command)],
        check=False,
        capture_output=True,
        text=True,
    )


def _remote_success(args: argparse.Namespace, command: list[str]) -> bool:
    return _remote_result(args, command).returncode == 0


def _scp(args: argparse.Namespace, source: str, destination: str) -> None:
    if _is_local_host(args.host):
        prefix = f"{args.host}:"

        def strip_local_endpoint(value: str) -> str:
            return value[len(prefix) :] if value.startswith(prefix) else value

        local_source = Path(strip_local_endpoint(source)).expanduser()
        local_destination = Path(strip_local_endpoint(destination)).expanduser()
        local_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_source, local_destination)
        return
    _run(["scp", "-q", source, destination])


def _validate(args: argparse.Namespace) -> None:
    for name in ("remote_repo_root", "remote_work_root"):
        value = getattr(args, name)
        if not value.startswith("/") or any(character.isspace() for character in value):
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path without whitespace")
    for name in ("remote_zstd_bin", "remote_conda_bin", "remote_python_bin"):
        executable = getattr(args, name)
        if not executable or ("/" in executable and not executable.startswith("/")):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a command name or absolute path"
            )


def _install_worker(args: argparse.Namespace) -> str:
    local_worker = Path(__file__).with_name("remote_recap_worker.sh").resolve()
    local_reservation = Path(__file__).with_name("reserve_gpu_memory.py").resolve()
    local_g05_preparer = Path(__file__).with_name(
        "prepare_g05_inference_checkpoint.py"
    ).resolve()
    local_value_renderer = Path(__file__).with_name(
        "render_rollout_value_videos.py"
    ).resolve()
    remote_bin = f"{args.remote_work_root}/bin"
    remote_worker = f"{remote_bin}/remote_recap_worker.sh"
    _remote(args, ["mkdir", "-p", remote_bin])
    for local_path, remote_path in (
        (local_worker, remote_worker),
        (local_reservation, f"{remote_bin}/reserve_gpu_memory.py"),
        (local_g05_preparer, f"{remote_bin}/prepare_g05_inference_checkpoint.py"),
        (local_value_renderer, f"{remote_bin}/render_rollout_value_videos.py"),
    ):
        temporary = f"{remote_path}.tmp-{os.getpid()}"
        _scp(args, str(local_path), f"{args.host}:{temporary}")
        _remote(args, ["mv", temporary, remote_path])
    _remote(args, ["chmod", "755", remote_worker])
    return remote_worker


def _install_g05_adapter(args: argparse.Namespace) -> None:
    local_adapter = (
        Path(__file__).resolve().parents[2] / "XPolicyLab/policy/G05/model.py"
    )
    remote_adapter = f"{args.remote_repo_root}/XPolicyLab/policy/G05/model.py"
    temporary = f"{remote_adapter}.tmp-{os.getpid()}"
    _scp(args, str(local_adapter), f"{args.host}:{temporary}")
    _remote(args, ["mv", temporary, remote_adapter])
    print("[RECAP remote] installed current G05 policy adapter", flush=True)


def _resolve_pi05_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if (checkpoint / "params").is_dir():
        return checkpoint
    candidates = sorted(
        (
            child
            for child in checkpoint.iterdir()
            if child.is_dir() and child.name.isdigit() and (child / "params").is_dir()
        ),
        key=lambda child: int(child.name),
    ) if checkpoint.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No Pi0.5 params checkpoint below: {checkpoint}")
    return candidates[-1]


def _g05_bundle(checkpoint: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        candidates = [checkpoint / "checkpoints" / "checkpoint"]
        candidates.extend(sorted(checkpoint.glob("checkpoints/step_*.pt")))
        candidates.extend(checkpoint.glob("**/model_state_dict.pt"))
        candidates = [candidate for candidate in candidates if candidate.is_file()]
        if not candidates:
            raise FileNotFoundError(f"No G05 checkpoint below: {checkpoint}")
        checkpoint = candidates[-1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"G05 checkpoint does not exist: {checkpoint}")
    for root in checkpoint.parents:
        if (root / ".hydra" / "config.yaml").is_file():
            for sidecar in ("dataset_stats.json", "action_tokenizer.pt"):
                if not (root / sidecar).is_file():
                    raise FileNotFoundError(f"G05 checkpoint bundle is missing {sidecar}: {root}")
            return root, checkpoint
    raise FileNotFoundError(f"No G05 Hydra run and sidecars above: {checkpoint}")


def _package_checkpoint(policy: str, checkpoint: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    if policy == "pi05":
        checkpoint = _resolve_pi05_checkpoint(checkpoint)
        command = [
            "tar", "--zstd", "-cf", str(temporary), "--exclude=./train_state",
            "-C", str(checkpoint), ".",
        ]
        metadata = checkpoint / "robodojo_pi05_model.json"
        parent_metadata = checkpoint.parent / "robodojo_pi05_model.json"
        if not metadata.is_file() and parent_metadata.is_file():
            command.extend(["-C", str(checkpoint.parent), parent_metadata.name])
    elif policy == "g05":
        root, checkpoint = _g05_bundle(checkpoint)
        command = [
            "tar", "--zstd", "-cf", str(temporary), "-C", str(root),
            str(checkpoint.relative_to(root)), ".hydra", "dataset_stats.json",
            "action_tokenizer.pt",
        ]
        metadata = root / "robodojo_g05_model.json"
        if metadata.is_file():
            command.append(metadata.name)
    else:
        raise ValueError(f"Unsupported RECAP policy: {policy}")
    _run(command)
    temporary.replace(archive)


def _package_rollout_cache(rollout_root: Path, archive: Path, expected: int) -> None:
    rollout_root = rollout_root.expanduser().resolve()
    if not rollout_root.is_dir():
        raise FileNotFoundError(f"Local rollout source is missing: {rollout_root}")
    manifests = list((rollout_root / "episodes").glob("*/manifest.json"))
    if len(manifests) < expected:
        raise ValueError(
            f"Local rollout source has {len(manifests)} complete episodes, "
            f"need {expected}: {rollout_root}"
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    _run(
        [
            "tar",
            "--zstd",
            "-cf",
            str(temporary),
            "--exclude=./_in_progress",
            "-C",
            str(rollout_root),
            ".",
        ]
    )
    temporary.replace(archive)


def _extract(archive: Path, destination: Path, member: str) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix="recap-transfer-", dir=destination.parent))
    try:
        _run(["tar", "--zstd", "-xf", str(archive), "-C", str(temporary_parent)])
        extracted = temporary_parent / member
        if not extracted.is_dir():
            raise FileNotFoundError(f"Remote archive does not contain {member}/")
        if destination.exists():
            shutil.rmtree(destination)
        extracted.replace(destination)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _job_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    job_root = f"{args.remote_work_root}/jobs/{args.job_id}"
    inbox = f"{job_root}/inbox"
    result = f"{job_root}/{args.action}.tar.zst"
    return job_root, inbox, result


def _worker_environment(args: argparse.Namespace, job_root: str, action: str) -> dict[str, str]:
    return {
        "RECAP_REMOTE_ACTION": action,
        "RECAP_REMOTE_REPO_ROOT": args.remote_repo_root,
        "RECAP_REMOTE_WORK_ROOT": args.remote_work_root,
        "RECAP_REMOTE_JOB_ROOT": job_root,
        "RECAP_REMOTE_ZSTD_BIN": args.remote_zstd_bin,
        "RECAP_REMOTE_CONDA_BIN": args.remote_conda_bin,
        "RECAP_REMOTE_PYTHON_BIN": args.remote_python_bin,
    }


def _invoke_worker(
    args: argparse.Namespace,
    worker: str,
    environment: dict[str, str],
    *,
    isolated_process_group: bool = False,
) -> None:
    command = ["env", *(f"{key}={value}" for key, value in environment.items()), worker]
    if isolated_process_group:
        command[:0] = ["setsid", "--wait"]
    _remote(args, command)


def _reserve_remote_gpus(args: argparse.Namespace, job_root: str, gpu_ids: list[int]) -> None:
    if not args.gpu_reservation:
        return
    worker = _install_worker(args)
    environment = _worker_environment(args, job_root, "reserve")
    environment.update(
        {
            "RECAP_REMOTE_RESERVATION_GPUS": ",".join(map(str, sorted(set(gpu_ids)))),
            "RECAP_REMOTE_RESERVATION_LEAVE_FREE_MIB": str(
                args.gpu_reservation_leave_free_mib
            ),
            "RECAP_REMOTE_RESERVATION_IDLE_USED_MAX_MIB": str(
                args.gpu_reservation_idle_used_max_mib
            ),
            "RECAP_REMOTE_RESERVATION_MAX_HOLD_SECONDS": str(
                args.gpu_reservation_remote_max_hold_seconds
            ),
        }
    )
    _invoke_worker(args, worker, environment)


def _cancel_remote_job(args: argparse.Namespace, job_root: str) -> None:
    try:
        worker = _install_worker(args)
        environment = _worker_environment(args, job_root, "cancel")
        _invoke_worker(args, worker, environment)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[RECAP remote] cleanup warning for {job_root}: {error}", file=sys.stderr)


def preflight(args: argparse.Namespace) -> None:
    _validate(args)
    failures = 0

    def check(
        label: str,
        command: list[str],
        fix: str,
        *,
        output_contains: str | None = None,
    ) -> bool:
        nonlocal failures
        result = _remote_result(args, command)
        passed = result.returncode == 0 and (
            output_contains is None or output_contains in result.stdout
        )
        if passed:
            print(f"[RECAP remote] OK: {label}", flush=True)
            return True
        failures += 1
        print(f"[RECAP remote] FAIL: {label}", file=sys.stderr)
        detail = (result.stderr or result.stdout).strip()
        if detail:
            print(f"  {detail}", file=sys.stderr)
        print(f"  Fix: {fix}", file=sys.stderr, flush=True)
        return False

    if not check(
        "passwordless SSH",
        ["true"],
        f"Run `ssh {shlex.quote(args.host)} true` and configure the SSH key/alias.",
    ):
        raise RuntimeError(f"Remote preflight failed for {args.host}: SSH is unavailable")

    if "/" in args.remote_zstd_bin:
        zstd_check = ["test", "-x", args.remote_zstd_bin]
    else:
        zstd_check = ["command", "-v", args.remote_zstd_bin]
    check(
        f"zstd available to the non-login worker ({args.remote_zstd_bin})",
        zstd_check,
        "Install zstd or set RECAP_REMOTE_ZSTD_BIN=/absolute/path/to/zstd.",
    )

    check(
        f"conda available to the non-login worker ({args.remote_conda_bin})",
        [args.remote_conda_bin, "info", "--base"],
        "Install conda or set RECAP_REMOTE_CONDA_BIN=/absolute/path/to/conda.",
    )
    check(
        f"bootstrap Python provides PyYAML ({args.remote_python_bin})",
        [args.remote_python_bin, "-c", "import yaml"],
        "Set RECAP_REMOTE_PYTHON_BIN to a remote Python environment containing PyYAML.",
    )

    eval_env = getattr(args, "eval_env", "")
    if eval_env:
        eval_selector = (
            ["--prefix", eval_env]
            if eval_env.startswith("/")
            else ["--name", eval_env]
        )
        check(
            f"remote simulator conda environment exists ({eval_env})",
            [
                args.remote_conda_bin,
                "run",
                *eval_selector,
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ],
            (
                "Set rollout.remote.eval_env to a conda environment that exists on "
                f"{args.host}; do not use a path from another host."
            ),
        )

    if (
        getattr(args, "policy", "pi05") == "g05"
        and getattr(args, "policy_env", "")
    ):
        policy_env = args.policy_env.rstrip("/")
        policy_python = f"{policy_env}/bin/python"
        policy_python_script = (
            'if test -x "$1/bin/python"; then python="$1/bin/python"; '
            'elif test -x "$1"; then python="$1"; '
            'else exit 127; fi; shift; exec "$python" "$@"'
        )
        policy_python_command = [
            "sh",
            "-c",
            policy_python_script,
            "sh",
            policy_env,
            "-c",
            (
                "import sys; print(sys.executable); "
                "import XPolicyLab.client_server.ws.protocol.codec"
            ),
        ]
        check(
            f"G05 policy Python exists ({policy_env})",
            [
                "sh",
                "-c",
                'test -x "$1/bin/python" || test -x "$1"',
                "sh",
                policy_env,
            ],
            "Set environment.policy_env to the remote G05 venv or Python executable.",
        )
        check(
            f"G05 policy Python imports the XPolicyLab WebSocket codec ({policy_env})",
            [
                "env",
                (
                    f"PYTHONPATH={args.remote_repo_root}:"
                    f"{args.remote_repo_root}/XPolicyLab"
                ),
                *policy_python_command,
            ],
            (
                "Install XPolicyLab runtime dependencies into the G05 environment, e.g. "
                f"`uv pip install --python {policy_python} -e "
                f"{args.remote_repo_root}/XPolicyLab`."
            ),
        )

    robodojo = f"{args.remote_repo_root}/scripts/robodojo.sh"
    check(
        f"RoboDojo launcher exists and is readable ({robodojo})",
        ["sh", "-c", 'test -f "$1" && test -r "$1"', "sh", robodojo],
        "Set RECAP_REMOTE_REPO_ROOT to the remote checkout's absolute path.",
    )
    check(
        "GNU tar supports --zstd",
        ["tar", "--help"],
        "Install GNU tar with zstd support on the remote host.",
        output_contains="--zstd",
    )
    check(
        "setsid --wait is available for isolated remote worker process groups",
        ["setsid", "--wait", "true"],
        "Install util-linux (provides setsid) on the remote host.",
    )
    for gpu in sorted(set(args.gpu)):
        check(
            f"GPU {gpu} is visible",
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
            f"Check `nvidia-smi -i {gpu}` on {args.host} or choose an available remote GPU.",
        )
    if args.require_wcm or args.gpu_reservation:
        wcm_python = f"{args.remote_repo_root}/external_dependencies/WCM/.venv/bin/python"
        check(
            f"WCM/reservation Python exists and is executable ({wcm_python})",
            ["test", "-x", wcm_python],
            f"Run `bash {args.remote_repo_root}/scripts/posttrain/install_wcm.sh` on {args.host}.",
        )
    if failures:
        raise RuntimeError(f"Remote preflight failed for {args.host}: {failures} check(s) failed")
    _install_worker(args)
    print("[RECAP remote] OK: remote worker installed", flush=True)


def rollout(args: argparse.Namespace) -> None:
    #_remote(args, ["pkill", "-9", "-u", "mingyang", "-x", "python"])
    _validate(args)
    if args.policy == "g05" and (
        not args.g05_processor_path.startswith("/")
        or any(character.isspace() for character in args.g05_processor_path)
    ):
        raise ValueError(
            "--g05-processor-path must be an absolute remote path without whitespace"
        )
    checkpoint = Path(args.checkpoint)
    local_output = Path(args.output)
    transfer_dir = local_output.parent / ".remote_transfers"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    job_root, inbox, result = _job_paths(args)
    result_local = transfer_dir / f"{args.job_id}-rollout.tar.zst"
    if _remote_success(args, ["test", "-f", result]):
        _scp(args, f"{args.host}:{result}", str(result_local))
        _extract(result_local, local_output, "rollouts")
        result_local.unlink(missing_ok=True)
        return
    archive = transfer_dir / f"{args.job_id}-policy.tar.zst"
    remote_checkpoint = f"{inbox}/policy.tar.zst"
    try:
        _reserve_remote_gpus(args, job_root, [args.policy_gpu, args.env_gpu])
        worker = _install_worker(args)
        if args.policy == "g05":
            _install_g05_adapter(args)
        if not _remote_success(args, ["test", "-f", f"{job_root}/policy/.extracted"]):
            _package_checkpoint(args.policy, checkpoint, archive)
            _remote(args, ["mkdir", "-p", inbox])
            _scp(args, str(archive), f"{args.host}:{remote_checkpoint}.tmp")
            _remote(args, ["mv", f"{remote_checkpoint}.tmp", remote_checkpoint])
        environment = _worker_environment(args, job_root, "rollout")
        environment.update(
            {
                "RECAP_REMOTE_CHECKPOINT_ARCHIVE": remote_checkpoint,
                "RECAP_REMOTE_POLICY": args.policy,
                "RECAP_REMOTE_G05_ROOT": args.g05_root,
                "RECAP_REMOTE_G05_PROCESSOR_PATH": args.g05_processor_path,
                "RECAP_REMOTE_G05_ACTION_SOURCE": args.g05_action_source,
                "RECAP_REMOTE_RESULT_ARCHIVE": result,
                "RECAP_REMOTE_TASK": args.task,
                "RECAP_REMOTE_EPISODES": str(args.episodes),
                "RECAP_REMOTE_MAX_STEPS": str(args.max_steps),
                "RECAP_REMOTE_FIXED_HORIZON": (
                    "1" if getattr(args, "fixed_horizon", False) else "0"
                ),
                "RECAP_REMOTE_LAYOUT_SEED": str(args.layout_seed),
                "RECAP_REMOTE_LAYOUT_OFFSET": str(args.layout_offset),
                "RECAP_REMOTE_POLICY_GPU": str(args.policy_gpu),
                "RECAP_REMOTE_ENV_GPU": str(args.env_gpu),
                "RECAP_REMOTE_ENV_CFG": args.env_cfg,
                "RECAP_REMOTE_ACTION_TYPE": args.action_type,
                "RECAP_REMOTE_POLICY_ENV": args.policy_env,
                "RECAP_REMOTE_EVAL_ENV": args.eval_env,
            }
        )
        _invoke_worker(args, worker, environment, isolated_process_group=True)
        _scp(args, f"{args.host}:{result}", str(result_local))
        _extract(result_local, local_output, "rollouts")
        archive.unlink(missing_ok=True)
        result_local.unlink(missing_ok=True)
    finally:
        _cancel_remote_job(args, job_root)


def value_video(args: argparse.Namespace) -> None:
    _validate(args)
    job_root, inbox, result = _job_paths(args)
    transfer_dir = Path(args.output).parent / ".remote_transfers"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    result_local = transfer_dir / f"{args.job_id}-value-video.tar.zst"
    if _remote_success(args, ["test", "-f", result]):
        _scp(args, f"{args.host}:{result}", str(result_local))
        _extract(result_local, Path(args.output), "value_videos")
        result_local.unlink(missing_ok=True)
        return
    try:
        _reserve_remote_gpus(args, job_root, [args.gpu])
        worker = _install_worker(args)
        rollout_archive = transfer_dir / f"{args.job_id}-rollouts.tar.zst"
        remote_rollout = f"{inbox}/rollouts.tar.zst"
        if not _remote_success(args, ["test", "-d", f"{job_root}/rollouts/episodes"]):
            print("[RECAP remote] rollout cache missing; rebuilding it from local artifacts")
            _package_rollout_cache(Path(args.rollout_root), rollout_archive, args.episodes)
            _remote(args, ["mkdir", "-p", inbox])
            _scp(args, str(rollout_archive), f"{args.host}:{remote_rollout}.tmp")
            _remote(args, ["mv", f"{remote_rollout}.tmp", remote_rollout])
        wcm_archive = transfer_dir / f"{args.job_id}-wcm.pt.zst"
        _run(["zstd", "-q", "-f", args.wcm_checkpoint, "-o", str(wcm_archive)])
        remote_wcm = f"{inbox}/wcm.pt.zst"
        _remote(args, ["mkdir", "-p", inbox])
        _scp(args, str(wcm_archive), f"{args.host}:{remote_wcm}.tmp")
        _remote(args, ["mv", f"{remote_wcm}.tmp", remote_wcm])
        environment = _worker_environment(args, job_root, "value-video")
        environment.update(
            {
                "RECAP_REMOTE_WCM_ARCHIVE": remote_wcm,
                "RECAP_REMOTE_ROLLOUT_ARCHIVE": remote_rollout,
                "RECAP_REMOTE_RESULT_ARCHIVE": result,
                "RECAP_REMOTE_VALUE_EPISODES": str(args.episodes),
                "RECAP_REMOTE_VALUE_GPU": str(args.gpu),
                "RECAP_REMOTE_VALUE_BATCH_SIZE": str(args.batch_size),
                "RECAP_REMOTE_VALUE_DEVICE": args.device,
                "RECAP_REMOTE_VALUE_PRECISION": args.precision,
                "RECAP_REMOTE_VALUE_BACKEND": args.backend,
                "RECAP_REMOTE_VALUE_SPEED": str(args.speed),
                "RECAP_REMOTE_VALUE_Y_MIN": str(args.y_min),
                "RECAP_REMOTE_VALUE_Y_MAX": str(args.y_max),
                "RECAP_REMOTE_VALUE_TITLE": args.title,
            }
        )
        _invoke_worker(args, worker, environment, isolated_process_group=True)
        _scp(args, f"{args.host}:{result}", str(result_local))
        _extract(result_local, Path(args.output), "value_videos")
        wcm_archive.unlink(missing_ok=True)
        rollout_archive.unlink(missing_ok=True)
        result_local.unlink(missing_ok=True)
    finally:
        _cancel_remote_job(args, job_root)


def cancel(args: argparse.Namespace) -> None:
    _validate(args)
    job_root, _, _ = _job_paths(args)
    _cancel_remote_job(args, job_root)


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-repo-root", required=True)
    parser.add_argument("--remote-work-root", required=True)
    parser.add_argument(
        "--remote-zstd-bin",
        default=os.environ.get("RECAP_REMOTE_ZSTD_BIN", "zstd"),
        help="zstd command name or absolute path in the remote non-login shell",
    )
    parser.add_argument(
        "--remote-conda-bin",
        default=os.environ.get("RECAP_REMOTE_CONDA_BIN", "conda"),
        help="conda command name or absolute path in the remote non-login shell",
    )
    parser.add_argument(
        "--remote-python-bin",
        default=os.environ.get("RECAP_REMOTE_PYTHON_BIN", "python"),
        help="Python with PyYAML for remote policy bootstrap scripts",
    )
    parser.add_argument(
        "--gpu-reservation",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GPU_RESERVATION_ENABLED", "1") != "0",
        help="reserve otherwise-idle remote GPUs during transfer and CPU-only preparation",
    )
    parser.add_argument(
        "--gpu-reservation-leave-free-mib",
        type=int,
        default=int(os.environ.get("GPU_RESERVATION_FREE_MIB", "2048")),
    )
    parser.add_argument(
        "--gpu-reservation-idle-used-max-mib",
        type=int,
        default=int(os.environ.get("GPU_RESERVATION_IDLE_USED_MAX_MIB", "64")),
        help="maximum existing memory use for a remote GPU to count as idle",
    )
    parser.add_argument(
        "--gpu-reservation-remote-max-hold-seconds",
        type=int,
        default=int(os.environ.get("GPU_RESERVATION_REMOTE_MAX_HOLD_SECONDS", "1800")),
        help="failsafe lifetime for a detached remote GPU reservation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    common(preflight_parser)
    preflight_parser.add_argument("--policy", choices=("pi05", "g05"), default="pi05")
    preflight_parser.add_argument("--policy-env", default="")
    preflight_parser.add_argument("--eval-env", required=True)
    preflight_parser.add_argument("--gpu", type=int, action="append", default=[])
    preflight_parser.add_argument("--require-wcm", action="store_true")
    preflight_parser.set_defaults(function=preflight)
    rollout_parser = subparsers.add_parser("rollout")
    common(rollout_parser)
    rollout_parser.add_argument("--job-id", required=True)
    rollout_parser.add_argument("--checkpoint", required=True)
    rollout_parser.add_argument("--policy", choices=("pi05", "g05"), required=True)
    rollout_parser.add_argument("--g05-root", default="")
    rollout_parser.add_argument("--g05-processor-path", default="")
    rollout_parser.add_argument("--g05-action-source", choices=("fm", "ar"), default="fm")
    rollout_parser.add_argument("--output", required=True)
    rollout_parser.add_argument("--task", required=True)
    rollout_parser.add_argument("--episodes", type=int, required=True)
    rollout_parser.add_argument("--max-steps", type=int, required=True)
    rollout_parser.add_argument("--fixed-horizon", action="store_true")
    rollout_parser.add_argument("--layout-seed", type=int, required=True)
    rollout_parser.add_argument("--layout-offset", type=int, default=0)
    rollout_parser.add_argument("--policy-gpu", type=int, required=True)
    rollout_parser.add_argument("--env-gpu", type=int, required=True)
    rollout_parser.add_argument("--env-cfg", required=True)
    rollout_parser.add_argument("--action-type", required=True)
    rollout_parser.add_argument("--policy-env", required=True)
    rollout_parser.add_argument("--eval-env", required=True)
    rollout_parser.set_defaults(function=rollout)
    video_parser = subparsers.add_parser("value-video")
    common(video_parser)
    video_parser.add_argument("--job-id", required=True)
    video_parser.add_argument("--wcm-checkpoint", required=True)
    video_parser.add_argument("--rollout-root", required=True)
    video_parser.add_argument("--output", required=True)
    video_parser.add_argument("--episodes", type=int, required=True)
    video_parser.add_argument("--gpu", type=int, required=True)
    video_parser.add_argument("--batch-size", type=int, default=16)
    video_parser.add_argument("--device", default="cuda")
    video_parser.add_argument("--precision", default="bf16")
    video_parser.add_argument("--backend", default="auto")
    video_parser.add_argument("--speed", type=float, default=1.0)
    video_parser.add_argument("--y-min", type=float, default=-1.0)
    video_parser.add_argument("--y-max", type=float, default=1.0)
    video_parser.add_argument("--title", default="WCM RECAP")
    video_parser.set_defaults(function=value_video)
    cancel_parser = subparsers.add_parser("cancel")
    common(cancel_parser)
    cancel_parser.add_argument("--job-id", required=True)
    cancel_parser.set_defaults(function=cancel)
    arguments = parser.parse_args()
    if arguments.action == "rollout" and arguments.max_steps < 1:
        parser.error("--max-steps must be a positive integer")
    if arguments.gpu_reservation_leave_free_mib < 256:
        parser.error("--gpu-reservation-leave-free-mib must be at least 256")
    if arguments.gpu_reservation_idle_used_max_mib < 0:
        parser.error("--gpu-reservation-idle-used-max-mib cannot be negative")
    if arguments.gpu_reservation_remote_max_hold_seconds < 60:
        parser.error("--gpu-reservation-remote-max-hold-seconds must be at least 60")

    def terminate(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        arguments.function(arguments)
    except KeyboardInterrupt:
        parser.exit(130, "remote_recap: interrupted; remote cleanup requested\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"remote_recap: {error}\n")
    except subprocess.CalledProcessError as error:
        command = shlex.join(str(part) for part in error.cmd)
        parser.exit(
            error.returncode or 1,
            f"remote_recap: command failed with exit status {error.returncode}: {command}\n",
        )


if __name__ == "__main__":
    main()
