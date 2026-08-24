"""Transfer compressed RECAP jobs to a remote simulator host over SSH."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _remote(args: argparse.Namespace, command: list[str]) -> None:
    _run(["ssh", "-o", "BatchMode=yes", args.host, shlex.join(command)])


def _remote_success(args: argparse.Namespace, command: list[str]) -> bool:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host, shlex.join(command)], check=False
    )
    return result.returncode == 0


def _scp(args: argparse.Namespace, source: str, destination: str) -> None:
    _run(["scp", "-q", source, destination])


def _validate(args: argparse.Namespace) -> None:
    for name in ("remote_repo_root", "remote_work_root"):
        value = getattr(args, name)
        if not value.startswith("/") or any(character.isspace() for character in value):
            raise ValueError(f"--{name.replace('_', '-')} must be an absolute path without whitespace")


def _install_worker(args: argparse.Namespace) -> str:
    local_worker = Path(__file__).with_name("remote_recap_worker.sh").resolve()
    remote_bin = f"{args.remote_work_root}/bin"
    remote_worker = f"{remote_bin}/remote_recap_worker.sh"
    _remote(args, ["mkdir", "-p", remote_bin])
    temporary = f"{remote_worker}.tmp-{os.getpid()}"
    _scp(args, str(local_worker), f"{args.host}:{temporary}")
    _remote(args, ["mv", temporary, remote_worker])
    _remote(args, ["chmod", "755", remote_worker])
    return remote_worker


def _resolve_checkpoint(checkpoint: Path) -> Path:
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


def _package_checkpoint(checkpoint: Path, archive: Path) -> None:
    checkpoint = _resolve_checkpoint(checkpoint)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    command = [
        "tar",
        "--zstd",
        "-cf",
        str(temporary),
        "--exclude=./train_state",
        "-C",
        str(checkpoint),
        ".",
    ]
    metadata = checkpoint / "robodojo_pi05_model.json"
    parent_metadata = checkpoint.parent / "robodojo_pi05_model.json"
    if not metadata.is_file() and parent_metadata.is_file():
        command.extend(["-C", str(checkpoint.parent), parent_metadata.name])
    _run(command)
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


def preflight(args: argparse.Namespace) -> None:
    _validate(args)
    _remote(
        args,
        [
            "bash",
            "-lc",
            f"command -v zstd >/dev/null && test -x {shlex.quote(args.remote_repo_root + '/scripts/robodojo.sh')}",
        ],
    )
    for gpu in sorted(set(args.gpu)):
        _remote(args, ["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"])
    if args.require_wcm:
        _remote(
            args,
            ["test", "-x", f"{args.remote_repo_root}/external_dependencies/WCM/.venv/bin/python"],
        )
    _install_worker(args)


def rollout(args: argparse.Namespace) -> None:
    _validate(args)
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
    worker = _install_worker(args)
    remote_checkpoint = f"{inbox}/policy.tar.zst"
    if not _remote_success(args, ["test", "-f", f"{job_root}/policy/.extracted"]):
        _package_checkpoint(checkpoint, archive)
        _remote(args, ["mkdir", "-p", inbox])
        _scp(args, str(archive), f"{args.host}:{remote_checkpoint}.tmp")
        _remote(args, ["mv", f"{remote_checkpoint}.tmp", remote_checkpoint])
    environment = {
        "RECAP_REMOTE_ACTION": "rollout",
        "RECAP_REMOTE_REPO_ROOT": args.remote_repo_root,
        "RECAP_REMOTE_JOB_ROOT": job_root,
        "RECAP_REMOTE_CHECKPOINT_ARCHIVE": remote_checkpoint,
        "RECAP_REMOTE_RESULT_ARCHIVE": result,
        "RECAP_REMOTE_TASK": args.task,
        "RECAP_REMOTE_EPISODES": str(args.episodes),
        "RECAP_REMOTE_LAYOUT_SEED": str(args.layout_seed),
        "RECAP_REMOTE_LAYOUT_OFFSET": str(args.layout_offset),
        "RECAP_REMOTE_POLICY_GPU": str(args.policy_gpu),
        "RECAP_REMOTE_ENV_GPU": str(args.env_gpu),
        "RECAP_REMOTE_ENV_CFG": args.env_cfg,
        "RECAP_REMOTE_ACTION_TYPE": args.action_type,
        "RECAP_REMOTE_POLICY_ENV": args.policy_env,
        "RECAP_REMOTE_EVAL_ENV": args.eval_env,
    }
    command = ["env", *(f"{key}={value}" for key, value in environment.items()), worker]
    _remote(args, command)
    _scp(args, f"{args.host}:{result}", str(result_local))
    _extract(result_local, local_output, "rollouts")
    archive.unlink(missing_ok=True)
    result_local.unlink(missing_ok=True)


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
    worker = _install_worker(args)
    wcm_archive = transfer_dir / f"{args.job_id}-wcm.pt.zst"
    _run(["zstd", "-q", "-f", args.wcm_checkpoint, "-o", str(wcm_archive)])
    remote_wcm = f"{inbox}/wcm.pt.zst"
    _remote(args, ["mkdir", "-p", inbox])
    _scp(args, str(wcm_archive), f"{args.host}:{remote_wcm}.tmp")
    _remote(args, ["mv", f"{remote_wcm}.tmp", remote_wcm])
    environment = {
        "RECAP_REMOTE_ACTION": "value-video",
        "RECAP_REMOTE_REPO_ROOT": args.remote_repo_root,
        "RECAP_REMOTE_JOB_ROOT": job_root,
        "RECAP_REMOTE_WCM_ARCHIVE": remote_wcm,
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
    command = ["env", *(f"{key}={value}" for key, value in environment.items()), worker]
    _remote(args, command)
    _scp(args, f"{args.host}:{result}", str(result_local))
    _extract(result_local, Path(args.output), "value_videos")
    wcm_archive.unlink(missing_ok=True)
    result_local.unlink(missing_ok=True)


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-repo-root", required=True)
    parser.add_argument("--remote-work-root", required=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    common(preflight_parser)
    preflight_parser.add_argument("--gpu", type=int, action="append", default=[])
    preflight_parser.add_argument("--require-wcm", action="store_true")
    preflight_parser.set_defaults(function=preflight)
    rollout_parser = subparsers.add_parser("rollout")
    common(rollout_parser)
    rollout_parser.add_argument("--job-id", required=True)
    rollout_parser.add_argument("--checkpoint", required=True)
    rollout_parser.add_argument("--output", required=True)
    rollout_parser.add_argument("--task", required=True)
    rollout_parser.add_argument("--episodes", type=int, required=True)
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
    arguments = parser.parse_args()
    arguments.function(arguments)
