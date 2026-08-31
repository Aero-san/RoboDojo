"""Run GPU-bound RECAP stages on a local or SSH-addressable host.

The local process owns the RECAP state machine.  This module only stages the
inputs for one stage, invokes the existing RoboDojo trainer through the shared
remote worker, and returns the stage output.  ``host: local`` (or localhost)
uses the same code path without SSH; any other host is treated as an SSH alias.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
from types import SimpleNamespace

try:  # Running as ``python scripts/posttrain/remote_training.py``.
    import g05_remote
    import remote_recap
except ModuleNotFoundError:  # Running as ``python -m scripts.posttrain.remote_training``.
    from . import g05_remote, remote_recap


MARKER_PREFIX = "@"


def _install_remote_wcm_support(args: argparse.Namespace) -> None:
    """Atomically install the tracked WCM adapters used by remote stages."""

    repo_root = Path(__file__).resolve().parents[2]
    remote_posttrain = f"{args.remote_repo_root}/scripts/posttrain"
    files = (
        ("run_wcm.sh", "run_wcm.sh"),
        ("run_wcm.py", "run_wcm.py"),
        ("wcm_checkpoint.py", "run_wcm.py"),
        ("annotate_recap_advantages.py", "annotate_recap_advantages.py"),
        ("render_rollout_value_videos.py", "render_rollout_value_videos.py"),
    )
    for name, reference_name in files:
        local_path = repo_root / "scripts/posttrain" / name
        remote_path = f"{remote_posttrain}/{name}"
        reference_path = f"{remote_posttrain}/{reference_name}"
        temporary = f"{remote_path}.tmp-{os.getpid()}"
        remote_recap._scp(args, str(local_path), f"{args.host}:{temporary}")
        remote_recap._remote(
            args,
            ["chown", "--reference", reference_path, temporary],
        )
        remote_recap._remote(
            args,
            ["chmod", "--reference", reference_path, temporary],
        )
        remote_recap._remote(args, ["mv", temporary, remote_path])
    print("[RECAP remote training] installed current WCM support", flush=True)


def _backend(args: argparse.Namespace, action: str = "run") -> SimpleNamespace:
    return SimpleNamespace(
        action=action,
        host=args.host,
        remote_repo_root=args.remote_repo_root,
        remote_work_root=args.remote_work_root,
        remote_zstd_bin=args.remote_zstd_bin,
        remote_conda_bin=args.remote_conda_bin,
        remote_python_bin=args.remote_python_bin,
        gpu_reservation=args.gpu_reservation,
        gpu_reservation_leave_free_mib=args.gpu_reservation_leave_free_mib,
        gpu_reservation_idle_used_max_mib=args.gpu_reservation_idle_used_max_mib,
        gpu_reservation_remote_max_hold_seconds=args.gpu_reservation_remote_max_hold_seconds,
        gpu=getattr(args, "gpu", []),
        require_wcm=getattr(args, "require_wcm", True),
    )


def _validate_host(args: argparse.Namespace) -> None:
    backend = _backend(args)
    remote_recap._validate(backend)
    if not args.host.strip() or any(character.isspace() for character in args.host):
        raise ValueError("--host must be a non-empty SSH alias or local")


def _remote_path(job_root: str, name: str) -> str:
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"remote input name must be relative and safe: {name!r}")
    return f"{job_root}/inputs/{name}"


def _upload_directory(backend: SimpleNamespace, local: Path, destination: str) -> None:
    local = local.expanduser().resolve()
    if not local.is_dir():
        raise FileNotFoundError(f"Remote stage directory input not found: {local}")
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix="robodojo-remote-stage-") as temporary:
        archive = Path(temporary) / "input.tar.zst"
        remote_recap._run(["tar", "--zstd", "-cf", str(archive), "-C", str(local), "."])
        remote_recap._remote(backend, ["rm", "-rf", str(destination)])
        remote_recap._remote(backend, ["mkdir", "-p", str(destination)])
        remote_archive = f"{destination}.tar.zst"
        remote_recap._scp(backend, str(archive), f"{backend.host}:{remote_archive}")
        remote_recap._remote(
            backend,
            ["tar", "--zstd", "-xf", remote_archive, "-C", str(destination)],
        )
        remote_recap._remote(backend, ["rm", "-f", remote_archive])


def _upload_file(backend: SimpleNamespace, local: Path, destination: str) -> None:
    local = local.expanduser().resolve()
    if not local.is_file():
        raise FileNotFoundError(f"Remote stage file input not found: {local}")
    remote_recap._remote(backend, ["mkdir", "-p", str(Path(destination).parent)])
    remote_recap._scp(backend, str(local), f"{backend.host}:{destination}.tmp")
    remote_recap._remote(backend, ["mv", f"{destination}.tmp", destination])


def _download_result(
    backend: SimpleNamespace,
    result_archive: str,
    output: Path,
    output_kind: str,
) -> None:
    output = output.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="robodojo-remote-result-") as temporary:
        archive = Path(temporary) / "result.tar.zst"
        remote_recap._scp(backend, f"{backend.host}:{result_archive}", str(archive))
        result = Path(temporary) / "result"
        remote_recap._extract(archive, result, "result")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output_kind == "directory":
            if output.exists():
                shutil.rmtree(output)
            shutil.copytree(result, output)
        else:
            candidates = list(result.iterdir())
            if len(candidates) != 1 or not candidates[0].is_file():
                raise RuntimeError(f"Remote result for {output} is not exactly one file")
            if output.exists():
                output.unlink()
            shutil.copy2(candidates[0], output)


def _expand(
    value: str,
    job_root: str,
    repo_root: str,
    inputs: dict[str, str],
    output: str,
) -> str:
    replacements = {"@repo": repo_root, "@job": job_root, "@output": output}
    for name, path in inputs.items():
        replacements[f"@input/{name}"] = path
    expanded = value
    for marker, replacement in sorted(replacements.items(), key=lambda item: -len(item[0])):
        expanded = expanded.replace(marker, replacement)
    if MARKER_PREFIX in expanded and "@" in expanded:
        raise ValueError(f"Unresolved remote stage marker in value: {value!r}")
    return expanded


def _stage_inputs(
    backend: SimpleNamespace,
    job_root: str,
    directory_inputs: dict[str, Path],
    file_inputs: dict[str, Path],
) -> dict[str, str]:
    remote_inputs: dict[str, str] = {}
    for name, local in directory_inputs.items():
        destination = _remote_path(job_root, name)
        _upload_directory(backend, local, destination)
        remote_inputs[name] = destination
    for name, local in file_inputs.items():
        destination = _remote_path(job_root, name)
        _upload_file(backend, local, destination)
        remote_inputs[name] = destination
    return remote_inputs


def _run_stage(
    args: argparse.Namespace,
    *,
    job_id: str,
    command: str,
    gpu_ids: str,
    directory_inputs: dict[str, Path],
    file_inputs: dict[str, Path],
    output: Path,
    output_kind: str,
    output_name: str | None = None,
    environment: dict[str, str] | None = None,
    resume_output: bool = False,
) -> None:
    if output_kind not in {"directory", "file"}:
        raise ValueError(f"Unsupported remote output kind: {output_kind}")
    if output_kind == "directory" and output.exists() and not output.is_dir():
        raise ValueError(f"Remote directory output is not a directory: {output}")
    if output_kind == "file" and output.exists() and not output.is_file():
        raise ValueError(f"Remote file output is not a file: {output}")
    if output_kind == "file" and not output_name:
        output_name = output.name

    backend = _backend(args)
    job_root = f"{args.remote_work_root}/jobs/{job_id}"
    result_archive = f"{job_root}/run.tar.zst"
    remote_output = f"{job_root}/output"
    if remote_recap._remote_success(backend, ["test", "-f", result_archive]):
        _download_result(backend, result_archive, output, output_kind)
        return

    try:
        remote_recap._reserve_remote_gpus(backend, job_root, [int(value) for value in gpu_ids.split(",")])
        remote_recap._remote(backend, ["mkdir", "-p", f"{job_root}/inputs"])
        if resume_output and output.exists():
            _upload_directory(backend, output, remote_output)
        else:
            remote_recap._remote(backend, ["rm", "-rf", remote_output])
            remote_recap._remote(backend, ["mkdir", "-p", remote_output])
        remote_inputs = _stage_inputs(backend, job_root, directory_inputs, file_inputs)
        expanded_command = _expand(command, job_root, args.remote_repo_root, remote_inputs, remote_output)
        expanded_environment = {
            name: _expand(value, job_root, args.remote_repo_root, remote_inputs, remote_output)
            for name, value in (environment or {}).items()
        }
        expanded_environment.update(
            {
                "RECAP_REMOTE_COMMAND": expanded_command,
                "RECAP_REMOTE_RESULT_ARCHIVE": result_archive,
                "RECAP_REMOTE_OUTPUT_PATH": remote_output
                if output_kind == "directory"
                else f"{remote_output}/{output_name}",
                "RECAP_REMOTE_OUTPUT_KIND": output_kind,
            }
        )
        worker = remote_recap._install_worker(backend)
        environment_for_worker = remote_recap._worker_environment(backend, job_root, "run")
        environment_for_worker.update(expanded_environment)
        remote_recap._invoke_worker(backend, worker, environment_for_worker, isolated_process_group=True)
        _download_result(backend, result_archive, output, output_kind)
    finally:
        remote_recap._cancel_remote_job(backend, job_root)


def _common_stage_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-repo-root", required=True)
    parser.add_argument("--remote-work-root", required=True)
    parser.add_argument("--remote-zstd-bin", default="zstd")
    parser.add_argument("--remote-conda-bin", default="conda")
    parser.add_argument("--remote-python-bin", default="python")
    parser.add_argument("--remote-policy-python", default="")
    parser.add_argument("--remote-wcm-python", default="")
    parser.add_argument("--gpu-reservation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-reservation-leave-free-mib", type=int, default=2048)
    parser.add_argument("--gpu-reservation-idle-used-max-mib", type=int, default=64)
    parser.add_argument("--gpu-reservation-remote-max-hold-seconds", type=int, default=1800)


def _add_stage_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id", required=True)


def _remote_python(args: argparse.Namespace, kind: str) -> str:
    if kind == "wcm":
        return args.remote_wcm_python or f"{args.remote_repo_root}/external_dependencies/WCM/.venv/bin/python"
    if args.remote_policy_python:
        return args.remote_policy_python
    if kind == "pi05":
        return f"{args.remote_repo_root}/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python"
    raise ValueError("G05 remote training requires training.remote.policy_python")


def run_wcm(args: argparse.Namespace) -> None:
    _install_remote_wcm_support(args)
    dataset = Path(args.dataset).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    init_checkpoint = Path(args.init_checkpoint).expanduser().resolve() if args.init_checkpoint else None
    output = Path(args.output).expanduser().resolve()
    remote_python = _remote_python(args, "wcm")
    command = (
        "bash " + shlex.quote(f"{args.remote_repo_root}/scripts/posttrain/run_wcm.sh") + ' --task "$WCM_TASK_NAME"'
    )
    environment = {
        "PYTHON_BIN": remote_python,
        "WCM_CONFIG": "@input/config",
        "WCM_DATASET_ROOT": "@input/dataset",
        "WCM_OUTPUT_DIR": "@output",
        "WCM_SUCCESS_LABELS": "@input/dataset/meta/success_labels.json",
        "WCM_ASSUME_SUCCESS": "0",
        "WCM_FAILURE_PENALTY": str(args.failure_penalty),
        "WCM_GAMMA": str(args.gamma),
        "WCM_EPOCHS": str(args.epochs),
        "WCM_NUM_WORKERS": str(args.num_workers),
        "WCM_PER_DEVICE_BATCH_SIZE": str(args.per_device_batch_size),
        "WCM_PRECISION": args.precision,
        "WCM_VIDEO_DECODER": args.video_decoder,
        "WCM_TASK_NAME": args.task,
        "CUDA_VISIBLE_DEVICES": args.gpus,
    }
    if args.learning_rate:
        environment["WCM_LR"] = args.learning_rate
    if args.warmup_steps:
        environment["WCM_WARMUP_STEPS"] = args.warmup_steps
    if init_checkpoint:
        environment["WCM_INIT_CHECKPOINT"] = "@input/init_wcm"
    if args.resume:
        environment["WCM_RESUME"] = "@output/checkpoints/last.pt"
    _run_stage(
        args,
        job_id=args.job_id,
        command=command,
        gpu_ids=args.gpus,
        directory_inputs={"dataset": dataset},
        file_inputs={"config": config, **({"init_wcm": init_checkpoint} if init_checkpoint else {})},
        output=output,
        output_kind="directory",
        environment=environment,
        resume_output=args.resume,
    )


def run_advantages(args: argparse.Namespace) -> None:
    _install_remote_wcm_support(args)
    buffer = Path(args.buffer).expanduser().resolve()
    checkpoint = Path(args.wcm_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    remote_python = _remote_python(args, "wcm")
    gpu_count = len(args.gpus.split(","))
    launcher = [remote_python]
    if gpu_count > 1:
        launcher.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpu_count}",
            ]
        )
    command = shlex.join(
        [
            *launcher,
            f"{args.remote_repo_root}/scripts/posttrain/annotate_recap_advantages.py",
            "--wcm-checkpoint",
            "@input/wcm_checkpoint",
            "--dataset-root",
            "@input/buffer",
            "--output",
            "@output/" + output.name,
            "--task",
            args.task,
            "--lookahead",
            str(args.lookahead),
            "--gamma",
            str(args.gamma),
            "--failure-penalty",
            str(args.failure_penalty),
            "--positive-fraction",
            str(args.positive_fraction),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--expected-world-size",
            str(gpu_count),
        ]
    )
    _run_stage(
        args,
        job_id=args.job_id,
        command=command,
        gpu_ids=args.gpus,
        directory_inputs={"buffer": buffer},
        file_inputs={"wcm_checkpoint": checkpoint},
        output=output,
        output_kind="file",
        environment={"CUDA_VISIBLE_DEVICES": args.gpus},
    )


def _rewrite_pi05_args(args: argparse.Namespace) -> list[str]:
    values = list(args.train_arg)
    rewritten: list[str] = []
    index = 0
    replacements = {
        "--openpi-root": "@repo/XPolicyLab/policy/Pi_05/openpi",
        "--checkpoint-dir": "@output",
        "--norm-stats-dir": "@input/norm_stats",
        "--init-checkpoint": "@input/init_policy",
    }
    while index < len(values):
        value = values[index]
        rewritten.append(value)
        if value in replacements:
            index += 1
            if index >= len(values):
                raise ValueError(f"Missing value after Pi0.5 option: {value}")
            rewritten.append(replacements[value])
        index += 1
    return rewritten


def run_pi05(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    norm_stats = Path(args.norm_stats).expanduser().resolve()
    init_policy = Path(args.init_policy).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    train_args = _rewrite_pi05_args(args)
    repo_id = ""
    for index, value in enumerate(train_args[:-1]):
        if value == "--repo-id":
            repo_id = train_args[index + 1]
            break
    if not repo_id or Path(repo_id).name != repo_id:
        raise ValueError("Pi0.5 train args must contain a simple --repo-id value")
    command = shlex.join(
        [_remote_python(args, "pi05"), f"{args.remote_repo_root}/scripts/posttrain/train_pi05.py", *train_args]
    )
    environment = {
        "HF_LEROBOT_HOME": "@job/inputs/lerobot",
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": args.xla_memory_fraction,
    }
    _run_stage(
        args,
        job_id=args.job_id,
        command=command,
        gpu_ids=args.gpus,
        directory_inputs={"lerobot/" + repo_id: dataset, "norm_stats": norm_stats, "init_policy": init_policy},
        file_inputs={},
        output=output,
        output_kind="directory",
        environment=environment,
        resume_output=args.resume,
    )


def run_render(args: argparse.Namespace) -> None:
    _install_remote_wcm_support(args)
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    checkpoint = Path(args.wcm_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    command = shlex.join(
        [
            _remote_python(args, "wcm"),
            f"{args.remote_repo_root}/scripts/posttrain/render_rollout_value_videos.py",
            "--wcm-checkpoint",
            "@input/wcm_checkpoint",
            "--rollout-root",
            "@input/rollouts",
            "--output-dir",
            "@output",
            "--max-episodes",
            str(args.episodes),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--backend",
            args.backend,
            "--speed",
            str(args.speed),
            "--y-min",
            str(args.y_min),
            "--y-max",
            str(args.y_max),
            "--title",
            args.title,
        ]
    )
    _run_stage(
        args,
        job_id=args.job_id,
        command=command,
        gpu_ids=str(args.gpu),
        directory_inputs={"rollouts": rollout_root},
        file_inputs={"wcm_checkpoint": checkpoint},
        output=output,
        output_kind="directory",
        environment={"CUDA_VISIBLE_DEVICES": str(args.gpu)},
    )


def preflight(args: argparse.Namespace) -> None:
    backend = _backend(args, action="preflight")
    remote_recap.preflight(backend)
    _install_remote_wcm_support(args)
    checks = (
        (
            f"{args.policy} Python",
            _remote_python(args, args.policy),
            f"Set training.remote.policy_python to an executable Python on {args.host}.",
        ),
        (
            "WCM Python",
            _remote_python(args, "wcm"),
            f"Set training.remote.wcm_python to an executable Python on {args.host}.",
        ),
    )
    failures: list[str] = []
    for label, executable, fix in checks:
        result = remote_recap._remote_result(backend, ["test", "-x", executable])
        if result.returncode == 0:
            print(f"[RECAP remote training] OK: {label} ({executable})", flush=True)
        else:
            failures.append(f"{label} is not executable: {executable}\n  Fix: {fix}")
    policy_trainer = (
        ("Pi0.5 trainer", f"{args.remote_repo_root}/scripts/posttrain/train_pi05.py")
        if args.policy == "pi05"
        else ("G05 trainer", f"{args.remote_repo_root}/scripts/posttrain/train_g05.py")
    )
    trainer_checks = [
        ("WCM trainer", f"{args.remote_repo_root}/scripts/posttrain/run_wcm.sh"),
        policy_trainer,
    ]
    if args.policy == "g05":
        trainer_checks.append(("G05 upstream trainer", f"{args.g05_root}/scripts/finetune.py"))
    for label, path in trainer_checks:
        result = remote_recap._remote_result(backend, ["sh", "-c", 'test -f "$1" && test -r "$1"', "sh", path])
        if result.returncode == 0:
            print(f"[RECAP remote training] OK: {label} ({path})", flush=True)
        else:
            failures.append(
                f"{label} is missing or unreadable: {path}\n  Fix: synchronize the remote RoboDojo checkout."
            )
    if failures:
        raise RuntimeError("Remote training preflight failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _common_stage_parser(preflight_parser)
    preflight_parser.add_argument("--gpu", action="append", type=int, default=[])
    preflight_parser.add_argument("--policy", choices=("pi05", "g05"), required=True)
    preflight_parser.add_argument("--g05-root", default="")
    preflight_parser.set_defaults(function=preflight)

    wcm_parser = subparsers.add_parser("wcm")
    _common_stage_parser(wcm_parser)
    _add_stage_identity(wcm_parser)
    wcm_parser.add_argument("--dataset", required=True)
    wcm_parser.add_argument("--config", required=True)
    wcm_parser.add_argument("--output", required=True)
    wcm_parser.add_argument("--task", required=True)
    wcm_parser.add_argument("--gpus", required=True)
    wcm_parser.add_argument("--epochs", required=True)
    wcm_parser.add_argument("--num-workers", required=True)
    wcm_parser.add_argument("--per-device-batch-size", required=True)
    wcm_parser.add_argument("--precision", required=True)
    wcm_parser.add_argument("--video-decoder", required=True)
    wcm_parser.add_argument("--failure-penalty", required=True)
    wcm_parser.add_argument("--gamma", required=True)
    wcm_parser.add_argument("--learning-rate", default="")
    wcm_parser.add_argument("--warmup-steps", default="")
    wcm_parser.add_argument("--init-checkpoint", default="")
    wcm_parser.add_argument("--resume", action="store_true")
    wcm_parser.set_defaults(function=run_wcm)

    advantage_parser = subparsers.add_parser("advantages")
    _common_stage_parser(advantage_parser)
    _add_stage_identity(advantage_parser)
    advantage_parser.add_argument("--buffer", required=True)
    advantage_parser.add_argument("--wcm-checkpoint", required=True)
    advantage_parser.add_argument("--output", required=True)
    advantage_parser.add_argument("--task", required=True)
    advantage_parser.add_argument("--gpus", required=True)
    advantage_parser.add_argument("--lookahead", required=True)
    advantage_parser.add_argument("--gamma", required=True)
    advantage_parser.add_argument("--failure-penalty", required=True)
    advantage_parser.add_argument("--positive-fraction", required=True)
    advantage_parser.add_argument("--batch-size", required=True)
    advantage_parser.add_argument("--num-workers", required=True)
    advantage_parser.add_argument("--device", required=True)
    advantage_parser.set_defaults(function=run_advantages)

    pi_parser = subparsers.add_parser("pi05")
    _common_stage_parser(pi_parser)
    _add_stage_identity(pi_parser)
    pi_parser.add_argument("--dataset", required=True)
    pi_parser.add_argument("--norm-stats", required=True)
    pi_parser.add_argument("--init-policy", required=True)
    pi_parser.add_argument("--output", required=True)
    pi_parser.add_argument("--gpus", required=True)
    pi_parser.add_argument("--xla-memory-fraction", required=True)
    pi_parser.add_argument("--train-arg", action="append", default=[])
    pi_parser.add_argument("--resume", action="store_true")
    pi_parser.set_defaults(function=run_pi05)

    render_parser = subparsers.add_parser("render")
    _common_stage_parser(render_parser)
    _add_stage_identity(render_parser)
    render_parser.add_argument("--rollout-root", required=True)
    render_parser.add_argument("--wcm-checkpoint", required=True)
    render_parser.add_argument("--output", required=True)
    render_parser.add_argument("--episodes", required=True)
    render_parser.add_argument("--gpu", required=True)
    render_parser.add_argument("--batch-size", required=True)
    render_parser.add_argument("--device", required=True)
    render_parser.add_argument("--precision", required=True)
    render_parser.add_argument("--backend", required=True)
    render_parser.add_argument("--speed", required=True)
    render_parser.add_argument("--y-min", required=True)
    render_parser.add_argument("--y-max", required=True)
    render_parser.add_argument("--title", required=True)
    g05_remote.add_parser(subparsers, _common_stage_parser, _add_stage_identity, _run_stage)

    render_parser.set_defaults(function=run_render)

    args = parser.parse_args()
    if args.gpu_reservation_leave_free_mib < 256:
        parser.error("--gpu-reservation-leave-free-mib must be at least 256")
    if args.gpu_reservation_idle_used_max_mib < 0:
        parser.error("--gpu-reservation-idle-used-max-mib cannot be negative")
    if args.gpu_reservation_remote_max_hold_seconds < 60:
        parser.error("--gpu-reservation-remote-max-hold-seconds must be at least 60")

    def stop_on_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, stop_on_signal)
    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        _validate_host(args)
        args.function(args)
    except KeyboardInterrupt:
        parser.exit(130, "remote_training: interrupted; remote job cleanup completed\n")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"remote_training: {error}\n")


if __name__ == "__main__":
    main()
