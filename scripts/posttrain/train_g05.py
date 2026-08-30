"""Launch upstream G05 training with RoboDojo-owned RECAP configs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def _checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"G05 checkpoint does not exist: {path}")
    candidates = [path / "last.pt", path / "checkpoints/checkpoint"]
    candidates.extend(sorted(path.glob("checkpoints/step_*.pt"), key=_step))
    candidates.extend(sorted(path.glob("**/model_state_dict.pt")))
    for candidate in reversed(candidates):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No G05 checkpoint file below: {path}")


def _step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return -1


def _sidecar(checkpoint: Path, name: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"G05 {name} does not exist: {path}")
    for parent in (checkpoint.parent, *checkpoint.parents):
        candidate = parent / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No {name} beside G05 checkpoint: {checkpoint}")


def _portable_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _latest_resume(output: Path) -> Path:
    candidates = sorted((output / "checkpoints").glob("step_*.pt"), key=_step)
    candidates.extend(path for path in (output / "last.pt",) if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No resumable G05 checkpoint below: {output}")
    return candidates[-1]


def main(args: argparse.Namespace) -> None:
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.decay_learning_rate < 0 or args.decay_learning_rate > args.learning_rate:
        raise ValueError("--decay-learning-rate must be between zero and --learning-rate.")
    if not 0 < args.decay_start_ratio < 1:
        raise ValueError("--decay-start-ratio must be in (0, 1).")
    if int(args.steps * args.decay_start_ratio) < args.warmup_steps:
        raise ValueError("--decay-start-ratio starts decay before warmup completes.")
    lr_min_ratio = args.decay_learning_rate / args.learning_rate

    root = Path(args.g05_root).expanduser().resolve()
    trainer = root / "scripts/finetune.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"Upstream G05 trainer is missing: {trainer}")
    guarded_entrypoint = Path(__file__).with_name("g05_finetune_entry.py").resolve()
    if not guarded_entrypoint.is_file():
        raise FileNotFoundError(
            f"RoboDojo G05 finetune entrypoint is missing: {guarded_entrypoint}"
        )
    dataset = Path(args.dataset).expanduser().resolve()
    if json.loads((dataset / "meta/info.json").read_text())["codebase_version"] != "v3.0":
        raise ValueError("G05 training accepts only a LeRobot v3.0 dataset.")
    output = Path(args.output).expanduser().resolve()
    initial = _checkpoint(Path(args.init_checkpoint))
    stats = _sidecar(initial, "dataset_stats.json", args.dataset_stats)
    tokenizer = _sidecar(initial, "action_tokenizer.pt", args.action_tokenizer)
    processor = Path(args.processor_path).expanduser()
    if not processor.is_absolute():
        processor = root / processor
    if not processor.is_dir():
        raise FileNotFoundError(f"G05 processor directory is missing: {processor}")

    config_dir = Path(__file__).resolve().parents[2] / "configs/g05"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(args.gpus),
        str(guarded_entrypoint),
        "--config-dir",
        str(config_dir),
        f"task={args.task_config}",
        f"hydra.run.dir={output}",
        f"exp_name={args.experiment_name}",
        f"model.max_steps={args.steps}",
        "model.max_epochs=null",
        f"model.batch_size={args.batch_size}",
        f"model.num_workers={args.num_workers}",
        f"model.grad_accumulation_steps={args.grad_accumulation_steps}",
        f"model.learning_rate={args.learning_rate}",
        f"model.warmup_steps={args.warmup_steps}",
        "model.lr_scheduler_type=warmup_constant_cosine",
        f"model.lr_min_ratio={lr_min_ratio:.12g}",
        f"model.constant_end_ratio={args.decay_start_ratio}",
        f"model.weight_decay={args.weight_decay}",
        f"model.model_arch.hf_processor_path={processor}",
        f"tokenizer.vq_config.ckpt_dir={tokenizer}",
        f"checkpointing_steps={args.save_interval}",
        f"logger.mode={'online' if args.wandb else 'disabled'}",
    ]
    if args.resume:
        command.append(f"resume_ckpt={_latest_resume(output)}")
    else:
        command.append(f"model.pretrained_ckpt={initial}")

    environment = dict(os.environ)
    environment.update(
        {
            "ROBODOJO_RECAP_DATASET": str(dataset),
            "ROBODOJO_G05_DATA_STATS": str(stats),
            "ROBODOJO_G05_FINETUNE_SCRIPT": str(trainer),
            "ROBODOJO_G05_MAX_GETITEM_ATTEMPTS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(root / "src"), str(root), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("[G05 RECAP] " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
    _portable_copy(stats, output / "dataset_stats.json")
    _portable_copy(tokenizer, output / "action_tokenizer.pt")
    (output / "robodojo_g05_model.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "g05",
                "recap_conditioning": True,
                "recap_inference_condition": "positive",
                "steps": args.steps,
                "optimizer": {
                    "learning_rate": args.learning_rate,
                    "decay_learning_rate": args.decay_learning_rate,
                    "decay_start_ratio": args.decay_start_ratio,
                    "scheduler": "warmup_constant_cosine",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g05-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--dataset-stats", default="")
    parser.add_argument("--action-tokenizer", default="")
    parser.add_argument("--processor-path", default="checkpoints/qwen3_5_2b_base_processor")
    parser.add_argument("--task-config", default="robodojo_recap")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--grad-accumulation-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--decay-learning-rate", type=float, required=True)
    parser.add_argument("--decay-start-ratio", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
