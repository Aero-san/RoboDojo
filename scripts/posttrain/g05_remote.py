"""Remote-stage adapter for G05 RECAP training."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import shlex
from typing import Callable

try:
    import remote_recap
except ModuleNotFoundError:
    from . import remote_recap


def _run(args: argparse.Namespace, run_stage: Callable[..., None]) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    bundle, checkpoint = remote_recap._g05_bundle(Path(args.init_policy))
    output = Path(args.output).expanduser().resolve()
    checkpoint_name = checkpoint.name
    command = shlex.join(
        [
            args.remote_policy_python,
            f"{args.remote_repo_root}/scripts/posttrain/train_g05.py",
            "--g05-root",
            args.g05_root,
            "--dataset",
            "@input/dataset",
            "--output",
            "@output",
            "--init-checkpoint",
            f"@input/init_policy/checkpoints/{checkpoint_name}",
            "--processor-path",
            args.processor_path,
            "--task-config",
            args.task_config,
            "--experiment-name",
            args.experiment_name,
            "--gpus",
            str(len(args.gpus.split(","))),
            "--steps",
            str(args.steps),
            "--save-interval",
            str(args.save_interval),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--grad-accumulation-steps",
            str(args.grad_accumulation_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--warmup-steps",
            str(args.warmup_steps),
            "--decay-learning-rate",
            str(args.decay_learning_rate),
            "--decay-start-ratio",
            str(args.decay_start_ratio),
            "--weight-decay",
            str(args.weight_decay),
            "--wandb" if args.wandb else "--no-wandb",
            *(["--resume"] if args.resume else []),
        ]
    )
    run_stage(
        args,
        job_id=args.job_id,
        command=command,
        gpu_ids=args.gpus,
        directory_inputs={"dataset": dataset},
        file_inputs={
            f"init_policy/checkpoints/{checkpoint_name}": checkpoint,
            "init_policy/.hydra/config.yaml": bundle / ".hydra/config.yaml",
            "init_policy/dataset_stats.json": bundle / "dataset_stats.json",
            "init_policy/action_tokenizer.pt": bundle / "action_tokenizer.pt",
        },
        output=output,
        output_kind="directory",
        environment={"CUDA_VISIBLE_DEVICES": args.gpus},
        resume_output=args.resume,
    )


def add_parser(subparsers, common_parser, stage_identity, run_stage) -> None:
    parser = subparsers.add_parser("g05")
    common_parser(parser)
    stage_identity(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--init-policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--g05-root", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--experiment-name", required=True)
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
    parser.set_defaults(function=partial(_run, run_stage=run_stage))
