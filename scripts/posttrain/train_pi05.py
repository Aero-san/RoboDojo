"""Run OpenPI Pi0.5 training with explicit parameter-selection policies.

The OpenPI checkout remains the source of the model and optimizer.  This
launcher only constructs its ``TrainConfig`` so RoboDojo can select a useful
subset of Pi0.5 parameters without editing the XPolicyLab submodule.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from types import ModuleType

import numpy as np


@dataclasses.dataclass(frozen=True)
class RecapTokenizePrompt:
    """Route one labeled RECAP sample to its CFG training prompt."""

    tokenizer: object
    discrete_state_input: bool = False
    unconditional_prob: float = 0.1

    def __call__(self, data):
        prompt = data.get("prompt")
        if prompt is None:
            raise ValueError("RECAP requires a prompt with an advantage condition.")
        prompt = str(np.asarray(prompt).item())
        base_prompt = re.sub(
            r"\s*Advantage:\s*(positive|negative)\s*$",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        if base_prompt == prompt:
            raise ValueError(f"RECAP prompt has no binary advantage condition: {prompt!r}")
        is_positive = bool(re.search(r"Advantage:\s*positive\s*$", prompt, flags=re.IGNORECASE))
        # positive_only_conditional routing from RECAP: negative samples are
        # always unconditional; positive samples are dropped to unconditional
        # with the configured CFG regularization probability.
        routed_prompt = (
            prompt
            if is_positive and np.random.random() >= self.unconditional_prob
            else base_prompt
        )
        state = data.get("state") if self.discrete_state_input else None
        if self.discrete_state_input and state is None:
            raise ValueError("Pi0.5 RECAP tokenization requires the normalized robot state.")
        tokens, mask = self.tokenizer.tokenize(routed_prompt, state)
        result = {key: value for key, value in data.items() if key != "prompt"}
        return {
            **result,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
        }


@dataclasses.dataclass(frozen=True)
class PosttrainDataConfig:
    """Add robot-specific action transforms and RECAP prompt conditioning."""

    base: object
    recap: bool = False
    recap_unconditional_prob: float = 0.1
    delta_action_mask: tuple[bool, ...] | None = None
    norm_stats_dir: str = ""

    def create(self, assets_dirs, model_config):
        import openpi.transforms as transforms

        data = self.base.create(assets_dirs, model_config)
        if self.delta_action_mask is not None:
            data_transforms = data.data_transforms.push(
                inputs=[transforms.DeltaActions(self.delta_action_mask)],
                outputs=[transforms.AbsoluteActions(self.delta_action_mask)],
            )
            data = dataclasses.replace(data, data_transforms=data_transforms)
        if self.recap:
            inputs = []
            replaced = False
            for transform in data.model_transforms.inputs:
                if isinstance(transform, transforms.TokenizePrompt):
                    inputs.append(
                        RecapTokenizePrompt(
                            transform.tokenizer,
                            transform.discrete_state_input,
                            self.recap_unconditional_prob,
                        )
                    )
                    replaced = True
                else:
                    inputs.append(transform)
            if not replaced:
                raise TypeError("Pi0.5 RECAP requires OpenPI's TokenizePrompt transform.")
            model_transforms = transforms.Group(
                inputs=tuple(inputs),
                outputs=data.model_transforms.outputs,
            )
            data = dataclasses.replace(data, model_transforms=model_transforms)
        if self.norm_stats_dir:
            import openpi.shared.normalize as normalize

            data = dataclasses.replace(data, norm_stats=normalize.load(self.norm_stats_dir))
        return data


def _load_train_module(openpi_root: Path) -> ModuleType:
    train_path = openpi_root / "scripts" / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"OpenPI trainer not found: {train_path}")
    sys.path.insert(0, str(openpi_root / "src"))
    spec = importlib.util.spec_from_file_location("robodojo_openpi_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenPI trainer: {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_and_freeze_filter(config, mode: str, action_expert_variant: str | None, paligemma_variant: str | None):
    import flax.nnx as nnx
    import openpi.models.pi0_config as pi0_config
    import openpi.shared.nnx_utils as nnx_utils

    if not isinstance(config.model, pi0_config.Pi0Config) or not config.model.pi05:
        raise ValueError("The selected OpenPI train config must be a Pi0.5 Pi0Config.")

    model = config.model
    if action_expert_variant:
        model = dataclasses.replace(model, action_expert_variant=action_expert_variant)
    if paligemma_variant:
        model = dataclasses.replace(model, paligemma_variant=paligemma_variant)

    # Pi0.5's action expert consists of the second Gemma stream plus the
    # action/timestep input and output projections owned by Pi0 itself.
    action_expert_path = ".*(llm.*_1|action_in_proj|time_mlp_in|time_mlp_out|action_out_proj).*"
    action_expert_lora_path = ".*(llm.*_1.*lora|action_in_proj|time_mlp_in|time_mlp_out|action_out_proj).*"
    paligemma_lora_path = ".*llm(?!.*_1).*lora.*"
    lora_path = ".*lora.*"
    if mode == "full":
        freeze_filter = nnx.Nothing
    elif mode == "action_expert":
        # Train the complete flow-matching action expert and freeze vision,
        # PaliGemma and all other policy parameters.
        freeze_filter = nnx.Not(nnx_utils.PathRegex(action_expert_path))
    elif mode == "action_expert_lora":
        model = dataclasses.replace(
            model,
            action_expert_variant=action_expert_variant or "gemma_300m_lora",
        )
        freeze_filter = nnx.Not(nnx_utils.PathRegex(action_expert_lora_path))
    elif mode == "paligemma_lora":
        model = dataclasses.replace(
            model,
            paligemma_variant=paligemma_variant or "gemma_2b_lora",
        )
        freeze_filter = nnx.Not(nnx_utils.PathRegex(paligemma_lora_path))
    elif mode == "all_lora":
        model = dataclasses.replace(
            model,
            paligemma_variant=paligemma_variant or "gemma_2b_lora",
            action_expert_variant=action_expert_variant or "gemma_300m_lora",
        )
        freeze_filter = nnx.Not(nnx_utils.PathRegex(lora_path))
    else:
        raise ValueError(f"Unsupported Pi0.5 fine-tune mode: {mode}")
    return model, freeze_filter


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--finetune-mode",
        choices=("full", "action_expert", "action_expert_lora", "paligemma_lora", "all_lora"),
        default="action_expert_lora",
    )
    parser.add_argument("--action-expert-variant", default="")
    parser.add_argument("--paligemma-variant", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-steps", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--decay-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=-1.0)
    parser.add_argument("--clip-gradient-norm", type=float, default=0.0)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--recap", action="store_true")
    parser.add_argument("--recap-unconditional-prob", type=float, default=0.1)
    parser.add_argument("--recap-guidance-scale", type=float, default=1.0)
    parser.add_argument("--env-cfg-type", default="")
    parser.add_argument("--action-type", choices=("joint", "ee"), default="joint")
    parser.add_argument("--norm-stats-dir", default="")
    return parser


def _robot_delta_mask(env_cfg_type: str, action_type: str) -> tuple[bool, ...] | None:
    if not env_cfg_type or action_type != "joint":
        return None
    import yaml

    root = Path(__file__).resolve().parents[2]
    env_cfg = yaml.safe_load((root / "env_cfg" / f"{env_cfg_type}.yml").read_text(encoding="utf-8"))
    robot_name = env_cfg["config"]["robot"]
    robot_info = json.loads((root / "env_cfg" / "robot" / "_robot_info.json").read_text(encoding="utf-8"))[
        robot_name
    ]
    mask: list[bool] = []
    for arm_dim, ee_dim in zip(robot_info["arm_dim"], robot_info["ee_dim"], strict=True):
        mask.extend([True] * int(arm_dim))
        mask.extend([False] * int(ee_dim))
    return tuple(mask)


def _robot_action_dim(env_cfg_type: str, action_type: str) -> int | None:
    if not env_cfg_type:
        return None
    import yaml

    root = Path(__file__).resolve().parents[2]
    env_cfg = yaml.safe_load((root / "env_cfg" / f"{env_cfg_type}.yml").read_text(encoding="utf-8"))
    robot_name = env_cfg["config"]["robot"]
    robot_info = json.loads((root / "env_cfg" / "robot" / "_robot_info.json").read_text(encoding="utf-8"))[
        robot_name
    ]
    arm_dim = sum(robot_info["arm_dim"]) if action_type == "joint" else 7 * len(robot_info["arm_dim"])
    return int(arm_dim + sum(robot_info["ee_dim"]))


def _write_checkpoint_model_metadata(
    checkpoint_dir: str | Path,
    model,
    finetune_mode: str,
    *,
    recap: bool,
    recap_unconditional_prob: float,
    recap_guidance_scale: float,
) -> None:
    root = Path(checkpoint_dir).expanduser().resolve()
    candidates = [root]
    if root.exists():
        candidates.extend(
            child for child in root.iterdir() if child.is_dir() and (child / "params").exists()
        )
    metadata = {
        "schema_version": 1,
        "model_type": "pi05",
        "finetune_mode": finetune_mode,
        "paligemma_variant": model.paligemma_variant,
        "action_expert_variant": model.action_expert_variant,
        "recap_conditioning": bool(recap),
        "recap_inference_condition": "positive" if recap else None,
        "recap_unconditional_prob": recap_unconditional_prob if recap else None,
        "recap_guidance_scale": recap_guidance_scale if recap else None,
    }
    for candidate in candidates:
        if (candidate / "params").exists():
            (candidate / "robodojo_pi05_model.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )


def main(args: argparse.Namespace) -> None:
    if not 0.0 <= args.recap_unconditional_prob <= 1.0:
        raise ValueError("--recap-unconditional-prob must be in [0, 1].")
    if args.recap and args.recap_guidance_scale != 1.0:
        raise ValueError(
            "This OpenPI integration supports RECAP guidance scale 1.0. Other values "
            "require two policy branches at every denoising step."
        )
    openpi_root = Path(args.openpi_root).expanduser().resolve()
    train_module = _load_train_module(openpi_root)
    import openpi.training.config as config_lib

    config = config_lib.get_config(args.train_config_name)
    model, freeze_filter = _model_and_freeze_filter(
        config,
        args.finetune_mode,
        args.action_expert_variant or None,
        args.paligemma_variant or None,
    )
    data = dataclasses.replace(config.data, repo_id=args.repo_id)
    output_action_dim = _robot_action_dim(args.env_cfg_type, args.action_type)
    if output_action_dim is not None and hasattr(data, "output_action_dim"):
        data = dataclasses.replace(data, output_action_dim=output_action_dim)
    delta_action_mask = _robot_delta_mask(args.env_cfg_type, args.action_type)
    if delta_action_mask is not None and hasattr(data, "use_delta_joint_actions"):
        data = dataclasses.replace(data, use_delta_joint_actions=False)
    if args.recap or delta_action_mask is not None or args.norm_stats_dir:
        data = PosttrainDataConfig(
            data,
            recap=args.recap,
            recap_unconditional_prob=args.recap_unconditional_prob,
            delta_action_mask=delta_action_mask,
            norm_stats_dir=args.norm_stats_dir,
        )
    updates = {
        "model": model,
        "freeze_filter": freeze_filter,
        "data": data,
        "exp_name": args.exp_name,
        "checkpoint_dir_override": args.checkpoint_dir,
        "seed": args.seed,
        "fsdp_devices": args.fsdp_devices,
        "overwrite": not args.resume,
        "resume": args.resume,
    }
    if args.init_checkpoint:
        if args.resume:
            raise ValueError("--init-checkpoint and --resume are mutually exclusive.")
        import openpi.training.weight_loaders as weight_loaders

        checkpoint = Path(args.init_checkpoint).expanduser().resolve()
        if not (checkpoint / "params").exists():
            candidates = [child for child in checkpoint.iterdir() if child.is_dir() and (child / "params").exists()]
            if not candidates:
                raise FileNotFoundError(f"No OpenPI params found below --init-checkpoint={checkpoint}")
            checkpoint = max(
                candidates,
                key=lambda path: int(path.name) if path.name.isdigit() else -1,
            )
        updates["weight_loader"] = weight_loaders.CheckpointWeightLoader(str(checkpoint / "params"))
    if args.batch_size > 0:
        updates["batch_size"] = args.batch_size
    if args.num_workers > 0:
        updates["num_workers"] = args.num_workers
    if args.num_train_steps > 0:
        updates["num_train_steps"] = args.num_train_steps
    if args.save_interval > 0:
        updates["save_interval"] = args.save_interval
    if args.log_interval > 0:
        updates["log_interval"] = args.log_interval
    if args.disable_wandb:
        updates["wandb_enabled"] = False

    if args.learning_rate > 0 or args.warmup_steps > 0 or args.decay_lr > 0:
        schedule = config.lr_schedule
        schedule_updates = {}
        if args.learning_rate > 0:
            schedule_updates["peak_lr"] = args.learning_rate
        if args.warmup_steps > 0:
            schedule_updates["warmup_steps"] = args.warmup_steps
        if args.decay_lr > 0:
            schedule_updates["decay_lr"] = args.decay_lr
        updates["lr_schedule"] = dataclasses.replace(schedule, **schedule_updates)
    if args.weight_decay >= 0 or args.clip_gradient_norm > 0:
        optimizer = config.optimizer
        optimizer_updates = {}
        if args.weight_decay >= 0:
            optimizer_updates["weight_decay"] = args.weight_decay
        if args.clip_gradient_norm > 0:
            optimizer_updates["clip_gradient_norm"] = args.clip_gradient_norm
        updates["optimizer"] = dataclasses.replace(optimizer, **optimizer_updates)

    config = dataclasses.replace(config, **updates)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "Starting Pi0.5 fine-tuning: mode=%s repo_id=%s checkpoint_dir=%s",
        args.finetune_mode,
        args.repo_id,
        args.checkpoint_dir,
    )
    train_module.main(config)
    _write_checkpoint_model_metadata(
        config.checkpoint_dir,
        config.model,
        args.finetune_mode,
        recap=args.recap,
        recap_unconditional_prob=args.recap_unconditional_prob,
        recap_guidance_scale=args.recap_guidance_scale,
    )


if __name__ == "__main__":
    main(_make_parser().parse_args())
