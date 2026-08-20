"""Launch the official WCM trainer/evaluator with the RoboDojo adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _argument_value(args: list[str], name: str) -> str | None:
    for index, value in enumerate(args):
        if value == name and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return None


def _configured_epochs(args: list[str]) -> int | None:
    override = os.environ.get("WCM_EPOCHS")
    if override:
        try:
            return int(override)
        except ValueError:
            # Let the official WCM runtime-overrides parser report the same
            # invalid value using its normal error path.
            return None
    config_path = _argument_value(args, "--config")
    if not config_path:
        return None
    try:
        from world_critic.config import load_config

        return int(load_config(config_path).epochs)
    except (OSError, TypeError, ValueError, ImportError):
        # The official parser remains the source of truth.  A missing or
        # malformed config will still produce its original error message.
        return None


def _install_ddp_model_compatibility() -> None:
    """Exclude the unused Hugging Face ViT pooler from DDP synchronization.

    WCM consumes ``last_hidden_state`` and never consumes ViT's pooled output.
    Some ViT checkpoints still register a trainable ``pooler.dense`` module,
    which produces no gradient and causes DDP's default unused-parameter check
    to fail on the next iteration.  Keep this compatibility patch in the
    tracked GR00T adapter so a clean official WCM submodule gets the same fix.
    """

    from world_critic.model import VisionEncoder

    original_init = VisionEncoder.__init__
    if getattr(original_init, "_gr00t_pooler_compat", False):
        return

    def patched_init(self, config):
        original_init(self, config)
        pooler = getattr(self.backbone, "pooler", None)
        if pooler is not None:
            pooler.requires_grad_(False)

    patched_init._gr00t_pooler_compat = True
    VisionEncoder.__init__ = patched_init


def _install_initial_checkpoint(command) -> None:
    """Initialize a fresh WCM optimization run from prior deploy weights.

    Official resume checkpoints also restore the old optimizer, epoch count,
    and split manifest, which is incorrect after appending rollout episodes.
    RECAP iterations start a fresh optimizer and data split while carrying
    model parameters and their action-normalization coordinate system forward.
    """

    checkpoint_path = os.environ.get("WCM_INIT_CHECKPOINT", "").strip()
    if not checkpoint_path:
        return
    import torch
    from world_critic.training import config_from_checkpoint_payload

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"WCM_INIT_CHECKPOINT is not a WCM checkpoint: {checkpoint_path}")
    saved_config = config_from_checkpoint_payload(payload)
    if saved_config.data.action_mean is None or saved_config.data.action_std is None:
        raise ValueError(f"WCM_INIT_CHECKPOINT has no action statistics: {checkpoint_path}")

    original_apply_runtime_overrides = command.apply_runtime_overrides
    original_build_model = command.build_model

    def apply_runtime_overrides(config):
        config = original_apply_runtime_overrides(config)
        config.data.action_mean = list(saved_config.data.action_mean)
        config.data.action_std = list(saved_config.data.action_std)
        return config

    def build_model(config):
        model = original_build_model(config)
        model.load_state_dict(payload["model"], strict=True)
        return model

    command.apply_runtime_overrides = apply_runtime_overrides
    command.build_model = build_model


def _install_gpu_reservation_release(command) -> None:
    """Release launcher-held memory immediately before WCM model construction."""

    from reserve_gpu_memory import release_gpu_reservation_from_environment

    original_build_model = command.build_model

    def build_model(config):
        release_gpu_reservation_from_environment()
        return original_build_model(config)

    command.build_model = build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    wcm_root = repo_root / "external_dependencies" / "WCM"
    sys.path.insert(0, str(wcm_root))
    # The upstream WCM checkout also has a top-level ``scripts`` package.  Do
    # not rely on namespace-package merging here; import this adapter directory
    # explicitly so it remains unambiguous in both single-process and torchrun.
    sys.path.insert(0, str(repo_root / "scripts" / "posttrain"))

    from progress import install_eval_progress, install_train_progress, tqdm_print_bridge
    from robodojo_dataset import load_robodojo_dataset
    import world_critic.data as wcm_data

    _install_ddp_model_compatibility()
    # The official trainer keeps the temporal dataset/window/model/loss code;
    # only its LeRobot constructor is replaced for RoboDojo's v2.1 files.
    def load_dataset(config):
        dataset_root = os.environ.get("WCM_DATASET_ROOT")
        if dataset_root:
            config.root = dataset_root
        if os.environ.get("WCM_TASK_NAME"):
            config.split_manifest = None
        return load_robodojo_dataset(config)

    wcm_data.load_lerobot_dataset = load_dataset
    if parsed.mode == "train":
        import world_critic.train as command

        command.load_lerobot_dataset = load_dataset
        _install_initial_checkpoint(command)
        _install_gpu_reservation_release(command)
        install_train_progress(command, train_epochs=_configured_epochs(parsed.args))
    else:
        import world_critic.evaluate as command

        command.load_lerobot_dataset = load_dataset
        install_eval_progress(command)

    sys.argv = [f"world_critic.{parsed.mode}", *parsed.args]
    with tqdm_print_bridge():
        command.run()


if __name__ == "__main__":
    main()
