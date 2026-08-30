"""Prepare a copied G05 checkpoint config for dataset-free remote rollout."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

_TRAINING_ONLY_LOGGER_KEYS = ("dir", "project", "workspace")
_TRAINING_ONLY_DATA_KEYS = frozenset({"dataset_dirs", "dataset_groups"})
_REMOTE_HF_PROCESSOR_PATH = "${oc.env:G05_HF_PROCESSOR_PATH}"


def _strip_training_data_locations(
    value: Any,
    *,
    path: tuple[str, ...],
    removed: list[str],
) -> None:
    if isinstance(value, dict):
        for key in list(value):
            child_path = (*path, str(key))
            if key in _TRAINING_ONLY_DATA_KEYS:
                removed.append(".".join(child_path))
                del value[key]
            else:
                _strip_training_data_locations(
                    value[key],
                    path=child_path,
                    removed=removed,
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _strip_training_data_locations(
                item,
                path=(*path, str(index)),
                removed=removed,
            )


def _materialize_action_tokenizer_config(
    payload: dict[str, Any],
    *,
    changes: list[str],
) -> None:
    """Detach AT_CONFIG from its interpolation before sidecar path patching."""

    model = payload.get("model")
    if not isinstance(model, dict):
        raise TypeError("G05 checkpoint config has no model mapping.")
    model_arch = model.get("model_arch")
    if not isinstance(model_arch, dict):
        raise TypeError("G05 checkpoint config has no model.model_arch mapping.")

    tokenizer = model.get("tokenizer")
    if not isinstance(tokenizer, dict):
        tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise TypeError("G05 checkpoint config has no tokenizer mapping.")
    vq_config = tokenizer.get("vq_config")
    if not isinstance(vq_config, dict) or not vq_config.get("vqvae_type"):
        raise ValueError("G05 checkpoint tokenizer.vq_config has no vqvae_type.")
    tokenizer_target = tokenizer.get("_target_")
    if not isinstance(tokenizer_target, str) or not tokenizer_target:
        raise ValueError("G05 checkpoint tokenizer has no _target_.")

    if model_arch.get("action_tokenizer") != tokenizer_target:
        model_arch["action_tokenizer"] = tokenizer_target
        changes.append("materialized:model.model_arch.action_tokenizer")

    current = model_arch.get("AT_CONFIG")
    if isinstance(current, dict) and current.get("vqvae_type"):
        return
    materialized = deepcopy(vq_config)
    if isinstance(current, dict):
        materialized.update(current)
    model_arch["AT_CONFIG"] = materialized
    changes.append("materialized:model.model_arch.AT_CONFIG")


def _make_hf_processor_path_portable(
    payload: dict[str, Any],
    *,
    changes: list[str],
) -> None:
    """Replace the training-host processor path with the worker-provided path."""

    model = payload.get("model")
    if not isinstance(model, dict):
        raise TypeError("G05 checkpoint config has no model mapping.")
    model_arch = model.get("model_arch")
    if not isinstance(model_arch, dict):
        raise TypeError("G05 checkpoint config has no model.model_arch mapping.")
    if model_arch.get("hf_processor_path") != _REMOTE_HF_PROCESSOR_PATH:
        model_arch["hf_processor_path"] = _REMOTE_HF_PROCESSOR_PATH
        changes.append("portable:model.model_arch.hf_processor_path")


def prepare_g05_inference_checkpoint(checkpoint_root: str | Path) -> list[str]:
    """Remove training paths and detach the action-tokenizer config interpolation."""

    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    config_path = checkpoint_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"G05 checkpoint Hydra config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"G05 checkpoint Hydra config must be a mapping: {config_path}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"G05 checkpoint Hydra config has no data mapping: {config_path}")

    changes: list[str] = []
    _strip_training_data_locations(data, path=("data",), removed=changes)
    logger = payload.get("logger")
    if isinstance(logger, dict):
        for key in _TRAINING_ONLY_LOGGER_KEYS:
            if key in logger:
                changes.append(f"logger.{key}")
                del logger[key]

    _make_hf_processor_path_portable(payload, changes=changes)
    _materialize_action_tokenizer_config(payload, changes=changes)

    if changes:
        temporary = config_path.with_name(f"{config_path.name}.tmp")
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        temporary.chmod(config_path.stat().st_mode)
        temporary.replace(config_path)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()
    changes = prepare_g05_inference_checkpoint(args.checkpoint_root)
    print(
        json.dumps(
            {
                "checkpoint_root": str(Path(args.checkpoint_root).expanduser().resolve()),
                "inference_config_changes": changes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
