"""Remove training-only dataset locations from a copied G05 rollout checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

_TRAINING_ONLY_LOGGER_KEYS = ("dir", "project", "workspace")
_TRAINING_ONLY_DATA_KEYS = frozenset({"dataset_dirs", "dataset_groups"})


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


def prepare_g05_inference_checkpoint(checkpoint_root: str | Path) -> list[str]:
    """Make a transferred checkpoint config independent of training datasets."""

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

    removed: list[str] = []
    _strip_training_data_locations(data, path=("data",), removed=removed)
    logger = payload.get("logger")
    if isinstance(logger, dict):
        for key in _TRAINING_ONLY_LOGGER_KEYS:
            if key in logger:
                removed.append(f"logger.{key}")
                del logger[key]

    if removed:
        temporary = config_path.with_name(f"{config_path.name}.tmp")
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        temporary.chmod(config_path.stat().st_mode)
        temporary.replace(config_path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()
    removed = prepare_g05_inference_checkpoint(args.checkpoint_root)
    print(
        json.dumps(
            {
                "checkpoint_root": str(Path(args.checkpoint_root).expanduser().resolve()),
                "removed_training_data_fields": removed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
