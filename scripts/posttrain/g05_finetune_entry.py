"""Launch upstream G05 finetuning with RoboDojo runtime guards."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    raw_attempts = os.environ.get("ROBODOJO_G05_MAX_GETITEM_ATTEMPTS", "1")
    try:
        max_attempts = int(raw_attempts)
    except ValueError as exc:
        raise ValueError(
            "ROBODOJO_G05_MAX_GETITEM_ATTEMPTS must be an integer"
        ) from exc
    if max_attempts < 1:
        raise ValueError("ROBODOJO_G05_MAX_GETITEM_ATTEMPTS must be at least 1")

    from g05.data import base_lerobot_dataset

    base_lerobot_dataset.MAX_GETITEM_ATTEMPT = max_attempts

    trainer = Path(os.environ["ROBODOJO_G05_FINETUNE_SCRIPT"]).expanduser().resolve()
    if not trainer.is_file():
        raise FileNotFoundError(f"G05 finetune script does not exist: {trainer}")

    trainer_directory = str(trainer.parent)
    original_argv0 = sys.argv[0]
    sys.path.insert(0, trainer_directory)
    sys.argv[0] = str(trainer)
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    finally:
        sys.argv[0] = original_argv0
        sys.path.remove(trainer_directory)


if __name__ == "__main__":
    main()
