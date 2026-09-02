"""Launch upstream G05 finetuning with RoboDojo runtime guards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys


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

    sampling_report = None
    if "ROBODOJO_G05_DEMO_SAMPLING_WEIGHT" in os.environ:
        try:
            from g05_source_sampling import install_source_balancing
        except ModuleNotFoundError:
            from scripts.posttrain.g05_source_sampling import install_source_balancing

        sampling_report = install_source_balancing(
            os.environ["ROBODOJO_RECAP_DATASET"],
            float(os.environ["ROBODOJO_G05_DEMO_SAMPLING_WEIGHT"]),
            float(os.environ["ROBODOJO_G05_ROLLOUT_SAMPLING_WEIGHT"]),
            int(os.environ.get("ROBODOJO_G05_SAMPLING_SEED", "0")),
        )

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

    report_path = os.environ.get("ROBODOJO_G05_SAMPLING_REPORT")
    if report_path and sampling_report is not None and os.environ.get("RANK", "0") == "0":
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(sampling_report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
