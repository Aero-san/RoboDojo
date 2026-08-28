"""Freeze G05 dataset statistics and tokenizer sidecars for a RECAP run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def _sidecar(checkpoint: Path, name: str, explicit: str) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"G05 {name} does not exist: {candidate}")
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (start, *start.parents):
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No {name} beside G05 checkpoint: {checkpoint}")


def _portable_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = _sidecar(checkpoint, "dataset_stats.json", args.dataset_stats)
    tokenizer = _sidecar(checkpoint, "action_tokenizer.pt", args.action_tokenizer)
    _portable_copy(stats, output / "dataset_stats.json")
    _portable_copy(stats, output / "norm_stats.json")
    _portable_copy(tokenizer, output / "action_tokenizer.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-stats", default="")
    parser.add_argument("--action-tokenizer", default="")
    main(parser.parse_args())
