"""Source-balanced frame sampling for the G05 RECAP trainer.

G05's ``MixtureLerobotDataset`` can weight dataset groups, but a RECAP
dataset is one LeRobot directory containing both demonstrations and rollouts.
This module wraps the instantiated training dataset with a deterministic
virtual index stream so the source weights apply without copying video data
or modifying unrelated policy code.
"""

from __future__ import annotations

import json
import logging
import operator
import os
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
SOURCE_KINDS = ("demo", "rollout")


def _manifest_path(dataset_root: str | os.PathLike[str]) -> Path:
    return Path(dataset_root).expanduser().resolve() / "meta" / "recap_incremental.json"


def frame_groups_from_manifest(dataset_root: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Return global frame indices grouped by RECAP source kind."""
    manifest_path = _manifest_path(dataset_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RECAP sampling manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities = payload.get("episodes")
    if not isinstance(identities, list):
        raise ValueError(f"RECAP sampling manifest has no episode list: {manifest_path}")

    groups: dict[str, list[int]] = {name: [] for name in SOURCE_KINDS}
    offset = 0
    for episode_index, episode in enumerate(identities):
        if not isinstance(episode, dict):
            raise ValueError(
                f"RECAP sampling manifest episode {episode_index} is not an object"
            )
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(
                f"RECAP sampling manifest episode {episode_index} has invalid length {length}"
            )
        kind = str(episode.get("source_kind", ""))
        if kind in groups:
            groups[kind].extend(range(offset, offset + length))
        offset += length

    return {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
    }


def _allocations(
    groups: dict[str, np.ndarray],
    demo_weight: float,
    rollout_weight: float,
) -> dict[str, int]:
    weights = {"demo": float(demo_weight), "rollout": float(rollout_weight)}
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("RECAP demo and rollout sampling weights must both be positive.")
    missing = {
        name: int(len(groups.get(name, ())))
        for name in SOURCE_KINDS
        if len(groups.get(name, ())) == 0
    }
    if missing:
        raise ValueError(
            "RECAP source balancing requires non-empty demo and rollout frame sets; "
            f"found { {name: len(groups.get(name, ())) for name in SOURCE_KINDS} }"
        )

    total_frames = sum(len(groups[name]) for name in SOURCE_KINDS)
    demo_count = max(
        1,
        round(total_frames * weights["demo"] / (weights["demo"] + weights["rollout"])),
    )
    rollout_count = total_frames - demo_count
    if rollout_count < 1:
        rollout_count = 1
        demo_count = total_frames - 1
    return {"demo": int(demo_count), "rollout": int(rollout_count)}


def build_virtual_indices(
    groups: dict[str, np.ndarray],
    demo_weight: float,
    rollout_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a fixed-size virtual frame stream with the requested source ratio."""
    weights = {"demo": float(demo_weight), "rollout": float(rollout_weight)}
    allocations = _allocations(groups, demo_weight, rollout_weight)
    rng = np.random.default_rng(seed)
    pieces = []
    for name in SOURCE_KINDS:
        source = np.asarray(groups[name], dtype=np.int64)
        count = allocations[name]
        # ``choice`` avoids constructing one full permutation for every repeat
        # when one source is much smaller than the other.
        pieces.append(rng.choice(source, size=count, replace=count > len(source)))
    indices = np.concatenate(pieces)
    rng.shuffle(indices)
    report = {
        "schema_version": 1,
        "type": "recap_source_balanced_sampling",
        "source_frames": {name: int(len(groups[name])) for name in SOURCE_KINDS},
        "weights": weights,
        "virtual_frames": allocations,
        "total_virtual_frames": int(len(indices)),
        "seed": int(seed),
    }
    return indices, report


def report_from_manifest(
    dataset_root: str | os.PathLike[str],
    demo_weight: float,
    rollout_weight: float,
    seed: int,
) -> dict[str, Any]:
    """Build the report without instantiating the image-backed dataset."""
    _, report = build_virtual_indices(
        frame_groups_from_manifest(dataset_root), demo_weight, rollout_weight, seed
    )
    return report


class SourceBalancedDataset:
    """Delegate a G05 dataset through a source-balanced virtual index stream."""

    def __init__(
        self,
        dataset: Any,
        groups: dict[str, np.ndarray],
        demo_weight: float,
        rollout_weight: float,
        seed: int,
    ) -> None:
        indices, report = build_virtual_indices(
            groups, demo_weight, rollout_weight, seed
        )
        if len(dataset) != sum(len(group) for group in groups.values()):
            raise ValueError(
                "RECAP sampling manifest frame count does not match G05 dataset: "
                f"manifest={sum(len(group) for group in groups.values())}, "
                f"dataset={len(dataset)}"
            )
        self._dataset = dataset
        self._indices = indices
        self.report = report

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        return self._dataset[int(self._indices[operator.index(index)])]

    def __getattr__(self, name: str):
        # Preserve the processor, stats, and diagnostics interfaces expected
        # by the upstream trainer while changing only index resolution.
        return getattr(self._dataset, name)


def install_source_balancing(
    dataset_root: str | os.PathLike[str],
    demo_weight: float,
    rollout_weight: float,
    seed: int,
) -> dict[str, Any]:
    """Patch G05's dataset factory before the upstream trainer imports it."""
    import g05.utils.data.processor_utils as processor_utils

    groups = frame_groups_from_manifest(dataset_root)
    original = processor_utils.instantiate_dataset

    def instantiate_balanced(cfg, **kwargs):
        dataset = original(cfg, **kwargs)
        if kwargs.get("is_training_set") is not True:
            return dataset
        balanced = SourceBalancedDataset(
            dataset, groups, demo_weight, rollout_weight, seed
        )
        LOGGER.info("RECAP source-balanced sampling: %s", balanced.report)
        return balanced

    processor_utils.instantiate_dataset = instantiate_balanced
    return report_from_manifest(dataset_root, demo_weight, rollout_weight, seed)
