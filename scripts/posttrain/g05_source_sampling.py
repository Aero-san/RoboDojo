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


def _training_frame_range(dataset: Any) -> tuple[int, int]:
    """Return the raw frame interval represented by one G05 RECAP dataset.

    G05 applies its train/validation split inside ``BaseLerobotDataset`` after
    loading the complete LeRobot directory. RECAP uses exactly one dataset
    group, so its logical indices correspond to one contiguous raw interval.
    """
    inner_datasets = getattr(dataset, "datasets", None)
    if not isinstance(inner_datasets, (list, tuple)) or len(inner_datasets) != 1:
        raise ValueError(
            "G05 RECAP source balancing requires exactly one dataset group."
        )
    inner = inner_datasets[0]
    if not hasattr(inner, "_start_idx") or not hasattr(inner, "_end_idx"):
        raise TypeError(
            "G05 RECAP dataset does not expose its train/validation frame range."
        )
    start = int(inner._start_idx)
    end = int(inner._end_idx)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid G05 dataset frame range: [{start}, {end})")
    if end - start != len(dataset):
        raise ValueError(
            "G05 RECAP source balancing requires an unweighted contiguous dataset "
            f"range: range={end - start}, dataset={len(dataset)}"
        )
    return start, end


def _groups_for_frame_range(
    groups: dict[str, np.ndarray], start: int, end: int
) -> tuple[dict[str, np.ndarray], int]:
    """Restrict manifest indices to a contiguous dataset range and rebase them."""
    all_indices = np.concatenate(
        [np.asarray(groups[name], dtype=np.int64) for name in SOURCE_KINDS]
    )
    if len(all_indices) == 0:
        raise ValueError("RECAP sampling manifest contains no source frames.")
    ordered = np.sort(all_indices)
    manifest_frames = int(ordered[-1]) + 1
    if not np.array_equal(ordered, np.arange(manifest_frames, dtype=np.int64)):
        raise ValueError(
            "RECAP sampling manifest source kinds do not partition all frames."
        )
    if start < 0 or end <= start or end > manifest_frames:
        raise ValueError(
            f"G05 dataset frame range [{start}, {end}) is outside the "
            f"RECAP manifest with {manifest_frames} frames."
        )

    selected = {
        name: indices[(indices >= start) & (indices < end)] - start
        for name, indices in groups.items()
    }
    selected_indices = np.sort(np.concatenate(list(selected.values())))
    if not np.array_equal(
        selected_indices, np.arange(end - start, dtype=np.int64)
    ):
        raise ValueError(
            "RECAP sampling manifest does not cover the complete G05 dataset split."
        )
    return selected, manifest_frames


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
        frame_range: tuple[int, int] | None = None,
    ) -> None:
        if frame_range is None:
            frame_range = (0, sum(len(group) for group in groups.values()))
        groups, manifest_frames = _groups_for_frame_range(groups, *frame_range)
        indices, report = build_virtual_indices(
            groups, demo_weight, rollout_weight, seed
        )
        selected_frames = sum(len(group) for group in groups.values())
        if len(dataset) != selected_frames:
            raise ValueError(
                "RECAP sampling manifest split does not match G05 dataset: "
                f"manifest_split={selected_frames}, dataset={len(dataset)}"
            )
        report.update(
            {
                "manifest_frames": manifest_frames,
                "dataset_frame_range": list(frame_range),
                "excluded_frames": manifest_frames - selected_frames,
            }
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
    sampling_report = report_from_manifest(
        dataset_root, demo_weight, rollout_weight, seed
    )

    def instantiate_balanced(cfg, **kwargs):
        dataset = original(cfg, **kwargs)
        if kwargs.get("is_training_set") is not True:
            return dataset
        balanced = SourceBalancedDataset(
            dataset,
            groups,
            demo_weight,
            rollout_weight,
            seed,
            frame_range=_training_frame_range(dataset),
        )
        sampling_report.clear()
        sampling_report.update(balanced.report)
        LOGGER.info("RECAP source-balanced sampling: %s", balanced.report)
        return balanced

    processor_utils.instantiate_dataset = instantiate_balanced
    return sampling_report
