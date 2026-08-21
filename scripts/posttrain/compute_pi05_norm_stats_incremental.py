"""Incrementally update OpenPI normalization statistics for RECAP datasets."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import torch

try:
    from progress import progress_iter
except ModuleNotFoundError:
    from scripts.posttrain.progress import progress_iter

try:
    from train_pi05 import PosttrainDataConfig, _robot_action_dim, _robot_delta_mask
except ModuleNotFoundError:
    from scripts.posttrain.train_pi05 import PosttrainDataConfig, _robot_action_dim, _robot_delta_mask


class RemoveStrings:
    def __call__(self, value: dict) -> dict:
        return {
            key: item
            for key, item in value.items()
            if not np.issubdtype(np.asarray(item).dtype, np.str_)
        }


class _NoVideoDataset:
    """Keep LeRobot state/action queries while supplying unused placeholder images."""

    def __init__(self, dataset, image_keys: list[str]):
        self._dataset = dataset
        self._image_keys = image_keys
        self._placeholder = np.zeros((3, 1, 1), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        item = self._dataset[index]
        for key in self._image_keys:
            item[key] = self._placeholder
        return item


def _replace_video_dataset(dataset):
    parent = None
    base = dataset
    while hasattr(base, "_dataset"):
        parent = base
        base = base._dataset
    image_keys = [
        key
        for key, feature in base.meta.info["features"].items()
        if feature.get("dtype") in {"video", "image"}
    ]
    if not image_keys:
        return dataset, base
    # Image-mode datasets keep pixels in parquet and do not invoke a video
    # decoder. They can still use incremental statistics; only the video-mode
    # fast path needs to replace visual observations.
    if any(base.meta.info["features"][key].get("dtype") == "image" for key in image_keys):
        return dataset, base
    for key in image_keys:
        del base.meta.info["features"][key]
    replacement = _NoVideoDataset(base, image_keys)
    if parent is None:
        return replacement, base
    parent._dataset = replacement
    return dataset, base


ACCUMULATOR = "running_stats.npz"
MANIFEST = "incremental_norm.json"


def _build_dataset(args: argparse.Namespace):
    openpi_root = Path(args.openpi_root).expanduser().resolve()
    sys.path.insert(0, str(openpi_root / "src"))
    import openpi.training.config as config_lib
    import openpi.training.data_loader as data_loader

    config = config_lib.get_config(args.train_config_name)
    factory = dataclasses.replace(config.data, repo_id=args.repo_id)
    output_action_dim = _robot_action_dim(args.env_cfg_type, args.action_type)
    if output_action_dim is not None and hasattr(factory, "output_action_dim"):
        factory = dataclasses.replace(factory, output_action_dim=output_action_dim)
    mask = _robot_delta_mask(args.env_cfg_type, args.action_type)
    if mask is not None and hasattr(factory, "use_delta_joint_actions"):
        factory = dataclasses.replace(factory, use_delta_joint_actions=False)
    factory = PosttrainDataConfig(factory, delta_action_mask=mask)
    data_config = factory.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset, base_dataset = _replace_video_dataset(dataset)
    dataset = data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )
    episode_rows = base_dataset.meta.episodes
    episode_lengths = [int(episode_rows[index]["length"]) for index in range(len(episode_rows))]
    if sum(episode_lengths) != len(dataset):
        raise ValueError(
            f"LeRobot episode lengths sum to {sum(episode_lengths)}, dataset has {len(dataset)} frames."
        )
    return dataset, data_loader, episode_lengths


def _restore_running_stats(normalize, path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=False)
    result = {}
    for key in ("state", "actions"):
        running = normalize.RunningStats()
        running._count = int(payload[f"{key}_count"].item())
        running._mean = payload[f"{key}_mean"].copy()
        running._mean_of_squares = payload[f"{key}_mean_of_squares"].copy()
        running._min = payload[f"{key}_min"].copy()
        running._max = payload[f"{key}_max"].copy()
        running._histograms = [row.copy() for row in payload[f"{key}_histograms"]]
        running._bin_edges = [row.copy() for row in payload[f"{key}_bin_edges"]]
        result[key] = running
    return result


def _save_running_stats(path: Path, stats: dict[str, object]) -> None:
    values = {}
    for key, running in stats.items():
        values.update(
            {
                f"{key}_count": np.asarray(running._count, dtype=np.int64),
                f"{key}_mean": np.asarray(running._mean),
                f"{key}_mean_of_squares": np.asarray(running._mean_of_squares),
                f"{key}_min": np.asarray(running._min),
                f"{key}_max": np.asarray(running._max),
                f"{key}_histograms": np.stack(running._histograms),
                f"{key}_bin_edges": np.stack(running._bin_edges),
            }
        )
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _load_previous(
    args: argparse.Namespace,
    normalize,
    episode_lengths: list[int],
) -> tuple[dict[str, object], int, int]:
    if not args.previous_stats:
        return {key: normalize.RunningStats() for key in ("state", "actions")}, 0, 0
    previous = Path(args.previous_stats).expanduser().resolve()
    accumulator = previous / ACCUMULATOR
    manifest_path = previous / MANIFEST
    if not accumulator.is_file() or not manifest_path.is_file():
        print("[Pi0.5 normalization] preceding statistics have no accumulator; rebuilding once")
        return {key: normalize.RunningStats() for key in ("state", "actions")}, 0, 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "train_config_name": args.train_config_name,
        "env_cfg_type": args.env_cfg_type,
        "action_type": args.action_type,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Previous normalization accumulator {key}={manifest.get(key)!r}, expected {value!r}."
            )
    prior_frames = int(manifest.get("dataset_frames", -1))
    prior_episodes = int(manifest.get("dataset_episodes", -1))
    dataset_size = sum(episode_lengths)
    if prior_episodes < 0 or prior_episodes > len(episode_lengths):
        raise ValueError(
            f"Previous normalization covers {prior_episodes} episodes, current dataset has {len(episode_lengths)}."
        )
    expected_prior_frames = sum(episode_lengths[:prior_episodes])
    if prior_frames != expected_prior_frames or prior_frames > dataset_size:
        raise ValueError(
            f"Previous normalization prefix has {prior_episodes} episodes/{prior_frames} frames, "
            f"expected {expected_prior_frames} frames in the current dataset."
        )
    return _restore_running_stats(normalize, accumulator), prior_frames, prior_episodes


def main(args: argparse.Namespace) -> None:
    if args.max_frames > 0:
        raise ValueError(
            "Incremental normalization requires OPENPI_NORM_MAX_FRAMES=0 so the dataset is a stable prefix."
        )
    openpi_root = Path(args.openpi_root).expanduser().resolve()
    sys.path.insert(0, str(openpi_root / "src"))
    import openpi.shared.normalize as normalize

    dataset, data_loader, episode_lengths = _build_dataset(args)
    stats, start, prior_episodes = _load_previous(args, normalize, episode_lengths)
    batches: list[list[int]] = []
    offset = start
    for length in episode_lengths[prior_episodes:]:
        episode_end = offset + length
        batches.extend(
            list(range(batch_start, min(batch_start + args.batch_size, episode_end)))
            for batch_start in range(offset, episode_end, args.batch_size)
        )
        offset = episode_end
    if batches:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=data_loader._collate_fn,
        )
        for batch in progress_iter(
            loader,
            desc="Incremental Pi0.5 normalization",
            total=len(loader),
            unit="batch",
        ):
            for key, running in stats.items():
                running.update(np.asarray(batch[key]))
    else:
        print(f"[Pi0.5 normalization] no new frames; reusing accumulator for {start} frames")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".incremental_update_in_progress"
    marker.write_text("incremental normalization update in progress\n", encoding="utf-8")
    normalize.save(output, {key: running.get_statistics() for key, running in stats.items()})
    _save_running_stats(output / ACCUMULATOR, stats)
    manifest = {
        "schema_version": 1,
        "type": "pi05_incremental_normalization",
        "repo_id": args.repo_id,
        "train_config_name": args.train_config_name,
        "env_cfg_type": args.env_cfg_type,
        "action_type": args.action_type,
        "dataset_episodes": len(episode_lengths),
        "dataset_frames": len(dataset),
        "reused_frames": start,
        "new_frames": len(dataset) - start,
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    marker.unlink()
    print(
        f"saved incremental Pi0.5 normalization: {output / 'norm_stats.json'} "
        f"reused={start} new={len(dataset) - start} total={len(dataset)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-cfg-type", required=True)
    parser.add_argument("--action-type", choices=("joint", "ee"), default="joint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous-stats", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    main(parser.parse_args())
