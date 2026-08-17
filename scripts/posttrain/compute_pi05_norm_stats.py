"""Compute robot-specific OpenPI normalization assets for a RECAP buffer."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

try:
    from progress import progress_iter
except ModuleNotFoundError:
    from scripts.posttrain.progress import progress_iter

from train_pi05 import PosttrainDataConfig, _robot_action_dim, _robot_delta_mask


class RemoveStrings:
    def __call__(self, value: dict) -> dict:
        return {key: item for key, item in value.items() if not np.issubdtype(np.asarray(item).dtype, np.str_)}


def main(args: argparse.Namespace) -> None:
    openpi_root = Path(args.openpi_root).expanduser().resolve()
    sys.path.insert(0, str(openpi_root / "src"))
    import openpi.shared.normalize as normalize
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

    dataset = data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )
    batch_size = min(args.batch_size, len(dataset))
    num_batches = max(1, len(dataset) // batch_size)
    if args.max_frames > 0:
        num_batches = min(num_batches, max(1, args.max_frames // batch_size))
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=args.max_frames > 0 and args.max_frames < len(dataset),
        num_batches=num_batches,
        num_workers=args.num_workers,
        framework="jax",
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in progress_iter(
        loader,
        desc="Pi0.5 normalization",
        total=num_batches,
        unit="batch",
    ):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))
    output = Path(args.output).expanduser().resolve()
    normalize.save(output, {key: running.get_statistics() for key, running in stats.items()})
    print(f"saved Pi0.5 normalization statistics: {output / 'norm_stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", required=True)
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-cfg-type", required=True)
    parser.add_argument("--action-type", choices=("joint", "ee"), default="joint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    main(parser.parse_args())
