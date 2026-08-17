"""Compute RECAP N-step advantages and binary conditioning labels with WCM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
WCM_ROOT = ROOT_DIR / "external_dependencies" / "WCM"
sys.path.insert(0, str(WCM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import progress_iter  # noqa: E402
from robodojo_dataset import load_robodojo_dataset  # noqa: E402
from world_critic.data import LeRobotWorldCriticDataset, WorldCriticCollator, build_processor  # noqa: E402
from world_critic.distributed import (  # noqa: E402
    DistributedContext,
    DistributedEvalSampler,
    cleanup_distributed,
    gather_objects,
    initialize_distributed,
)
from world_critic.model import WorldCriticModel  # noqa: E402
from world_critic.training import config_from_checkpoint_payload  # noqa: E402


def _provenance(root: Path) -> dict[int, str]:
    path = root / "meta" / "provenance.jsonl"
    if not path.exists():
        return {}
    return {
        int(row["episode_index"]): str(row.get("source_kind", ""))
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") not in {"deploy", "full_resume"}:
        raise ValueError("--wcm-checkpoint must be an official WCM deploy.pt, best.pt, or last.pt.")
    config = config_from_checkpoint_payload(payload)
    model = WorldCriticModel(config.model).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return config, model


def _merge_value_shards(
    shards: list[tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], dict[tuple[int, int], int]]],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    value_sum: dict[tuple[int, int], float] = {}
    value_count: dict[tuple[int, int], int] = {}
    value_priority: dict[tuple[int, int], int] = {}
    for shard_sum, shard_count, shard_priority in shards:
        for key, priority in shard_priority.items():
            current_priority = value_priority.get(key, -1)
            if priority > current_priority:
                value_priority[key] = priority
                value_sum[key] = shard_sum[key]
                value_count[key] = shard_count[key]
            elif priority == current_priority:
                value_sum[key] += shard_sum[key]
                value_count[key] += shard_count[key]
    return value_sum, value_count


def main(args: argparse.Namespace) -> None:
    if args.lookahead < 1:
        raise ValueError("--lookahead must be positive.")
    if not 0.0 < args.positive_fraction < 1.0:
        raise ValueError("--positive-fraction must be in (0, 1).")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1].")
    if args.failure_penalty <= 0.0:
        raise ValueError("--failure-penalty must be positive.")
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size > 1:
        if args.device != "cuda":
            raise ValueError("Distributed advantage inference requires --device cuda.")
        ctx = initialize_distributed(args.expected_world_size)
        device = ctx.device
    else:
        if args.expected_world_size != 1:
            raise RuntimeError(
                f"Expected world size {args.expected_world_size}, but the annotator was not launched with torchrun."
            )
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, device=device)

    try:
        _run(args, ctx, device)
    finally:
        cleanup_distributed(ctx)


def _run(args: argparse.Namespace, ctx: DistributedContext, device: torch.device) -> None:
    root = Path(args.dataset_root).expanduser().resolve()
    labels_path = root / "meta" / "success_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Replay-buffer success labels are missing: {labels_path}")
    os.environ["WCM_DATASET_ROOT"] = str(root)
    os.environ["WCM_SUCCESS_LABELS"] = str(labels_path)
    os.environ["WCM_ASSUME_SUCCESS"] = "0"
    os.environ["WCM_FAILURE_PENALTY"] = str(args.failure_penalty)
    os.environ["WCM_GAMMA"] = str(args.gamma)
    if args.task:
        os.environ["WCM_TASK_NAME"] = args.task

    config, model = _load_model(Path(args.wcm_checkpoint).expanduser().resolve(), device)
    config.data.root = str(root)
    config.data.split_manifest = None
    import world_critic.data as wcm_data

    wcm_data.load_lerobot_dataset = load_robodojo_dataset
    dataset = load_robodojo_dataset(config.data)
    episode_ids = sorted(set(map(int, dataset._row_episode.tolist())))
    windows = LeRobotWorldCriticDataset(dataset, config.data, episode_ids)
    collator = WorldCriticCollator(
        build_processor(config.model),
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    loader = DataLoader(
        windows,
        batch_size=args.batch_size,
        sampler=DistributedEvalSampler(windows, ctx.rank, ctx.world_size) if ctx.distributed else None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    value_sum: dict[tuple[int, int], float] = {}
    value_count: dict[tuple[int, int], int] = {}
    value_priority: dict[tuple[int, int], int] = {}
    with torch.inference_mode():
        for batch in progress_iter(
            loader,
            desc="RECAP value inference",
            total=len(loader),
            unit="batch",
        ):
            output = model(
                batch["images"].to(device),
                batch["actions"].to(device),
                batch["instruction_input_ids"].to(device),
                batch["instruction_attention_mask"].to(device),
                batch["valid_mask"].to(device),
            )
            values = output.value.squeeze(-1).cpu().numpy()
            frames = batch["frame_indices"].numpy()
            for row, episode in enumerate(batch["episode_id"].tolist()):
                for column, frame in enumerate(frames[row].tolist()):
                    key = (int(episode), int(frame))
                    # Prefer the estimate with the longest available causal
                    # history. The same frame appears in overlapping windows,
                    # but averaging short- and full-history contexts changes
                    # the meaning of V(o_t).
                    priority = value_priority.get(key, -1)
                    if column > priority:
                        value_priority[key] = column
                        value_sum[key] = float(values[row, column])
                        value_count[key] = 1
                    elif column == priority:
                        value_sum[key] += float(values[row, column])
                        value_count[key] += 1

    gathered_shards = gather_objects((value_sum, value_count, value_priority), ctx)
    if not ctx.is_main:
        return
    assert gathered_shards is not None
    value_sum, value_count = _merge_value_shards(gathered_shards)

    provenance = _provenance(root)
    records = []
    all_advantages: dict[str, list[float]] = {}
    rollout_advantages: dict[str, list[float]] = {}
    for episode in progress_iter(
        episode_ids,
        desc="RECAP advantages",
        total=len(episode_ids),
        unit="episode",
    ):
        rows = np.flatnonzero(dataset._row_episode == episode)
        frames = dataset._row_local_frame[rows].astype(np.int64)
        returns = dataset._returns[rows].astype(np.float32)
        task = str(dataset._row_instruction[rows[0]])
        values = np.asarray(
            [
                value_sum[(episode, int(frame))] / value_count[(episode, int(frame))]
                if (episode, int(frame)) in value_count
                else float(returns[index])
                for index, frame in enumerate(frames)
            ],
            dtype=np.float32,
        )
        advantages = np.empty_like(values)
        for index in range(len(rows)):
            future = min(index + args.lookahead, len(rows) - 1)
            steps = future - index
            discount = args.gamma**steps
            # Both terms use the exact affine transform fitted to WCM's global
            # min-max value targets.  Taking a difference of normalized full
            # returns yields the correctly scaled discounted N-step reward,
            # including the affine-offset correction when gamma < 1.
            reward_sum = float(returns[index] - discount * returns[future])
            advantages[index] = reward_sum + discount * values[future] - values[index]
        all_advantages.setdefault(task, []).extend(map(float, advantages))
        if provenance.get(episode) == "rollout":
            rollout_advantages.setdefault(task, []).extend(map(float, advantages))
        records.append(
            {
                "episode_index": episode,
                "task": task,
                "source_kind": provenance.get(episode, "unknown"),
                "values": values.tolist(),
                "advantages": advantages.tolist(),
            }
        )

    thresholds = {}
    percentile = 100.0 * (1.0 - args.positive_fraction)
    for task, values in all_advantages.items():
        threshold_values = rollout_advantages.get(task) or values
        thresholds[task] = float(np.percentile(np.asarray(threshold_values), percentile))
    positive_count = 0
    frame_count = 0
    for record in records:
        threshold = thresholds[record["task"]]
        positive = np.asarray(record["advantages"]) > threshold
        if args.force_demonstrations_positive and record["source_kind"] == "demo":
            positive[:] = True
        record["positive"] = positive.tolist()
        record["threshold"] = threshold
        positive_count += int(positive.sum())
        frame_count += len(positive)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "schema_version": 1,
        "type": "recap_advantages",
        "lookahead": args.lookahead,
        "gamma": args.gamma,
        "failure_penalty": args.failure_penalty,
        "positive_fraction": args.positive_fraction,
        "thresholds": thresholds,
        "return_normalization": "global_minmax",
        "return_raw_min": dataset._return_raw_min,
        "return_raw_max": dataset._return_raw_max,
        "wcm_checkpoint": str(Path(args.wcm_checkpoint).expanduser().resolve()),
    }
    output.write_text(
        json.dumps(header) + "\n" + "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        f"saved RECAP labels: {output} episodes={len(records)} frames={frame_count} "
        f"positive={positive_count} ({positive_count / max(frame_count, 1):.1%}) "
        f"world_size={ctx.world_size}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcm-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=300.0)
    parser.add_argument("--positive-fraction", type=float, default=0.4)
    parser.add_argument("--force-demonstrations-positive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-world-size", type=int, default=1)
    main(parser.parse_args())
