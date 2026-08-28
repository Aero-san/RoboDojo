"""Rank RoboDojo demonstrations with a trained WCM and emit episode labels.

The output is intentionally the same ``episode_index -> bool`` JSON accepted
by ``prepare_policy_dataset.py`` and the WCM data adapter.  This lets the direct
Pi0.5 fine-tune path use WCM scores without changing OpenPI's trainer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
WCM_ROOT = ROOT_DIR / "external_dependencies" / "WCM"
sys.path.insert(0, str(WCM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import progress_iter  # noqa: E402
from robodojo_dataset import load_robodojo_dataset  # noqa: E402
from world_critic.data import (  # noqa: E402
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    episode_ids_from_dataset,
    validate_action_normalization,
)
from world_critic.model import WorldCriticModel  # noqa: E402
from world_critic.training import config_from_checkpoint_payload  # noqa: E402


def _load(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") not in {"deploy", "full_resume"}:
        raise ValueError("--wcm-checkpoint must be an official WCM deploy.pt, best.pt, or last.pt.")
    config = config_from_checkpoint_payload(payload)
    model = WorldCriticModel(config.model).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, model


def main(args: argparse.Namespace) -> None:
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    if args.task:
        os.environ["WCM_TASK_NAME"] = args.task
    config, model = _load(args.wcm_checkpoint, device)
    config.data.root = str(Path(args.dataset_root).expanduser().resolve())
    config.data.repo_id = config.data.repo_id or "RoboDojo/lerobot_v21_video"
    import world_critic.data as wcm_data

    wcm_data.load_lerobot_dataset = load_robodojo_dataset
    dataset = load_robodojo_dataset(config.data)
    validate_action_normalization(dataset, config.data)
    all_episode_ids = sorted(set(map(int, episode_ids_from_dataset(dataset).tolist())))
    episode_ids = all_episode_ids
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]
    windows = LeRobotWorldCriticDataset(dataset, config.data, episode_ids)
    processor = build_processor(config.model)
    collator = WorldCriticCollator(
        processor,
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    loader = DataLoader(
        windows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    sums: dict[int, float] = {episode: 0.0 for episode in episode_ids}
    counts: dict[int, int] = {episode: 0 for episode in episode_ids}
    with torch.inference_mode():
        for batch in progress_iter(
            loader,
            desc="WCM episode scoring",
            total=len(loader),
            unit="batch",
        ):
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)
            output = model(
                images,
                actions,
                batch["instruction_input_ids"].to(device),
                batch["instruction_attention_mask"].to(device),
                batch["valid_mask"].to(device),
            )
            values = output.value.squeeze(-1).mean(dim=1).cpu().tolist()
            for episode, score in zip(batch["episode_id"].tolist(), values, strict=True):
                episode = int(episode)
                sums[episode] += float(score)
                counts[episode] += 1
    scores = {episode: sums[episode] / max(counts[episode], 1) for episode in episode_ids}
    ranked = sorted(scores, key=lambda episode: (scores[episode], -episode), reverse=True)
    selected_count = max(1, round(len(ranked) * args.fraction))
    selected = set(ranked[:selected_count])
    # Emit a complete map because the converter deliberately rejects partial
    # label files; episodes outside --max-episodes remain unselected.
    labels = {str(episode): episode in selected for episode in all_episode_ids}
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"saved WCM episode selection: {output} "
        f"selected={selected_count}/{len(ranked)} scored, total={len(all_episode_ids)} score_range="
        f"[{min(scores.values()):.6f}, {max(scores.values()):.6f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcm-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--task",
        default="",
        help="One task instruction or benchmark slug such as stack_bowls.",
    )
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
