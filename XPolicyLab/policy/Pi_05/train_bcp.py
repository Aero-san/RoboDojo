#!/usr/bin/env python3
"""Train the Pi0.5 BCP continuation head from grouped simulator rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import torch

from XPolicyLab.utils.bernoulli_continuation import (
    clipped_grpo_loss,
    load_bcp_checkpoint,
    normalized_group_advantages,
    replanning_efficiency_reward,
    save_bcp_checkpoint,
)


def _load_rollouts(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(record, dict) or record.get("format_version") != 1:
            continue
        record["_path"] = str(path)
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No BCP rollout .pt files found below {root}.")
    return records


def _make_groups(
    records: list[dict[str, Any]],
    group_size: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("group_id", "")),
            str(record.get("task_name", "")),
            str(record.get("instance_id", "")),
        )
        if not all(key):
            raise ValueError(
                f"Rollout is missing group_id/task_name/instance_id: {record['_path']}"
            )
        grouped[key].append(record)

    result = []
    for key, items in grouped.items():
        references = [item for item in items if item.get("reference", False)]
        adaptive = [item for item in items if not item.get("reference", False)]
        if len(references) != 1 or len(adaptive) != group_size - 1:
            raise ValueError(
                f"BCP group {key} needs one reference and {group_size - 1} adaptive "
                f"rollouts, found {len(references)} and {len(adaptive)}."
            )
        if any(not item.get("decisions") for item in adaptive):
            raise ValueError(f"BCP group {key} has an adaptive rollout without decisions.")
        result.append((references[0], adaptive))
    return result


def _decision_batch(
    adaptive: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    visual_tokens = []
    actions = []
    velocities = []
    visual_masks = []
    horizon_indices = []
    old_log_probs = []
    trajectory_indices = []
    for trajectory_index, trajectory in enumerate(adaptive):
        for decision in trajectory["decisions"]:
            visual_tokens.append(decision["visual_tokens"])
            actions.append(decision["actions"])
            velocities.append(decision["velocities"])
            visual_masks.append(decision["visual_mask"])
            horizon_indices.append(int(decision["horizon_index"]))
            old_log_probs.append(float(decision["old_log_prob"]))
            trajectory_indices.append(trajectory_index)
    return (
        torch.stack(visual_tokens).to(device),
        torch.stack(actions).to(device),
        torch.stack(velocities).to(device),
        torch.stack(visual_masks).to(device),
        torch.tensor(horizon_indices, dtype=torch.long, device=device),
        torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        torch.tensor(trajectory_indices, dtype=torch.long, device=device),
    )


def _group_loss(head, reference, adaptive, device):
    successes = torch.tensor([item["success"] for item in adaptive], device=device)
    calls = torch.tensor([item["calls"] for item in adaptive], device=device)
    reference_calls = torch.full_like(calls, int(reference["calls"]))
    adaptive_rewards = replanning_efficiency_reward(
        successes,
        calls,
        reference_calls,
        delta_positive=head.config.delta_positive,
        delta_negative=head.config.delta_negative,
    )
    reference_reward = torch.tensor(float(reference["success"]), device=device)
    rewards = torch.cat((adaptive_rewards, reference_reward[None]))
    if torch.count_nonzero(rewards).item() == 0:
        return None
    advantages = normalized_group_advantages(rewards)[:-1]
    (
        visual,
        actions,
        velocities,
        mask,
        indices,
        old_log_probs,
        trajectory_indices,
    ) = _decision_batch(adaptive, device)
    distribution = head.distribution(visual, actions, velocities, mask)
    new_log_probs = distribution.log_prob(indices)
    loss = clipped_grpo_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        trajectory_indices,
        clip_low=head.config.clip_low,
        clip_high=head.config.clip_high,
    )
    return loss, int(new_log_probs.numel())


def _finish_update(optimizer, head, decision_count: int, max_grad_norm: float) -> None:
    if decision_count < 1:
        return
    for parameter in head.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(decision_count)
    torch.nn.utils.clip_grad_norm_(head.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=512, help="Adaptive horizon decisions per update.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.group_size < 2 or args.epochs < 1 or args.global_batch_size < 1:
        raise ValueError("Group size, epochs, and global batch size must be positive (group size >= 2).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    head, payload = load_bcp_checkpoint(args.checkpoint, device)
    head.train()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["weight_decay"] = args.weight_decay
    groups = _make_groups(_load_rollouts(args.rollout_dir), args.group_size)
    step = int(payload.get("step", 0))
    optimized = 0
    discarded = 0
    accumulated_decisions = 0
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.epochs):
        random.shuffle(groups)
        for reference, adaptive in groups:
            result = _group_loss(head, reference, adaptive, device)
            if result is None:
                discarded += 1
                continue
            loss, decisions = result
            (loss * decisions).backward()
            accumulated_decisions += decisions
            optimized += 1
            if accumulated_decisions >= args.global_batch_size:
                _finish_update(optimizer, head, accumulated_decisions, args.max_grad_norm)
                accumulated_decisions = 0
                step += 1
    if accumulated_decisions:
        _finish_update(optimizer, head, accumulated_decisions, args.max_grad_norm)
        step += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_bcp_checkpoint(
        args.output,
        head,
        optimizer=optimizer,
        step=step,
        metadata={
            "source_checkpoint": str(args.checkpoint),
            "rollout_dir": str(args.rollout_dir),
            "optimized_groups": optimized,
            "discarded_all_failure_groups": discarded,
        },
    )
    print(
        f"Saved BCP checkpoint to {args.output}; optimized_groups={optimized}, "
        f"discarded_all_failure_groups={discarded}, step={step}."
    )


if __name__ == "__main__":
    main()
