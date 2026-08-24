"""Select the best evaluated checkpoint and enforce a promotion safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from check_recap_rollouts import summarize
except ModuleNotFoundError:
    from scripts.posttrain.check_recap_rollouts import summarize


def _candidate(value: str) -> tuple[int, Path, Path]:
    parts = value.split("::", 2)
    if len(parts) != 3:
        raise ValueError("--candidate must be STEP::CHECKPOINT::ROLLOUT_ROOT")
    return int(parts[0]), Path(parts[1]).resolve(), Path(parts[2]).resolve()


def main(args: argparse.Namespace) -> None:
    baseline_root = Path(args.baseline_rollouts).expanduser().resolve()
    baseline = summarize(baseline_root)
    candidates = []
    for raw in args.candidate:
        step, checkpoint, rollout_root = _candidate(raw)
        metrics = summarize(rollout_root)
        if metrics["episodes"] != baseline["episodes"]:
            raise ValueError(
                f"Candidate step {step} has {metrics['episodes']} evaluations; "
                f"baseline has {baseline['episodes']}."
            )
        candidates.append(
            {"step": step, "checkpoint": str(checkpoint), "rollout_root": str(rollout_root), **metrics}
        )
    if not candidates:
        raise ValueError("No evaluated policy candidates were provided.")
    selected = max(candidates, key=lambda row: (row["success_rate"], row["mean_score"], row["step"]))
    required_rate = max(args.min_success_rate, baseline["success_rate"] - args.max_success_drop)
    promoted = selected["success_rate"] >= required_rate
    payload = {
        "schema_version": 1,
        "type": "recap_policy_promotion",
        "baseline": {"checkpoint": str(Path(args.baseline_checkpoint).resolve()), **baseline},
        "candidates": candidates,
        "selected": selected,
        "required_success_rate": required_rate,
        "max_success_drop": args.max_success_drop,
        "min_success_rate": args.min_success_rate,
        "promoted": promoted,
        "policy": selected["checkpoint"] if promoted else str(Path(args.baseline_checkpoint).resolve()),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["policy"])
    if not promoted:
        raise SystemExit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--baseline-rollouts", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--max-success-drop", type=float, default=0.1)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
