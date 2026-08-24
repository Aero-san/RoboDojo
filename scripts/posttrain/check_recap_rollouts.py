"""Summarize rollout manifests and enforce RECAP data-quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict[str, object]:
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "episodes").glob("*/manifest.json"))
    ]
    successes = sum(bool(row.get("success")) for row in manifests)
    episodes = len(manifests)
    return {
        "episodes": episodes,
        "successes": successes,
        "failures": episodes - successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "mean_score": (
            sum(float(row.get("score", bool(row.get("success")))) for row in manifests) / episodes
            if episodes
            else 0.0
        ),
    }


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    summary = summarize(root)
    failures: list[str] = []
    if summary["episodes"] != args.expected:
        failures.append(f"episodes={summary['episodes']} expected={args.expected}")
    if summary["successes"] < args.min_successes:
        failures.append(f"successes={summary['successes']} minimum={args.min_successes}")
    if summary["failures"] < args.min_failures:
        failures.append(f"failures={summary['failures']} minimum={args.min_failures}")
    payload = {
        "schema_version": 1,
        "type": "recap_rollout_quality",
        "root": str(root),
        **summary,
        "required": {
            "episodes": args.expected,
            "min_successes": args.min_successes,
            "min_failures": args.min_failures,
        },
        "passed": not failures,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if failures:
        raise SystemExit("RECAP rollout quality gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--min-successes", type=int, default=0)
    parser.add_argument("--min-failures", type=int, default=0)
    parser.add_argument("--output", default="")
    main(parser.parse_args())
