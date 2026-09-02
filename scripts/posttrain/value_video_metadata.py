"""Backfill and validate value-video instructions from rollout manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rollout_instructions(rollout_root: Path) -> dict[int, str]:
    instructions: dict[int, str] = {}
    for manifest_path in (rollout_root / "episodes").glob("*/manifest.json"):
        manifest = _json(manifest_path)
        episode = int(manifest["episode_index"])
        instruction = str(manifest.get("task", "")).strip()
        if not instruction:
            raise ValueError(
                f"Rollout manifest has no task instruction: {manifest_path}"
            )
        previous = instructions.get(episode)
        if previous is not None and previous != instruction:
            raise ValueError(
                f"Conflicting rollout instructions for episode={episode}: "
                f"{previous!r} != {instruction!r}"
            )
        instructions[episode] = instruction
    if not instructions:
        raise FileNotFoundError(
            f"No rollout manifests below {rollout_root / 'episodes'}"
        )
    return instructions


def backfill_value_video_instructions(
    output_dir: str | Path,
    rollout_root: str | Path,
) -> int:
    output_dir = Path(output_dir).expanduser().resolve()
    rollout_root = Path(rollout_root).expanduser().resolve()
    curve_path = output_dir / "episode_curves.json"
    summary_path = output_dir / "summary.json"
    curves = _json(curve_path)
    summary = _json(summary_path)
    if not isinstance(curves, list) or not all(
        isinstance(curve, dict) for curve in curves
    ):
        raise ValueError(f"Expected a JSON curve list: {curve_path}")
    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
        raise ValueError(f"Expected summary.episodes list: {summary_path}")

    instructions = _rollout_instructions(rollout_root)
    changes = 0
    by_render_episode: dict[int, str] = {}
    for curve in curves:
        render_episode = int(curve["episode_id"])
        source_episode = int(curve.get("source_episode_id", render_episode))
        try:
            instruction = instructions[source_episode]
        except KeyError as exc:
            raise ValueError(
                f"No rollout instruction for value-video source_episode_id="
                f"{source_episode}."
            ) from exc
        if curve.get("instruction") != instruction:
            curve["instruction"] = instruction
            changes += 1
        by_render_episode[render_episode] = instruction

    summary_episodes = summary["episodes"]
    if not all(isinstance(episode, dict) for episode in summary_episodes):
        raise ValueError(f"summary.episodes contains a non-object: {summary_path}")
    for episode in summary_episodes:
        episode_id = int(episode["episode_id"])
        try:
            instruction = by_render_episode[episode_id]
        except KeyError as exc:
            raise ValueError(
                f"No curve instruction for summary episode_id={episode_id}."
            ) from exc
        if episode.get("instruction") != instruction:
            episode["instruction"] = instruction
            changes += 1

    if changes:
        _atomic_json(curve_path, curves)
        _atomic_json(summary_path, summary)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-root", required=True)
    args = parser.parse_args()
    changes = backfill_value_video_instructions(
        args.output_dir,
        args.rollout_root,
    )
    print(
        f"value-video instruction metadata ready: "
        f"{Path(args.output_dir).expanduser()} changes={changes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
