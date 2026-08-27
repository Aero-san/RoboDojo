"""Build and record policy-evaluation artifacts without duplicating rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from check_recap_rollouts import summarize
except ModuleNotFoundError:
    from scripts.posttrain.check_recap_rollouts import summarize


def _episodes(root: Path) -> list[Path]:
    episodes = list((root / "episodes").glob("*/manifest.json"))
    return [
        path.parent
        for path in sorted(
            episodes,
            key=lambda path: int(
                json.loads(path.read_text(encoding="utf-8"))["episode_index"]
            ),
        )
    ]


def _link_episodes(source: Path, count: int, destination: Path) -> int:
    episodes = _episodes(source)
    if len(episodes) < count:
        raise ValueError(f"{source} has {len(episodes)} episodes, need {count}")
    for episode in episodes[:count]:
        link = destination / episode.name
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Duplicate evaluation episode name: {episode.name}")
        link.symlink_to(os.path.relpath(episode.resolve(), link.parent), target_is_directory=True)
    return count


def _write_record(
    root: Path,
    *,
    checkpoint: str,
    label: str,
    source: str,
    layout_seed: int,
    layout_offset: int,
    reused_episodes: int,
    remote_episodes: int,
) -> None:
    metrics = summarize(root)
    payload = {
        "schema_version": 1,
        "type": "recap_policy_evaluation",
        "label": label,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "source": source,
        "layout_seed": layout_seed,
        "layout_offset": layout_offset,
        "reused_rollout_episodes": reused_episodes,
        "remote_episodes": remote_episodes,
        **metrics,
    }
    (root / "evaluation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def reuse(args: argparse.Namespace) -> None:
    source = Path(args.rollout_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    episode_output = output / "episodes"
    episode_output.mkdir(parents=True)
    reused = min(args.episodes, args.reuse_episodes)
    _link_episodes(source, reused, episode_output)
    remote_count = args.episodes - reused
    if remote_count:
        if not args.remote_root:
            raise ValueError(f"Missing remote evaluation source for {remote_count} episodes")
        _link_episodes(Path(args.remote_root).expanduser().resolve(), remote_count, episode_output)
    _write_record(
        output,
        checkpoint=args.checkpoint,
        label=args.label,
        source="rollout" if not remote_count else "rollout+remote",
        layout_seed=args.layout_seed,
        layout_offset=args.layout_offset,
        reused_episodes=reused,
        remote_episodes=remote_count,
    )


def record(args: argparse.Namespace) -> None:
    root = Path(args.output).expanduser().resolve()
    if len(_episodes(root)) != args.episodes:
        raise ValueError(f"Remote evaluation is incomplete: {root}")
    _write_record(
        root,
        checkpoint=args.checkpoint,
        label=args.label,
        source="remote",
        layout_seed=args.layout_seed,
        layout_offset=args.layout_offset,
        reused_episodes=0,
        remote_episodes=args.episodes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--label", required=True)
    common.add_argument("--episodes", type=int, required=True)
    common.add_argument("--layout-seed", type=int, required=True)
    common.add_argument("--layout-offset", type=int, required=True)

    reuse_parser = subparsers.add_parser("reuse", parents=[common])
    reuse_parser.add_argument("--rollout-root", required=True)
    reuse_parser.add_argument("--reuse-episodes", type=int, required=True)
    reuse_parser.add_argument("--remote-root", default="")
    reuse_parser.set_defaults(function=reuse)

    record_parser = subparsers.add_parser("record", parents=[common])
    record_parser.set_defaults(function=record)

    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.action == "reuse" and not 0 <= args.reuse_episodes <= args.episodes:
        parser.error("--reuse-episodes must be between zero and --episodes")
    args.function(args)


if __name__ == "__main__":
    main()
