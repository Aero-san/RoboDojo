"""Validate and recoverably archive RECAP stage artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_buffer(path: Path) -> bool:
    required = (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/success_labels.json",
        "meta/provenance.jsonl",
        "meta/replay_buffer.json",
    )
    if not all((path / item).is_file() for item in required):
        return False
    info = _json(path / "meta/info.json")
    episodes = [line for line in (path / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    return int(info.get("total_episodes", -1)) == len(episodes) > 0


def _check_advantages(path: Path) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows) > 1 and rows[0].get("type") == "recap_advantages"


def _check_pi_dataset(path: Path) -> bool:
    return (
        (path / "meta/info.json").is_file()
        and (path / "meta/stats.json").is_file()
        and (path / "data").is_dir()
    )


def _check_policy(path: Path) -> bool:
    if not (path / "robodojo_pi05_model.json").is_file():
        return False
    return any(
        child.is_dir()
        and (child / "params").is_dir()
        and (child / "assets").is_dir()
        and (child / "_CHECKPOINT_METADATA").is_file()
        for child in path.iterdir()
    )


def _check_policy_resume(path: Path) -> bool:
    return any(
        child.is_dir()
        and (child / "params").is_dir()
        and (child / "train_state").is_dir()
        and (child / "_CHECKPOINT_METADATA").is_file()
        for child in path.iterdir()
    ) if path.is_dir() else False


def _check_rollout(path: Path, expected: int) -> bool:
    manifests = list((path / "episodes").glob("*/manifest.json"))
    if len(manifests) != expected:
        return False
    for manifest_path in manifests:
        manifest = _json(manifest_path)
        episode_dir = manifest_path.parent
        if not (episode_dir / "trajectory.npz").is_file():
            return False
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if not (episode_dir / f"{camera}.mp4").is_file():
                return False
        if int(manifest.get("length", 0)) < 1:
            return False
    return True


def check(stage: str, path: Path, expected: int) -> bool:
    if stage == "buffer":
        return _check_buffer(path)
    if stage == "wcm":
        return (path / "deploy.pt").is_file()
    if stage == "advantages":
        return _check_advantages(path)
    if stage == "pi_dataset":
        return _check_pi_dataset(path)
    if stage == "norm":
        return (path / "norm_stats.json").is_file()
    if stage == "policy":
        return _check_policy(path)
    if stage == "policy_resume":
        return _check_policy_resume(path)
    if stage == "rollout":
        return _check_rollout(path, expected)
    if stage == "value_videos":
        return (
            (path / "summary.json").is_file()
            and len(list((path / "videos").glob("episode-*.mp4"))) == expected
        )
    raise ValueError(f"Unknown stage: {stage}")


def archive(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.incomplete-{timestamp}")
    suffix = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.incomplete-{timestamp}-{suffix}")
        suffix += 1
    shutil.move(str(path), str(destination))
    print(f"[RECAP resume] archived incomplete artifact: {path} -> {destination}")


def main(args: argparse.Namespace) -> None:
    path = Path(args.path).expanduser().resolve()
    if args.command == "check":
        if not check(args.stage, path, args.expected):
            raise SystemExit(1)
        return
    archive(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    checker = subparsers.add_parser("check")
    checker.add_argument("--stage", required=True)
    checker.add_argument("--path", required=True)
    checker.add_argument("--expected", type=int, default=0)
    archiver = subparsers.add_parser("archive")
    archiver.add_argument("--path", required=True)
    main(parser.parse_args())
