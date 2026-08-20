"""Materialize successful demonstrations and policy rollouts into one buffer.

The output intentionally uses the compact RoboDojo LeRobot-v2.1 layout read
by ``robodojo_dataset.py``.  Videos are hard-linked when possible, so each
RECAP iteration can expose the complete aggregated dataset without duplicating
large media files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

try:
    from progress import progress_iter
    from robodojo_dataset import filter_episode_metadata
except ModuleNotFoundError:
    from scripts.posttrain.progress import progress_iter
    from scripts.posttrain.robodojo_dataset import filter_episode_metadata


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _video_source(root: Path, info: dict[str, Any], episode: int, camera: str) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    keys = [f"observation.images.{camera}"]
    if camera in {"cam_left_wrist", "cam_right_wrist"}:
        keys.append("observation.images.cam_wrist")
    for key in keys:
        path = root / template.format(
            episode_chunk=episode // chunk_size,
            episode_index=episode,
            video_key=key,
        )
        if path.exists():
            return path
    raise FileNotFoundError(f"No {camera} video for episode={episode} below {root}")


def _episode_table(
    actions: np.ndarray,
    reference_actions: np.ndarray,
    states: np.ndarray,
    episode_index: int,
    task_index: int,
    global_start: int,
    fps: float,
) -> pa.Table:
    actions = np.asarray(actions, dtype=np.float32)
    reference_actions = np.asarray(reference_actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if (
        actions.ndim != 2
        or reference_actions.shape != actions.shape
        or states.ndim != 2
        or len(actions) != len(states)
    ):
        raise ValueError(
            "Invalid trajectory arrays: "
            f"action={actions.shape}, reference={reference_actions.shape}, state={states.shape}"
        )
    length = len(actions)
    return pa.Table.from_arrays(
        [
            pa.array(states.tolist(), type=pa.list_(pa.float32(), states.shape[1])),
            pa.array(actions.tolist(), type=pa.list_(pa.float32(), actions.shape[1])),
            pa.array(reference_actions.tolist(), type=pa.list_(pa.float32(), actions.shape[1])),
            pa.array(np.arange(length, dtype=np.float32) / float(fps)),
            pa.array(np.arange(length, dtype=np.int64)),
            pa.array(np.full(length, episode_index, dtype=np.int64)),
            pa.array(np.arange(global_start, global_start + length, dtype=np.int64)),
            pa.array(np.full(length, task_index, dtype=np.int64)),
        ],
        names=[
            "observation.state",
            "action",
            "reference_action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )


def _demo_records(root: Path, task: str, max_episodes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = _json(root / "meta" / "info.json")
    episodes, _ = filter_episode_metadata(_jsonl(root / "meta" / "episodes.jsonl"), task)
    episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = int(info.get("chunks_size", 1000))
    records = []
    for metadata in progress_iter(
        episodes,
        desc="Loading demonstrations",
        total=len(episodes),
        unit="episode",
    ):
        episode = int(metadata["episode_index"])
        path = root / template.format(episode_chunk=episode // chunk_size, episode_index=episode)
        table = parquet.read_table(path, columns=["observation.state", "action"])
        records.append(
            {
                "kind": "demo",
                "source": str(root),
                "source_episode": episode,
                "task": str(metadata["tasks"][0]),
                "success": True,
                "score": 1.0,
                "fps": float(info.get("fps", 25)),
                "states": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
                "actions": np.asarray(table["action"].to_pylist(), dtype=np.float32),
                "reference_actions": np.asarray(table["action"].to_pylist(), dtype=np.float32),
                "videos": {camera: _video_source(root, info, episode, camera) for camera in CAMERAS},
            }
        )
    return info, records


def _rollout_records(root: Path, task: str, max_episodes: int, seed: int) -> list[dict[str, Any]]:
    candidates = sorted((root / "episodes").glob("*/manifest.json"))
    records = []
    for manifest_path in progress_iter(
        candidates,
        desc="Loading rollouts",
        total=len(candidates),
        unit="episode",
    ):
        manifest = _json(manifest_path)
        if task:
            try:
                filter_episode_metadata(
                    [{"episode_index": 0, "tasks": [str(manifest["task"])]}], task
                )
            except ValueError:
                continue
        episode_dir = manifest_path.parent
        with np.load(episode_dir / "trajectory.npz") as arrays:
            states = np.asarray(arrays["observation_state"], dtype=np.float32)
            actions = np.asarray(arrays["action"], dtype=np.float32)
            if "reference_action" in arrays.files:
                reference_actions = np.asarray(arrays["reference_action"], dtype=np.float32)
            else:
                reference_actions = actions.copy()
        videos = {camera: episode_dir / f"{camera}.mp4" for camera in CAMERAS}
        missing = [str(path) for path in videos.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Rollout episode is missing camera videos: {missing}")
        records.append(
            {
                "kind": "rollout",
                "source": str(root),
                "source_episode": int(manifest["episode_index"]),
                "run_id": str(manifest.get("run_id", "")),
                "task": str(manifest["task"]),
                "success": bool(manifest["success"]),
                "score": float(manifest.get("score", float(manifest["success"]))),
                "fps": float(manifest["fps"]),
                "states": states,
                "actions": actions,
                "reference_actions": reference_actions,
                "videos": videos,
            }
        )
    if max_episodes > 0 and len(records) > max_episodes:
        records = random.Random(seed).sample(records, max_episodes)
    return records


def main(args: argparse.Namespace) -> None:
    if args.max_demo_episodes < 0:
        raise ValueError("--max-demo-episodes must be non-negative.")
    if args.max_rollout_episodes < 0:
        raise ValueError("--max-rollout-episodes must be non-negative.")
    demo_root = Path(args.demo_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Replay buffer output already exists: {output}")
    source_info, records = _demo_records(demo_root, args.task, args.max_demo_episodes)
    for index, raw_root in enumerate(args.rollout_root):
        records.extend(
            _rollout_records(
                Path(raw_root).expanduser().resolve(),
                args.task,
                args.max_rollout_episodes,
                args.seed + index + 1,
            )
        )
    if not records:
        raise ValueError("The replay buffer has no episodes.")

    dimensions = {(record["states"].shape[1], record["actions"].shape[1]) for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"Replay buffer mixes incompatible robot/action dimensions: {sorted(dimensions)}")
    state_dim, action_dim = dimensions.pop()
    output.mkdir(parents=True)
    (output / "meta").mkdir()

    tasks: dict[str, int] = {}
    episodes_meta = []
    provenance = []
    labels: dict[str, bool] = {}
    global_index = 0
    total_videos = 0
    for episode_index, record in enumerate(
        progress_iter(
            records,
            desc="Building replay buffer",
            total=len(records),
            unit="episode",
        )
    ):
        task_index = tasks.setdefault(record["task"], len(tasks))
        table = _episode_table(
            record["actions"],
            record["reference_actions"],
            record["states"],
            episode_index,
            task_index,
            global_index,
            record["fps"],
        )
        chunk = episode_index // args.chunk_size
        parquet_path = output / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, parquet_path)
        for camera, source in record["videos"].items():
            destination = (
                output
                / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4"
            )
            _link(Path(source), destination)
            total_videos += 1
        length = len(record["actions"])
        episodes_meta.append({"episode_index": episode_index, "tasks": [record["task"]], "length": length})
        labels[str(episode_index)] = bool(record["success"])
        provenance.append(
            {
                "episode_index": episode_index,
                "source_kind": record["kind"],
                "source": record["source"],
                "source_episode": record["source_episode"],
                "run_id": record.get("run_id", ""),
                "success": bool(record["success"]),
                "score": float(record["score"]),
            }
        )
        global_index += length

    features = dict(source_info["features"])
    features["observation.state"] = dict(features["observation.state"], shape=[state_dim])
    features["action"] = dict(features["action"], shape=[action_dim])
    features["reference_action"] = dict(features["action"])
    features = {
        key: value
        for key, value in features.items()
        if not key.startswith("observation.images.")
        or key in {f"observation.images.{camera}" for camera in CAMERAS}
    }
    info = dict(source_info)
    info.update(
        {
            "total_episodes": len(records),
            "total_frames": global_index,
            "total_tasks": len(tasks),
            "total_videos": total_videos,
            "total_chunks": (len(records) + args.chunk_size - 1) // args.chunk_size,
            "chunks_size": args.chunk_size,
            "splits": {"train": f"0:{len(records)}"},
            "features": features,
        }
    )
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (output / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes_meta), encoding="utf-8"
    )
    (output / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for task, index in tasks.items()),
        encoding="utf-8",
    )
    (output / "meta" / "success_labels.json").write_text(
        json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "meta" / "provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "task": args.task,
        "episodes": len(records),
        "demonstrations": sum(row["source_kind"] == "demo" for row in provenance),
        "rollouts": sum(row["source_kind"] == "rollout" for row in provenance),
        "successes": sum(labels.values()),
        "failures": len(labels) - sum(labels.values()),
        "frames": global_index,
    }
    (output / "meta" / "replay_buffer.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", required=True)
    parser.add_argument("--rollout-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--max-demo-episodes", type=int, default=0)
    parser.add_argument("--max-rollout-episodes", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3072)
    main(parser.parse_args())
