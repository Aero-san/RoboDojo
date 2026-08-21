"""Incrementally materialize an advantage-conditioned Pi0.5 dataset.

The first materialization uses the regular converter.  Later RECAP rounds copy
the preceding LeRobot-v3 dataset, hard-link its immutable packed videos, append
only newly collected episodes, and rewrite the lightweight task-conditioning
columns for all frames using the current advantage labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError as exc:
    if exc.name != "lerobot.datasets":
        raise
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

try:
    import prepare_pi05_dataset as full_converter
    from progress import progress_iter, tqdm_print_bridge
    from robodojo_dataset import filter_episode_metadata
except ModuleNotFoundError:
    from scripts.posttrain import prepare_pi05_dataset as full_converter
    from scripts.posttrain.progress import progress_iter, tqdm_print_bridge
    from scripts.posttrain.robodojo_dataset import filter_episode_metadata


MANIFEST = "meta/recap_incremental.json"
IN_PROGRESS = "meta/.recap_update_in_progress"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    parquet.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def _advantage_labels(path: Path) -> tuple[dict[str, Any], dict[int, list[bool]]]:
    rows = _jsonl(path)
    if not rows or rows[0].get("schema_version") != 1 or rows[0].get("type") != "recap_advantages":
        raise ValueError(f"Unsupported RECAP advantage artifact: {path}")
    labels = {int(row["episode_index"]): [bool(value) for value in row["positive"]] for row in rows[1:]}
    return rows[0], labels


def _source_episodes(root: Path, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = _json(root / "meta/info.json")
    episodes = _jsonl(root / "meta/episodes.jsonl")
    task_path = root / "meta/tasks.jsonl"
    task_rows = _jsonl(task_path) if task_path.exists() else None
    episodes, _ = filter_episode_metadata(episodes, task, task_rows)
    return info, episodes


def _source_identity(root: Path, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provenance_path = root / "meta/provenance.jsonl"
    provenance = {
        int(row["episode_index"]): row
        for row in (_jsonl(provenance_path) if provenance_path.exists() else [])
    }
    identities = []
    for row in episodes:
        episode = int(row["episode_index"])
        source = provenance.get(episode, {})
        identities.append(
            {
                "episode_index": episode,
                "length": int(row["length"]),
                "task": str(row.get("tasks", [""])[0]),
                "source_kind": str(source.get("source_kind", "unknown")),
                "source": str(source.get("source", "")),
                "source_episode": int(source.get("source_episode", episode)),
                "run_id": str(source.get("run_id", "")),
            }
        )
    return identities


def _v3_episode_lengths(root: Path) -> list[int]:
    records: list[tuple[int, int]] = []
    for path in sorted((root / "meta/episodes").glob("**/*.parquet")):
        table = parquet.read_table(path, columns=["episode_index", "length"])
        records.extend(
            (int(episode), int(length))
            for episode, length in zip(table["episode_index"].to_pylist(), table["length"].to_pylist(), strict=True)
        )
    records.sort()
    if [episode for episode, _ in records] != list(range(len(records))):
        raise ValueError(f"Pi0.5 dataset has non-contiguous episode indices: {root}")
    return [length for _, length in records]


def _validate_prefix(output: Path, identities: list[dict[str, Any]]) -> int:
    info = _json(output / "meta/info.json")
    count = int(info.get("total_episodes", -1))
    if count < 0 or count > len(identities):
        raise ValueError(
            f"Existing Pi0.5 dataset has {count} episodes but the replay buffer has {len(identities)}."
        )
    lengths = _v3_episode_lengths(output)
    expected_lengths = [row["length"] for row in identities[:count]]
    if lengths != expected_lengths:
        raise ValueError("Existing Pi0.5 episodes are not a length-preserving prefix of the replay buffer.")
    manifest_path = output / MANIFEST
    if manifest_path.exists():
        previous = _json(manifest_path).get("episodes", [])
        if previous != identities[:count]:
            raise ValueError("Existing Pi0.5 dataset provenance is not a prefix of the current replay buffer.")
    return count


def _clone_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    source = source.resolve()

    def copy_file(src: str, dst: str) -> str:
        relative = Path(src).resolve().relative_to(source)
        if relative.parts and relative.parts[0] == "videos":
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=copy_file)


def _source_paths(root: Path, info: dict[str, Any], episode: int) -> tuple[Path, dict[str, Path]]:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode // chunk_size
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    data_path = root / data_template.format(episode_chunk=chunk, episode_index=episode)
    videos: dict[str, Path] = {}
    for camera in CAMERAS:
        key = f"observation.images.{camera}"
        path = root / video_template.format(video_key=key, episode_chunk=chunk, episode_index=episode)
        if not path.exists() and camera in {"cam_left_wrist", "cam_right_wrist"}:
            path = root / video_template.format(
                video_key="observation.images.cam_wrist",
                episode_chunk=chunk,
                episode_index=episode,
            )
        if not path.exists():
            raise FileNotFoundError(f"No {camera} video for episode={episode}: {path}")
        videos[camera] = path
    return data_path, videos


def _append_new_episodes(
    output: Path,
    repo_id: str,
    source_root: Path,
    source_info: dict[str, Any],
    episodes: list[dict[str, Any]],
    labels: dict[int, list[bool]],
    start: int,
) -> None:
    if start == len(episodes):
        return
    dataset = LeRobotDataset(repo_id=repo_id, root=output)
    with tqdm_print_bridge():
        for output_episode in progress_iter(
            range(start, len(episodes)),
            desc="Appending new Pi0.5 rollout videos",
            total=len(episodes) - start,
            unit="episode",
        ):
            metadata = episodes[output_episode]
            source_episode = int(metadata["episode_index"])
            data_path, videos = _source_paths(source_root, source_info, source_episode)
            table = parquet.read_table(data_path, columns=["action", "observation.state"])
            episode_labels = labels[source_episode]
            if len(episode_labels) != table.num_rows:
                raise ValueError(
                    f"RECAP labels for episode={source_episode} have {len(episode_labels)} frames, "
                    f"expected {table.num_rows}."
                )
            actions = table["action"].to_pylist()
            states = table["observation.state"].to_pylist()
            task = str(metadata.get("tasks", ["Perform the RoboDojo task."])[0])
            with (
                full_converter._SequentialVideo(videos["cam_high"]) as high_video,
                full_converter._SequentialVideo(videos["cam_left_wrist"]) as left_video,
                full_converter._SequentialVideo(videos["cam_right_wrist"]) as right_video,
            ):
                for frame in range(table.num_rows):
                    condition = "positive" if episode_labels[frame] else "negative"
                    dataset.add_frame(
                        {
                            "observation.state": np.asarray(states[frame], dtype=np.float32),
                            "action": np.asarray(actions[frame], dtype=np.float32),
                            "task": f"{task}\nAdvantage: {condition}",
                            "observation.images.cam_high": high_video.read(),
                            "observation.images.cam_left_wrist": left_video.read(),
                            "observation.images.cam_right_wrist": right_video.read(),
                        }
                    )
            dataset.save_episode()
    dataset.finalize()


def _stats(values: np.ndarray) -> dict[str, list[int | float]]:
    values = values.astype(np.float64, copy=False)
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def _rewrite_advantage_conditions(
    output: Path,
    episodes: list[dict[str, Any]],
    labels: dict[int, list[bool]],
) -> None:
    task_names: list[str] = []
    frame_tasks: dict[int, np.ndarray] = {}
    episode_task_names: dict[int, list[str]] = {}
    for output_episode, metadata in enumerate(episodes):
        source_episode = int(metadata["episode_index"])
        base_task = str(metadata.get("tasks", ["Perform the RoboDojo task."])[0])
        names = np.asarray(
            [f"{base_task}\nAdvantage: {'positive' if value else 'negative'}" for value in labels[source_episode]],
            dtype=object,
        )
        frame_tasks[output_episode] = names
        episode_task_names[output_episode] = list(dict.fromkeys(names.tolist()))
        for name in names:
            if name not in task_names:
                task_names.append(str(name))
    task_to_index = {task: index for index, task in enumerate(task_names)}
    all_task_indices: list[np.ndarray] = []

    for path in sorted((output / "data").glob("**/*.parquet")):
        table = parquet.read_table(path)
        episode_values = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_values = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        indices = np.empty(table.num_rows, dtype=np.int64)
        for episode in np.unique(episode_values):
            mask = episode_values == episode
            local_frames = frame_values[mask]
            names = frame_tasks[int(episode)][local_frames]
            indices[mask] = np.asarray([task_to_index[str(name)] for name in names], dtype=np.int64)
        column = table.schema.get_field_index("task_index")
        table = table.set_column(column, "task_index", pa.array(indices, type=pa.int64()))
        _atomic_parquet(path, table)
        all_task_indices.append(indices)

    for path in sorted((output / "meta/episodes").glob("**/*.parquet")):
        table = parquet.read_table(path)
        output_episodes = [int(value) for value in table["episode_index"].to_pylist()]
        replacements: dict[str, pa.Array] = {
            "tasks": pa.array(
                [episode_task_names[episode] for episode in output_episodes],
                type=table.schema.field("tasks").type,
            )
        }
        per_episode_stats = {
            episode: _stats(
                np.asarray([task_to_index[str(name)] for name in frame_tasks[episode]], dtype=np.int64)
            )
            for episode in output_episodes
        }
        for key in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            name = f"stats/task_index/{key}"
            if name in table.column_names:
                replacements[name] = pa.array(
                    [per_episode_stats[episode][key] for episode in output_episodes],
                    type=table.schema.field(name).type,
                )
        for name, values in replacements.items():
            table = table.set_column(table.schema.get_field_index(name), name, values)
        _atomic_parquet(path, table)

    tasks = pd.DataFrame({"task_index": range(len(task_names))}, index=task_names)
    tasks.index.name = None
    tasks_path = output / "meta/tasks.parquet"
    temporary_tasks = tasks_path.with_suffix(".parquet.tmp")
    tasks.to_parquet(temporary_tasks)
    os.replace(temporary_tasks, tasks_path)

    info_path = output / "meta/info.json"
    info = _json(info_path)
    info["total_tasks"] = len(task_names)
    _atomic_json(info_path, info)
    stats_path = output / "meta/stats.json"
    stats = _json(stats_path)
    stats["task_index"] = _stats(np.concatenate(all_task_indices))
    _atomic_json(stats_path, stats)


def _full_materialization(args: argparse.Namespace) -> None:
    full_converter.main(
        argparse.Namespace(
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            episode_labels="",
            advantage_labels=args.advantage_labels,
            task=args.task,
            max_episodes=0,
            mode=args.mode,
        )
    )


def main(args: argparse.Namespace) -> None:
    source_root = Path(args.dataset_root).expanduser().resolve()
    output = (HF_LEROBOT_HOME / args.repo_id).resolve()
    previous = Path(args.previous_dataset).expanduser().resolve() if args.previous_dataset else None
    source_info, episodes = _source_episodes(source_root, args.task)
    advantage_header, labels = _advantage_labels(Path(args.advantage_labels).expanduser().resolve())
    missing = [int(row["episode_index"]) for row in episodes if int(row["episode_index"]) not in labels]
    if missing:
        raise ValueError(f"RECAP labels are missing {len(missing)} episodes, first={missing[:5]}")
    identities = _source_identity(source_root, episodes)

    if not output.exists():
        if previous is None:
            print("[Pi0.5 incremental] no preceding dataset; performing the one-time full materialization")
            _full_materialization(args)
        else:
            _validate_prefix(previous, identities)
            print(f"[Pi0.5 incremental] cloning preceding dataset with hard-linked videos: {previous}")
            _clone_dataset(previous, output)

    marker = output / IN_PROGRESS
    marker.write_text("incremental Pi0.5 RECAP update in progress\n", encoding="utf-8")
    existing = _validate_prefix(output, identities)
    if existing < len(episodes):
        print(
            f"[Pi0.5 incremental] reusing {existing} episodes and materializing "
            f"{len(episodes) - existing} new rollout episodes"
        )
        _append_new_episodes(output, args.repo_id, source_root, source_info, episodes, labels, existing)
    else:
        print(f"[Pi0.5 incremental] reusing all {existing} episode videos")

    print(f"[Pi0.5 incremental] updating advantage conditions for {len(episodes)} episodes")
    _rewrite_advantage_conditions(output, episodes, labels)
    _atomic_json(
        output / MANIFEST,
        {
            "schema_version": 1,
            "type": "pi05_recap_incremental",
            "source_dataset": str(source_root),
            "advantage_checkpoint": advantage_header.get("wcm_checkpoint", ""),
            "episodes": identities,
        },
    )
    # Loading through the same API used by OpenPI catches malformed parquet,
    # metadata, task maps, and missing packed videos before training starts.
    validated = LeRobotDataset(repo_id=args.repo_id, root=output)
    if validated.meta.total_episodes != len(episodes):
        raise RuntimeError("Incremental Pi0.5 dataset failed final episode-count validation.")
    marker.unlink()
    print(
        f"saved incremental LeRobot dataset: {output} episodes={len(episodes)} "
        f"reused={existing} appended={len(episodes) - existing}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--advantage-labels", required=True)
    parser.add_argument("--previous-dataset", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--mode", choices=("image", "video"), default="video")
    main(parser.parse_args())
