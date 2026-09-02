"""Append one rollout round to a preceding RoboDojo replay buffer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import pyarrow.parquet as parquet

try:
    import build_replay_buffer as replay
    from progress import progress_iter
except ModuleNotFoundError:
    from scripts.posttrain import build_replay_buffer as replay
    from scripts.posttrain.progress import progress_iter


def _clone(source: Path, destination: Path) -> None:
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


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _normalize_task_slugs(episodes: list[dict], task: str) -> None:
    if not task.strip():
        raise ValueError("--task is required for a single-task RECAP replay buffer.")
    for episode in episodes:
        episode["task_slug"] = task


def main(args: argparse.Namespace) -> None:
    previous = Path(args.previous_buffer).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Replay buffer output already exists: {output}")
    info = replay._json(previous / "meta/info.json")
    episodes = replay._jsonl(previous / "meta/episodes.jsonl")
    _normalize_task_slugs(episodes, args.task)
    provenance = replay._jsonl(previous / "meta/provenance.jsonl")
    labels = {str(key): bool(value) for key, value in replay._json(previous / "meta/success_labels.json").items()}
    task_rows = replay._jsonl(previous / "meta/tasks.jsonl")
    tasks = {str(row["task"]): int(row["task_index"]) for row in task_rows}
    if int(info.get("total_episodes", -1)) != len(episodes) or len(provenance) != len(episodes):
        raise ValueError(f"Previous replay buffer metadata is incomplete: {previous}")

    records = replay._rollout_records(
        rollout_root,
        args.task,
        args.max_rollout_episodes,
        args.seed,
    )
    if not records:
        raise ValueError(f"No new rollout episodes found below {rollout_root}")
    prior_sources = {
        (str(row.get("source", "")), int(row.get("source_episode", -1)), str(row.get("run_id", "")))
        for row in provenance
    }
    duplicates = [
        (str(record["source"]), int(record["source_episode"]), str(record.get("run_id", "")))
        for record in records
        if (str(record["source"]), int(record["source_episode"]), str(record.get("run_id", ""))) in prior_sources
    ]
    if duplicates:
        raise ValueError(f"New rollout round overlaps the preceding replay buffer: {duplicates[:5]}")

    dimensions = {(record["states"].shape[1], record["actions"].shape[1]) for record in records}
    expected = (
        int(info["features"]["observation.state"]["shape"][0]),
        int(info["features"]["action"]["shape"][0]),
    )
    if dimensions != {expected}:
        raise ValueError(f"New rollout dimensions {sorted(dimensions)} do not match preceding buffer {expected}.")

    _clone(previous, output)
    marker = output / "meta/.incremental_update_in_progress"
    marker.write_text("incremental replay-buffer update in progress\n", encoding="utf-8")
    chunk_size = int(info.get("chunks_size", 1000))
    global_index = int(info["total_frames"])
    next_episode = len(episodes)
    for offset, record in enumerate(
        progress_iter(records, desc="Appending replay rollouts", total=len(records), unit="episode")
    ):
        episode_index = next_episode + offset
        task_index = tasks.setdefault(record["task"], len(tasks))
        table = replay._episode_table(
            record["actions"],
            record["reference_actions"],
            record["states"],
            episode_index,
            task_index,
            global_index,
            record["fps"],
        )
        chunk = episode_index // chunk_size
        data_path = output / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, data_path)
        for camera, source in record["videos"].items():
            replay._link(
                Path(source),
                output
                / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{episode_index:06d}.mp4",
            )
        length = len(record["actions"])
        episode_meta = {
            "episode_index": episode_index,
            "tasks": [record["task"]],
            "length": length,
        }
        if record.get("task_slug"):
            episode_meta["task_slug"] = str(record["task_slug"])
        episodes.append(episode_meta)
        labels[str(episode_index)] = bool(record["success"])
        provenance.append(
            {
                "episode_index": episode_index,
                "source_kind": "rollout",
                "source": record["source"],
                "source_episode": record["source_episode"],
                "run_id": record.get("run_id", ""),
                "success": bool(record["success"]),
                "score": float(record["score"]),
            }
        )
        global_index += length

    info.update(
        {
            "total_episodes": len(episodes),
            "total_frames": global_index,
            "total_tasks": len(tasks),
            "total_videos": 3 * len(episodes),
            "total_chunks": (len(episodes) + chunk_size - 1) // chunk_size,
            "splits": {"train": f"0:{len(episodes)}"},
        }
    )
    _write_json(output / "meta/info.json", info)
    (output / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (output / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": task}) + "\n"
            for task, index in sorted(tasks.items(), key=lambda item: item[1])
        ),
        encoding="utf-8",
    )
    _write_json(output / "meta/success_labels.json", labels)
    (output / "meta/provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "task": args.task,
        "episodes": len(episodes),
        "demonstrations": sum(row.get("source_kind") == "demo" for row in provenance),
        "rollouts": sum(row.get("source_kind") == "rollout" for row in provenance),
        "successes": sum(labels.values()),
        "failures": len(labels) - sum(labels.values()),
        "frames": global_index,
        "incremental_parent": str(previous),
        "incremental_rollout": str(rollout_root),
    }
    _write_json(output / "meta/replay_buffer.json", summary)
    marker.unlink()
    print(
        f"saved incremental replay buffer: {output} reused={next_episode} "
        f"appended={len(records)} total={len(episodes)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-buffer", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-rollout-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
