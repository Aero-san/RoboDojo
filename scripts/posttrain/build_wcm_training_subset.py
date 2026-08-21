"""Build one WCM update set from replayed old episodes plus all new episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

try:
    import build_replay_buffer as replay
    from progress import progress_iter
except ModuleNotFoundError:
    from scripts.posttrain import build_replay_buffer as replay
    from scripts.posttrain.progress import progress_iter


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Replay episode is missing required column {name!r}.")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def main(args: argparse.Namespace) -> None:
    source = Path(args.buffer).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"WCM training subset already exists: {output}")
    if args.old_episode_count < 0 or args.replay_episodes < 0:
        raise ValueError("Episode counts must be non-negative.")

    info = _json(source / "meta/info.json")
    episodes = _jsonl(source / "meta/episodes.jsonl")
    provenance = _jsonl(source / "meta/provenance.jsonl")
    labels = {int(key): bool(value) for key, value in _json(source / "meta/success_labels.json").items()}
    total = int(info.get("total_episodes", -1))
    if total != len(episodes) or total != len(provenance) or len(labels) != total:
        raise ValueError(f"Source replay buffer metadata is incomplete: {source}")
    if args.old_episode_count > total:
        raise ValueError(
            f"Old prefix has {args.old_episode_count} episodes but current buffer has {total}."
        )

    replay_count = min(args.replay_episodes, args.old_episode_count)
    old_ids = sorted(random.Random(args.seed).sample(range(args.old_episode_count), replay_count))
    new_ids = list(range(args.old_episode_count, total))
    selected = old_ids + new_ids
    if not selected:
        raise ValueError("WCM training subset would contain no episodes.")

    output.mkdir(parents=True)
    (output / "meta").mkdir()
    marker = output / "meta/.incremental_update_in_progress"
    marker.write_text("WCM training subset construction in progress\n", encoding="utf-8")

    source_chunk_size = int(info.get("chunks_size", 1000))
    output_chunk_size = int(args.chunk_size)
    data_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    tasks: dict[str, int] = {}
    output_episodes = []
    output_provenance = []
    output_labels: dict[str, bool] = {}
    global_index = 0
    total_videos = 0

    for output_episode, source_episode in enumerate(
        progress_iter(selected, desc="Building WCM update subset", total=len(selected), unit="episode")
    ):
        metadata = episodes[source_episode]
        task = str(metadata.get("tasks", [""])[0])
        task_index = tasks.setdefault(task, len(tasks))
        source_path = source / data_template.format(
            episode_chunk=source_episode // source_chunk_size,
            episode_index=source_episode,
        )
        table = parquet.read_table(source_path)
        length = table.num_rows
        table = _replace_column(table, "episode_index", np.full(length, output_episode, dtype=np.int64))
        table = _replace_column(table, "frame_index", np.arange(length, dtype=np.int64))
        table = _replace_column(
            table, "index", np.arange(global_index, global_index + length, dtype=np.int64)
        )
        table = _replace_column(table, "task_index", np.full(length, task_index, dtype=np.int64))
        output_chunk = output_episode // output_chunk_size
        output_path = output / f"data/chunk-{output_chunk:03d}/episode_{output_episode:06d}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(table, output_path)

        for camera in replay.CAMERAS:
            video = replay._video_source(source, info, source_episode, camera)
            replay._link(
                video,
                output
                / f"videos/chunk-{output_chunk:03d}/observation.images.{camera}/episode_{output_episode:06d}.mp4",
            )
            total_videos += 1

        output_episodes.append({"episode_index": output_episode, "tasks": [task], "length": length})
        output_labels[str(output_episode)] = labels[source_episode]
        source_provenance = dict(provenance[source_episode])
        source_provenance.update(
            {
                "episode_index": output_episode,
                "buffer_episode_index": source_episode,
                "sampling_role": "old_replay" if source_episode < args.old_episode_count else "new",
            }
        )
        output_provenance.append(source_provenance)
        global_index += length

    output_info = dict(info)
    output_info.update(
        {
            "total_episodes": len(selected),
            "total_frames": global_index,
            "total_tasks": len(tasks),
            "total_videos": total_videos,
            "total_chunks": (len(selected) + output_chunk_size - 1) // output_chunk_size,
            "chunks_size": output_chunk_size,
            "splits": {"train": f"0:{len(selected)}"},
        }
    )
    _write_json(output / "meta/info.json", output_info)
    (output / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in output_episodes), encoding="utf-8"
    )
    (output / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": task}) + "\n"
            for task, index in sorted(tasks.items(), key=lambda item: item[1])
        ),
        encoding="utf-8",
    )
    _write_json(output / "meta/success_labels.json", output_labels)
    (output / "meta/provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in output_provenance), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "type": "wcm_update_subset",
        "source_buffer": str(source),
        "source_episodes": total,
        "old_prefix_episodes": args.old_episode_count,
        "requested_old_replay_episodes": args.replay_episodes,
        "sampled_old_episodes": len(old_ids),
        "new_episodes": len(new_ids),
        "episodes": len(selected),
        "frames": global_index,
        "successes": sum(output_labels.values()),
        "failures": len(output_labels) - sum(output_labels.values()),
        "seed": args.seed,
        "selected_buffer_episode_ids": selected,
    }
    _write_json(output / "meta/replay_buffer.json", summary)
    marker.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-episode-count", type=int, required=True)
    parser.add_argument("--replay-episodes", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3072)
    main(parser.parse_args())
