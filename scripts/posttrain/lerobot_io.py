"""Strict readers for RoboDojo LeRobot v2.1 and v3.0 datasets.

The two layouts are intentionally handled by separate code paths. Callers
must declare the expected format; a dataset is never silently read as a
different version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as parquet

SUPPORTED_FORMATS = ("v2.1", "v3.0")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_format(value: str) -> str:
    normalized = str(value).strip().lower().removeprefix("lerobot_")
    aliases = {"2.1": "v2.1", "3.0": "v3.0", "v21": "v2.1", "v30": "v3.0"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported LeRobot format {value!r}; expected one of {SUPPORTED_FORMATS}."
        )
    return normalized


class LeRobotLayout:
    """Version-specific metadata, parquet, and video path access."""

    def __init__(self, root: str | Path, expected_format: str) -> None:
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata is missing: {info_path}")
        self.info = _json(info_path)
        self.format = normalize_format(expected_format)
        actual = normalize_format(str(self.info.get("codebase_version", "")))
        if actual != self.format:
            raise ValueError(
                f"Configured LeRobot format is {self.format}, but {info_path} declares {actual}. "
                "One RECAP run may read only the configured source format."
            )
        self.chunk_size = int(self.info.get("chunks_size", 1000))

    def episodes(self) -> list[dict[str, Any]]:
        if self.format == "v2.1":
            path = self.root / "meta/episodes.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"LeRobot v2.1 episode metadata is missing: {path}")
            rows = _jsonl(path)
        else:
            paths = sorted((self.root / "meta/episodes").glob("**/*.parquet"))
            if not paths:
                raise FileNotFoundError(
                    f"LeRobot v3.0 episode metadata is missing below {self.root / 'meta/episodes'}"
                )
            rows = []
            for path in paths:
                rows.extend(parquet.read_table(path).to_pylist())
        rows.sort(key=lambda row: int(row["episode_index"]))
        indices = [int(row["episode_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate episode_index values in {self.root}")
        return rows

    def tasks(self) -> list[dict[str, Any]]:
        if self.format == "v2.1":
            path = self.root / "meta/tasks.jsonl"
            return _jsonl(path) if path.is_file() else []

        path = self.root / "meta/tasks.parquet"
        if not path.is_file():
            return []
        rows = parquet.read_table(path).to_pylist()
        result = []
        for row in rows:
            task = row.get("task")
            if task is None:
                task = row.get("__index_level_0__")
            if task is None:
                raise ValueError(f"LeRobot v3.0 task row has no task text: {row}")
            result.append({"task_index": int(row["task_index"]), "task": str(task)})
        return result

    def read_episode(
        self,
        metadata: dict[str, Any],
        columns: Iterable[str],
    ):
        episode = int(metadata["episode_index"])
        if self.format == "v2.1":
            template = self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            relative = template.format(
                episode_chunk=episode // self.chunk_size,
                episode_index=episode,
            )
            path = self.root / relative
            table = parquet.read_table(path, columns=list(columns))
        else:
            chunk = int(metadata["data/chunk_index"])
            file_index = int(metadata["data/file_index"])
            template = self.info.get(
                "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            )
            path = self.root / template.format(chunk_index=chunk, file_index=file_index)
            table = parquet.read_table(
                path,
                columns=list(columns),
                filters=[("episode_index", "=", episode)],
            )
        expected = int(metadata.get("length", table.num_rows))
        if table.num_rows != expected:
            raise ValueError(
                f"Episode {episode} metadata length={expected}, parquet rows={table.num_rows}: {path}"
            )
        return table

    def video_path(self, metadata: dict[str, Any], video_key: str) -> Path:
        episode = int(metadata["episode_index"])
        if self.format == "v2.1":
            template = self.info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            relative = template.format(
                video_key=video_key,
                episode_chunk=episode // self.chunk_size,
                episode_index=episode,
            )
        else:
            prefix = f"videos/{video_key}"
            chunk = int(metadata[f"{prefix}/chunk_index"])
            file_index = int(metadata[f"{prefix}/file_index"])
            from_timestamp = float(metadata.get(f"{prefix}/from_timestamp", 0.0))
            if abs(from_timestamp) > 1e-6:
                raise ValueError(
                    "RoboDojo's RECAP reader requires one v3.0 video file per episode; "
                    f"episode={episode}, key={video_key} starts at {from_timestamp}s."
                )
            template = self.info.get(
                "video_path",
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            )
            relative = template.format(
                video_key=video_key,
                chunk_index=chunk,
                file_index=file_index,
            )
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing episode video for {video_key}: {path}")
        return path
