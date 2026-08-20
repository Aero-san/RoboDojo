"""Convert RoboDojo's v2.1 video export into the LeRobot tree used by Pi0.5.

This keeps the source export untouched and optionally filters episodes using a
WCM success-label JSON file.  The resulting repo can be passed to OpenPI with
``OPENPI_LEROBOT_REPO_ID``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as parquet

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError as exc:
    if exc.name != "lerobot.datasets":
        raise
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

try:
    from progress import progress_iter, tqdm_print_bridge
    from robodojo_dataset import filter_episode_metadata
except ModuleNotFoundError:
    from scripts.posttrain.progress import progress_iter, tqdm_print_bridge
    from scripts.posttrain.robodojo_dataset import filter_episode_metadata


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels(path: str | None) -> dict[int, bool]:
    if not path:
        return {}
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return {int(key): bool(value) for key, value in payload.items()}


def _advantage_labels(path: str | None) -> dict[int, list[bool]]:
    if not path:
        return {}
    lines = [line for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("RECAP advantage label file is empty.")
    header = json.loads(lines[0])
    if header.get("schema_version") != 1 or header.get("type") != "recap_advantages":
        raise ValueError("Unsupported RECAP advantage label format.")
    result: dict[int, list[bool]] = {}
    for line in lines[1:]:
        row = json.loads(line)
        episode = int(row["episode_index"])
        result[episode] = [bool(value) for value in row["positive"]]
    return result


class _SequentialVideo:
    """Decode one episode without materializing all camera frames in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = None
        self.container = None
        self.iterator = None
        self.frame_index = 0
        if os.environ.get("WCM_VIDEO_DECODER", "pyav").lower() == "opencv":
            self.capture = cv2.VideoCapture(str(path))
        if self.capture is None or not self.capture.isOpened():
            if self.capture is not None:
                self.capture.release()
            self._use_pyav()

    def _use_pyav(self) -> None:
        import av

        self.container = av.open(str(self.path))
        self.iterator = self.container.decode(video=0)

    def read(self) -> np.ndarray:
        if self.iterator is not None:
            try:
                image = next(self.iterator).to_ndarray(format="rgb24")
            except StopIteration as exc:
                raise RuntimeError(f"Could not decode frame {self.frame_index} from {self.path}") from exc
        else:
            try:
                ok, image = self.capture.read()
            except Exception:
                ok = False
                image = None
            if not ok:
                self.capture.release()
                self._use_pyav()
                return self.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.frame_index += 1
        return image

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.container is not None:
            self.container.close()

    def __enter__(self) -> _SequentialVideo:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _feature_names(info: dict[str, Any], key: str, dim: int) -> list[str]:
    names = info["features"].get(key, {}).get("names")
    if isinstance(names, list) and names and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or len(names) != dim:
        return [f"joint_{index}" for index in range(dim)]
    return [str(name) for name in names]


def _create_dataset(repo_id: str, info: dict[str, Any], mode: str) -> LeRobotDataset:
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    state_names = _feature_names(info, "observation.state", state_dim)
    action_names = _feature_names(info, "action", action_dim)
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": state_names},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
    }
    for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        features[f"observation.images.{name}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    output = HF_LEROBOT_HOME / repo_id
    if output.exists():
        raise FileExistsError(f"Output dataset already exists; remove it first: {output}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(info.get("fps", 25)),
        robot_type=str(info.get("robot_type", "unified_robot")),
        features=features,
        use_videos=mode == "video",
    )


def main(args: argparse.Namespace) -> None:
    root = Path(args.dataset_root).expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _jsonl(root / "meta" / "episodes.jsonl")
    task_metadata_path = root / "meta" / "tasks.jsonl"
    task_metadata = _jsonl(task_metadata_path) if task_metadata_path.exists() else None
    episodes, _ = filter_episode_metadata(episodes, args.task, task_metadata)
    labels = _labels(args.episode_labels)
    advantage_labels = _advantage_labels(args.advantage_labels)
    if labels and any(int(row["episode_index"]) not in labels for row in episodes):
        missing = [int(row["episode_index"]) for row in episodes if int(row["episode_index"]) not in labels]
        raise ValueError(f"Episode label file is missing {len(missing)} episodes, first={missing[:5]}")
    if advantage_labels and any(int(row["episode_index"]) not in advantage_labels for row in episodes):
        missing = [int(row["episode_index"]) for row in episodes if int(row["episode_index"]) not in advantage_labels]
        raise ValueError(f"RECAP labels are missing {len(missing)} episodes, first={missing[:5]}")
    selected = [
        row
        for row in episodes
        if not labels or labels[int(row["episode_index"])]
    ]
    if args.max_episodes > 0:
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("No episodes remain after WCM label filtering.")
    dataset = _create_dataset(args.repo_id, info, args.mode)
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunks_size = int(info.get("chunks_size", 1000))
    with tqdm_print_bridge():
        for episode_meta in progress_iter(
            selected,
            desc="Converting Pi0.5 dataset",
            total=len(selected),
            unit="episode",
        ):
            episode = int(episode_meta["episode_index"])
            chunk = episode // chunks_size
            parquet_path = root / data_template.format(episode_chunk=chunk, episode_index=episode)
            table = parquet.read_table(parquet_path, columns=["action", "observation.state", "frame_index"])
            frame_count = table.num_rows
            video_paths: dict[str, Path] = {}
            for output_name, source_key in (
                ("cam_high", "observation.images.cam_high"),
                ("cam_left_wrist", "observation.images.cam_left_wrist"),
                ("cam_right_wrist", "observation.images.cam_right_wrist"),
            ):
                source_path = root / video_template.format(
                    video_key=source_key,
                    episode_chunk=chunk,
                    episode_index=episode,
                )
                if not source_path.exists():
                    # Single-arm exports may have only cam_wrist.  Duplicating it
                    # is the least surprising way to satisfy Pi0.5's three-input
                    # ALOHA transform while preserving the real head view.
                    source_path = root / video_template.format(
                        video_key="observation.images.cam_wrist",
                        episode_chunk=chunk,
                        episode_index=episode,
                    )
                if not source_path.exists():
                    raise FileNotFoundError(f"No camera video for episode {episode}: {source_key}")
                video_paths[output_name] = source_path
            task = str(episode_meta.get("tasks", ["Perform the RoboDojo task."])[0])
            episode_advantages = advantage_labels.get(episode)
            if episode_advantages is not None and len(episode_advantages) != frame_count:
                raise ValueError(
                    f"RECAP labels for episode={episode} have {len(episode_advantages)} frames, expected {frame_count}."
                )
            actions = table["action"].to_pylist()
            states = table["observation.state"].to_pylist()
            with (
                _SequentialVideo(video_paths["cam_high"]) as high_video,
                _SequentialVideo(video_paths["cam_left_wrist"]) as left_video,
                _SequentialVideo(video_paths["cam_right_wrist"]) as right_video,
            ):
                for index in range(frame_count):
                    frame_task = task
                    if episode_advantages is not None:
                        condition = "positive" if episode_advantages[index] else "negative"
                        frame_task = f"{task}\nAdvantage: {condition}"
                    frame: dict[str, Any] = {
                        "observation.state": np.asarray(states[index], dtype=np.float32),
                        "action": np.asarray(actions[index], dtype=np.float32),
                        "task": frame_task,
                        "observation.images.cam_high": high_video.read(),
                        "observation.images.cam_left_wrist": left_video.read(),
                        "observation.images.cam_right_wrist": right_video.read(),
                    }
                    dataset.add_frame(frame)
            dataset.save_episode()
    dataset.hf_dataset = dataset.create_hf_dataset()
    print(f"saved LeRobot dataset: {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode-labels", default="")
    parser.add_argument(
        "--advantage-labels",
        default="",
        help="RECAP frame labels emitted by annotate_recap_advantages.py; keeps both successful and failed episodes.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="One task instruction or benchmark slug such as stack_bowls.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--mode", choices=("image", "video"), default="image")
    main(parser.parse_args())
