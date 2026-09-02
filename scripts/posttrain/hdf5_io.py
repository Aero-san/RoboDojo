"""Strict, streaming reader for RoboDojo trajectory HDF5 demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import cv2
import h5py
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit

JOINT_FIELDS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)
JOINT_NAMES = (
    *(f"left_arm_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_arm_joint_{index}" for index in range(6)),
    "right_gripper",
)
CAMERA_DATASETS = {
    "cam_high": (
        "vision/cam_head/colors",
        "vision/cam_high/colors",
        "vision/cam_third_view/colors",
    ),
    "cam_left_wrist": (
        "vision/cam_left_wrist/colors",
        "vision/cam_wrist/colors",
    ),
    "cam_right_wrist": (
        "vision/cam_right_wrist/colors",
        "vision/cam_wrist/colors",
    ),
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug.removesuffix("_random")


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (bytes, np.bytes_)):
        return [bytes(value).decode("utf-8")]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if decoded != value:
            return _strings(decoded)
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _strings(value.item())
        result: list[str] = []
        for item in value.reshape(-1).tolist():
            result.extend(_strings(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_strings(item))
        return result
    return [str(value)]


def _sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"RoboDojo HDF5 sidecar must be a JSON object: {sidecar}")
    return payload


def _episode_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"HDF5 episode filename has no numeric suffix: {path}")
    return int(match.group(1))


def _array_2d(dataset: h5py.Dataset, name: str) -> np.ndarray:
    value = np.asarray(dataset, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"{name} must be [T,D], got {value.shape}")
    return value


def _joint_vector(group: h5py.Group, name: str) -> np.ndarray:
    missing = [field for field in JOINT_FIELDS if field not in group]
    if missing:
        raise KeyError(f"{name} is missing joint fields {missing}")
    parts = [_array_2d(group[field], f"{name}/{field}") for field in JOINT_FIELDS]
    horizons = {len(part) for part in parts}
    if len(horizons) != 1:
        raise ValueError(f"{name} joint fields have mismatched horizons: {sorted(horizons)}")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _actions(handle: h5py.File, states: np.ndarray) -> np.ndarray:
    if "action" in handle:
        node = handle["action"]
        if isinstance(node, h5py.Group):
            actions = _joint_vector(node, "action")
        else:
            actions = _array_2d(node, "action")
        if actions.shape != states.shape:
            raise ValueError(
                f"HDF5 state/action shape mismatch: {states.shape} vs {actions.shape}"
            )
        return actions

    # RoboDojo's public RGB subset omits privileged action targets. Follow the
    # repository's existing Pi conversion convention: the next observed joint
    # state is the position target and the final target holds the last state.
    actions = np.empty_like(states, dtype=np.float32)
    if len(states) > 1:
        actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


def _instruction(handle: h5py.File, sidecar: dict[str, Any], task: str) -> str:
    for key in ("instruction", "instructions"):
        if key in handle:
            values = [value.strip() for value in _strings(handle[key][()]) if value.strip()]
            if values:
                return values[0]
    values = [value.strip() for value in _strings(sidecar.get("instruction")) if value.strip()]
    return values[0] if values else task


def _fps(handle: h5py.File, sidecar: dict[str, Any]) -> float:
    value = handle.attrs.get("fps")
    if value is None and "additional_info/frequency" in handle:
        value = handle["additional_info/frequency"][()]
    if value is None:
        value = sidecar.get("fps", 25)
    fps = float(np.asarray(value).reshape(-1)[0])
    if fps <= 0:
        raise ValueError(f"HDF5 fps must be positive, got {fps}")
    return fps


def _decode_frame(value: Any, source: str) -> np.ndarray:
    image = np.asarray(decode_image_bit(value))
    if image.ndim == 4 and len(image) == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Decoded HDF5 image must be HWC RGB, got {image.shape}: {source}")
    return image.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class Hdf5VideoSource:
    """A camera stream kept inside one HDF5 episode until output materialization."""

    episode_path: Path
    dataset_path: str
    frames: int

    def frame_shape(self) -> tuple[int, int, int]:
        with h5py.File(self.episode_path, "r") as handle:
            return tuple(
                _decode_frame(handle[self.dataset_path][0], str(self.episode_path)).shape
            )

    def materialize(self, destination: Path, fps: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
        temporary.unlink(missing_ok=True)
        writer: cv2.VideoWriter | None = None
        expected_shape: tuple[int, int, int] | None = None
        try:
            with h5py.File(self.episode_path, "r") as handle:
                dataset = handle[self.dataset_path]
                if len(dataset) != self.frames:
                    raise ValueError(
                        f"HDF5 camera horizon changed for {self.dataset_path}: "
                        f"expected {self.frames}, got {len(dataset)}"
                    )
                for index, raw in enumerate(dataset):
                    image = _decode_frame(
                        raw,
                        f"{self.episode_path}:{self.dataset_path}[{index}]",
                    )
                    if writer is None:
                        height, width = image.shape[:2]
                        expected_shape = tuple(image.shape)
                        writer = cv2.VideoWriter(
                            str(temporary),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create HDF5 camera video: {temporary}")
                    elif tuple(image.shape) != expected_shape:
                        raise ValueError(
                            f"HDF5 camera changes image shape from {expected_shape} "
                            f"to {tuple(image.shape)}: {self.episode_path}:{self.dataset_path}"
                        )
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if writer is None:
                raise ValueError(
                    f"HDF5 camera contains no frames: {self.episode_path}:{self.dataset_path}"
                )
            writer.release()
            writer = None
            temporary.replace(destination)
        except Exception:
            if writer is not None:
                writer.release()
            temporary.unlink(missing_ok=True)
            raise


class Hdf5DemoSource:
    """Discover and load exactly one configured RoboDojo HDF5 task source."""

    def __init__(self, root: str | Path, task: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.task = task.strip()
        if not self.root.exists():
            raise FileNotFoundError(f"RoboDojo HDF5 root does not exist: {self.root}")

    def _candidate_files(self) -> list[Path]:
        if self.root.is_file():
            if self.root.suffix.lower() not in {".hdf5", ".h5"}:
                raise ValueError(f"Configured HDF5 source is not .hdf5/.h5: {self.root}")
            return [self.root]

        search_root = self.root
        task_root = self.root / self.task
        if self.task and task_root.is_dir():
            search_root = task_root
        files = sorted(search_root.rglob("episode_*.hdf5"))
        files.extend(sorted(search_root.rglob("episode_*.h5")))
        unique = {path.resolve(): path.resolve() for path in files}
        if not unique:
            raise FileNotFoundError(f"No episode_*.hdf5/.h5 files below: {search_root}")
        return sorted(unique.values())

    def load(
        self,
        max_episodes: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        available_tasks: set[str] = set()
        requested_slug = _slug(self.task)
        for path in self._candidate_files():
            sidecar = _sidecar(path)
            with h5py.File(path, "r") as handle:
                raw_task = handle.attrs.get("task", sidecar.get("task", ""))
                canonical_task = (_strings(raw_task) or [self.task])[0].strip()
                if canonical_task:
                    available_tasks.add(canonical_task)
                if requested_slug and canonical_task and _slug(canonical_task) != requested_slug:
                    continue
                if "state" not in handle or not isinstance(handle["state"], h5py.Group):
                    raise KeyError(f"RoboDojo HDF5 episode has no state group: {path}")
                states = _joint_vector(handle["state"], "state")
                if not len(states):
                    raise ValueError(f"RoboDojo HDF5 episode contains no frames: {path}")
                actions = _actions(handle, states)
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError(f"RoboDojo HDF5 state/action contains non-finite values: {path}")
                fps = _fps(handle, sidecar)
                videos: dict[str, Hdf5VideoSource] = {}
                for camera, candidates in CAMERA_DATASETS.items():
                    dataset_path = next((key for key in candidates if key in handle), None)
                    if dataset_path is None:
                        raise KeyError(f"RoboDojo HDF5 episode is missing {camera}: {path}")
                    dataset = handle[dataset_path]
                    if not isinstance(dataset, h5py.Dataset) or len(dataset) != len(states):
                        raise ValueError(
                            f"HDF5 {camera} horizon does not match state: {path}:{dataset_path}"
                        )
                    videos[camera] = Hdf5VideoSource(path, dataset_path, len(states))
                records.append(
                    {
                        "kind": "demo",
                        "source": str(path),
                        "source_episode": _episode_index(path),
                        "task": _instruction(handle, sidecar, canonical_task or self.task),
                        "task_slug": canonical_task or self.task,
                        "success": True,
                        "score": 1.0,
                        "fps": fps,
                        "states": states,
                        "actions": actions,
                        "reference_actions": actions.copy(),
                        "videos": videos,
                    }
                )
            if max_episodes > 0 and len(records) >= max_episodes:
                break

        if not records:
            raise ValueError(
                f"Task {self.task!r} matched no HDF5 episodes below {self.root}; "
                f"available tasks={sorted(available_tasks)}"
            )
        fps_values = {record["fps"] for record in records}
        if len(fps_values) != 1:
            raise ValueError(f"HDF5 demonstrations mix fps values: {sorted(fps_values)}")
        dimensions = {
            (record["states"].shape[1], record["actions"].shape[1])
            for record in records
        }
        if len(dimensions) != 1:
            raise ValueError(f"HDF5 demonstrations mix dimensions: {sorted(dimensions)}")
        state_dim, action_dim = dimensions.pop()
        first_videos = records[0]["videos"]
        camera_shapes = {
            camera: source.frame_shape() for camera, source in first_videos.items()
        }
        for record in records[1:]:
            for camera, source in record["videos"].items():
                shape = source.frame_shape()
                if shape != camera_shapes[camera]:
                    raise ValueError(
                        f"HDF5 {camera} image shape {shape} does not match "
                        f"{camera_shapes[camera]}: {source.episode_path}"
                    )
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": list(JOINT_NAMES) if state_dim == len(JOINT_NAMES) else None,
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": list(JOINT_NAMES) if action_dim == len(JOINT_NAMES) else None,
            },
        }
        for camera, (height, width, channels) in camera_shapes.items():
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": [channels, height, width],
                "names": ["channels", "height", "width"],
            }
        info = {
            "codebase_version": "hdf5",
            "source_format": "hdf5",
            "fps": fps_values.pop(),
            "robot_type": "arx_x5",
            "features": features,
        }
        return info, records
