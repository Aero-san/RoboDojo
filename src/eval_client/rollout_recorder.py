"""Stream RoboDojo policy rollouts to a trainer-independent raw format.

The simulator environment intentionally does not depend on LeRobot or
PyArrow.  Each completed episode contains a compressed state/action archive,
one MP4 per policy camera, and a JSON outcome manifest.  Post-training tools
materialize these episodes into the WCM/LeRobot replay buffer afterwards.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from utils.save_file import VideoStreamWriter

_CAMERA_CANDIDATES = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left", "cam_wrist"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right", "cam_wrist"),
}


def _state_mapping(observation: dict[str, Any]) -> dict[str, Any]:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("Rollout observations must contain a dictionary-valued 'state'.")
    return state


def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in mapping:
            return np.asarray(mapping[name], dtype=np.float32).reshape(-1)
    raise KeyError(f"None of the required state/action keys are present: {names}")


def pack_robot_vector(
    mapping: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict[str, Any],
) -> np.ndarray:
    """Pack single/dual-arm RoboDojo state dictionaries for every robot config."""

    arm_dims = list(robot_action_dim_info["arm_dim"])
    ee_dims = list(robot_action_dim_info["ee_dim"])
    if len(arm_dims) != len(ee_dims) or len(arm_dims) not in (1, 2):
        raise ValueError(f"Unsupported robot action layout: {robot_action_dim_info}")

    prefixes = ("",) if len(arm_dims) == 1 else ("left_", "right_")
    parts: list[np.ndarray] = []
    for prefix, arm_dim, ee_dim in zip(prefixes, arm_dims, ee_dims, strict=True):
        if action_type == "joint":
            arm_names = (
                f"{prefix}arm_joint_state",
                f"{prefix}joint_state",
            )
        elif action_type == "ee":
            arm_names = (
                f"{prefix}ee_pose",
                f"{prefix}tcp_pose",
                f"{prefix}delta_ee_pose",
            )
            # Robot metadata stores joint action dimensions.  End-effector
            # actions always use xyz + quaternion in RoboDojo.
            arm_dim = 7
        else:
            raise ValueError(f"Unsupported rollout action_type={action_type!r}")
        ee_names = (f"{prefix}ee_joint_state",)
        arm = _first(mapping, arm_names)
        ee = _first(mapping, ee_names)
        if arm.size != arm_dim or ee.size != ee_dim:
            raise ValueError(
                f"Rollout vector dimension mismatch for prefix={prefix!r}: "
                f"arm={arm.size}/{arm_dim}, ee={ee.size}/{ee_dim}."
            )
        parts.extend((arm, ee))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _color_frame(observation: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray | None:
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        return None
    for name in candidates:
        value = vision.get(name)
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        if value is None:
            continue
        image = np.asarray(value)
        if image.ndim != 3:
            continue
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (3, 4):
            image = np.transpose(image, (1, 2, 0))
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8, copy=False)
        if image.shape[-1] not in (3, 4):
            continue
        return np.ascontiguousarray(image)
    return None


class RolloutRecorder:
    """Record only completed, simulator-labelled policy trajectories."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        run_id: str,
        fps: float,
        robot_action_dim_info: dict[str, Any],
        task_name: str,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = run_id
        self.fps = float(fps)
        self.robot_action_dim_info = robot_action_dim_info
        self.task_name = task_name
        self.in_progress_root = self.root / "_in_progress" / run_id
        self.episodes_root = self.root / "episodes"
        self.in_progress_root.mkdir(parents=True, exist_ok=True)
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self._pending: dict[int, dict[str, Any]] = {}
        self._states: dict[int, list[np.ndarray]] = {}
        self._actions: dict[int, list[np.ndarray]] = {}
        self._reference_actions: dict[int, list[np.ndarray]] = {}
        self._writers: dict[int, dict[str, VideoStreamWriter]] = {}
        self._action_types: dict[int, str] = {}
        self._tasks: dict[int, str] = {}

    def observe(self, env_idx: int, observation: dict[str, Any]) -> None:
        """Cache the pre-action observation; it is committed with the next action."""

        state = {key: np.asarray(value).copy() for key, value in _state_mapping(observation).items()}
        images: dict[str, np.ndarray] = {}
        high = _color_frame(observation, _CAMERA_CANDIDATES["cam_high"])
        for name, candidates in _CAMERA_CANDIDATES.items():
            image = _color_frame(observation, candidates)
            if image is None:
                image = high
            if image is None:
                raise KeyError(f"Rollout observation has no usable image for {name}.")
            images[name] = image.copy()
        instruction = str(observation.get("instruction") or self.task_name).strip()
        self._pending[env_idx] = {"state": state, "images": images, "instruction": instruction}

    def record_action(
        self,
        env_idx: int,
        action: dict[str, Any],
        action_type: str,
        *,
        reference_action: dict[str, Any] | None = None,
    ) -> None:
        observation = self._pending.pop(env_idx, None)
        if observation is None:
            raise RuntimeError(f"No pre-action observation was recorded for env_idx={env_idx}.")
        previous_type = self._action_types.setdefault(env_idx, action_type)
        if previous_type != action_type:
            raise ValueError(f"Episode changed action type from {previous_type} to {action_type}.")
        state = pack_robot_vector(observation["state"], action_type, self.robot_action_dim_info)
        packed_action = pack_robot_vector(action, action_type, self.robot_action_dim_info)
        previous_task = self._tasks.setdefault(env_idx, observation["instruction"])
        if previous_task != observation["instruction"]:
            raise ValueError(f"Episode changed instruction from {previous_task!r} to {observation['instruction']!r}.")
        self._states.setdefault(env_idx, []).append(state)
        self._actions.setdefault(env_idx, []).append(packed_action)
        if reference_action is not None:
            packed_reference = pack_robot_vector(
                reference_action,
                action_type,
                self.robot_action_dim_info,
            )
            self._reference_actions.setdefault(env_idx, []).append(packed_reference)
        elif env_idx in self._reference_actions:
            raise RuntimeError("Reference actions disappeared partway through a rollout episode.")

        writers = self._writers.setdefault(env_idx, {})
        env_dir = self.in_progress_root / f"env_{env_idx:03d}"
        env_dir.mkdir(parents=True, exist_ok=True)
        for camera, image in observation["images"].items():
            writer = writers.get(camera)
            if writer is None:
                height, width, channels = image.shape
                writer = VideoStreamWriter(
                    str(env_dir / f"{camera}.mp4"),
                    height,
                    width,
                    channels,
                    fps=self.fps,
                )
                writers[camera] = writer
            writer.append(image)

    def finalize(
        self,
        env_idx: int,
        episode_index: int,
        *,
        success: bool,
        score: float,
        episode_seed: int,
        layout_id: int,
    ) -> Path:
        states = self._states.pop(env_idx, [])
        actions = self._actions.pop(env_idx, [])
        reference_actions = self._reference_actions.pop(env_idx, [])
        self._pending.pop(env_idx, None)
        action_type = self._action_types.pop(env_idx, "joint")
        task = self._tasks.pop(env_idx, self.task_name)
        if not states or len(states) != len(actions):
            self.abort([env_idx])
            raise RuntimeError(
                f"Cannot finalize rollout env={env_idx}: states={len(states)} actions={len(actions)}."
            )
        if reference_actions and len(reference_actions) != len(actions):
            self.abort([env_idx])
            raise RuntimeError(
                f"Incomplete reference actions for env={env_idx}: "
                f"reference={len(reference_actions)} actions={len(actions)}."
            )
        writers = self._writers.pop(env_idx, {})
        for writer in writers.values():
            writer.close(announce=False)

        source_dir = self.in_progress_root / f"env_{env_idx:03d}"
        trajectory = {
            "observation_state": np.stack(states),
            "action": np.stack(actions),
        }
        if reference_actions:
            trajectory["reference_action"] = np.stack(reference_actions)
        np.savez_compressed(source_dir / "trajectory.npz", **trajectory)
        manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "layout_id": int(layout_id),
            "task": task,
            "success": bool(success),
            "score": float(score),
            "length": len(actions),
            "fps": self.fps,
            "action_type": action_type,
            "state_dim": int(states[0].size),
            "action_dim": int(actions[0].size),
            "has_reference_actions": bool(reference_actions),
            "cameras": sorted(writers),
        }
        (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        final_dir = self.episodes_root / f"{self.run_id}_episode_{episode_index:07d}"
        if final_dir.exists():
            raise FileExistsError(f"Rollout episode already exists: {final_dir}")
        source_dir.replace(final_dir)
        return final_dir

    def abort(self, env_idx_list: list[int] | None = None) -> None:
        if env_idx_list is None:
            env_idx_list = sorted(
                set(self._pending) | set(self._states) | set(self._actions) | set(self._writers)
                | set(self._reference_actions)
            )
        for env_idx in env_idx_list:
            self._pending.pop(env_idx, None)
            self._states.pop(env_idx, None)
            self._actions.pop(env_idx, None)
            self._reference_actions.pop(env_idx, None)
            self._action_types.pop(env_idx, None)
            self._tasks.pop(env_idx, None)
            for writer in self._writers.pop(env_idx, {}).values():
                writer.abort()
            shutil.rmtree(self.in_progress_root / f"env_{env_idx:03d}", ignore_errors=True)

    def close(self) -> None:
        self.abort()
