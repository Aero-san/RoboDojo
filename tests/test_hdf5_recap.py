from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import pyarrow.parquet as parquet

from scripts.posttrain.build_replay_buffer import (
    _rollout_records,
    main as build_replay_buffer,
)
from scripts.posttrain.build_replay_buffer_incremental import _normalize_task_slugs
from scripts.posttrain.build_wcm_training_subset import main as build_wcm_training_subset
from scripts.posttrain.hdf5_io import Hdf5DemoSource
from scripts.posttrain.robodojo_dataset import RoboDojoDataset, filter_episode_metadata


def _jpeg(rgb: np.ndarray) -> np.ndarray:
    # RoboDojo records RGB arrays directly through OpenCV's encoder. Preserve
    # that convention here so decode_image_bit() returns an RGB array.
    ok, encoded = cv2.imencode(".jpg", rgb)
    if not ok:
        raise RuntimeError("Could not encode test JPEG")
    return encoded


def _write_episode(path: Path, *, explicit_actions: bool) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True)
    left_arm = np.arange(18, dtype=np.float32).reshape(3, 6)
    left_gripper = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    right_arm = left_arm + 100
    right_gripper = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
    states = np.concatenate(
        [left_arm, left_gripper[:, None], right_arm, right_gripper[:, None]],
        axis=1,
    )
    actions = states + 10
    frames = [
        _jpeg(np.full((24, 32, 3), [index * 20, 80, 160], dtype=np.uint8))
        for index in range(3)
    ]
    variable_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as handle:
        handle.attrs.update(task="pour_by_language", fps=25)
        handle.create_dataset(
            "instruction",
            data=np.asarray([b"Pour each bottle into its matching bowl."]),
        )
        state = handle.create_group("state")
        for name, value in (
            ("left_arm_joint_states", left_arm),
            ("left_ee_joint_states", left_gripper),
            ("right_arm_joint_states", right_arm),
            ("right_ee_joint_states", right_gripper),
        ):
            state.create_dataset(name, data=value)
        if explicit_actions:
            action = handle.create_group("action")
            action.create_dataset("left_arm_joint_states", data=actions[:, :6])
            action.create_dataset("left_ee_joint_states", data=actions[:, 6])
            action.create_dataset("right_arm_joint_states", data=actions[:, 7:13])
            action.create_dataset("right_ee_joint_states", data=actions[:, 13])
        vision = handle.create_group("vision")
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            colors = vision.create_group(camera).create_dataset(
                "colors", shape=(len(frames),), dtype=variable_uint8
            )
            for index, frame in enumerate(frames):
                colors[index] = frame
    return states, actions


class Hdf5DemoSourceTest(unittest.TestCase):
    def test_rollout_and_incremental_metadata_use_canonical_task_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episodes/000000"
            episode.mkdir(parents=True)
            (episode / "manifest.json").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "task": "Pour each bottle into its matching bowl.",
                        "success": True,
                        "fps": 25,
                    }
                ),
                encoding="utf-8",
            )
            np.savez(
                episode / "trajectory.npz",
                observation_state=np.zeros((2, 14), dtype=np.float32),
                action=np.zeros((2, 14), dtype=np.float32),
            )
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                (episode / f"{camera}.mp4").touch()

            records = _rollout_records(root, "pour_by_language", 0, 0)
            self.assertEqual(records[0]["task_slug"], "pour_by_language")

            metadata = [
                {"episode_index": 0, "task_slug": "old instruction"},
                {"episode_index": 1},
            ]
            _normalize_task_slugs(metadata, "pour_by_language")
            self.assertEqual(
                [row["task_slug"] for row in metadata],
                ["pour_by_language", "pour_by_language"],
            )

    def test_public_rgb_episode_derives_actions_and_builds_internal_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = (
                root
                / "RoboDojo/pour_by_language/arx_x5/data/episode_0000000.hdf5"
            )
            states, _ = _write_episode(episode, explicit_actions=False)
            output = root / "buffer"

            build_replay_buffer(
                argparse.Namespace(
                    demo_root=str(root / "RoboDojo"),
                    demo_format="hdf5",
                    rollout_root=[],
                    output=str(output),
                    task="pour_by_language",
                    max_demo_episodes=0,
                    max_rollout_episodes=0,
                    chunk_size=1000,
                    seed=0,
                )
            )

            info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
            episodes = [
                json.loads(line)
                for line in (output / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            table = parquet.read_table(
                output / "data/chunk-000/episode_000000.parquet"
            )
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)

            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["source_format"], "hdf5")
            np.testing.assert_allclose(actions[:-1], states[1:])
            np.testing.assert_allclose(actions[-1], states[-1])
            self.assertEqual(episodes[0]["task_slug"], "pour_by_language")
            self.assertEqual(
                episodes[0]["tasks"],
                ["Pour each bottle into its matching bowl."],
            )
            selected, _ = filter_episode_metadata(episodes, "pour_by_language")
            self.assertEqual(len(selected), 1)
            wcm_dataset = RoboDojoDataset(
                output,
                task_selector="pour_by_language",
            )
            self.assertEqual(len(wcm_dataset), 3)

            subset = root / "wcm_subset"
            build_wcm_training_subset(
                argparse.Namespace(
                    buffer=str(output),
                    output=str(subset),
                    old_episode_count=0,
                    replay_episodes=0,
                    chunk_size=1000,
                    seed=0,
                )
            )
            subset_episodes = [
                json.loads(line)
                for line in (subset / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(subset_episodes[0]["task_slug"], "pour_by_language")
            subset_dataset = RoboDojoDataset(
                subset,
                task_selector="pour_by_language",
            )
            self.assertEqual(len(subset_dataset), 3)

            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                video = (
                    output
                    / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
                )
                self.assertGreater(video.stat().st_size, 0)
                if camera == "cam_high":
                    capture = cv2.VideoCapture(str(video))
                    ok, bgr = capture.read()
                    capture.release()
                    self.assertTrue(ok)
                    rgb_mean = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).mean(axis=(0, 1))
                    self.assertGreater(rgb_mean[2], rgb_mean[1])
                    self.assertGreater(rgb_mean[1], rgb_mean[0])

    def test_explicit_hdf5_action_group_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "pour_by_language/arx_x5/data/episode_0000007.hdf5"
            _, expected_actions = _write_episode(episode, explicit_actions=True)

            _, records = Hdf5DemoSource(root, "pour_by_language").load(0)

            self.assertEqual(records[0]["source_episode"], 7)
            np.testing.assert_allclose(records[0]["actions"], expected_actions)


if __name__ == "__main__":
    unittest.main()
