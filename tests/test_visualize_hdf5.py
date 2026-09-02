from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np

from scripts.RoboDojo.visualize_hdf5 import main, sample_episodes


def _write_episode(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(4):
        rgb = np.full((24, 32, 3), [value, index * 30, 180], dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", rgb)
        if not ok:
            raise RuntimeError("Could not encode test frame")
        frames.append(encoded)
    variable_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(path, "w") as handle:
        handle.attrs.update(task="align_blocks", fps=20)
        handle.create_dataset("instruction", data=np.asarray([b"Align the blocks."]))
        vision = handle.create_group("vision")
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            dataset = vision.create_group(camera).create_dataset(
                "colors", shape=(len(frames),), dtype=variable_uint8
            )
            for index, frame in enumerate(frames):
                dataset[index] = frame


class VisualizeHdf5Test(unittest.TestCase):
    def test_even_sampling_covers_first_and_last_episode(self):
        paths = [Path(f"episode_{index:07d}.hdf5") for index in range(10)]
        self.assertEqual(sample_episodes(paths, 3, "even", 0), [paths[0], paths[4], paths[9]])

    def test_main_renders_only_selected_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/RoboDojo"
            for index in range(3):
                _write_episode(
                    source / f"align_blocks/arx_x5/data/episode_{index:07d}.hdf5",
                    20 + index * 20,
                )
            output = root / "previews"

            exit_code = main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--num-episodes",
                    "2",
                    "--sampling",
                    "even",
                    "--panel-width",
                    "96",
                ]
            )

            self.assertEqual(exit_code, 0)
            videos = sorted(output.rglob("*.mp4"))
            self.assertEqual(len(videos), 2)
            self.assertEqual(
                [video.stem for video in videos],
                ["episode_0000000", "episode_0000002"],
            )
            capture = cv2.VideoCapture(str(videos[0]))
            ok, frame = capture.read()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame_count, 4)
            self.assertGreater(frame.size, 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["available_episodes"], 3)
            self.assertEqual(manifest["selected_episodes"], 2)
            self.assertEqual([row["status"] for row in manifest["results"]], ["written", "written"])


if __name__ == "__main__":
    unittest.main()
