from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.posttrain.recap_artifacts import check
from scripts.posttrain.value_video_metadata import (
    backfill_value_video_instructions,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ValueVideoInstructionMetadataTest(unittest.TestCase):
    def test_remote_renderer_resolves_support_from_remote_repo(self):
        source = Path(
            "scripts/posttrain/render_rollout_value_videos.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"RECAP_REMOTE_REPO_ROOT"', source)
        self.assertIn('POSTTRAIN_DIR = ROOT_DIR / "scripts" / "posttrain"', source)

    def test_backfills_historical_value_video_json_from_rollout_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollouts"
            output = root / "value_videos"
            _write_json(
                rollout / "episodes/episode_7/manifest.json",
                {
                    "episode_index": 7,
                    "task": "Pick up the red mug by 10 cm.",
                },
            )
            _write_json(
                output / "episode_curves.json",
                [
                    {
                        "episode_id": 0,
                        "source_episode_id": 7,
                        "frame_indices": [2],
                        "values": [-0.5],
                    }
                ],
            )
            _write_json(
                output / "summary.json",
                {"episodes": [{"episode_id": 0, "success": False}]},
            )
            video = output / "videos/episode-000000.mp4"
            video.parent.mkdir(parents=True)
            video.touch()

            self.assertFalse(check("value_videos", output, 1))
            self.assertEqual(
                backfill_value_video_instructions(output, rollout),
                2,
            )
            self.assertEqual(
                backfill_value_video_instructions(output, rollout),
                0,
            )

            curves = json.loads(
                (output / "episode_curves.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            instruction = "Pick up the red mug by 10 cm."
            self.assertEqual(curves[0]["instruction"], instruction)
            self.assertEqual(summary["episodes"][0]["instruction"], instruction)
            self.assertTrue(check("value_videos", output, 1))


if __name__ == "__main__":
    unittest.main()
