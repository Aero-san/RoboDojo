from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.posttrain.recap_advantage_metadata import (
    backfill_advantage_statistics,
    positive_statistics,
)


class RecapPositiveStatisticsTest(unittest.TestCase):
    def test_counts_positive_frames_by_outcome_and_tracks_failure_positions(self):
        records = [
            {
                "source_kind": "demo",
                "success": True,
                "positive": [True, True],
            },
            {
                "source_kind": "rollout",
                "success": True,
                "positive": [True, False, True],
            },
            {
                "source_kind": "rollout",
                "success": False,
                "positive": [False, True, False, True],
            },
            {
                "source_kind": "unknown",
                "success": False,
                "positive": [True],
            },
        ]

        statistics = positive_statistics(records)

        self.assertEqual(
            statistics["positive_frame_counts"],
            {
                "total": 7,
                "demonstration": 2,
                "successful_rollout": 2,
                "failed_rollout": 2,
                "unknown_source": 1,
            },
        )
        histogram = statistics[
            "failed_rollout_positive_normalized_position_histogram"
        ]
        self.assertEqual(histogram["counts"], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        self.assertEqual(len(histogram["bin_edges"]), 11)

    def test_backfills_existing_advantage_records_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            advantages = root / "advantages.jsonl"
            labels = root / "success_labels.json"
            advantages.write_text(
                json.dumps({"type": "recap_advantages"})
                + "\n"
                + json.dumps(
                    {
                        "episode_index": 0,
                        "source_kind": "rollout",
                        "positive": [True, False],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            labels.write_text(json.dumps({"0": False}), encoding="utf-8")

            self.assertEqual(
                backfill_advantage_statistics(advantages, labels),
                2,
            )
            self.assertEqual(
                backfill_advantage_statistics(advantages, labels),
                0,
            )
            rows = [
                json.loads(line)
                for line in advantages.read_text(encoding="utf-8").splitlines()
            ]
            self.assertFalse(rows[1]["success"])
            self.assertEqual(
                rows[0]["positive_statistics"]["positive_frame_counts"][
                    "failed_rollout"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
