from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.posttrain.g05_source_sampling import (
    SourceBalancedDataset,
    build_virtual_indices,
    report_from_manifest,
)


class G05SourceSamplingTest(unittest.TestCase):
    def test_virtual_stream_uses_configured_source_ratio(self):
        groups = {
            "demo": np.arange(3, dtype=np.int64),
            "rollout": np.arange(3, 13, dtype=np.int64),
        }

        indices, report = build_virtual_indices(groups, 2.0, 1.0, seed=7)

        self.assertEqual(len(indices), 13)
        self.assertEqual(report["virtual_frames"], {"demo": 9, "rollout": 4})
        self.assertEqual(sum(np.isin(indices, groups["demo"])), 9)
        self.assertEqual(sum(np.isin(indices, groups["rollout"])), 4)

    def test_wrapper_preserves_inner_dataset_interfaces(self):
        groups = {
            "demo": np.arange(2, dtype=np.int64),
            "rollout": np.arange(2, 6, dtype=np.int64),
        }

        class Dataset:
            def __len__(self):
                return 6

            def __getitem__(self, index):
                return index

            marker = "inner"

        dataset = SourceBalancedDataset(Dataset(), groups, 1.0, 1.0, seed=0)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.marker, "inner")
        self.assertTrue(all(0 <= dataset[index] < 6 for index in range(len(dataset))))
        
    def test_wrapper_aligns_manifest_groups_to_training_episode_prefix(self):
        groups = {
            "demo": np.arange(4, dtype=np.int64),
            "rollout": np.arange(4, 10, dtype=np.int64),
        }

        class Dataset:
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return index

        dataset = SourceBalancedDataset(
            Dataset(),
            groups,
            1.0,
            1.0,
            seed=0,
            frame_range=(0, 8),
        )

        self.assertEqual(dataset.report["manifest_frames"], 10)
        self.assertEqual(dataset.report["dataset_frame_range"], [0, 8])
        self.assertEqual(dataset.report["excluded_frames"], 2)
        self.assertEqual(dataset.report["source_frames"], {"demo": 4, "rollout": 4})
        self.assertTrue(all(0 <= dataset[index] < 8 for index in range(len(dataset))))

    def test_report_reads_recap_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/recap_incremental.json").write_text(
                json.dumps(
                    {
                        "episodes": [
                            {"length": 2, "source_kind": "demo"},
                            {"length": 3, "source_kind": "rollout"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = report_from_manifest(root, 1.0, 1.0, seed=3)

        self.assertEqual(report["source_frames"], {"demo": 2, "rollout": 3})
        self.assertEqual(report["virtual_frames"], {"demo": 2, "rollout": 3})


if __name__ == "__main__":
    unittest.main()
