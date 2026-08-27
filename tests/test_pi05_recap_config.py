from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.posttrain.pi05_recap_config import resolve

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/posttrain/pi05_recap.yaml.example"


class Pi05RecapConfigTest(unittest.TestCase):
    def _resolve_payload(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            return resolve(path)

    def test_example_resolves_explicit_pi05_runtime(self):
        resolved, environment = resolve(EXAMPLE)
        self.assertEqual(resolved["pi05"]["parameter_dtype"], "bfloat16")
        self.assertEqual(environment["OPENPI_SHARDING_STRATEGY"], "full_shard")
        self.assertEqual(environment["TRAIN_GPUS"], "0,1,2,3")

    def test_unknown_field_is_rejected(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["pi05"]["paramter_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "Unknown.*paramter_dtype"):
            self._resolve_payload(payload)

    def test_obsolete_promotion_gate_is_rejected(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["evaluation"]["promotion"] = {"stop_on_rejection": True}
        with self.assertRaisesRegex(ValueError, "Unknown.*evaluation.promotion"):
            self._resolve_payload(payload)

    def test_fsdp_axis_must_divide_gpu_count(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload = deepcopy(payload)
        payload["devices"]["pi05_train"] = [0, 1, 2]
        payload["pi05"]["fsdp_devices"] = 2
        with self.assertRaisesRegex(ValueError, "fsdp_devices"):
            self._resolve_payload(payload)

    def test_zero_pi_workers_is_supported(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["pi05"]["num_workers"] = 0
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["OPENPI_NUM_WORKERS"], "0")

    def test_rollout_evaluation_reuse_resolves(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["evaluation"]["reuse_rollout"] = True
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["RECAP_POLICY_EVAL_REUSE_ROLLOUT"], "1")

    def test_completed_artifact_reuse_requires_resume(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["run"]["reuse_completed_artifacts"] = True
        with self.assertRaisesRegex(ValueError, "requires run.resume"):
            self._resolve_payload(payload)

        payload["run"]["resume"] = True
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["RECAP_REUSE_COMPLETED_ARTIFACTS"], "1")

    def test_remote_training_hosts_and_gpu_lists_resolve(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["training"]["remote"].update(
            {
                "enabled": True,
                "host": "train4090",
                "repo_root": "/share/user/RoboDojo",
                "work_root": "/share/user/recap_remote_jobs",
                "pi05_gpus": [4, 5],
                "wcm_gpus": [6, 7],
            }
        )
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_HOST"], "train4090")
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_PI05_GPUS"], "4,5")
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_WCM_GPUS"], "6,7")

    def test_remote_gpu_count_replaces_local_count_for_pi_validation(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["devices"]["pi05_train"] = [0, 1, 2]
        payload["training"]["remote"].update(
            {
                "enabled": True,
                "host": "train4090",
                "repo_root": "/share/user/RoboDojo",
                "work_root": "/share/user/recap_remote_jobs",
                "pi05_gpus": [4, 5],
            }
        )
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["OPENPI_FSDP_DEVICES"], "2")

    def test_remote_training_requires_host_paths(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["training"]["remote"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "Remote training requires"):
            self._resolve_payload(payload)

    def test_zero_learning_rate_is_rejected(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["pi05"]["optimizer"]["learning_rate"] = 0
        with self.assertRaisesRegex(ValueError, "learning_rate must be positive"):
            self._resolve_payload(payload)


if __name__ == "__main__":
    unittest.main()
