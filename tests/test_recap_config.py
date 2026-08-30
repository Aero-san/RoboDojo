from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.posttrain.recap_config import resolve

ROOT = Path(__file__).resolve().parents[1]
G05_EXAMPLE = ROOT / "configs/posttrain/g05_recap.yaml.example"
EXAMPLE = ROOT / "configs/posttrain/pi05_recap.yaml.example"


class RecapConfigTest(unittest.TestCase):
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
        payload["devices"]["policy_train"] = [0, 1, 2]
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
                "policy_gpus": [4, 5],
                "wcm_gpus": [6, 7],
            }
        )
        _, environment = self._resolve_payload(payload)
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_HOST"], "train4090")
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_POLICY_GPUS"], "4,5")
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_WCM_GPUS"], "6,7")

    def test_remote_gpu_count_replaces_local_count_for_pi_validation(self):
        payload = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
        payload["devices"]["policy_train"] = [0, 1, 2]
        payload["training"]["remote"].update(
            {
                "enabled": True,
                "host": "train4090",
                "repo_root": "/share/user/RoboDojo",
                "work_root": "/share/user/recap_remote_jobs",
                "policy_gpus": [4, 5],
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

    def test_g05_example_resolves_v3_runtime(self):
        resolved, environment = resolve(G05_EXAMPLE)
        self.assertEqual(resolved["policy"]["name"], "g05")
        self.assertEqual(resolved["data"]["format"], "v3.0")
        self.assertEqual(environment["ROBODOJO_G05_ACTION_SOURCE"], "fm")
        self.assertEqual(environment["RECAP_TRAINING_REMOTE_POLICY_GPUS"], "0,1,2,3")
        self.assertEqual(environment["G05_DECAY_LEARNING_RATE"], "1e-06")
        self.assertEqual(environment["G05_DECAY_START_RATIO"], "0.5")
        self.assertEqual(
            environment["RECAP_REMOTE_G05_PROCESSOR_PATH"],
            "/absolute/path/to/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor",
        )
        self.assertEqual(
            environment["RECAP_REMOTE_POLICY_ENV"],
            "/absolute/path/to/g05-runtime",
        )
        self.assertEqual(
            environment["RECAP_REMOTE_EVAL_ENV"],
            "/absolute/path/to/robodojo-conda-env",
        )

    def test_remote_rollout_requires_host_specific_runtime_environments(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["rollout"]["remote"]["eval_env"] = None
        with self.assertRaisesRegex(ValueError, "rollout.remote.eval_env"):
            self._resolve_payload(payload)

    def test_remote_g05_requires_processor_path(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["rollout"]["remote"]["g05_processor_path"] = None
        with self.assertRaisesRegex(ValueError, "g05_processor_path"):
            self._resolve_payload(payload)

    def test_g05_rejects_v21_input(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["data"]["format"] = "v2.1"
        with self.assertRaisesRegex(ValueError, "requires data.format: v3.0"):
            self._resolve_payload(payload)

    def test_g05_rejects_ar_action_source(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["g05"]["action_source"] = "ar"
        with self.assertRaisesRegex(ValueError, "requires g05.action_source: fm"):
            self._resolve_payload(payload)

    def test_g05_rejects_decay_rate_above_initial_rate(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["g05"]["optimizer"]["decay_learning_rate"] = 2.0e-5
        with self.assertRaisesRegex(ValueError, "decay_learning_rate"):
            self._resolve_payload(payload)

    def test_g05_rejects_decay_before_warmup_finishes(self):
        payload = yaml.safe_load(G05_EXAMPLE.read_text(encoding="utf-8"))
        payload["g05"]["optimizer"]["decay_start_ratio"] = 0.1
        with self.assertRaisesRegex(ValueError, "before warmup"):
            self._resolve_payload(payload)



if __name__ == "__main__":
    unittest.main()
