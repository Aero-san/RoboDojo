from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from scripts.posttrain import remote_recap, remote_training
from scripts.posttrain.prepare_g05_inference_checkpoint import (
    prepare_g05_inference_checkpoint,
)


class RemoteTrainingHelpersTest(unittest.TestCase):
    def test_g05_inference_checkpoint_drops_only_training_data_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / ".hydra/config.yaml"
            config_path.parent.mkdir()
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "action_size": 16,
                            "dataset_dirs": ["${oc.env:ROBODOJO_LEROBOT_V30_ROOT}"],
                            "embodiment_datasets": {
                                "robodojo": {
                                    "embodiment_type": "robodojo",
                                    "dataset_groups": [
                                        {
                                            "weight": 1.0,
                                            "dataset_dirs": [
                                                "${oc.env:ROBODOJO_LEROBOT_V30_ROOT}"
                                            ],
                                        }
                                    ],
                                    "shape_meta": {"action": []},
                                }
                            },
                            "processors": {"robodojo": {"shape_meta": {"action": []}}},
                        },
                        "model": {
                            "model_arch": {
                                "hf_processor_path": "${oc.env:G05_HF_PROCESSOR_PATH}"
                            }
                        },
                        "logger": {
                            "type": "wandb",
                            "mode": "disabled",
                            "dir": "${oc.env:G05_OUTPUT_DIR}",
                            "project": "${oc.env:WANDB_PROJECT,g05}",
                            "workspace": "${oc.env:WANDB_ENTITY,Galaxea-AI}",
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            removed = prepare_g05_inference_checkpoint(root)
            result = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            self.assertEqual(
                removed,
                [
                    "data.dataset_dirs",
                    "data.embodiment_datasets.robodojo.dataset_groups",
                    "logger.dir",
                    "logger.project",
                    "logger.workspace",
                ],
            )
            self.assertNotIn("dataset_dirs", result["data"])
            self.assertNotIn(
                "dataset_groups",
                result["data"]["embodiment_datasets"]["robodojo"],
            )
            self.assertEqual(result["data"]["action_size"], 16)
            self.assertEqual(
                result["model"]["model_arch"]["hf_processor_path"],
                "${oc.env:G05_HF_PROCESSOR_PATH}",
            )
            self.assertEqual(result["logger"], {"type": "wandb", "mode": "disabled"})

    def test_worker_installer_uploads_g05_inference_preparer(self):
        args = SimpleNamespace(host="XYZ4090", remote_work_root="/remote/jobs")
        with (
            mock.patch.object(remote_recap, "_remote"),
            mock.patch.object(remote_recap, "_scp") as scp,
        ):
            remote_recap._install_worker(args)

        destinations = [call.args[2] for call in scp.call_args_list]
        self.assertTrue(
            any(
                "prepare_g05_inference_checkpoint.py.tmp-" in destination
                for destination in destinations
            )
        )


    def test_local_host_aliases_are_not_sent_to_ssh(self):
        self.assertTrue(remote_recap._is_local_host("local"))
        self.assertTrue(remote_recap._is_local_host("localhost"))

    def test_stage_markers_use_the_remote_checkout(self):
        expanded = remote_training._expand(
            "@repo/scripts/posttrain/train.py @input/lerobot/foo --out @output",
            "/remote/work/jobs/example",
            "/remote/RoboDojo",
            {"lerobot/foo": "/remote/work/jobs/example/inputs/lerobot/foo"},
            "/remote/work/jobs/example/output",
        )
        self.assertEqual(
            expanded,
            "/remote/RoboDojo/scripts/posttrain/train.py "
            "/remote/work/jobs/example/inputs/lerobot/foo --out "
            "/remote/work/jobs/example/output",
        )

    def test_pi05_paths_are_rewritten_to_remote_stage_paths(self):
        args = SimpleNamespace(
            train_arg=[
                "--openpi-root",
                "/local/Pi_05/openpi",
                "--norm-stats-dir",
                "/local/norm",
                "--checkpoint-dir",
                "/local/output",
                "--init-checkpoint",
                "/local/init",
                "--repo-id",
                "RoboDojo-recap-task-iter-1",
            ]
        )
        self.assertEqual(
            remote_training._rewrite_pi05_args(args),
            [
                "--openpi-root",
                "@repo/XPolicyLab/policy/Pi_05/openpi",
                "--norm-stats-dir",
                "@input/norm_stats",
                "--checkpoint-dir",
                "@output",
                "--init-checkpoint",
                "@input/init_policy",
                "--repo-id",
                "RoboDojo-recap-task-iter-1",
            ],
        )

    def test_directory_upload_uses_exact_destination_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "arbitrary-local-name"
            source.mkdir()
            destination = root / "remote" / "inputs" / "dataset"
            backend = SimpleNamespace(host="local")

            with (
                mock.patch.object(remote_recap, "_run") as run,
                mock.patch.object(remote_recap, "_remote") as remote,
                mock.patch.object(remote_recap, "_scp"),
            ):
                remote_training._upload_directory(backend, source, str(destination))

            self.assertEqual(run.call_args.args[0][-2:], [str(source), "."])
            self.assertIn(
                mock.call(backend, ["mkdir", "-p", str(destination)]),
                remote.call_args_list,
            )
            extract = remote.call_args_list[-2].args[1]
            self.assertEqual(extract[-2:], ["-C", str(destination)])

    def test_g05_rollout_forwards_remote_processor_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processor = "/remote/GalaxeaVLA/checkpoints/processor"
            args = SimpleNamespace(
                action="rollout",
                host="XYZ4090",
                remote_repo_root="/remote/RoboDojo",
                remote_work_root="/remote/jobs",
                job_id="task-iter-01",
                checkpoint=str(root / "checkpoint"),
                output=str(root / "rollouts"),
                policy="g05",
                g05_root="/remote/GalaxeaVLA",
                g05_processor_path=processor,
                g05_action_source="fm",
                task="general_pickup",
                episodes=3,
                layout_seed=0,
                layout_offset=0,
                policy_gpu=0,
                env_gpu=1,
                env_cfg="arx_x5",
                action_type="joint",
                policy_env="/remote/GalaxeaVLA/.venv",
                eval_env="/remote/RoboDojo",
            )

            with (
                mock.patch.object(remote_recap, "_validate"),
                mock.patch.object(
                    remote_recap, "_remote_success", side_effect=[False, False]
                ),
                mock.patch.object(remote_recap, "_reserve_remote_gpus"),
                mock.patch.object(remote_recap, "_install_worker", return_value="worker"),
                mock.patch.object(remote_recap, "_package_checkpoint"),
                mock.patch.object(remote_recap, "_remote"),
                mock.patch.object(remote_recap, "_scp"),
                mock.patch.object(remote_recap, "_worker_environment", return_value={}),
                mock.patch.object(remote_recap, "_invoke_worker") as invoke,
                mock.patch.object(remote_recap, "_extract"),
                mock.patch.object(remote_recap, "_cancel_remote_job"),
            ):
                remote_recap.rollout(args)

            environment = invoke.call_args.args[2]
            self.assertEqual(
                environment["RECAP_REMOTE_G05_PROCESSOR_PATH"],
                processor,
            )

    def test_value_video_rebuilds_missing_remote_rollout_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollouts"
            rollout.mkdir()
            args = SimpleNamespace(
                action="value-video",
                host="XYZ4090",
                remote_repo_root="/remote/RoboDojo",
                remote_work_root="/remote/jobs",
                remote_zstd_bin="/usr/bin/zstd",
                remote_conda_bin="/remote/conda",
                remote_python_bin="/remote/python",
                gpu_reservation=True,
                gpu_reservation_leave_free_mib=2048,
                gpu_reservation_idle_used_max_mib=1024,
                gpu_reservation_remote_max_hold_seconds=1800,
                job_id="task-iter-01",
                wcm_checkpoint=str(root / "deploy.pt"),
                rollout_root=str(rollout),
                output=str(root / "value_videos"),
                episodes=3,
                gpu=4,
                batch_size=16,
                device="cuda",
                precision="bf16",
                backend="auto",
                speed=1.0,
                y_min=-1.0,
                y_max=1.0,
                title="RECAP",
            )

            with (
                mock.patch.object(remote_recap, "_validate"),
                mock.patch.object(
                    remote_recap, "_remote_success", side_effect=[False, False]
                ),
                mock.patch.object(remote_recap, "_reserve_remote_gpus"),
                mock.patch.object(remote_recap, "_install_worker", return_value="worker"),
                mock.patch.object(remote_recap, "_package_rollout_cache") as package,
                mock.patch.object(remote_recap, "_remote"),
                mock.patch.object(remote_recap, "_scp") as scp,
                mock.patch.object(remote_recap, "_run"),
                mock.patch.object(remote_recap, "_invoke_worker"),
                mock.patch.object(remote_recap, "_extract"),
                mock.patch.object(remote_recap, "_cancel_remote_job"),
            ):
                remote_recap.value_video(args)

            package.assert_called_once_with(
                rollout.resolve(),
                root / ".remote_transfers/task-iter-01-rollouts.tar.zst",
                3,
            )
            self.assertTrue(
                any(
                    call.args[2].endswith("/inbox/rollouts.tar.zst.tmp")
                    for call in scp.call_args_list
                )
            )


if __name__ == "__main__":
    unittest.main()
