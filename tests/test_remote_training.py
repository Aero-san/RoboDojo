from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts.posttrain import remote_recap, remote_training


class RemoteTrainingHelpersTest(unittest.TestCase):
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
