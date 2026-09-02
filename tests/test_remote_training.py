from __future__ import annotations

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from scripts.posttrain import g05_remote, remote_recap, remote_training
from scripts.posttrain.prepare_g05_inference_checkpoint import (
    prepare_g05_inference_checkpoint,
)
from scripts.posttrain.train_g05 import _memory_overrides


class RemoteTrainingHelpersTest(unittest.TestCase):
    def test_g05_native_memory_overrides_are_explicit(self):
        args = SimpleNamespace(
            model_weights_to_bf16=True,
            use_8bit_optimizer=True,
            checkpoint_vision=False,
            checkpoint_vlm=True,
            checkpoint_action_expert=True,
        )

        self.assertEqual(
            _memory_overrides(args),
            [
                "model.model_weights_to_bf16=true",
                "model.use_8bit_optimizer=true",
                "model.model_arch.checkpoint_vision=false",
                "model.model_arch.checkpoint_vlm=true",
                "model.model_arch.checkpoint_action_expert=true",
            ],
        )

    def test_g05_remote_forwards_native_memory_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            checkpoint = bundle / "checkpoints/step_100.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            args = SimpleNamespace(
                dataset=str(root / "dataset"),
                init_policy=str(bundle),
                output=str(root / "output"),
                remote_repo_root="/remote/RoboDojo",
                remote_policy_python="/remote/g05/python",
                g05_root="/remote/GalaxeaVLA",
                processor_path="/remote/GalaxeaVLA/processor",
                task_config="robodojo_recap",
                experiment_name="recap-test",
                gpus="4,5",
                steps=100,
                save_interval=50,
                batch_size=2,
                num_workers=0,
                grad_accumulation_steps=1,
                learning_rate=1e-5,
                warmup_steps=10,
                decay_learning_rate=1e-6,
                decay_start_ratio=0.5,
                weight_decay=1e-4,
                model_weights_to_bf16=True,
                use_8bit_optimizer=True,
                checkpoint_vision=False,
                checkpoint_vlm=True,
                checkpoint_action_expert=True,
                wandb=False,
                resume=False,
                job_id="g05-memory-test",
                seed=17,
                recap_demo_weight=3.0,
                recap_rollout_weight=1.0,
            )
            with (
                mock.patch.object(g05_remote, "_install_remote_trainer"),
                mock.patch.object(
                    g05_remote.remote_recap,
                    "_g05_bundle",
                    return_value=(bundle, checkpoint),
                ),
            ):
                run_stage = mock.Mock()
                g05_remote._run(args, run_stage)

            command = run_stage.call_args.kwargs["command"]
            self.assertIn("--model-weights-to-bf16", command)
            self.assertIn("--use-8bit-optimizer", command)
            self.assertIn("--no-checkpoint-vision", command)
            self.assertIn("--checkpoint-vlm", command)
            self.assertIn("--checkpoint-action-expert", command)
            self.assertIn("--seed 17", command)
            self.assertIn("--recap-demo-weight 3.0", command)
            self.assertIn("--recap-rollout-weight 1.0", command)

    def test_wcm_remote_support_installer_targets_checkout_atomically(self):
        args = SimpleNamespace(
            host="XYZ6226",
            remote_repo_root="/remote/RoboDojo",
        )
        with (
            mock.patch.object(remote_recap, "_remote") as remote,
            mock.patch.object(remote_recap, "_scp") as scp,
        ):
            remote_training._install_remote_wcm_support(args)

        uploads = [call.args[1:] for call in scp.call_args_list]
        self.assertEqual(
            [Path(source).name for source, _ in uploads],
            [
                "run_wcm.sh",
                "run_wcm.py",
                "wcm_checkpoint.py",
                "annotate_recap_advantages.py",
                "recap_advantage_metadata.py",
                "render_rollout_value_videos.py",
            ],
        )
        self.assertTrue(
            all(
                destination.startswith("XYZ6226:/remote/RoboDojo/scripts/posttrain/")
                and ".tmp-" in destination
                for _, destination in uploads
            )
        )
        self.assertEqual(
            [
                command[-1]
                for command in (call.args[1] for call in remote.call_args_list)
                if command[0] == "mv"
            ],
            [
                "/remote/RoboDojo/scripts/posttrain/run_wcm.sh",
                "/remote/RoboDojo/scripts/posttrain/run_wcm.py",
                "/remote/RoboDojo/scripts/posttrain/wcm_checkpoint.py",
                "/remote/RoboDojo/scripts/posttrain/annotate_recap_advantages.py",
                "/remote/RoboDojo/scripts/posttrain/recap_advantage_metadata.py",
                "/remote/RoboDojo/scripts/posttrain/render_rollout_value_videos.py",
            ],
        )

    def test_advantage_inference_uses_nonpersistent_spawn_workers(self):
        module = ast.parse(
            Path("scripts/posttrain/annotate_recap_advantages.py").read_text(
                encoding="utf-8"
            )
        )
        loaders = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DataLoader"
        ]

        self.assertEqual(len(loaders), 1)
        keywords = {keyword.arg: keyword.value for keyword in loaders[0].keywords}
        self.assertIs(ast.literal_eval(keywords["persistent_workers"]), False)
        context = keywords["multiprocessing_context"]
        self.assertIsInstance(context, ast.IfExp)
        self.assertEqual(ast.literal_eval(context.body), "spawn")

    def test_advantage_inference_is_offline_and_synchronizes_finalization(self):
        source = Path("scripts/posttrain/annotate_recap_advantages.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)
        self.assertIn("completion = broadcast_object(completion, ctx, src=0)", source)
        self.assertIn("labels finalized; exiting", source)

    def test_remote_multigpu_advantages_use_torchrun(self):
        args = SimpleNamespace(
            buffer="/local/buffer",
            wcm_checkpoint="/local/deploy.pt",
            output="/local/advantages.jsonl",
            task="general_pickup",
            gpus="0,1,2,3",
            lookahead="10",
            gamma="1.0",
            failure_penalty="300",
            positive_fraction="0.3",
            batch_size="96",
            num_workers="8",
            device="cuda",
            remote_wcm_python="/remote/wcm/python",
            remote_repo_root="/remote/RoboDojo",
            job_id="general-pickup-iter-03-advantages",
        )
        with (
            mock.patch.object(remote_training, "_install_remote_wcm_support"),
            mock.patch.object(remote_training, "_run_stage") as run_stage,
        ):
            remote_training.run_advantages(args)

        command = run_stage.call_args.kwargs["command"]
        self.assertIn(
            "/remote/wcm/python -m torch.distributed.run --standalone "
            "--nproc_per_node=4",
            command,
        )
        self.assertIn("--expected-world-size 4", command)

    def test_g05_remote_trainer_installer_targets_checkout_atomically(self):
        args = SimpleNamespace(
            host="XYZ6226",
            remote_repo_root="/remote/RoboDojo",
        )
        with (
            mock.patch.object(remote_recap, "_remote") as remote,
            mock.patch.object(remote_recap, "_scp") as scp,
        ):
            g05_remote._install_remote_trainer(args)

        uploads = [call.args[1:] for call in scp.call_args_list]
        self.assertEqual(
            [Path(source).name for source, _ in uploads],
            [
                "train_g05.py",
                "g05_finetune_entry.py",
                "g05_source_sampling.py",
                "robodojo_recap.yaml",
                "robodojo_recap.yaml",
            ],
        )
        self.assertIn(
            "XYZ6226:/remote/RoboDojo/scripts/posttrain/train_g05.py.tmp-",
            uploads[0][1],
        )
        self.assertIn(
            "XYZ6226:/remote/RoboDojo/scripts/posttrain/g05_finetune_entry.py.tmp-",
            uploads[1][1],
        )
        self.assertIn(
            "XYZ6226:/remote/RoboDojo/scripts/posttrain/g05_source_sampling.py.tmp-",
            uploads[2][1],
        )
        self.assertIn(
            "XYZ6226:/remote/RoboDojo/configs/g05/data/robodojo_recap.yaml.tmp-",
            uploads[3][1],
        )
        self.assertIn(
            "XYZ6226:/remote/RoboDojo/configs/g05/task/robodojo_recap.yaml.tmp-",
            uploads[4][1],
        )
        commands = [call.args[1] for call in remote.call_args_list]
        self.assertEqual(commands[0][:2], ["chown", "--reference"])
        self.assertEqual(commands[1][:2], ["chmod", "--reference"])
        self.assertEqual(
            [command[-1] for command in commands if command[0] == "mv"],
            [
                "/remote/RoboDojo/scripts/posttrain/train_g05.py",
                "/remote/RoboDojo/scripts/posttrain/g05_finetune_entry.py",
                "/remote/RoboDojo/scripts/posttrain/g05_source_sampling.py",
                "/remote/RoboDojo/configs/g05/data/robodojo_recap.yaml",
                "/remote/RoboDojo/configs/g05/task/robodojo_recap.yaml",
            ],
        )

    def test_remote_preflight_checks_g05_policy_websocket_runtime(self):
        args = SimpleNamespace(
            host="XYZ4090",
            remote_repo_root="/remote/RoboDojo",
            remote_work_root="/remote/jobs",
            remote_zstd_bin="/usr/bin/zstd",
            remote_conda_bin="/remote/conda",
            remote_python_bin="/remote/bootstrap/bin/python",
            policy="g05",
            policy_env="/remote/GalaxeaVLA/.venv",
            eval_env="/remote/miniconda3/envs/RoboDojo",
            gpu=[],
            require_wcm=False,
            gpu_reservation=False,
        )
        success = SimpleNamespace(returncode=0, stdout="--zstd\n", stderr="")
        with (
            mock.patch.object(remote_recap, "_validate"),
            mock.patch.object(
                remote_recap, "_remote_result", return_value=success
            ) as remote_result,
            mock.patch.object(remote_recap, "_install_worker"),
        ):
            remote_recap.preflight(args)

        commands = [call.args[1] for call in remote_result.call_args_list]
        codec_checks = [
            command
            for command in commands
            if "XPolicyLab.client_server.ws.protocol.codec" in " ".join(command)
        ]
        self.assertEqual(len(codec_checks), 1)
        self.assertIn(
            "PYTHONPATH=/remote/RoboDojo:/remote/RoboDojo/XPolicyLab",
            codec_checks[0],
        )
        self.assertIn("/remote/GalaxeaVLA/.venv", codec_checks[0])
        eval_checks = [
            command
            for command in commands
            if "--prefix" in command
            and "/remote/miniconda3/envs/RoboDojo" in command
        ]
        self.assertEqual(len(eval_checks), 1)

    def test_g05_adapter_matches_policy_inferencer_constructor(self):
        adapter_path = Path("XPolicyLab/policy/G05/model.py")
        module = ast.parse(adapter_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PolicyInferencer"
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual([keyword.arg for keyword in calls[0].keywords], ["device"])

    def test_g05_inference_checkpoint_prepares_dataset_free_rollout_config(self):
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
                            "tokenizer": "${tokenizer}",
                            "model_arch": {
                                "hf_processor_path": "/training-host/qwen-processor",
                                "action_tokenizer": "${model.tokenizer._target_}",
                                "AT_CONFIG": "${model.tokenizer.vq_config}",
                            },
                        },
                        "tokenizer": {
                            "_target_": "g05.tokenizer.interface.vq_base.VQActionTokenizer",
                            "vq_config": {
                                "vqvae_type": "g05.tokenizer.models.actioncodec2_v2.wrapper.ActionCodecV2Wrapper",
                                "ckpt_dir": "/training-host/action_tokenizer.pt",
                            },
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

            changes = prepare_g05_inference_checkpoint(root)
            result = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            self.assertEqual(
                changes,
                [
                    "data.dataset_dirs",
                    "data.embodiment_datasets.robodojo.dataset_groups",
                    "logger.dir",
                    "logger.project",
                    "logger.workspace",
                    "portable:model.model_arch.hf_processor_path",
                    "portable:tokenizer.vq_config.ckpt_dir",
                    "materialized:model.model_arch.action_tokenizer",
                    "materialized:model.model_arch.AT_CONFIG",
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
            self.assertEqual(
                result["model"]["model_arch"]["action_tokenizer"],
                "g05.tokenizer.interface.vq_base.VQActionTokenizer",
            )
            self.assertEqual(
                result["tokenizer"]["vq_config"]["ckpt_dir"],
                "${oc.env:G05_ACTION_TOKENIZER_PATH}",
            )
            self.assertEqual(
                result["model"]["model_arch"]["AT_CONFIG"],
                {
                    "vqvae_type": "g05.tokenizer.models.actioncodec2_v2.wrapper.ActionCodecV2Wrapper",
                    "ckpt_dir": "${oc.env:G05_ACTION_TOKENIZER_PATH}",
                },
            )
            self.assertEqual(prepare_g05_inference_checkpoint(root), [])

    def test_g05_inference_checkpoint_repairs_materialized_tokenizer_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / ".hydra/config.yaml"
            config_path.parent.mkdir()
            tokenizer = {
                "_target_": "g05.tokenizer.interface.vq_base.VQActionTokenizer",
                "vq_config": {
                    "vqvae_type": "g05.tokenizer.models.actioncodec2_v2.wrapper.ActionCodecV2Wrapper",
                    "ckpt_dir": "/training-host/action_tokenizer.pt",
                },
            }
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"action_size": 16},
                        "model": {
                            "tokenizer": tokenizer,
                            "model_arch": {
                                "hf_processor_path": "${oc.env:G05_HF_PROCESSOR_PATH}",
                                "action_tokenizer": tokenizer["_target_"],
                                "AT_CONFIG": {
                                    **tokenizer["vq_config"],
                                    "ckpt_dir": "/training-host/action_tokenizer.pt",
                                },
                            },
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            changes = prepare_g05_inference_checkpoint(root)
            result = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            self.assertEqual(
                changes,
                [
                    "portable:tokenizer.vq_config.ckpt_dir",
                    "portable:model.model_arch.AT_CONFIG.ckpt_dir",
                ],
            )
            self.assertEqual(
                result["model"]["tokenizer"]["vq_config"]["ckpt_dir"],
                "${oc.env:G05_ACTION_TOKENIZER_PATH}",
            )
            self.assertEqual(
                result["model"]["model_arch"]["AT_CONFIG"]["ckpt_dir"],
                "${oc.env:G05_ACTION_TOKENIZER_PATH}",
            )
            self.assertEqual(prepare_g05_inference_checkpoint(root), [])

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
        self.assertTrue(
            any(
                "render_rollout_value_videos.py.tmp-" in destination
                for destination in destinations
            )
        )

    def test_g05_adapter_installer_targets_remote_checkout_atomically(self):
        args = SimpleNamespace(
            host="XYZ4090",
            remote_repo_root="/remote/RoboDojo",
        )
        with (
            mock.patch.object(remote_recap, "_remote") as remote,
            mock.patch.object(remote_recap, "_scp") as scp,
        ):
            remote_recap._install_g05_adapter(args)

        source, destination = scp.call_args.args[1:]
        self.assertEqual(Path(source).name, "model.py")
        self.assertIn(
            "XYZ4090:/remote/RoboDojo/XPolicyLab/policy/G05/model.py.tmp-",
            destination,
        )
        move = remote.call_args.args[1]
        self.assertEqual(move[0], "mv")
        self.assertEqual(
            move[-1],
            "/remote/RoboDojo/XPolicyLab/policy/G05/model.py",
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
                max_steps=40,
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
                mock.patch.object(
                    remote_recap, "_install_worker", return_value="worker"
                ),
                mock.patch.object(
                    remote_recap, "_install_g05_adapter"
                ) as install_adapter,
                mock.patch.object(remote_recap, "_package_checkpoint"),
                mock.patch.object(remote_recap, "_remote"),
                mock.patch.object(remote_recap, "_scp"),
                mock.patch.object(remote_recap, "_worker_environment", return_value={}),
                mock.patch.object(remote_recap, "_invoke_worker") as invoke,
                mock.patch.object(remote_recap, "_extract"),
                mock.patch.object(remote_recap, "_cancel_remote_job"),
            ):
                remote_recap.rollout(args)

            install_adapter.assert_called_once_with(args)
            environment = invoke.call_args.args[2]
            self.assertEqual(
                environment["RECAP_REMOTE_G05_PROCESSOR_PATH"],
                processor,
            )
            self.assertEqual(environment["RECAP_REMOTE_MAX_STEPS"], "40")

    def test_rollout_max_steps_reaches_local_and_remote_eval_launchers(self):
        run_recap = Path("scripts/posttrain/run_recap.sh").read_text(encoding="utf-8")
        worker = Path("scripts/posttrain/remote_recap_worker.sh").read_text(
            encoding="utf-8"
        )

        self.assertGreaterEqual(
            run_recap.count('--max-steps "${ROLLOUT_MAX_STEPS}"'), 3
        )
        self.assertIn('--max-steps "${RECAP_REMOTE_MAX_STEPS}"', worker)

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
