from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as parquet
import torch
import yaml

from scripts.posttrain import g05_finetune_entry, train_g05
from scripts.posttrain.build_wcm_training_subset import main as build_wcm_training_subset
from scripts.posttrain.lerobot_io import LeRobotLayout
from scripts.posttrain.recap_conditioning import strip_condition, training_prompt
from scripts.posttrain.robodojo_dataset import RoboDojoDataset
from scripts.posttrain.run_wcm import _install_resume_checkpoint_compatibility
from scripts.posttrain.wcm_checkpoint import adapt_wcm_state_dict

ROOT = Path(__file__).resolve().parents[1]


class LeRobotLayoutTest(unittest.TestCase):
    def test_configured_format_must_match_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "One RECAP run may read only"):
                LeRobotLayout(root, "v2.1")

    def test_wcm_loader_rejects_non_internal_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only the normalized internal"):
                RoboDojoDataset(root)


class WcmTrainingSubsetTest(unittest.TestCase):
    def test_subset_resolves_videos_through_lerobot_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "buffer"
            output = root / "subset"
            (source / "meta").mkdir(parents=True)
            cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
            info = {
                "codebase_version": "v2.1",
                "total_episodes": 1,
                "chunks_size": 1000,
                "features": {
                    f"observation.images.{camera}": {"dtype": "video"}
                    for camera in cameras
                },
            }
            (source / "meta/info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            (source / "meta/episodes.jsonl").write_text(
                json.dumps(
                    {"episode_index": 0, "tasks": ["pick"], "length": 2}
                )
                + "\n",
                encoding="utf-8",
            )
            (source / "meta/provenance.jsonl").write_text(
                json.dumps({"episode_index": 0, "source_kind": "demo"}) + "\n",
                encoding="utf-8",
            )
            (source / "meta/success_labels.json").write_text(
                json.dumps({"0": True}), encoding="utf-8"
            )
            data_path = source / "data/chunk-000/episode_000000.parquet"
            data_path.parent.mkdir(parents=True)
            parquet.write_table(
                pa.table(
                    {
                        "episode_index": [0, 0],
                        "frame_index": [0, 1],
                        "index": [0, 1],
                        "task_index": [0, 0],
                    }
                ),
                data_path,
            )
            for camera in cameras:
                video = (
                    source
                    / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
                )
                video.parent.mkdir(parents=True)
                video.touch()

            build_wcm_training_subset(
                argparse.Namespace(
                    buffer=str(source),
                    output=str(output),
                    old_episode_count=0,
                    replay_episodes=0,
                    chunk_size=1000,
                    seed=0,
                )
            )

            for camera in cameras:
                self.assertTrue(
                    (
                        output

                        / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
                    ).is_file()
                )

class WcmCheckpointCompatibilityTest(unittest.TestCase):
    def test_full_resume_loader_normalizes_before_strict_restore(self):
        source_key = (
            "vision_encoder.backbone.encoder.layer.0."
            "attention.attention.query.weight"
        )
        target_key = "vision_encoder.backbone.layers.0.attention.q_proj.weight"
        tensor = torch.zeros(2, 2)
        loader_globals = {
            "load_checkpoint_payload": lambda _path, _ctx: {
                "model": {source_key: tensor}
            }
        }
        exec(
            "def load_training_checkpoint(path, model, optimizer, scheduler, ctx, "
            "expected_config=None):\n"
            "    payload = load_checkpoint_payload(path, ctx)\n"
            "    model.load_state_dict(payload['model'], strict=True)\n"
            "    return payload\n",
            loader_globals,
        )
        original_payload_loader = loader_globals["load_checkpoint_payload"]
        command = SimpleNamespace(
            load_training_checkpoint=loader_globals["load_training_checkpoint"]
        )

        class Model:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {target_key: tensor}

            def load_state_dict(self, state_dict, *, strict):
                self.loaded = state_dict
                self.strict = strict

        model = Model()
        _install_resume_checkpoint_compatibility(command)
        command.load_training_checkpoint("last.pt", model, None, None, object())

        self.assertEqual(set(model.loaded), {target_key})
        self.assertTrue(model.strict)
        self.assertIs(
            loader_globals["load_checkpoint_payload"],
            original_payload_loader,
        )

    def test_transformers_vit_keys_are_normalized_in_both_directions(self):
        pairs = {
            "vision_encoder.backbone.layers.0.attention.q_proj.weight": (
                "vision_encoder.backbone.encoder.layer.0.attention.attention.query.weight"
            ),
            "vision_encoder.backbone.layers.1.attention.o_proj.bias": (
                "vision_encoder.backbone.encoder.layer.1.attention.output.dense.bias"
            ),
            "vision_encoder.backbone.layers.2.layernorm_before.weight": (
                "vision_encoder.backbone.encoder.layer.2.layernorm_before.weight"
            ),
            "vision_encoder.backbone.layers.3.mlp.fc1.weight": (
                "vision_encoder.backbone.encoder.layer.3.intermediate.dense.weight"
            ),
            "vision_encoder.backbone.layers.4.mlp.fc2.bias": (
                "vision_encoder.backbone.encoder.layer.4.output.dense.bias"
            ),
        }
        modern = {key: torch.zeros(2, 2) for key in pairs}
        legacy = {key: torch.zeros(2, 2) for key in pairs.values()}

        self.assertEqual(
            set(adapt_wcm_state_dict(modern, legacy)),
            set(legacy),
        )
        self.assertEqual(
            set(adapt_wcm_state_dict(legacy, modern)),
            set(modern),
        )

    def test_real_architecture_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "shape_mismatches"):
            adapt_wcm_state_dict(
                {"value_head.weight": torch.zeros(2, 2)},
                {"value_head.weight": torch.zeros(3, 2)},
            )


class G05TrainerTest(unittest.TestCase):
    def test_recap_task_checkpoints_all_large_model_components(self):
        config = yaml.safe_load(
            (ROOT / "configs/g05/task/robodojo_recap.yaml").read_text(
                encoding="utf-8"
            )
        )
        model_arch = config["model"]["model_arch"]

        self.assertTrue(model_arch["checkpoint_vision"])
        self.assertTrue(model_arch["checkpoint_vlm"])
        self.assertTrue(model_arch["checkpoint_action_expert"])

    def test_data_config_only_passes_supported_mixture_constructor_keys(self):
        config = yaml.safe_load(
            (ROOT / "configs/g05/data/robodojo_recap.yaml").read_text(
                encoding="utf-8"
            )
        )
        source = ast.parse(
            (
                ROOT
                / "XPolicyLab/policy/G05/GalaxeaVLA/src/g05/data/mixture_lerobot_dataset.py"
            ).read_text(encoding="utf-8")
        )
        dataset_class = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name == "MixtureLerobotDataset"
        )
        constructor = next(
            node
            for node in dataset_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        supported = {argument.arg for argument in constructor.args.args} - {"self"}
        passed = set(config) - {"_target_", "processors"}

        self.assertEqual(passed - supported, set())
        self.assertEqual(
            config["embodiment_datasets"]["robodojo"]["shape_meta"],
            config["processors"]["robodojo"]["shape_meta"],
        )
        self.assertFalse(config["in_memory"])

    def test_guarded_entrypoint_stops_after_first_sample_failure(self):
        base_dataset = ModuleType("g05.data.base_lerobot_dataset")
        base_dataset.MAX_GETITEM_ATTEMPT = 10_000
        data_package = ModuleType("g05.data")
        data_package.base_lerobot_dataset = base_dataset
        g05_package = ModuleType("g05")
        g05_package.data = data_package
        runtime = {}

        def capture_runtime(path, run_name):
            runtime["path"] = path
            runtime["run_name"] = run_name
            runtime["sys_path_0"] = sys.path[0]
            runtime["argv0"] = sys.argv[0]

        with tempfile.TemporaryDirectory() as directory:
            trainer = Path(directory) / "finetune.py"
            trainer.touch()
            with (
                mock.patch.dict(
                    "sys.modules",
                    {
                        "g05": g05_package,
                        "g05.data": data_package,
                        "g05.data.base_lerobot_dataset": base_dataset,
                    },
                ),
                mock.patch.dict(
                    "os.environ",
                    {
                        "ROBODOJO_G05_FINETUNE_SCRIPT": str(trainer),
                        "ROBODOJO_G05_MAX_GETITEM_ATTEMPTS": "1",
                    },
                ),
                mock.patch.object(
                    g05_finetune_entry.runpy,
                    "run_path",
                    side_effect=capture_runtime,
                ) as run_path,
            ):
                g05_finetune_entry.main()

        self.assertEqual(base_dataset.MAX_GETITEM_ATTEMPT, 1)
        run_path.assert_called_once_with(str(trainer.resolve()), run_name="__main__")
        self.assertEqual(runtime["sys_path_0"], str(trainer.parent.resolve()))
        self.assertEqual(runtime["argv0"], str(trainer.resolve()))

    def test_dry_run_emits_learning_rate_decay_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g05_root = root / "g05"
            (g05_root / "scripts").mkdir(parents=True)
            (g05_root / "scripts/finetune.py").touch()
            processor = g05_root / "checkpoints/processor"
            processor.mkdir(parents=True)
            dataset = root / "dataset"
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta/info.json").write_text(
                json.dumps({"codebase_version": "v3.0"}), encoding="utf-8"
            )
            checkpoint = root / "checkpoint"
            checkpoint.touch()
            stats = root / "dataset_stats.json"
            stats.write_text("{}", encoding="utf-8")
            tokenizer = root / "action_tokenizer.pt"
            tokenizer.touch()
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                train_g05.main(
                    SimpleNamespace(
                        g05_root=str(g05_root),
                        dataset=str(dataset),
                        output=str(root / "output"),
                        init_checkpoint=str(checkpoint),
                        dataset_stats=str(stats),
                        action_tokenizer=str(tokenizer),
                        processor_path=str(processor),
                        task_config="robodojo_recap",
                        experiment_name="test",
                        gpus=1,
                        steps=1000,
                        save_interval=100,
                        batch_size=1,
                        num_workers=0,
                        grad_accumulation_steps=1,
                        learning_rate=1.0e-5,
                        warmup_steps=100,
                        decay_learning_rate=1.0e-6,
                        decay_start_ratio=0.5,
                        weight_decay=0.01,
                        wandb=False,
                        resume=False,
                        dry_run=True,
                    )
                )

            command = stdout.getvalue()
            self.assertIn("model.lr_scheduler_type=warmup_constant_cosine", command)
            self.assertIn("model.lr_min_ratio=0.1", command)
            self.assertIn("model.constant_end_ratio=0.5", command)
            self.assertIn(
                f"tokenizer.vq_config.ckpt_dir={tokenizer.resolve()}",
                command,
            )
            self.assertIn("g05_finetune_entry.py", command)



class RecapConditioningTest(unittest.TestCase):
    def test_pi05_keeps_positive_and_negative_labels(self):
        self.assertTrue(
            training_prompt(
                "pi05",
                "pick",
                False,
                unconditional_probability=0.1,
                seed=0,
                episode=0,
                frame=0,
            ).endswith("Advantage: negative")
        )

    def test_g05_failure_is_unconditional_and_success_is_positive(self):
        failure = training_prompt(
            "g05",
            "pick",
            False,
            unconditional_probability=0.0,
            seed=0,
            episode=0,
            frame=0,
        )
        success = training_prompt(
            "g05",
            "pick",
            True,
            unconditional_probability=0.0,
            seed=0,
            episode=0,
            frame=0,
        )
        self.assertEqual(failure, "pick")
        self.assertEqual(success, "pick\nAdvantage: positive")
        self.assertEqual(strip_condition(success), "pick")


if __name__ == "__main__":
    unittest.main()
