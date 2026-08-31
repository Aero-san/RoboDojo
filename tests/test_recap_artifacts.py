from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.posttrain.recap_artifacts import check, finalize_iteration


def _write(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _completed_iteration(root: Path) -> Path:
    iteration = root / "iteration_01"
    _write(iteration / "g05/checkpoints/step_100.pt")
    _write(iteration / "rollouts/.recap_stage.json", "{}")
    _write(iteration / "wcm/deploy.pt")
    _write(iteration / "replay_buffer/.recap_stage.json", "{}")
    baseline = iteration / "policy_evaluations/baseline"
    candidate = iteration / "policy_evaluations/step_100"
    _write(baseline / "evaluation.json", "{}")
    _write(candidate / "evaluation.json", "{}")
    _write(
        iteration / "selection.json",
        json.dumps(
            {
                "type": "recap_policy_selection",
                "iteration": 1,
                "baseline": {"rollout_root": "/training-host/baseline"},
                "candidates": [
                    {"step": 100, "rollout_root": "/training-host/step_100"}
                ],
                "continuation": {
                    "step": 100,
                    "checkpoint": "/training-host/g05/checkpoints/step_100.pt",
                },
            }
        ),
    )
    return iteration


class RecapAdvantageArtifactTest(unittest.TestCase):
    def test_expected_episode_count_rejects_partial_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "advantages.jsonl"
            _write(
                path,
                "\n".join(
                    [
                        json.dumps({"type": "recap_advantages"}),
                        json.dumps({"episode_index": 0}),
                    ]
                )
                + "\n",
            )

            self.assertTrue(check("advantages", path, 1))
            self.assertFalse(check("advantages", path, 2))


class RecapArtifactFinalizationTest(unittest.TestCase):
    def test_completed_iteration_merges_diagnostics_and_removes_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            iteration = _completed_iteration(root)
            _write(iteration / "g05/logs/train.log", "current")
            archived_policy = iteration / "g05.incomplete-20260830T010203Z"
            _write(archived_policy / "logs/train.log", "older")
            _write(archived_policy / "eval_snapshots/step_50.html", "snapshot")
            _write(archived_policy / "checkpoints/step_50.pt", "obsolete")
            _write(archived_policy / "action_tokenizer.pt", "obsolete")
            archived_rollout = iteration / "rollouts.incomplete-20260830T020304Z"
            _write(archived_rollout / "rollout_old.log", "diagnostic")
            _write(archived_rollout / "episodes/old/manifest.json", "{}")
            _write(
                iteration / "recap_advantages.jsonl.incomplete-20260830T030405Z",
                "obsolete",
            )
            _write(root / "fixed_norm_stats.incomplete-20260830T040506Z/data.bin")

            result = finalize_iteration(root, iteration)

            self.assertFalse(any(".incomplete-" in path.name for path in root.rglob("*")))
            self.assertEqual(
                (iteration / "g05/eval_snapshots/step_50.html").read_text(),
                "snapshot",
            )
            self.assertEqual(
                (
                    iteration
                    / "g05/history/20260830T010203Z/logs/train.log"
                ).read_text(),
                "older",
            )
            self.assertFalse((iteration / "g05/checkpoints/step_50.pt").exists())
            self.assertFalse((iteration / "rollouts/episodes/old").exists())
            self.assertEqual(
                (iteration / "rollouts/rollout_old.log").read_text(),
                "diagnostic",
            )
            self.assertEqual(len(result["removed_artifacts"]), 4)
            selection = json.loads(
                (iteration / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                selection["continuation"]["checkpoint"],
                str(iteration / "g05/checkpoints/step_100.pt"),
            )
            evaluation = json.loads(
                (
                    iteration / "policy_evaluations/step_100/evaluation.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                evaluation["checkpoint"],
                str(iteration / "g05/checkpoints/step_100.pt"),
            )
            self.assertEqual(result, finalize_iteration(root, iteration))

    def test_selected_incomplete_policy_is_promoted_to_formal_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            iteration = _completed_iteration(root)
            archived = iteration / "g05.incomplete-20260830T010203Z"
            (iteration / "g05").rename(archived)
            selection_path = iteration / "selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection["continuation"]["checkpoint"] = (
                "/training-host/iteration_01/"
                "g05.incomplete-20260830T010203Z/checkpoints/step_100.pt"
            )
            selection_path.write_text(json.dumps(selection), encoding="utf-8")

            result = finalize_iteration(root, iteration)

            self.assertTrue((iteration / "g05/checkpoints/step_100.pt").is_file())
            self.assertFalse(archived.exists())
            selection = json.loads(
                (iteration / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                selection["continuation"]["checkpoint"],
                str(iteration / "g05/checkpoints/step_100.pt"),
            )
            self.assertEqual(
                result["promoted_artifacts"],
                [
                    {
                        "source": "iteration_01/g05.incomplete-20260830T010203Z",
                        "destination": "iteration_01/g05",
                    }
                ],
            )

    def test_unfinished_iteration_is_never_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            iteration = root / "iteration_01"
            archived = iteration / "g05.incomplete-20260830T010203Z"
            _write(archived / "train.log")

            with self.assertRaisesRegex(RuntimeError, "selection.json"):
                finalize_iteration(root, iteration)

            self.assertTrue(archived.is_dir())


if __name__ == "__main__":
    unittest.main()
