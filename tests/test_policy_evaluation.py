from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.posttrain import policy_evaluation, select_recap_policy, write_recap_report


def _episode(root: Path, name: str, index: int, success: bool) -> None:
    episode = root / "episodes" / name
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text(
        json.dumps({"episode_index": index, "success": success, "score": float(success)}),
        encoding="utf-8",
    )
    (episode / "trajectory.npz").touch()
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        (episode / f"{camera}.mp4").touch()


class PolicyEvaluationTest(unittest.TestCase):
    def test_reuses_rollouts_and_appends_only_remote_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = root / "rollout"
            remote = root / "remote"
            output = root / "evaluation"
            _episode(rollout, "rollout-0", 0, True)
            _episode(rollout, "rollout-1", 1, False)
            _episode(remote, "remote-2", 2, True)
            args = SimpleNamespace(
                rollout_root=str(rollout),
                remote_root=str(remote),
                output=str(output),
                checkpoint=str(root / "checkpoint"),
                label="baseline",
                episodes=3,
                reuse_episodes=2,
                layout_seed=7,
                layout_offset=0,
            )

            policy_evaluation.reuse(args)

            episodes = sorted((output / "episodes").iterdir())
            self.assertEqual(len(episodes), 3)
            self.assertTrue(all(path.is_symlink() for path in episodes))
            record = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(record["source"], "rollout+remote")
            self.assertEqual(record["reused_rollout_episodes"], 2)
            self.assertEqual(record["remote_episodes"], 1)
            self.assertEqual(record["success_rate"], 2 / 3)

    def test_report_records_evaluation_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            iteration = root / "iteration_01"
            evaluation = iteration / "policy_evaluations/baseline"
            evaluation.mkdir(parents=True)
            (evaluation / "evaluation.json").write_text(
                json.dumps(
                    {
                        "checkpoint": "/checkpoint/best",
                        "label": "baseline",
                        "source": "rollout",
                        "success_rate": 0.5,
                        "mean_score": 0.6,
                    }
                ),
                encoding="utf-8",
            )

            report = write_recap_report.build(root)

            self.assertEqual(
                report["iterations"][0]["evaluations"][0]["source"], "rollout"
            )
            self.assertEqual(
                report["best_checkpoint"]["checkpoint"], "/checkpoint/best"
            )

    def test_last_checkpoint_continues_even_when_baseline_scores_higher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            step_100 = root / "step_100"
            step_499 = root / "step_499"
            for index in range(2):
                _episode(baseline, f"baseline-{index}", index, True)
                _episode(step_100, f"step-100-{index}", index, index == 0)
                _episode(step_499, f"step-499-{index}", index, False)

            args = SimpleNamespace(
                iteration=1,
                baseline_checkpoint=str(root / "initial"),
                baseline_rollouts=str(baseline),
                candidate=[
                    f"100::{root / 'policy/100'}::{step_100}",
                    f"499::{root / 'policy/499'}::{step_499}",
                ],
                output=str(root / "selection.json"),
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                select_recap_policy.main(args)

            selection = json.loads((root / "selection.json").read_text())
            self.assertEqual(
                selection["best_evaluated"]["checkpoint"],
                str((root / "initial").resolve()),
            )
            self.assertEqual(
                selection["continuation"]["checkpoint"],
                str((root / "policy/499").resolve()),
            )
            self.assertEqual(
                stdout.getvalue().strip(), str((root / "policy/499").resolve())
            )
