"""Write machine-readable and Markdown summaries for a RECAP run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build(run_root: Path) -> dict[str, object]:
    iterations: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []
    for iteration_dir in sorted(run_root.glob("iteration_[0-9][0-9]")):
        iteration = int(iteration_dir.name.rsplit("_", 1)[1])
        evaluations = []
        for evaluation in sorted(
            (iteration_dir / "policy_evaluations").glob("*/evaluation.json")
        ):
            payload = _read(evaluation)
            if payload:
                evaluations.append(payload)
                evaluated.append({"iteration": iteration, **payload})
        iterations.append(
            {
                "iteration": iteration,
                "rollout": _read(iteration_dir / "rollouts/quality.json"),
                "evaluations": evaluations,
                "selection": _read(iteration_dir / "selection.json"),
                "value_video": _read(iteration_dir / "value_videos/summary.json"),
            }
        )
    best_checkpoint = (
        max(
            evaluated,
            key=lambda row: (
                float(row.get("success_rate", 0.0)),
                float(row.get("mean_score", 0.0)),
                int(row["iteration"]),
            ),
        )
        if evaluated
        else None
    )
    return {
        "schema_version": 1,
        "type": "recap_run_report",
        "run_root": str(run_root),
        "best_checkpoint": best_checkpoint,
        "iterations": iterations,
    }


def _rate(value: object) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def markdown(report: dict[str, object]) -> str:
    lines = [
        "# RECAP report",
        "",
        "| Iteration | Rollout | Baseline eval | Best eval | Continue from | Value video |",
        "| ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in report["iterations"]:
        selection = row["selection"] or {}
        baseline = selection.get("baseline", {})
        best_evaluated = selection.get("best_evaluated", {})
        continuation = selection.get("continuation", {})
        rollout = row["rollout"] or {}
        value_video = row["value_video"] or {}
        video_episodes = value_video.get("episodes")
        video_count = len(video_episodes) if isinstance(video_episodes, list) else "—"
        lines.append(
            "| {iteration} | {rollout} | {baseline} | {best} | {last} | {videos} |".format(
                iteration=row["iteration"],
                rollout=_rate(rollout.get("success_rate")),
                baseline=_rate(baseline.get("success_rate")),
                best=_rate(best_evaluated.get("success_rate")),
                last=continuation.get("step", "—"),
                videos=video_count,
            )
        )
    lines.extend(["", "Evaluation provenance and per-checkpoint metrics are in `report.json`.", ""])
    return "\n".join(lines)


def best_markdown(best: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Best evaluated checkpoint",
            "",
            f"- Path: `{best['checkpoint']}`",
            f"- Iteration: {best['iteration']}",
            f"- Evaluation: {best['label']}",
            f"- Success rate: {float(best['success_rate']):.1%}",
            f"- Mean score: {float(best['mean_score']):.6f}",
            f"- Episodes: {best['episodes']}",
            f"- Source: {best['source']}",
            "",
            "Training continuation is intentionally independent of this ranking and always uses "
            "the last checkpoint from each iteration.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    report = build(root)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "report.md").write_text(markdown(report), encoding="utf-8")
    best = report["best_checkpoint"]
    if best:
        (root / "best_checkpoint.json").write_text(
            json.dumps(best, indent=2) + "\n", encoding="utf-8"
        )
        (root / "best_checkpoint.md").write_text(
            best_markdown(best), encoding="utf-8"
        )
        (root / "best_checkpoint.txt").write_text(
            str(best["checkpoint"]) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
