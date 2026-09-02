"""Validate and recoverably archive RECAP stage artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import filecmp
import hashlib
import json
from pathlib import Path
import re
import shutil

_INCOMPLETE_NAME = re.compile(
    r"^(?P<base>.+)\.incomplete-(?P<token>\d{8}T\d{6}Z(?:-\d+)?)$"
)
_MERGEABLE_DIAGNOSTIC_DIRECTORIES = frozenset(
    {"logs", "eval_snapshots", "train_snapshots"}
)
_MERGEABLE_DIAGNOSTIC_SUFFIXES = frozenset({".html", ".log"})


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_buffer(path: Path) -> bool:
    required = (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/success_labels.json",
        "meta/provenance.jsonl",
        "meta/replay_buffer.json",
    )
    if not all((path / item).is_file() for item in required):
        return False
    if (path / "meta/.incremental_update_in_progress").exists():
        return False
    info = _json(path / "meta/info.json")
    try:
        episodes = [
            json.loads(line)
            for line in (path / "meta/episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, TypeError):
        return False
    summary = _json(path / "meta/replay_buffer.json")
    task = str(summary.get("task", "")).strip()
    return (
        int(info.get("total_episodes", -1)) == len(episodes) > 0
        and bool(task)
        and all(episode.get("task_slug") == task for episode in episodes)
    )


def _check_advantages(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        len(rows) > 1
        and rows[0].get("type") == "recap_advantages"
        and (expected <= 0 or len(rows) - 1 == expected)
    )


def _check_policy_dataset(path: Path, expected: int) -> bool:
    complete = (
        (path / "meta/info.json").is_file()
        and (path / "meta/stats.json").is_file()
        and (path / "data").is_dir()
        and not (path / "meta/.recap_update_in_progress").exists()
    )
    if not complete:
        return False
    return expected <= 0 or int(_json(path / "meta/info.json").get("total_episodes", -1)) == expected


def _g05_checkpoints(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    candidates = list((path / "checkpoints").glob("step_*.pt"))
    candidates.extend(candidate for candidate in (path / "last.pt",) if candidate.is_file())
    return candidates


def _g05_step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return 0


def _check_policy(path: Path, expected: int, policy: str) -> bool:
    if policy == "g05":
        required = ("robodojo_g05_model.json", ".hydra/config.yaml", "dataset_stats.json", "action_tokenizer.pt")
        if not all((path / item).is_file() for item in required):
            return False
        checkpoints = _g05_checkpoints(path)
        return bool(checkpoints) and (expected <= 0 or max(map(_g05_step, checkpoints)) >= expected - 1)
    if not (path / "robodojo_pi05_model.json").is_file():
        return False
    checkpoints = [
        child.is_dir()
        and (child / "params").is_dir()
        and (child / "assets").is_dir()
        and (child / "_CHECKPOINT_METADATA").is_file()
        for child in path.iterdir()
    ]
    if expected > 0:
        final = path / str(expected - 1)
        return (
            final.is_dir()
            and (final / "params").is_dir()
            and (final / "assets").is_dir()
            and (final / "_CHECKPOINT_METADATA").is_file()
        )
    return any(checkpoints)


def _check_policy_resume(path: Path, policy: str) -> bool:
    if policy == "g05":
        return bool(_g05_checkpoints(path))
    return any(
        child.is_dir()
        and (child / "params").is_dir()
        and (child / "train_state").is_dir()
        and (child / "_CHECKPOINT_METADATA").is_file()
        for child in path.iterdir()
    ) if path.is_dir() else False


def _check_rollout(path: Path, expected: int) -> bool:
    manifests = list((path / "episodes").glob("*/manifest.json"))
    if len(manifests) != expected:
        return False
    for manifest_path in manifests:
        manifest = _json(manifest_path)
        episode_dir = manifest_path.parent
        if not (episode_dir / "trajectory.npz").is_file():
            return False
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if not (episode_dir / f"{camera}.mp4").is_file():
                return False
        if int(manifest.get("length", 0)) < 1:
            return False
    return True


def _check_value_videos(path: Path, expected: int) -> bool:
    summary_path = path / "summary.json"
    curves_path = path / "episode_curves.json"
    if not summary_path.is_file() or not curves_path.is_file():
        return False
    try:
        summary = _json(summary_path)
        curves = _json(curves_path)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(summary, dict) or not isinstance(curves, list):
        return False
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        return False
    if expected > 0 and (len(episodes) != expected or len(curves) != expected):
        return False
    summary_instructions = {
        int(episode["episode_id"]): str(episode.get("instruction", "")).strip()
        for episode in episodes
        if isinstance(episode, dict) and "episode_id" in episode
    }
    curve_instructions = {
        int(curve["episode_id"]): str(curve.get("instruction", "")).strip()
        for curve in curves
        if isinstance(curve, dict) and "episode_id" in curve
    }
    return (
        len(summary_instructions) == len(episodes)
        and len(curve_instructions) == len(curves)
        and all(summary_instructions.values())
        and summary_instructions == curve_instructions
        and len(list((path / "videos").glob("episode-*.mp4"))) == expected
    )


def _state_path(path: Path) -> Path:
    if path.is_dir() or not path.suffix:
        return path / ".recap_stage.json"
    return path.with_name(f"{path.name}.recap_stage.json")


def _check_fingerprint(path: Path, stage: str, fingerprint: str) -> bool:
    if not fingerprint:
        return True
    state_path = _state_path(path)
    if not state_path.is_file():
        return False
    state = _json(state_path)
    return state.get("stage") == stage and state.get("fingerprint") == fingerprint


def check(stage: str, path: Path, expected: int, fingerprint: str = "", policy: str = "pi05") -> bool:
    if not _check_fingerprint(path, stage, fingerprint):
        return False
    if stage == "buffer":
        return _check_buffer(path)
    if stage == "wcm":
        return (path / "deploy.pt").is_file()
    if stage == "advantages":
        return _check_advantages(path, expected)
    if stage == "policy_dataset":
        return _check_policy_dataset(path, expected)
    if stage == "norm":
        if policy == "g05":
            return (
                (path / "norm_stats.json").is_file()
                and (path / "dataset_stats.json").is_file()
                and (path / "action_tokenizer.pt").is_file()
                and not (path / ".incremental_update_in_progress").exists()
            )
        return (
            (path / "norm_stats.json").is_file()
            and not (path / ".incremental_update_in_progress").exists()
        )
    if stage == "policy":
        return _check_policy(path, expected, policy)
    if stage == "policy_resume":
        return _check_policy_resume(path, policy)
    if stage == "rollout":
        return _check_rollout(path, expected)
    if stage == "value_videos":
        return _check_value_videos(path, expected)
    raise ValueError(f"Unknown stage: {stage}")


def fingerprint(stage: str, entries: list[str]) -> str:
    values: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Fingerprint entry must be KEY=VALUE, got {entry!r}.")
        key, value = entry.split("=", 1)
        if not key or key in values:
            raise ValueError(f"Invalid or duplicate fingerprint key: {key!r}.")
        values[key] = value
    payload = json.dumps(
        {"schema_version": 1, "stage": stage, "values": values},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def mark(stage: str, path: Path, fingerprint_value: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Cannot mark missing RECAP artifact: {path}")
    state_path = _state_path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "fingerprint": fingerprint_value,
                "completed_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)


def archive(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.incomplete-{timestamp}")
    suffix = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.incomplete-{timestamp}-{suffix}")
        suffix += 1
    shutil.move(str(path), str(destination))
    print(f"[RECAP resume] archived incomplete artifact: {path} -> {destination}")


def _selected_checkpoint(iteration_root: Path, continuation: dict) -> Path:
    step = continuation.get("step")
    if step is None:
        raise ValueError("RECAP continuation checkpoint has no step")
    candidates = [
        iteration_root / "g05/checkpoints" / f"step_{int(step)}.pt",
        iteration_root / "pi05" / str(int(step)),
    ]
    recorded = Path(str(continuation["checkpoint"]))
    for index, part in enumerate(recorded.parts):
        match = _INCOMPLETE_NAME.fullmatch(part)
        if match and match.group("base") in {"g05", "pi05"}:
            candidates.append(iteration_root / part / Path(*recorded.parts[index + 1 :]))
            break
    existing = [
        candidate for candidate in candidates if candidate.is_file() or candidate.is_dir()
    ]
    if existing:
        return existing[0]
    raise FileNotFoundError(
        "Selected continuation checkpoint is missing from the completed iteration: "
        + ", ".join(map(str, candidates))
    )


def _validate_completed_iteration(iteration_root: Path) -> dict:
    selection_path = iteration_root / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError(
            f"Refusing to clean an unfinished iteration without selection.json: "
            f"{iteration_root}"
        )
    selection = _json(selection_path)
    if selection.get("type") != "recap_policy_selection":
        raise ValueError(f"Invalid RECAP selection artifact: {selection_path}")
    continuation = selection.get("continuation")
    if not isinstance(continuation, dict) or not continuation.get("checkpoint"):
        raise ValueError(f"RECAP selection has no continuation checkpoint: {selection_path}")
    _selected_checkpoint(iteration_root, continuation)
    required = (
        iteration_root / "rollouts/.recap_stage.json",
        iteration_root / "wcm/deploy.pt",
        iteration_root / "replay_buffer/.recap_stage.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Refusing to clean an unfinished iteration; required artifacts are "
            f"missing: {missing}"
        )
    baseline = selection.get("baseline")
    candidates = selection.get("candidates")
    if not isinstance(baseline, dict) or not isinstance(candidates, list):
        raise ValueError(f"Invalid RECAP evaluation selection: {selection_path}")
    evaluation_paths = [
        iteration_root / "policy_evaluations/baseline/evaluation.json"
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("step") is None:
            raise ValueError(f"Invalid RECAP evaluation candidate: {candidate!r}")
        evaluation_paths.append(
            iteration_root
            / "policy_evaluations"
            / f"step_{int(candidate['step'])}"
            / "evaluation.json"
        )
    missing_evaluations = [str(path) for path in evaluation_paths if not path.is_file()]
    if missing_evaluations:
        raise RuntimeError(
            "Refusing to clean before every selected policy evaluation is complete: "
            f"{missing_evaluations}"
        )
    return selection


def _normalize_selection_paths(
    iteration_root: Path,
    selection: dict,
    selected_checkpoint: Path,
) -> None:
    is_g05 = selected_checkpoint.parent.name == "checkpoints"
    policy_root = (
        selected_checkpoint.parent.parent if is_g05 else selected_checkpoint.parent
    )

    def checkpoint_for(step: int) -> Path:
        if is_g05:
            return policy_root / "checkpoints" / f"step_{step}.pt"
        return policy_root / str(step)

    baseline = selection["baseline"]
    baseline["rollout_root"] = str(
        iteration_root / "policy_evaluations" / "baseline"
    )
    for candidate in selection["candidates"]:
        step = int(candidate["step"])
        checkpoint = checkpoint_for(step)
        candidate["checkpoint"] = str(checkpoint)
        evaluation_root = iteration_root / "policy_evaluations" / f"step_{step}"
        candidate["rollout_root"] = str(evaluation_root)
        evaluation_path = evaluation_root / "evaluation.json"
        evaluation = _json(evaluation_path)
        evaluation["checkpoint"] = str(checkpoint)
        evaluation_temporary = evaluation_path.with_suffix(".json.tmp")
        evaluation_temporary.write_text(
            json.dumps(evaluation, indent=2) + "\n",
            encoding="utf-8",
        )
        evaluation_temporary.replace(evaluation_path)
    best = selection.get("best_evaluated")
    if isinstance(best, dict):
        step = best.get("step")
        best["rollout_root"] = str(
            iteration_root
            / "policy_evaluations"
            / ("baseline" if step is None else f"step_{int(step)}")
        )
        if step is not None:
            best["checkpoint"] = str(checkpoint_for(int(step)))
    continuation = selection["continuation"]
    continuation["checkpoint"] = str(selected_checkpoint)
    continuation["rollout_root"] = str(
        iteration_root
        / "policy_evaluations"
        / f"step_{int(continuation['step'])}"
    )
    selection["policy"] = str(selected_checkpoint)
    selection_path = iteration_root / "selection.json"
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    temporary.replace(selection_path)


def _incomplete_roots(run_root: Path) -> list[tuple[Path, Path, str]]:
    candidates: list[tuple[Path, Path, str]] = []
    for path in run_root.rglob("*"):
        match = _INCOMPLETE_NAME.fullmatch(path.name)
        if match:
            candidates.append(
                (path, path.with_name(match.group("base")), match.group("token"))
            )
    candidates.sort(key=lambda item: len(item[0].parts))
    roots: list[tuple[Path, Path, str]] = []
    for candidate in candidates:
        if any(candidate[0].is_relative_to(root[0]) for root in roots):
            continue
        roots.append(candidate)
    return roots


def _is_mergeable_diagnostic(relative: Path) -> bool:
    return (
        bool(relative.parts)
        and relative.parts[0] in _MERGEABLE_DIAGNOSTIC_DIRECTORIES
    ) or relative.suffix in _MERGEABLE_DIAGNOSTIC_SUFFIXES


def _same_file(left: Path, right: Path) -> bool:
    return (
        left.is_file()
        and right.is_file()
        and filecmp.cmp(left, right, shallow=False)
    )


def _unique_history_target(canonical: Path, token: str, relative: Path) -> Path:
    target = canonical / "history" / token / relative
    suffix = 1
    while target.exists() or target.is_symlink():
        target = target.with_name(f"{target.name}.{suffix}")
        suffix += 1
    return target


def _merge_diagnostics(
    source: Path,
    canonical: Path,
    token: str,
) -> tuple[list[str], int]:
    if not source.is_dir() or not canonical.is_dir():
        return [], 0
    merged: list[str] = []
    duplicates = 0
    files = [
        path
        for path in source.rglob("*")
        if (path.is_file() or path.is_symlink())
        and _is_mergeable_diagnostic(path.relative_to(source))
    ]
    for path in files:
        relative = path.relative_to(source)
        destination = canonical / relative
        if destination.exists() or destination.is_symlink():
            if not path.is_symlink() and _same_file(path, destination):
                duplicates += 1
                continue
            destination = _unique_history_target(canonical, token, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        merged.append(str(destination))
    return merged, duplicates


def finalize_iteration(run_root: Path, iteration_root: Path) -> dict:
    run_root = run_root.expanduser().resolve()
    iteration_root = iteration_root.expanduser().resolve()
    if not iteration_root.is_relative_to(run_root):
        raise ValueError(f"Iteration root must be inside run root: {iteration_root}")
    selection = _validate_completed_iteration(iteration_root)
    selected_checkpoint = _selected_checkpoint(
        iteration_root,
        selection["continuation"],
    )
    sources = _incomplete_roots(run_root)
    selected_source = next(
        (
            source
            for source, _, _ in sources
            if selected_checkpoint.is_relative_to(source)
        ),
        None,
    )
    manifest_path = iteration_root / "incomplete_cleanup.json"
    if not sources and manifest_path.is_file():
        _normalize_selection_paths(iteration_root, selection, selected_checkpoint)
        print("[RECAP cleanup] no incomplete artifacts remain", flush=True)
        return _json(manifest_path)
    merged: list[str] = []
    promoted: list[dict[str, str]] = []
    duplicates = 0
    discarded_files = 0
    removed: list[str] = []
    for source, canonical, token in sources:
        if source == selected_source:
            if canonical.exists() or canonical.is_symlink():
                canonical_files = (
                    sum(
                        1
                        for path in canonical.rglob("*")
                        if path.is_file() or path.is_symlink()
                    )
                    if canonical.is_dir()
                    else 1
                )
                preserved, preserved_duplicates = _merge_diagnostics(
                    canonical,
                    source,
                    "superseded-canonical",
                )
                merged.extend(
                    str((canonical / Path(path).relative_to(source)).relative_to(run_root))
                    for path in preserved
                )
                duplicates += preserved_duplicates
                discarded_files += (
                    canonical_files - len(preserved) - preserved_duplicates
                )
                if canonical.is_dir() and not canonical.is_symlink():
                    shutil.rmtree(canonical)
                else:
                    canonical.unlink(missing_ok=True)
            shutil.move(str(source), str(canonical))
            selected_checkpoint = canonical / selected_checkpoint.relative_to(source)
            removed.append(str(source.relative_to(run_root)))
            promoted.append(
                {
                    "source": str(source.relative_to(run_root)),
                    "destination": str(canonical.relative_to(run_root)),
                }
            )
            continue
        source_files = (
            sum(1 for path in source.rglob("*") if path.is_file() or path.is_symlink())
            if source.is_dir()
            else 1
        )
        merged_files, duplicate_files = _merge_diagnostics(
            source,
            canonical,
            token,
        )
        merged.extend(
            str(Path(path).relative_to(run_root)) for path in merged_files
        )
        duplicates += duplicate_files
        discarded_files += source_files - len(merged_files) - duplicate_files
        removed.append(str(source.relative_to(run_root)))
        if source.is_dir() and not source.is_symlink():
            shutil.rmtree(source)
        else:
            source.unlink(missing_ok=True)

    _normalize_selection_paths(iteration_root, selection, selected_checkpoint)
    manifest = {
        "schema_version": 1,
        "type": "recap_incomplete_cleanup",
        "iteration": selection.get("iteration"),
        "completed_at": datetime.now(UTC).isoformat(),
        "removed_artifacts": removed,
        "promoted_artifacts": promoted,
        "merged_diagnostics": merged,
        "duplicate_diagnostics": duplicates,
        "discarded_files": discarded_files,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        "[RECAP cleanup] removed "
        f"{len(removed)} incomplete artifact(s); promoted {len(promoted)} complete "
        f"artifact(s); merged {len(merged)} diagnostic file(s); discarded "
        f"{discarded_files} obsolete file(s)",
        flush=True,
    )
    return manifest


def main(args: argparse.Namespace) -> None:
    if args.command == "fingerprint":
        print(fingerprint(args.stage, args.entry))
        return
    if args.command == "finalize-iteration":
        finalize_iteration(Path(args.run_root), Path(args.iteration_root))
        return
    if args.command == "iteration-complete":
        _validate_completed_iteration(Path(args.iteration_root).expanduser().resolve())
        return
    if args.command == "selected-checkpoint":
        iteration_root = Path(args.iteration_root).expanduser().resolve()
        selection = _json(iteration_root / "selection.json")
        print(_selected_checkpoint(iteration_root, selection["continuation"]))
        return
    path = Path(args.path).expanduser().resolve()
    if args.command == "matches":
        if not _check_fingerprint(path, args.stage, args.fingerprint):
            raise SystemExit(1)
        return
    if args.command == "check":
        if not check(args.stage, path, args.expected, args.fingerprint, args.policy):
            raise SystemExit(1)
        return
    if args.command == "mark":
        mark(args.stage, path, args.fingerprint)
        return
    archive(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    checker = subparsers.add_parser("check")
    checker.add_argument("--stage", required=True)
    checker.add_argument("--path", required=True)
    checker.add_argument("--expected", type=int, default=0)
    checker.add_argument("--fingerprint", default="")
    checker.add_argument("--policy", choices=("pi05", "g05"), required=True)
    marker = subparsers.add_parser("mark")
    marker.add_argument("--stage", required=True)
    marker.add_argument("--path", required=True)
    marker.add_argument("--fingerprint", required=True)
    matcher = subparsers.add_parser("matches")
    matcher.add_argument("--stage", required=True)
    matcher.add_argument("--path", required=True)
    matcher.add_argument("--fingerprint", required=True)
    digester = subparsers.add_parser("fingerprint")
    digester.add_argument("--stage", required=True)
    digester.add_argument("--entry", action="append", default=[])
    archiver = subparsers.add_parser("archive")
    archiver.add_argument("--path", required=True)
    finalizer = subparsers.add_parser("finalize-iteration")
    finalizer.add_argument("--run-root", required=True)
    finalizer.add_argument("--iteration-root", required=True)
    iteration_checker = subparsers.add_parser("iteration-complete")
    iteration_checker.add_argument("--iteration-root", required=True)
    selector = subparsers.add_parser("selected-checkpoint")
    selector.add_argument("--iteration-root", required=True)
    main(parser.parse_args())
