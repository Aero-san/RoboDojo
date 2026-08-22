#!/usr/bin/env python3
"""Reduce recorded policy initial noise and render rollout/task views."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

# UMAP imports Numba kernels with caching enabled. Conda environments can be
# read-only to the eval process, so point both runtime caches at a writable
# location before importing matplotlib or umap.
_runtime_cache = Path(tempfile.gettempdir()) / f"robodojo-viz-{os.getuid()}"
(_runtime_cache / "numba").mkdir(parents=True, exist_ok=True)
(_runtime_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_runtime_cache / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


DEEP_BLUE = "#08306b"
DEEP_RED = "#99000d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Root containing raw/**/*.npz recordings.")
    parser.add_argument("--output-dir", help="Plot directory (default: INPUT_DIR/plots).")
    parser.add_argument("--method", choices=("umap", "tsne"), default="umap")
    parser.add_argument("--highlight-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_points(root: Path) -> tuple[np.ndarray, list[dict]]:
    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    expected_dim: int | None = None
    for path in sorted((root / "raw").glob("**/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            noise = np.asarray(data["initial_noise_actions"], dtype=np.float32)
            steps = np.asarray(data["rollout_steps"], dtype=np.int64)
            task = str(data["task_id"].item())
            run_id = str(data["run_id"].item())
            episode_index = int(data["episode_index"].item())
            episode_seed = int(data["episode_seed"].item())
            layout_id = int(data["layout_id"].item())
            success = bool(data["success"].item())
        if noise.shape[0] != len(steps):
            raise ValueError(f"Noise/step count mismatch in {path}: {noise.shape[0]} != {len(steps)}")
        flat = noise.reshape(noise.shape[0], -1)
        if expected_dim is None:
            expected_dim = flat.shape[1]
        elif flat.shape[1] != expected_dim:
            raise ValueError(
                f"Cannot jointly reduce different noise dimensions: {flat.shape[1]} in {path}, expected {expected_dim}."
            )
        rollout_id = f"{task}/{run_id}/episode_{episode_index:07d}"
        for point_index, (vector, step) in enumerate(zip(flat, steps, strict=True)):
            vectors.append(vector)
            rows.append(
                {
                    "task_id": task,
                    "rollout_id": rollout_id,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "layout_id": layout_id,
                    "point_index": point_index,
                    "rollout_step": int(step),
                    "success": success,
                    "source": str(path),
                }
            )
    if not vectors:
        raise ValueError(f"No initial-noise recordings found below {root / 'raw'}.")
    return np.stack(vectors), rows


def reduce_points(vectors: np.ndarray, method: str, seed: int) -> np.ndarray:
    if len(vectors) < 3:
        raise ValueError("At least three policy inference points are required for a 2-D visualization.")
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(vectors)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP requires the 'umap-learn' package.") from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(vectors) - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
    else:
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(1.0, (len(vectors) - 1) / 3.0))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
    return np.asarray(reducer.fit_transform(scaled), dtype=np.float32)


def choose_highlights(rows: list[dict], k: int) -> dict[str, list[int]]:
    if k < 1:
        raise ValueError("--highlight-k must be a positive integer.")
    by_task: dict[str, dict[bool, dict[str, list[int]]]] = {}
    for index, row in enumerate(rows):
        by_task.setdefault(row["task_id"], {True: {}, False: {}})[row["success"]].setdefault(
            row["rollout_id"], []
        ).append(index)
    candidates = []
    for task, outcomes in by_task.items():
        if outcomes[True] and outcomes[False]:
            success_id = max(outcomes[True], key=lambda item: len(outcomes[True][item]))
            failure_id = max(outcomes[False], key=lambda item: len(outcomes[False][item]))
            score = min(len(outcomes[True][success_id]), len(outcomes[False][failure_id]))
            candidates.append((score, task, success_id, failure_id))
    if not candidates:
        return {}
    _, _task, success_id, failure_id = max(candidates, key=lambda item: (item[0], item[1]))

    def evenly_spaced(indices: list[int]) -> list[int]:
        count = min(k, len(indices))
        ordered = sorted(indices, key=lambda index: rows[index]["rollout_step"])
        steps = np.asarray([rows[index]["rollout_step"] for index in ordered], dtype=np.float32)
        targets = np.linspace(float(steps[0]), float(steps[-1]), count)
        available = set(range(len(ordered)))
        selected = []
        for target in targets:
            position = min(available, key=lambda item: (abs(float(steps[item]) - target), item))
            available.remove(position)
            selected.append(ordered[position])
        return sorted(selected, key=lambda index: rows[index]["rollout_step"])

    return {
        "success": evenly_spaced(by_task[_task][True][success_id]),
        "failure": evenly_spaced(by_task[_task][False][failure_id]),
    }


def mark_highlights(ax: plt.Axes, embedding: np.ndarray, highlights: dict[str, list[int]]) -> None:
    styles = {
        "success": ("o", "#00ffff", "selected success"),
        "failure": ("D", "#ffd700", "selected failure"),
    }
    for outcome, indices in highlights.items():
        marker, edge, label = styles[outcome]
        xy = embedding[indices]
        ax.plot(xy[:, 0], xy[:, 1], color=edge, linewidth=1.5, alpha=0.8, zorder=4)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=125,
            marker=marker,
            facecolors="none",
            edgecolors=edge,
            linewidths=2.2,
            label=label,
            zorder=5,
        )
        prefix = "S" if outcome == "success" else "F"
        for sequence, (x, y) in enumerate(xy, start=1):
            ax.annotate(
                f"{prefix}{sequence}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color=edge,
                zorder=6,
            )


def plot_outcomes(embedding: np.ndarray, rows: list[dict], highlights: dict[str, list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    success_mask = np.asarray([row["success"] for row in rows], dtype=bool)
    failed_indices = np.flatnonzero(~success_mask)
    if success_mask.any():
        ax.scatter(
            embedding[success_mask, 0],
            embedding[success_mask, 1],
            c=DEEP_BLUE,
            s=24,
            alpha=0.8,
            label="success",
        )
    if len(failed_indices):
        progress = np.zeros(len(failed_indices), dtype=np.float32)
        failure_rollouts: dict[str, list[tuple[int, int]]] = {}
        for local_index, global_index in enumerate(failed_indices):
            row = rows[global_index]
            failure_rollouts.setdefault(row["rollout_id"], []).append((local_index, row["rollout_step"]))
        for points in failure_rollouts.values():
            min_step = min(step for _, step in points)
            max_step = max(step for _, step in points)
            for local_index, step in points:
                span = max_step - min_step
                progress[local_index] = (step - min_step) / span if span > 0 else 0.0
        cmap = LinearSegmentedColormap.from_list("failure_progress", [DEEP_BLUE, DEEP_RED])
        scatter = ax.scatter(
            embedding[failed_indices, 0],
            embedding[failed_indices, 1],
            c=progress,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=26,
            alpha=0.85,
            label="failure",
        )
        fig.colorbar(scatter, ax=ax, label="failure rollout progress (early → late)")
    mark_highlights(ax, embedding, highlights)
    ax.set_title("Initial action noise by rollout outcome")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_tasks(embedding: np.ndarray, rows: list[dict], highlights: dict[str, list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    tasks = sorted({row["task_id"] for row in rows})
    palette = "tab20" if len(tasks) <= 20 else "turbo"
    cmap = plt.get_cmap(palette, max(1, len(tasks)))
    for task_index, task in enumerate(tasks):
        mask = np.asarray([row["task_id"] == task for row in rows], dtype=bool)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=cmap(task_index),
            s=24,
            alpha=0.8,
            label=task,
        )
    mark_highlights(ax, embedding, highlights)
    ax.set_title("Initial action noise by task ID")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(alpha=0.15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    vectors, rows = load_points(root)
    embedding = reduce_points(vectors, args.method, args.seed)
    highlights = choose_highlights(rows, args.highlight_k)
    highlighted_indices = {
        index: name for name, indices in highlights.items() for index in indices
    }
    for index, (row, (x, y)) in enumerate(zip(rows, embedding, strict=True)):
        row["component_1"] = float(x)
        row["component_2"] = float(y)
        row["highlight"] = highlighted_indices.get(index, "")
    coordinates_path = output / f"{args.method}_coordinates.csv"
    with coordinates_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outcome_path = output / f"{args.method}_by_outcome.png"
    task_path = output / f"{args.method}_by_task.png"
    plot_outcomes(embedding, rows, highlights, outcome_path)
    plot_tasks(embedding, rows, highlights, task_path)
    manifest = {
        "method": args.method,
        "points": len(rows),
        "features": int(vectors.shape[1]),
        "tasks": sorted({row["task_id"] for row in rows}),
        "highlighted_rollouts": {
            name: rows[indices[0]]["rollout_id"] for name, indices in highlights.items() if indices
        },
        "outputs": [str(outcome_path), str(task_path), str(coordinates_path)],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not highlights:
        print("[action-noise] no task has both a successful and failed rollout; trajectory highlights omitted.")
    print(f"[action-noise] wrote {outcome_path}")
    print(f"[action-noise] wrote {task_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"[action-noise] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
