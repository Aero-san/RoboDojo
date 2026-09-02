"""Pure metadata summaries for RECAP frame-level advantage labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def positive_statistics(records: list[dict[str, Any]]) -> dict[str, object]:
    """Summarize where frame-level positive labels came from."""

    counts = {
        "total": 0,
        "demonstration": 0,
        "successful_rollout": 0,
        "failed_rollout": 0,
        "unknown_source": 0,
    }
    failed_positions: list[float] = []
    for record in records:
        positive = np.asarray(record["positive"], dtype=bool)
        positive_count = int(positive.sum())
        counts["total"] += positive_count
        source_kind = str(record["source_kind"])
        if source_kind == "demo":
            counts["demonstration"] += positive_count
        elif source_kind == "rollout" and bool(record["success"]):
            counts["successful_rollout"] += positive_count
        elif source_kind == "rollout":
            counts["failed_rollout"] += positive_count
            denominator = max(len(positive) - 1, 1)
            failed_positions.extend(
                float(index) / denominator for index in np.flatnonzero(positive)
            )
        else:
            counts["unknown_source"] += positive_count

    bin_edges = np.linspace(0.0, 1.0, 11)
    histogram, _ = np.histogram(failed_positions, bins=bin_edges)
    return {
        "positive_frame_counts": counts,
        "failed_rollout_positive_normalized_position_histogram": {
            "bin_edges": bin_edges.tolist(),
            "counts": histogram.astype(int).tolist(),
        },
    }


def backfill_advantage_statistics(
    advantages_path: str | Path,
    success_labels_path: str | Path,
) -> int:
    advantages_path = Path(advantages_path).expanduser().resolve()
    success_labels_path = Path(success_labels_path).expanduser().resolve()
    lines = [
        line
        for line in advantages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"RECAP advantage labels are incomplete: {advantages_path}")
    header = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    if header.get("type") != "recap_advantages":
        raise ValueError(f"Unsupported RECAP advantage labels: {advantages_path}")
    labels = {
        int(key): bool(value)
        for key, value in json.loads(
            success_labels_path.read_text(encoding="utf-8")
        ).items()
    }

    changes = 0
    for record in records:
        episode = int(record["episode_index"])
        try:
            success = labels[episode]
        except KeyError as exc:
            raise ValueError(
                f"Success labels are missing episode_index={episode}."
            ) from exc
        if record.get("success") is not success:
            record["success"] = success
            changes += 1
    statistics = positive_statistics(records)
    if header.get("positive_statistics") != statistics:
        header["positive_statistics"] = statistics
        changes += 1

    if changes:
        temporary = advantages_path.with_name(f"{advantages_path.name}.tmp")
        temporary.write_text(
            json.dumps(header)
            + "\n"
            + "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(advantages_path)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advantages", required=True)
    parser.add_argument("--success-labels", required=True)
    args = parser.parse_args()
    changes = backfill_advantage_statistics(
        args.advantages,
        args.success_labels,
    )
    print(
        f"RECAP advantage statistics ready: "
        f"{Path(args.advantages).expanduser()} changes={changes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
