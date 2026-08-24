"""Copy immutable normalization statistics from the initial Pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def _checkpoint_root(path: Path) -> Path:
    if (path / "params").is_dir():
        return path
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "params").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No Pi0.5 parameter checkpoint found below {path}")
    return max(candidates, key=lambda child: int(child.name))


def _norm_path(checkpoint: Path, asset_id: str) -> Path:
    assets = checkpoint / "assets"
    if asset_id:
        candidate = assets / asset_id / "norm_stats.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"Initial checkpoint norm stats not found: {candidate}")
        return candidate
    candidates = sorted(assets.glob("*/norm_stats.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one norm_stats.json below {assets}, found {len(candidates)}; "
            "set --asset-id explicitly."
        )
    return candidates[0]


def main(args: argparse.Namespace) -> None:
    requested = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _checkpoint_root(requested)
    source = _norm_path(checkpoint, args.asset_id)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "norm_stats.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "type": "fixed_pi05_normalization",
        "requested_checkpoint": str(requested),
        "resolved_checkpoint": str(checkpoint),
        "source": str(source),
        "asset_id": source.parent.name,
        "sha256": digest,
    }
    manifest_path = output / "fixed_norm_stats.json"
    if destination.exists() or manifest_path.exists():
        if not destination.is_file() or not manifest_path.is_file():
            raise FileExistsError(f"Incomplete fixed normalization artifact: {output}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if previous != manifest or current_digest != digest:
            raise ValueError(
                f"Fixed normalization at {output} does not match initial checkpoint {checkpoint}."
            )
        print(f"reusing fixed Pi0.5 normalization: {destination}")
        return
    shutil.copy2(source, destination)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"copied fixed Pi0.5 normalization: {source} -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-id", default="")
    main(parser.parse_args())
