"""Score raw RoboDojo rollouts with WCM and render official value overlays."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(
    os.environ.get(
        "RECAP_REMOTE_REPO_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()
WCM_ROOT = ROOT_DIR / "external_dependencies" / "WCM"
POSTTRAIN_DIR = ROOT_DIR / "scripts" / "posttrain"
sys.path.insert(0, str(WCM_ROOT))
sys.path.insert(0, str(POSTTRAIN_DIR))

from episode_value_video.curves import load_episode_curves  # noqa: E402
from episode_value_video.render import RenderOptions, render_episodes  # noqa: E402
from episode_value_video.sources import VideoMapRepository  # noqa: E402
from episode_value_video.video_io import iter_video_frames  # noqa: E402
from progress import progress_iter  # noqa: E402
from wcm_checkpoint import adapt_wcm_state_dict  # noqa: E402
from world_critic.data import WorldCriticCollator, build_processor  # noqa: E402
from world_critic.model import WorldCriticModel  # noqa: E402
from world_critic.training import config_from_checkpoint_payload  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_model(path: Path, device: torch.device) -> tuple[Any, WorldCriticModel]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") not in {
        "deploy",
        "full_resume",
    }:
        raise ValueError("--wcm-checkpoint must be an official WCM deploy.pt, best.pt, or last.pt.")
    config = config_from_checkpoint_payload(payload)
    model = WorldCriticModel(config.model).to(device).eval()
    model.load_state_dict(
        adapt_wcm_state_dict(payload["model"], model.state_dict()),
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, model


def _selected_episodes(root: Path, max_episodes: int) -> list[tuple[Path, dict[str, Any]]]:
    entries = [
        (manifest_path.parent, _json(manifest_path))
        for manifest_path in (root / "episodes").glob("*/manifest.json")
    ]
    entries.sort(key=lambda item: (int(item[1]["episode_index"]), item[0].name))
    if not entries:
        raise FileNotFoundError(f"No completed rollout manifests below {root / 'episodes'}")
    selected = entries[:max_episodes]
    return [
        (episode_dir, {**manifest, "_render_episode_id": render_episode_id})
        for render_episode_id, (episode_dir, manifest) in enumerate(selected)
    ]


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _flush_batch(
    samples: list[dict[str, Any]],
    *,
    collator: WorldCriticCollator,
    model: WorldCriticModel,
    device: torch.device,
    precision: str,
    frame_indices: list[int],
    values: list[float],
) -> None:
    if not samples:
        return
    batch = collator(samples)
    with torch.inference_mode(), _autocast(device, precision):
        output = model(
            batch["images"].to(device, non_blocking=True),
            batch["actions"].to(device, non_blocking=True),
            batch["instruction_input_ids"].to(device, non_blocking=True),
            batch["instruction_attention_mask"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
    endpoint_values = output.value[:, -1, 0].float().cpu().tolist()
    frame_indices.extend(int(sample["frame_indices"][-1]) for sample in samples)
    values.extend(float(value) for value in endpoint_values)
    samples.clear()


def _score_episode(
    episode_dir: Path,
    manifest: dict[str, Any],
    *,
    config: Any,
    model: WorldCriticModel,
    collator: WorldCriticCollator,
    device: torch.device,
    precision: str,
    batch_size: int,
    backend: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_episode_id = int(manifest["episode_index"])
    episode_id = int(manifest.get("_render_episode_id", source_episode_id))
    instruction = str(manifest.get("task", "")).strip()
    if not instruction:
        raise ValueError(
            f"episode={source_episode_id} rollout manifest has no task instruction."
        )
    length = int(manifest["length"])
    history_size = int(config.data.history_size)
    window_size = history_size + int(config.data.prediction_horizon)
    if int(config.data.prediction_horizon) != 1:
        raise NotImplementedError("Rollout value videos require WCM prediction_horizon=1.")
    if length < window_size:
        raise ValueError(
            f"episode={episode_id} length={length} is shorter than WCM window={window_size}."
        )

    trajectory_path = episode_dir / "trajectory.npz"
    with np.load(trajectory_path) as trajectory:
        actions = np.asarray(trajectory["action"], dtype=np.float32)
    if actions.ndim != 2 or len(actions) != length:
        raise ValueError(
            f"episode={episode_id} trajectory action shape={actions.shape}, manifest length={length}."
        )
    if actions.shape[1] != int(config.model.action_dim):
        raise ValueError(
            f"episode={episode_id} action_dim={actions.shape[1]}, WCM action_dim={config.model.action_dim}."
        )
    if config.data.normalize_action:
        mean = np.asarray(config.data.action_mean, dtype=np.float32)
        std = np.asarray(config.data.action_std, dtype=np.float32)
        actions = (actions - mean) / std

    camera_names = [str(key).rsplit(".", 1)[-1] for key in config.data.image_keys]
    camera_paths = [episode_dir / f"{camera}.mp4" for camera in camera_names]
    missing = [str(path) for path in camera_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode={episode_id} is missing WCM camera videos: {missing}")
    streams = [iter_video_frames(path, backend=backend) for path in camera_paths]
    for path, (probe, _) in zip(camera_paths, streams, strict=True):
        if probe.frame_count is not None and probe.frame_count != length:
            raise ValueError(
                f"episode={episode_id} video {path.name} has {probe.frame_count} frames, expected {length}."
            )

    sentinel = object()
    frame_window: deque[list[np.ndarray]] = deque(maxlen=window_size)
    pending: list[dict[str, Any]] = []
    curve_frames: list[int] = []
    curve_values: list[float] = []
    decoded_frames = 0
    iterators = [iterator for _, iterator in streams]
    for frame_index, frame_group in enumerate(
        zip_longest(*iterators, fillvalue=sentinel)
    ):
        if any(frame is sentinel for frame in frame_group):
            raise ValueError(f"episode={episode_id} camera videos have different frame counts.")
        if frame_index >= length:
            raise ValueError(f"episode={episode_id} camera videos contain more than {length} frames.")
        frame_window.append(
            [np.asarray(frame, dtype=np.uint8) for frame in frame_group]
        )
        decoded_frames += 1
        if len(frame_window) < window_size:
            continue
        start = frame_index - window_size + 1
        pending.append(
            {
                "images": list(frame_window),
                "actions": torch.from_numpy(actions[start : start + history_size]),
                "instruction": instruction,
                "valid_mask": torch.ones(history_size, dtype=torch.bool),
                "episode_id": episode_id,
                "frame_indices": torch.arange(
                    start,
                    start + history_size,
                    dtype=torch.long,
                ),
                "sample_id": f"{episode_id}:{start}",
            }
        )
        if len(pending) >= batch_size:
            _flush_batch(
                pending,
                collator=collator,
                model=model,
                device=device,
                precision=precision,
                frame_indices=curve_frames,
                values=curve_values,
            )
    _flush_batch(
        pending,
        collator=collator,
        model=model,
        device=device,
        precision=precision,
        frame_indices=curve_frames,
        values=curve_values,
    )
    if decoded_frames != length:
        raise ValueError(
            f"episode={episode_id} decoded {decoded_frames} frames, expected {length}."
        )
    expected_curve_points = length - history_size
    if len(curve_values) != expected_curve_points:
        raise RuntimeError(
            f"episode={episode_id} produced {len(curve_values)} values, expected {expected_curve_points}."
        )
    if not all(math.isfinite(value) for value in curve_values):
        raise ValueError(f"episode={episode_id} produced non-finite WCM values.")

    curve = {
        "episode_id": episode_id,
        "frame_indices": curve_frames,
        "values": curve_values,
        "success": bool(manifest["success"]),
        "score": float(manifest["score"]),
        "source": str(episode_dir),
        "source_episode_id": source_episode_id,
        "instruction": instruction,
    }
    video_map = {
        "path": str((episode_dir / "cam_high.mp4").resolve()),
        "fps": float(manifest["fps"]),
        "frame_offset": 0,
        "frame_count": length,
        "history_size": history_size,
        "camera_key": "cam_high",
    }
    return curve, video_map


def main(args: argparse.Namespace) -> None:
    if args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("--precision must be fp32, fp16, or bf16.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not math.isfinite(args.y_min) or not math.isfinite(args.y_max):
        raise ValueError("--y-min and --y-max must be finite.")
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be smaller than --y-max.")
    if not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--speed must be finite and positive.")

    rollout_root = Path(args.rollout_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    config, model = _load_model(
        Path(args.wcm_checkpoint).expanduser().resolve(),
        device,
    )
    collator = WorldCriticCollator(
        build_processor(config.model),
        config.model.vision.image_size,
        config.model.language.max_length,
    )

    selected = _selected_episodes(rollout_root, args.max_episodes)
    curves: list[dict[str, Any]] = []
    video_map: dict[str, dict[str, Any]] = {}
    for episode_dir, manifest in progress_iter(
        selected,
        desc="WCM rollout value inference",
        total=len(selected),
        unit="episode",
    ):
        curve, source = _score_episode(
            episode_dir,
            manifest,
            config=config,
            model=model,
            collator=collator,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
            backend=args.backend,
        )
        curves.append(curve)
        video_map[str(curve["episode_id"])] = source

    curve_path = output_dir / "episode_curves.json"
    video_map_path = output_dir / "head_video_map.json"
    curve_path.write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    video_map_path.write_text(json.dumps(video_map, indent=2) + "\n", encoding="utf-8")

    official_curves = load_episode_curves(curve_path)
    repository = VideoMapRepository(
        video_map_path,
        backend=args.backend,
        history_size=int(config.data.history_size),
    )
    render_result = render_episodes(
        official_curves,
        repository,
        options=RenderOptions(
            output_dir=output_dir / "videos",
            speed=args.speed,
            backend=args.backend,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            scale_mode="global",
            y_min=args.y_min,
            y_max=args.y_max,
            accent=args.accent,
            title=args.title,
        ),
        curve_artifact=curve_path,
    )
    summary = {
        "schema_version": 1,
        "wcm_checkpoint": str(Path(args.wcm_checkpoint).expanduser().resolve()),
        "rollout_root": str(rollout_root),
        "curves": str(curve_path.resolve()),
        "head_video_map": str(video_map_path.resolve()),
        "render_manifest": render_result["manifest"],
        "episodes": [
            {
                "episode_id": curve["episode_id"],
                "success": curve["success"],
                "score": curve["score"],
                "instruction": curve["instruction"],
                "value_min": min(curve["values"]),
                "value_max": max(curve["values"]),
            }
            for curve in curves
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved WCM rollout value videos: {output_dir / 'videos'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcm-checkpoint", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--backend", choices=("auto", "pyav", "ffmpeg"), default="auto")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--codec", default="h264")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--accent", default="#61E4FF")
    parser.add_argument("--title", default="WCM RECAP")
    main(parser.parse_args())
