#!/usr/bin/env python3
"""Sample RoboDojo HDF5 episodes and render multi-camera MP4 previews."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

import cv2
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.save_file import save_json
from XPolicyLab.utils.process_data import decode_image_bit

CAMERA_CANDIDATES = {
    "cam_head": (
        "vision/cam_head/colors",
        "vision/cam_high/colors",
        "vision/cam_third_view/colors",
    ),
    "cam_left_wrist": (
        "vision/cam_left_wrist/colors",
        "vision/cam_wrist/colors",
    ),
    "cam_right_wrist": (
        "vision/cam_right_wrist/colors",
        "vision/cam_wrist/colors",
    ),
}
IMAGE_FIELD_NAMES = {"color", "colors", "image", "images", "rgb"}


@dataclass(frozen=True)
class EpisodeMetadata:
    path: Path
    task: str
    instruction: str
    fps: float
    frames: int
    cameras: tuple[tuple[str, str], ...]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (bytes, np.bytes_)):
        return [bytes(value).decode("utf-8")]
    if isinstance(value, str):
        return [value]
    array = np.asarray(value)
    if array.ndim == 0:
        return _strings(array.item())
    result: list[str] = []
    for item in array.reshape(-1).tolist():
        result.extend(_strings(item))
    return result


def _sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _fallback_task(path: Path) -> str:
    if path.parent.name == "data" and len(path.parents) >= 3:
        return path.parents[2].name
    return path.parent.name


def _task_name(handle: h5py.File, sidecar: dict[str, Any], path: Path) -> str:
    values = _strings(handle.attrs.get("task", sidecar.get("task")))
    return values[0].strip() if values and values[0].strip() else _fallback_task(path)


def _instruction(handle: h5py.File, sidecar: dict[str, Any], task: str) -> str:
    for key in ("instruction", "instructions"):
        if key in handle:
            values = [value.strip() for value in _strings(handle[key][()]) if value.strip()]
            if values:
                return values[0]
    values = [value.strip() for value in _strings(sidecar.get("instruction")) if value.strip()]
    return values[0] if values else task


def _fps(handle: h5py.File, sidecar: dict[str, Any]) -> float:
    value = handle.attrs.get("fps")
    if value is None and "additional_info/frequency" in handle:
        value = handle["additional_info/frequency"][()]
    if value is None:
        value = sidecar.get("fps", 25)
    fps = float(np.asarray(value).reshape(-1)[0])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return fps


def _camera_datasets(handle: h5py.File) -> list[tuple[str, str]]:
    cameras: list[tuple[str, str]] = []
    used_paths: set[str] = set()
    for label, candidates in CAMERA_CANDIDATES.items():
        dataset_path = next(
            (
                candidate
                for candidate in candidates
                if candidate in handle and isinstance(handle[candidate], h5py.Dataset)
            ),
            None,
        )
        if dataset_path is not None and dataset_path not in used_paths:
            cameras.append((label, dataset_path))
            used_paths.add(dataset_path)

    extras: list[str] = []

    def find_extra(name: str, node: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(node, h5py.Dataset) or name in used_paths:
            return
        parts = name.casefold().split("/")
        if parts[-1] not in IMAGE_FIELD_NAMES or not node.shape or node.shape[0] == 0:
            return
        if "vision" in parts or "observation" in parts or "observations" in parts:
            extras.append(name)

    handle.visititems(find_extra)
    for dataset_path in sorted(extras):
        parent = dataset_path.rsplit("/", 1)[0]
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", parent.rsplit("/", 1)[-1])
        cameras.append((label, dataset_path))
    return cameras


def inspect_episode(path: Path, selected_cameras: set[str] | None = None) -> EpisodeMetadata:
    sidecar = _sidecar(path)
    with h5py.File(path, "r") as handle:
        task = _task_name(handle, sidecar, path)
        cameras = _camera_datasets(handle)
        if selected_cameras is not None:
            cameras = [camera for camera in cameras if camera[0] in selected_cameras]
            missing = selected_cameras - {label for label, _ in cameras}
            if missing:
                raise KeyError(f"Missing requested cameras {sorted(missing)} in {path}")
        if not cameras:
            raise KeyError(f"No RGB camera datasets found in {path}")
        horizons = [len(handle[dataset_path]) for _, dataset_path in cameras]
        if not all(horizons):
            raise ValueError(f"Camera stream contains no frames in {path}: {horizons}")
        return EpisodeMetadata(
            path=path.resolve(),
            task=task,
            instruction=_instruction(handle, sidecar, task),
            fps=_fps(handle, sidecar),
            frames=min(horizons),
            cameras=tuple(cameras),
        )


def discover_episodes(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.casefold() not in {".h5", ".hdf5"}:
            raise ValueError(f"Input file must end in .h5 or .hdf5: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = [*input_path.rglob("episode_*.hdf5"), *input_path.rglob("episode_*.h5")]
    files = sorted({path.resolve() for path in files})
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 or episode_*.h5 files below {input_path}")
    return files


def filter_tasks(paths: Sequence[Path], tasks: set[str] | None) -> list[Path]:
    if not tasks:
        return list(paths)
    normalized = {task.casefold() for task in tasks}
    matches: list[Path] = []
    for path in paths:
        sidecar = _sidecar(path)
        with h5py.File(path, "r") as handle:
            task = _task_name(handle, sidecar, path)
        if task.casefold() in normalized:
            matches.append(path)
    if not matches:
        raise ValueError(f"No episodes matched tasks {sorted(tasks)}")
    return matches


def sample_episodes(
    paths: Sequence[Path],
    count: int,
    strategy: str,
    seed: int,
) -> list[Path]:
    if count <= 0:
        raise ValueError("Episode sample count must be positive")
    count = min(count, len(paths))
    if strategy == "first":
        return list(paths[:count])
    if strategy == "random":
        return sorted(random.Random(seed).sample(list(paths), count))
    if strategy == "even":
        if count == 1:
            return [paths[0]]
        indices = [round(index * (len(paths) - 1) / (count - 1)) for index in range(count)]
        return [paths[index] for index in indices]
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _decode_frame(raw: Any, source: str) -> np.ndarray:
    frame = np.asarray(decode_image_bit(raw))
    if frame.ndim == 4 and len(frame) == 1:
        frame = frame[0]
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError(f"Decoded image must be HWC RGB/RGBA, got {frame.shape}: {source}")
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        raise ValueError(f"Decoded image must have uint8 dtype, got {frame.dtype}: {source}")
    return np.ascontiguousarray(frame)


def _ascii(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def _wrap_text(text: str, max_width: int, font_scale: float, thickness: int) -> list[str]:
    words = _ascii(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def _annotate_panel(panel: np.ndarray, label: str) -> None:
    text = _ascii(label)
    font_scale = 0.58
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(panel, (8, 8), (20 + text_width, 18 + text_height + baseline), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (14, 14 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _compose_frame(
    frames: Sequence[np.ndarray],
    labels: Sequence[str],
    metadata: EpisodeMetadata,
    source_index: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    columns = 1 if len(frames) == 1 else 2
    rows = math.ceil(len(frames) / columns)
    header_height = 112
    canvas = np.full(
        (header_height + rows * panel_height, columns * panel_width, 3),
        24,
        dtype=np.uint8,
    )
    title = (
        f"{metadata.task} | {metadata.path.name} | "
        f"frame {source_index + 1}/{metadata.frames} | {source_index / metadata.fps:.2f}s"
    )
    cv2.putText(
        canvas,
        _ascii(title),
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    instruction_lines = _wrap_text(
        f"Instruction: {metadata.instruction}", canvas.shape[1] - 32, 0.52, 1
    )[:3]
    for line_index, line in enumerate(instruction_lines):
        cv2.putText(
            canvas,
            line,
            (16, 58 + 22 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (205, 225, 255),
            1,
            cv2.LINE_AA,
        )
    for index, (frame, label) in enumerate(zip(frames, labels)):
        panel = _fit_panel(frame, panel_width, panel_height)
        _annotate_panel(panel, label)
        row, column = divmod(index, columns)
        y = header_height + row * panel_height
        x = column * panel_width
        canvas[y : y + panel_height, x : x + panel_width] = panel
    return canvas


def render_episode(
    metadata: EpisodeMetadata,
    destination: Path,
    *,
    stride: int,
    max_frames: int,
    panel_width: int,
    fps_override: float | None,
    overwrite: bool,
) -> dict[str, Any]:
    if destination.exists() and not overwrite:
        return {
            "status": "skipped",
            "source": str(metadata.path),
            "output": str(destination.resolve()),
            "task": metadata.task,
            "instruction": metadata.instruction,
        }
    with h5py.File(metadata.path, "r") as handle:
        datasets = [(label, handle[dataset_path]) for label, dataset_path in metadata.cameras]
        first_frames = [
            _decode_frame(dataset[0], f"{metadata.path}:{dataset.name}[0]")
            for _, dataset in datasets
        ]
        panel_height = max(
            2,
            round(max(frame.shape[0] / frame.shape[1] for frame in first_frames) * panel_width),
        )
        panel_height += panel_height % 2
        columns = 1 if len(datasets) == 1 else 2
        rows = math.ceil(len(datasets) / columns)
        output_width = columns * panel_width
        output_height = 112 + rows * panel_height
        output_width += output_width % 2
        output_height += output_height % 2
        output_fps = fps_override if fps_override is not None else metadata.fps / stride
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
        temporary.unlink(missing_ok=True)
        writer: cv2.VideoWriter | None = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            writer.release()
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"OpenCV could not create MPEG-4 video: {temporary}")
        written = 0
        try:
            stop = metadata.frames
            if max_frames > 0:
                stop = min(stop, max_frames * stride)
            for source_index in range(0, stop, stride):
                frames = [
                    _decode_frame(
                        dataset[source_index],
                        f"{metadata.path}:{dataset.name}[{source_index}]",
                    )
                    for _, dataset in datasets
                ]
                canvas = _compose_frame(
                    frames,
                    [label for label, _ in datasets],
                    metadata,
                    source_index,
                    panel_width,
                    panel_height,
                )
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                written += 1
            writer.release()
            writer = None
            temporary.replace(destination)
        except Exception:
            if writer is not None:
                writer.release()
            temporary.unlink(missing_ok=True)
            raise
    return {
        "status": "written",
        "source": str(metadata.path),
        "output": str(destination.resolve()),
        "task": metadata.task,
        "instruction": metadata.instruction,
        "source_fps": metadata.fps,
        "output_fps": output_fps,
        "source_frames": metadata.frames,
        "output_frames": written,
        "cameras": [label for label, _ in metadata.cameras],
    }


def _output_path(input_path: Path, output_dir: Path, episode: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        relative = episode.relative_to(input_path)
    else:
        relative = Path(episode.name)
    return (output_dir / relative).with_suffix(".mp4")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample RoboDojo HDF5 episodes and render labelled multi-camera MP4 previews."
    )
    parser.add_argument("--input", type=Path, default=Path("data/RoboDojo"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hdf5_visualizations"))
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--sampling", choices=("even", "random", "first"), default="even")
    parser.add_argument("--seed", type=int, default=0, help="Seed used by random sampling.")
    parser.add_argument("--task", action="append", help="Task name to include; repeat for multiple tasks.")
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera label to include; repeat as needed (default: every RGB camera).",
    )
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth source frame.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum output frames per video; 0 keeps the complete episode.",
    )
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--fps", type=float, help="Override output FPS (default: source FPS / stride).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.panel_width < 64:
        parser.error("--panel-width must be at least 64")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    paths = filter_tasks(discover_episodes(input_path), set(args.task) if args.task else None)
    selected = sample_episodes(paths, args.num_episodes, args.sampling, args.seed)
    selected_cameras = set(args.camera) if args.camera else None
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures = 0
    print(f"Found {len(paths)} matching episodes; selected {len(selected)} with {args.sampling!r} sampling.")
    for index, path in enumerate(selected, start=1):
        try:
            metadata = inspect_episode(path, selected_cameras)
            destination = _output_path(input_path, output_dir, path)
            print(f"[{index}/{len(selected)}] {path} -> {destination}")
            result = render_episode(
                metadata,
                destination,
                stride=args.stride,
                max_frames=args.max_frames,
                panel_width=args.panel_width,
                fps_override=args.fps,
                overwrite=args.overwrite,
            )
            results.append(result)
        except Exception as exc:
            failures += 1
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            results.append({"status": "error", "source": str(path), "error": str(exc)})

    manifest = {
        "input": str(input_path),
        "output": str(output_dir),
        "available_episodes": len(paths),
        "selected_episodes": len(selected),
        "sampling": args.sampling,
        "seed": args.seed,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "panel_width": args.panel_width,
        "fps_override": args.fps,
        "tasks": args.task,
        "cameras": args.camera,
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    save_json(manifest, manifest_path)
    written = sum(result["status"] == "written" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    print(f"Done: written={written}, skipped={skipped}, failed={failures}; manifest={manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
