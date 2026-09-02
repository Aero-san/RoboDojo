"""Read RoboDojo's internal LeRobot-v2.1 replay dataset for the official WCM.

The ingestion layer normalizes each supported source format into this internal
layout. This module implements the small dataset interface consumed by
``external_dependencies/WCM/world_critic/data.py`` so the official WCM model
and trainer can run against RoboDojo files directly without mutating the
downloaded dataset.

It is deliberately an adapter, not a second WCM implementation: temporal
window construction, image preprocessing, language tokenization, model
forward, losses, checkpoints, and evaluation remain in the WCM submodule.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as parquet

try:
    from .progress import progress_iter
except ImportError:  # The WCM launcher imports this adapter as a top-level module.
    from progress import progress_iter


class _ArrowLikeColumn:
    def __init__(self, values: Any) -> None:
        self.values = values

    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray:
        del zero_copy_only
        return np.asarray(self.values)


class _ColumnTable:
    """Minimal HF/Arrow column facade used by WCM's schema helpers."""

    def __init__(self, columns: dict[str, Any]) -> None:
        self._columns = columns
        self.column_names = list(columns)
        self.data = self

    def __getitem__(self, key: str) -> Any:
        return self._columns[key]

    def column(self, key: str) -> _ArrowLikeColumn:
        return _ArrowLikeColumn(self._columns[key])


class _VideoCache:
    """Small per-worker OpenCV cache for random-access temporal windows."""

    def __init__(self, max_open_files: int = 8) -> None:
        self.max_open_files = max_open_files
        self._captures: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._av_containers: OrderedDict[str, Any] = OrderedDict()
        self._av_iters: dict[str, Any] = {}
        self._pyav_paths: set[str] = set()
        self._last_frame: dict[str, int] = {}

    def _capture(self, path: Path) -> cv2.VideoCapture:
        key = str(path)
        capture = self._captures.pop(key, None)
        if capture is None:
            capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open RoboDojo video: {path}")
        self._captures[key] = capture
        while len(self._captures) > self.max_open_files:
            old_key, old_capture = self._captures.popitem(last=False)
            old_capture.release()
            self._last_frame.pop(old_key, None)
        return capture

    def read(self, path: Path, frame_index: int) -> np.ndarray:
        key = str(path)
        if key in self._pyav_paths or os.environ.get("WCM_VIDEO_DECODER", "pyav").lower() == "pyav":
            self._pyav_paths.add(key)
            return self._read_pyav(path, frame_index)
        try:
            capture = self._capture(path)
        except RuntimeError:
            self._pyav_paths.add(key)
            return self._read_pyav(path, frame_index)
        previous = self._last_frame.get(key)
        if previous is None or frame_index <= previous:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            # Sequential reads are substantially faster than seeking for the
            # K=3 windows used by WCM.
            capture.set(cv2.CAP_PROP_POS_FRAMES, previous + 1)
        image = None
        current = previous + 1 if previous is not None and frame_index > previous else frame_index
        while current <= frame_index:
            try:
                ok, image = capture.read()
            except Exception:
                self._pyav_paths.add(key)
                return self._read_pyav(path, frame_index)
            if not ok:
                self._pyav_paths.add(key)
                return self._read_pyav(path, frame_index)
            current += 1
        self._last_frame[key] = frame_index
        # OpenCV returns BGR while the WCM image processor expects RGB.
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_pyav(self, path: Path, frame_index: int) -> np.ndarray:
        """Decode AV1 when the host OpenCV build lacks a software decoder."""
        import av

        key = str(path)
        container = self._av_containers.pop(key, None)
        if container is None:
            container = av.open(key)
        self._av_containers[key] = container
        previous = self._last_frame.get(key)
        if previous is None or frame_index <= previous or key not in self._av_iters:
            container.seek(0, stream=container.streams.video[0])
            iterator = container.decode(video=0)
            current = 0
        else:
            iterator = self._av_iters[key]
            current = previous + 1
        image = None
        while current <= frame_index:
            try:
                image = next(iterator).to_ndarray(format="rgb24")
            except StopIteration as exc:
                raise RuntimeError(f"Could not decode frame {frame_index} from {path}") from exc
            current += 1
        self._av_iters[key] = iterator
        self._last_frame[key] = frame_index
        while len(self._av_containers) > self.max_open_files:
            old_key, old_container = self._av_containers.popitem(last=False)
            old_container.close()
            self._av_iters.pop(old_key, None)
            self._last_frame.pop(old_key, None)
        return image

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        for container in self._av_containers.values():
            container.close()
        self._captures.clear()
        self._av_containers.clear()
        self._av_iters.clear()
        self._pyav_paths.clear()
        self._last_frame.clear()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scalar(values: Any) -> np.ndarray:
    return np.asarray(values.to_pylist() if hasattr(values, "to_pylist") else values).reshape(-1)


def _array_column(table: Any, key: str) -> list[np.ndarray]:
    return [np.asarray(value) for value in table[key].to_pylist()]


def _resolve_root(root: str | Path, repo_id: str | None) -> Path:
    path = Path(root).expanduser().resolve()
    if (path / "meta" / "info.json").exists():
        return path
    if repo_id:
        nested = path.joinpath(*repo_id.split("/"))
        if (nested / "meta" / "info.json").exists():
            return nested
    raise FileNotFoundError(
        f"RoboDojo dataset metadata not found below {path}; expected meta/info.json."
    )


def _task_map(root: Path, fallback: str) -> dict[int, str]:
    task_path = root / "meta" / "tasks.jsonl"
    if not task_path.exists():
        return {0: fallback}
    result: dict[int, str] = {}
    for row in _load_jsonl(task_path):
        task_index = int(row["task_index"])
        text = str(row.get("task", row.get("instruction", fallback))).strip()
        if text:
            result[task_index] = text
    return result or {0: fallback}


def _task_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


_TASK_SELECTOR_STOPWORDS = {
    "a",
    "all",
    "and",
    "at",
    "board",
    "by",
    "from",
    "in",
    "into",
    "of",
    "on",
    "play",
    "the",
    "then",
    "to",
    "using",
    "with",
}

_TASK_SELECTOR_SYNONYMS = {
    "classify": {"sort"},
    "fasten": {"tighten"},
    "imitate": {"observe"},
    "sequence": {"order"},
    "sorting": {"order"},
}

# A few benchmark names are intentionally much shorter than their natural
# language instructions.  These hints are still matched against metadata;
# they do not hard-code episode indices or alter the source dataset.
_TASK_SELECTOR_HINTS = {
    "organize_table": {"mouse", "keyboard", "figurine", "drawer"},
    "swap_blocks": {"empty", "mat", "button"},
    "swap_t": {"t", "shaped", "orientation"},
}
_MULTI_INSTRUCTION_HINTS = {
    "general_pickup": {"pick", "up"},
}


def _task_selector_tokens(value: str) -> set[str]:
    """Return the content words used for a benchmark-style task selector.

    The task names used by RoboDojo (for example ``fill_egg_holder``) are
    concise labels, whereas the v2.1 metadata stores natural-language
    instructions (for example ``Place the four eggs ...``).  Comparing all
    words literally therefore misses valid tasks.  Removing grammatical
    words and normalising simple plurals gives us a small, deterministic
    bridge without introducing a fuzzy-edit-distance dependency.
    """
    tokens = set(_task_slug(value).split("_")) - _TASK_SELECTOR_STOPWORDS
    normalised: set[str] = set()
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        elif token == "toaster":
            token = "toast"
        normalised.add(token)
        normalised.update(_TASK_SELECTOR_SYNONYMS.get(token, ()))
    return normalised


def _episode_task_text(episode: dict[str, Any], task_map: dict[int, str]) -> str:
    tasks = episode.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        return ""
    value = tasks[0]
    if isinstance(value, int):
        return task_map.get(value, "")
    text = str(value).strip()
    if text.isdigit():
        return task_map.get(int(text), "")
    return text


def filter_episode_metadata(
    episodes: list[dict[str, Any]],
    selector: str | None,
    task_metadata: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Select one RoboDojo task by text or a benchmark-style slug.

    RoboDojo's v2.1 episode metadata stores the natural-language instruction,
    while benchmark commands usually use names such as ``stack_bowls``.  An
    exact instruction, an exact normalized slug, or a unique content-word
    match is accepted.  Ambiguous short selectors fail instead of silently
    mixing tasks.  ``task_metadata`` is read from ``meta/tasks.jsonl`` when
    available; it also lets exports that store a numeric task id in
    ``episodes.jsonl`` be filtered correctly.
    """
    if selector is None or not selector.strip():
        return episodes, None
    requested = selector.strip()
    requested_slug = _task_slug(requested).removesuffix("_random")
    selected_by_slug = [
        episode
        for episode in episodes
        if _task_slug(str(episode.get("task_slug", ""))).removesuffix("_random")
        == requested_slug
    ]
    if selected_by_slug:
        return selected_by_slug, requested
    task_map = {
        int(row["task_index"]): str(row.get("task", row.get("instruction", ""))).strip()
        for row in (task_metadata or [])
        if "task_index" in row
    }
    by_text: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        text = _episode_task_text(episode, task_map)
        if text:
            by_text.setdefault(text, []).append(episode)
    if not by_text:
        raise ValueError("RoboDojo episode metadata contains no task instructions.")

    candidates = list(by_text)
    requested_folded = requested.casefold()
    requested_slug = _task_slug(requested)
    selector_hint = _TASK_SELECTOR_HINTS.get(requested_slug.removesuffix("_random"))
    matches = [text for text in candidates if text.casefold() == requested_folded]
    if not matches:
        if requested_slug.isdigit() and int(requested_slug) in task_map:
            matches = [task_map[int(requested_slug)]]
        else:
            matches = [text for text in candidates if _task_slug(text) == requested_slug]
    multi_hint = _MULTI_INSTRUCTION_HINTS.get(requested_slug.removesuffix("_random"))
    if not matches and multi_hint is not None:
        matches = [
            text for text in candidates
            if multi_hint <= set(_task_slug(text).split("_"))
        ]
    if not matches and selector_hint is None:
        requested_tokens = set(_task_slug(requested).split("_")) - {""}
        matches = [
            text
            for text in candidates
            if requested_tokens and requested_tokens <= set(_task_slug(text).split("_"))
        ]
    if not matches:
        # Canonical RoboDojo names and metadata instructions use different
        # verbs (``fill`` vs ``place``, ``put`` vs ``throw``) and sometimes
        # different grammatical forms (``bottles`` vs ``bottle``). Match on
        # the distinctive content words as a final, deterministic step.
        requested_tokens = selector_hint or _task_selector_tokens(requested)
        scored = [
            (len(requested_tokens & _task_selector_tokens(text)), text)
            for text in candidates
        ]
        best_score = max((score for score, _ in scored), default=0)
        best_candidates = [text for score, text in scored if score == best_score]
        if best_score >= 2 or (best_score == 1 and len(best_candidates) == 1):
            matches = best_candidates
    if not matches:
        available = "; ".join(candidates[:12])
        raise ValueError(
            f"Task selector {selector!r} did not match RoboDojo metadata. "
            f"Available examples: {available}"
        )
    if len(matches) > 1 and multi_hint is None:
        raise ValueError(
            f"Task selector {selector!r} is ambiguous; matches={matches}. "
            "Use the complete task instruction."
        )
    selected_text = matches[0] if len(matches) == 1 else requested
    matched_texts = set(matches)
    selected = [
        episode
        for episode in episodes
        if _episode_task_text(episode, task_map) in matched_texts
    ]
    if not selected:
        raise ValueError(f"Task selector {selector!r} matched no episodes.")
    return selected, selected_text


def _success_labels(path: str | None) -> dict[int, bool]:
    if not path:
        return {}
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WCM_SUCCESS_LABELS must be a JSON object mapping episode_index to bool.")
    result: dict[int, bool] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            parsed = value
        elif isinstance(value, int) and value in (0, 1):
            parsed = bool(value)
        else:
            raise ValueError(f"Success label for episode_index={key} must be bool or 0/1.")
        result[int(key)] = parsed
    return result


def _returns(
    episode_ids: np.ndarray,
    rewards: dict[int, np.ndarray],
    labels: dict[int, bool],
    failure_penalty: float,
    gamma: float = 1.0,
    normalization_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success_by_episode: dict[int, bool] = {}
    for episode, values in rewards.items():
        if episode in labels:
            success_by_episode[episode] = labels[episode]
        else:
            success_by_episode[episode] = bool(np.any(values > 0))
    if not success_by_episode or (not any(success_by_episode.values()) and not labels):
        raise ValueError(
            "RoboDojo data contains no positive success signal. Pass "
            "WCM_SUCCESS_LABELS=/path/to/episode_labels.json for rollout failures."
        )

    if not 0.0 < gamma <= 1.0:
        raise ValueError("WCM_GAMMA must be in (0, 1].")
    if not np.isfinite(failure_penalty) or failure_penalty <= 0.0:
        raise ValueError("WCM_FAILURE_PENALTY must be a finite positive number.")

    raw = np.empty((len(episode_ids),), dtype=np.float32)
    for episode in np.unique(episode_ids):
        episode = int(episode)
        rows = np.flatnonzero(episode_ids == episode)
        terminal = 0.0 if success_by_episode[episode] else -failure_penalty
        values = np.empty(len(rows), dtype=np.float32)
        values[-1] = terminal
        for index in range(len(rows) - 2, -1, -1):
            values[index] = -1.0 + gamma * values[index + 1]
        raw[rows] = values
    minimum, maximum = normalization_bounds or (float(raw.min()), float(raw.max()))
    if float(raw.min()) < minimum - 1e-5 or float(raw.max()) > maximum + 1e-5:
        raise ValueError(
            f"Return values [{float(raw.min())}, {float(raw.max())}] exceed the supplied "
            f"global normalization bounds [{minimum}, {maximum}]."
        )
    normalized = np.full_like(raw, -0.5)
    if maximum - minimum >= 1e-8:
        # RECAP trains its critic in normalized return space [-1, 0].  Keep
        # this exact affine transform available to advantage annotation so
        # the N-step reward term is normalized in the same coordinate system.
        normalized[:] = (raw - minimum) / (maximum - minimum) - 1.0
    success_rows = np.asarray([success_by_episode[int(value)] for value in episode_ids], dtype=np.int8)
    return normalized, raw, success_rows


def _return_reference_bounds(
    dataset_root: Path,
    failure_penalty: float,
    gamma: float,
) -> tuple[float, float] | None:
    summary_path = dataset_root / "meta/replay_buffer.json"
    if not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    if summary.get("type") != "wcm_update_subset" or not summary.get("source_buffer"):
        return None
    reference_root = Path(summary["source_buffer"]).expanduser().resolve()
    episodes = _load_jsonl(reference_root / "meta/episodes.jsonl")
    labels = _success_labels(str(reference_root / "meta/success_labels.json"))
    if not episodes:
        raise ValueError(f"Return-reference replay buffer has no episodes: {reference_root}")
    episode_ids = np.concatenate(
        [np.full(int(row["length"]), int(row["episode_index"]), dtype=np.int64) for row in episodes]
    )
    rewards = {
        int(row["episode_index"]): np.zeros(int(row["length"]), dtype=np.float32)
        for row in episodes
    }
    _, raw, _ = _returns(episode_ids, rewards, labels, failure_penalty, gamma)
    return float(raw.min()), float(raw.max())


class RoboDojoDataset:
    """A WCM-compatible view over RoboDojo's local v2.1 dataset."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        task_selector: str | None = None,
    ) -> None:
        self.root = _resolve_root(root, repo_id)
        self.info = _load_json(self.root / "meta" / "info.json")
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(
                "RoboDojoDataset reads only the normalized internal LeRobot v2.1 layout; "
                f"{self.root / 'meta/info.json'} declares {self.info.get('codebase_version')!r}."
            )
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.task_fallback = self.root.parent.parent.name
        source_task_map = _task_map(self.root, self.task_fallback)
        self.meta = SimpleNamespace(tasks=source_task_map)
        self._video_cache = _VideoCache()
        self._video_paths: dict[tuple[int, str], Path] = {}
        self._episodes = _load_jsonl(self.root / "meta" / "episodes.jsonl")
        if not self._episodes:
            raise ValueError(f"Dataset has no episodes: {self.root}")
        task_metadata_path = self.root / "meta" / "tasks.jsonl"
        task_metadata = _load_jsonl(task_metadata_path) if task_metadata_path.exists() else None
        self._episodes, self.task_name = filter_episode_metadata(
            self._episodes, task_selector, task_metadata
        )

        rewards: dict[int, np.ndarray] = {}
        self._row_episode: list[int] = []
        self._row_local_frame: list[int] = []
        self._row_task: list[int] = []
        self._row_instruction: list[str] = []
        self._row_action: list[np.ndarray] = []
        self._row_reference_action: list[np.ndarray] = []
        self._row_state: list[np.ndarray] = []
        task_ids_by_instruction: dict[str, int] = {}
        task_text_by_id: dict[int, str] = {}
        self._has_state = "observation.state" in self.info.get("features", {})

        data_template = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        video_template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        image_keys = [
            key for key, feature in self.info.get("features", {}).items()
            if key.startswith("observation.images.") and feature.get("dtype") == "video"
        ]
        if not image_keys:
            raise ValueError(f"No observation.images.* video features found in {self.root / 'meta/info.json'}.")
        self.image_keys = image_keys

        for episode_meta in progress_iter(
            self._episodes,
            desc="Loading RoboDojo episodes",
            total=len(self._episodes),
            unit="episode",
        ):
            episode = int(episode_meta["episode_index"])
            chunk = episode // self.chunks_size
            parquet_path = self.root / data_template.format(episode_chunk=chunk, episode_index=episode)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")
            columns = ["action", "episode_index", "frame_index", "task_index"]
            has_reference_action = "reference_action" in self.info.get("features", {})
            if has_reference_action:
                columns.append("reference_action")
            if self._has_state:
                columns.append("observation.state")
            reward_key = "next.reward" if "next.reward" in self.info.get("features", {}) else None
            if reward_key:
                columns.append(reward_key)
            table = parquet.read_table(parquet_path, columns=columns)
            actual_length = table.num_rows
            expected_length = int(episode_meta.get("length", actual_length))
            if actual_length != expected_length:
                raise ValueError(f"Episode {episode} metadata length={expected_length}, parquet rows={actual_length}.")
            ep_ids = _scalar(table["episode_index"]).astype(np.int64)
            if not np.all(ep_ids == episode):
                raise ValueError(f"Episode parquet contains mixed episode_index values: {parquet_path}")
            frame_ids = _scalar(table["frame_index"]).astype(np.int64)
            source_task_ids = _scalar(table["task_index"]).astype(np.int64)
            episode_tasks = episode_meta.get("tasks", [])
            if isinstance(episode_tasks, list) and episode_tasks:
                instruction = str(episode_tasks[0]).strip()
            else:
                instruction = str(source_task_map.get(int(source_task_ids[0]), self.task_fallback)).strip()
            if not instruction:
                raise ValueError(f"Episode {episode} has an empty task instruction.")
            if instruction not in task_ids_by_instruction:
                task_id = len(task_ids_by_instruction)
                task_ids_by_instruction[instruction] = task_id
                task_text_by_id[task_id] = instruction
            task_ids = np.full(actual_length, task_ids_by_instruction[instruction], dtype=np.int64)
            if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) == 1):
                raise ValueError(f"Episode {episode} has non-consecutive frame_index values.")
            actions = _array_column(table, "action")
            reference_actions = (
                _array_column(table, "reference_action") if has_reference_action else actions
            )
            states = _array_column(table, "observation.state") if self._has_state else []
            self._row_episode.extend([episode] * actual_length)
            self._row_local_frame.extend(range(actual_length))
            self._row_task.extend(task_ids.tolist())
            self._row_instruction.extend([instruction] * actual_length)
            self._row_action.extend(actions)
            self._row_reference_action.extend(reference_actions)
            self._row_state.extend(states)
            if reward_key:
                rewards[episode] = _scalar(table[reward_key]).astype(np.float32)
            for key in image_keys:
                relative = video_template.format(video_key=key, episode_chunk=chunk, episode_index=episode)
                video_path = self.root / relative
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing episode video for {key}: {video_path}")
                self._video_paths[(episode, key)] = video_path

        self._row_episode = np.asarray(self._row_episode, dtype=np.int64)
        self._row_local_frame = np.asarray(self._row_local_frame, dtype=np.int64)
        self._row_task = np.asarray(self._row_task, dtype=np.int64)
        self.meta.tasks = task_text_by_id
        action_dim = int(self._row_action[0].size)
        if self._has_state:
            state_values = np.stack([value.reshape(-1) for value in self._row_state]).astype(np.float32)
        else:
            state_values = None
        labels_path = os.environ.get("WCM_SUCCESS_LABELS")
        labels = _success_labels(labels_path)
        if not rewards:
            if not labels and os.environ.get("WCM_ASSUME_SUCCESS", "1") == "1":
                # RoboDojo's released demonstrations are successful expert
                # trajectories and do not contain a reward column.  A rollout
                # dataset with failures should always provide explicit labels.
                labels = {int(meta["episode_index"]): True for meta in self._episodes}
            if not labels:
                raise ValueError(
                    "RoboDojo dataset has no reward column. Pass WCM_SUCCESS_LABELS="
                    "/path/to/episode_labels.json or set WCM_ASSUME_SUCCESS=1 for expert-only data."
                )
            rewards = {
                int(episode_meta["episode_index"]): np.zeros(int(episode_meta["length"]), dtype=np.float32)
                for episode_meta in self._episodes
            }
        if labels:
            for episode in self._row_episode:
                if int(episode) not in labels:
                    raise ValueError(f"WCM_SUCCESS_LABELS is missing episode_index={int(episode)}.")
        failure_penalty_raw = os.environ.get("WCM_FAILURE_PENALTY", "")
        failure_penalty = float(failure_penalty_raw) if failure_penalty_raw else 300.0
        gamma = float(os.environ.get("WCM_GAMMA", "1.0"))
        normalization_bounds = _return_reference_bounds(self.root, failure_penalty, gamma)
        returns, raw_returns, success_rows = _returns(
            self._row_episode,
            rewards,
            labels,
            failure_penalty,
            gamma,
            normalization_bounds,
        )
        self._returns = returns
        self._raw_returns = raw_returns
        self._success_rows = success_rows
        self._return_normalization = "global_minmax"
        self._return_raw_min = (
            normalization_bounds[0] if normalization_bounds is not None else float(raw_returns.min())
        )
        self._return_raw_max = (
            normalization_bounds[1] if normalization_bounds is not None else float(raw_returns.max())
        )
        self._return_gamma = gamma
        self._failure_penalty = failure_penalty
        self._columns = {
            "episode_index": self._row_episode,
            "frame_index": np.concatenate(
                [np.arange(int(meta["length"]), dtype=np.int64) for meta in self._episodes]
            ),
            "task_index": self._row_task,
            "action": [value for value in self._row_action],
            "reference_action": [value for value in self._row_reference_action],
            "return": self._returns,
            "return_raw": self._raw_returns,
            "episode_success": self._success_rows,
        }
        if state_values is not None:
            self._columns["observation.state"] = [value for value in self._row_state]
        self.hf_dataset = _ColumnTable(self._columns)
        features = dict(self.info.get("features", {}))
        features["return"] = {"dtype": "float32", "shape": (1,)}
        features["return_raw"] = {"dtype": "float32", "shape": (1,)}
        features["episode_success"] = {"dtype": "int8", "shape": (1,)}
        self.features = features
        self._action_dim = action_dim
        self._state_dim = int(state_values.shape[-1]) if state_values is not None else None

    def __len__(self) -> int:
        return int(self._row_episode.size)

    def close(self) -> None:
        """Release decoder handles held by this dataset worker."""
        self._video_cache.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_video_cache"] = _VideoCache()
        return state

    def __del__(self) -> None:
        cache = getattr(self, "_video_cache", None)
        if cache is not None:
            cache.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = int(self._row_episode[index])
        local_frame = int(self._row_local_frame[index])
        sample: dict[str, Any] = {
            "action": self._row_action[index],
            "reference_action": self._row_reference_action[index],
            "return": np.asarray(self._returns[index], dtype=np.float32),
            "return_raw": np.asarray(self._raw_returns[index], dtype=np.float32),
            "episode_success": np.asarray(self._success_rows[index], dtype=np.int8),
            "episode_index": np.asarray(episode, dtype=np.int64),
            "frame_index": np.asarray(local_frame, dtype=np.int64),
            "task_index": np.asarray(self._row_task[index], dtype=np.int64),
            "task": self._row_instruction[index],
        }
        if self._has_state:
            sample["observation.state"] = self._row_state[index]
        for key in self.image_keys:
            sample[key] = self._video_cache.read(self._video_paths[(episode, key)], local_frame)
        return sample


def load_robodojo_dataset(config: Any) -> RoboDojoDataset:
    if not config.root:
        raise ValueError("WCM_DATASET_ROOT/config.data.root must point to a local RoboDojo dataset.")
    return RoboDojoDataset(
        config.root,
        getattr(config, "repo_id", None),
        os.environ.get("WCM_TASK_NAME"),
    )
