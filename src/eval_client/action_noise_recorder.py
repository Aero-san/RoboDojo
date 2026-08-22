"""Record policy initial-action noise and simulator-labelled outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ActionNoiseRecorder:
    """Collect one initial-noise tensor per policy inference/action chunk."""

    def __init__(self, root: str, run_id: str, task_name: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = run_id
        self.task_name = task_name
        self.raw_root = self.root / "raw" / task_name
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self._noise: dict[int, list[np.ndarray]] = {}
        self._steps: dict[int, list[int]] = {}

    def record(self, env_idx: int, initial_noise: Any, rollout_step: int) -> None:
        noise = np.asarray(initial_noise, dtype=np.float32)
        if noise.size == 0 or not np.isfinite(noise).all():
            raise ValueError("initial_noise_actions must be a non-empty finite tensor.")
        self._noise.setdefault(env_idx, []).append(noise.copy())
        self._steps.setdefault(env_idx, []).append(int(rollout_step))

    def finalize(
        self,
        env_idx: int,
        episode_index: int,
        *,
        success: bool,
        episode_seed: int,
        layout_id: int,
    ) -> Path | None:
        noise = self._noise.pop(env_idx, [])
        steps = self._steps.pop(env_idx, [])
        if not noise:
            return None
        shapes = {item.shape for item in noise}
        if len(shapes) != 1:
            raise ValueError(f"Initial-noise shape changed during an episode: {sorted(shapes)}")
        path = self.raw_root / f"{self.run_id}_episode_{episode_index:07d}.npz"
        np.savez_compressed(
            path,
            initial_noise_actions=np.stack(noise),
            rollout_steps=np.asarray(steps, dtype=np.int32),
            task_id=np.asarray(self.task_name),
            run_id=np.asarray(self.run_id),
            episode_index=np.asarray(episode_index, dtype=np.int64),
            episode_seed=np.asarray(episode_seed, dtype=np.int64),
            layout_id=np.asarray(layout_id, dtype=np.int64),
            success=np.asarray(success, dtype=np.bool_),
        )
        return path

    def abort(self, env_idx_list: list[int] | None = None) -> None:
        indices = set(self._noise) | set(self._steps) if env_idx_list is None else set(env_idx_list)
        for env_idx in indices:
            self._noise.pop(env_idx, None)
            self._steps.pop(env_idx, None)

    def close(self) -> None:
        self.abort()
