from collections.abc import Iterable, Mapping
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from env.global_configs import ASSETS_PATH, BENCHMARK
from utils.load_file import *


class SeedManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.num_envs: int = int(self.config["num_envs"])

        # config fields used for directory layout
        self.task_name: str = str(self.config["task_name"])
        self.config_name: str = str(self.config["config_name"])

        self.type: str

        self._current_batch_seeds: List[int] | None = None
        self._layout_ids: List[int] = []
        self._layout_id_set: set[int] = set()
        self._layout_count = 0
        self._excluded_episode_seeds: set[int] = set()
        self._schedule_index = 0

    def init_eval(
        self,
        completed_episode_seeds: Iterable[int] | None = None,
        abandoned_episode_seeds: Iterable[int] | None = None,
    ):
        self.eval_seed = self.config.get("seed", 0)
        layout_dir = Path(ASSETS_PATH, "Eval_Layout", BENCHMARK, self.config_name, str(self.eval_seed))
        if not layout_dir.is_dir():
            config_root = layout_dir.parent
            available_seeds = sorted(
                (path.name for path in config_root.iterdir() if path.is_dir()),
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            ) if config_root.is_dir() else []
            available = ", ".join(available_seeds) if available_seeds else "none"
            raise FileNotFoundError(
                f"Evaluation layout seed {self.eval_seed!r} is unavailable for "
                f"config {self.config_name!r}: {layout_dir}. Available seeds: {available}. "
                "The eval seed selects a layout directory; it must not be derived from a "
                "training iteration or episode index."
            )
        pattern = re.compile(rf"{re.escape(self.task_name)}_\d+\.json")
        matching_files = sorted(
            [p for p in layout_dir.iterdir() if pattern.fullmatch(p.name)],
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )

        matching_files = [str(p) for p in matching_files]
        self.seed_info = {}
        for idx, file_path in enumerate(matching_files):
            self.seed_info[idx] = {"scene_layout": file_path}

        shard_index = int(self.config.get("layout_shard_index", 0))
        shard_count = int(self.config.get("layout_shard_count", 1))
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError(
                "Layout shard must satisfy shard_count >= 1 and "
                f"0 <= shard_index < shard_count, got {shard_index}/{shard_count}."
            )
        all_layout_ids = list(range(len(matching_files)))[shard_index::shard_count]
        layout_offset = int(self.config.get("layout_offset", 0))
        if layout_offset < 0:
            raise ValueError(f"Layout offset must be non-negative, got {layout_offset}.")
        if layout_offset:
            all_layout_ids = all_layout_ids[layout_offset:]
            print(
                f"[SeedManager] layout offset={layout_offset} "
                f"remaining={len(all_layout_ids)}/{len(matching_files)}"
            )
        if not all_layout_ids:
            raise ValueError(
                f"No evaluation layouts remain for task {self.task_name!r} after "
                f"applying shard {shard_index}/{shard_count} and offset {layout_offset}."
            )
        self._layout_ids = all_layout_ids
        self._layout_id_set = set(all_layout_ids)
        self._layout_count = len(matching_files)
        self._excluded_episode_seeds = set(int(s) for s in (completed_episode_seeds or [])) | set(
            int(s) for s in (abandoned_episode_seeds or [])
        )
        if self._excluded_episode_seeds:
            print(
                "[SeedManager] init_eval resume filter: "
                f"excluded_episode_seeds={len(self._excluded_episode_seeds)}"
            )
        if shard_count > 1:
            print(
                f"[SeedManager] layout shard={shard_index}/{shard_count} "
                f"layouts={len(self._layout_ids)}/{len(matching_files)}"
            )
        requested_episodes = int(self.config.get("eval_num", len(self._layout_ids)))
        if requested_episodes > len(self._layout_ids):
            print(
                f"[SeedManager] requested episodes={requested_episodes} exceed available "
                f"layouts={len(self._layout_ids)}; layouts will repeat deterministically."
            )

        self.type = "eval"
        self._schedule_index = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Episode seeds are unbounded and unique. The caller owns the requested
        episode count; saved layouts repeat deterministically when that count
        exceeds the number of layout files.
        """
        batch_size = self.num_envs if max_count is None else min(self.num_envs, max(0, int(max_count)))
        if batch_size == 0:
            return None
        batch: List[int] = []
        while len(batch) < batch_size:
            cycle, position = divmod(self._schedule_index, len(self._layout_ids))
            layout_id = self._layout_ids[position]
            episode_seed = cycle * self._layout_count + layout_id
            self._schedule_index += 1
            if episode_seed not in self._excluded_episode_seeds:
                batch.append(episode_seed)
        self._current_batch_seeds = batch
        return batch

    def get_layout_id(self, episode_seed: int) -> int:
        """Return the saved-layout id selected by a unique episode seed."""
        layout_id = int(episode_seed) % self._layout_count
        if layout_id not in self._layout_id_set:
            raise ValueError(
                f"Episode seed {episode_seed} maps to layout {layout_id}, which is outside "
                "the configured layout shard."
            )
        return layout_id

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        layout_id = self.get_layout_id(seed)
        seed_info = self.seed_info.get(layout_id)
        if seed_info is None:
            raise ValueError(f"Layout {layout_id} for episode seed {seed} was not found.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(
                f"Scene layout file not found for episode seed {seed}, layout {layout_id}, "
                f"at expected path {file_path}."
            )
        data = load_json(file_path)
        return data

    def eval_step(self):
        self._current_batch_seeds = None
