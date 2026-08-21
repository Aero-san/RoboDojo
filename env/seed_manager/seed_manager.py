from collections.abc import Iterable, Mapping
from copy import deepcopy
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

        self.st_idx: int
        self.ed_idx: int
        self.type: str

        self._current_batch_seeds: List[int] | None = None

    def init_eval(
        self,
        completed_layout_ids: Iterable[int] | None = None,
        abandoned_layout_ids: Iterable[int] | None = None,
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
        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(
            int(s) for s in (abandoned_layout_ids or [])
        )
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            print(
                f"[SeedManager] init_eval resume filter: excluded={len(excluded)} "
                f"remaining={len(self.seed_list)}/{len(all_layout_ids)}"
            )
        else:
            self.seed_list = all_layout_ids
        if shard_count > 1:
            print(
                f"[SeedManager] layout shard={shard_index}/{shard_count} "
                f"layouts={len(self.seed_list)}/{len(matching_files)}"
            )
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Returns None when enough episodes have been successfully collected.
        """

        if self.idx >= self.ed_idx:
            return None
        if max_count is not None:
            batch_size = min(self.num_envs, max(0, int(max_count)))
            if batch_size == 0:
                return None
            batch = self.seed_list[self.idx : min(self.idx + batch_size, self.ed_idx)]
            self.idx += len(batch)
            self._current_batch_seeds = batch
            return batch
        if self.idx + self.num_envs > self.ed_idx:
            batch = self.seed_list[self.idx : self.ed_idx]
            result = deepcopy(batch)
            for _ in range(self.num_envs - len(result)):
                batch.append(self.seed_list[self.ed_idx - 1])  # pad with last seed if not enough remaining
        else:
            batch = self.seed_list[self.idx : self.idx + self.num_envs]
        self.idx += self.num_envs
        self._current_batch_seeds = batch
        return batch

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        seed_info = self.seed_info.get(seed)
        if seed_info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Scene layout file not found for seed {seed} at expected path {file_path}.")
        data = load_json(file_path)
        return data

    def eval_step(self):
        self._current_batch_seeds = None
