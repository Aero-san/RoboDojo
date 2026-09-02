"""Pi0.5 adapter for Bernoulli-Continuation Policy inference and rollouts."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from XPolicyLab.utils.bernoulli_continuation import (
    BCPConfig,
    BernoulliContinuationHead,
    load_bcp_checkpoint,
    save_bcp_checkpoint,
    validate_candidate_horizons,
)


def _boolean(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"Expected a boolean value, got {value!r}.")
    return normalized in {"1", "true", "yes"}


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


class Pi05BCPRuntime:
    def __init__(self, values: Mapping[str, Any], policy_dir: Path) -> None:
        raw = dict(values)
        environment = {
            "enabled": ("BCP_ENABLED", _boolean),
            "checkpoint": ("BCP_CHECKPOINT", str),
            "deterministic": ("BCP_DETERMINISTIC", _boolean),
            "reference": ("BCP_REFERENCE", _boolean),
            "rollout_dir": ("BCP_ROLLOUT_DIR", str),
            "group_id": ("BCP_GROUP_ID", str),
            "initialize_checkpoint": ("BCP_INITIALIZE_CHECKPOINT", str),
            "seed": ("BCP_SEED", int),
        }
        for key, (name, parser) in environment.items():
            if name in os.environ:
                raw[key] = parser(os.environ[name])

        self.enabled = _boolean(raw.pop("enabled", False))
        self.deterministic = _boolean(raw.pop("deterministic", True))
        self.reference = _boolean(raw.pop("reference", False))
        self.seed = int(raw.pop("seed", 0))
        self.checkpoint = self._path(raw.pop("checkpoint", None), policy_dir)
        self.initialize_checkpoint = self._path(
            raw.pop("initialize_checkpoint", None), policy_dir
        )
        self.rollout_dir = self._path(raw.pop("rollout_dir", None), policy_dir)
        self.group_id = str(raw.pop("group_id", "")).strip()
        self.config = BCPConfig.from_mapping(raw)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head: BernoulliContinuationHead | None = None
        self.checkpoint_payload: dict[str, Any] = {}
        self.decisions: dict[int, list[dict[str, Any]]] = {}
        self.calls: dict[int, int] = {}

        if not self.enabled:
            return
        torch.manual_seed(self.seed)
        if self.reference and self.checkpoint is not None:
            raise ValueError("BCP reference rollout does not use a continuation-head checkpoint.")
        if not self.reference and self.checkpoint is None and self.initialize_checkpoint is None:
            raise ValueError(
                "BCP requires BCP_CHECKPOINT, or BCP_INITIALIZE_CHECKPOINT for the first "
                "randomly initialized rollout policy."
            )
        if self.rollout_dir is not None and not self.group_id:
            raise ValueError("BCP rollout collection requires BCP_GROUP_ID.")
        if self.checkpoint is not None:
            self.head, self.checkpoint_payload = load_bcp_checkpoint(self.checkpoint, self.device)
            if self.head.config != self.config:
                raise ValueError(
                    "BCP deploy configuration does not match the continuation-head checkpoint."
                )
            self.head.eval()

    @staticmethod
    def _path(value: Any, policy_dir: Path) -> Path | None:
        if value in {None, "", "null", "none"}:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (policy_dir / path).resolve()

    def reset(self) -> None:
        self.decisions.clear()
        self.calls.clear()

    def _ensure_head(
        self,
        visual_tokens: np.ndarray,
        actions: np.ndarray,
        velocities: np.ndarray,
    ) -> BernoulliContinuationHead:
        if self.head is None:
            torch.manual_seed(self.seed)
            self.head = BernoulliContinuationHead(
                visual_tokens.shape[-1],
                actions.shape[-1],
                velocities.shape[-1],
                self.config,
            ).to(self.device)
            self.head.eval()
            assert self.initialize_checkpoint is not None
            self.initialize_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_bcp_checkpoint(
                self.initialize_checkpoint,
                self.head,
                metadata={"initialized_by": "Pi_05", "seed": self.seed},
            )
        expected = (self.head.visual_dim, self.head.action_dim, self.head.velocity_dim)
        actual = (visual_tokens.shape[-1], actions.shape[-1], velocities.shape[-1])
        if actual != expected:
            raise ValueError(f"BCP feature dimensions {actual} do not match checkpoint {expected}.")
        return self.head

    def select_horizon(
        self,
        prediction: Mapping[str, Any],
        env_idx: int,
    ) -> int:
        actions = np.asarray(prediction["normalized_actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Pi0.5 BCP actions must be [horizon, dim], got {actions.shape}.")
        validate_candidate_horizons(self.config.candidate_horizons, actions.shape[0])
        env_idx = int(env_idx)
        self.calls[env_idx] = self.calls.get(env_idx, 0) + 1
        if self.reference:
            return actions.shape[0]

        features = prediction.get("bcp_features")
        if not isinstance(features, Mapping):
            raise ValueError("Pi0.5 inference did not return BCP features.")
        visual = np.asarray(features["visual_tokens"], dtype=np.float32)
        visual_mask = np.asarray(features["visual_mask"], dtype=bool)
        velocities = np.asarray(features["velocities"], dtype=np.float32)
        head = self._ensure_head(visual, actions, velocities)
        with torch.no_grad():
            horizon, index, log_prob = head.select(
                torch.from_numpy(visual)[None].to(self.device),
                torch.from_numpy(actions)[None].to(self.device),
                torch.from_numpy(velocities)[None].to(self.device),
                torch.from_numpy(visual_mask)[None].to(self.device),
                deterministic=self.deterministic,
            )
        if self.rollout_dir is not None:
            self.decisions.setdefault(env_idx, []).append(
                {
                    "visual_tokens": torch.from_numpy(visual).to(torch.float16),
                    "visual_mask": torch.from_numpy(visual_mask),
                    "actions": torch.from_numpy(actions).to(torch.float16),
                    "velocities": torch.from_numpy(velocities).to(torch.float16),
                    "horizon_index": int(index.item()),
                    "old_log_prob": float(log_prob.item()),
                }
            )
        return int(horizon.item())

    def on_trial_end(self, payload: Mapping[str, Any]) -> dict[str, int] | None:
        if not self.enabled or self.rollout_dir is None:
            return None
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError("BCP trial_end payload must contain an episodes list.")
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for episode in episodes:
            env_idx = int(episode["env_idx"])
            instance_id = str(episode["layout_id"])
            decisions = self.decisions.get(env_idx, [])
            if not self.reference and not decisions:
                raise ValueError(f"Adaptive BCP rollout for env {env_idx} has no decisions.")
            record = {
                "format_version": 1,
                "group_id": self.group_id,
                "instance_id": instance_id,
                "task_name": payload.get("task_name"),
                "reference": self.reference,
                "success": bool(episode["success"]),
                "calls": int(self.calls.get(env_idx, 0)),
                "steps": int(episode.get("steps", 0)),
                "decisions": decisions,
                "checkpoint": str(self.checkpoint or self.initialize_checkpoint or ""),
            }
            filename = (
                f"{_safe_name(self.group_id)}__{_safe_name(payload.get('task_name'))}__"
                f"{_safe_name(instance_id)}__{'reference' if self.reference else 'adaptive'}__"
                f"{uuid.uuid4().hex}.pt"
            )
            torch.save(record, self.rollout_dir / filename)
            written += 1
        return {"written": written}
