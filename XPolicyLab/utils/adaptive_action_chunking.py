"""Inference-time Adaptive Action Chunking (AAC).

This module implements Algorithm 1 and Appendix A of Liang et al.,
"Adaptive Action Chunking at Inference-time for Vision-Language-Action
Models" (CVPR 2026).  It is policy-agnostic: callers provide sampled action
chunks in normalized and execution spaces plus the continuous/discrete action
layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionLayout:
    """Partition action dimensions into continuous groups and discrete values."""

    continuous_groups: tuple[tuple[int, ...], ...]
    discrete_indices: tuple[int, ...] = ()

    def validate(self, action_dim: int) -> None:
        indices = [index for group in self.continuous_groups for index in group]
        indices.extend(self.discrete_indices)
        if not indices:
            raise ValueError("AAC action layout must contain at least one action dimension.")
        if any(index < 0 or index >= action_dim for index in indices):
            raise ValueError(
                f"AAC action layout contains an index outside action_dim={action_dim}: {indices}"
            )
        if len(indices) != len(set(indices)):
            raise ValueError("AAC action layout dimensions must not overlap.")
        if any(not group for group in self.continuous_groups):
            raise ValueError("AAC continuous action groups must not be empty.")


@dataclass(frozen=True)
class AdaptiveActionChunkingConfig:
    """AAC inference parameters.

    Defaults reproduce the paper/reference implementation: 20 samples,
    Gaussian/Bernoulli entropy, a minimum chunk of 2, and minimum movement
    energy alpha=3.  Continuous entropy is computed in normalized action
    space; discrete entropy and the movement floor use executable actions.
    """

    enabled: bool = False
    num_samples: int = 20
    min_chunk_size: int = 2
    max_chunk_size: int | None = None
    movement_threshold: float = 3.0
    covariance_regularization: float = 1e-6
    discrete_threshold: float = 0.0
    magnitude_discrete_threshold: float = 0.0
    continuous_entropy_weight: float = 1.0
    discrete_entropy_weight: float = 1.0
    continuous_magnitude_weight: float = 1.0
    discrete_magnitude_weight: float = 1.0
    candidate_index: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> AdaptiveActionChunkingConfig:
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown adaptive_action_chunking options: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("AAC enabled must be a boolean.")
        if self.num_samples < 2:
            raise ValueError("AAC num_samples must be at least 2 to estimate action entropy.")
        if self.min_chunk_size < 1:
            raise ValueError("AAC min_chunk_size must be at least 1.")
        if self.max_chunk_size is not None and self.max_chunk_size < self.min_chunk_size:
            raise ValueError("AAC max_chunk_size must be no smaller than min_chunk_size.")
        if self.movement_threshold < 0:
            raise ValueError("AAC movement_threshold must be non-negative.")
        if self.covariance_regularization <= 0:
            raise ValueError("AAC covariance_regularization must be positive.")
        weights = (
            self.continuous_entropy_weight,
            self.discrete_entropy_weight,
            self.continuous_magnitude_weight,
            self.discrete_magnitude_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("AAC entropy and magnitude weights must be non-negative.")
        if self.candidate_index < 0 or self.candidate_index >= self.num_samples:
            raise ValueError("AAC candidate_index must select one of num_samples candidates.")


@dataclass(frozen=True)
class AdaptiveChunkingResult:
    chunk_size: int
    entropy_chunk_size: int
    movement_floor: int
    step_entropy: np.ndarray
    prefix_mean_entropy: np.ndarray
    prefix_magnitude: np.ndarray


class AdaptiveActionChunker:
    """Select an execution prefix from parallel action-chunk samples."""

    def __init__(self, config: AdaptiveActionChunkingConfig, layout: ActionLayout):
        config.validate()
        self.config = config
        self.layout = layout

    def select(
        self,
        normalized_samples: np.ndarray,
        action_samples: np.ndarray,
        magnitude_samples: np.ndarray | None = None,
    ) -> AdaptiveChunkingResult:
        normalized = _validate_samples("normalized_samples", normalized_samples)
        actions = _validate_samples("action_samples", action_samples)
        if normalized.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "normalized_samples and action_samples must have identical sample/horizon dimensions, "
                f"got {normalized.shape[:2]} and {actions.shape[:2]}."
            )
        magnitude_actions = (
            actions
            if magnitude_samples is None
            else _validate_samples("magnitude_samples", magnitude_samples)
        )
        if magnitude_actions.shape != actions.shape:
            raise ValueError(
                "magnitude_samples and action_samples must have identical shapes, "
                f"got {magnitude_actions.shape} and {actions.shape}."
            )
        if normalized.shape[0] != self.config.num_samples:
            raise ValueError(
                f"AAC expected {self.config.num_samples} samples, got {normalized.shape[0]}."
            )
        if self.config.candidate_index >= actions.shape[0]:
            raise ValueError("AAC candidate_index is outside the sampled action batch.")

        horizon = normalized.shape[1]
        if self.config.max_chunk_size is not None:
            horizon = min(horizon, self.config.max_chunk_size)
        if horizon < self.config.min_chunk_size:
            raise ValueError(
                f"AAC action horizon {horizon} is smaller than min_chunk_size={self.config.min_chunk_size}."
            )

        self.layout.validate(min(normalized.shape[2], actions.shape[2]))
        normalized = normalized[:, :horizon]
        selected_magnitude_actions = magnitude_actions[self.config.candidate_index, :horizon]

        step_entropy = self._step_entropy(normalized, actions[:, :horizon])
        prefix_mean_entropy = np.cumsum(step_entropy) / np.arange(1, horizon + 1)
        # Equation (5): h indexes the left prefix in E_bar[h+1] - E_bar[h].
        entropy_chunk_size = (
            int(np.argmax(np.diff(prefix_mean_entropy))) + 1 if horizon > 1 else 1
        )
        entropy_chunk_size = max(entropy_chunk_size, self.config.min_chunk_size)

        prefix_magnitude = self._prefix_magnitude(selected_magnitude_actions)
        above_threshold = np.flatnonzero(prefix_magnitude > self.config.movement_threshold)
        movement_floor = int(above_threshold[0]) + 1 if above_threshold.size else horizon
        movement_floor = max(movement_floor, self.config.min_chunk_size)

        chunk_size = min(horizon, max(entropy_chunk_size, movement_floor))
        return AdaptiveChunkingResult(
            chunk_size=chunk_size,
            entropy_chunk_size=entropy_chunk_size,
            movement_floor=movement_floor,
            step_entropy=step_entropy,
            prefix_mean_entropy=prefix_mean_entropy,
            prefix_magnitude=prefix_magnitude,
        )

    def select_actions(
        self,
        normalized_samples: np.ndarray,
        action_samples: np.ndarray,
        magnitude_samples: np.ndarray | None = None,
    ) -> tuple[np.ndarray, AdaptiveChunkingResult]:
        result = self.select(normalized_samples, action_samples, magnitude_samples)
        actions = np.asarray(action_samples)[self.config.candidate_index, : result.chunk_size]
        return actions, result

    def _step_entropy(
        self,
        continuous_samples: np.ndarray,
        discrete_samples: np.ndarray,
    ) -> np.ndarray:
        horizon = continuous_samples.shape[1]
        total = np.zeros(horizon, dtype=np.float64)
        for timestep in range(horizon):
            for group in self.layout.continuous_groups:
                total[timestep] += self.config.continuous_entropy_weight * _gaussian_entropy(
                    continuous_samples[:, timestep, group],
                    self.config.covariance_regularization,
                )
            for index in self.layout.discrete_indices:
                binary = (
                    discrete_samples[:, timestep, index] >= self.config.discrete_threshold
                )
                total[timestep] += self.config.discrete_entropy_weight * _bernoulli_entropy(binary)
        return total

    def _prefix_magnitude(self, actions: np.ndarray) -> np.ndarray:
        horizon = actions.shape[0]
        magnitude = np.zeros(horizon, dtype=np.float64)
        for prefix_index in range(horizon):
            prefix = actions[: prefix_index + 1]
            continuous = sum(
                np.linalg.norm(np.sum(prefix[:, group], axis=0))
                for group in self.layout.continuous_groups
            )
            discrete = sum(
                bool(
                    np.any(
                        np.diff(
                            prefix[:, index] >= self.config.magnitude_discrete_threshold,
                        )
                    )
                )
                for index in self.layout.discrete_indices
            )
            magnitude[prefix_index] = (
                self.config.continuous_magnitude_weight * continuous
                + self.config.discrete_magnitude_weight * discrete
            )
        return magnitude


def absolute_actions_to_offsets(
    actions: np.ndarray,
    current_action: np.ndarray,
    layout: ActionLayout,
) -> np.ndarray:
    """Convert absolute continuous targets to Appendix A action offsets.

    Discrete values stay absolute so gripper state switches remain detectable.
    """
    action_chunks = _validate_samples("actions", actions)
    current = np.asarray(current_action, dtype=np.float64).reshape(-1)
    if current.shape[0] < action_chunks.shape[2]:
        raise ValueError(
            f"current_action has {current.shape[0]} dimensions, expected {action_chunks.shape[2]}."
        )
    layout.validate(action_chunks.shape[2])
    offsets = np.array(action_chunks, copy=True)
    for group in layout.continuous_groups:
        indices = list(group)
        previous = np.concatenate(
            (
                np.broadcast_to(current[indices], (action_chunks.shape[0], 1, len(indices))),
                action_chunks[:, :-1, indices],
            ),
            axis=1,
        )
        offsets[:, :, indices] = action_chunks[:, :, indices] - previous
    return offsets


def _validate_samples(name: str, samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [samples, horizon, action_dim], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _gaussian_entropy(samples: np.ndarray, regularization: float) -> float:
    sample_count, dimensions = samples.shape
    if sample_count <= 1:
        return 0.0
    covariance = np.atleast_2d(np.cov(samples, rowvar=False))
    covariance += regularization * np.eye(dimensions)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("AAC covariance is not positive definite after regularization.")
    return float(0.5 * (dimensions * np.log(2 * np.pi * np.e) + log_determinant))


def _bernoulli_entropy(samples: Sequence[bool]) -> float:
    probability = float(np.mean(samples))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-probability * np.log(probability) - (1 - probability) * np.log(1 - probability))
