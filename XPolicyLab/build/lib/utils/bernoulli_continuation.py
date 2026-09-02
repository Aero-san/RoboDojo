"""Bernoulli-Continuation Policy components.

The continuation head is deliberately policy-agnostic.  A policy adapter only
needs to provide visual-language tokens, the denoised action chunk, and the
final flow/denoising velocity for that chunk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BCPConfig:
    candidate_horizons: tuple[int, ...] = tuple(range(15, 51, 5))
    model_dim: int = 512
    num_layers: int = 2
    num_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.0
    delta_positive: float = 0.7
    delta_negative: float = 0.3
    clip_low: float = 0.2
    clip_high: float = 0.2

    def __post_init__(self) -> None:
        horizons = self.candidate_horizons
        if len(horizons) < 2 or any(isinstance(value, bool) or value < 1 for value in horizons):
            raise ValueError("candidate_horizons must contain at least two positive integers.")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("candidate_horizons must be strictly increasing.")
        if self.model_dim < 1 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads.")
        if self.num_layers < 1 or self.feedforward_dim < 1:
            raise ValueError("num_layers and feedforward_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not self.delta_positive > self.delta_negative > 0:
            raise ValueError("BCP requires delta_positive > delta_negative > 0.")
        if self.clip_low < 0 or self.clip_high < 0:
            raise ValueError("GRPO clipping bounds must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> BCPConfig:
        values = dict(values or {})
        if "candidate_horizons" in values:
            values["candidate_horizons"] = tuple(int(value) for value in values["candidate_horizons"])
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown BCP configuration keys: {unknown}")
        return cls(**values)


class BernoulliContinuationHead(nn.Module):
    """Paper Eq. (2)-(4): ordered continue-or-replan horizon policy."""

    def __init__(
        self,
        visual_dim: int,
        action_dim: int,
        velocity_dim: int,
        config: BCPConfig,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.velocity_dim = int(velocity_dim)
        self.config = config
        self.visual_projection = nn.Linear(self.visual_dim, config.model_dim)
        self.action_projection = nn.Linear(
            self.action_dim + self.velocity_dim,
            config.model_dim,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.continue_logits = nn.Linear(config.model_dim, len(config.candidate_horizons) - 1)
        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _position_encoding(length: int, width: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32, device=device)[:, None]
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32, device=device)
            * (-math.log(10_000.0) / width)
        )
        encoding = torch.zeros((length, width), dtype=torch.float32, device=device)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding

    def forward(
        self,
        visual_tokens: Tensor,
        actions: Tensor,
        velocities: Tensor,
        visual_mask: Tensor | None = None,
    ) -> Tensor:
        if visual_tokens.ndim != 3 or actions.ndim != 3 or velocities.ndim != 3:
            raise ValueError("BCP inputs must have shapes [batch, tokens/steps, features].")
        if actions.shape[:2] != velocities.shape[:2]:
            raise ValueError("BCP actions and velocities must share batch and horizon dimensions.")
        if visual_tokens.shape[0] != actions.shape[0]:
            raise ValueError("BCP visual and action features must share the batch dimension.")
        batch_size = actions.shape[0]
        visual = self.visual_projection(visual_tokens.float())
        action = self.action_projection(torch.cat((velocities, actions), dim=-1).float())
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, visual, action), dim=1)
        tokens = tokens + self._position_encoding(
            tokens.shape[1], tokens.shape[2], tokens.device
        )[None]
        padding_mask = None
        if visual_mask is not None:
            visual_mask = visual_mask.to(device=tokens.device, dtype=torch.bool)
            if visual_mask.shape != visual_tokens.shape[:2]:
                raise ValueError("visual_mask must have shape [batch, visual_tokens].")
            padding_mask = torch.cat(
                (
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
                    ~visual_mask,
                    torch.zeros(actions.shape[:2], dtype=torch.bool, device=tokens.device),
                ),
                dim=1,
            )
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.continue_logits(self.output_norm(encoded[:, 0]))

    @staticmethod
    def horizon_log_probs(continue_logits: Tensor) -> Tensor:
        """Return log pi(E=e^k|s) for every candidate horizon (paper Eq. 4)."""
        log_continue = torch.nn.functional.logsigmoid(continue_logits)
        log_stop = torch.nn.functional.logsigmoid(-continue_logits)
        prefix = torch.cat(
            (
                torch.zeros((*continue_logits.shape[:-1], 1), device=continue_logits.device),
                torch.cumsum(log_continue, dim=-1),
            ),
            dim=-1,
        )
        return torch.cat((prefix[..., :-1] + log_stop, prefix[..., -1:]), dim=-1)

    def distribution(
        self,
        visual_tokens: Tensor,
        actions: Tensor,
        velocities: Tensor,
        visual_mask: Tensor | None = None,
    ) -> torch.distributions.Categorical:
        logits = self.forward(visual_tokens, actions, velocities, visual_mask)
        return torch.distributions.Categorical(logits=self.horizon_log_probs(logits))

    def select(
        self,
        visual_tokens: Tensor,
        actions: Tensor,
        velocities: Tensor,
        visual_mask: Tensor | None = None,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(visual_tokens, actions, velocities, visual_mask)
        indices = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(indices)
        candidates = torch.as_tensor(
            self.config.candidate_horizons,
            device=indices.device,
            dtype=torch.long,
        )
        return candidates[indices], indices, log_probs


def replanning_efficiency_reward(
    success: Tensor,
    calls: Tensor,
    reference_calls: Tensor,
    *,
    delta_positive: float = 0.7,
    delta_negative: float = 0.3,
) -> Tensor:
    """Paper Eq. (7)-(8), evaluated for adaptive trajectories."""
    if torch.any(calls <= 0) or torch.any(reference_calls <= 0):
        raise ValueError("VLA call counts must be positive.")
    efficiency = torch.tanh(torch.log(reference_calls.float() / calls.float()))
    adjustment = delta_positive * efficiency.clamp_min(0) + delta_negative * efficiency.clamp_max(0)
    return success.float() * (1.0 + adjustment)


def normalized_group_advantages(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """Paper Eq. (6), including the fixed-horizon reference reward."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A BCP group must contain at least two trajectory rewards.")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def clipped_grpo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    trajectory_indices: Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> Tensor:
    """Negative paper Eq. (5), normalized over all adaptive decisions."""
    if not (new_log_probs.shape == old_log_probs.shape == trajectory_indices.shape):
        raise ValueError("GRPO decision tensors must have identical shapes.")
    decision_advantages = advantages[trajectory_indices]
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * decision_advantages
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high) * decision_advantages
    return -torch.minimum(unclipped, clipped).mean()


def save_bcp_checkpoint(
    path: str | Path,
    head: BernoulliContinuationHead,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "config": asdict(head.config),
        "visual_dim": head.visual_dim,
        "action_dim": head.action_dim,
        "velocity_dim": head.velocity_dim,
        "state_dict": head.state_dict(),
        "step": int(step),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_bcp_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[BernoulliContinuationHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported BCP checkpoint: {path}")
    config = BCPConfig.from_mapping(payload["config"])
    head = BernoulliContinuationHead(
        payload["visual_dim"],
        payload["action_dim"],
        payload["velocity_dim"],
        config,
    ).to(device)
    head.load_state_dict(payload["state_dict"])
    return head, payload


def validate_candidate_horizons(horizons: Sequence[int], action_horizon: int) -> None:
    if horizons[-1] > action_horizon:
        raise ValueError(
            f"BCP candidate horizon {horizons[-1]} exceeds Pi0.5 action horizon {action_horizon}."
        )
