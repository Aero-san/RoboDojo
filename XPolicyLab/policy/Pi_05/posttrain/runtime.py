"""Runtime helpers for Pi0.5 WCM/RLToken post-training artifacts.

The model server remains an ordinary XPolicyLab adapter.  This module only
loads the small PyTorch reference-conditioned actor and converts its normalized action
space back to the physical robot action space used by RoboDojo.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rl_token import RLTokenActor, RLTokenConfig, RLTokenEncoderDecoder


def _as_stats(value: Any, *, name: str, dim: int, device: torch.device) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Post-training checkpoint is missing {name} statistics.")
    result = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
    if result.numel() != dim:
        raise ValueError(f"{name} has {result.numel()} values, expected action_dim={dim}.")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def load_posttrain_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Post-training artifact must be a dictionary: {path}")
    artifact_type = payload.get("artifact_type", payload.get("format", ""))
    if artifact_type not in {
        "robodojo_pi05_rltoken_v1",
        "robodojo_pi05_wcm_actor_v1",
        "robodojo_pi05_posttrain_v1",
    }:
        raise ValueError(
            f"Unsupported Pi0.5 post-training artifact {artifact_type!r}; "
            "use a checkpoint produced by scripts/posttrain/train_pi05_rltoken.py."
        )
    config_values = dict(payload.get("config", {}))
    config_fields = {field.name for field in fields(RLTokenConfig)}
    token_config = RLTokenConfig(
        **{key: value for key, value in config_values.items() if key in config_fields}
    )
    action_dim = int(config_values.get("action_dim", payload.get("action_dim", 0)))
    state_dim = int(config_values.get("state_dim", payload.get("state_dim", 0)))
    chunk_steps = int(config_values.get("chunk_steps", payload.get("chunk_steps", 1)))
    if action_dim < 1 or state_dim < 1 or chunk_steps < 1:
        raise ValueError("Post-training checkpoint has invalid action_dim/state_dim/chunk_steps.")
    encoder_state = payload.get("encoder")
    actor_state = payload.get("actor")
    if not isinstance(encoder_state, dict) or not isinstance(actor_state, dict):
        raise ValueError("Post-training checkpoint must contain 'encoder' and 'actor' state dicts.")
    encoder = RLTokenEncoderDecoder(state_dim, token_config).to(device)
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    actor = RLTokenActor(
        token_config.token_dim,
        state_dim,
        (chunk_steps, action_dim),
        token_config,
    ).to(device)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    action_mean = _as_stats(
        config_values.get("action_mean", payload.get("action_mean")),
        name="action_mean",
        dim=action_dim,
        device=device,
    )
    action_std = _as_stats(
        config_values.get("action_std", payload.get("action_std")),
        name="action_std",
        dim=action_dim,
        device=device,
    ).clamp_min(1.0e-6)
    return {
        "encoder": encoder,
        "actor": actor,
        "chunk_steps": chunk_steps,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "action_mean": action_mean,
        "action_std": action_std,
        "artifact_type": artifact_type,
    }


def normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (actions - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def unnormalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return actions * std.view(1, 1, -1) + mean.view(1, 1, -1)


def physical_action_chunk(
    encoder: RLTokenEncoderDecoder,
    actor: RLTokenActor,
    state: np.ndarray,
    reference: np.ndarray,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    *,
    deterministic: bool = True,
) -> np.ndarray:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=action_mean.device).reshape(1, -1)
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=action_mean.device).reshape(
        1, reference.shape[0], reference.shape[1]
    )
    reference_normalized = normalize_actions(reference_tensor, action_mean, action_std)
    with torch.no_grad():
        token, _ = encoder(state_tensor[:, None, :])
        normalized = actor.sample(
            token,
            state_tensor,
            reference_normalized,
            deterministic=deterministic,
        )
        physical = unnormalize_actions(normalized, action_mean, action_std)
    return physical[0].detach().cpu().numpy().astype(np.float32)
