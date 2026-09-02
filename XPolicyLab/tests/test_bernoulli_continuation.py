from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.policy.Pi_05.bcp import Pi05BCPRuntime
from XPolicyLab.utils.bernoulli_continuation import (
    BCPConfig,
    BernoulliContinuationHead,
    clipped_grpo_loss,
    normalized_group_advantages,
    replanning_efficiency_reward,
)


def _head() -> BernoulliContinuationHead:
    return BernoulliContinuationHead(
        visual_dim=6,
        action_dim=3,
        velocity_dim=3,
        config=BCPConfig(
            candidate_horizons=(2, 4, 6),
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
        ),
    )


def test_horizon_log_probs_follow_bernoulli_factorization() -> None:
    probabilities = torch.tensor([[0.25, 0.6]])
    logits = torch.logit(probabilities)
    actual = _head().horizon_log_probs(logits).exp()
    expected = torch.tensor([[0.75, 0.25 * 0.4, 0.25 * 0.6]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))


def test_head_returns_one_probability_per_candidate_horizon() -> None:
    head = _head()
    distribution = head.distribution(
        torch.randn(2, 5, 6),
        torch.randn(2, 6, 3),
        torch.randn(2, 6, 3),
        torch.tensor([[True, True, False, False, False], [True] * 5]),
    )
    assert distribution.probs.shape == (2, 3)
    torch.testing.assert_close(distribution.probs.sum(dim=-1), torch.ones(2))


def test_replanning_efficiency_reward_is_asymmetric_and_success_gated() -> None:
    rewards = replanning_efficiency_reward(
        torch.tensor([1, 1, 0]),
        torch.tensor([5, 20, 1]),
        torch.tensor([10, 10, 10]),
    )
    efficiency = math.tanh(math.log(2.0))
    torch.testing.assert_close(rewards[0], torch.tensor(1.0 + 0.7 * efficiency))
    torch.testing.assert_close(rewards[1], torch.tensor(1.0 - 0.3 * efficiency))
    assert rewards[2].item() == 0.0


def test_group_advantages_include_reference_and_grpo_backpropagates() -> None:
    rewards = torch.tensor([1.2, 0.0, 1.0])
    advantages = normalized_group_advantages(rewards)
    torch.testing.assert_close(advantages.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    new_log_probs = torch.tensor([-0.4, -0.8], requires_grad=True)
    loss = clipped_grpo_loss(
        new_log_probs,
        torch.tensor([-0.5, -0.7]),
        advantages[:2],
        torch.tensor([0, 1]),
        clip_low=0.2,
        clip_high=0.2,
    )
    loss.backward()
    assert new_log_probs.grad is not None
    assert torch.isfinite(new_log_probs.grad).all()


def _runtime_config(tmp_path: Path) -> dict:
    return {
        "enabled": True,
        "initialize_checkpoint": str(tmp_path / "initial.pt"),
        "rollout_dir": str(tmp_path / "rollouts"),
        "group_id": "test-group",
        "deterministic": False,
        "candidate_horizons": [2, 4, 6],
        "model_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "feedforward_dim": 16,
    }


def test_pi05_runtime_initializes_selects_and_records(tmp_path: Path) -> None:
    runtime = Pi05BCPRuntime(_runtime_config(tmp_path), tmp_path)
    prediction = {
        "normalized_actions": np.zeros((6, 3), dtype=np.float32),
        "bcp_features": {
            "visual_tokens": np.zeros((5, 6), dtype=np.float32),
            "visual_mask": np.ones(5, dtype=bool),
            "velocities": np.zeros((6, 3), dtype=np.float32),
        },
    }
    assert runtime.select_horizon(prediction, env_idx=3) in {2, 4, 6}
    assert (tmp_path / "initial.pt").is_file()
    result = runtime.on_trial_end(
        {
            "task_name": "stack_bowls",
            "episodes": [{"env_idx": 3, "layout_id": 9, "success": True, "steps": 4}],
        }
    )
    assert result == {"written": 1}
    paths = list((tmp_path / "rollouts").glob("*.pt"))
    assert len(paths) == 1
    record = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert record["calls"] == 1
    assert record["instance_id"] == "9"
    assert len(record["decisions"]) == 1
