from types import SimpleNamespace

import jax
import numpy as np
import torch

from openpi.policies import policy as policy_module


class _FakeTorchFlowModel:
    config = SimpleNamespace(action_horizon=3, action_dim=2)

    def sample_noise(self, shape, device):
        return torch.arange(np.prod(shape), dtype=torch.float32, device=device).reshape(shape)

    def sample_actions(self, _device, _observation, *, noise):
        return noise


def test_infer_returns_parallel_normalized_and_transformed_samples():
    policy = object.__new__(policy_module.Policy)
    vars(policy).update(
        {
            "_input_transform": lambda value: value,
            "_output_transform": lambda value: {"actions": np.asarray(value["actions"]) * 2},
            "_is_pytorch_model": False,
            "_sample_actions_accepts_noise": True,
            "_sample_kwargs": {},
            "_rng": jax.random.key(0),
            "_model": SimpleNamespace(action_horizon=3, action_dim=2),
            "_sample_actions": lambda _rng, _observation, *, noise: noise,
        }
    )

    observation = {
        "image": {},
        "image_mask": {},
        "state": np.zeros(2, dtype=np.float32),
    }
    result = policy.infer(observation, num_samples=4, return_normalized_actions=True)

    assert result["actions"].shape == (4, 3, 2)
    assert result["normalized_actions"].shape == (4, 3, 2)
    assert result["initial_noise_actions"].shape == (4, 3, 2)
    np.testing.assert_allclose(result["actions"], result["normalized_actions"] * 2)
    np.testing.assert_allclose(result["initial_noise_actions"], result["normalized_actions"])


def test_pytorch_infer_returns_the_exact_noise_given_to_the_flow_sampler():
    model = _FakeTorchFlowModel()
    policy = object.__new__(policy_module.Policy)
    vars(policy).update(
        {
            "_input_transform": lambda value: value,
            "_output_transform": lambda value: value,
            "_is_pytorch_model": True,
            "_pytorch_device": "cpu",
            "_sample_actions_accepts_noise": True,
            "_sample_kwargs": {},
            "_model": model,
            "_sample_actions": model.sample_actions,
        }
    )

    observation = {
        "image": {},
        "image_mask": {},
        "state": np.zeros(2, dtype=np.float32),
    }
    result = policy.infer(observation)

    assert result["actions"].shape == (3, 2)
    assert result["initial_noise_actions"].shape == (3, 2)
    np.testing.assert_allclose(result["actions"], result["initial_noise_actions"])
