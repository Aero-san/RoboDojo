from types import MethodType, SimpleNamespace

import numpy as np

from XPolicyLab.policy.Pi_05.model import Model


class _FakePolicy:
    def __init__(self, actions, initial_noise):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.initial_noise = np.asarray(initial_noise, dtype=np.float32)

    def infer(self, _observation, **_kwargs):
        return {
            "actions": self.actions.copy(),
            "initial_noise_actions": self.initial_noise.copy(),
        }


def _model(*, posttrain=False):
    model = object.__new__(Model)
    actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    initial_noise = -np.arange(8, dtype=np.float32).reshape(4, 2)
    vars(model).update(
        {
            "observation_window": {
                "state": np.zeros((1, 2), dtype=np.float32),
                "images": {
                    "cam_high": np.zeros((1, 3, 2, 2), dtype=np.uint8),
                    "cam_left_wrist": np.zeros((1, 3, 2, 2), dtype=np.uint8),
                    "cam_right_wrist": np.zeros((1, 3, 2, 2), dtype=np.uint8),
                },
                "prompt": ["test"],
            },
            "_latest_env_idx_list": [0],
            "_latest_reference_action_list": [],
            "_latest_initial_noise_action_list": [],
            "policy": _FakePolicy(actions, initial_noise),
            "bcp": SimpleNamespace(enabled=False),
            "adaptive_action_chunker": None,
            "last_adaptive_chunking_results": {},
            "robot_action_dim_info": None,
            "posttrain": object() if posttrain else None,
        }
    )
    if posttrain:
        model._apply_posttrain = MethodType(
            lambda _self, _observation, reference: reference + 100.0,
            model,
        )
    return model, actions, initial_noise


def test_base_pi05_deployment_returns_initial_noise_with_actions():
    model, actions, initial_noise = _model()

    result = model.get_action()

    np.testing.assert_array_equal(result["actions"], actions)
    np.testing.assert_array_equal(result["initial_noise_actions"], initial_noise)
    assert "reference_actions" not in result


def test_posttrained_pi05_keeps_reference_noise_separate_from_modified_actions():
    model, reference_actions, initial_noise = _model(posttrain=True)

    result = model.get_action()

    np.testing.assert_array_equal(result["actions"], reference_actions + 100.0)
    np.testing.assert_array_equal(result["reference_actions"], reference_actions)
    np.testing.assert_array_equal(result["initial_noise_actions"], initial_noise)
