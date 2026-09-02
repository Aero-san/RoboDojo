import numpy as np
import pytest

from XPolicyLab.utils.adaptive_action_chunking import (
    ActionLayout,
    AdaptiveActionChunker,
    AdaptiveActionChunkingConfig,
    absolute_actions_to_offsets,
)


def test_entropy_elbow_and_movement_floor_select_longer_prefix():
    config = AdaptiveActionChunkingConfig(
        enabled=True,
        num_samples=4,
        min_chunk_size=2,
        movement_threshold=1.5,
    )
    chunker = AdaptiveActionChunker(config, ActionLayout(((0,),), (1,)))

    normalized = np.zeros((4, 5, 2), dtype=np.float32)
    normalized[:, 3, 0] = [-3.0, -1.0, 1.0, 3.0]
    normalized[:, 4, 0] = [-3.0, -1.0, 1.0, 3.0]
    normalized[:, 3, 1] = [-1.0, -1.0, 1.0, 1.0]
    actions = np.zeros_like(normalized)
    actions[0, :, 0] = 0.6

    result = chunker.select(normalized, actions)

    assert result.entropy_chunk_size == 3
    assert result.movement_floor == 3
    assert result.chunk_size == 3
    assert result.step_entropy[3] > result.step_entropy[2]


def test_movement_floor_uses_gripper_switch():
    config = AdaptiveActionChunkingConfig(
        enabled=True,
        num_samples=2,
        min_chunk_size=1,
        movement_threshold=0.5,
    )
    chunker = AdaptiveActionChunker(config, ActionLayout((), (0,)))
    samples = np.full((2, 4, 1), -1.0, dtype=np.float32)
    samples[0, 2:, 0] = 1.0

    result = chunker.select(samples, samples)

    assert result.movement_floor == 3
    assert result.chunk_size == 3


def test_discrete_entropy_uses_executable_gripper_values():
    config = AdaptiveActionChunkingConfig(
        enabled=True,
        num_samples=4,
        min_chunk_size=1,
        movement_threshold=0.0,
        discrete_threshold=0.5,
    )
    chunker = AdaptiveActionChunker(config, ActionLayout((), (0,)))
    normalized = np.ones((4, 2, 1), dtype=np.float32)
    actions = np.zeros_like(normalized)
    actions[:, 1, 0] = [0.0, 0.0, 1.0, 1.0]

    result = chunker.select(normalized, actions)

    assert result.step_entropy[0] == pytest.approx(0.0)
    assert result.step_entropy[1] == pytest.approx(np.log(2.0))


def test_select_actions_uses_configured_candidate_and_horizon_cap():
    config = AdaptiveActionChunkingConfig(
        enabled=True,
        num_samples=3,
        min_chunk_size=2,
        max_chunk_size=3,
        movement_threshold=100.0,
        candidate_index=1,
    )
    chunker = AdaptiveActionChunker(config, ActionLayout(((0,),)))
    samples = np.zeros((3, 6, 1), dtype=np.float32)
    samples[1, :, 0] = np.arange(6)

    selected, result = chunker.select_actions(samples, samples)

    assert result.chunk_size == 3
    np.testing.assert_array_equal(selected[:, 0], [0.0, 1.0, 2.0])


def test_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown adaptive_action_chunking options"):
        AdaptiveActionChunkingConfig.from_mapping({"not_a_parameter": 1})


def test_absolute_targets_are_converted_to_offsets_without_changing_gripper():
    layout = ActionLayout(((0, 1),), (2,))
    actions = np.array([[[[2.0, 4.0, 0.0], [3.0, 7.0, 1.0]]]], dtype=np.float32).reshape(1, 2, 3)

    offsets = absolute_actions_to_offsets(actions, np.array([1.0, 1.0, 0.0]), layout)

    np.testing.assert_array_equal(offsets[0, :, :2], [[1.0, 3.0], [1.0, 3.0]])
    np.testing.assert_array_equal(offsets[0, :, 2], [0.0, 1.0])
