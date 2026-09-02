import dataclasses

import flax.nnx as nnx
import jax
import pytest

from openpi.training import config as _config


def _resolved(mode: _config.FinetuneMode) -> _config.TrainConfig:
    config = dataclasses.replace(
        _config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0"),
        finetune_mode=mode,
    )
    return _config.resolve_finetune_config(config)


def _trainable_paths(config: _config.TrainConfig) -> set[str]:
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    state = nnx.state(model, config.trainable_filter).flat_state()
    return {"/".join(map(str, path)) for path in state}


@pytest.mark.parametrize(
    ("mode", "paligemma_variant", "action_expert_variant"),
    [
        ("full", "gemma_2b", "gemma_300m"),
        ("action_expert", "gemma_2b", "gemma_300m"),
        ("action_expert_lora", "gemma_2b", "gemma_300m_lora"),
        ("paligemma_lora", "gemma_2b_lora", "gemma_300m"),
        ("all_lora", "gemma_2b_lora", "gemma_300m_lora"),
    ],
)
def test_finetune_mode_model_variants(mode, paligemma_variant, action_expert_variant):
    config = _resolved(mode)
    assert config.model.paligemma_variant == paligemma_variant
    assert config.model.action_expert_variant == action_expert_variant


def test_action_expert_trainable_paths():
    paths = _trainable_paths(_resolved("action_expert"))
    assert paths
    assert any("_1/" in path for path in paths)
    assert any("action_in_proj" in path for path in paths)
    assert any("action_out_proj" in path for path in paths)
    assert all("img/" not in path for path in paths)
    assert all("_1/" in path or "proj" in path or "time_mlp" in path for path in paths)


def test_action_expert_lora_trainable_paths():
    paths = _trainable_paths(_resolved("action_expert_lora"))
    assert any("_1/" in path and "lora" in path for path in paths)
    assert any("action_in_proj" in path for path in paths)
    assert all("img/" not in path for path in paths)
    assert all("lora" in path or "proj" in path or "time_mlp" in path for path in paths)


def test_paligemma_lora_trainable_paths():
    paths = _trainable_paths(_resolved("paligemma_lora"))
    assert paths
    assert all("lora" in path for path in paths)
    assert all("_1/" not in path for path in paths)


def test_all_lora_trainable_paths():
    paths = _trainable_paths(_resolved("all_lora"))
    assert paths
    assert all("lora" in path for path in paths)
    assert any("_1/" in path for path in paths)
    assert any("_1/" not in path for path in paths)


def test_full_mode_trains_non_expert_parameters():
    paths = _trainable_paths(_resolved("full"))
    assert any("img/" in path for path in paths)
    assert any("llm/layers" in path for path in paths)
    assert any("_1/" in path for path in paths)
