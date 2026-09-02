import jax

from openpi.models import pi0_config
from openpi.training import config
from openpi.training import sharding

from . import train


def test_cpu_offload_sharding_accepts_typed_train_state():
    train_config = config.TrainConfig(
        name="cpu_offload_sharding_test",
        exp_name="test",
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=8,
            action_horizon=4,
            max_token_len=16,
        ),
        batch_size=jax.device_count(),
        fsdp_devices=1,
        cpu_offload=True,
    )

    state_shape, runtime_sharding, device_opt_sharding = train.init_train_state(
        train_config,
        jax.random.key(0),
        sharding.make_mesh(1),
        resume=True,
    )

    assert jax.tree.structure(runtime_sharding) == jax.tree.structure(state_shape)
    assert jax.tree.structure(device_opt_sharding) == jax.tree.structure(state_shape.opt_state)
