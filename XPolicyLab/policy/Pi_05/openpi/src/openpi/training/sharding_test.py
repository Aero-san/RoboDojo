import jax
import jax.numpy as jnp
import pytest

from openpi.training import sharding


def _large_shape_tree():
    shape = jax.ShapeDtypeStruct((2048, 2048), jnp.bfloat16)
    return {
        "params": {"weight": shape},
        "opt_state": {"weight": shape},
        "ema_params": {"weight": shape},
    }


@pytest.mark.skipif(jax.device_count() < 2, reason="requires multiple JAX devices")
def test_fsdp_sharding_strategies():
    mesh = sharding.make_mesh(jax.device_count())

    full_shard = sharding.fsdp_sharding(_large_shape_tree(), mesh, strategy="full_shard")
    shard_grad_op = sharding.fsdp_sharding(_large_shape_tree(), mesh, strategy="shard_grad_op")
    no_shard = sharding.fsdp_sharding(_large_shape_tree(), mesh, strategy="no_shard")

    sharded = jax.sharding.PartitionSpec(None, sharding.FSDP_AXIS)
    replicated = jax.sharding.PartitionSpec()
    assert full_shard["params"]["weight"].spec == sharded
    assert full_shard["opt_state"]["weight"].spec == sharded
    assert full_shard["ema_params"]["weight"].spec == sharded
    assert shard_grad_op["params"]["weight"].spec == replicated
    assert shard_grad_op["opt_state"]["weight"].spec == sharded
    assert shard_grad_op["ema_params"]["weight"].spec == replicated
    assert no_shard["params"]["weight"].spec == replicated
    assert no_shard["opt_state"]["weight"].spec == replicated
    assert no_shard["ema_params"]["weight"].spec == replicated


def test_fsdp_sharding_rejects_unknown_strategy():
    mesh = sharding.make_mesh(1)
    with pytest.raises(ValueError, match="Unknown sharding strategy"):
        sharding.fsdp_sharding(_large_shape_tree(), mesh, strategy="zero_4")


def test_make_mesh_rejects_zero_fsdp_devices():
    with pytest.raises(ValueError, match="must be positive"):
        sharding.make_mesh(0)


def test_put_tree_preserves_values_and_requested_memory_kind():
    mesh = sharding.make_mesh(1)
    device_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    host_sharding = device_sharding.with_memory_kind(
        sharding.resolve_memory_kind(cpu_offload=True)
    )
    values = {"momentum": jnp.arange(8, dtype=jnp.float32)}

    host_values = sharding.put_tree(values, {"momentum": host_sharding})
    device_values = sharding.put_tree(host_values, {"momentum": device_sharding})

    assert host_values["momentum"].sharding.memory_kind == host_sharding.memory_kind
    assert device_values["momentum"].sharding.memory_kind == device_sharding.memory_kind
    assert jnp.array_equal(device_values["momentum"], values["momentum"])

