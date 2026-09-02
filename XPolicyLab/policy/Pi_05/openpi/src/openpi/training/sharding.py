import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)
SHARDING_STRATEGIES = ("full_shard", "shard_grad_op", "no_shard")


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    device_count = jax.device_count()
    if num_fsdp_devices < 1:
        raise ValueError(f"Number of FSDP devices must be positive, got {num_fsdp_devices}.")
    if device_count % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {device_count} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (device_count // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremely cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def resolve_memory_kind(*, cpu_offload: bool) -> str | None:
    """Return the JAX memory kind used for CPU-offloaded train state.

    JAX models host offload as a memory kind on the same device mesh. GPUs
    normally expose ``pinned_host`` memory; CPU-only test runs expose
    ``unpinned_host`` instead. The latter is already the CPU device's native
    memory and keeps the option testable without pretending that a CPU run has
    GPU offload support.
    """
    if not cpu_offload:
        return None

    devices = jax.devices()
    if not devices:
        raise RuntimeError("CPU offload requested, but JAX did not expose any devices.")

    memory_kinds = [
        {memory.kind for memory in device.addressable_memories()}
        for device in devices
    ]
    if all("pinned_host" in kinds for kinds in memory_kinds):
        return "pinned_host"
    if all(
        device.platform == "cpu" and "unpinned_host" in kinds
        for device, kinds in zip(devices, memory_kinds, strict=True)
    ):
        return "unpinned_host"

    available = sorted({kind for kinds in memory_kinds for kind in kinds})
    raise ValueError(
        "CPU offload requires every JAX device to expose pinned_host memory; "
        f"available memory kinds are {available}."
    )


def with_memory_kind(pytree, memory_kind: str):
    """Apply a JAX memory kind to every sharding leaf in a sharding pytree."""
    return jax.tree.map(
        lambda value: value.with_memory_kind(memory_kind)
        if isinstance(value, jax.sharding.Sharding)
        else value,
        pytree,
    )


def put_tree(pytree, sharding_pytree):
    """Move array leaves to ``sharding_pytree`` while preserving static leaves."""

    def _put(value, target):
        if isinstance(value, jax.Array | np.ndarray | np.generic):
            return jax.device_put(value, target)
        return value

    return jax.tree.map(_put, pytree, sharding_pytree)


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    strategy: str = "full_shard",
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        strategy: ``full_shard`` shards parameters, gradients and optimizer
          state; ``shard_grad_op`` keeps model/EMA parameters replicated while
          sharding optimizer state; ``no_shard`` replicates the whole state.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    if strategy not in SHARDING_STRATEGIES:
        raise ValueError(f"Unknown sharding strategy {strategy!r}; choose from {SHARDING_STRATEGIES}.")
    min_size_bytes = min_size_mbytes * 2**20

    def _named(spec):
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

    def _root_name(path):
        if not path:
            return None
        root = path[0]
        return getattr(root, "name", getattr(root, "key", None))

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # ``no_shard`` is useful when the model fits and communication is the
        # bottleneck. ``shard_grad_op`` mirrors FSDP SHARD_GRAD_OP: parameters
        # stay replicated between steps while optimizer state is sharded.
        if strategy == "no_shard" or mesh.shape[FSDP_AXIS] == 1:
            return _named(())
        if strategy == "shard_grad_op" and _root_name(kp) in {"params", "ema_params"}:
            return _named(())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return _named(())
        if len(array.shape) < 2:
            return _named(())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return _named(())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return _named(spec)

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return _named(())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
