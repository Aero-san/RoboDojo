import numpy as np
import torch


def _to_host_vector(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(3)


def make_root_velocities(linear_velocity=None, angular_velocity=None):
    """Build a single (linear, angular) root-velocity batch for Isaac Sim."""
    velocities = np.zeros((1, 6), dtype=np.float32)
    if linear_velocity is not None:
        velocities[0, :3] = _to_host_vector(linear_velocity)
    if angular_velocity is not None:
        velocities[0, 3:] = _to_host_vector(angular_velocity)
    return velocities
