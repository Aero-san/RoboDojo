from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._sample_actions_accepts_noise = "noise" in inspect.signature(model.sample_actions).parameters

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_with_bcp_features = None
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            bcp_sampler = getattr(model, "sample_actions_with_bcp_features", None)
            self._sample_actions_with_bcp_features = (
                nnx_utils.module_jit(bcp_sampler) if callable(bcp_sampler) else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_samples: int = 1,
        return_normalized_actions: bool = False,
        return_bcp_features: bool = False,
    ) -> dict:  # type: ignore[misc]
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        if return_bcp_features and num_samples != 1:
            raise ValueError("BCP feature extraction requires num_samples=1.")
        if return_bcp_features and self._is_pytorch_model:
            raise NotImplementedError("BCP feature extraction currently supports JAX Pi0.5 checkpoints.")
        if return_bcp_features and self._sample_actions_with_bcp_features is None:
            raise NotImplementedError("This OpenPI model does not expose BCP features.")
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        is_batched = False
        if "state" in inputs:
            is_batched = np.asarray(inputs["state"]).ndim > 1
        elif images := inputs.get("image"):
            first_image = next(iter(images.values()))
            is_batched = np.asarray(first_image).ndim > 3

        original_batch_size = int(np.asarray(inputs["state"]).shape[0]) if is_batched else 1

        if not self._is_pytorch_model:
            if not is_batched:
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            else:
                inputs = jax.tree.map(jnp.asarray, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            if not is_batched:
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs
                )
            else:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        if self._is_pytorch_model and num_samples > 1:
            inputs = jax.tree.map(lambda x: torch.repeat_interleave(x, num_samples, dim=0), inputs)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        initial_noise = None
        if noise is not None and not self._sample_actions_accepts_noise:
            raise TypeError(f"{type(self._model).__name__}.sample_actions does not accept noise.")
        if num_samples > 1 and not self._sample_actions_accepts_noise:
            raise TypeError(f"{type(self._model).__name__}.sample_actions does not support multiple noise samples.")
        if noise is not None:
            initial_noise = (
                torch.as_tensor(noise, device=self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )
            if not is_batched:
                expected_unbatched_ndim = 3 if num_samples > 1 else 2
                if initial_noise.ndim == expected_unbatched_ndim:
                    initial_noise = initial_noise[None, ...]
        elif self._sample_actions_accepts_noise and self._is_pytorch_model:
            effective_batch_size = original_batch_size * num_samples
            initial_noise = self._model.sample_noise(
                (
                    effective_batch_size,
                    self._model.config.action_horizon,
                    self._model.config.action_dim,
                ),
                self._pytorch_device,
            )
        elif self._sample_actions_accepts_noise:
            noise_shape = (
                (
                    original_batch_size,
                    num_samples,
                    self._model.action_horizon,
                    self._model.action_dim,
                )
                if num_samples > 1
                else (
                    original_batch_size,
                    self._model.action_horizon,
                    self._model.action_dim,
                )
            )
            initial_noise = jax.random.normal(
                sample_rng_or_pytorch_device,
                noise_shape,
            )
        if self._is_pytorch_model and num_samples > 1 and initial_noise is not None and initial_noise.ndim == 4:
            initial_noise = initial_noise.reshape(
                original_batch_size * num_samples,
                *initial_noise.shape[2:],
            )
        if initial_noise is not None:
            sample_kwargs["noise"] = initial_noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        bcp_features = None
        if return_bcp_features:
            sampled = self._sample_actions_with_bcp_features(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
            sampled_actions, visual_tokens, visual_mask, velocities = sampled
            bcp_features = {
                "visual_tokens": visual_tokens,
                "visual_mask": visual_mask,
                "velocities": velocities,
            }
        else:
            sampled_actions = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
        outputs = {"state": inputs["state"], "actions": sampled_actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model and num_samples > 1:
            outputs["actions"] = outputs["actions"].reshape(
                original_batch_size,
                num_samples,
                *outputs["actions"].shape[1:],
            )
            initial_noise = initial_noise.reshape(
                original_batch_size,
                num_samples,
                *initial_noise.shape[1:],
            )
        normalized_actions = outputs["actions"]
        if self._is_pytorch_model:
            if not is_batched:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
                normalized_actions = np.asarray(normalized_actions[0, ...].detach().cpu())
                initial_noise_actions = (
                    np.asarray(initial_noise[0, ...].detach().cpu()) if initial_noise is not None else None
                )
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
                normalized_actions = np.asarray(normalized_actions.detach().cpu())
                initial_noise_actions = np.asarray(initial_noise.detach().cpu()) if initial_noise is not None else None
        elif not is_batched:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            normalized_actions = np.asarray(normalized_actions[0, ...])
            initial_noise_actions = np.asarray(initial_noise[0, ...]) if initial_noise is not None else None
        else:
            outputs = jax.tree.map(np.asarray, outputs)
            normalized_actions = np.asarray(normalized_actions)
            initial_noise_actions = np.asarray(initial_noise) if initial_noise is not None else None

        outputs = self._output_transform(outputs)
        if initial_noise_actions is not None:
            outputs["initial_noise_actions"] = initial_noise_actions
        if return_normalized_actions:
            outputs["normalized_actions"] = normalized_actions
        if bcp_features is not None:
            if not is_batched:
                bcp_features = jax.tree.map(lambda x: np.asarray(x[0, ...]), bcp_features)
            else:
                bcp_features = jax.tree.map(np.asarray, bcp_features)
            outputs["bcp_features"] = bcp_features
            outputs["normalized_actions"] = normalized_actions
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
