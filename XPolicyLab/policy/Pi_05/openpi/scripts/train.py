import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

_METRICS_FILENAME = "training_metrics.json"


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_metric_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Invalid training metrics file: {path}")
    return records


def _metrics_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"schema_version": 1, "records": records},
        indent=2,
        sort_keys=True,
    ) + "\n"


def _write_metric_records(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    """Atomically write the run-level metric history."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(_metrics_json(records), encoding="utf-8")
    temporary_path.replace(path)


def _upsert_metric_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    step = record["step"]
    records[:] = [existing for existing in records if existing.get("step") != step]
    records.append(record)
    records.sort(key=lambda existing: existing["step"])


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any, Any | None]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # ``model.dtype`` controls forward-pass compute, but many Linen/NNX
        # parameter initializers default to float32. Cast the complete train
        # state explicitly so full fine-tuning does not retain a float32 copy
        # of Pi0.5 on every device.
        parameter_dtype = jnp.dtype(config.parameter_dtype)
        params = nnx_utils.state_map(
            params,
            nnx.Param,
            lambda p: p.replace(p.value.astype(parameter_dtype)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    device_state_sharding = sharding.fsdp_sharding(
        train_state_shape,
        mesh,
        strategy=config.sharding_strategy,
        log=True,
    )
    device_opt_state_sharding = None
    if config.cpu_offload:
        device_opt_state_sharding = device_state_sharding.opt_state

    if resume:
        return train_state_shape, device_state_sharding, device_opt_state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=device_state_sharding,
    )(init_rng, partial_params)

    return train_state, device_state_sharding, device_opt_state_sharding


def compute_policy_loss(
    config: _config.TrainConfig,
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
):
    if config.recap_beta is None:
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    if (
        observation.unconditional_tokenized_prompt is None
        or observation.unconditional_tokenized_prompt_mask is None
    ):
        raise ValueError("RECAP training requires unconditional prompt tokens in every batch.")
    unconditional_observation = dataclasses.replace(
        observation,
        tokenized_prompt=observation.unconditional_tokenized_prompt,
        tokenized_prompt_mask=observation.unconditional_tokenized_prompt_mask,
        unconditional_tokenized_prompt=None,
        unconditional_tokenized_prompt_mask=None,
    )
    # Reusing rng gives both branches identical image augmentation, flow noise,
    # and time samples, so beta controls only the policy objective.
    unconditional_loss = model.compute_loss(rng, unconditional_observation, actions, train=True)
    conditioned_loss = model.compute_loss(rng, observation, actions, train=True)
    return jnp.mean(unconditional_loss) + config.recap_beta * jnp.mean(conditioned_loss)


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        return compute_policy_loss(config, model, rng, observation, actions)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    config = _config.resolve_finetune_config(config)
    logging.info(f"Running on: {platform.node()}")
    logging.info(
        "Fine-tuning mode=%s, parameter_dtype=%s, sharding_strategy=%s, cpu_offload=%s, "
        "paligemma_variant=%s, action_expert_variant=%s",
        config.finetune_mode,
        config.parameter_dtype,
        config.sharding_strategy,
        config.cpu_offload,
        getattr(config.model, "paligemma_variant", None),
        getattr(config.model, "action_expert_variant", None),
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax_cache_dir = epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()
    jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    metrics_path = config.checkpoint_dir / _METRICS_FILENAME
    metric_records = _load_metric_records(metrics_path) if resuming else []
    checkpoint_metadata = {
        "schema_version": 1,
        "model_type": "pi05",
        "finetune_mode": config.finetune_mode,
        "parameter_dtype": config.parameter_dtype,
        "sharding_strategy": config.sharding_strategy,
        "cpu_offload": config.cpu_offload,
        "fsdp_devices": config.fsdp_devices,
        "paligemma_variant": getattr(config.model, "paligemma_variant", None),
        "action_expert_variant": getattr(config.model, "action_expert_variant", None),
        "recap_conditioning": config.recap_beta is not None,
        "recap_inference_condition": "positive" if config.recap_beta is not None else None,
        "recap_beta": config.recap_beta,
    }
    metadata_text = json.dumps(checkpoint_metadata, indent=2) + "\n"
    (config.checkpoint_dir / "robodojo_pi05_model.json").write_text(metadata_text, encoding="utf-8")
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding, device_opt_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            state_sharding=train_state_sharding,
        )

    # Keep compiled optimizer/model computation on the accelerator. CPU
    # offload is performed explicitly between steps; exposing host-memory
    # optimizer state as a jit input makes XLA lower optimizer operations to
    # the host and triggers host_offloader warnings.
    host_opt_state_sharding = None
    if device_opt_state_sharding is not None:
        host_opt_state_sharding = sharding.with_memory_kind(
            device_opt_state_sharding,
            sharding.resolve_memory_kind(cpu_offload=True),
        )
        jax.block_until_ready(train_state.opt_state)
        train_state = dataclasses.replace(
            train_state,
            opt_state=sharding.put_tree(train_state.opt_state, host_opt_state_sharding),
        )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    if resuming:
        # A process may have logged after the latest durable checkpoint and
        # then crashed. Keep metrics aligned with the resumed state.
        metric_records[:] = [record for record in metric_records if record.get("step", -1) <= start_step]
    _write_metric_records(metrics_path, metric_records)

    lr_schedule = config.lr_schedule.create()
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        if device_opt_state_sharding is not None:
            train_state = dataclasses.replace(
                train_state,
                opt_state=sharding.put_tree(train_state.opt_state, device_opt_state_sharding),
            )
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if host_opt_state_sharding is not None:
            jax.block_until_ready(train_state.opt_state)
            train_state = dataclasses.replace(
                train_state,
                opt_state=sharding.put_tree(train_state.opt_state, host_opt_state_sharding),
            )
        latest_metrics = {
            "loss": float(jax.device_get(info["loss"])),
            "lr": float(lr_schedule(step)),
            "grad_norm": float(jax.device_get(info["grad_norm"])),
            "param_norm": float(jax.device_get(info["param_norm"])),
        }
        pbar.set_postfix(
            loss=f"{latest_metrics['loss']:.5f}",
            lr=f"{latest_metrics['lr']:.3e}",
            grad_norm=f"{latest_metrics['grad_norm']:.5f}",
            param_norm=f"{latest_metrics['param_norm']:.5f}",
            refresh=True,
        )
        infos.append(info)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {key: float(value) for key, value in reduced_info.items()}
            record = {
                "step": int(step),
                "timestamp": time.time(),
                "lr": float(lr_schedule(step)),
                **reduced_info,
            }
            _upsert_metric_record(metric_records, record)
            _write_metric_records(metrics_path, metric_records)
            info_str = ", ".join(f"{key}={value:.5f}" for key, value in reduced_info.items())
            info_str += f", lr={record['lr']:.3e}"
            pbar.write(f"Step {step}: {info_str}")
            wandb.log({**reduced_info, "lr": record["lr"]}, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    checkpoint_manager.wait_until_finished()
    for step_dir in config.checkpoint_dir.iterdir():
        if step_dir.is_dir() and (step_dir / "params").exists():
            (step_dir / "robodojo_pi05_model.json").write_text(metadata_text, encoding="utf-8")
            try:
                checkpoint_step = int(step_dir.name)
            except ValueError:
                checkpoint_records = metric_records
            else:
                checkpoint_records = [record for record in metric_records if record["step"] <= checkpoint_step]
            (step_dir / _METRICS_FILENAME).write_text(_metrics_json(checkpoint_records), encoding="utf-8")


if __name__ == "__main__":
    main(_config.cli())
