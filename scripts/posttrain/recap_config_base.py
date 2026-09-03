"""Validate and resolve the unified Pi0.5 RECAP YAML configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml

_MISSING = object()


@dataclass(frozen=True)
class Field:
    env: str
    kind: str
    default: Any = _MISSING
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


FIELDS: dict[str, Field] = {
    "run.task": Field("TASK_NAME", "str"),
    "run.output_root": Field("OUTPUT_ROOT", "str", "outputs/recap"),
    "run.iterations": Field("RECAP_ITERATIONS", "int", 3, minimum=1),
    "run.resume": Field("RECAP_RESUME", "bool", False),
    "run.reuse_completed_artifacts": Field("RECAP_REUSE_COMPLETED_ARTIFACTS", "bool", False),
    "run.seed": Field("SEED", "int", 0, minimum=0),
    "environment.env_cfg": Field("ENV_CFG_TYPE", "str", "arx_x5"),
    "environment.action_type": Field("ACTION_TYPE", "str", "joint", ("joint", "ee")),
    "environment.policy_env": Field("POLICY_ENV", "str", "openpi"),
    "environment.eval_env": Field("EVAL_ENV", "str", "RoboDojo"),
    "checkpoints.initial_policy": Field("INITIAL_POLICY_CHECKPOINT", "str"),
    "checkpoints.initial_wcm": Field("INITIAL_WCM_CHECKPOINT", "optional_str", None),
    "data.demo_root": Field("DEMO_ROOT", "str"),
    "data.max_demo_episodes": Field("RECAP_MAX_DEMO_EPISODES", "int", 100, minimum=0),
    "data.mode": Field("OPENPI_DATA_MODE", "str", "video", ("video", "image")),
    "data.norm_asset_id": Field("OPENPI_NORM_ASSET_ID", "optional_str", None),
    "devices.pi05_train": Field("TRAIN_GPUS", "gpu_list"),
    "devices.wcm_train": Field("WCM_TRAIN_GPUS", "gpu_list"),
    "devices.rollout_policy": Field("POLICY_GPU", "int", 0, minimum=0),
    "devices.rollout_environment": Field("ENV_GPU", "int", 1, minimum=0),
    "devices.value_video": Field("RECAP_VALUE_VIDEO_GPU", "int", 0, minimum=0),
    "devices.reservation.enabled": Field("GPU_RESERVATION_ENABLED", "bool", True),
    "devices.reservation.leave_free_mib": Field("GPU_RESERVATION_FREE_MIB", "int", 2048, minimum=256),
    "devices.reservation.idle_used_max_mib": Field(
        "GPU_RESERVATION_IDLE_USED_MAX_MIB", "int", 64, minimum=0
    ),
    "devices.reservation.local_max_hold_seconds": Field(
        "GPU_RESERVATION_LOCAL_MAX_HOLD_SECONDS", "int", 1800, minimum=60
    ),
    "devices.reservation.remote_max_hold_seconds": Field(
        "GPU_RESERVATION_REMOTE_MAX_HOLD_SECONDS", "int", 1800, minimum=60
    ),
    "training.remote.enabled": Field("RECAP_TRAINING_REMOTE_ENABLED", "bool", False),
    "training.remote.host": Field("RECAP_TRAINING_REMOTE_HOST", "optional_str", None),
    "training.remote.repo_root": Field("RECAP_TRAINING_REMOTE_REPO_ROOT", "optional_str", None),
    "training.remote.work_root": Field("RECAP_TRAINING_REMOTE_WORK_ROOT", "optional_str", None),
    "training.remote.zstd": Field("RECAP_TRAINING_REMOTE_ZSTD_BIN", "str", "zstd"),
    "training.remote.conda": Field("RECAP_TRAINING_REMOTE_CONDA_BIN", "str", "conda"),
    "training.remote.python": Field("RECAP_TRAINING_REMOTE_PYTHON_BIN", "str", "python"),
    "training.remote.pi_python": Field("RECAP_TRAINING_REMOTE_PI_PYTHON", "optional_str", None),
    "training.remote.wcm_python": Field("RECAP_TRAINING_REMOTE_WCM_PYTHON", "optional_str", None),
    "training.remote.pi05_gpus": Field("RECAP_TRAINING_REMOTE_PI05_GPUS", "gpu_list", [0, 1]),
    "training.remote.wcm_gpus": Field("RECAP_TRAINING_REMOTE_WCM_GPUS", "gpu_list", [0, 1]),
    "training.remote.value_video_gpu": Field("RECAP_TRAINING_REMOTE_VALUE_VIDEO_GPU", "int", 0, minimum=0),
    "training.remote.render_value_video": Field("RECAP_TRAINING_REMOTE_RENDER_VALUE_VIDEO", "bool", True),
    "rollout.episodes": Field("RECAP_ROLLOUT_EPISODES", "int", 100, minimum=1),
    "rollout.max_steps": Field("RECAP_ROLLOUT_MAX_STEPS", "int", 40, minimum=1),
    "rollout.fixed_horizon": Field(
        "RECAP_ROLLOUT_FIXED_HORIZON", "bool", False
    ),
    "rollout.layout_seed": Field("RECAP_ROLLOUT_LAYOUT_SEED", "int", 0, minimum=0),
    "rollout.minimum.total": Field("RECAP_MIN_ROLLOUT_EPISODES", "int", 50, minimum=1),
    "rollout.minimum.successes": Field("RECAP_MIN_SUCCESS_EPISODES", "int", 5, minimum=0),
    "rollout.minimum.failures": Field("RECAP_MIN_FAILURE_EPISODES", "int", 5, minimum=0),
    "rollout.remote.enabled": Field("RECAP_REMOTE_ENABLED", "bool", False),
    "rollout.remote.host": Field("RECAP_REMOTE_ROLLOUT_HOST", "optional_str", None),
    "rollout.remote.repo_root": Field("RECAP_REMOTE_REPO_ROOT", "optional_str", None),
    "rollout.remote.work_root": Field("RECAP_REMOTE_WORK_ROOT", "optional_str", None),
    "rollout.remote.zstd": Field("RECAP_REMOTE_ZSTD_BIN", "str", "zstd"),
    "rollout.remote.conda": Field("RECAP_REMOTE_CONDA_BIN", "str", "conda"),
    "rollout.remote.python": Field("RECAP_REMOTE_PYTHON_BIN", "str", "python"),
    "rollout.remote.policy_env": Field("RECAP_REMOTE_POLICY_ENV", "optional_str", None),
    "rollout.remote.eval_env": Field("RECAP_REMOTE_EVAL_ENV", "optional_str", None),
    "rollout.remote.policy_gpu": Field("RECAP_REMOTE_POLICY_GPU", "int", 0, minimum=0),
    "rollout.remote.environment_gpu": Field("RECAP_REMOTE_ENV_GPU", "int", 0, minimum=0),
    "rollout.remote.value_video_gpu": Field("RECAP_REMOTE_VALUE_VIDEO_GPU", "int", 0, minimum=0),
    "rollout.remote.policy_evaluation": Field("RECAP_REMOTE_POLICY_EVAL", "bool", False),
    "wcm.train.config": Field("WCM_CONFIG", "str", "configs/wcm/robodojo_pi05.yaml"),
    "wcm.train.epochs": Field("RECAP_WCM_EPOCHS", "int", 5, minimum=1),
    "wcm.train.replay_episodes": Field("RECAP_WCM_REPLAY_EPISODES", "int", 20, minimum=0),
    "wcm.train.per_device_batch_size": Field("WCM_PER_DEVICE_BATCH_SIZE", "int", 16, minimum=1),
    "wcm.train.num_workers": Field("WCM_NUM_WORKERS", "int", 8, minimum=0),
    "wcm.train.precision": Field("WCM_PRECISION", "str", "bf16", ("fp32", "fp16", "bf16")),
    "wcm.train.learning_rate": Field("WCM_LR", "optional_float", None, minimum=0),
    "wcm.train.warmup_steps": Field("WCM_WARMUP_STEPS", "optional_int", None, minimum=0),
    "wcm.train.video_decoder": Field("WCM_VIDEO_DECODER", "str", "pyav", ("pyav", "opencv")),
    "wcm.inference.batch_size": Field("RECAP_WCM_INFER_BATCH_SIZE", "int", 8, minimum=1),
    "wcm.inference.num_workers": Field("RECAP_WCM_NUM_WORKERS", "int", 2, minimum=0),
    "wcm.inference.device": Field("RECAP_WCM_DEVICE", "str", "cuda"),
    "recap.lookahead": Field("RECAP_LOOKAHEAD", "int", 10, minimum=1),
    "recap.gamma": Field("RECAP_GAMMA", "float", 1.0, minimum=0, maximum=1),
    "recap.failure_penalty": Field("WCM_FAILURE_PENALTY", "float", 300.0, minimum=0),
    "recap.positive_fraction": Field("RECAP_POSITIVE_FRACTION", "float", 0.3, minimum=0, maximum=1),
    "recap.unconditional_probability": Field("RECAP_UNCONDITIONAL_PROB", "float", 0.1, minimum=0, maximum=1),
    "recap.guidance_scale": Field("RECAP_GUIDANCE_SCALE", "float", 1.0, minimum=0),
    "recap.sampling.demonstrations": Field("RECAP_DEMO_SAMPLING_WEIGHT", "float", 1.0, minimum=0),
    "recap.sampling.rollouts": Field("RECAP_ROLLOUT_SAMPLING_WEIGHT", "float", 1.0, minimum=0),
    "pi05.train_config": Field("OPENPI_TRAIN_CONFIG_NAME", "str", "pi05_base_aloha_full_sim_arx-x5_seed_0"),
    "pi05.finetune_mode": Field(
        "PI05_FINETUNE_MODE",
        "str",
        "action_expert_lora",
        ("full", "action_expert", "action_expert_lora", "paligemma_lora", "all_lora"),
    ),
    "pi05.steps": Field("OPENPI_NUM_TRAIN_STEPS", "int", 3000, minimum=1),
    "pi05.batch_size": Field("OPENPI_BATCH_SIZE", "int", 64, minimum=1),
    "pi05.num_workers": Field("OPENPI_NUM_WORKERS", "int", 4, minimum=0),
    "pi05.parameter_dtype": Field("OPENPI_PARAMETER_DTYPE", "str", "bfloat16", ("bfloat16", "float32")),
    "pi05.sharding_strategy": Field(
        "OPENPI_SHARDING_STRATEGY",
        "str",
        "full_shard",
        ("full_shard", "shard_grad_op", "no_shard"),
    ),
    "pi05.fsdp_devices": Field("OPENPI_FSDP_DEVICES", "int", 2, minimum=1),
    "pi05.cpu_offload": Field("OPENPI_CPU_OFFLOAD", "bool", False),
    "pi05.ema_decay": Field("OPENPI_EMA_DECAY", "optional_float", 0.99, minimum=0, maximum=1),
    "pi05.action_expert_variant": Field("OPENPI_ACTION_EXPERT_VARIANT", "optional_str", None),
    "pi05.paligemma_variant": Field("OPENPI_PALIGEMMA_VARIANT", "optional_str", None),
    "pi05.xla_memory_fraction": Field("XLA_PYTHON_CLIENT_MEM_FRACTION", "float", 0.9, minimum=0, maximum=1),
    "pi05.wandb_enabled": Field("OPENPI_WANDB_ENABLED", "bool", True),
    "pi05.optimizer.learning_rate": Field("OPENPI_LEARNING_RATE", "float", 5e-6, minimum=0),
    "pi05.optimizer.warmup_steps": Field("OPENPI_WARMUP_STEPS", "int", 500, minimum=0),
    "pi05.optimizer.decay_lr": Field("OPENPI_DECAY_LR", "optional_float", None, minimum=0),
    "pi05.optimizer.weight_decay": Field("OPENPI_WEIGHT_DECAY", "float", 1e-10, minimum=0),
    "pi05.optimizer.clip_gradient_norm": Field("OPENPI_CLIP_GRADIENT_NORM", "float", 1.0, minimum=0),
    "pi05.log_interval": Field("OPENPI_LOG_INTERVAL", "int", 100, minimum=1),
    "evaluation.interval": Field("RECAP_POLICY_EVAL_INTERVAL", "int", 1000, minimum=1),
    "evaluation.episodes": Field("RECAP_POLICY_EVAL_EPISODES", "int", 20, minimum=1),
    "evaluation.reuse_rollout": Field("RECAP_POLICY_EVAL_REUSE_ROLLOUT", "bool", True),
    "evaluation.layout_seed": Field("RECAP_POLICY_EVAL_LAYOUT_SEED", "int", 1, minimum=0),
    "evaluation.layout_offset": Field("RECAP_POLICY_EVAL_LAYOUT_OFFSET", "int", 0, minimum=0),
    "value_video.episodes": Field("RECAP_VALUE_VIDEO_EPISODES", "int", 3, minimum=0),
    "value_video.batch_size": Field("RECAP_VALUE_VIDEO_BATCH_SIZE", "int", 8, minimum=1),
    "value_video.device": Field("RECAP_VALUE_VIDEO_DEVICE", "str", "cuda"),
    "value_video.precision": Field("RECAP_VALUE_VIDEO_PRECISION", "str", "bf16", ("fp32", "fp16", "bf16")),
    "value_video.backend": Field("RECAP_VALUE_VIDEO_BACKEND", "str", "auto", ("auto", "pyav", "ffmpeg")),
    "value_video.speed": Field("RECAP_VALUE_VIDEO_SPEED", "float", 1.0, minimum=0),
    "value_video.y_min": Field("RECAP_VALUE_VIDEO_Y_MIN", "float", -1.0),
    "value_video.y_max": Field("RECAP_VALUE_VIDEO_Y_MAX", "float", 1.0),
    "runtime.pi_python": Field("PI_PYTHON_BIN", "optional_str", None),
    "runtime.wcm_python": Field("WCM_PYTHON_BIN", "optional_str", None),
    "runtime.policy_dir": Field("POLICY_DIR", "optional_str", None),
}


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Pi0.5 RECAP config must be a YAML mapping.")
    result: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _number(value: Any, path: str, *, integer: bool) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        expected = "integer" if integer else "number"
        raise TypeError(f"{path} must be a YAML {expected}.")
    if integer and not isinstance(value, int):
        raise TypeError(f"{path} must be a YAML integer.")
    return int(value) if integer else float(value)


def _normalize_value(path: str, field: Field, value: Any) -> Any:
    optional = field.kind.startswith("optional_")
    kind = field.kind.removeprefix("optional_")
    if value is None:
        if optional:
            return None
        raise TypeError(f"{path} cannot be null.")
    if kind == "str":
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{path} must be a non-empty YAML string.")
        normalized: Any = value
    elif kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be true or false.")
        normalized = value
    elif kind == "int":
        normalized = _number(value, path, integer=True)
    elif kind == "float":
        normalized = _number(value, path, integer=False)
    elif kind == "gpu_list":
        if not isinstance(value, list) or not value:
            raise TypeError(f"{path} must be a non-empty YAML list of GPU indices.")
        normalized = [_number(item, path, integer=True) for item in value]
        if any(item < 0 for item in normalized):
            raise ValueError(f"{path} cannot contain negative GPU indices.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{path} contains duplicate GPU indices.")
    else:
        raise AssertionError(f"Unknown field kind: {field.kind}")
    if field.choices and normalized not in field.choices:
        raise ValueError(f"{path} must be one of {field.choices}, got {normalized!r}.")
    if field.minimum is not None and normalized < field.minimum:
        raise ValueError(f"{path} must be at least {field.minimum}.")
    if field.maximum is not None and normalized > field.maximum:
        raise ValueError(f"{path} must be at most {field.maximum}.")
    return normalized


def _set_nested(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    flat = _flatten(payload)
    version = flat.pop("schema_version", None)
    if version != 1:
        raise ValueError(f"schema_version must be 1, got {version!r}.")
    unknown = sorted(set(flat) - set(FIELDS))
    if unknown:
        raise ValueError("Unknown Pi0.5 RECAP config field(s): " + ", ".join(unknown))

    values: dict[str, Any] = {}
    for field_path, field in FIELDS.items():
        raw = flat.get(field_path, field.default)
        if raw is _MISSING:
            raise ValueError(f"Missing required Pi0.5 RECAP config field: {field_path}")
        values[field_path] = _normalize_value(field_path, field, raw)

    if values["run.reuse_completed_artifacts"] and not values["run.resume"]:
        raise ValueError("run.reuse_completed_artifacts requires run.resume: true.")

    pi_gpus = (
        values["training.remote.pi05_gpus"]
        if values["training.remote.enabled"]
        else values["devices.pi05_train"]
    )
    fsdp_devices = values["pi05.fsdp_devices"]
    if fsdp_devices > len(pi_gpus) or len(pi_gpus) % fsdp_devices:
        raise ValueError("pi05.fsdp_devices must divide the effective Pi0.5 GPU count.")
    if values["pi05.batch_size"] % len(pi_gpus):
        raise ValueError("pi05.batch_size must be divisible by the effective Pi0.5 GPU count.")
    episodes = values["rollout.episodes"]
    if values["rollout.minimum.total"] > episodes:
        raise ValueError("rollout.minimum.total cannot exceed rollout.episodes.")
    if values["rollout.minimum.successes"] + values["rollout.minimum.failures"] > episodes:
        raise ValueError("rollout success/failure minimums cannot exceed rollout.episodes.")
    if values["value_video.episodes"] > episodes:
        raise ValueError("value_video.episodes cannot exceed rollout.episodes.")
    if values["value_video.y_min"] >= values["value_video.y_max"]:
        raise ValueError("value_video.y_min must be smaller than value_video.y_max.")
    if values["recap.gamma"] <= 0 or values["recap.positive_fraction"] in {0.0, 1.0}:
        raise ValueError("recap.gamma must be positive and recap.positive_fraction must be in (0, 1).")
    if values["recap.sampling.demonstrations"] <= 0 or values["recap.sampling.rollouts"] <= 0:
        raise ValueError("RECAP sampling weights must be positive.")
    if values["pi05.optimizer.warmup_steps"] >= values["pi05.steps"]:
        raise ValueError("pi05.optimizer.warmup_steps must be smaller than pi05.steps.")
    if values["evaluation.interval"] >= values["pi05.steps"]:
        raise ValueError("evaluation.interval must be smaller than pi05.steps.")
    if values["wcm.train.learning_rate"] is not None and values["wcm.train.learning_rate"] <= 0:
        raise ValueError("wcm.train.learning_rate must be positive when specified.")
    if values["pi05.optimizer.learning_rate"] <= 0:
        raise ValueError("pi05.optimizer.learning_rate must be positive.")
    if values["pi05.xla_memory_fraction"] <= 0:
        raise ValueError("pi05.xla_memory_fraction must be in (0, 1].")
    if values["value_video.speed"] <= 0:
        raise ValueError("value_video.speed must be positive.")

    training_remote_enabled = values["training.remote.enabled"]
    training_remote_required = (
        "training.remote.host",
        "training.remote.repo_root",
        "training.remote.work_root",
    )
    if training_remote_enabled:
        missing = [name for name in training_remote_required if values[name] is None]
        if missing:
            raise ValueError("Remote training requires: " + ", ".join(missing))

    remote_enabled = values["rollout.remote.enabled"]
    remote_required = (
        "rollout.remote.host",
        "rollout.remote.repo_root",
        "rollout.remote.work_root",
    )
    if remote_enabled:
        missing = [name for name in remote_required if values[name] is None]
        if missing:
            raise ValueError("Remote rollout requires: " + ", ".join(missing))
    else:
        for name in remote_required:
            if values[name] is not None:
                raise ValueError(f"{name} must be null when rollout.remote.enabled is false.")
        if values["rollout.remote.policy_evaluation"]:
            raise ValueError("rollout.remote.policy_evaluation requires rollout.remote.enabled.")

    resolved: dict[str, Any] = {"schema_version": 1}
    environment: dict[str, str] = {}
    for field_path, field in FIELDS.items():
        value = values[field_path]
        _set_nested(resolved, field_path, value)
        if isinstance(value, bool):
            environment[field.env] = "1" if value else "0"
        elif isinstance(value, list):
            environment[field.env] = ",".join(map(str, value))
        elif value is None:
            environment[field.env] = ""
        else:
            environment[field.env] = str(value)
    return resolved, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--format", choices=("env", "yaml"), default="env")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    resolved, environment = resolve(config_path)
    if args.format == "env":
        if args.output:
            parser.error("--output is only valid with --format yaml")
        for name, value in environment.items():
            sys.stdout.buffer.write(name.encode() + b"\0" + value.encode() + b"\0")
        return
    rendered = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
