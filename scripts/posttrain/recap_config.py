"""Validate and resolve the model-selectable RECAP YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

try:
    from recap_config_base import (
        _MISSING,
        FIELDS as PI05_FIELDS,
        Field,
        _flatten,
        _normalize_value,
        _set_nested,
    )
except ModuleNotFoundError:
    from scripts.posttrain.recap_config_base import (
        _MISSING,
        FIELDS as PI05_FIELDS,
        Field,
        _flatten,
        _normalize_value,
        _set_nested,
    )


FIELDS = dict(PI05_FIELDS)
for obsolete in (
    "devices.pi05_train",
    "training.remote.pi_python",
    "training.remote.pi05_gpus",
    "runtime.pi_python",
):
    FIELDS.pop(obsolete)

FIELDS.update(
    {
        "policy.name": Field("RECAP_POLICY_NAME", "str", "pi05", ("pi05", "g05")),
        "data.format": Field("LEROBOT_DATA_FORMAT", "str", "v2.1", ("v2.1", "v3.0")),
        "rollout.remote.g05_root": Field(
            "RECAP_REMOTE_G05_ROOT", "optional_str", None
        ),
        "devices.policy_train": Field("TRAIN_GPUS", "gpu_list"),
        "training.remote.policy_python": Field(
            "RECAP_TRAINING_REMOTE_POLICY_PYTHON", "optional_str", None
        ),
        "training.remote.policy_gpus": Field(
            "RECAP_TRAINING_REMOTE_POLICY_GPUS", "gpu_list", [0, 1]
        ),
        "g05.root": Field("G05_ROOT", "optional_str", None),
        "g05.task_config": Field("G05_TASK_CONFIG", "str", "robodojo_recap"),
        "g05.processor_path": Field(
            "G05_PROCESSOR_PATH", "str", "checkpoints/qwen3_5_2b_base_processor"
        ),
        "g05.action_tokenizer": Field(
            "G05_ACTION_TOKENIZER_PATH", "optional_str", None
        ),
        "g05.action_source": Field(
            "ROBODOJO_G05_ACTION_SOURCE", "str", "fm", ("fm", "ar")
        ),
        "g05.steps": Field("G05_NUM_TRAIN_STEPS", "int", 3000, minimum=1),
        "g05.batch_size": Field("G05_BATCH_SIZE", "int", 8, minimum=1),
        "g05.num_workers": Field("G05_NUM_WORKERS", "int", 8, minimum=0),
        "g05.grad_accumulation_steps": Field(
            "G05_GRAD_ACCUMULATION_STEPS", "int", 1, minimum=1
        ),
        "g05.optimizer.learning_rate": Field(
            "G05_LEARNING_RATE", "float", 1e-5, minimum=0
        ),
        "g05.optimizer.warmup_steps": Field(
            "G05_WARMUP_STEPS", "int", 500, minimum=0
        ),
        "g05.optimizer.decay_learning_rate": Field(
            "G05_DECAY_LEARNING_RATE", "float", 1e-6, minimum=0
        ),
        "g05.optimizer.decay_start_ratio": Field(
            "G05_DECAY_START_RATIO", "float", 0.5, minimum=0, maximum=1
        ),
        "g05.optimizer.weight_decay": Field(
            "G05_WEIGHT_DECAY", "float", 0.01, minimum=0
        ),
        "g05.wandb_enabled": Field("G05_WANDB_ENABLED", "bool", True),
        "runtime.data_python": Field(
            "DATA_PYTHON_BIN", "optional_str", None
        ),
        "runtime.policy_python": Field("POLICY_PYTHON_BIN", "optional_str", None),
    }
)


def _validate(values: dict[str, Any]) -> None:
    policy = values["policy.name"]
    if values["runtime.data_python"] is None:
        raise ValueError("RECAP requires runtime.data_python for policy dataset materialization.")
    if values["run.reuse_completed_artifacts"] and not values["run.resume"]:
        raise ValueError("run.reuse_completed_artifacts requires run.resume: true.")

    policy_gpus = (
        values["training.remote.policy_gpus"]
        if values["training.remote.enabled"]
        else values["devices.policy_train"]
    )
    if policy == "pi05":
        fsdp_devices = values["pi05.fsdp_devices"]
        if fsdp_devices > len(policy_gpus) or len(policy_gpus) % fsdp_devices:
            raise ValueError("pi05.fsdp_devices must divide the effective Pi0.5 GPU count.")
        if values["pi05.batch_size"] % len(policy_gpus):
            raise ValueError("pi05.batch_size must be divisible by the effective Pi0.5 GPU count.")
    else:
        if values["environment.action_type"] != "joint":
            raise ValueError("G05 RECAP currently requires environment.action_type: joint.")
        if values["data.format"] != "v3.0":
            raise ValueError("G05 RECAP requires data.format: v3.0.")
        if values["g05.root"] is None:
            raise ValueError("G05 RECAP requires g05.root.")
        if values["g05.action_source"] != "fm":
            raise ValueError("G05 RECAP currently requires g05.action_source: fm.")
        if not values["training.remote.enabled"] and values["runtime.policy_python"] is None:
            raise ValueError("Local G05 RECAP training requires runtime.policy_python.")
        if values["training.remote.enabled"] and values["training.remote.policy_python"] is None:
            raise ValueError("Remote G05 training requires training.remote.policy_python.")
        if values["recap.guidance_scale"] != 1.0:
            raise ValueError("G05 RECAP currently supports recap.guidance_scale: 1.0.")

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
        raise ValueError(
            "recap.gamma must be positive and recap.positive_fraction must be in (0, 1)."
        )
    if values["recap.sampling.demonstrations"] <= 0 or values["recap.sampling.rollouts"] <= 0:
        raise ValueError("RECAP sampling weights must be positive.")
    if policy == "g05" and (
        values["recap.sampling.demonstrations"] != 1.0
        or values["recap.sampling.rollouts"] != 1.0
    ):
        raise ValueError("G05 currently requires equal RECAP source sampling weights (1.0/1.0).")

    steps = values[f"{policy}.steps"]
    warmup = values[f"{policy}.optimizer.warmup_steps"]
    if warmup >= steps:
        raise ValueError(f"{policy}.optimizer.warmup_steps must be smaller than {policy}.steps.")
    if policy == "g05":
        learning_rate = values["g05.optimizer.learning_rate"]
        decay_learning_rate = values["g05.optimizer.decay_learning_rate"]
        decay_start_ratio = values["g05.optimizer.decay_start_ratio"]
        if decay_learning_rate > learning_rate:
            raise ValueError(
                "g05.optimizer.decay_learning_rate must not exceed "
                "g05.optimizer.learning_rate."
            )
        if not 0 < decay_start_ratio < 1:
            raise ValueError("g05.optimizer.decay_start_ratio must be in (0, 1).")
        if int(steps * decay_start_ratio) < warmup:
            raise ValueError(
                "g05.optimizer.decay_start_ratio starts decay before warmup completes."
            )
    if values["evaluation.interval"] >= steps:
        raise ValueError(f"evaluation.interval must be smaller than {policy}.steps.")
    if values[f"{policy}.optimizer.learning_rate"] <= 0:
        raise ValueError(f"{policy}.optimizer.learning_rate must be positive.")
    if values["wcm.train.learning_rate"] is not None and values["wcm.train.learning_rate"] <= 0:
        raise ValueError("wcm.train.learning_rate must be positive when specified.")
    if values["pi05.xla_memory_fraction"] <= 0:
        raise ValueError("pi05.xla_memory_fraction must be in (0, 1].")
    if values["value_video.speed"] <= 0:
        raise ValueError("value_video.speed must be positive.")

    if values["training.remote.enabled"]:
        required = (
            "training.remote.host",
            "training.remote.repo_root",
            "training.remote.work_root",
        )
        missing = [name for name in required if values[name] is None]
        if missing:
            raise ValueError("Remote training requires: " + ", ".join(missing))

    remote_required = (
        "rollout.remote.host",
        "rollout.remote.repo_root",
        "rollout.remote.work_root",
    )
    if values["rollout.remote.enabled"]:
        missing = [name for name in remote_required if values[name] is None]
        if policy == "g05" and values["rollout.remote.g05_root"] is None:
            missing.append("rollout.remote.g05_root")
        if missing:
            raise ValueError("Remote rollout requires: " + ", ".join(missing))
    else:
        for name in remote_required:
            if values[name] is not None:
                raise ValueError(f"{name} must be null when rollout.remote.enabled is false.")
        if values["rollout.remote.policy_evaluation"]:
            raise ValueError("rollout.remote.policy_evaluation requires rollout.remote.enabled.")


def resolve(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    flat = _flatten(payload)
    version = flat.pop("schema_version", None)
    if version != 2:
        raise ValueError(f"schema_version must be 2, got {version!r}.")
    unknown = sorted(set(flat) - set(FIELDS))
    if unknown:
        raise ValueError("Unknown RECAP config field(s): " + ", ".join(unknown))

    values: dict[str, Any] = {}
    for field_path, field in FIELDS.items():
        raw = flat.get(field_path, field.default)
        if raw is _MISSING:
            raise ValueError(f"Missing required RECAP config field: {field_path}")
        values[field_path] = _normalize_value(field_path, field, raw)
    _validate(values)

    resolved: dict[str, Any] = {"schema_version": 2}
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
    resolved, environment = resolve(Path(args.config).expanduser().resolve())
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
