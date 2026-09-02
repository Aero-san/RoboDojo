#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.adaptive_action_chunking import (
    ActionLayout,
    AdaptiveActionChunker,
    AdaptiveActionChunkingConfig,
    absolute_actions_to_offsets,
)
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .bcp import Pi05BCPRuntime

_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"Expected a boolean value, got {value!r}.")
    return normalized in {"1", "true", "yes"}


def _parse_optional_int(value: Any) -> int | None:
    normalized = str(value).strip().lower()
    return None if normalized in {"", "none", "null"} else int(normalized)


def _as_action_chunk(actions: Any) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Pi0.5 returned actions with unsupported shape {array.shape}.")
    return array


def _as_sampled_action_chunks(actions: Any) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            "Pi0.5 AAC inference must return [samples, horizon, action_dim], "
            f"got {array.shape}."
        )
    return array


def _trim_action_dim(actions: np.ndarray, expected_dim: int, action_type: str) -> np.ndarray:
    if actions.shape[-1] < expected_dim:
        raise ValueError(
            f"Pi0.5 returned {actions.shape[-1]} action dimensions, "
            f"but {action_type} control for this robot requires {expected_dim}."
        )
    return actions[..., :expected_dim]


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self._latest_reference_action_list: list[Any] = []
        self._latest_initial_noise_action_list: list[np.ndarray] = []

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy
        bcp_config = model_cfg.get("bernoulli_continuation") or {}
        if not isinstance(bcp_config, dict):
            raise TypeError("bernoulli_continuation must be a YAML mapping.")
        self.bcp = Pi05BCPRuntime(bcp_config, _POLICY_DIR)
        self.adaptive_action_chunker = self._build_adaptive_action_chunker(model_cfg)
        self.last_adaptive_chunking_results: dict[int, Any] = {}
        self.posttrain_mode = str(model_cfg.get("posttrain_mode", "base")).lower()
        self.posttrain = self._load_posttrain(model_cfg)
        deterministic_raw = str(
            os.environ.get(
                "POSTTRAIN_DETERMINISTIC",
                model_cfg.get("posttrain_deterministic", "1"),
            )
        ).strip().lower()
        if deterministic_raw not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "POSTTRAIN_DETERMINISTIC/posttrain_deterministic must be a boolean value."
            )
        self.posttrain_deterministic = deterministic_raw in {"1", "true", "yes"}
        self.recap_condition = getattr(self, "_checkpoint_metadata", {}).get(
            "recap_inference_condition"
        )

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        if self.robot_action_dim_info is not None and hasattr(config.data, "output_action_dim"):
            arm_dim = (
                sum(self.robot_action_dim_info["arm_dim"])
                if self.action_type == "joint"
                else 7 * len(self.robot_action_dim_info["arm_dim"])
            )
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(
                    config.data,
                    output_action_dim=arm_dim + sum(self.robot_action_dim_info["ee_dim"]),
                ),
            )
        metadata_path = model_root / "robodojo_pi05_model.json"
        if not metadata_path.exists():
            metadata_path = model_root.parent / "robodojo_pi05_model.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Pi0.5 checkpoint metadata must be an object: {metadata_path}")
            model_updates = {
                key: metadata[key]
                for key in ("paligemma_variant", "action_expert_variant")
                if key in metadata
            }
            if model_updates:
                config = dataclasses.replace(
                    config,
                    model=dataclasses.replace(config.model, **model_updates),
                )
            self._checkpoint_metadata = metadata
        else:
            self._checkpoint_metadata = {}
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        if self.recap_condition:
            for encoded in encoded_obs_list:
                prompt = str(encoded.get("prompt") or self.task_name)
                prompt = prompt.rsplit("\nAdvantage:", 1)[0]
                encoded["prompt"] = f"{prompt}\nAdvantage: {self.recap_condition}"
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        result = {
            "actions": action_list[0],
            "initial_noise_actions": self._latest_initial_noise_action_list[0],
        }
        if self.posttrain is not None:
            result["reference_actions"] = self._latest_reference_action_list[0]
        return result

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []
        reference_action_list = []
        initial_noise_action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            if self.bcp.enabled:
                prediction = self.policy.infer(
                    single_observation,
                    return_normalized_actions=True,
                    return_bcp_features=not self.bcp.reference,
                    **kwargs,
                )
                full_candidate = _as_action_chunk(prediction["actions"])
                horizon = self.bcp.select_horizon(
                    prediction,
                    int(env_idx_list[batch_index]),
                )
                actions = full_candidate[:horizon]
            elif self.adaptive_action_chunker is None:
                prediction = self.policy.infer(single_observation, **kwargs)
                actions = _as_action_chunk(prediction["actions"])
            else:
                prediction = self.policy.infer(
                    single_observation,
                    num_samples=self.adaptive_action_chunker.config.num_samples,
                    return_normalized_actions=True,
                    **kwargs,
                )
                sampled_actions = _as_sampled_action_chunks(prediction["actions"])
                normalized_samples = _as_sampled_action_chunks(prediction["normalized_actions"])
                expected_dim = self._expected_action_dim()
                sampled_actions = _trim_action_dim(sampled_actions, expected_dim, self.action_type)
                normalized_samples = _trim_action_dim(normalized_samples, expected_dim, self.action_type)
                magnitude_samples = absolute_actions_to_offsets(
                    sampled_actions,
                    np.asarray(single_observation["state"], dtype=np.float32)[:expected_dim],
                    self.adaptive_action_chunker.layout,
                )
                adaptive_result = self.adaptive_action_chunker.select(
                    normalized_samples,
                    sampled_actions,
                    magnitude_samples,
                )
                candidate_index = self.adaptive_action_chunker.config.candidate_index
                full_candidate = sampled_actions[candidate_index]
                actions = full_candidate[: adaptive_result.chunk_size]
                self.last_adaptive_chunking_results[int(env_idx_list[batch_index])] = adaptive_result

            initial_noise_actions = np.asarray(
                prediction["initial_noise_actions"],
                dtype=np.float32,
            )
            if self.adaptive_action_chunker is not None and not self.bcp.enabled:
                initial_noise_actions = initial_noise_actions[candidate_index]
            if initial_noise_actions.ndim != 2:
                raise ValueError(
                    "Pi0.5 initial action noise must have shape [horizon, action_dim] "
                    f"after candidate selection, got {initial_noise_actions.shape}."
                )

            if self.robot_action_dim_info is not None:
                actions = _trim_action_dim(actions, self._expected_action_dim(), self.action_type)
            reference_actions = actions.copy()
            if self.posttrain is not None:
                if self.adaptive_action_chunker is None and not self.bcp.enabled:
                    actions = self._apply_posttrain(single_observation, actions)
                else:
                    posttrained = self._apply_posttrain(single_observation, full_candidate)
                    actions = posttrained[: len(reference_actions)]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
                reference_action_list.append(reference_actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
                reference_action_list.append(
                    unpack_robot_state(
                        reference_actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
            initial_noise_action_list.append(initial_noise_actions)

        self._latest_reference_action_list = reference_action_list
        self._latest_initial_noise_action_list = initial_noise_action_list
        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._latest_reference_action_list = []
        self._latest_initial_noise_action_list = []
        self.last_adaptive_chunking_results = {}
        self.bcp.reset()

    def on_trial_end(self, payload: dict[str, Any]):
        return self.bcp.on_trial_end(payload)

    def reset_obsrvationwindows(self):
        self.reset()

    def _expected_action_dim(self) -> int:
        if self.robot_action_dim_info is None:
            raise ValueError("env_cfg_type is required to determine the Pi0.5 action layout.")
        arm_count = len(self.robot_action_dim_info["arm_dim"])
        arm_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            if self.action_type == "joint"
            else 7 * arm_count
        )
        return arm_dim + sum(self.robot_action_dim_info["ee_dim"])

    def _build_adaptive_action_chunker(
        self,
        model_cfg: dict[str, Any],
    ) -> AdaptiveActionChunker | None:
        raw_config = model_cfg.get("adaptive_action_chunking") or {}
        if not isinstance(raw_config, dict):
            raise TypeError("adaptive_action_chunking must be a YAML mapping.")
        config_values = dict(raw_config)
        environment_overrides = {
            "enabled": ("AAC_ENABLED", _parse_bool),
            "num_samples": ("AAC_NUM_SAMPLES", int),
            "min_chunk_size": ("AAC_MIN_CHUNK_SIZE", int),
            "max_chunk_size": ("AAC_MAX_CHUNK_SIZE", _parse_optional_int),
            "movement_threshold": ("AAC_MOVEMENT_THRESHOLD", float),
        }
        for key, (environment_name, parser) in environment_overrides.items():
            if environment_name in os.environ:
                config_values[key] = parser(os.environ[environment_name])

        config = AdaptiveActionChunkingConfig.from_mapping(config_values)
        # BCP and AAC both own execution-horizon selection. BCP takes explicit
        # precedence so AAC's deploy default cannot silently add multi-sample
        # inference or override the learned horizon.
        if not config.enabled or self.bcp.enabled:
            return None
        if self.robot_action_dim_info is None:
            raise ValueError("AAC requires env_cfg_type to infer the robot action layout.")

        continuous_groups = []
        discrete_indices = []
        offset = 0
        for arm_dim, end_effector_dim in zip(
            self.robot_action_dim_info["arm_dim"],
            self.robot_action_dim_info["ee_dim"],
            strict=True,
        ):
            action_arm_dim = arm_dim if self.action_type == "joint" else 7
            if self.action_type == "ee":
                continuous_groups.append(tuple(range(offset, offset + 3)))
                continuous_groups.append(tuple(range(offset + 3, offset + action_arm_dim)))
            else:
                continuous_groups.append(tuple(range(offset, offset + action_arm_dim)))
            offset += action_arm_dim
            discrete_indices.extend(range(offset, offset + end_effector_dim))
            offset += end_effector_dim

        layout = ActionLayout(tuple(continuous_groups), tuple(discrete_indices))
        layout.validate(offset)
        return AdaptiveActionChunker(config, layout)

    def _load_posttrain(self, model_cfg: dict[str, Any]):
        if self.posttrain_mode in {"base", "pi05", "none", ""}:
            return None
        if self.posttrain_mode not in {"rltoken", "wcm_actor"}:
            raise ValueError(
                f"Unsupported posttrain_mode={self.posttrain_mode!r}; "
                "choose base, rltoken, or wcm_actor."
            )
        import torch

        from .posttrain.runtime import load_posttrain_checkpoint

        if self.robot_action_dim_info is None:
            raise ValueError("Pi0.5 post-training requires env_cfg_type to define the RoboDojo action layout.")
        raw_path = model_cfg.get("posttrain_checkpoint") or os.environ.get("POSTTRAIN_CHECKPOINT")
        if not raw_path:
            raise ValueError(
                "Pi_05 post-training mode requires deploy.yml posttrain_checkpoint "
                "or POSTTRAIN_CHECKPOINT=/path/to/artifact.pt."
            )
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (_POLICY_DIR / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pi0.5 post-training checkpoint not found: {path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = load_posttrain_checkpoint(path, device)
        arm_dim = (
            sum(self.robot_action_dim_info["arm_dim"])
            if self.action_type == "joint"
            else 7 * len(self.robot_action_dim_info["arm_dim"])
        )
        expected_action_dim = arm_dim + sum(self.robot_action_dim_info["ee_dim"])
        if loaded["action_dim"] != expected_action_dim:
            raise ValueError(
                "Post-training action dimension does not match the selected RoboDojo robot: "
                f"{loaded['action_dim']} != {expected_action_dim}."
            )
        if loaded["state_dim"] != loaded["action_dim"]:
            raise ValueError(
                "Pi0.5 post-training currently expects packed state and action dimensions to match; "
                f"got state_dim={loaded['state_dim']} and action_dim={loaded['action_dim']}."
            )
        return loaded

    def _apply_posttrain(self, observation: dict[str, Any], reference: np.ndarray) -> np.ndarray:
        from .posttrain.runtime import physical_action_chunk

        if reference.shape[-1] != self.posttrain["action_dim"]:
            raise ValueError(
                f"Pi0.5 action dimension {reference.shape[-1]} does not match post-training "
                f"checkpoint {self.posttrain['action_dim']}."
            )
        encoded = encode_obs(observation, self.action_type, self.robot_action_dim_info)
        state = np.asarray(encoded["state"], dtype=np.float32).reshape(-1)
        chunk_steps = self.posttrain["chunk_steps"]
        if reference.shape[0] < chunk_steps:
            pad = np.repeat(reference[-1:], chunk_steps - reference.shape[0], axis=0)
            reference = np.concatenate((reference, pad), axis=0)
        reference = reference[:chunk_steps]
        return physical_action_chunk(
            self.posttrain["encoder"],
            self.posttrain["actor"],
            state,
            reference,
            self.posttrain["action_mean"],
            self.posttrain["action_std"],
            deterministic=self.posttrain_deterministic,
        )


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        source_images = observation["images"]
        high = source_images.get("cam_high")
        if high is None:
            high = source_images.get("cam_head")
        wrist = source_images.get("cam_wrist")
        left = source_images.get("cam_left_wrist")
        if left is None:
            left = wrist if wrist is not None else high
        right = source_images.get("cam_right_wrist")
        if right is None:
            right = wrist if wrist is not None else high
        if high is None or left is None or right is None:
            raise KeyError("Pi0.5 observations need a head camera and at least one usable wrist camera.")
        images = {
            "cam_high": ensure_chw_uint8(high),
            "cam_left_wrist": ensure_chw_uint8(left),
            "cam_right_wrist": ensure_chw_uint8(right),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    high = extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
    left = extract_image(
        observation,
        ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left", "cam_wrist"],
        fallback=high,
    )
    right = extract_image(
        observation,
        ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right", "cam_wrist"],
        fallback=high,
    )
    images = {
        "cam_high": ensure_chw_uint8(high),
        "cam_left_wrist": ensure_chw_uint8(left),
        "cam_right_wrist": ensure_chw_uint8(right),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names, fallback=None):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    if fallback is not None:
        return fallback
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
