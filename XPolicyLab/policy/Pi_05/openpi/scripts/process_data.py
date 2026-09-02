import argparse
import dataclasses
import os
from pathlib import Path
import random
import shutil
from typing import Any, Literal

import h5py
import numpy as np
from tqdm import tqdm
from XPolicyLab.utils.load_file import load_json
from XPolicyLab.utils.load_file import load_yaml
from XPolicyLab.utils.process_data import decode_image_bit

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError as exc:
    if exc.name != "lerobot.datasets":
        raise
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT_PATH = Path(__file__).resolve().parents[5]

CAMERA_ALIASES = {
    "cam_head": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    robot_action_dim_info: dict,
) -> LeRobotDataset:
    motors = [
        *[f"left_{i}" for i in range(robot_action_dim_info["arm_dim"][0])],
        *[f"left_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][0])],
        *[f"right_{i}" for i in range(robot_action_dim_info["arm_dim"][1])],
        *[f"right_ee_{i}" for i in range(robot_action_dim_info["ee_dim"][1])]
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
    }

    for camera_name in CAMERA_ALIASES.values():
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def _load_compressed_images(group: h5py.Group, key: str) -> np.ndarray:
    return np.asarray(decode_image_bit(group[key]))

def _make_action_from_state(state: np.ndarray) -> np.ndarray:
    action = np.empty_like(state, dtype=np.float32)
    if len(state) == 1:
        action[0] = state[0]
        return action

    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action

def load_data(ep_path) -> dict[str, Any]:
    with h5py.File(ep_path, "r") as ep:
        right_state = np.concatenate(
            [ep["state/right_arm_joint_states"][:], ep["state/right_ee_joint_states"][:][:, None]],
            axis=1,
        )
        left_state = np.concatenate(
            [ep["state/left_arm_joint_states"][:], ep["state/left_ee_joint_states"][:][:, None]],
            axis=1,
        )
        state = np.concatenate([left_state, right_state], axis=1).astype(np.float32)
        action = _make_action_from_state(state)

        images = {}
        for source_name, output_name in CAMERA_ALIASES.items():
            if source_name in ep["vision"]:
                images[output_name] = _load_compressed_images(ep["vision"][source_name], "colors")
        instructions = None
        if "instructions" in ep:
            raw_instructions = np.atleast_1d(ep["instructions"][()]).reshape(-1)
            instructions = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in raw_instructions
            ]

    return {
        "images": images,
        "state": state,
        "action": action,
        "velocity": None,
        "effort": None,
        "timestamps": None,
        "instructions": instructions,
    }

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="Dataset bench name (e.g., RoboDojo)")
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("action_type", type=str, help="Action type for artifact naming (e.g., joint)")
    parser.add_argument(
        "expert_data_num",
        type=str,
        nargs="?",
        default=None,
        help="Optional number of episodes to process; non-numeric values are treated as raw_task_dirs.",
    )
    parser.add_argument(
        "raw_task_dirs",
        type=str,
        nargs="?",
        default=None,
        help="Optional raw task dir or comma-separated dirs under data/<bench>/; defaults to ckpt_name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Whether to store images as videos or individual image files",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Do your job.",
        help="Default instruction when not present in HDF5",
    )
    args = parser.parse_args()

    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type
    repo_id = f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    mode = args.mode
    instruction = args.instruction
    expert_data_num = None
    raw_task_dirs_arg = args.raw_task_dirs
    if args.expert_data_num is not None:
        try:
            expert_data_num = int(args.expert_data_num)
        except ValueError:
            if args.raw_task_dirs is not None:
                raise ValueError("raw_task_dirs was provided twice.") from None
            raw_task_dirs_arg = args.expert_data_num
    raw_task_dirs = [item.strip() for item in (raw_task_dirs_arg or ckpt_name).split(",") if item.strip()]

    env_cfg = load_yaml(os.path.join(ROOT_PATH, "./env_cfg", f"{env_cfg_type}.yml"))
    robot_type = env_cfg["config"]["robot"]

    robot_action_dim_info = load_json(os.path.join(ROOT_PATH, "env_cfg/robot", "_robot_info.json"))[robot_type]

    episode_files = []
    for raw_task_dir in raw_task_dirs:
        load_data_dir = Path(ROOT_PATH) / "data" / str(bench_name) / raw_task_dir / str(env_cfg_type)
        task_episode_files = sorted(load_data_dir.glob("data/episode_*.hdf5"))
        if not task_episode_files:
            task_episode_files = sorted(load_data_dir.glob("*.hdf5"))
        episode_files.extend(task_episode_files)
    if expert_data_num is not None:
        episode_files = episode_files[:expert_data_num]
    if not episode_files:
        searched = ", ".join(
            str(Path(ROOT_PATH) / "data" / str(bench_name) / task / str(env_cfg_type))
            for task in raw_task_dirs
        )
        raise FileNotFoundError(f"No HDF5 episodes found for task data under: {searched}")

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=50, # pi default to 50
        mode=mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
        robot_action_dim_info=robot_action_dim_info,
    )

    processed_episodes = 0
    for ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file)
            num_frames = data["state"].shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "task": instruction if data["instructions"] is None else random.choice(data["instructions"]),
                }
                for camera_name, images in data["images"].items():
                    frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame)

            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()
            processed_episodes += 1
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")
    if processed_episodes == 0:
        raise RuntimeError("All selected HDF5 episodes failed during Pi0.5 data conversion.")

if __name__ == "__main__":
    main()
