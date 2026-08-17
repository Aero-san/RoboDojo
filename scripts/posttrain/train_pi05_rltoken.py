"""Train a Pi0.5 residual RL Token actor from a WCM replay buffer.

The frozen Pi0.5 policy supplies ``reference_action`` at deployment. Training
uses the same reference stored in rollout buffers, while ``action`` remains
the behavior/demo target and the action consumed by WCM.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
WCM_ROOT = ROOT_DIR / "external_dependencies" / "WCM"
POSTTRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WCM_ROOT))
sys.path.insert(0, str(POSTTRAIN_DIR))

from progress import BAR_FORMAT, progress_enabled  # noqa: E402
from robodojo_dataset import load_robodojo_dataset  # noqa: E402
from world_critic.data import (  # noqa: E402
    WorldCriticCollator,
    build_datasets,
    build_processor,
    infer_feature_dim,
    task_for_sample,
)
from world_critic.distributed import (  # noqa: E402
    DistributedContext,
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from world_critic.model import WorldCriticModel  # noqa: E402
from world_critic.training import config_from_checkpoint_payload  # noqa: E402

from XPolicyLab.policy.Pi_05.posttrain.rl_token import (  # noqa: E402
    RLTokenActor,
    RLTokenConfig,
    RLTokenEncoderDecoder,
)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def _load_payload(path: str, label: str) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} checkpoint is not a dictionary: {resolved}")
    return payload


def _checkpoint_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    values = payload.get("config", payload.get("rl_token_config", {}))
    return dict(values) if isinstance(values, dict) else {}


class ActorWindowDataset(Dataset):
    """Causal WCM history followed by a future actor action chunk."""

    def __init__(self, dataset: Any, config: Any, episode_ids: list[int], chunk_steps: int):
        self.dataset = dataset
        self.config = config
        self.chunk_steps = chunk_steps
        self.current_rows: list[int] = []
        episode_by_row = np.asarray(dataset._row_episode, dtype=np.int64)
        for episode in sorted(map(int, episode_ids)):
            rows = np.flatnonzero(episode_by_row == episode)
            if len(rows) < config.history_size + chunk_steps - 1:
                continue
            first_current = config.history_size - 1
            last_current = len(rows) - chunk_steps
            self.current_rows.extend(map(int, rows[first_current : last_current + 1]))

    def __len__(self) -> int:
        return len(self.current_rows)

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_action:
            return values
        mean = torch.as_tensor(self.config.action_mean, dtype=values.dtype)
        std = torch.as_tensor(self.config.action_std, dtype=values.dtype)
        return (values - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        current = self.current_rows[index]
        history_rows = range(current - self.config.history_size + 1, current + 1)
        action_rows = range(current, current + self.chunk_steps)
        history = [self.dataset[row] for row in history_rows]
        future = [self.dataset[row] for row in action_rows]
        episode = int(future[0]["episode_index"])
        if any(int(sample["episode_index"]) != episode for sample in [*history, *future]):
            raise RuntimeError("Actor window crossed an episode boundary.")
        instruction = task_for_sample(self.dataset, future[0])
        actions = torch.stack(
            [torch.as_tensor(sample["action"], dtype=torch.float32).reshape(-1) for sample in future]
        )
        references = torch.stack(
            [
                torch.as_tensor(sample["reference_action"], dtype=torch.float32).reshape(-1)
                for sample in future
            ]
        )
        state_key = self.config.state_key
        state_value = future[0][state_key] if state_key else future[0]["action"]
        return {
            "images": [[sample[key] for key in self.config.image_keys] for sample in history],
            "actions": self._normalize(actions),
            "reference_actions": self._normalize(references),
            "current_state_vector": torch.as_tensor(state_value, dtype=torch.float32).reshape(-1),
            "return_targets": torch.stack(
                [torch.as_tensor(sample[self.config.return_key], dtype=torch.float32).reshape(1) for sample in future]
            ),
            "instruction": instruction,
            "valid_mask": torch.ones(self.config.history_size, dtype=torch.bool),
            "episode_id": episode,
            "frame_indices": torch.as_tensor(
                [int(sample["frame_index"]) for sample in future], dtype=torch.long
            ),
            "sample_id": f"{episode}:{int(future[0]['frame_index'])}",
        }


class ActorCollator(WorldCriticCollator):
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().__call__(samples)
        output["reference_actions"] = torch.stack(
            [sample["reference_actions"] for sample in samples]
        )
        output["current_state_vector"] = torch.stack(
            [sample["current_state_vector"] for sample in samples]
        )
        return output


def _load_wcm_checkpoint(path: str | Path) -> tuple[Any, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") not in {"deploy", "full_resume"}:
        raise ValueError("--wcm-checkpoint must be an official WCM deploy.pt, best.pt, or last.pt.")
    return config_from_checkpoint_payload(payload), payload


def _wcm_rollout_values(
    model: WorldCriticModel,
    images: torch.Tensor,
    actions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    history_latents = model.pool_views(model.vision_encoder(images))
    text_tokens, text_mask = model.language_encoder(input_ids, attention_mask)
    latents = history_latents
    values = []
    for step in range(actions.size(1)):
        history = latents[:, -model.config.max_history :]
        valid = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        context = model.encode_context(history, text_tokens, text_mask, valid)
        values.append(model.value_head(context[:, -1:]))
        latents = torch.cat(
            (
                latents,
                model.dynamics(
                    current_state_latent=history[:, -1:],
                    context=context[:, -1:],
                    actions=actions[:, step : step + 1],
                ),
            ),
            dim=1,
        )
    return torch.cat(values, dim=1).squeeze(-1)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcm-checkpoint", required=True)
    parser.add_argument("--dataset-root", default="data/RoboDojo_lerobot_v21_video")
    parser.add_argument("--task", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", choices=("wcm_actor", "rltoken"), default="wcm_actor")
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=2048)
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-layers", type=int, default=6)
    parser.add_argument(
        "--actor-mode",
        choices=("direct", "residual"),
        default="direct",
        help=(
            "Fresh-actor parameterization. 'direct' behavior-clones the target action; "
            "'residual' starts as an exact reference-action pass-through and therefore "
            "requires successful samples whose action differs from reference_action."
        ),
    )
    parser.add_argument("--fixed-std", type=float, default=0.04)
    parser.add_argument("--action-low", type=float, default=-5.0)
    parser.add_argument("--action-high", type=float, default=5.0)
    parser.add_argument("--bc-weight", type=float, default=0.25)
    parser.add_argument("--baseline-loss-penalty", type=float, default=1.0)
    parser.add_argument("--wcm-value-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--encoder-lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--encoder-warmup-steps", type=int, default=0)
    parser.add_argument("--bc-init-steps", type=int, default=0)
    parser.add_argument("--train-encoder-with-actor", action="store_true")
    parser.add_argument("--encoder-checkpoint", default="")
    parser.add_argument("--bc-checkpoint", default="")
    parser.add_argument("--encoder-resume", default="")
    parser.add_argument("--bc-resume", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers per process.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3072)
    return parser


def _make_loader(
    dataset: Dataset,
    collator: ActorCollator,
    args: argparse.Namespace,
    ctx: DistributedContext,
    *,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if ctx.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("Actor dataset has no complete causal windows.")
    return loader, sampler


def _infinite_batches(
    loader: DataLoader,
    sampler: DistributedSampler | None,
) -> Iterator[dict[str, Any]]:
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _reconstruction_loss(reconstruction: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return (reconstruction[:, 0] - state.detach()).square().mean()


def _successful_action_stats(
    dataset: Any,
    config: Any,
    episode_ids: list[int],
    *,
    max_rows: int = 8192,
) -> tuple[float, float, float]:
    """Measure BC supervision without decoding any videos."""

    rows = np.flatnonzero(np.isin(dataset._row_episode, np.asarray(episode_ids, dtype=np.int64)))
    if len(rows) > max_rows:
        rows = rows[np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)]
    actions = np.stack([dataset._row_action[int(row)] for row in rows]).astype(np.float32)
    references = np.stack(
        [dataset._row_reference_action[int(row)] for row in rows]
    ).astype(np.float32)
    if config.normalize_action:
        mean = np.asarray(config.action_mean, dtype=np.float32)
        std = np.asarray(config.action_std, dtype=np.float32)
        actions = (actions - mean) / std
        references = (references - mean) / std
    return (
        float(np.mean(np.square(actions - references))),
        float(np.mean(np.abs(actions))),
        float(np.mean(np.abs(references))),
    )


def _save_checkpoint(
    path: Path,
    *,
    artifact_type: str,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    config: dict[str, Any],
    total_step: int,
    encoder_steps: int,
    bc_steps: int,
    actor_optimizer: torch.optim.Optimizer | None = None,
    encoder_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": artifact_type,
        "encoder": _unwrap(encoder).state_dict(),
        "actor": _unwrap(actor).state_dict(),
        "config": config,
        "step": int(total_step),
        "encoder_steps": int(encoder_steps),
        "bc_steps": int(bc_steps),
    }
    if actor_optimizer is not None:
        payload["actor_optimizer"] = actor_optimizer.state_dict()
    if encoder_optimizer is not None:
        payload["encoder_optimizer"] = encoder_optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.action_low >= args.action_high:
        raise ValueError("--action-low must be smaller than --action-high.")
    for name in ("epochs", "batch_size", "chunk_steps", "expected_world_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("max_steps", "encoder_warmup_steps", "bc_init_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    for name in (
        "bc_weight",
        "baseline_loss_penalty",
        "wcm_value_weight",
        "reconstruction_weight",
        "lr",
        "encoder_lr",
        "weight_decay",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.resume and (args.encoder_resume or args.bc_resume):
        raise ValueError("Use --resume or standalone --encoder-resume/--bc-resume, not both.")


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size > 1:
        if args.device != "cuda":
            raise ValueError("Distributed actor training requires --device cuda.")
        ctx = initialize_distributed(args.expected_world_size)
        device = ctx.device
    else:
        if args.expected_world_size != 1:
            raise RuntimeError(
                f"Expected world size {args.expected_world_size}, but training was not launched with torchrun."
            )
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, device=device)
    try:
        _run(args, ctx, device)
    finally:
        cleanup_distributed(ctx)


def _run(args: argparse.Namespace, ctx: DistributedContext, device: torch.device) -> None:
    torch.manual_seed(args.seed + ctx.rank)
    task_name = args.task or os.environ.get("WCM_TASK_NAME", "")
    if task_name:
        os.environ["WCM_TASK_NAME"] = task_name
    wcm_config, wcm_payload = _load_wcm_checkpoint(args.wcm_checkpoint)
    wcm_config.data.root = str(Path(args.dataset_root).expanduser().resolve())
    os.environ["WCM_DATASET_ROOT"] = wcm_config.data.root
    os.environ.setdefault("WCM_ASSUME_SUCCESS", "1")
    if task_name:
        wcm_config.data.split_manifest = None

    import world_critic.data as wcm_data

    wcm_data.load_lerobot_dataset = load_robodojo_dataset
    dataset, _unused_train, _unused_val, split = build_datasets(wcm_config.data)
    del _unused_train, _unused_val
    action_dim = infer_feature_dim(dataset, wcm_config.data.action_key)
    state_dim = infer_feature_dim(dataset, wcm_config.data.state_key) if wcm_config.data.state_key else action_dim
    if not 1 <= args.chunk_steps:
        raise ValueError("--chunk-steps must be positive.")

    full_resume = _load_payload(args.resume, "Full actor resume")
    encoder_resume = _load_payload(args.encoder_resume, "Encoder resume")
    bc_resume = _load_payload(args.bc_resume, "BC actor resume")
    # Encoder-only artifacts historically carried a snapshot of the actor
    # config because both modules shared one dataclass.  Those actor fields do
    # not describe an actor being resumed and must not override fresh-actor
    # arguments such as --actor-mode.  Conversely, a full/BC actor resume must
    # preserve its parameterization for state-dict and optimizer consistency.
    encoder_source_config = _checkpoint_config(
        full_resume or encoder_resume or bc_resume
    )
    actor_source_config = _checkpoint_config(full_resume or bc_resume)

    def saved(config: dict[str, Any], name: str, default: Any) -> Any:
        value = config.get(name, default)
        return default if value is None else value

    token_config = RLTokenConfig(
        token_dim=int(saved(encoder_source_config, "token_dim", args.token_dim)),
        num_layers=int(
            saved(encoder_source_config, "num_layers", RLTokenConfig.num_layers)
        ),
        num_heads=int(
            saved(encoder_source_config, "num_heads", RLTokenConfig.num_heads)
        ),
        decoder_layers=int(
            saved(
                encoder_source_config,
                "decoder_layers",
                RLTokenConfig.decoder_layers,
            )
        ),
        mlp_dim=int(saved(encoder_source_config, "mlp_dim", RLTokenConfig.mlp_dim)),
        actor_hidden_dim=int(
            saved(actor_source_config, "actor_hidden_dim", args.actor_hidden_dim)
        ),
        actor_layers=int(
            saved(actor_source_config, "actor_layers", args.actor_layers)
        ),
        fixed_std=float(saved(actor_source_config, "fixed_std", args.fixed_std)),
        action_low=float(saved(actor_source_config, "action_low", args.action_low)),
        action_high=float(saved(actor_source_config, "action_high", args.action_high)),
        reference_dropout=float(saved(actor_source_config, "reference_dropout", 0.0)),
        actor_residual=bool(
            saved(
                actor_source_config,
                "actor_residual",
                args.actor_mode == "residual",
            )
        ),
        max_sequence_length=int(
            saved(
                encoder_source_config,
                "max_sequence_length",
                RLTokenConfig.max_sequence_length,
            )
        ),
    )
    saved_chunk_steps = actor_source_config.get("chunk_steps")
    if saved_chunk_steps is not None and int(saved_chunk_steps) != args.chunk_steps:
        raise ValueError(
            f"Resume checkpoint chunk_steps={saved_chunk_steps}, requested={args.chunk_steps}."
        )
    if (
        "state_dim" in encoder_source_config
        and int(encoder_source_config["state_dim"]) != state_dim
    ):
        raise ValueError(
            f"Encoder checkpoint state_dim={encoder_source_config['state_dim']}, dataset={state_dim}."
        )
    for name, current in (("action_dim", action_dim), ("state_dim", state_dim)):
        if name in actor_source_config and int(actor_source_config[name]) != current:
            raise ValueError(
                f"Actor checkpoint {name}={actor_source_config[name]}, dataset={current}."
            )
    if ctx.is_main:
        actor_mode = "residual" if token_config.actor_residual else "direct"
        actor_source = "full resume" if full_resume else "BC resume" if bc_resume else "fresh"
        print(f"[RLToken init] actor_mode={actor_mode} actor_source={actor_source}")

    encoder = RLTokenEncoderDecoder(state_dim, token_config).to(device)
    actor = RLTokenActor(
        token_config.token_dim,
        state_dim,
        (args.chunk_steps, action_dim),
        token_config,
    ).to(device)
    if full_resume is not None:
        encoder.load_state_dict(full_resume["encoder"], strict=True)
        actor.load_state_dict(full_resume["actor"], strict=True)
    else:
        if encoder_resume is not None:
            encoder.load_state_dict(encoder_resume["encoder"], strict=True)
        elif bc_resume is not None:
            if "encoder" not in bc_resume:
                raise ValueError(
                    "--bc-resume does not contain its paired encoder; also pass --encoder-resume."
                )
            encoder.load_state_dict(bc_resume["encoder"], strict=True)
        if bc_resume is not None:
            actor.load_state_dict(bc_resume["actor"], strict=True)

    for name, current in (
        ("action_mean", wcm_config.data.action_mean),
        ("action_std", wcm_config.data.action_std),
    ):
        saved_stats = actor_source_config.get(name)
        if saved_stats is None:
            continue
        saved_array = np.asarray(saved_stats, dtype=np.float32)
        current_array = np.asarray(current, dtype=np.float32)
        if saved_array.shape != current_array.shape or not np.allclose(
            saved_array, current_array, rtol=1.0e-5, atol=1.0e-6
        ):
            raise ValueError(
                f"Resume checkpoint {name} does not match the current WCM. "
                "Resume the paired WCM checkpoint so actor coordinates remain stable."
            )

    if ctx.distributed:
        encoder = DistributedDataParallel(encoder, device_ids=[device.index])
        actor = DistributedDataParallel(actor, device_ids=[device.index])
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if full_resume is not None:
        if full_resume.get("encoder_optimizer"):
            encoder_optimizer.load_state_dict(full_resume["encoder_optimizer"])
        if full_resume.get("actor_optimizer"):
            actor_optimizer.load_state_dict(full_resume["actor_optimizer"])
        elif full_resume.get("optimizer"):
            actor_optimizer.load_state_dict(full_resume["optimizer"])
    elif encoder_resume is not None and encoder_resume.get("encoder_optimizer"):
        encoder_optimizer.load_state_dict(encoder_resume["encoder_optimizer"])
    if full_resume is None and bc_resume is not None and bc_resume.get("actor_optimizer"):
        actor_optimizer.load_state_dict(bc_resume["actor_optimizer"])

    prior_encoder_steps = int(
        (full_resume or encoder_resume or bc_resume or {}).get("encoder_steps", 0)
    )
    prior_bc_steps = int((full_resume or bc_resume or {}).get("bc_steps", 0))
    warmup_steps = 0 if full_resume is not None else max(args.encoder_warmup_steps - prior_encoder_steps, 0)
    bc_init_steps = 0 if full_resume is not None else max(args.bc_init_steps - prior_bc_steps, 0)
    total_step = int((full_resume or {}).get("step", 0))

    successful_episodes = [
        episode
        for episode in split.train
        if bool(dataset._success_rows[np.flatnonzero(dataset._row_episode == episode)[0]])
    ]
    actor_dataset = ActorWindowDataset(dataset, wcm_config.data, split.train, args.chunk_steps)
    success_dataset = ActorWindowDataset(
        dataset,
        wcm_config.data,
        successful_episodes,
        args.chunk_steps,
    )
    if len(success_dataset) == 0:
        raise ValueError("The replay buffer has no successful actor windows for BC initialization.")
    if bc_init_steps > 0:
        target_reference_mse, target_abs_mean, reference_abs_mean = _successful_action_stats(
            dataset,
            wcm_config.data,
            successful_episodes,
        )
        if token_config.actor_residual and target_reference_mse <= 1.0e-12:
            raise ValueError(
                "Residual actor BC has zero supervision: every successful target action equals "
                "reference_action, while residual actors start as an exact reference pass-through. "
                "Use --actor-mode direct for SFT-only initialization, or provide successful rollout "
                "samples whose executed action differs from the Pi0.5 reference action."
            )
        if ctx.is_main:
            actor_mode = "residual" if token_config.actor_residual else "direct"
            print(
                f"[RLToken BC] actor_mode={actor_mode} "
                f"target_reference_mse={target_reference_mse:.3e} "
                f"target_abs_mean={target_abs_mean:.3e} "
                f"reference_abs_mean={reference_abs_mean:.3e}"
            )
    collator = ActorCollator(
        build_processor(wcm_config.model),
        wcm_config.model.vision.image_size,
        wcm_config.model.language.max_length,
    )
    actor_loader, actor_sampler = _make_loader(
        actor_dataset, collator, args, ctx, seed=args.seed
    )
    success_loader, success_sampler = _make_loader(
        success_dataset, collator, args, ctx, seed=args.seed + 1
    )

    success_batches = _infinite_batches(success_loader, success_sampler)
    show_progress = ctx.is_main and progress_enabled()
    encoder.train()
    encoder_progress = tqdm(
        range(warmup_steps),
        desc="RLToken encoder warmup",
        unit="step",
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        disable=not show_progress or warmup_steps == 0,
    )
    for _ in encoder_progress:
        batch = next(success_batches)
        state = batch["current_state_vector"].to(device)
        _, reconstruction = encoder(state[:, None, :])
        loss = _reconstruction_loss(reconstruction, state)
        encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        encoder_optimizer.step()
        if show_progress:
            encoder_progress.set_postfix(loss=f"{loss.item():.3e}", refresh=False)
    encoder_progress.close()
    encoder_steps = prior_encoder_steps + warmup_steps

    if not args.train_encoder_with_actor:
        _unwrap(encoder).freeze()
    else:
        encoder.train()
    actor.train()
    bc_progress = tqdm(
        range(bc_init_steps),
        desc="RLToken actor BC",
        unit="step",
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        disable=not show_progress or bc_init_steps == 0,
    )
    for _ in bc_progress:
        batch = next(success_batches)
        state = batch["current_state_vector"].to(device)
        reference = batch["reference_actions"].to(device)
        target = batch["actions"].to(device)
        with torch.set_grad_enabled(args.train_encoder_with_actor):
            token, reconstruction = encoder(state[:, None, :])
        candidate = actor(token, state, reference)
        action_loss = (candidate - target).square().mean()
        baseline_loss = (reference - target).square().mean()
        loss = action_loss + args.baseline_loss_penalty * (
            action_loss - baseline_loss.detach()
        ).clamp_min(0).square()
        if args.train_encoder_with_actor:
            loss = loss + args.reconstruction_weight * _reconstruction_loss(reconstruction, state)
            encoder_optimizer.zero_grad(set_to_none=True)
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()
        if args.train_encoder_with_actor:
            encoder_optimizer.step()
        if show_progress:
            bc_progress.set_postfix(
                loss=f"{loss.item():.3e}",
                action=f"{action_loss.item():.3e}",
                ref_mse=f"{baseline_loss.item():.3e}",
                refresh=False,
            )
    bc_progress.close()
    bc_steps = prior_bc_steps + bc_init_steps

    auxiliary_config = asdict(token_config)
    auxiliary_config.update(
        {
            "action_dim": action_dim,
            "state_dim": state_dim,
            "chunk_steps": args.chunk_steps,
            "action_mean": list(map(float, wcm_config.data.action_mean)),
            "action_std": list(map(float, wcm_config.data.action_std)),
            "task": task_name,
        }
    )

    if ctx.is_main and args.encoder_checkpoint and warmup_steps > 0:
        _save_checkpoint(
            Path(args.encoder_checkpoint).expanduser(),
            artifact_type="robodojo_pi05_rltoken_encoder_v1",
            encoder=encoder,
            actor=actor,
            config=auxiliary_config,
            total_step=total_step,
            encoder_steps=encoder_steps,
            bc_steps=bc_steps,
            encoder_optimizer=encoder_optimizer,
        )
    if ctx.is_main and args.bc_checkpoint and (bc_init_steps > 0 or bc_resume is not None):
        _save_checkpoint(
            Path(args.bc_checkpoint).expanduser(),
            artifact_type="robodojo_pi05_rltoken_bc_v1",
            encoder=encoder,
            actor=actor,
            config=auxiliary_config,
            total_step=total_step,
            encoder_steps=encoder_steps,
            bc_steps=bc_steps,
            actor_optimizer=actor_optimizer,
        )

    wcm_model = None
    if args.objective == "wcm_actor":
        wcm_model = WorldCriticModel(wcm_config.model).to(device).eval()
        wcm_model.load_state_dict(wcm_payload["model"], strict=True)
        for parameter in wcm_model.parameters():
            parameter.requires_grad_(False)

    actor_updates = 0
    planned_actor_updates = args.epochs * len(actor_loader)
    if args.max_steps > 0:
        planned_actor_updates = min(planned_actor_updates, args.max_steps)
    actor_progress = tqdm(
        total=planned_actor_updates,
        desc=f"RLToken {args.objective}",
        unit="step",
        file=sys.stdout,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
        disable=not show_progress,
    )
    for epoch in range(args.epochs):
        if actor_sampler is not None:
            actor_sampler.set_epoch(epoch)
        for batch in actor_loader:
            state = batch["current_state_vector"].to(device)
            reference = batch["reference_actions"].to(device)
            target = batch["actions"].to(device)
            with torch.set_grad_enabled(args.train_encoder_with_actor):
                token, reconstruction = encoder(state[:, None, :])
            candidate = actor(token, state, reference)
            if args.objective == "wcm_actor":
                assert wcm_model is not None
                bc_loss = (candidate - reference).square().mean()
                rollout_action = torch.cat((candidate, candidate[:, -1:].detach()), dim=1)
                values = _wcm_rollout_values(
                    wcm_model,
                    batch["images"].to(device),
                    rollout_action,
                    batch["instruction_input_ids"].to(device),
                    batch["instruction_attention_mask"].to(device),
                )
                wcm_value = values[:, -1].mean()
                loss = -args.wcm_value_weight * wcm_value + args.bc_weight * bc_loss
            else:
                bc_loss = (candidate - target).square().mean()
                targets = batch["return_targets"][:, :, 0].to(device)
                weights = torch.softmax(targets.mean(dim=1), dim=0).detach() * targets.size(0)
                loss = (weights[:, None, None] * (candidate - target).square()).mean()
                wcm_value = targets.mean()
            if args.train_encoder_with_actor:
                loss = loss + args.reconstruction_weight * _reconstruction_loss(reconstruction, state)
                encoder_optimizer.zero_grad(set_to_none=True)
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            if args.train_encoder_with_actor:
                encoder_optimizer.step()
            actor_updates += 1
            total_step += 1
            actor_progress.update(1)
            if show_progress:
                actor_progress.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    loss=f"{loss.item():.3e}",
                    bc=f"{bc_loss.item():.3e}",
                    value=f"{wcm_value.item():.3e}",
                    refresh=False,
                )
            if args.max_steps > 0 and actor_updates >= args.max_steps:
                break
        if args.max_steps > 0 and actor_updates >= args.max_steps:
            break
    actor_progress.close()
    config = asdict(token_config)
    config.update(
        {
            "action_dim": action_dim,
            "state_dim": state_dim,
            "chunk_steps": args.chunk_steps,
            "action_mean": list(map(float, wcm_config.data.action_mean)),
            "action_std": list(map(float, wcm_config.data.action_std)),
            "wcm_checkpoint": str(Path(args.wcm_checkpoint).expanduser().resolve()),
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "task": task_name,
            "objective": args.objective,
        }
    )
    artifact_type = (
        "robodojo_pi05_wcm_actor_v1"
        if args.objective == "wcm_actor"
        else "robodojo_pi05_rltoken_v1"
    )
    barrier(ctx)
    output_path = Path(args.output).expanduser()
    if ctx.is_main:
        _save_checkpoint(
            output_path,
            artifact_type=artifact_type,
            encoder=encoder,
            actor=actor,
            config=config,
            total_step=total_step,
            encoder_steps=encoder_steps,
            bc_steps=bc_steps,
            actor_optimizer=actor_optimizer,
            encoder_optimizer=encoder_optimizer,
        )
        print(f"saved {artifact_type}: {output_path}")
    barrier(ctx)


if __name__ == "__main__":
    main(_make_parser().parse_args())
