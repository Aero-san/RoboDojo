"""RL Token (RLT) building blocks for frozen GR00T policies.

This module intentionally has no dependency on a particular VLA.  A VLA only
has to provide a sequence of final-layer embeddings and a reference action
chunk.  This makes the implementation usable with N1.7, while keeping the
online learner small enough for RoboCasa365.

The implementation follows Physical Intelligence's RLT recipe: a
transformer bottleneck readout, a stochastic Gaussian chunk actor conditioned
on the reference chunk, and a clipped/double-Q TD3-style critic.  Rewards are
not shaped here; the environment adapter should provide the sparse terminal
success reward.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class RLTokenConfig:
    token_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    decoder_layers: int = 2
    mlp_dim: int = 1024
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    actor_layers: int = 2
    critic_layers: int = 2
    fixed_std: float = 0.05
    reference_dropout: float = 0.5
    discount: float = 0.99
    tau: float = 0.005
    bc_weight: float = 1.0
    actor_update_period: int = 2
    target_noise_std: float = 0.02
    target_noise_clip: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    # When enabled, preserve the frozen VLA behavior at initialization and
    # learn a bounded correction around its reference action.  This is
    # important for sparse online rewards: a randomly initialized full-action
    # actor would discard any success probability already present in the VLA.
    actor_residual: bool = False
    max_sequence_length: int = 4096


class RLTokenEncoderDecoder(nn.Module):
    """Compress VLA sequence embeddings into one learned RL token.

    ``forward`` returns the token and, optionally, the reconstruction.  The
    reconstruction objective is used only during task-specific adaptation;
    call ``freeze`` before collecting online RL data.
    """

    def __init__(self, input_dim: int, config: RLTokenConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.token = nn.Parameter(torch.randn(1, 1, config.token_dim) * 0.02)
        self.input_projection = nn.Linear(input_dim, config.token_dim)
        enc_layer = nn.TransformerEncoderLayer(
            config.token_dim,
            config.num_heads,
            config.mlp_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, config.num_layers)
        dec_layer = nn.TransformerDecoderLayer(
            config.token_dim,
            config.num_heads,
            config.mlp_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, config.decoder_layers)
        self.output_projection = nn.Linear(config.token_dim, input_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length + 1, config.token_dim)
        )
        self.query = nn.Parameter(torch.randn(1, 1, config.token_dim) * 0.02)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None):
        if features.ndim != 3:
            raise ValueError(f"features must be [B,S,D], got {tuple(features.shape)}")
        # N1.7 inference commonly exposes bf16 backbone features while the
        # lightweight adapter is trained in fp32.
        features = features.float()
        sequence_length = features.shape[1]
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"features sequence length {sequence_length} exceeds "
                f"max_sequence_length={self.config.max_sequence_length}"
            )
        x = self.input_projection(features)
        b = x.shape[0]
        token = self.token.expand(b, -1, -1)
        if padding_mask is not None:
            padding_mask = torch.cat(
                (
                    padding_mask.to(device=x.device),
                    torch.zeros(b, 1, dtype=torch.bool, device=x.device),
                ),
                dim=1,
            )
        encoder_input = torch.cat((x, token), dim=1)
        encoder_input = encoder_input + self.position_embedding[:, : sequence_length + 1]
        encoded = self.encoder(encoder_input, src_key_padding_mask=padding_mask)
        rl_token = encoded[:, -1]
        # Autoregressive teacher forcing: the decoder sees BOS followed by the
        # previous projected input embedding. Without this, repeated queries
        # and no positional encoding make every reconstructed position
        # indistinguishable. The target is frozen VLA output, so stop gradients
        # through the teacher-forcing inputs.
        decoder_input = torch.cat(
            (
                self.query.expand(b, 1, -1),
                x.detach()[:, :-1],
            ),
            dim=1,
        )
        decoder_input = decoder_input + self.position_embedding[:, :sequence_length]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=features.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            decoder_input,
            rl_token[:, None, :],
            tgt_mask=causal_mask,
        )
        reconstruction = self.output_projection(decoded)
        return rl_token, reconstruction

    def reconstruction_loss(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        token, reconstruction = self(features, padding_mask)
        target = features.detach()
        loss = (reconstruction - target).square().mean(dim=-1)
        if padding_mask is not None:
            loss = loss.masked_fill(padding_mask, 0.0)
            return loss.sum() / (~padding_mask).sum().clamp_min(1)
        return loss.mean()

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for i in range(layers):
        blocks += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
    blocks.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*blocks)


class RLTokenActor(nn.Module):
    """Gaussian chunk actor with explicit VLA reference-action pass-through."""

    def __init__(
        self, token_dim: int, proprio_dim: int, action_shape: tuple[int, int], config: RLTokenConfig
    ):
        super().__init__()
        self.action_shape = action_shape
        self.action_size = action_shape[0] * action_shape[1]
        self.config = config
        self.net = _mlp(
            token_dim + proprio_dim + self.action_size,
            self.action_size,
            config.actor_hidden_dim,
            config.actor_layers,
        )
        if config.actor_residual:
            # Start from an exact reference-action pass-through.  The network
            # then learns only a correction, while keeping the same state-dict
            # shape as the non-residual actor for checkpoint compatibility.
            last_layer = self.net[-1]
            assert isinstance(last_layer, nn.Linear)
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def mean(self, token: Tensor, proprio: Tensor, reference: Tensor) -> Tensor:
        ref = reference.reshape(reference.shape[0], -1)
        correction = self.net(torch.cat((token, proprio, ref), dim=-1))
        mean = ref + correction if self.config.actor_residual else correction
        return mean.reshape(-1, *self.action_shape).clamp(
            self.config.action_low, self.config.action_high
        )

    def forward(self, token: Tensor, proprio: Tensor, reference: Tensor) -> Tensor:
        """Return the deterministic mean; this entry point is DDP-compatible."""
        return self.mean(token, proprio, reference)

    def distribution(
        self, token: Tensor, proprio: Tensor, reference: Tensor, *, training: bool = True
    ):
        if training and self.config.reference_dropout > 0:
            dropped = (
                torch.rand(reference.shape[0], device=reference.device)
                < self.config.reference_dropout
            )
            ref = torch.where(dropped[:, None, None], torch.zeros_like(reference), reference)
        else:
            ref = reference
        mean = self.mean(token, proprio, ref)
        std = torch.full_like(mean, self.config.fixed_std)
        return Normal(mean, std), mean

    def sample(
        self,
        token: Tensor,
        proprio: Tensor,
        reference: Tensor,
        *,
        deterministic: bool = False,
        reparameterized: bool = False,
    ):
        dist, mean = self.distribution(token, proprio, reference, training=self.training)
        if deterministic:
            action = mean
        else:
            action = dist.rsample() if reparameterized else dist.sample()
        return action.clamp(self.config.action_low, self.config.action_high)


class TwinQCritic(nn.Module):
    """Twin action-value critics used by RLT's TD3-style update."""

    def __init__(
        self, token_dim: int, proprio_dim: int, action_shape: tuple[int, int], config: RLTokenConfig
    ):
        super().__init__()
        dim = token_dim + proprio_dim + action_shape[0] * action_shape[1]
        self.q1 = _mlp(dim, 1, config.critic_hidden_dim, config.critic_layers)
        self.q2 = _mlp(dim, 1, config.critic_hidden_dim, config.critic_layers)

    def forward(self, token: Tensor, proprio: Tensor, action: Tensor):
        x = torch.cat((token, proprio, action.reshape(action.shape[0], -1)), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


@dataclass
class RLTTransition:
    token: Tensor
    proprio: Tensor
    reference: Tensor
    action: Tensor
    reward: Tensor
    next_token: Tensor
    next_proprio: Tensor
    next_reference: Tensor
    done: Tensor


class RLTReplayBuffer:
    """Small CPU/GPU-agnostic replay buffer; stores chunk-boundary transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[RLTTransition] = []
        self.position = 0

    def add(self, transition: RLTTransition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int, device: torch.device | str) -> RLTTransition:
        if len(self.data) < batch_size:
            raise ValueError(f"replay has {len(self.data)} samples, need {batch_size}")
        batch = random.sample(self.data, batch_size)
        fields = [
            "token",
            "proprio",
            "reference",
            "action",
            "reward",
            "next_token",
            "next_proprio",
            "next_reference",
            "done",
        ]
        return RLTTransition(
            **{name: torch.stack([getattr(x, name) for x in batch]).to(device) for name in fields}
        )


class RLTokenAgent(nn.Module):
    """Online off-policy RLT learner with sparse success rewards."""

    def __init__(
        self,
        token_dim: int,
        proprio_dim: int,
        action_shape: tuple[int, int],
        config: RLTokenConfig | None = None,
    ):
        super().__init__()
        self.config = config or RLTokenConfig()
        self.actor = RLTokenActor(token_dim, proprio_dim, action_shape, self.config)
        self.critic = TwinQCritic(token_dim, proprio_dim, action_shape, self.config)
        self.target_critic = copy.deepcopy(self.critic)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.update_step = 0
        self.chunk_discount = self.config.discount ** action_shape[0]

    @torch.no_grad()
    def act(
        self, token: Tensor, proprio: Tensor, reference: Tensor, deterministic: bool = False
    ) -> Tensor:
        was_training = self.actor.training
        self.actor.eval()
        action = self.actor.sample(token, proprio, reference, deterministic=deterministic)
        self.actor.train(was_training)
        return action

    def update(
        self,
        batch: RLTTransition,
        actor_optimizer,
        critic_optimizer,
        *,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """Perform one critic update and optionally the RL actor update.

        ``update_actor=False`` is used by the diagnostic behavioral-cloning
        mode: the critic still learns from online replay, while the actor is
        updated separately from demonstration video/action pairs.
        """
        c = self.config
        with torch.no_grad():
            # Reference dropout regularizes the actor update, but must not be
            # applied to the target policy: otherwise the TD target randomly
            # replaces the frozen VLA reference with a zero reference.
            actor_was_training = self.actor.training
            self.actor.eval()
            next_action = self.actor.sample(
                batch.next_token, batch.next_proprio, batch.next_reference, deterministic=False
            )
            self.actor.train(actor_was_training)
            noise = (torch.randn_like(next_action) * c.target_noise_std).clamp(
                -c.target_noise_clip, c.target_noise_clip
            )
            next_action = (next_action + noise).clamp(c.action_low, c.action_high)
            tq1, tq2 = self.target_critic(batch.next_token, batch.next_proprio, next_action)
            target = batch.reward + self.chunk_discount * (1.0 - batch.done) * torch.minimum(
                tq1, tq2
            )
        q1, q2 = self.critic(batch.token, batch.proprio, batch.action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()
        self.update_step += 1
        actor_loss_value = torch.zeros((), device=target.device)
        if update_actor and self.update_step % c.actor_update_period == 0:
            action = self.actor.sample(
                batch.token,
                batch.proprio,
                batch.reference,
                deterministic=False,
                reparameterized=True,
            )
            q1_actor, q2_actor = self.critic(batch.token, batch.proprio, action)
            # TD3 uses only Q1 for the actor objective.  In multi-GPU mode the
            # twin critic is wrapped by DDP, which still requires every output
            # branch to participate in the backward graph on this iteration.
            # Keep Q2's gradient exactly zero while retaining its autograd/DDP
            # path; otherwise the next critic update fails with "unfinished
            # reduction" because Q2 was unused in this actor update.
            q1_actor = q1_actor + q2_actor * 0.0
            bc = (action - batch.reference).square().mean()
            actor_loss = -q1_actor.mean() + c.bc_weight * bc
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
            actor_loss_value = actor_loss.detach()
        if self.update_step % c.actor_update_period == 0:
            with torch.no_grad():
                for target_param, param in zip(
                    self.target_critic.parameters(), self.critic.parameters()
                ):
                    target_param.lerp_(param, c.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss_value),
            "q": float(torch.minimum(q1, q2).mean().detach()),
        }

    def update_critic_from_successful_actions(
        self,
        token: Tensor,
        proprio: Tensor,
        action: Tensor,
        critic_optimizer,
        target_return: float | Tensor = 1.0,
    ) -> dict[str, float]:
        """Anchor the critic with actions from known-success demonstrations.

        This is a low-cost critic bootstrap for sparse-reward tasks.  The
        dataset is expected to contain successful demonstrations.  The target
        may be a scalar or a per-sample return-to-go tensor.  Online TD updates
        subsequently replace this approximation with the environment's actual
        terminal reward structure.
        """
        q1, q2 = self.critic(token, proprio, action)
        target = torch.as_tensor(target_return, device=q1.device, dtype=q1.dtype)
        if target.ndim == 0:
            target = target.expand_as(q1)
        else:
            target = target.reshape(-1)
            if target.numel() != q1.numel():
                raise ValueError(
                    "target_return must be scalar or contain one value per demo sample; "
                    f"got {target.numel()} values for {q1.numel()} samples"
                )
            target = target.reshape_as(q1)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(param, self.config.tau)
        return {
            "critic_demo_loss": float(critic_loss.detach()),
            "demo_q": float(q1.detach().mean()),
            "demo_target": float(target.detach().mean()),
        }


def sparse_success_reward(success: bool, done: bool) -> float:
    """RLT's terminal sparse reward: +1 on successful completion, otherwise 0."""
    return 1.0 if bool(success) and bool(done) else 0.0
