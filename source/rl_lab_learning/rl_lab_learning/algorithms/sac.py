"""Standalone SAC components independent of RSL-RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1.0e-6


@dataclass
class SACConfig:
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    alpha_learning_rate: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_buffer_size: int = 1_000_000
    initial_random_steps: int = 10_000
    updates_per_step: int = 1
    hidden_dim: int = 256
    automatic_entropy_tuning: bool = True
    alpha: float = 0.2
    target_entropy: float | None = None
    max_grad_norm: float = 10.0


@dataclass
class SACBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


@dataclass
class SACUpdateStats:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    action_std_mean: float


class ReplayBuffer:
    """Vectorized replay buffer for off-policy algorithms."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device | str):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch_size = obs.shape[0]
        indices = (torch.arange(batch_size, device=self.device) + self.pos) % self.capacity
        self.obs[indices].copy_(obs)
        self.actions[indices].copy_(actions)
        self.rewards[indices].copy_(rewards)
        self.next_obs[indices].copy_(next_obs)
        self.dones[indices].copy_(dones.float())
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> SACBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay buffer has {self.size} samples, but batch_size={batch_size}.")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return SACBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )


class SquashedGaussianActor(nn.Module):
    """Gaussian actor with tanh squashing and action bounds."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_low: torch.Tensor, action_high: torch.Tensor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias
        log_prob = dist.log_prob(raw_action) - torch.log(self.action_scale * (1.0 - squashed.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action, std

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _, deterministic_action, _ = self.sample(obs)
        return deterministic_action if deterministic else action


class QNetwork(nn.Module):
    """State-action value function Q(s, a)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, actions), dim=-1)).squeeze(-1)


class SACTrainer:
    """Owns SAC networks, optimizers, replay buffer, and update logic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: SACConfig,
        device: torch.device | str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ):
        self.cfg = config
        self.device = device
        action_low = action_low.to(device=device, dtype=torch.float32)
        action_high = action_high.to(device=device, dtype=torch.float32)

        self.actor = SquashedGaussianActor(obs_dim, action_dim, config.hidden_dim, action_low, action_high).to(device)
        self.q1 = QNetwork(obs_dim, action_dim, config.hidden_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim, config.hidden_dim).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim, config.hidden_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim, config.hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.critic_learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.critic_learning_rate)

        target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_dim)
        self.target_entropy = torch.tensor(target_entropy, device=device)
        if config.automatic_entropy_tuning:
            self.log_alpha = torch.tensor(config.alpha, device=device).log().detach().clone().requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)
        else:
            self.log_alpha = torch.tensor(config.alpha, device=device).log()
            self.alpha_optimizer = None

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size, obs_dim, action_dim, device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor.act(obs, deterministic=deterministic)

    def update(self) -> SACUpdateStats:
        batch = self.replay_buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.actor.sample(batch.next_obs)
            q1_next = self.q1_target(batch.next_obs, next_actions)
            q2_next = self.q2_target(batch.next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
            target_q = batch.rewards + self.cfg.gamma * (1.0 - batch.dones) * q_next

        q1 = self.q1(batch.obs, batch.actions)
        q2 = self.q2(batch.obs, batch.actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.max_grad_norm)
        self.q2_optimizer.step()

        new_actions, log_probs, _, std = self.actor.sample(batch.obs)
        q1_new = self.q1(batch.obs, new_actions)
        q2_new = self.q2(batch.obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        if self.cfg.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return SACUpdateStats(
            actor_loss=actor_loss.item(),
            critic_loss=(q1_loss + q2_loss).item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
            entropy=(-log_probs).mean().item(),
            q1_mean=q1.mean().item(),
            q2_mean=q2.mean().item(),
            target_q_mean=target_q.mean().item(),
            action_std_mean=std.mean().item(),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.cfg.tau)
                target_param.data.add_(self.cfg.tau * source_param.data)

    def save_checkpoint(self, path: Path, step: int, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "q1_optimizer_state_dict": self.q1_optimizer.state_dict(),
                "q2_optimizer_state_dict": self.q2_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "config": self.cfg.__dict__,
                "extra": extra or {},
            },
            path,
        )
