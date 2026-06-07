"""Standalone PPO components independent of RSL-RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class PPOConfig:
    learning_rate: float = 1.0e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 1.0
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.01
    schedule: str = "adaptive"
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-2
    max_grad_norm: float = 1.0
    update_epochs: int = 5
    num_mini_batches: int = 4
    hidden_dim: int = 32
    min_log_std: float = -5.0
    max_log_std: float = 2.0


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float
    action_std_mean: float


class ActorCritic(nn.Module):
    """Small Gaussian actor-critic for continuous-control tasks."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.value(obs)
        return actions, log_probs, values

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value(obs)
        return log_probs, entropy, values

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class RolloutBuffer:
    """Fixed-size rollout buffer with GAE computation."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device | str):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones.float())
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        advantage = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            next_not_done = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_not_done - self.values[step]
            advantage = delta + gamma * gae_lambda * next_not_done * advantage
            self.advantages[step] = advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1.0e-8)

    def flatten(self) -> RolloutBatch:
        return RolloutBatch(
            obs=self.obs.reshape(-1, self.obs.shape[-1]),
            actions=self.actions.reshape(-1, self.actions.shape[-1]),
            log_probs=self.log_probs.reshape(-1),
            rewards=self.rewards.reshape(-1),
            dones=self.dones.reshape(-1),
            values=self.values.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=self.advantages.reshape(-1),
        )

    def clear(self) -> None:
        self.step = 0


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    variance = torch.var(y_true)
    if variance == 0:
        return torch.tensor(0.0, device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (variance + 1.0e-8)


class PPOTrainer:
    """Owns PPO update logic and policy checkpointing."""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, device: torch.device | str):
        self.cfg = config
        self.policy = ActorCritic(obs_dim, action_dim, config.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.act(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy.value(obs)

    def update(self, batch: RolloutBatch) -> PPOUpdateStats:
        if self.cfg.update_epochs < 1:
            raise ValueError("PPOConfig.update_epochs must be at least 1.")
        if self.cfg.num_mini_batches < 1:
            raise ValueError("PPOConfig.num_mini_batches must be at least 1.")

        batch_size = batch.obs.shape[0]
        mini_batch_size = max(1, batch_size // self.cfg.num_mini_batches)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_fraction_sum = 0.0
        updates = 0

        for _ in range(self.cfg.update_epochs):
            indices = torch.randperm(batch_size, device=batch.obs.device)
            for start in range(0, batch_size, mini_batch_size):
                mb_idx = indices[start : start + mini_batch_size]
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(batch.obs[mb_idx], batch.actions[mb_idx])

                log_ratio = new_log_probs - batch.log_probs[mb_idx]
                ratio = log_ratio.exp()
                unclipped = ratio * batch.advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * batch.advantages[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                if self.cfg.use_clipped_value_loss:
                    value_clipped = batch.values[mb_idx] + (new_values - batch.values[mb_idx]).clamp(
                        -self.cfg.clip_eps, self.cfg.clip_eps
                    )
                    value_losses = (new_values - batch.returns[mb_idx]).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns[mb_idx]).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (batch.returns[mb_idx] - new_values).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean()
                    self._adapt_learning_rate(approx_kl)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.policy.log_std.clamp_(self.cfg.min_log_std, self.cfg.max_log_std)

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy_loss.item()
                approx_kl_sum += approx_kl.item()
                clip_fraction_sum += clip_fraction.item()
                updates += 1

        return PPOUpdateStats(
            policy_loss=policy_loss_sum / updates,
            value_loss=value_loss_sum / updates,
            entropy=entropy_sum / updates,
            approx_kl=approx_kl_sum / updates,
            clip_fraction=clip_fraction_sum / updates,
            explained_variance=explained_variance(batch.values, batch.returns).item(),
            learning_rate=self.optimizer.param_groups[0]["lr"],
            action_std_mean=self.policy.log_std.exp().mean().item(),
        )

    def _adapt_learning_rate(self, approx_kl: torch.Tensor) -> None:
        if self.cfg.schedule != "adaptive" or self.cfg.desired_kl <= 0.0:
            return
        if approx_kl > self.cfg.desired_kl * 2.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = max(self.cfg.min_learning_rate, param_group["lr"] / 1.5)
        elif 0.0 < approx_kl < self.cfg.desired_kl / 2.0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = min(self.cfg.max_learning_rate, param_group["lr"] * 1.5)

    def save_checkpoint(self, path: Path, iteration: int, extra: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
                "extra": extra or {},
            },
            path,
        )
