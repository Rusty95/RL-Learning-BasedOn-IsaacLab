"""Train an Isaac Lab task with this repo's standalone SAC implementation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


parser = argparse.ArgumentParser(description="Train RLLab Cartpole with standalone SAC.")
parser.add_argument("--task", type=str, default="RLLab-Cartpole-Direct-v0", help="Gymnasium task id.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel Isaac Lab environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of environment-control iterations.")
parser.add_argument("--steps_per_iteration", type=int, default=1, help="Environment steps collected per iteration.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--actor_learning_rate", type=float, default=3.0e-4, help="Actor Adam learning rate.")
parser.add_argument("--critic_learning_rate", type=float, default=3.0e-4, help="Critic Adam learning rate.")
parser.add_argument("--alpha_learning_rate", type=float, default=3.0e-4, help="Temperature Adam learning rate.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--tau", type=float, default=0.005, help="Target critic soft-update coefficient.")
parser.add_argument("--batch_size", type=int, default=4096, help="Replay mini-batch size.")
parser.add_argument("--replay_buffer_size", type=int, default=1_000_000, help="Replay buffer transition capacity.")
parser.add_argument("--initial_random_steps", type=int, default=20_000, help="Global environment steps before policy actions.")
parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per environment step.")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden width for actor and Q networks.")
parser.add_argument("--alpha", type=float, default=0.2, help="Initial/fixed entropy temperature.")
parser.add_argument("--target_entropy", type=float, default=None, help="Target entropy. Defaults to -action_dim.")
parser.add_argument("--automatic_entropy_tuning", action=argparse.BooleanOptionalAction, default=True, help="Learn alpha.")
parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Gradient clipping norm.")
parser.add_argument("--save_interval", type=int, default=250, help="Checkpoint interval in iterations.")
parser.add_argument("--experiment_name", type=str, default="cartpole", help="Experiment folder name.")
parser.add_argument("--run_name", type=str, default="", help="Optional run name suffix.")
parser.add_argument("--deterministic_eval", action=argparse.BooleanOptionalAction, default=False, help="Use mean actions.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between videos in steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import rl_lab_learning.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rl_lab_learning.algorithms.sac import SACConfig, SACTrainer  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
except ImportError:  # pragma: no cover - tensorboard is optional.
    SummaryWriter = None


METRIC_FIELDS = [
    "iteration",
    "global_step",
    "replay_size",
    "mean_reward_100",
    "mean_length_100",
    "actor_loss",
    "critic_loss",
    "alpha_loss",
    "alpha",
    "entropy",
    "q1_mean",
    "q2_mean",
    "target_q_mean",
    "action_std_mean",
    "total_fps",
    "iteration_time",
]


def get_policy_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return obs_dict["policy"]


def build_sac_config(args: argparse.Namespace) -> SACConfig:
    return SACConfig(
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        alpha_learning_rate=args.alpha_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        initial_random_steps=args.initial_random_steps,
        updates_per_step=args.updates_per_step,
        hidden_dim=args.hidden_dim,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        alpha=args.alpha,
        target_entropy=args.target_entropy,
        max_grad_norm=args.max_grad_norm,
    )


def get_action_bounds(env, device: torch.device | str, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    action_space = env.unwrapped.single_action_space
    low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32)
    high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)
    finite = torch.isfinite(low).all() and torch.isfinite(high).all()
    if not finite:
        low = torch.full((action_dim,), -1.0, device=device)
        high = torch.full((action_dim,), 1.0, device=device)
    return low, high


def sample_random_actions(num_envs: int, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    return torch.rand(num_envs, action_low.shape[0], device=action_low.device) * (action_high - action_low) + action_low


def log_tensorboard(
    writer: SummaryWriter | None,
    iteration: int,
    total_time: float,
    global_step: int,
    replay_size: int,
    mean_reward: float,
    mean_length: float,
    stats,
    fps: int,
    iteration_time: float,
) -> None:
    if writer is None or stats is None:
        return

    writer.add_scalar("Loss/actor", stats.actor_loss, iteration)
    writer.add_scalar("Loss/critic", stats.critic_loss, iteration)
    writer.add_scalar("Loss/alpha", stats.alpha_loss, iteration)
    writer.add_scalar("Loss/alpha_value", stats.alpha, iteration)
    writer.add_scalar("Loss/entropy", stats.entropy, iteration)
    writer.add_scalar("Loss/q1_mean", stats.q1_mean, iteration)
    writer.add_scalar("Loss/q2_mean", stats.q2_mean, iteration)
    writer.add_scalar("Loss/target_q_mean", stats.target_q_mean, iteration)
    writer.add_scalar("Policy/mean_noise_std", stats.action_std_mean, iteration)
    writer.add_scalar("Replay/size", replay_size, iteration)
    writer.add_scalar("Perf/total_fps", fps, iteration)
    writer.add_scalar("Perf/iteration_time", iteration_time, iteration)
    writer.add_scalar("Train/mean_reward", mean_reward, iteration)
    writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
    writer.add_scalar("Train/mean_reward/time", mean_reward, total_time)
    writer.add_scalar("Train/mean_episode_length/time", mean_length, total_time)
    writer.add_scalar("Charts/global_step", global_step, iteration)


def empty_stats_row() -> dict[str, float]:
    return {
        "actor_loss": math.nan,
        "critic_loss": math.nan,
        "alpha_loss": math.nan,
        "alpha": math.nan,
        "entropy": math.nan,
        "q1_mean": math.nan,
        "q2_mean": math.nan,
        "target_q_mean": math.nan,
        "action_std_mean": math.nan,
    }


def main() -> None:
    torch.manual_seed(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    log_root = PROJECT_ROOT / "logs" / "standalone" / "sac" / args_cli.experiment_name
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_dir += f"_{args_cli.run_name}"
    log_dir = log_root / run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging standalone SAC run to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(log_dir / "videos" / "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None
    metrics_file = (log_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
    metrics_writer.writeheader()

    try:
        obs_dict, _ = env.reset()
        obs = get_policy_obs(obs_dict)
        device = obs.device
        num_envs = env.unwrapped.num_envs
        obs_dim = obs.shape[-1]
        action_dim = env.unwrapped.single_action_space.shape[0]
        action_low, action_high = get_action_bounds(env, device, action_dim)
        print(
            f"[INFO] num_envs={num_envs}, obs_dim={obs_dim}, action_dim={action_dim}, "
            f"device={device}, action_low={action_low.detach().cpu().tolist()}, action_high={action_high.detach().cpu().tolist()}"
        )

        config = build_sac_config(args_cli)
        trainer = SACTrainer(obs_dim, action_dim, config, device, action_low, action_high)

        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device)
        completed_rewards: list[float] = []
        completed_lengths: list[float] = []
        global_step = 0
        total_time = 0.0

        for iteration in range(args_cli.max_iterations):
            iteration_start_time = time.time()
            last_stats = None

            for _ in range(args_cli.steps_per_iteration):
                if global_step < config.initial_random_steps:
                    actions = sample_random_actions(num_envs, action_low, action_high)
                else:
                    actions = trainer.act(obs, deterministic=args_cli.deterministic_eval)

                next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)
                next_obs = get_policy_obs(next_obs_dict)
                dones = terminated | truncated
                trainer.replay_buffer.add(obs, actions, rewards, next_obs, dones)

                episode_rewards += rewards
                episode_lengths += 1
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    completed_rewards.extend(episode_rewards[done_ids].detach().cpu().tolist())
                    completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())
                    episode_rewards[done_ids] = 0
                    episode_lengths[done_ids] = 0

                obs = next_obs
                global_step += num_envs

                if trainer.replay_buffer.size >= config.batch_size and global_step >= config.initial_random_steps:
                    for _ in range(config.updates_per_step):
                        last_stats = trainer.update()

            mean_reward = sum(completed_rewards[-100:]) / max(1, len(completed_rewards[-100:]))
            mean_length = sum(completed_lengths[-100:]) / max(1, len(completed_lengths[-100:]))
            iteration_time = time.time() - iteration_start_time
            total_time += iteration_time
            fps = int((args_cli.steps_per_iteration * num_envs) / max(iteration_time, 1.0e-8))

            log_tensorboard(
                writer,
                iteration,
                total_time,
                global_step,
                trainer.replay_buffer.size,
                mean_reward,
                mean_length,
                last_stats,
                fps,
                iteration_time,
            )

            stats_row = empty_stats_row()
            if last_stats is not None:
                stats_row.update(
                    {
                        "actor_loss": last_stats.actor_loss,
                        "critic_loss": last_stats.critic_loss,
                        "alpha_loss": last_stats.alpha_loss,
                        "alpha": last_stats.alpha,
                        "entropy": last_stats.entropy,
                        "q1_mean": last_stats.q1_mean,
                        "q2_mean": last_stats.q2_mean,
                        "target_q_mean": last_stats.target_q_mean,
                        "action_std_mean": last_stats.action_std_mean,
                    }
                )
            metrics_writer.writerow(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "replay_size": trainer.replay_buffer.size,
                    "mean_reward_100": mean_reward,
                    "mean_length_100": mean_length,
                    **stats_row,
                    "total_fps": fps,
                    "iteration_time": iteration_time,
                }
            )
            metrics_file.flush()

            if last_stats is None:
                stats_text = "warming_up_replay"
            else:
                stats_text = (
                    f"actor={last_stats.actor_loss:8.4f} critic={last_stats.critic_loss:8.4f} "
                    f"alpha={last_stats.alpha:7.4f} entropy={last_stats.entropy:7.4f} std={last_stats.action_std_mean:6.3f}"
                )
            print(
                f"iter={iteration:04d} step={global_step:09d} replay={trainer.replay_buffer.size:07d} "
                f"reward100={mean_reward:8.3f} len100={mean_length:6.1f} {stats_text}"
            )

            if iteration % args_cli.save_interval == 0:
                trainer.save_checkpoint(log_dir / f"model_{iteration}.pt", iteration, extra={"args": vars(args_cli)})

        final_iteration = max(0, args_cli.max_iterations - 1)
        trainer.save_checkpoint(log_dir / f"model_{final_iteration}.pt", final_iteration, extra={"args": vars(args_cli)})
    finally:
        metrics_file.close()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
