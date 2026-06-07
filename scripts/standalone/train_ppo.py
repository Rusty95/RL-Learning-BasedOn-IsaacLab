"""Train an Isaac Lab task with this repo's standalone PPO implementation."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


parser = argparse.ArgumentParser(description="Train RLLab Cartpole with standalone PPO.")
parser.add_argument("--task", type=str, default="RLLab-Cartpole-Direct-v0", help="Gymnasium task id.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel Isaac Lab environments.")
parser.add_argument("--max_iterations", type=int, default=150, help="Number of PPO iterations.")
parser.add_argument("--num_steps", type=int, default=16, help="Rollout length per environment.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Adam learning rate.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda.")
parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon.")
parser.add_argument("--entropy_coef", type=float, default=0.001, help="Entropy bonus coefficient.")
parser.add_argument("--value_coef", type=float, default=1.0, help="Value loss coefficient.")
parser.add_argument(
    "--use_clipped_value_loss",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use PPO-style clipped value loss.",
)
parser.add_argument("--desired_kl", type=float, default=0.01, help="Target KL for adaptive learning-rate scheduling.")
parser.add_argument("--schedule", choices=["fixed", "adaptive"], default="adaptive", help="Learning-rate schedule.")
parser.add_argument("--min_learning_rate", type=float, default=1.0e-5, help="Lower bound for adaptive learning rate.")
parser.add_argument("--max_learning_rate", type=float, default=1.0e-2, help="Upper bound for adaptive learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
parser.add_argument("--update_epochs", type=int, default=5, help="Epochs per PPO update.")
parser.add_argument("--num_mini_batches", type=int, default=4, help="Mini-batches per epoch.")
parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden width for actor and critic MLPs.")
parser.add_argument("--min_log_std", type=float, default=-5.0, help="Minimum actor log standard deviation.")
parser.add_argument("--max_log_std", type=float, default=2.0, help="Maximum actor log standard deviation.")
parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint interval in iterations.")
parser.add_argument("--experiment_name", type=str, default="cartpole", help="Experiment folder name.")
parser.add_argument("--run_name", type=str, default="", help="Optional run name suffix.")
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
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import rl_lab_learning.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rl_lab_learning.algorithms.ppo import PPOConfig, PPOTrainer, RolloutBuffer  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
except ImportError:  # pragma: no cover - tensorboard is optional.
    SummaryWriter = None


METRIC_FIELDS = [
    "iteration",
    "global_step",
    "mean_reward_100",
    "mean_length_100",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "learning_rate",
    "action_std_mean",
    "total_fps",
    "iteration_time",
]


def get_policy_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract Isaac Lab's policy observation group."""
    return obs_dict["policy"]


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        use_clipped_value_loss=args.use_clipped_value_loss,
        desired_kl=args.desired_kl,
        schedule=args.schedule,
        min_learning_rate=args.min_learning_rate,
        max_learning_rate=args.max_learning_rate,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        num_mini_batches=args.num_mini_batches,
        hidden_dim=args.hidden_dim,
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
    )


def log_tensorboard(
    writer: SummaryWriter | None,
    iteration: int,
    total_time: float,
    global_step: int,
    mean_reward: float,
    mean_length: float,
    stats,
    fps: int,
    iteration_time: float,
) -> None:
    if writer is None:
        return

    # These tags mirror RSL-RL where possible, so TensorBoard comparisons line up.
    writer.add_scalar("Loss/surrogate", stats.policy_loss, iteration)
    writer.add_scalar("Loss/value_function", stats.value_loss, iteration)
    writer.add_scalar("Loss/entropy", stats.entropy, iteration)
    writer.add_scalar("Loss/approx_kl", stats.approx_kl, iteration)
    writer.add_scalar("Loss/clip_fraction", stats.clip_fraction, iteration)
    writer.add_scalar("Loss/explained_variance", stats.explained_variance, iteration)
    writer.add_scalar("Loss/learning_rate", stats.learning_rate, iteration)
    writer.add_scalar("Policy/mean_noise_std", stats.action_std_mean, iteration)
    writer.add_scalar("Perf/total_fps", fps, iteration)
    writer.add_scalar("Perf/iteration_time", iteration_time, iteration)
    writer.add_scalar("Train/mean_reward", mean_reward, iteration)
    writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
    writer.add_scalar("Train/mean_reward/time", mean_reward, total_time)
    writer.add_scalar("Train/mean_episode_length/time", mean_length, total_time)
    writer.add_scalar("Charts/global_step", global_step, iteration)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    log_root = PROJECT_ROOT / "logs" / "standalone" / "ppo" / args_cli.experiment_name
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        run_dir += f"_{args_cli.run_name}"
    log_dir = log_root / run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging standalone PPO run to: {log_dir}")

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
        print(f"[INFO] num_envs={num_envs}, obs_dim={obs_dim}, action_dim={action_dim}, device={device}")

        config = build_ppo_config(args_cli)
        trainer = PPOTrainer(obs_dim, action_dim, config, device)
        buffer = RolloutBuffer(args_cli.num_steps, num_envs, obs_dim, action_dim, device)

        episode_rewards = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device)
        completed_rewards: list[float] = []
        completed_lengths: list[float] = []
        global_step = 0
        total_time = 0.0

        for iteration in range(args_cli.max_iterations):
            iteration_start_time = time.time()
            buffer.clear()

            for _ in range(args_cli.num_steps):
                with torch.no_grad():
                    actions, log_probs, values = trainer.act(obs)

                next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)
                dones = terminated | truncated
                buffer.add(obs, actions, log_probs, rewards, dones, values)

                episode_rewards += rewards
                episode_lengths += 1
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    completed_rewards.extend(episode_rewards[done_ids].detach().cpu().tolist())
                    completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())
                    episode_rewards[done_ids] = 0
                    episode_lengths[done_ids] = 0

                obs = get_policy_obs(next_obs_dict)
                global_step += num_envs

            with torch.no_grad():
                last_values = trainer.value(obs)
            buffer.compute_returns_and_advantages(last_values, config.gamma, config.gae_lambda)
            stats = trainer.update(buffer.flatten())

            mean_reward = sum(completed_rewards[-100:]) / max(1, len(completed_rewards[-100:]))
            mean_length = sum(completed_lengths[-100:]) / max(1, len(completed_lengths[-100:]))
            iteration_time = time.time() - iteration_start_time
            total_time += iteration_time
            fps = int((args_cli.num_steps * num_envs) / max(iteration_time, 1.0e-8))

            log_tensorboard(writer, iteration, total_time, global_step, mean_reward, mean_length, stats, fps, iteration_time)
            metrics_writer.writerow(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "mean_reward_100": mean_reward,
                    "mean_length_100": mean_length,
                    "policy_loss": stats.policy_loss,
                    "value_loss": stats.value_loss,
                    "entropy": stats.entropy,
                    "approx_kl": stats.approx_kl,
                    "clip_fraction": stats.clip_fraction,
                    "explained_variance": stats.explained_variance,
                    "learning_rate": stats.learning_rate,
                    "action_std_mean": stats.action_std_mean,
                    "total_fps": fps,
                    "iteration_time": iteration_time,
                }
            )
            metrics_file.flush()

            print(
                f"iter={iteration:04d} step={global_step:09d} "
                f"reward100={mean_reward:8.3f} len100={mean_length:6.1f} "
                f"policy={stats.policy_loss:8.4f} value={stats.value_loss:8.4f} "
                f"entropy={stats.entropy:7.4f} kl={stats.approx_kl:8.5f} "
                f"clip={stats.clip_fraction:5.3f} ev={stats.explained_variance:6.3f} "
                f"std={stats.action_std_mean:6.3f}"
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
