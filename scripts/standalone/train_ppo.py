"""Train an Isaac Lab task with this repo's standalone PPO implementation.

Cartpole default data shapes used in the comments below:

- num_envs = 1024
- num_steps = 16
- obs_dim = 4
- action_dim = 1
- one rollout batch = num_steps * num_envs = 16384 samples

Typical tensors:

- obs: [1024, 4], one observation vector for each parallel environment.
- actions: [1024, 1], one continuous force command for each cart.
- rewards: [1024], one scalar reward for each environment step.
- values: [1024], critic V(s) estimate for each environment.
- flattened rollout obs: [16384, 4], PPO update batch after collecting 16 steps.
"""

from __future__ import annotations

import argparse  # 解析命令行参数，比如 --num_envs 1024。
import csv  # 把每一轮训练指标写到 metrics.csv，方便后处理画图。
import sys  # 修改 sys.path，让脚本能直接 import 本仓库 package。
import time  # 统计每轮 PPO iteration 耗时和 FPS。
from datetime import datetime  # 给每次训练生成时间戳目录。
from pathlib import Path  # 用 pathlib 处理工程路径，比字符串拼接稳。

from isaaclab.app import AppLauncher  # Isaac Lab/Isaac Sim 的应用启动器，必须先启动再 import 仿真相关模块。


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../RL-Learning-BasedOn-IsaacLab。
SOURCE_ROOT = PROJECT_ROOT / "source" / "rl_lab_learning"  # 本仓库 Python package 的源码根目录。

if str(SOURCE_ROOT) not in sys.path:  # 如果没有 pip install -e，也能直接运行脚本。
    sys.path.insert(0, str(SOURCE_ROOT))  # 把 source/rl_lab_learning 放到 import 搜索路径最前面。


parser = argparse.ArgumentParser(description="Train RLLab Cartpole with standalone PPO.")  # 创建 CLI 参数解析器。
parser.add_argument("--task", type=str, default="RLLab-Cartpole-Direct-v0", help="Gymnasium task id.")  # 注册的任务名。
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel Isaac Lab environments.")  # 并行环境数量。
parser.add_argument("--max_iterations", type=int, default=150, help="Number of PPO iterations.")  # PPO 外层迭代次数。
parser.add_argument("--num_steps", type=int, default=16, help="Rollout length per environment.")  # 每个环境采样 16 步。
parser.add_argument("--seed", type=int, default=42, help="Random seed.")  # 随机种子，方便复现实验。
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Adam learning rate.")  # Adam 初始学习率。
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")  # 折扣因子，越接近 1 越看重长期奖励。
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda.")  # GAE 的 lambda，平衡偏差和方差。
parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon.")  # PPO ratio 裁剪范围 1±0.2。
parser.add_argument("--entropy_coef", type=float, default=0.001, help="Entropy bonus coefficient.")  # 熵奖励系数，鼓励探索。
parser.add_argument("--value_coef", type=float, default=1.0, help="Value loss coefficient.")  # critic loss 在总 loss 中的权重。
parser.add_argument(
    "--use_clipped_value_loss",  # 是否对 value loss 也做 PPO 风格裁剪。
    action=argparse.BooleanOptionalAction,  # 允许 --use_clipped_value_loss 和 --no-use_clipped_value_loss。
    default=True,  # 默认开启，和很多 PPO/RSL-RL 实现保持一致。
    help="Use PPO-style clipped value loss.",
)
parser.add_argument("--desired_kl", type=float, default=0.01, help="Target KL for adaptive learning-rate scheduling.")  # 目标 KL。
parser.add_argument("--schedule", choices=["fixed", "adaptive"], default="adaptive", help="Learning-rate schedule.")  # 学习率策略。
parser.add_argument("--min_learning_rate", type=float, default=1.0e-5, help="Lower bound for adaptive learning rate.")  # LR 下限。
parser.add_argument("--max_learning_rate", type=float, default=1.0e-2, help="Upper bound for adaptive learning rate.")  # LR 上限。
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")  # 梯度裁剪，防止梯度爆炸。
parser.add_argument("--update_epochs", type=int, default=5, help="Epochs per PPO update.")  # 同一批 rollout 重复训练 5 轮。
parser.add_argument("--num_mini_batches", type=int, default=4, help="Mini-batches per epoch.")  # 每轮切成 4 个 mini-batch。
parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden width for actor and critic MLPs.")  # actor/critic 隐藏层宽度。
parser.add_argument("--min_log_std", type=float, default=-5.0, help="Minimum actor log standard deviation.")  # 动作 std 下限。
parser.add_argument("--max_log_std", type=float, default=2.0, help="Maximum actor log standard deviation.")  # 动作 std 上限。
parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint interval in iterations.")  # 每 50 轮保存一次模型。
parser.add_argument("--experiment_name", type=str, default="cartpole", help="Experiment folder name.")  # 日志大目录名。
parser.add_argument("--run_name", type=str, default="", help="Optional run name suffix.")  # 可选 run 后缀，方便标实验。
parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")  # 是否录制视频。
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in steps.")  # 每段视频长度。
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between videos in steps.")  # 视频录制间隔。
AppLauncher.add_app_launcher_args(parser)  # 加入 Isaac Lab 通用参数，比如 --headless、--device cuda:0。
args_cli = parser.parse_args()  # 真正解析命令行，得到 args_cli.xxx。

if args_cli.video:  # 如果要录视频，Isaac Sim 必须启用 camera 相关能力。
    args_cli.enable_cameras = True  # 等价于自动帮你加 --enable_cameras。

app_launcher = AppLauncher(args_cli)  # 启动 Isaac Sim app；这一步必须在 gym/env 相关 import 前面。
simulation_app = app_launcher.app  # 保存 app 句柄，脚本结束时要 close。


import gymnasium as gym  # noqa: E402  # AppLauncher 之后再 import Gym/Isaac 环境相关模块。
import torch  # noqa: E402  # PyTorch，用于 tensor、网络、优化器。

import isaaclab_tasks  # noqa: F401, E402  # 导入 Isaac Lab 官方任务包，触发官方任务注册。
import rl_lab_learning.tasks  # noqa: F401, E402  # 导入本仓库任务包，触发 RLLab-Cartpole-Direct-v0 注册。
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402  # 根据 task id 解析 Isaac Lab env cfg。
from rl_lab_learning.algorithms.ppo import PPOConfig, PPOTrainer, RolloutBuffer  # noqa: E402  # 我们自己的 PPO 实现。

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402  # TensorBoard 写 event 文件。
except ImportError:  # pragma: no cover - tensorboard is optional.
    SummaryWriter = None  # 没装 tensorboard 也能训练，只是不写 event。


METRIC_FIELDS = [  # metrics.csv 的列名，一行对应一个 PPO iteration。
    "iteration",  # 当前 PPO 迭代编号，例如 0..149。
    "global_step",  # 到目前为止总共采样了多少环境步，例如 iteration 0 后是 16*1024=16384。
    "mean_reward_100",  # 最近 100 个完成 episode 的平均 reward。
    "mean_length_100",  # 最近 100 个完成 episode 的平均长度。
    "policy_loss",  # PPO actor loss，也叫 surrogate loss。
    "value_loss",  # critic value function loss。
    "entropy",  # 策略分布熵，越大说明探索越强。
    "approx_kl",  # 新旧策略近似 KL，衡量 policy 更新幅度。
    "clip_fraction",  # ratio 被 clip 的样本比例。
    "explained_variance",  # critic 对 returns 的解释能力。
    "learning_rate",  # 当前 optimizer lr，adaptive schedule 会改它。
    "action_std_mean",  # Gaussian policy 的平均动作标准差。
    "total_fps",  # 采样+训练吞吐。
    "iteration_time",  # 当前 iteration 耗时。
]


def get_policy_obs(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract Isaac Lab's policy observation group.

    Isaac Lab 通常返回 dict，例如:
    obs_dict["policy"].shape == [1024, 4]
    """
    return obs_dict["policy"]  # Cartpole: 每个环境 4 维观测，返回 [num_envs, obs_dim]。


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """把命令行参数转成算法层 PPOConfig。"""
    return PPOConfig(
        learning_rate=args.learning_rate,  # Adam 初始 lr，例如 0.001。
        gamma=args.gamma,  # GAE/return 折扣，例如 0.99。
        gae_lambda=args.gae_lambda,  # GAE lambda，例如 0.95。
        clip_eps=args.clip_eps,  # PPO ratio clip，例如 0.2。
        entropy_coef=args.entropy_coef,  # entropy bonus 权重。
        value_coef=args.value_coef,  # value loss 权重。
        use_clipped_value_loss=args.use_clipped_value_loss,  # 是否启用 clipped value loss。
        desired_kl=args.desired_kl,  # adaptive lr 的目标 KL。
        schedule=args.schedule,  # "adaptive" 或 "fixed"。
        min_learning_rate=args.min_learning_rate,  # adaptive lr 下限。
        max_learning_rate=args.max_learning_rate,  # adaptive lr 上限。
        max_grad_norm=args.max_grad_norm,  # 梯度裁剪阈值。
        update_epochs=args.update_epochs,  # 每批数据训练几轮。
        num_mini_batches=args.num_mini_batches,  # 每轮切几个 mini-batch。
        hidden_dim=args.hidden_dim,  # MLP 隐藏层宽度。
        min_log_std=args.min_log_std,  # actor log_std clamp 下限。
        max_log_std=args.max_log_std,  # actor log_std clamp 上限。
    )


def log_tensorboard(
    writer: SummaryWriter | None,  # SummaryWriter 或 None。
    iteration: int,  # x 轴 step，这里用 PPO iteration。
    total_time: float,  # 从训练开始累计耗时。
    global_step: int,  # 累计环境交互步数。
    mean_reward: float,  # 最近 100 个 episode 平均 reward。
    mean_length: float,  # 最近 100 个 episode 平均长度。
    stats,  # PPOTrainer.update 返回的统计量对象。
    fps: int,  # 当前 iteration 的吞吐。
    iteration_time: float,  # 当前 iteration 耗时。
) -> None:
    if writer is None:  # 没装 tensorboard 时直接跳过。
        return

    # These tags mirror RSL-RL where possible, so TensorBoard comparisons line up.
    writer.add_scalar("Loss/surrogate", stats.policy_loss, iteration)  # actor loss，shape 是 Python float。
    writer.add_scalar("Loss/value_function", stats.value_loss, iteration)  # critic loss。
    writer.add_scalar("Loss/entropy", stats.entropy, iteration)  # policy entropy。
    writer.add_scalar("Loss/approx_kl", stats.approx_kl, iteration)  # 新旧策略 KL。
    writer.add_scalar("Loss/clip_fraction", stats.clip_fraction, iteration)  # clip 比例。
    writer.add_scalar("Loss/explained_variance", stats.explained_variance, iteration)  # critic 解释方差。
    writer.add_scalar("Loss/learning_rate", stats.learning_rate, iteration)  # 当前 lr。
    writer.add_scalar("Policy/mean_noise_std", stats.action_std_mean, iteration)  # actor std 均值。
    writer.add_scalar("Perf/total_fps", fps, iteration)  # 性能指标。
    writer.add_scalar("Perf/iteration_time", iteration_time, iteration)  # 单轮耗时。
    writer.add_scalar("Train/mean_reward", mean_reward, iteration)  # 训练 reward 曲线。
    writer.add_scalar("Train/mean_episode_length", mean_length, iteration)  # episode length 曲线。
    writer.add_scalar("Train/mean_reward/time", mean_reward, total_time)  # 以 wall time 为横轴的 reward。
    writer.add_scalar("Train/mean_episode_length/time", mean_length, total_time)  # 以 wall time 为横轴的 episode length。
    writer.add_scalar("Charts/global_step", global_step, iteration)  # 额外记录总环境步。


def main() -> None:
    torch.manual_seed(args_cli.seed)  # 固定 PyTorch 随机种子。
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32，加速矩阵乘。
    torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32。
    torch.backends.cudnn.deterministic = False  # 不强制完全确定性，换速度。
    torch.backends.cudnn.benchmark = False  # 不让 cuDNN 做输入尺寸搜索；这里 MLP 影响不大。

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)  # 构建环境配置。
    env_cfg.seed = args_cli.seed  # 把 seed 也写进 Isaac Lab env cfg。

    log_root = PROJECT_ROOT / "logs" / "standalone" / "ppo" / args_cli.experiment_name  # logs/standalone/ppo/cartpole。
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 例如 2026-06-07_21-41-11。
    if args_cli.run_name:  # 如果传了 --run_name test，会拼到目录名后。
        run_dir += f"_{args_cli.run_name}"
    log_dir = log_root / run_dir  # 本次 run 的完整日志目录。
    log_dir.mkdir(parents=True, exist_ok=True)  # 创建日志目录。
    env_cfg.log_dir = str(log_dir)  # 让 Isaac Lab 内部也知道日志路径。
    print(f"[INFO] Logging standalone PPO run to: {log_dir}")  # 打印日志路径。

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)  # 创建 Isaac Lab Gym 环境。
    if args_cli.video:  # 如果开启视频录制，外面包一层 RecordVideo。
        env = gym.wrappers.RecordVideo(
            env,  # 原始 Isaac Lab 环境。
            video_folder=str(log_dir / "videos" / "train"),  # 视频保存目录。
            step_trigger=lambda step: step % args_cli.video_interval == 0,  # 每隔多少 global step 录一次。
            video_length=args_cli.video_length,  # 每个视频录多少 step。
            disable_logger=True,  # 关掉 Gym wrapper 自己的啰嗦日志。
        )

    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None  # TensorBoard event writer。
    metrics_file = (log_dir / "metrics.csv").open("w", newline="", encoding="utf-8")  # CSV 指标文件。
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)  # 用 dict 写每一行指标。
    metrics_writer.writeheader()  # 写 CSV 表头。

    try:
        obs_dict, _ = env.reset()  # 重置所有并行环境；obs_dict["policy"] shape 通常是 [1024, 4]。
        obs = get_policy_obs(obs_dict)  # 取 policy 观测，Cartpole: obs.shape == [1024, 4]。
        device = obs.device  # Isaac Lab 会直接给 cuda tensor，例如 cuda:0。
        num_envs = env.unwrapped.num_envs  # 实际并行环境数，例如 1024。
        obs_dim = obs.shape[-1]  # 单个环境观测维度，Cartpole 是 4。
        action_dim = env.unwrapped.single_action_space.shape[0]  # 单个环境动作维度，Cartpole 是 1。
        print(f"[INFO] num_envs={num_envs}, obs_dim={obs_dim}, action_dim={action_dim}, device={device}")

        config = build_ppo_config(args_cli)  # 把 CLI 参数整理成 PPOConfig。
        trainer = PPOTrainer(obs_dim, action_dim, config, device)  # 创建 actor、critic、optimizer。
        buffer = RolloutBuffer(args_cli.num_steps, num_envs, obs_dim, action_dim, device)  # 形状如 obs=[16,1024,4]。

        episode_rewards = torch.zeros(num_envs, device=device)  # 每个环境当前 episode 累计 reward，shape [1024]。
        episode_lengths = torch.zeros(num_envs, device=device)  # 每个环境当前 episode 长度，shape [1024]。
        completed_rewards: list[float] = []  # 已结束 episode 的总 reward，CPU list。
        completed_lengths: list[float] = []  # 已结束 episode 的长度，CPU list。
        global_step = 0  # 累计环境 step；每 env.step 一次增加 num_envs。
        total_time = 0.0  # 累计训练 wall time。

        for iteration in range(args_cli.max_iterations):  # 外层 PPO loop，例如 150 轮。
            iteration_start_time = time.time()  # 记录本轮开始时间。
            buffer.clear()  # 清空 rollout 写入位置；底层 tensor 复用，不重新分配。

            for _ in range(args_cli.num_steps):  # 收集一段 rollout，例如每个环境 16 步。
                with torch.no_grad():  # 采样阶段不反传，只需要 action/log_prob/value。
                    actions, log_probs, values = trainer.act(obs)  # obs [1024,4] -> actions [1024,1], log_probs [1024], values [1024]。

                next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)  # actions [1024,1] 送进仿真。
                dones = terminated | truncated  # terminated/truncated shape [1024]，合成 episode 是否结束。
                buffer.add(obs, actions, log_probs, rewards, dones, values)  # 写入当前 step 数据到 buffer 的第 t 行。

                episode_rewards += rewards  # rewards shape [1024]，累加到每个环境自己的 episode reward。
                episode_lengths += 1  # 每个还在跑的环境长度 +1；shape [1024]。
                done_ids = dones.nonzero(as_tuple=False).flatten()  # 找到刚结束的环境编号，例如 tensor([3, 91, ...])。
                if done_ids.numel() > 0:  # 如果这一仿真步里有环境结束。
                    completed_rewards.extend(episode_rewards[done_ids].detach().cpu().tolist())  # 把结束环境 reward 存到 CPU list。
                    completed_lengths.extend(episode_lengths[done_ids].detach().cpu().tolist())  # 把结束环境 length 存到 CPU list。
                    episode_rewards[done_ids] = 0  # 已结束环境的累计 reward 清零，准备新 episode。
                    episode_lengths[done_ids] = 0  # 已结束环境的长度清零。

                obs = get_policy_obs(next_obs_dict)  # 更新当前观测，shape 仍是 [1024, 4]。
                global_step += num_envs  # 一次 env.step 等于并行走了 1024 个环境步。

            with torch.no_grad():  # rollout 结束后，用 critic 估计最后一个 next state 的 V(s)。
                last_values = trainer.value(obs)  # obs [1024,4] -> last_values [1024]。
            buffer.compute_returns_and_advantages(last_values, config.gamma, config.gae_lambda)  # 计算 returns/advantages，shape [16,1024]。
            stats = trainer.update(buffer.flatten())  # flatten 后 obs [16384,4]，actions [16384,1]，做 PPO 多轮更新。

            mean_reward = sum(completed_rewards[-100:]) / max(1, len(completed_rewards[-100:]))  # 最近 100 个 episode 平均回报。
            mean_length = sum(completed_lengths[-100:]) / max(1, len(completed_lengths[-100:]))  # 最近 100 个 episode 平均长度。
            iteration_time = time.time() - iteration_start_time  # 当前 iteration 总耗时。
            total_time += iteration_time  # 累加 wall time。
            fps = int((args_cli.num_steps * num_envs) / max(iteration_time, 1.0e-8))  # 环境步吞吐，例如 16*1024/time。

            log_tensorboard(writer, iteration, total_time, global_step, mean_reward, mean_length, stats, fps, iteration_time)  # 写 TB。
            metrics_writer.writerow(
                {
                    "iteration": iteration,  # 当前 PPO iteration。
                    "global_step": global_step,  # 累计环境步。
                    "mean_reward_100": mean_reward,  # 最近 100 episode reward。
                    "mean_length_100": mean_length,  # 最近 100 episode length。
                    "policy_loss": stats.policy_loss,  # actor loss。
                    "value_loss": stats.value_loss,  # critic loss。
                    "entropy": stats.entropy,  # policy entropy。
                    "approx_kl": stats.approx_kl,  # policy 更新幅度。
                    "clip_fraction": stats.clip_fraction,  # PPO clip 比例。
                    "explained_variance": stats.explained_variance,  # critic 解释能力。
                    "learning_rate": stats.learning_rate,  # 当前 lr。
                    "action_std_mean": stats.action_std_mean,  # 当前探索噪声。
                    "total_fps": fps,  # 吞吐。
                    "iteration_time": iteration_time,  # 耗时。
                }
            )
            metrics_file.flush()  # 立刻刷盘，训练中断也能看到最新 CSV。

            print(
                f"iter={iteration:04d} step={global_step:09d} "  # 迭代号和环境步。
                f"reward100={mean_reward:8.3f} len100={mean_length:6.1f} "  # 最近 100 episode 表现。
                f"policy={stats.policy_loss:8.4f} value={stats.value_loss:8.4f} "  # actor/critic loss。
                f"entropy={stats.entropy:7.4f} kl={stats.approx_kl:8.5f} "  # 探索和 KL。
                f"clip={stats.clip_fraction:5.3f} ev={stats.explained_variance:6.3f} "  # clip 比例和 critic 解释方差。
                f"std={stats.action_std_mean:6.3f}"  # action std，越低通常说明策略越确定。
            )

            if iteration % args_cli.save_interval == 0:  # 每 save_interval 轮保存一次。
                trainer.save_checkpoint(log_dir / f"model_{iteration}.pt", iteration, extra={"args": vars(args_cli)})  # 保存模型和参数。

        final_iteration = max(0, args_cli.max_iterations - 1)  # 训练结束时最后一轮编号。
        trainer.save_checkpoint(log_dir / f"model_{final_iteration}.pt", final_iteration, extra={"args": vars(args_cli)})  # 保存最终模型。
    finally:
        metrics_file.close()  # 无论是否异常，都关闭 CSV 文件。
        if writer is not None:  # 如果 TensorBoard writer 存在。
            writer.close()  # flush 并关闭 event 文件。
        env.close()  # 关闭 Gym/Isaac Lab 环境。


if __name__ == "__main__":
    try:
        main()  # 运行训练主流程。
    finally:
        simulation_app.close()  # 关闭 Isaac Sim app，释放显存和仿真资源。
