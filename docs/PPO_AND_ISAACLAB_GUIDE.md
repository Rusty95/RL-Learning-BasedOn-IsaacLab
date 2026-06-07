# PPO and Isaac Lab Project Guide

This repo is meant to be a learning lab: Isaac Lab provides simulation and task APIs, while this project owns the algorithm code you are studying.

## 1. Project Shape

```text
RL-Learning-BasedOn-IsaacLab/
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/
│   │   ├── train.py
│   │   └── play.py
│   └── standalone/
│       ├── train_ppo.py
│       └── plot_metrics.py
├── source/rl_lab_learning/
│   └── rl_lab_learning/
│       ├── algorithms/
│       │   └── ppo.py
│       └── tasks/
│           └── cartpole/
│               ├── cartpole_env_cfg.py
│               └── agents/rsl_rl_ppo_cfg.py
└── logs/
```

The key engineering decision is this:

- `tasks/` describes the world, observations, actions, rewards, resets, and task registration.
- `algorithms/` contains learning logic independent of Isaac Lab runners.
- `scripts/rsl_rl/` delegates to Isaac Lab's official RSL-RL scripts for a baseline.
- `scripts/standalone/` runs our own implementation.

That means you can compare against RSL-RL without being forced to write new algorithms in RSL-RL's internal style.

## 2. Environment Registration

The Cartpole task is registered in:

```text
source/rl_lab_learning/rl_lab_learning/tasks/cartpole/__init__.py
```

The registered task id is:

```text
RLLab-Cartpole-Direct-v0
```

When this package is imported, Gymnasium learns that `RLLab-Cartpole-Direct-v0` maps to Isaac Lab's direct Cartpole environment:

```python
entry_point="isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv"
```

The custom config is:

```python
env_cfg_entry_point="rl_lab_learning.tasks.cartpole.cartpole_env_cfg:RLLabCartpoleEnvCfg"
```

The RSL-RL baseline config is:

```python
rsl_rl_cfg_entry_point="rl_lab_learning.tasks.cartpole.agents.rsl_rl_ppo_cfg:RLLabCartpolePPORunnerCfg"
```

So one task id can be used by both RSL-RL and our standalone PPO.

## 3. Task Config

The file:

```text
source/rl_lab_learning/rl_lab_learning/tasks/cartpole/cartpole_env_cfg.py
```

inherits Isaac Lab's built-in `CartpoleEnvCfg`.

Important fields:

```python
self.scene.num_envs = 1024
self.actions.joint_effort.scale = 100.0
self.rewards.alive.weight = 1.0
self.rewards.terminating.weight = -2.0
self.rewards.pole_pos.weight = -1.0
self.rewards.cart_vel.weight = -0.01
self.rewards.pole_vel.weight = -0.005
```

Interpretation:

- More parallel environments means faster sample collection.
- Action scale controls how strong the cart force can be.
- Reward weights define what the policy is actually optimizing.
- Cartpole is a good first PPO task because the observation and action spaces are small.

## 4. PPO Core

The standalone algorithm lives in:

```text
source/rl_lab_learning/rl_lab_learning/algorithms/ppo.py
```

Main classes:

- `PPOConfig`: all PPO hyperparameters.
- `ActorCritic`: policy network plus value network.
- `RolloutBuffer`: stores one batch of environment interaction.
- `PPOTrainer`: computes PPO losses and applies optimizer steps.

### Actor-Critic

`ActorCritic` has two neural networks:

```python
self.actor = nn.Sequential(...)
self.critic = nn.Sequential(...)
self.log_std = nn.Parameter(torch.zeros(action_dim))
```

The actor outputs the action mean. `log_std` is learned, so the policy is a Gaussian:

```python
Normal(mean, std)
```

During training:

```python
actions = dist.sample()
log_probs = dist.log_prob(actions).sum(dim=-1)
values = self.value(obs)
```

The sampled action explores. The log probability is saved because PPO later asks: "How much did the new policy change the probability of the old sampled action?"

During inference, we usually use:

```python
return self.actor(obs)
```

That gives the deterministic mean action.

### Rollout Buffer

PPO is on-policy. It collects fresh data, updates on that data, then throws it away.

For each timestep, the buffer stores:

- observation
- action
- old log probability
- reward
- done flag
- value prediction

After rollout collection, it computes GAE:

```python
delta = reward + gamma * next_value * next_not_done - value
advantage = delta + gamma * gae_lambda * next_not_done * advantage
return = advantage + value
```

Here `delta` is the TD error. It is not the state value itself. It is the one-step value prediction error:

```text
TD error = r + gamma * V(next_state) - V(state)
```

PPO uses the advantage to tell the actor whether the sampled action was better or worse than expected.

## 5. PPO Loss

The policy ratio is:

```python
ratio = exp(new_log_prob - old_log_prob)
```

If `ratio > 1`, the new policy likes the old action more than before.
If `ratio < 1`, it likes that action less than before.

The unclipped objective is:

```python
ratio * advantage
```

The clipped objective is:

```python
clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
```

PPO takes the conservative one:

```python
policy_loss = -min(unclipped, clipped).mean()
```

This is the core PPO trick: learn from the batch, but do not let one update move the policy too far.

The value loss trains the critic:

```python
0.5 * (return - value)^2
```

This implementation also supports clipped value loss, matching the common PPO/RSL-RL style.

The entropy term keeps exploration alive:

```python
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

## 6. Training Script Flow

The entry point is:

```text
scripts/standalone/train_ppo.py
```

High-level flow:

1. Start Isaac Sim through `AppLauncher`.
2. Import and register `rl_lab_learning.tasks`.
3. Parse the Isaac Lab environment config.
4. Create the Gymnasium environment.
5. Reset environment and read `obs["policy"]`.
6. Build `PPOTrainer` and `RolloutBuffer`.
7. Collect `num_steps * num_envs` samples.
8. Compute returns and advantages.
9. Run PPO update.
10. Write CSV metrics, TensorBoard scalars, and checkpoints.

The script defaults to GUI mode. Add `--headless` only when you want faster unattended training.

## 7. TensorBoard Metrics

The standalone PPO writes RSL-RL-style tags:

```text
Loss/surrogate
Loss/value_function
Loss/entropy
Loss/approx_kl
Loss/clip_fraction
Loss/explained_variance
Loss/learning_rate
Policy/mean_noise_std
Perf/total_fps
Perf/iteration_time
Train/mean_reward
Train/mean_episode_length
```

That is why you can compare our standalone PPO and RSL-RL PPO in one TensorBoard page:

```bash
tensorboard --logdir logs
```

You do not need to wait for training to finish. TensorBoard can read event files while training is still running.

## 8. Commands

Activate Isaac Lab:

```bash
source /home/hall/code/activate_isaaclab.sh
```

If your shell has not initialized conda:

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
source /home/hall/code/activate_isaaclab.sh
```

Install this package in editable mode:

```bash
cd /home/hall/code/RL-Learning-BasedOn-IsaacLab
python -m pip install -e source/rl_lab_learning
```

List registered tasks:

```bash
python scripts/list_envs.py
```

Train RSL-RL baseline:

```bash
python scripts/rsl_rl/train.py --task RLLab-Cartpole-Direct-v0 --num_envs 1024 --max_iterations 150
```

Train standalone PPO:

```bash
python scripts/standalone/train_ppo.py --task RLLab-Cartpole-Direct-v0 --num_envs 1024 --max_iterations 150
```

Open TensorBoard:

```bash
tensorboard --logdir logs
```

Generate an offline PNG:

```bash
python scripts/standalone/plot_metrics.py
```

## 9. Reading Curves

Healthy Cartpole PPO usually looks like this:

- `Train/mean_reward` rises and then stabilizes.
- `Train/mean_episode_length` approaches the environment max length.
- `Loss/approx_kl` stays near `desired_kl`, not constantly exploding.
- `Loss/clip_fraction` is not always zero and not always near one.
- `Policy/mean_noise_std` gradually decreases as the policy becomes confident.
- `Loss/explained_variance` rises toward one when the critic starts predicting returns well.

Common problems:

- Reward flat: learning rate too low, action scale wrong, reward too sparse, or observations wrong.
- KL spikes: learning rate too high or too many update epochs.
- Clip fraction near one: policy update is too aggressive.
- Noise std never drops: entropy coefficient too high or task not being solved.
- Value loss huge: critic learning is unstable, reward scale is too large, or rollout length is too short.

## 10. Adding Another Algorithm

Use the same pattern:

```text
source/rl_lab_learning/rl_lab_learning/algorithms/sac.py
scripts/standalone/train_sac.py
docs/SAC_AND_ISAACLAB_GUIDE.md
```

Keep Isaac Lab task registration separate from algorithm code. Then one task can be reused across PPO, SAC, TD3, DDPG, or your own new method.
