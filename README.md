# RL Learning Based On Isaac Lab

This repository is a learning-first Isaac Lab project for implementing and comparing reinforcement-learning algorithms.

Current contents:

- Isaac Lab 2.3.0 / Isaac Sim 5.1.0 compatible project layout
- Cartpole task registration
- RSL-RL PPO baseline wrappers
- Standalone PPO implemented in this repository, independent of RSL-RL algorithm classes
- Standalone SAC implemented in this repository, independent of RSL-RL algorithm classes
- TensorBoard and CSV metrics
- Offline PNG curve plotting

## Activate Environment

```bash
source /home/hall/code/activate_isaaclab.sh
```

If `conda activate` complains in a non-interactive shell:

```bash
source /home/hall/miniconda3/etc/profile.d/conda.sh
source /home/hall/code/activate_isaaclab.sh
```

## Install Editable Package

```bash
cd /home/hall/code/RL-Learning-BasedOn-IsaacLab
python -m pip install -e source/rl_lab_learning
```

The scripts also add `source/rl_lab_learning` to `PYTHONPATH`, so editable install is convenient but not required for local runs.

## Project Layout

```text
RL-Learning-BasedOn-IsaacLab/
├── docs/
│   └── PPO_AND_ISAACLAB_GUIDE.md
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/
│   │   ├── play.py
│   │   └── train.py
│   └── standalone/
│       ├── plot_metrics.py
│       ├── train_ppo.py
│       └── train_sac.py
└── source/
    └── rl_lab_learning/
        ├── pyproject.toml
        ├── setup.py
        └── rl_lab_learning/
            ├── algorithms/
            │   ├── ppo.py
            │   └── sac.py
            └── tasks/
                └── cartpole/
                    ├── cartpole_env_cfg.py
                    └── agents/
                        └── rsl_rl_ppo_cfg.py
```

## Verify Task Registration

```bash
python scripts/list_envs.py
```

Expected:

```text
RLLab-Cartpole-Direct-v0
```

## Train RSL-RL PPO Baseline

```bash
python scripts/rsl_rl/train.py \
  --task RLLab-Cartpole-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 150
```

Add `--headless` for faster non-GUI training.

## Train Standalone PPO

```bash
python scripts/standalone/train_ppo.py \
  --task RLLab-Cartpole-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 150
```

Logs:

```text
logs/standalone/ppo/
```

## Train Cartpole Swing-Up PPO

The swing-up task uses full-angle initialization, a five-dimensional
`[sin(theta), cos(theta), pole_vel, cart_pos, cart_vel]` observation, and does
not terminate when the pole passes 90 degrees.

```bash
python scripts/standalone/train_ppo.py \
  --task RLLab-Cartpole-SwingUp-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 200 \
  --num_steps 32 \
  --learning_rate 3.0e-4 \
  --gamma 0.995 \
  --entropy_coef 0.003 \
  --hidden_dim 128 \
  --save_interval 100 \
  --experiment_name cartpole_swingup_centered \
  --run_name reward_v2 \
  --headless
```

The checkpoints are written under:

```text
logs/standalone/ppo/cartpole_swingup_centered/
```

Use `model_200.pt` for deterministic playback. Longer training is not
necessarily better: monitor KL divergence and episode length, and retain the
last checkpoint before a destabilizing PPO update.

## Train Standalone SAC

```bash
python scripts/standalone/train_sac.py \
  --task RLLab-Cartpole-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 1500
```

SAC is off-policy, so it writes transitions to a replay buffer and starts gradient updates after
`--initial_random_steps` global environment steps. Logs:

```text
logs/standalone/sac/
```

TensorBoard:

```bash
tensorboard --logdir logs
```

Offline plot:

```bash
python scripts/standalone/plot_metrics.py
```

## Adding New Algorithms

Recommended pattern:

```text
source/rl_lab_learning/rl_lab_learning/algorithms/<algorithm>.py
scripts/standalone/train_<algorithm>.py
docs/<ALGORITHM>_GUIDE.md
```

Keep algorithm code independent from Isaac Lab startup code where possible. Scripts should launch Isaac Sim, create environments, and call algorithm modules.
