# RL Learning Based On Isaac Lab

This repository is a learning-first Isaac Lab project for implementing and comparing reinforcement-learning algorithms.

Current contents:

- Isaac Lab 2.3.0 / Isaac Sim 5.1.0 compatible project layout
- Cartpole task registration
- RSL-RL PPO baseline wrappers
- Standalone PPO implemented in this repository, independent of RSL-RL algorithm classes
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
│       └── train_ppo.py
└── source/
    └── rl_lab_learning/
        ├── pyproject.toml
        ├── setup.py
        └── rl_lab_learning/
            ├── algorithms/
            │   └── ppo.py
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

