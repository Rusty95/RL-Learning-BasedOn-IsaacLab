"""Cartpole task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="RLLab-Cartpole-Direct-v0",
    entry_point="isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:RLLabCartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RLLabCartpolePPORunnerCfg",
    },
)

gym.register(
    id="RLLab-Cartpole-SwingUp-Direct-v0",
    entry_point=f"{__name__}.cartpole_swingup_env:CartpoleSwingUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_swingup_env:RLLabCartpoleSwingUpEnvCfg",
    },
)

gym.register(
    id="RLLab-Cartpole-SwingUp-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.cartpole_swingup_camera_env:CartpoleSwingUpCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.cartpole_swingup_camera_env:RLLabCartpoleSwingUpCameraEnvCfg"
        ),
    },
)
