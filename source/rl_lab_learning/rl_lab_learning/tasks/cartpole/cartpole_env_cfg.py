"""Cartpole environment config for RL learning experiments."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg


@configclass
class RLLabCartpoleEnvCfg(CartpoleEnvCfg):
    """Direct Cartpole config with explicit learning knobs."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    episode_length_s = 5.0
    action_scale = 100.0

    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

