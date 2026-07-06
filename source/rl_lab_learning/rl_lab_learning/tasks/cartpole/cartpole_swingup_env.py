"""Cartpole swing-up task for standalone PPO training."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg


@configclass
class RLLabCartpoleSwingUpEnvCfg(CartpoleEnvCfg):
    """Cartpole configuration for full-angle swing-up and balancing."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    episode_length_s = 10.0
    action_scale = 100.0
    observation_space = 5

    max_cart_pos = 3.0
    initial_pole_angle_range_rad = [-math.pi, math.pi]
    initial_pole_velocity_range = [-0.25, 0.25]
    initial_cart_position_range = [-0.5, 0.5]
    initial_cart_velocity_range = [-0.25, 0.25]

    rew_scale_upright = 3.0
    rew_scale_centered = 1.5
    rew_scale_cart_pos = -0.4
    rew_scale_cart_vel = -0.04
    rew_scale_pole_vel = -0.005
    rew_scale_action = -0.002
    rew_scale_cart_boundary = -2.0
    cart_boundary_start = 2.0
    rew_scale_terminated = -20.0


class CartpoleSwingUpEnv(CartpoleEnv):
    """Cartpole that learns to swing up from arbitrary pole angles."""

    cfg: RLLabCartpoleSwingUpEnvCfg

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
        obs = torch.stack(
            (
                torch.sin(pole_pos),
                torch.cos(pole_pos),
                self.joint_vel[:, self._pole_dof_idx[0]],
                self.joint_pos[:, self._cart_dof_idx[0]],
                self.joint_vel[:, self._cart_dof_idx[0]],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        normalized_action = self.actions[:, 0] / self.action_scale

        # upright is 1 at theta=0, 0 at theta=pi; squaring sharpens the
        # balancing objective while retaining a dense swing-up gradient.
        upright = 0.5 * (torch.cos(pole_pos) + 1.0)
        upright_reward = self.cfg.rew_scale_upright * upright.square()
        centered_reward = self.cfg.rew_scale_centered * upright * torch.exp(-0.5 * cart_pos.square())
        cart_pos_penalty = self.cfg.rew_scale_cart_pos * cart_pos.square()
        cart_vel_penalty = self.cfg.rew_scale_cart_vel * cart_vel.square()
        pole_vel_penalty = self.cfg.rew_scale_pole_vel * pole_vel.square()
        action_penalty = self.cfg.rew_scale_action * normalized_action.square()
        boundary_penalty = self.cfg.rew_scale_cart_boundary * torch.relu(
            torch.abs(cart_pos) - self.cfg.cart_boundary_start
        ).square()
        termination_penalty = self.cfg.rew_scale_terminated * self.reset_terminated.float()

        return (
            upright_reward
            + centered_reward
            + cart_pos_penalty
            + cart_vel_penalty
            + pole_vel_penalty
            + action_penalty
            + boundary_penalty
            + termination_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cart_out_of_bounds = torch.abs(self.joint_pos[:, self._cart_dof_idx[0]]) > self.cfg.max_cart_pos
        return cart_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        DirectRLEnv._reset_idx(self, env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids].clone()
        joint_vel = self.cartpole.data.default_joint_vel[env_ids].clone()

        pole_shape = joint_pos[:, self._pole_dof_idx].shape
        cart_shape = joint_pos[:, self._cart_dof_idx].shape
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range_rad[0],
            self.cfg.initial_pole_angle_range_rad[1],
            pole_shape,
            joint_pos.device,
        )
        joint_pos[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_position_range[0],
            self.cfg.initial_cart_position_range[1],
            cart_shape,
            joint_pos.device,
        )
        joint_vel[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_velocity_range[0],
            self.cfg.initial_pole_velocity_range[1],
            pole_shape,
            joint_vel.device,
        )
        joint_vel[:, self._cart_dof_idx] += sample_uniform(
            self.cfg.initial_cart_velocity_range[0],
            self.cfg.initial_cart_velocity_range[1],
            cart_shape,
            joint_vel.device,
        )

        default_root_state = self.cartpole.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
