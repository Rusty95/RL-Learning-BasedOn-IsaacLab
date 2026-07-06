"""Full-angle Cartpole swing-up task with raw RGB observations."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleCameraEnv, CartpoleRGBCameraEnvCfg


@configclass
class RLLabCartpoleSwingUpCameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """Camera task matching the dynamics and resets of the swing-up policy."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=256,
        height=256,
    )
    observation_space = [256, 256, 3]

    episode_length_s = 10.0
    action_scale = 100.0
    max_cart_pos = 3.0

    initial_pole_angle_range_rad = [-math.pi, math.pi]
    initial_pole_velocity_range = [-0.35, 0.35]
    initial_cart_position_range = [-0.75, 0.75]
    initial_cart_velocity_range = [-0.35, 0.35]

    rew_scale_upright = 3.0
    rew_scale_centered = 1.5
    rew_scale_cart_pos = -0.4
    rew_scale_cart_vel = -0.04
    rew_scale_pole_vel = -0.005
    rew_scale_action = -0.002
    rew_scale_cart_boundary = -2.0
    cart_boundary_start = 2.0
    rew_scale_terminated = -20.0


class CartpoleSwingUpCameraEnv(CartpoleCameraEnv):
    """Full-angle swing-up dynamics without terminating when the pole falls."""

    cfg: RLLabCartpoleSwingUpCameraEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        ground_cfg = sim_utils.GroundPlaneCfg(
            color=(0.01, 0.01, 0.01),
            size=(10.0, 10.0),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": self._tiled_camera.data.output["rgb"][..., :3].clone()}

    def _get_rewards(self) -> torch.Tensor:
        pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
        pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        normalized_action = self.actions[:, 0] / self.action_scale

        upright = 0.5 * (torch.cos(pole_pos) + 1.0)
        return (
            self.cfg.rew_scale_upright * upright.square()
            + self.cfg.rew_scale_centered * upright * torch.exp(-0.5 * cart_pos.square())
            + self.cfg.rew_scale_cart_pos * cart_pos.square()
            + self.cfg.rew_scale_cart_vel * cart_vel.square()
            + self.cfg.rew_scale_pole_vel * pole_vel.square()
            + self.cfg.rew_scale_action * normalized_action.square()
            + self.cfg.rew_scale_cart_boundary
            * torch.relu(torch.abs(cart_pos) - self.cfg.cart_boundary_start).square()
            + self.cfg.rew_scale_terminated * self.reset_terminated.float()
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self._cartpole.data.joint_pos
        self.joint_vel = self._cartpole.data.joint_vel
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cart_out_of_bounds = torch.abs(self.joint_pos[:, self._cart_dof_idx[0]]) > self.cfg.max_cart_pos
        return cart_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._cartpole._ALL_INDICES
        DirectRLEnv._reset_idx(self, env_ids)

        joint_pos = self._cartpole.data.default_joint_pos[env_ids].clone()
        joint_vel = self._cartpole.data.default_joint_vel[env_ids].clone()
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

        default_root_state = self._cartpole.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self._cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
