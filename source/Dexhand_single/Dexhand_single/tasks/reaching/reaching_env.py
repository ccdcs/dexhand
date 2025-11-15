# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .reaching_env_cfg import ReachingEnvCfg


class ReachingEnv(DirectRLEnv):
    cfg: ReachingEnvCfg

    def __init__(self, cfg: ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.target = RigidObject(self.cfg.target)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["target"] = self.target

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        pos_action = self.actions[:, :3]
        orn_action = self.actions[:, 3:]

        orn_quat = quat_from_euler_xyz(
            orn_action[:, 0], orn_action[:, 1], orn_action[:, 2]
        )

        # Create a zero tensor for velocities
        zero_velocities = torch.zeros_like(self.robot.data.root_vel_w)

        # Concatenate into a (N, 13) tensor
        root_state = torch.cat([pos_action, orn_quat, zero_velocities], dim=-1)

        self.robot.write_root_state_to_sim(root_state)

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        target_pos = self.target.data.root_pos_w.expand(self.num_envs, -1)

        joint_pos = self.robot.data.joint_pos

        obs = torch.cat([root_pos, root_quat, target_pos, joint_pos], dim=-1)

        observations = {"policy": obs, "critic": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.robot.data.root_pos_w,
            self.target.data.root_pos_w,
            self.actions,
            self.reset_buf,
            self.cfg.rew_scale_dist,
            self.cfg.rew_scale_success,
            self.cfg.action_penalty,
            self.cfg.success_tolerance,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        dist_to_target = torch.norm(
            self.robot.data.root_pos_w - self.target.data.root_pos_w, dim=-1
        )

        terminated = dist_to_target < self.cfg.success_tolerance

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset robot root state
        root_state = self.robot.data.default_root_state[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Reset finger joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        super()._reset_idx(env_ids)


@torch.jit.script
def compute_rewards(
    root_pos: torch.Tensor,
    target_pos: torch.Tensor,
    actions: torch.Tensor,
    reset_buf: torch.Tensor,
    rew_scale_dist: float,
    rew_scale_success: float,
    action_penalty: float,
    success_tolerance: float,
) -> torch.Tensor:

    dist_to_target = torch.norm(root_pos - target_pos, dim=-1)

    # Distance reward
    rew = rew_scale_dist * dist_to_target

    # Success reward
    rew += rew_scale_success * (dist_to_target < success_tolerance)

    # Action penalty
    rew += action_penalty * torch.sum(torch.square(actions), dim=1)

    # Apply termination penalty
    rew = torch.where(reset_buf, torch.full_like(rew, -2.0), rew)

    return rew
