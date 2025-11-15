# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .reaching_env_cfg import ReachingEnvCfg


class ReachingEnv(DirectRLEnv):
    cfg: ReachingEnvCfg

    def __init__(self, cfg: ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        self.target_joint_pos = torch.zeros(
            (cfg.scene.num_envs, 6), device=cfg.sim.device
        )
        super().__init__(cfg, render_mode, **kwargs)

        self.l1_idx, _ = self.robot.find_joints(self.cfg.l1_dof_name)
        self.l2_idx, _ = self.robot.find_joints(self.cfg.l2_dof_name)
        self.l3_idx, _ = self.robot.find_joints(self.cfg.l3_dof_name)
        self.r1_idx, _ = self.robot.find_joints(self.cfg.r1_dof_name)
        self.r2_idx, _ = self.robot.find_joints(self.cfg.r2_dof_name)
        self.r3_idx, _ = self.robot.find_joints(self.cfg.r3_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.test_count = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos,
                self.target_joint_pos,
            ),
            dim=-1,
        )
        observations = {"policy": obs, "critic": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pose_error,
            self.joint_pos,
            self.target_joint_pos,
            self.reset_terminated,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        max_angle = max(
            abs(self.cfg.target_joint_angle_range[0]),
            abs(self.cfg.target_joint_angle_range[1]),
        )
        out_of_bounds = torch.any(torch.abs(self.joint_pos) > max_angle, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # set random target pose
        self.target_joint_pos[env_ids] = sample_uniform(
            self.cfg.target_joint_angle_range[0],
            self.cfg.target_joint_angle_range[1],
            (len(env_ids), 6),
            device=self.device,
        )

        # reset state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pose_error: float,
    current_joint_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    pose_error = torch.sum(torch.square(current_joint_pos - target_joint_pos), dim=-1)
    rew_pose = rew_scale_pose_error * pose_error
    return rew_alive + rew_termination + rew_pose
