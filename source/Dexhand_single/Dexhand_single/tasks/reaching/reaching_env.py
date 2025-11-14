# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .reaching_env_cfg import ReachingEnvCfg


class ReachingEnv(DirectRLEnv):
    cfg: ReachingEnvCfg

    def __init__(self, cfg: ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # get assets
        self.robot = self.scene.articulations["robot"]
        self.target = self.scene.scene_objects["target"]

    def _setup_scene(self):
        # spawn ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # scale actions
        forces = self.actions[:, :3] * self.cfg.action_scale
        torques = self.actions[:, 3:] * self.cfg.action_scale
        # apply forces and torques
        self.robot.set_external_force_and_torque(forces, torques)

    def _get_observations(self) -> dict:
        # robot root state (pos, quat, lin_vel, ang_vel)
        robot_root_state = self.robot.data.root_state_w.clone()
        # target position
        target_pos = self.target.data.root_pos_w.clone()
        
        # observations (robot_state, target_pos)
        obs = torch.cat((robot_root_state, target_pos), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # robot and target positions
        robot_pos = self.robot.data.root_pos_w[:, :3]
        target_pos = self.target.data.root_pos_w[:, :3]
        
        # distance to target
        distance = torch.norm(robot_pos - target_pos, dim=-1)
        
        # reward for decreasing distance to target
        rew_distance = self.cfg.rew_scale_distance * distance
        
        # success reward
        rew_success = torch.zeros_like(rew_distance)
        is_success = distance < self.cfg.success_tolerance
        rew_success[is_success] = self.cfg.rew_scale_success
        
        # total reward
        total_reward = rew_distance + rew_success + self.cfg.rew_scale_alive
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # robot and target positions
        robot_pos = self.robot.data.root_pos_w[:, :3]
        target_pos = self.target.data.root_pos_w[:, :3]
        
        # distance to target
        distance = torch.norm(robot_pos - target_pos, dim=-1)
        
        # success condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reached_target = distance < self.cfg.success_tolerance
        
        # reset if out of bounds (workspace is a 2m cube centered at origin)
        out_of_bounds = torch.any(torch.abs(robot_pos) > 2.0, dim=1)

        return (reached_target | out_of_bounds), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # number of envs to reset
        num_resets = len(env_ids)

        # randomize robot root state
        robot_pos = self.robot.data.default_root_state[env_ids, :3]
        robot_pos[:, 0] = sample_uniform(self.cfg.robot_init_pos_range["x"][0], self.cfg.robot_init_pos_range["x"][1], (num_resets,), device=self.device)
        robot_pos[:, 1] = sample_uniform(self.cfg.robot_init_pos_range["y"][0], self.cfg.robot_init_pos_range["y"][1], (num_resets,), device=self.device)
        robot_pos[:, 2] = sample_uniform(self.cfg.robot_init_pos_range["z"][0], self.cfg.robot_init_pos_range["z"][1], (num_resets,), device=self.device)
        
        # set robot root state
        new_robot_root_state = self.robot.data.default_root_state[env_ids]
        new_robot_root_state[:, :3] = robot_pos
        # set velocity to zero
        new_robot_root_state[:, 3:] = 0.0
        self.robot.write_root_state_to_sim(new_robot_root_state, env_ids)

        # randomize target position
        target_pos = self.target.data.default_root_state[env_ids, :3]
        target_pos[:, 0] = sample_uniform(self.cfg.target_init_pos_range["x"][0], self.cfg.target_init_pos_range["x"][1], (num_resets,), device=self.device)
        target_pos[:, 1] = sample_uniform(self.cfg.target_init_pos_range["y"][0], self.cfg.target_init_pos_range["y"][1], (num_resets,), device=self.device)
        target_pos[:, 2] = sample_uniform(self.cfg.target_init_pos_range["z"][0], self.cfg.target_init_pos_range["z"][1], (num_resets,), device=self.device)

        # set target root state
        new_target_root_state = self.target.data.default_root_state[env_ids]
        new_target_root_state[:, :3] = target_pos
        self.target.write_root_state_to_sim(new_target_root_state, env_ids)

        # reset buffers
        super()._reset_idx(env_ids)
