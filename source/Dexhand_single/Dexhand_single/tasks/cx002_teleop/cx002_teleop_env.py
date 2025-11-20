# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .cx002_teleop_env_cfg import Cx002TeleopEnvCfg


class Cx002TeleopEnv(DirectRLEnv):
    cfg: Cx002TeleopEnvCfg

    def __init__(self, cfg: Cx002TeleopEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Store joint data references
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
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
        # For now, just store actions (no control yet)
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # For now, do nothing - just spawn the robot
        # Later: self.robot.set_joint_position_target(self.actions)
        pass

    def _get_observations(self) -> dict:
        # Simple observation: just return joint positions for now
        # Get all joint positions
        obs = self.joint_pos
        observations = {"policy": obs, "critic": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Simple reward: just keep alive for now
        rew_alive = self.cfg.rew_scale_alive * torch.ones(
            self.num_envs, device=self.device
        )
        return rew_alive

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # No termination conditions for now - just timeout
        out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset to default joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

