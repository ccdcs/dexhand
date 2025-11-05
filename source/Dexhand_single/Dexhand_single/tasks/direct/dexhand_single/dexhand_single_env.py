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

from .dexhand_single_env_cfg import DexhandSingleEnvCfg
from .kinematics.compute_joint import compute_jacobian, inverse_kinematics_with_jacobian
from .kinematics.compliance_utils import compliance_cl


class DexhandSingleEnv(DirectRLEnv):
    cfg: DexhandSingleEnvCfg

    def __init__(self, cfg: DexhandSingleEnvCfg, render_mode: str | None = None, **kwargs):
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
        # self.actions = actions.clone()
        N = 50  # 每个自由度变化的步数
        total_dim = 6
        # 你可以根据实际可达区间设置
        l_min, l_max = 0.015, 0.035

        # 计算当前要动哪个自由度
        phase = (self.test_count // N) % total_dim

        # 生成 action
        action = torch.full((self.num_envs, total_dim), (l_min + l_max) / 2, device=self.device)
        # 当前自由度做线性变化
        t = (self.test_count % N) / (N - 1)
        value = l_min + (l_max - l_min) * t
        action[:, phase] = value
        self.actions = action.clone()

    def _apply_action(self) -> None:
        l1 = self.actions[:, 0].cpu().numpy()
        l2 = self.actions[:, 1].cpu().numpy()
        l3 = self.actions[:, 2].cpu().numpy()
        r1 = self.actions[:, 3].cpu().numpy()
        r2 = self.actions[:, 4].cpu().numpy()
        r3 = self.actions[:, 5].cpu().numpy()
        print("--------------------------------")
        print(self.actions[:, ])
        print("--------------------------------")
        theta1_list, theta2_list, theta3_list = [], [], []
        theta4_list, theta5_list, theta6_list = [], [], []
        for i in range(self.num_envs):
            # 获取当前关节角度作为初值
            current_theta1 = self.joint_pos[i, self.l1_idx[0]].item()
            current_theta2 = self.joint_pos[i, self.l2_idx[0]].item()
            # 逆解
            t1, t2, t3 = inverse_kinematics_with_jacobian(
                l1[i], l2[i], l3[i],
                self.cfg.R, self.cfg.D1, self.cfg.D2, self.cfg.D3, self.cfg.D4, self.cfg.L, self.cfg.S,
                current_theta1=current_theta1,
                current_theta2=current_theta2
            )
            # 失败时用当前角度
            if t1 is None: t1 = current_theta1
            if t2 is None: t2 = current_theta2
            if t3 is None: t3 = self.joint_pos[i, self.l3_idx[0]].item()
            theta1_list.append(t1)
            theta2_list.append(t2)
            theta3_list.append(t3)
            current_theta4 = self.joint_pos[i, self.r1_idx[0]].item()
            current_theta5 = self.joint_pos[i, self.r2_idx[0]].item()
            t4, t5, t6 = inverse_kinematics_with_jacobian(
                r1[i], r2[i], r3[i],
                self.cfg.R, self.cfg.D1, self.cfg.D2, self.cfg.D3, self.cfg.D4, self.cfg.L, self.cfg.S,
                current_theta1=current_theta4,
                current_theta2=current_theta5
            )
            if t4 is None: t4 = current_theta4
            if t5 is None: t5 = current_theta5
            if t6 is None: t6 = self.joint_pos[i, self.r3_idx[0]].item()
            theta4_list.append(t4)
            theta5_list.append(t5)
            theta6_list.append(t6)
            # 转成 tensor
        theta_tensor = self.joint_pos.clone()
        theta_tensor[:, self.l1_idx[0]] = torch.tensor(theta1_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
        theta_tensor[:, self.l2_idx[0]] = torch.tensor(theta2_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
        theta_tensor[:, self.l3_idx[0]] = torch.tensor(theta3_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
        theta_tensor[:, self.r1_idx[0]] = torch.tensor(theta4_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
        theta_tensor[:, self.r2_idx[0]] = torch.tensor(theta5_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
        theta_tensor[:, self.r3_idx[0]] = torch.tensor(theta6_list, device=self.joint_pos.device, dtype=self.joint_pos.dtype)
    
        self.robot.set_joint_position_target(theta_tensor)
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self.l1_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.l2_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.l3_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.r1_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.r2_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self.r3_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs, "critic": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_l1_pos,
            self.cfg.rew_scale_l2_pos,
            self.cfg.rew_scale_l3_pos,
            self.cfg.rew_scale_r1_pos,
            self.cfg.rew_scale_r2_pos,
            self.cfg.rew_scale_r3_pos,
            self.joint_pos[:, self.l1_idx[0]],
            self.joint_vel[:, self.l1_idx[0]],
            self.joint_pos[:, self.l2_idx[0]],
            self.joint_vel[:, self.l2_idx[0]],
            self.joint_pos[:, self.l3_idx[0]],
            self.joint_vel[:, self.l3_idx[0]],
            self.joint_pos[:, self.r1_idx[0]],
            self.joint_vel[:, self.r1_idx[0]],
            self.joint_pos[:, self.r2_idx[0]],
            self.joint_vel[:, self.r2_idx[0]],
            self.joint_pos[:, self.r3_idx[0]],
            self.joint_vel[:, self.r3_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.l1_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.l2_idx]) > math.pi / 2, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.l3_idx]) > math.pi / 2, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.r1_idx]) > math.pi / 2, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.r2_idx]) > math.pi / 2, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.r3_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self.l1_idx] += sample_uniform(
            self.cfg.initial_l1_angle_range[0] * math.pi,
            self.cfg.initial_l1_angle_range[1] * math.pi,
            joint_pos[:, self.l1_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self.l2_idx] += sample_uniform(
            self.cfg.initial_l2_angle_range[0] * math.pi,
            self.cfg.initial_l2_angle_range[1] * math.pi,
            joint_pos[:, self.l2_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self.l3_idx] += sample_uniform(
            self.cfg.initial_l3_angle_range[0] * math.pi,
            self.cfg.initial_l3_angle_range[1] * math.pi,
            joint_pos[:, self.l3_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self.r1_idx] += sample_uniform(
            self.cfg.initial_r1_angle_range[0] * math.pi,
            self.cfg.initial_r1_angle_range[1] * math.pi,
            joint_pos[:, self.r1_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self.r2_idx] += sample_uniform(
            self.cfg.initial_r2_angle_range[0] * math.pi,
            self.cfg.initial_r2_angle_range[1] * math.pi,
            joint_pos[:, self.r2_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self.r3_idx] += sample_uniform(
            self.cfg.initial_r3_angle_range[0] * math.pi,
            self.cfg.initial_r3_angle_range[1] * math.pi,
            joint_pos[:, self.r3_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_l1_pos: float,
    rew_scale_l2_pos: float,
    rew_scale_l3_pos: float,
    rew_scale_r1_pos: float,
    rew_scale_r2_pos: float,
    rew_scale_r3_pos: float,
    l1_pos: torch.Tensor,
    l1_vel: torch.Tensor,
    l2_pos: torch.Tensor,
    l2_vel: torch.Tensor,
    l3_pos: torch.Tensor,
    l3_vel: torch.Tensor,
    r1_pos: torch.Tensor,
    r1_vel: torch.Tensor,
    r2_pos: torch.Tensor,
    r2_vel: torch.Tensor,
    r3_pos: torch.Tensor,
    r3_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_l1_pos = rew_scale_l1_pos * torch.sum(torch.square(l1_pos).unsqueeze(dim=1), dim=-1)
    rew_l2_pos = rew_scale_l2_pos * torch.sum(torch.square(l2_pos).unsqueeze(dim=1), dim=-1)
    rew_l3_pos = rew_scale_l3_pos * torch.sum(torch.square(l3_pos).unsqueeze(dim=1), dim=-1)
    rew_r1_pos = rew_scale_r1_pos * torch.sum(torch.square(r1_pos).unsqueeze(dim=1), dim=-1)
    rew_r2_pos = rew_scale_r2_pos * torch.sum(torch.square(r2_pos).unsqueeze(dim=1), dim=-1)
    rew_r3_pos = rew_scale_r3_pos * torch.sum(torch.square(r3_pos).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_l1_pos + rew_l2_pos + rew_l3_pos + rew_r1_pos + rew_r2_pos + rew_r3_pos
    return total_reward
