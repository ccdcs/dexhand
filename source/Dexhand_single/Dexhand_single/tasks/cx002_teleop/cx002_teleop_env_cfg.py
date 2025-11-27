# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

# CX002 robot configuration using the new USD file
CX002_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/cx002_description_new/cx002_robot/cx002_robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Spawn at height 1.0m
    ),
    # Basic actuators for main arm joints (can be expanded later)
    actuators={
        "left_arm_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_roll_joint",
                "left_elbow_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "right_arm_act": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_roll_joint",
                "right_elbow_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
    },
)


@configclass
class Cx002TeleopEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition (will be updated when we know which joints to control)
    action_space = 14  # 7 joints per arm (left + right)
    observation_space = 14
    state_space = 14

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CX002_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    # custom parameters/scales
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0

