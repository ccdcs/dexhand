# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


DEXHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/limit_1.57/Assem_DexCo_2/Assem_DexCo_2/Assem_DexCo_2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "L1_joint": 0.0,
            "L2_pre_joint": 0.0,
            "L3_pre_joint": 0.0,
            "R1_joint": 0.0,
            "R2_pre_joint": 0.0,
            "R3_pre_joint": 0.0,
        },
        pos=(0.0, 5.0, 5.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={},
)


@configclass
class ReachingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 7
    observation_space = 13
    state_space = 13  # State space should match observation space for simplicity

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(
            2.0,
            -2.0,
            1.5,
        ),  # Camera position: (X, Y, Z) - Set back, to the side, and up
        lookat=(
            0.0,
            0.0,
            0.7,
        ),  # Target position: Look directly at the hand's root position
    )
    # robot
    robot: ArticulationCfg = DEXHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # target pose
    target_position = [0.5, 0.0, 0.5]
    target_orientation = [1.0, 0.0, 0.0, 0.0]

    # - reward scales
    rew_scale_pos_potential = 10.0
    rew_scale_orn_potential = 5.0

    rew_success_bonus = 100.0  # bonus for successful episode
    # - action penalty
    action_penalty = -0.001
    # - action scales (for delta actions)
    action_scale_pos = 0.1  # [m]
    action_scale_rot = 0.1  # [rad]
    # - reset states/conditions
    # TODO: do i need workspace
    workspace = [
        (-1.0, -1.0, 0.0),
        (1.0, 1.0, 1.0),
    ]
    pos_tolerance = 0.05  # [m]
    orn_tolerance = 0.1745  # [rad] (~10 degrees)
