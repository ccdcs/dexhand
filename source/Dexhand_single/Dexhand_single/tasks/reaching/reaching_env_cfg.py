# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

##
# Pre-defined configs
##

TARGET_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Target",
    spawn=sim_utils.SphereCfg(
        radius=0.05,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ),
)

DEXHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/limit_1.57/Assem_DexCo_2/Assem_DexCo_2/Assem_DexCo_2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fixed_base=False,  # Make sure the base is floating
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Start at a reasonable height
    ),
    actuators={},  # No actuators, controlled by external forces
)


@configclass
class ReachingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 8.0
    # - spaces definition
    action_space = 6  # 3 forces, 3 torques
    observation_space = 16  # 13 robot state, 3 target pos
    state_space = 13  # robot root state

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        scene_objects={"target": TARGET_CFG},
        articulations={"robot": DEXHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")},
    )

    # custom parameters/scales
    # - action scale
    action_scale = 0.5
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_distance = -1.0
    rew_scale_success = 5.0
    # - reset states/conditions
    robot_init_pos_range = {"x": [-0.2, 0.2], "y": [-0.2, 0.2], "z": [0.3, 0.7]}
    target_init_pos_range = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.3, 1.0]}

    # success tolerance
    success_tolerance = 0.05
