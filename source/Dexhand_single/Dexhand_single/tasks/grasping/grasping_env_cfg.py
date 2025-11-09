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

DOFBOT_CONFIG = ArticulationCfg(
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
        pos=(0.25, -0.25, 2.0),
    ),
    actuators={
        "L1_act": ImplicitActuatorCfg(
            joint_names_expr=["L1_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "L2_act": ImplicitActuatorCfg(
            joint_names_expr=["L2_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "L3_act": ImplicitActuatorCfg(
            joint_names_expr=["L3_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R1_act": ImplicitActuatorCfg(
            joint_names_expr=["R1_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R2_act": ImplicitActuatorCfg(
            joint_names_expr=["R2_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R3_act": ImplicitActuatorCfg(
            joint_names_expr=["R3_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
    },
)


@configclass
class GraspingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 6
    observation_space = 12
    state_space = 6

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = DOFBOT_CONFIG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # custom parameters/scales
    # - controllable joint
    l1_dof_name = "L1_joint"
    l2_dof_name = "L2_pre_joint"
    l3_dof_name = "L3_pre_joint"
    r1_dof_name = "R1_joint"
    r2_dof_name = "R2_pre_joint"
    r3_dof_name = "R3_pre_joint"
    # - action scale
    action_scale = 1.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pose_error = -1.0
    # - reset states/conditions
    target_joint_angle_range = [-1.0, 1.0]

    R = 0.025
    D1 = 0.018
    D2 = 0.012
    D3 = 0.015
    D4 = 0.010
    L = 0.045
    S = 0.006

    action_space_low = [0.012, 0.012, 0.012, 0.012, 0.012, 0.012]
    action_space_high = [0.027, 0.027, 0.027, 0.027, 0.027, 0.027]
