from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
        pos=(0.0, 2.5, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={},
)


BALL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ball",
    spawn=sim_utils.SphereCfg(
        radius=0.01,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
    ),
)


@configclass
class ReachingEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 7
    observation_space = 13
    state_space = 13

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True
    )
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, -2.0, 1.5), lookat=(0.0, 0.0, 1.0))
    robot: ArticulationCfg = DEXHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    ball: RigidObjectCfg = BALL_CFG

    target_spawn_base_offset = [0.0, -2.5, 0.0]
    target_spawn_pos_min = [-0.2, -0.2, -0.2]
    target_spawn_pos_max = [0.2, 0.2, 0.2]
    target_orientation = [1.0, 0.0, 0.0, 0.0]
    grasp_offset = [0.0, 0.0, -0.15]

    # rewards
    rew_scale_pos_potential = 10.0
    rew_scale_orn_potential = 10.0
    rew_success_bonus = 100.0
    action_penalty = -0.001

    # action scales (for delta actions)
    action_scale_pos = 0.01  # [m]
    action_scale_rot = 0.1  # [rad]

    # reset states/conditions
    pos_tolerance = 0.05  # [m]
    orn_tolerance = 0.1745  # [rad] (~10 degrees)
