# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple keyboard teleoperation for cx002 robot base using WASD.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for cx002 robot base.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

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
        pos=(0.0, 0.0, 1.0),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
    },
)


class Cx002SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = CX002_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    scene_cfg = Cx002SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    input_interface = carb.input.acquire_input_interface()
    
    base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    base_speed = 0.5

    print("[INFO]: Setup complete...")
    print("[INFO]: Click on the 3D viewport to give it focus, then use:")
    print("        W: Move forward")
    print("        S: Move backward")
    print("        A: Move left")
    print("        D: Move right")
    print("        Ctrl+C: Exit")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        base_velocity.zero_()

        try:
            keyboard = carb.input.get_keyboard()
            if keyboard:
                if input_interface.get_keyboard_value(keyboard, ord('w')) or input_interface.get_keyboard_value(keyboard, ord('W')):
                    base_velocity[:, 0] = base_speed
                if input_interface.get_keyboard_value(keyboard, ord('s')) or input_interface.get_keyboard_value(keyboard, ord('S')):
                    base_velocity[:, 0] = -base_speed
                if input_interface.get_keyboard_value(keyboard, ord('a')) or input_interface.get_keyboard_value(keyboard, ord('A')):
                    base_velocity[:, 1] = base_speed
                if input_interface.get_keyboard_value(keyboard, ord('d')) or input_interface.get_keyboard_value(keyboard, ord('D')):
                    base_velocity[:, 1] = -base_speed
        except:
            pass

        root_state = robot.data.root_state_w.clone()
        root_state[:, 7:10] = base_velocity
        robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))

        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
