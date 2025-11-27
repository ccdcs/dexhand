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
    keys_pressed = {"w": False, "s": False, "a": False, "d": False}

    def handle_action_event(event):
        if event.action == "move_forward":
            keys_pressed["w"] = True
        elif event.action == "move_backward":
            keys_pressed["s"] = True
        elif event.action == "move_left":
            keys_pressed["a"] = True
        elif event.action == "move_right":
            keys_pressed["d"] = True
        return True

    def handle_action_release_event(event):
        if event.action == "move_forward":
            keys_pressed["w"] = False
        elif event.action == "move_backward":
            keys_pressed["s"] = False
        elif event.action == "move_left":
            keys_pressed["a"] = False
        elif event.action == "move_right":
            keys_pressed["d"] = False
        return True

    try:
        action_mapping_set = input_interface.create_action_mapping_set("teleop_actions")
        keyboard = carb.input.get_keyboard()
        if keyboard:
            input_interface.add_action_mapping(action_mapping_set, "move_forward", keyboard, ord('w'))
            input_interface.add_action_mapping(action_mapping_set, "move_backward", keyboard, ord('s'))
            input_interface.add_action_mapping(action_mapping_set, "move_left", keyboard, ord('a'))
            input_interface.add_action_mapping(action_mapping_set, "move_right", keyboard, ord('d'))
            
            input_interface.subscribe_to_action_events(action_mapping_set, "move_forward", handle_action_event)
            input_interface.subscribe_to_action_events(action_mapping_set, "move_backward", handle_action_event)
            input_interface.subscribe_to_action_events(action_mapping_set, "move_left", handle_action_event)
            input_interface.subscribe_to_action_events(action_mapping_set, "move_right", handle_action_event)
    except Exception as e:
        print(f"[WARNING]: Could not set up keyboard events: {e}")
        print("[INFO]: Keyboard controls may not work. Continuing anyway...")

    print("[INFO]: Setup complete...")
    print("[INFO]: Keyboard controls:")
    print("        W: Move forward")
    print("        S: Move backward")
    print("        A: Move left")
    print("        D: Move right")
    print("        Ctrl+C: Exit")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        base_velocity.zero_()

        if keys_pressed["w"]:
            base_velocity[:, 0] = base_speed
        if keys_pressed["s"]:
            base_velocity[:, 0] = -base_speed
        if keys_pressed["a"]:
            base_velocity[:, 1] = base_speed
        if keys_pressed["d"]:
            base_velocity[:, 1] = -base_speed

        root_state = robot.data.root_state_w.clone()
        root_state[:, 7:10] = base_velocity
        robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))

        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
