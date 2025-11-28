# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple keyboard teleoperation for cx002 robot base using I/J/K/L.
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
import omni.appwindow
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

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
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=500.0,
            velocity_limit_sim=100.0,
            stiffness=5000.0,
            damping=2000.0,
        ),
    },
)


class Cx002SceneCfg(InteractiveSceneCfg):
    robot = CX002_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    try:
        import omni.kit.viewport.utility as viewport_utils
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            viewport.disable_interactions()
            print("[INFO]: Viewport interactions disabled for keyboard input.")
    except Exception as e:
        print(f"[INFO]: Could not disable viewport interactions: {e}")
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    scene_cfg = Cx002SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    
    sim.reset()

    robot = scene["robot"]
    
    base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    base_speed = 0.5
    
    joint_targets = robot.data.default_joint_pos.clone()
    
    bow_pitch_01_idx, _ = robot.find_joints("bow_pitch_joint_01")
    bow_pitch_02_idx, _ = robot.find_joints("bow_pitch_joint_02")
    bow_pitch_03_idx, _ = robot.find_joints("bow_pitch_joint_03")
    bow_yaw_idx, _ = robot.find_joints("bow_yaw_joint")
    
    if bow_pitch_01_idx is not None and len(bow_pitch_01_idx) > 0:
        idx = bow_pitch_01_idx[0].item() if hasattr(bow_pitch_01_idx[0], 'item') else bow_pitch_01_idx[0]
        joint_targets[:, idx] = 0.0
    if bow_pitch_02_idx is not None and len(bow_pitch_02_idx) > 0:
        idx = bow_pitch_02_idx[0].item() if hasattr(bow_pitch_02_idx[0], 'item') else bow_pitch_02_idx[0]
        joint_targets[:, idx] = 0.0
    if bow_pitch_03_idx is not None and len(bow_pitch_03_idx) > 0:
        idx = bow_pitch_03_idx[0].item() if hasattr(bow_pitch_03_idx[0], 'item') else bow_pitch_03_idx[0]
        joint_targets[:, idx] = 0.0
    if bow_yaw_idx is not None and len(bow_yaw_idx) > 0:
        idx = bow_yaw_idx[0].item() if hasattr(bow_yaw_idx[0], 'item') else bow_yaw_idx[0]
        joint_targets[:, idx] = 0.0
    
    print("[INFO]: Setting robot to upright pose...")
    root_state = robot.data.root_state_w.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))
    
    for i in range(300):
        robot.set_joint_position_target(joint_targets)
        robot.set_joint_velocity_target(torch.zeros_like(joint_targets))
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
        robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    
    print("[INFO]: Robot stabilized.")
    print("[INFO]: IMPORTANT: Click on the 3D viewport window to give it focus!")
    print("[INFO]: Then use I/J/K/L keys to move the base:")
    print("        I: Move forward")
    print("        K: Move backward")
    print("        J: Move left")
    print("        L: Move right")
    print("        Ctrl+C: Exit")

    input_interface = carb.input.acquire_input_interface()
    keys_pressed = {"i": False, "k": False, "j": False, "l": False}
    
    def keyboard_event_handler(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.KEY_I:
                keys_pressed["i"] = True
                print("[DEBUG]: I pressed - moving forward")
            elif event.input == carb.input.KeyboardInput.KEY_K:
                keys_pressed["k"] = True
                print("[DEBUG]: K pressed - moving backward")
            elif event.input == carb.input.KeyboardInput.KEY_J:
                keys_pressed["j"] = True
                print("[DEBUG]: J pressed - moving left")
            elif event.input == carb.input.KeyboardInput.KEY_L:
                keys_pressed["l"] = True
                print("[DEBUG]: L pressed - moving right")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input == carb.input.KeyboardInput.KEY_I:
                keys_pressed["i"] = False
            elif event.input == carb.input.KeyboardInput.KEY_K:
                keys_pressed["k"] = False
            elif event.input == carb.input.KeyboardInput.KEY_J:
                keys_pressed["j"] = False
            elif event.input == carb.input.KeyboardInput.KEY_L:
                keys_pressed["l"] = False
    
    try:
        import omni.appwindow
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                subscription = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
                print("[INFO]: Keyboard event handler registered successfully.")
                print("[INFO]: Click on the Isaac Lab viewport window to give it focus, then use I/J/K/L keys.")
            else:
                print("[WARNING]: Could not get keyboard from app window.")
        else:
            print("[WARNING]: Could not get app window.")
    except Exception as e:
        print(f"[WARNING]: Could not register keyboard events: {e}")
        print("[INFO]: Make sure to click on the Isaac Lab viewport window to give it focus!")

    sim_dt = sim.get_physics_dt()
    frame_count = 0
    keyboard_working = False
    
    while simulation_app.is_running():
        base_velocity.zero_()
        frame_count += 1

        if keys_pressed["i"]:
            base_velocity[:, 0] = base_speed
            if not keyboard_working:
                print("[DEBUG]: I key detected - moving forward!")
                keyboard_working = True
        if keys_pressed["k"]:
            base_velocity[:, 0] = -base_speed
            if not keyboard_working:
                print("[DEBUG]: K key detected - moving backward!")
                keyboard_working = True
        if keys_pressed["j"]:
            base_velocity[:, 1] = base_speed
            if not keyboard_working:
                print("[DEBUG]: J key detected - moving left!")
                keyboard_working = True
        if keys_pressed["l"]:
            base_velocity[:, 1] = -base_speed
            if not keyboard_working:
                print("[DEBUG]: L key detected - moving right!")
                keyboard_working = True
        
        if torch.any(base_velocity != 0):
            print(f"[DEBUG]: Applying base velocity: {base_velocity[0].cpu().numpy()}")
        
        if frame_count % 300 == 0 and not keyboard_working:
            print(f"[DEBUG]: Frame {frame_count}, keys_pressed: {keys_pressed}")
            print("[INFO]: If keys aren't working, make sure you clicked the viewport window!")

        robot.set_joint_position_target(joint_targets)
        robot.set_joint_velocity_target(torch.zeros_like(joint_targets))
        
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
        root_state[:, 7:10] = base_velocity
        robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))

        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
