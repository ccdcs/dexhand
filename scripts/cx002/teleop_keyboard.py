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
    
    # Debug: Print all joint names to see if there are wheel joints
    all_joint_names = robot.joint_names
    print(f"[DEBUG]: Total joints: {len(all_joint_names)}")
    print(f"[DEBUG]: Joint names: {all_joint_names}")
    wheel_joints = [name for name in all_joint_names if 'wheel' in name.lower() or 'drive' in name.lower() or 'motor' in name.lower()]
    if wheel_joints:
        print(f"[DEBUG]: Found potential wheel/drive joints: {wheel_joints}")
    else:
        print("[DEBUG]: No wheel/drive joints found in joint names")
        print("[DEBUG]: Robot may use direct base movement (root state control)")
    
    base_displacement = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    base_speed = 10.0
    
    joint_targets = robot.data.default_joint_pos.clone()
    
    bow_pitch_01_idx, _ = robot.find_joints("bow_pitch_joint_01")
    bow_pitch_02_idx, _ = robot.find_joints("bow_pitch_joint_02")
    bow_pitch_03_idx, _ = robot.find_joints("bow_pitch_joint_03")
    bow_yaw_idx, _ = robot.find_joints("bow_yaw_joint")
    
    bow_pitch_01_idx_val = None
    bow_pitch_02_idx_val = None
    bow_pitch_03_idx_val = None
    bow_yaw_idx_val = None
    
    if bow_pitch_01_idx is not None and len(bow_pitch_01_idx) > 0:
        bow_pitch_01_idx_val = bow_pitch_01_idx[0].item() if hasattr(bow_pitch_01_idx[0], 'item') else bow_pitch_01_idx[0]
        joint_targets[:, bow_pitch_01_idx_val] = 0.0
    if bow_pitch_02_idx is not None and len(bow_pitch_02_idx) > 0:
        bow_pitch_02_idx_val = bow_pitch_02_idx[0].item() if hasattr(bow_pitch_02_idx[0], 'item') else bow_pitch_02_idx[0]
        joint_targets[:, bow_pitch_02_idx_val] = 0.0
    if bow_pitch_03_idx is not None and len(bow_pitch_03_idx) > 0:
        bow_pitch_03_idx_val = bow_pitch_03_idx[0].item() if hasattr(bow_pitch_03_idx[0], 'item') else bow_pitch_03_idx[0]
        joint_targets[:, bow_pitch_03_idx_val] = 0.0
    if bow_yaw_idx is not None and len(bow_yaw_idx) > 0:
        bow_yaw_idx_val = bow_yaw_idx[0].item() if hasattr(bow_yaw_idx[0], 'item') else bow_yaw_idx[0]
        joint_targets[:, bow_yaw_idx_val] = 0.0
    
    default_joint_targets = joint_targets.clone()
    
    print("[INFO]: Setting robot to upright pose...")
    root_state = robot.data.root_state_w.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    default_root_orientation = root_state[:, 3:7].clone()
    robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))
    
    for i in range(300):
        robot.set_joint_position_target(joint_targets)
        robot.set_joint_velocity_target(torch.zeros_like(joint_targets))
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    
    print("[INFO]: Robot stabilized.")
    print("[INFO]: IMPORTANT: Click on the 3D viewport window to give it focus!")
    print("[INFO]: Keyboard controls:")
    print("        Base Movement:")
    print("          I/K: Move forward/backward")
    print("          J/L: Move left/right")
    print("        Body Leaning (Bow Joints):")
    print("          W/S: Bow pitch 01 forward/backward")
    print("          A/D: Bow pitch 01 left/right")
    print("          Q/E: Bow pitch 02 up/down")
    print("          Z/C: Bow pitch 03 up/down")
    print("          R/F: Bow yaw left/right")
    print("        Ctrl+C: Exit")

    input_interface = carb.input.acquire_input_interface()
    keys_pressed = {
        "i": False, "k": False, "j": False, "l": False,  # Base movement
        "w": False, "s": False, "a": False, "d": False,  # Bow pitch 01 (forward/back, left/right lean)
        "q": False, "e": False,  # Bow pitch 02
        "z": False, "c": False,  # Bow pitch 03
        "r": False, "f": False,  # Bow yaw (left/right turn)
    }
    
    def keyboard_event_handler(event, *args, **kwargs):
        input_val = event.input
        input_str = str(input_val)
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Base movement
            if input_str.endswith(".I") or input_str == "KeyboardInput.I":
                keys_pressed["i"] = True
                print("[KEY PRESSED]: I - Moving forward")
            elif input_str.endswith(".K") or input_str == "KeyboardInput.K":
                keys_pressed["k"] = True
                print("[KEY PRESSED]: K - Moving backward")
            elif input_str.endswith(".J") or input_str == "KeyboardInput.J":
                keys_pressed["j"] = True
                print("[KEY PRESSED]: J - Moving left")
            elif input_str.endswith(".L") or input_str == "KeyboardInput.L":
                keys_pressed["l"] = True
                print("[KEY PRESSED]: L - Moving right")
            # Body leaning - Bow pitch 01
            elif input_str.endswith(".W") or input_str == "KeyboardInput.W":
                keys_pressed["w"] = True
                print("[KEY PRESSED]: W - Bow pitch 01 forward")
            elif input_str.endswith(".S") or input_str == "KeyboardInput.S":
                keys_pressed["s"] = True
                print("[KEY PRESSED]: S - Bow pitch 01 backward")
            elif input_str.endswith(".A") or input_str == "KeyboardInput.A":
                keys_pressed["a"] = True
                print("[KEY PRESSED]: A - Bow pitch 01 left")
            elif input_str.endswith(".D") or input_str == "KeyboardInput.D":
                keys_pressed["d"] = True
                print("[KEY PRESSED]: D - Bow pitch 01 right")
            # Bow pitch 02
            elif input_str.endswith(".Q") or input_str == "KeyboardInput.Q":
                keys_pressed["q"] = True
                print("[KEY PRESSED]: Q - Bow pitch 02 up")
            elif input_str.endswith(".E") or input_str == "KeyboardInput.E":
                keys_pressed["e"] = True
                print("[KEY PRESSED]: E - Bow pitch 02 down")
            # Bow pitch 03
            elif input_str.endswith(".Z") or input_str == "KeyboardInput.Z":
                keys_pressed["z"] = True
                print("[KEY PRESSED]: Z - Bow pitch 03 up")
            elif input_str.endswith(".C") or input_str == "KeyboardInput.C":
                keys_pressed["c"] = True
                print("[KEY PRESSED]: C - Bow pitch 03 down")
            # Bow yaw
            elif input_str.endswith(".R") or input_str == "KeyboardInput.R":
                keys_pressed["r"] = True
                print("[KEY PRESSED]: R - Bow yaw left")
            elif input_str.endswith(".F") or input_str == "KeyboardInput.F":
                keys_pressed["f"] = True
                print("[KEY PRESSED]: F - Bow yaw right")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if input_str.endswith(".I") or input_str == "KeyboardInput.I":
                keys_pressed["i"] = False
            elif input_str.endswith(".K") or input_str == "KeyboardInput.K":
                keys_pressed["k"] = False
            elif input_str.endswith(".J") or input_str == "KeyboardInput.J":
                keys_pressed["j"] = False
            elif input_str.endswith(".L") or input_str == "KeyboardInput.L":
                keys_pressed["l"] = False
            elif input_str.endswith(".W") or input_str == "KeyboardInput.W":
                keys_pressed["w"] = False
            elif input_str.endswith(".S") or input_str == "KeyboardInput.S":
                keys_pressed["s"] = False
            elif input_str.endswith(".A") or input_str == "KeyboardInput.A":
                keys_pressed["a"] = False
            elif input_str.endswith(".D") or input_str == "KeyboardInput.D":
                keys_pressed["d"] = False
            elif input_str.endswith(".Q") or input_str == "KeyboardInput.Q":
                keys_pressed["q"] = False
            elif input_str.endswith(".E") or input_str == "KeyboardInput.E":
                keys_pressed["e"] = False
            elif input_str.endswith(".Z") or input_str == "KeyboardInput.Z":
                keys_pressed["z"] = False
            elif input_str.endswith(".C") or input_str == "KeyboardInput.C":
                keys_pressed["c"] = False
            elif input_str.endswith(".R") or input_str == "KeyboardInput.R":
                keys_pressed["r"] = False
            elif input_str.endswith(".F") or input_str == "KeyboardInput.F":
                keys_pressed["f"] = False
    
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
    joint_speed = 50.0
    
    joint_offsets = torch.zeros_like(default_joint_targets)
    
    while simulation_app.is_running():
        base_displacement.zero_()
        frame_count += 1

        # Base movement
        if keys_pressed["i"]:
            base_displacement[:, 0] = base_speed
            if not keyboard_working:
                print("[DEBUG]: I key detected - moving forward!")
                keyboard_working = True
        if keys_pressed["k"]:
            base_displacement[:, 0] = -base_speed
            if not keyboard_working:
                print("[DEBUG]: K key detected - moving backward!")
                keyboard_working = True
        if keys_pressed["j"]:
            base_displacement[:, 1] = base_speed
            if not keyboard_working:
                print("[DEBUG]: J key detected - moving left!")
                keyboard_working = True
        if keys_pressed["l"]:
            base_displacement[:, 1] = -base_speed
            if not keyboard_working:
                print("[DEBUG]: L key detected - moving right!")
                keyboard_working = True
        
        # Body leaning - Bow pitch 01 (forward/backward, left/right) - additive to default
        if bow_pitch_01_idx_val is not None:
            if keys_pressed["w"]:
                joint_offsets[:, bow_pitch_01_idx_val] += joint_speed * sim_dt
                print(f"[BOW PITCH 01]: W pressed, offset: {joint_offsets[0, bow_pitch_01_idx_val].item():.3f}")
            if keys_pressed["s"]:
                joint_offsets[:, bow_pitch_01_idx_val] -= joint_speed * sim_dt
                print(f"[BOW PITCH 01]: S pressed, offset: {joint_offsets[0, bow_pitch_01_idx_val].item():.3f}")
            if keys_pressed["a"]:
                joint_offsets[:, bow_pitch_01_idx_val] += joint_speed * sim_dt * 0.5
                print(f"[BOW PITCH 01]: A pressed, offset: {joint_offsets[0, bow_pitch_01_idx_val].item():.3f}")
            if keys_pressed["d"]:
                joint_offsets[:, bow_pitch_01_idx_val] -= joint_speed * sim_dt * 0.5
                print(f"[BOW PITCH 01]: D pressed, offset: {joint_offsets[0, bow_pitch_01_idx_val].item():.3f}")
        
        # Bow pitch 02 - additive to default
        if bow_pitch_02_idx_val is not None:
            if keys_pressed["q"]:
                joint_offsets[:, bow_pitch_02_idx_val] += joint_speed * sim_dt
                print(f"[BOW PITCH 02]: Q pressed, offset: {joint_offsets[0, bow_pitch_02_idx_val].item():.3f}")
            if keys_pressed["e"]:
                joint_offsets[:, bow_pitch_02_idx_val] -= joint_speed * sim_dt
                print(f"[BOW PITCH 02]: E pressed, offset: {joint_offsets[0, bow_pitch_02_idx_val].item():.3f}")
        
        # Bow pitch 03 - additive to default
        if bow_pitch_03_idx_val is not None:
            if keys_pressed["z"]:
                joint_offsets[:, bow_pitch_03_idx_val] += joint_speed * sim_dt
                print(f"[BOW PITCH 03]: Z pressed, offset: {joint_offsets[0, bow_pitch_03_idx_val].item():.3f}")
            if keys_pressed["c"]:
                joint_offsets[:, bow_pitch_03_idx_val] -= joint_speed * sim_dt
                print(f"[BOW PITCH 03]: C pressed, offset: {joint_offsets[0, bow_pitch_03_idx_val].item():.3f}")
        
        # Bow yaw - additive to default
        if bow_yaw_idx_val is not None:
            if keys_pressed["r"]:
                joint_offsets[:, bow_yaw_idx_val] += joint_speed * sim_dt
                print(f"[BOW YAW]: R pressed, offset: {joint_offsets[0, bow_yaw_idx_val].item():.3f}")
            if keys_pressed["f"]:
                joint_offsets[:, bow_yaw_idx_val] -= joint_speed * sim_dt
                print(f"[BOW YAW]: F pressed, offset: {joint_offsets[0, bow_yaw_idx_val].item():.3f}")
        
        # Apply offsets to default targets
        joint_targets = default_joint_targets + joint_offsets
        
        if frame_count % 300 == 0 and not keyboard_working:
            print(f"[DEBUG]: Frame {frame_count}, keys_pressed: {keys_pressed}")
            print("[INFO]: If keys aren't working, make sure you clicked the viewport window!")

        robot.set_joint_position_target(joint_targets)
        robot.set_joint_velocity_target(torch.zeros_like(joint_targets))
        
        # Update base position (additive to current position, maintain default upright orientation)
        root_state = robot.data.root_state_w.clone()
        if torch.any(base_displacement != 0):
            root_state[:, 0:3] += base_displacement
            print(f"[MOVING]: Base position updated by {base_displacement[0].cpu().numpy()}, new pos: {root_state[0, 0:3].cpu().numpy()}")
        # Always maintain default upright orientation (don't force it to overwrite, just set it once)
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_state_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))

        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
