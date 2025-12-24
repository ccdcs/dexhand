# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleoperation for cx002 robot.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for cx002 robot.")
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
            effort_limit_sim=5000.0,
            velocity_limit_sim=1000.0,
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
    except Exception:
        pass
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0/60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.5])

    scene_cfg = Cx002SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    
    # Get all joint names from the robot
    all_joint_names = robot.joint_names
    print(f"[INFO]: Found {len(all_joint_names)} joints:")
    for name in all_joint_names:
        print(f"  - {name}")
    
    def get_joint_idx(joint_name):
        """Get joint index by name."""
        idx, _ = robot.find_joints(joint_name)
        if idx is None or len(idx) == 0:
            return None
        val = idx[0]
        return val.item() if hasattr(val, 'item') else int(val)
    
    # Register all joints with their indices
    joint_indices = {}
    for joint_name in all_joint_names:
        joint_indices[joint_name] = get_joint_idx(joint_name)
    
    default_joint_targets = robot.data.default_joint_pos.clone()
    
    root_state = robot.data.root_state_w.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    default_root_orientation = root_state[:, 3:7].clone()
    robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))
    
    for i in range(20):
        robot.set_joint_position_target(default_joint_targets)
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    
    # Professional keyboard control scheme
    # Organized by body parts with logical key groupings
    
    print("[INFO]: Keyboard controls:")
    print("=" * 60)
    print("BOW/TORSO (4 joints):")
    print("  W/S - Bow Pitch 01")
    print("  Q/E - Bow Pitch 02")
    print("  Z/C - Bow Pitch 03")
    print("  R/F - Bow Yaw")
    print()
    print("HEAD (2 joints):")
    print("  T/G - Head Yaw")
    print("  Y/H - Head Pitch")
    print()
    print("ARMS (Toggle with TAB, then use same keys for both arms):")
    print("  TAB - Switch between Left/Right arm")
    print("  U/O - Shoulder Pitch")
    print("  1/2 - Shoulder Roll")
    print("  3/4 - Shoulder Yaw")
    print("  5/6 - Elbow Roll")
    print("  7/8 - Elbow Yaw")
    print("  9/0 - Wrist Roll")
    print("  -/= - Wrist Pitch")
    print()
    print("NOTE: All joint controls use position-based steps (1 degree per press)")
    print("      Hold keys for continuous movement")
    print("=" * 60)

    input_interface = carb.input.acquire_input_interface()
    
    keys_pressed = {
        # Bow/Torso
        "w": False, "s": False,  # bow_pitch_joint_01
        "q": False, "e": False,  # bow_pitch_joint_02
        "z": False, "c": False,  # bow_pitch_joint_03
        "r": False, "f": False,  # bow_yaw_joint
        # Head
        "t": False, "g": False,  # head_yaw_joint
        "y": False, "h": False,  # head_pitch_joint
        # Arm controls (shared for left/right) - using number row
        "u": False, "o": False,  # shoulder_pitch
        "1": False, "2": False,  # shoulder_roll
        "3": False, "4": False,  # shoulder_yaw
        "5": False, "6": False,  # elbow_roll
        "7": False, "8": False,  # elbow_yaw
        "9": False, "0": False,  # wrist_roll
        "-": False, "=": False,  # wrist_pitch
        # Arm toggle
        "tab": False,
    }
    
    # Track which arm is active (True = left, False = right)
    active_arm_left = True
    
    keys_just_pressed = set()
    
    def keyboard_event_handler(event, *args, **kwargs):
        nonlocal keys_just_pressed, active_arm_left
        input_str = str(event.input)
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Bow/Torso
            if input_str.endswith(".W"): 
                keys_pressed["w"] = True
                keys_just_pressed.add("w")
            elif input_str.endswith(".S"): 
                keys_pressed["s"] = True
                keys_just_pressed.add("s")
            elif input_str.endswith(".Q"): 
                keys_pressed["q"] = True
                keys_just_pressed.add("q")
            elif input_str.endswith(".E"): 
                keys_pressed["e"] = True
                keys_just_pressed.add("e")
            elif input_str.endswith(".Z"): 
                keys_pressed["z"] = True
                keys_just_pressed.add("z")
            elif input_str.endswith(".C"): 
                keys_pressed["c"] = True
                keys_just_pressed.add("c")
            elif input_str.endswith(".R"): 
                keys_pressed["r"] = True
                keys_just_pressed.add("r")
            elif input_str.endswith(".F"): 
                keys_pressed["f"] = True
                keys_just_pressed.add("f")
            # Head
            elif input_str.endswith(".T"): 
                keys_pressed["t"] = True
                keys_just_pressed.add("t")
            elif input_str.endswith(".G"): 
                keys_pressed["g"] = True
                keys_just_pressed.add("g")
            elif input_str.endswith(".Y"): 
                keys_pressed["y"] = True
                keys_just_pressed.add("y")
            elif input_str.endswith(".H"): 
                keys_pressed["h"] = True
                keys_just_pressed.add("h")
            # Arms
            elif input_str.endswith(".U"): 
                keys_pressed["u"] = True
                keys_just_pressed.add("u")
            elif input_str.endswith(".O"): 
                keys_pressed["o"] = True
                keys_just_pressed.add("o")
            elif input_str.endswith(".1") or input_str.endswith(".ONE"): 
                keys_pressed["1"] = True
                keys_just_pressed.add("1")
            elif input_str.endswith(".2") or input_str.endswith(".TWO"): 
                keys_pressed["2"] = True
                keys_just_pressed.add("2")
            elif input_str.endswith(".3") or input_str.endswith(".THREE"): 
                keys_pressed["3"] = True
                keys_just_pressed.add("3")
            elif input_str.endswith(".4") or input_str.endswith(".FOUR"): 
                keys_pressed["4"] = True
                keys_just_pressed.add("4")
            elif input_str.endswith(".5") or input_str.endswith(".FIVE"): 
                keys_pressed["5"] = True
                keys_just_pressed.add("5")
            elif input_str.endswith(".6") or input_str.endswith(".SIX"): 
                keys_pressed["6"] = True
                keys_just_pressed.add("6")
            elif input_str.endswith(".7") or input_str.endswith(".SEVEN"): 
                keys_pressed["7"] = True
                keys_just_pressed.add("7")
            elif input_str.endswith(".8") or input_str.endswith(".EIGHT"): 
                keys_pressed["8"] = True
                keys_just_pressed.add("8")
            elif input_str.endswith(".9") or input_str.endswith(".NINE"): 
                keys_pressed["9"] = True
                keys_just_pressed.add("9")
            elif input_str.endswith(".0") or input_str.endswith(".ZERO"): 
                keys_pressed["0"] = True
                keys_just_pressed.add("0")
            elif input_str.endswith(".MINUS") or input_str.endswith(".-"): 
                keys_pressed["-"] = True
                keys_just_pressed.add("-")
            elif input_str.endswith(".EQUALS") or input_str.endswith(".="): 
                keys_pressed["="] = True
                keys_just_pressed.add("=")
            # Arm toggle
            elif input_str.endswith(".TAB"): 
                if "tab" not in keys_just_pressed:
                    active_arm_left = not active_arm_left
                    arm_name = "LEFT" if active_arm_left else "RIGHT"
                    print(f"[INFO]: Switched to {arm_name} arm control")
                keys_pressed["tab"] = True
                keys_just_pressed.add("tab")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Bow/Torso
            elif input_str.endswith(".W"): keys_pressed["w"] = False
            elif input_str.endswith(".S"): keys_pressed["s"] = False
            elif input_str.endswith(".Q"): keys_pressed["q"] = False
            elif input_str.endswith(".E"): keys_pressed["e"] = False
            elif input_str.endswith(".Z"): keys_pressed["z"] = False
            elif input_str.endswith(".C"): keys_pressed["c"] = False
            elif input_str.endswith(".R"): keys_pressed["r"] = False
            elif input_str.endswith(".F"): keys_pressed["f"] = False
            # Head
            elif input_str.endswith(".T"): keys_pressed["t"] = False
            elif input_str.endswith(".G"): keys_pressed["g"] = False
            elif input_str.endswith(".Y"): keys_pressed["y"] = False
            elif input_str.endswith(".H"): keys_pressed["h"] = False
            # Arms
            elif input_str.endswith(".U"): keys_pressed["u"] = False
            elif input_str.endswith(".O"): keys_pressed["o"] = False
            elif input_str.endswith(".1") or input_str.endswith(".ONE"): keys_pressed["1"] = False
            elif input_str.endswith(".2") or input_str.endswith(".TWO"): keys_pressed["2"] = False
            elif input_str.endswith(".3") or input_str.endswith(".THREE"): keys_pressed["3"] = False
            elif input_str.endswith(".4") or input_str.endswith(".FOUR"): keys_pressed["4"] = False
            elif input_str.endswith(".5") or input_str.endswith(".FIVE"): keys_pressed["5"] = False
            elif input_str.endswith(".6") or input_str.endswith(".SIX"): keys_pressed["6"] = False
            elif input_str.endswith(".7") or input_str.endswith(".SEVEN"): keys_pressed["7"] = False
            elif input_str.endswith(".8") or input_str.endswith(".EIGHT"): keys_pressed["8"] = False
            elif input_str.endswith(".9") or input_str.endswith(".NINE"): keys_pressed["9"] = False
            elif input_str.endswith(".0") or input_str.endswith(".ZERO"): keys_pressed["0"] = False
            elif input_str.endswith(".MINUS") or input_str.endswith(".-"): keys_pressed["-"] = False
            elif input_str.endswith(".EQUALS") or input_str.endswith(".="): keys_pressed["="] = False
            elif input_str.endswith(".TAB"): keys_pressed["tab"] = False
    
    try:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    except Exception:
        pass

    sim_dt = sim.get_physics_dt()
    step_size = 0.0175  # 1 degree in radians (position-based control)
    
    joint_targets = default_joint_targets.clone()
    joint_offsets = torch.zeros_like(default_joint_targets)
    
    # Joint name mappings for keyboard control
    joint_key_mappings = {
        # Bow/Torso
        "bow_pitch_joint_01": ("w", "s"),
        "bow_pitch_joint_02": ("q", "e"),
        "bow_pitch_joint_03": ("z", "c"),
        "bow_yaw_joint": ("r", "f"),
        # Head
        "head_yaw_joint": ("t", "g"),
        "head_pitch_joint": ("y", "h"),
    }
    
    # Arm joint mappings (will be prefixed with left_ or right_)
    arm_joint_mappings = {
        "shoulder_pitch_joint": ("u", "o"),
        "shoulder_roll_joint": ("1", "2"),
        "shoulder_yaw_joint": ("3", "4"),
        "elbow_roll_joint": ("5", "6"),
        "elbow_yaw_joint": ("7", "8"),
        "wrist_roll_joint": ("9", "0"),
        "wrist_pitch_joint": ("-", "="),
    }
    
    while simulation_app.is_running():
        # Bow/Torso and Head joints (discrete steps on key press)
        for joint_name, (inc_key, dec_key) in joint_key_mappings.items():
            if joint_name in joint_indices and joint_indices[joint_name] is not None:
                idx = joint_indices[joint_name]
                if inc_key in keys_just_pressed:
                    joint_offsets[:, idx] += step_size
                if dec_key in keys_just_pressed:
                    joint_offsets[:, idx] -= step_size
        
        # Arm joints (discrete steps, with left/right toggle)
        arm_prefix = "left_" if active_arm_left else "right_"
        for joint_suffix, (inc_key, dec_key) in arm_joint_mappings.items():
            joint_name = f"{arm_prefix}{joint_suffix}"
            if joint_name in joint_indices and joint_indices[joint_name] is not None:
                idx = joint_indices[joint_name]
                if inc_key in keys_just_pressed:
                    joint_offsets[:, idx] += step_size
                if dec_key in keys_just_pressed:
                    joint_offsets[:, idx] -= step_size
        
        keys_just_pressed.clear()
        
        joint_targets = default_joint_targets + joint_offsets
        
        robot.set_joint_position_target(joint_targets)
        
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
