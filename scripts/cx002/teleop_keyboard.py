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
    
    bow_pitch_01_idx, _ = robot.find_joints("bow_pitch_joint_01")
    bow_pitch_02_idx, _ = robot.find_joints("bow_pitch_joint_02")
    bow_pitch_03_idx, _ = robot.find_joints("bow_pitch_joint_03")
    bow_yaw_idx, _ = robot.find_joints("bow_yaw_joint")
    
    def get_joint_idx(idx_result):
        if idx_result is None or len(idx_result) == 0:
            return None
        val = idx_result[0]
        return val.item() if hasattr(val, 'item') else int(val)
    
    bow_pitch_01_idx_val = get_joint_idx(bow_pitch_01_idx)
    bow_pitch_02_idx_val = get_joint_idx(bow_pitch_02_idx)
    bow_pitch_03_idx_val = get_joint_idx(bow_pitch_03_idx)
    bow_yaw_idx_val = get_joint_idx(bow_yaw_idx)
    
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
    
    print("[INFO]: Keyboard controls:")
    print("  Base: I/K forward/back, J/L left/right")
    print("  Bow pitch 01: W/S forward/back, A/D left/right")
    print("  Bow pitch 02: Q/E up/down")
    print("  Bow pitch 03: Z/C up/down")
    print("  Bow yaw: R/F left/right")

    input_interface = carb.input.acquire_input_interface()
    keys_pressed = {
        "i": False, "k": False, "j": False, "l": False,
        "w": False, "s": False, "a": False, "d": False,
        "q": False, "e": False,
        "z": False, "c": False,
        "r": False, "f": False,
    }
    
    def keyboard_event_handler(event, *args, **kwargs):
        input_str = str(event.input)
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if input_str.endswith(".I"): keys_pressed["i"] = True
            elif input_str.endswith(".K"): keys_pressed["k"] = True
            elif input_str.endswith(".J"): keys_pressed["j"] = True
            elif input_str.endswith(".L"): keys_pressed["l"] = True
            elif input_str.endswith(".W"): keys_pressed["w"] = True
            elif input_str.endswith(".S"): keys_pressed["s"] = True
            elif input_str.endswith(".A"): keys_pressed["a"] = True
            elif input_str.endswith(".D"): keys_pressed["d"] = True
            elif input_str.endswith(".Q"): keys_pressed["q"] = True
            elif input_str.endswith(".E"): keys_pressed["e"] = True
            elif input_str.endswith(".Z"): keys_pressed["z"] = True
            elif input_str.endswith(".C"): keys_pressed["c"] = True
            elif input_str.endswith(".R"): keys_pressed["r"] = True
            elif input_str.endswith(".F"): keys_pressed["f"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if input_str.endswith(".I"): keys_pressed["i"] = False
            elif input_str.endswith(".K"): keys_pressed["k"] = False
            elif input_str.endswith(".J"): keys_pressed["j"] = False
            elif input_str.endswith(".L"): keys_pressed["l"] = False
            elif input_str.endswith(".W"): keys_pressed["w"] = False
            elif input_str.endswith(".S"): keys_pressed["s"] = False
            elif input_str.endswith(".A"): keys_pressed["a"] = False
            elif input_str.endswith(".D"): keys_pressed["d"] = False
            elif input_str.endswith(".Q"): keys_pressed["q"] = False
            elif input_str.endswith(".E"): keys_pressed["e"] = False
            elif input_str.endswith(".Z"): keys_pressed["z"] = False
            elif input_str.endswith(".C"): keys_pressed["c"] = False
            elif input_str.endswith(".R"): keys_pressed["r"] = False
            elif input_str.endswith(".F"): keys_pressed["f"] = False
    
    try:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    except Exception:
        pass

    sim_dt = sim.get_physics_dt()
    joint_speed = 200
    base_speed = 10
    
    joint_targets = default_joint_targets.clone()
    joint_offsets = torch.zeros_like(default_joint_targets)
    base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    
    while simulation_app.is_running():
        base_velocity.zero_()
        joint_offsets.zero_()
        
        if keys_pressed["i"]:
            base_velocity[:, 0] = base_speed
        if keys_pressed["k"]:
            base_velocity[:, 0] = -base_speed
        if keys_pressed["j"]:
            base_velocity[:, 1] = base_speed
        if keys_pressed["l"]:
            base_velocity[:, 1] = -base_speed
        
        if bow_pitch_01_idx_val is not None:
            if keys_pressed["w"]:
                joint_offsets[:, bow_pitch_01_idx_val] += joint_speed * sim_dt
            if keys_pressed["s"]:
                joint_offsets[:, bow_pitch_01_idx_val] -= joint_speed * sim_dt
            if keys_pressed["a"]:
                joint_offsets[:, bow_pitch_01_idx_val] += joint_speed * sim_dt * 0.5
            if keys_pressed["d"]:
                joint_offsets[:, bow_pitch_01_idx_val] -= joint_speed * sim_dt * 0.5
        
        if bow_pitch_02_idx_val is not None:
            if keys_pressed["q"]:
                joint_offsets[:, bow_pitch_02_idx_val] += joint_speed * sim_dt
            if keys_pressed["e"]:
                joint_offsets[:, bow_pitch_02_idx_val] -= joint_speed * sim_dt
        
        if bow_pitch_03_idx_val is not None:
            if keys_pressed["z"]:
                joint_offsets[:, bow_pitch_03_idx_val] += joint_speed * sim_dt
            if keys_pressed["c"]:
                joint_offsets[:, bow_pitch_03_idx_val] -= joint_speed * sim_dt
        
        if bow_yaw_idx_val is not None:
            if keys_pressed["r"]:
                joint_offsets[:, bow_yaw_idx_val] += joint_speed * sim_dt
            if keys_pressed["f"]:
                joint_offsets[:, bow_yaw_idx_val] -= joint_speed * sim_dt
        
        joint_targets = default_joint_targets + joint_offsets
        
        robot.set_joint_position_target(joint_targets)
        
        if torch.any(base_velocity != 0):
            root_state = robot.data.root_state_w.clone()
            root_state[:, 0:3] += base_velocity * sim_dt
            root_state[:, 3:7] = default_root_orientation
            robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))
        else:
            root_state = robot.data.root_state_w.clone()
            root_state[:, 3:7] = default_root_orientation
            robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
