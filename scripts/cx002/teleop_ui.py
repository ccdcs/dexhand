# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
UI-based teleoperation for cx002 robot using tkinter sliders.
Implements position-based control.
"""

import argparse
import queue
import threading
import tkinter as tk
from tkinter import ttk

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UI teleoperation for cx002 robot.")
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
            effort_limit_sim=5000.0,
            velocity_limit_sim=1000.0,
            stiffness=5000.0,
            damping=2000.0,
        ),
    },
)


class Cx002SceneCfg(InteractiveSceneCfg):

    robot = CX002_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class JointControlUI:
    """Tkinter UI for controlling robot joints."""
    
    def __init__(self, joint_data, joint_limits):
        self.joint_data = joint_data
        self.joint_limits = joint_limits
        self.root = tk.Tk()
        self.root.title("CX002 Robot Teleoperation - Position Control")
        self.root.geometry("1400x1600")
        
        self.sliders = {}
        self.value_labels = {}
        self.current_labels = {}
        
        self.create_ui()
        
    def create_ui(self):
        """Create the UI elements."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        row = 0
        
        ttk.Label(main_frame, text="Bow/Torso Joints", font=("Arial", 16, "bold")).grid(
            row=row, column=0, columnspan=4, pady=15, sticky=tk.W
        )
        row += 1
        
        joints_config = [
            ("bow_pitch_joint_01", "Bow Pitch 01", -2.2689, 0.87266),
            ("bow_pitch_joint_02", "Bow Pitch 02", -1.0472, 1.5708),
            ("bow_pitch_joint_03", "Bow Pitch 03", -2.0944, 1.4486),
            ("bow_yaw_joint", "Bow Yaw", -1.7453, 1.7453),
        ]
        
        for joint_name, display_name, min_val, max_val in joints_config:
            self.create_joint_slider(main_frame, row, joint_name, display_name, min_val, max_val)
            row += 1
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        ttk.Label(main_frame, text="Head Joints", font=("Arial", 16, "bold")).grid(
            row=row, column=0, columnspan=4, pady=15, sticky=tk.W
        )
        row += 1
        
        head_joints = [
            ("head_yaw_joint", "Head Yaw", -1.5708, 1.5708),
            ("head_pitch_joint", "Head Pitch", -0.8727, 0.3491),
        ]
        
        for joint_name, display_name, min_val, max_val in head_joints:
            self.create_joint_slider(main_frame, row, joint_name, display_name, min_val, max_val)
            row += 1
        
        reset_button = ttk.Button(main_frame, text="Reset All Joints", command=self.reset_all)
        reset_button.grid(row=row, column=0, columnspan=4, pady=20)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def create_joint_slider(self, parent, row, joint_name, display_name, min_val, max_val):
        """Create a slider for a joint."""
        label = ttk.Label(parent, text=display_name, width=22, font=("Arial", 14, "bold"))
        label.grid(row=row, column=0, padx=15, pady=12, sticky=tk.W)
        
        def make_callback(name):
            def callback(val):
                self.on_slider_change(name, val)
            return callback
        
        slider = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            command=make_callback(joint_name),
            length=500
        )
        slider.set(0.0)
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=15, pady=12)
        
        value_label = ttk.Label(parent, text="0.00", width=15, font=("Arial", 14))
        value_label.grid(row=row, column=2, padx=15, pady=12)
        
        current_label = ttk.Label(parent, text="Cur: 0.00", width=20, font=("Arial", 14))
        current_label.grid(row=row, column=3, padx=15, pady=12)
        
        self.sliders[joint_name] = slider
        self.value_labels[joint_name] = value_label
        self.current_labels[joint_name] = current_label
        
        self.joint_data["targets"][joint_name] = 0.0
        
    def on_slider_change(self, joint_name, value):
        """Handle slider value change with joint limit clamping."""
        val = float(value)
        with self.joint_data["lock"]:
            if joint_name in self.joint_limits:
                min_val, max_val = self.joint_limits[joint_name]
                val = max(min_val, min(max_val, val))
            self.joint_data["targets"][joint_name] = val
        if joint_name in self.value_labels:
            self.value_labels[joint_name].config(text=f"{val:.3f}")
        
    def reset_all(self):
        """Reset all sliders to zero."""
        for joint_name, slider in self.sliders.items():
            slider.set(0.0)
            self.joint_data["targets"][joint_name] = 0.0
            self.value_labels[joint_name].config(text="0.00")
            
    def update_current_positions(self, current_positions):
        """Update current joint position displays."""
        for joint_name, current_pos in current_positions.items():
            if joint_name in self.current_labels:
                self.current_labels[joint_name].config(text=f"Cur: {current_pos:.3f}")
    
    def update_slider_from_keyboard(self, joint_name, value):
        """Update slider position when keyboard changes joint target."""
        if joint_name in self.sliders:
            self.sliders[joint_name].set(value)
            if joint_name in self.value_labels:
                self.value_labels[joint_name].config(text=f"{value:.3f}")
    
    def poll_updates(self, update_queue):
        """Poll for updates from simulation."""
        try:
            while True:
                current_positions = update_queue.get_nowait()
                self.update_current_positions(current_positions)
        except queue.Empty:
            pass
    
    def run(self, update_queue, sim_step_callback, running_flag):
        """Start the UI"""
        def step_simulation():
            if running_flag["running"]:
                sim_step_callback()
                self.poll_updates(update_queue)
                # Run at ~200 Hz (5ms = 200 Hz) for smooth keyboard control
                self.root.after(5, step_simulation)
        
        self.root.after(50, step_simulation)
        self.root.mainloop()


def main():
    try:
        import omni.kit.viewport.utility as viewport_utils
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            viewport.disable_interactions()
    except Exception:
        pass
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0/200.0)
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
    
    def get_joint_idx(idx_result):
        if idx_result is None or len(idx_result) == 0:
            return None
        val = idx_result[0]
        return val.item() if hasattr(val, 'item') else int(val)
    
    joint_indices = {}
    joint_names = [
        "bow_pitch_joint_01", "bow_pitch_joint_02", "bow_pitch_joint_03", "bow_yaw_joint",
        "head_yaw_joint", "head_pitch_joint"
    ]
    
    for joint_name in joint_names:
        idx, _ = robot.find_joints(joint_name)
        joint_indices[joint_name] = get_joint_idx(idx)
    
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
    
    joint_limits = {
        "bow_pitch_joint_01": (-2.2689, 0.87266),
        "bow_pitch_joint_02": (-1.0472, 1.5708),
        "bow_pitch_joint_03": (-2.0944, 1.4486),
        "bow_yaw_joint": (-1.7453, 1.7453),
        "head_yaw_joint": (-1.5708, 1.5708),
        "head_pitch_joint": (-0.8727, 0.3491),
    }
    
    joint_data = {
        "targets": {},
        "lock": threading.Lock(),
        "running": True
    }
    
    # Initialize joint targets to default positions
    for joint_name in joint_names:
        if joint_name in joint_indices and joint_indices[joint_name] is not None:
            idx = joint_indices[joint_name]
            default_pos = default_joint_targets[0, idx].item()
            joint_data["targets"][joint_name] = default_pos
    
    update_queue = queue.Queue()
    
    # Setup keyboard controls (before UI so UI can access them)
    input_interface = carb.input.acquire_input_interface()
    keys_pressed = {
        "i": False, "k": False, "j": False, "l": False,  # Base movement
        "w": False, "s": False,  # bow_pitch_joint_01
        "q": False, "e": False,  # bow_pitch_joint_02
        "z": False, "c": False,  # bow_pitch_joint_03
        "r": False, "f": False,  # bow_yaw_joint
        "t": False, "g": False,  # head_yaw_joint
        "y": False, "h": False,  # head_pitch_joint
    }
    
    keys_just_pressed = set()
    # Position-based step size: 1 degree = ~0.0175 radians per frame when key is held
    keyboard_step_size = 0.0175  # 1 degree per step (can be adjusted to 0.5 deg = 0.0087)
    
    ui = JointControlUI(joint_data, joint_limits)
    
    # Initialize sliders to default positions
    for joint_name in joint_names:
        if joint_name in joint_data["targets"] and joint_name in ui.sliders:
            default_val = joint_data["targets"][joint_name]
            ui.sliders[joint_name].set(default_val)
            if joint_name in ui.value_labels:
                ui.value_labels[joint_name].config(text=f"{default_val:.3f}")
    
    def keyboard_event_handler(event, *args, **kwargs):
        input_str = str(event.input)
        
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if input_str.endswith(".I"): 
                keys_pressed["i"] = True
            elif input_str.endswith(".K"): 
                keys_pressed["k"] = True
            elif input_str.endswith(".J"): 
                keys_pressed["j"] = True
            elif input_str.endswith(".L"): 
                keys_pressed["l"] = True
            elif input_str.endswith(".W"): 
                keys_pressed["w"] = True
            elif input_str.endswith(".S"): 
                keys_pressed["s"] = True
            elif input_str.endswith(".Q"): 
                keys_pressed["q"] = True
            elif input_str.endswith(".E"): 
                keys_pressed["e"] = True
            elif input_str.endswith(".Z"): 
                keys_pressed["z"] = True
            elif input_str.endswith(".C"): 
                keys_pressed["c"] = True
            elif input_str.endswith(".R"): 
                keys_pressed["r"] = True
            elif input_str.endswith(".F"): 
                keys_pressed["f"] = True
            elif input_str.endswith(".T"): 
                keys_pressed["t"] = True
            elif input_str.endswith(".G"): 
                keys_pressed["g"] = True
            elif input_str.endswith(".Y"): 
                keys_pressed["y"] = True
            elif input_str.endswith(".H"): 
                keys_pressed["h"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if input_str.endswith(".I"): keys_pressed["i"] = False
            elif input_str.endswith(".K"): keys_pressed["k"] = False
            elif input_str.endswith(".J"): keys_pressed["j"] = False
            elif input_str.endswith(".L"): keys_pressed["l"] = False
            elif input_str.endswith(".W"): keys_pressed["w"] = False
            elif input_str.endswith(".S"): keys_pressed["s"] = False
            elif input_str.endswith(".Q"): keys_pressed["q"] = False
            elif input_str.endswith(".E"): keys_pressed["e"] = False
            elif input_str.endswith(".Z"): keys_pressed["z"] = False
            elif input_str.endswith(".C"): keys_pressed["c"] = False
            elif input_str.endswith(".R"): keys_pressed["r"] = False
            elif input_str.endswith(".F"): keys_pressed["f"] = False
            elif input_str.endswith(".T"): keys_pressed["t"] = False
            elif input_str.endswith(".G"): keys_pressed["g"] = False
            elif input_str.endswith(".Y"): keys_pressed["y"] = False
            elif input_str.endswith(".H"): keys_pressed["h"] = False
    
    try:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
                print("[INFO]: Keyboard event handler subscribed successfully.")
            else:
                print("[WARNING]: Could not get keyboard from app window.")
        else:
            print("[WARNING]: Could not get app window.")
    except Exception as e:
        print(f"[ERROR]: Failed to subscribe keyboard events: {e}")
        import traceback
        traceback.print_exc()
    
    def clamp_joint_value(joint_name, value):
        """Clamp joint value to limits for safety."""
        if joint_name in joint_limits:
            min_val, max_val = joint_limits[joint_name]
            return max(min_val, min(max_val, value))
        return value
    
    sim_dt = sim.get_physics_dt()
    
    def simulation_step():
        """Single simulation step - handles both keyboard and UI control."""
        nonlocal keys_pressed
        
        if not simulation_app.is_running() or not joint_data["running"]:
            return
        
        # Process keyboard inputs for joints (position-based, continuous on hold)
        with joint_data["lock"]:
            
            # bow_pitch_joint_01: W/S
            if "bow_pitch_joint_01" in joint_indices and joint_indices["bow_pitch_joint_01"] is not None:
                if keys_pressed.get("w", False):
                    joint_data["targets"]["bow_pitch_joint_01"] += keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_01"] = clamp_joint_value("bow_pitch_joint_01", joint_data["targets"]["bow_pitch_joint_01"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_01", joint_data["targets"]["bow_pitch_joint_01"])
                if keys_pressed.get("s", False):
                    joint_data["targets"]["bow_pitch_joint_01"] -= keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_01"] = clamp_joint_value("bow_pitch_joint_01", joint_data["targets"]["bow_pitch_joint_01"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_01", joint_data["targets"]["bow_pitch_joint_01"])
            
            if "bow_pitch_joint_02" in joint_indices and joint_indices["bow_pitch_joint_02"] is not None:
                if keys_pressed.get("q", False):
                    joint_data["targets"]["bow_pitch_joint_02"] += keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_02"] = clamp_joint_value("bow_pitch_joint_02", joint_data["targets"]["bow_pitch_joint_02"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_02", joint_data["targets"]["bow_pitch_joint_02"])
                if keys_pressed.get("e", False):
                    joint_data["targets"]["bow_pitch_joint_02"] -= keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_02"] = clamp_joint_value("bow_pitch_joint_02", joint_data["targets"]["bow_pitch_joint_02"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_02", joint_data["targets"]["bow_pitch_joint_02"])
            
            if "bow_pitch_joint_03" in joint_indices and joint_indices["bow_pitch_joint_03"] is not None:
                if keys_pressed.get("z", False):
                    joint_data["targets"]["bow_pitch_joint_03"] += keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_03"] = clamp_joint_value("bow_pitch_joint_03", joint_data["targets"]["bow_pitch_joint_03"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_03", joint_data["targets"]["bow_pitch_joint_03"])
                if keys_pressed.get("c", False):
                    joint_data["targets"]["bow_pitch_joint_03"] -= keyboard_step_size
                    joint_data["targets"]["bow_pitch_joint_03"] = clamp_joint_value("bow_pitch_joint_03", joint_data["targets"]["bow_pitch_joint_03"])
                    ui.update_slider_from_keyboard("bow_pitch_joint_03", joint_data["targets"]["bow_pitch_joint_03"])
            
            # bow_yaw_joint: R/F
            if "bow_yaw_joint" in joint_indices and joint_indices["bow_yaw_joint"] is not None:
                if keys_pressed.get("r", False):
                    joint_data["targets"]["bow_yaw_joint"] += keyboard_step_size
                    joint_data["targets"]["bow_yaw_joint"] = clamp_joint_value("bow_yaw_joint", joint_data["targets"]["bow_yaw_joint"])
                    ui.update_slider_from_keyboard("bow_yaw_joint", joint_data["targets"]["bow_yaw_joint"])
                if keys_pressed.get("f", False):
                    joint_data["targets"]["bow_yaw_joint"] -= keyboard_step_size
                    joint_data["targets"]["bow_yaw_joint"] = clamp_joint_value("bow_yaw_joint", joint_data["targets"]["bow_yaw_joint"])
                    ui.update_slider_from_keyboard("bow_yaw_joint", joint_data["targets"]["bow_yaw_joint"])
            
            # head_yaw_joint: T/G
            if "head_yaw_joint" in joint_indices and joint_indices["head_yaw_joint"] is not None:
                if keys_pressed.get("t", False):
                    joint_data["targets"]["head_yaw_joint"] += keyboard_step_size
                    joint_data["targets"]["head_yaw_joint"] = clamp_joint_value("head_yaw_joint", joint_data["targets"]["head_yaw_joint"])
                    ui.update_slider_from_keyboard("head_yaw_joint", joint_data["targets"]["head_yaw_joint"])
                if keys_pressed.get("g", False):
                    joint_data["targets"]["head_yaw_joint"] -= keyboard_step_size
                    joint_data["targets"]["head_yaw_joint"] = clamp_joint_value("head_yaw_joint", joint_data["targets"]["head_yaw_joint"])
                    ui.update_slider_from_keyboard("head_yaw_joint", joint_data["targets"]["head_yaw_joint"])
            
            # head_pitch_joint: Y/H
            if "head_pitch_joint" in joint_indices and joint_indices["head_pitch_joint"] is not None:
                if keys_pressed.get("y", False):
                    joint_data["targets"]["head_pitch_joint"] += keyboard_step_size
                    joint_data["targets"]["head_pitch_joint"] = clamp_joint_value("head_pitch_joint", joint_data["targets"]["head_pitch_joint"])
                    ui.update_slider_from_keyboard("head_pitch_joint", joint_data["targets"]["head_pitch_joint"])
                if keys_pressed.get("h", False):
                    joint_data["targets"]["head_pitch_joint"] -= keyboard_step_size
                    joint_data["targets"]["head_pitch_joint"] = clamp_joint_value("head_pitch_joint", joint_data["targets"]["head_pitch_joint"])
                    ui.update_slider_from_keyboard("head_pitch_joint", joint_data["targets"]["head_pitch_joint"])
            
            # Build joint targets from joint_data
            joint_targets = default_joint_targets.clone()
            
            for joint_name in joint_names:
                if joint_name in joint_indices and joint_indices[joint_name] is not None:
                    idx = joint_indices[joint_name]
                    current_pos = robot.data.joint_pos[0, idx].item()
                    
                    target_val = joint_data["targets"].get(joint_name, current_pos)
                    target_val = clamp_joint_value(joint_name, target_val)
                    
                    joint_targets[:, idx] = target_val
            
            current_positions = {}
            for joint_name in joint_names:
                if joint_name in joint_indices and joint_indices[joint_name] is not None:
                    current_positions[joint_name] = robot.data.joint_pos[0, joint_indices[joint_name]].item()
        
        # Handle base movement
        base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
        base_speed = 1.0 
        
        if keys_pressed.get("i", False):
            base_velocity[:, 0] = base_speed
        if keys_pressed.get("k", False):
            base_velocity[:, 0] = -base_speed
        if keys_pressed.get("j", False):
            base_velocity[:, 1] = base_speed
        if keys_pressed.get("l", False):
            base_velocity[:, 1] = -base_speed
        
        robot.set_joint_position_target(joint_targets)
        
        root_state = robot.data.root_state_w.clone()
        if torch.any(base_velocity != 0):
            root_state[:, 0:3] += base_velocity * sim_dt
            root_state[:, 3:7] = default_root_orientation
            robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))
        else:
            root_state[:, 3:7] = default_root_orientation
            robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))
        
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        
        try:
            update_queue.put_nowait(current_positions)
        except queue.Full:
            pass
    
    print("[INFO]: UI + Keyboard teleoperation started.")
    print("[INFO]: Control frequency: ~200 Hz")
    print("[INFO]: Keyboard controls:")
    print("  Base: I/K forward/back, J/L left/right")
    print("  Bow pitch 01: W/S")
    print("  Bow pitch 02: Q/E")
    print("  Bow pitch 03: Z/C")
    print("  Bow yaw: R/F")
    print("  Head yaw: T/G")
    print("  Head pitch: Y/H")
    
    try:
        ui.run(update_queue, simulation_step, joint_data)
    finally:
        joint_data["running"] = False


if __name__ == "__main__":
    main()
    simulation_app.close()

