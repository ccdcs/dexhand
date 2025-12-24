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

import threading
import tkinter as tk
from tkinter import ttk

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


class JointControlUI:
    def __init__(self, joint_data, joint_limits):
        self.joint_data = joint_data
        self.joint_limits = joint_limits
        self.root = tk.Tk()
        self.root.title("CX002 Teleoperation UI")
        self.root.geometry("1200x900")

        self.sliders = {}
        self.value_labels = {}

        self._build_ui()

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(main_frame, text="Bow / Torso", font=("Arial", 14, "bold")).grid(
            row=row, column=0, columnspan=3, pady=10, sticky=tk.W
        )
        row += 1

        bow_joints = [
            ("bow_pitch_joint_01", "Bow Pitch 01", -2.2689, 0.87266),
            ("bow_pitch_joint_02", "Bow Pitch 02", -1.0472, 1.5708),
            ("bow_pitch_joint_03", "Bow Pitch 03", -2.0944, 1.4486),
            ("bow_yaw_joint", "Bow Yaw", -1.7453, 1.7453),
        ]

        for joint_name, label, jmin, jmax in bow_joints:
            self._add_slider(main_frame, row, joint_name, label, jmin, jmax)
            row += 1

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        row += 1

        ttk.Label(main_frame, text="Head", font=("Arial", 14, "bold")).grid(
            row=row, column=0, columnspan=3, pady=10, sticky=tk.W
        )
        row += 1

        head_joints = [
            ("head_yaw_joint", "Head Yaw", -1.5708, 1.5708),
            ("head_pitch_joint", "Head Pitch", -0.8727, 0.3491),
        ]

        for joint_name, label, jmin, jmax in head_joints:
            self._add_slider(main_frame, row, joint_name, label, jmin, jmax)
            row += 1

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def _add_slider(self, parent, row, joint_name, display_name, jmin, jmax):
        ttk.Label(parent, text=display_name, width=18, font=("Arial", 12, "bold")).grid(
            row=row, column=0, padx=10, pady=8, sticky=tk.W
        )

        slider = ttk.Scale(
            parent,
            from_=jmin,
            to=jmax,
            orient=tk.HORIZONTAL,
            length=450,
            command=lambda val, n=joint_name: self.on_slider_change(n, val),
        )
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=8)

        value_label = ttk.Label(parent, text="0.000", width=10, font=("Arial", 12))
        value_label.grid(row=row, column=2, padx=10, pady=8)

        self.sliders[joint_name] = slider
        self.value_labels[joint_name] = value_label

    def on_slider_change(self, joint_name, value):
        val = float(value)
        if joint_name in self.joint_limits:
            jmin, jmax = self.joint_limits[joint_name]
            val = max(jmin, min(jmax, val))
        with self.joint_data["lock"]:
            self.joint_data["slider_targets"][joint_name] = val
        if joint_name in self.value_labels:
            self.value_labels[joint_name].config(text=f"{val:.3f}")

    def _on_key_press(self, event):
        key = event.keysym.lower()
        with self.joint_data["lock"]:
            if "keys_pressed" in self.joint_data and key in self.joint_data["keys_pressed"]:
                self.joint_data["keys_pressed"][key] = True

    def _on_key_release(self, event):
        key = event.keysym.lower()
        with self.joint_data["lock"]:
            if "keys_pressed" in self.joint_data and key in self.joint_data["keys_pressed"]:
                self.joint_data["keys_pressed"][key] = False

    def set_initial_values(self, initial_targets):
        for joint_name, target in initial_targets.items():
            if joint_name in self.sliders:
                self.sliders[joint_name].set(target)
                if joint_name in self.value_labels:
                    self.value_labels[joint_name].config(text=f"{target:.3f}")

    def run(self, simulation_step, joint_data):
        def loop():
            if not joint_data["running"]:
                return
            simulation_step()
            self.root.after(5, loop)
        self.root.after(5, loop)
        self.root.mainloop()


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
    
    print("[INFO]: Keyboard controls and UI:")
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

    keys_template = {
        "w": False, "s": False,
        "q": False, "e": False,
        "z": False, "c": False,
        "r": False, "f": False,
        "t": False, "g": False,
        "y": False, "h": False,
        "u": False, "o": False,
        "1": False, "2": False,
        "3": False, "4": False,
        "5": False, "6": False,
        "7": False, "8": False,
        "9": False, "0": False,
        "-": False, "=": False,
        "tab": False,
    }
    
    active_arm_left = True
    
    joint_limits = {
        "bow_pitch_joint_01": (-2.2689, 0.87266),
        "bow_pitch_joint_02": (-1.0472, 1.5708),
        "bow_pitch_joint_03": (-2.0944, 1.4486),
        "bow_yaw_joint": (-1.7453, 1.7453),
        "head_yaw_joint": (-1.5708, 1.5708),
        "head_pitch_joint": (-0.8727, 0.3491),
    }
    
    joint_data = {
        "slider_targets": {},
        "keys_pressed": keys_template.copy(),
        "lock": threading.Lock(),
        "running": True,
    }
    
    ui = JointControlUI(joint_data, joint_limits, [])
    
    initial_slider_targets = {}
    for name in joint_limits.keys():
        if name in joint_indices and joint_indices[name] is not None:
            idx = joint_indices[name]
            initial_slider_targets[name] = default_joint_targets[0, idx].item()
            joint_data["slider_targets"][name] = initial_slider_targets[name]
    
    ui.set_initial_values(initial_slider_targets)

    sim_dt = sim.get_physics_dt()
    step_size = 0.0175
    
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
    
    # Arm joint mappings (prefixed with left_ or right_)
    arm_joint_mappings = {
        "shoulder_pitch_joint": ("u", "o"),
        "shoulder_roll_joint": ("1", "2"),
        "shoulder_yaw_joint": ("3", "4"),
        "elbow_roll_joint": ("5", "6"),
        "elbow_yaw_joint": ("7", "8"),
        "wrist_roll_joint": ("9", "0"),
        "wrist_pitch_joint": ("-", "="),
    }
    
    def simulation_step():
        nonlocal active_arm_left, joint_offsets

        if not simulation_app.is_running():
            joint_data["running"] = False
            return

        with joint_data["lock"]:
            kp = joint_data["keys_pressed"]

            for joint_name, (inc_key, dec_key) in joint_key_mappings.items():
                if joint_name in joint_indices and joint_indices[joint_name] is not None:
                    idx = joint_indices[joint_name]
                    if kp.get(inc_key, False):
                        joint_offsets[:, idx] += step_size
                    if kp.get(dec_key, False):
                        joint_offsets[:, idx] -= step_size

            arm_prefix = "left_" if active_arm_left else "right_"
            if kp.get("tab", False):
                active_arm_left = not active_arm_left
                joint_data["keys_pressed"]["tab"] = False

            for joint_suffix, (inc_key, dec_key) in arm_joint_mappings.items():
                joint_name = f"{arm_prefix}{joint_suffix}"
                if joint_name in joint_indices and joint_indices[joint_name] is not None:
                    idx = joint_indices[joint_name]
                    if kp.get(inc_key, False):
                        joint_offsets[:, idx] += step_size
                    if kp.get(dec_key, False):
                        joint_offsets[:, idx] -= step_size
        
        joint_targets = default_joint_targets + joint_offsets
        
        with joint_data["lock"]:
            for joint_name, slider_val in joint_data["slider_targets"].items():
                if joint_name in joint_indices and joint_indices[joint_name] is not None:
                    idx = joint_indices[joint_name]
                    if joint_name in joint_limits:
                        jmin, jmax = joint_limits[joint_name]
                        slider_val = max(jmin, min(jmax, slider_val))
                    joint_targets[:, idx] = slider_val
        
        robot.set_joint_position_target(joint_targets)
        
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)

    ui.run(simulation_step, joint_data)


if __name__ == "__main__":
    main()
    simulation_app.close()
