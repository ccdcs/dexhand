# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import threading
import tkinter as tk
from tkinter import ttk

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Combined UI + keyboard teleoperation for cx002 robot (Tk focus-safe).")
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


class JointControlUI:
    def __init__(self, joint_data, joint_limits, keys_pressed, on_key_press=None, on_key_release=None):
        self.joint_data = joint_data
        self.joint_limits = joint_limits
        self.keys_pressed = keys_pressed

        self.root = tk.Tk()
        self.root.title("CX002 Teleoperation UI")
        self.root.geometry("1200x900")

        self.sliders = {}
        self.value_labels = {}

        if on_key_press is not None:
            self.root.bind("<KeyPress>", on_key_press)
        if on_key_release is not None:
            self.root.bind("<KeyRelease>", on_key_release)
        self.root.focus_set()

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(main_frame, text="All Joints", font=("Arial", 14, "bold")).grid(
            row=row, column=0, columnspan=3, pady=10, sticky=tk.W
        )
        row += 1

        for joint_name in sorted(self.joint_data["targets"].keys()):
            jmin, jmax = self.joint_limits.get(joint_name, (-3.14, 3.14))
            self._add_slider(main_frame, row, joint_name, joint_name, jmin, jmax)
            row += 1

        reset_button = ttk.Button(main_frame, text="Reset All Joints", command=self._reset_all)
        reset_button.grid(row=row, column=0, columnspan=3, pady=20)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def _add_slider(self, parent, row, joint_name, display_name, jmin, jmax):
        ttk.Label(parent, text=display_name, width=28, font=("Arial", 11, "bold")).grid(
            row=row, column=0, padx=10, pady=4, sticky=tk.W
        )

        slider = ttk.Scale(
            parent,
            from_=jmin,
            to=jmax,
            orient=tk.HORIZONTAL,
            length=500,
            command=lambda val, n=joint_name: self._on_slider_change(n, val),
        )
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=10, pady=4)

        value_label = ttk.Label(parent, text="0.000", width=10, font=("Arial", 11))
        value_label.grid(row=row, column=2, padx=10, pady=4)

        self.sliders[joint_name] = slider
        self.value_labels[joint_name] = value_label

    def _on_slider_change(self, joint_name, value):
        val = float(value)
        if joint_name in self.joint_limits:
            jmin, jmax = self.joint_limits[joint_name]
            val = max(jmin, min(jmax, val))
        with self.joint_data["lock"]:
            self.joint_data["targets"][joint_name] = val
            self.joint_data["dirty"].add(joint_name)
        if joint_name in self.value_labels:
            self.value_labels[joint_name].config(text=f"{val:.3f}")

    def _reset_all(self):
        with self.joint_data["lock"]:
            for joint_name, default_val in self.joint_data["defaults"].items():
                self.joint_data["targets"][joint_name] = default_val
                self.joint_data["dirty"].add(joint_name)
                if joint_name in self.sliders:
                    self.sliders[joint_name].set(default_val)
                if joint_name in self.value_labels:
                    self.value_labels[joint_name].config(text=f"{default_val:.3f}")

    def set_initial_values(self):
        for joint_name, val in self.joint_data["targets"].items():
            if joint_name in self.sliders:
                self.sliders[joint_name].set(val)
                if joint_name in self.value_labels:
                    self.value_labels[joint_name].config(text=f"{val:.3f}")


def main():
    try:
        import omni.kit.viewport.utility as viewport_utils
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            viewport.disable_interactions()
    except Exception:
        pass

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0)
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

    all_joint_names = robot.joint_names

    def get_joint_idx(joint_name):
        idx, _ = robot.find_joints(joint_name)
        if idx is None or len(idx) == 0:
            return None
        val = idx[0]
        return val.item() if hasattr(val, "item") else int(val)

    joint_indices = {name: get_joint_idx(name) for name in all_joint_names}

    default_joint_targets = robot.data.default_joint_pos.clone()

    root_state = robot.data.root_state_w.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)
    default_root_orientation = root_state[:, 3:7].clone()
    robot.write_root_state_to_sim(root_state, torch.arange(args_cli.num_envs, device=sim.device))

    for _ in range(20):
        robot.set_joint_position_target(default_joint_targets)
        root_state = robot.data.root_state_w.clone()
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    keys_pressed = {
        "i": False, "k": False, "j": False, "l": False,
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
    toggle_lock = False

    def tk_key_to_control(keysym: str):
        k = keysym.lower()
        if k in ("minus",):
            return "-"
        if k in ("equal",):
            return "="
        if k in ("tab",):
            return "tab"
        if k in ("return", "enter"):
            return None
        if len(k) == 1:
            return k
        return None

    def on_tk_key_press(event):
        nonlocal active_arm_left, toggle_lock
        k = tk_key_to_control(event.keysym)
        if k is None:
            return
        if k == "tab":
            if not toggle_lock:
                active_arm_left = not active_arm_left
                toggle_lock = True
            keys_pressed["tab"] = True
            return
        if k in keys_pressed:
            keys_pressed[k] = True

    def on_tk_key_release(event):
        nonlocal toggle_lock
        k = tk_key_to_control(event.keysym)
        if k is None:
            return
        if k == "tab":
            keys_pressed["tab"] = False
            toggle_lock = False
            return
        if k in keys_pressed:
            keys_pressed[k] = False

    input_interface = carb.input.acquire_input_interface()
    keyboard_sub = None

    def keyboard_event_handler(event, *args, **kwargs):
        nonlocal active_arm_left, toggle_lock
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
            elif input_str.endswith(".U"):
                keys_pressed["u"] = True
            elif input_str.endswith(".O"):
                keys_pressed["o"] = True
            elif input_str.endswith(".1") or input_str.endswith(".ONE"):
                keys_pressed["1"] = True
            elif input_str.endswith(".2") or input_str.endswith(".TWO"):
                keys_pressed["2"] = True
            elif input_str.endswith(".3") or input_str.endswith(".THREE"):
                keys_pressed["3"] = True
            elif input_str.endswith(".4") or input_str.endswith(".FOUR"):
                keys_pressed["4"] = True
            elif input_str.endswith(".5") or input_str.endswith(".FIVE"):
                keys_pressed["5"] = True
            elif input_str.endswith(".6") or input_str.endswith(".SIX"):
                keys_pressed["6"] = True
            elif input_str.endswith(".7") or input_str.endswith(".SEVEN"):
                keys_pressed["7"] = True
            elif input_str.endswith(".8") or input_str.endswith(".EIGHT"):
                keys_pressed["8"] = True
            elif input_str.endswith(".9") or input_str.endswith(".NINE"):
                keys_pressed["9"] = True
            elif input_str.endswith(".0") or input_str.endswith(".ZERO"):
                keys_pressed["0"] = True
            elif input_str.endswith(".MINUS") or input_str.endswith(".-"):
                keys_pressed["-"] = True
            elif input_str.endswith(".EQUALS") or input_str.endswith(".="):
                keys_pressed["="] = True
            elif input_str.endswith(".TAB"):
                if not toggle_lock:
                    active_arm_left = not active_arm_left
                    toggle_lock = True
                keys_pressed["tab"] = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if input_str.endswith(".I"):
                keys_pressed["i"] = False
            elif input_str.endswith(".K"):
                keys_pressed["k"] = False
            elif input_str.endswith(".J"):
                keys_pressed["j"] = False
            elif input_str.endswith(".L"):
                keys_pressed["l"] = False
            elif input_str.endswith(".W"):
                keys_pressed["w"] = False
            elif input_str.endswith(".S"):
                keys_pressed["s"] = False
            elif input_str.endswith(".Q"):
                keys_pressed["q"] = False
            elif input_str.endswith(".E"):
                keys_pressed["e"] = False
            elif input_str.endswith(".Z"):
                keys_pressed["z"] = False
            elif input_str.endswith(".C"):
                keys_pressed["c"] = False
            elif input_str.endswith(".R"):
                keys_pressed["r"] = False
            elif input_str.endswith(".F"):
                keys_pressed["f"] = False
            elif input_str.endswith(".T"):
                keys_pressed["t"] = False
            elif input_str.endswith(".G"):
                keys_pressed["g"] = False
            elif input_str.endswith(".Y"):
                keys_pressed["y"] = False
            elif input_str.endswith(".H"):
                keys_pressed["h"] = False
            elif input_str.endswith(".U"):
                keys_pressed["u"] = False
            elif input_str.endswith(".O"):
                keys_pressed["o"] = False
            elif input_str.endswith(".1") or input_str.endswith(".ONE"):
                keys_pressed["1"] = False
            elif input_str.endswith(".2") or input_str.endswith(".TWO"):
                keys_pressed["2"] = False
            elif input_str.endswith(".3") or input_str.endswith(".THREE"):
                keys_pressed["3"] = False
            elif input_str.endswith(".4") or input_str.endswith(".FOUR"):
                keys_pressed["4"] = False
            elif input_str.endswith(".5") or input_str.endswith(".FIVE"):
                keys_pressed["5"] = False
            elif input_str.endswith(".6") or input_str.endswith(".SIX"):
                keys_pressed["6"] = False
            elif input_str.endswith(".7") or input_str.endswith(".SEVEN"):
                keys_pressed["7"] = False
            elif input_str.endswith(".8") or input_str.endswith(".EIGHT"):
                keys_pressed["8"] = False
            elif input_str.endswith(".9") or input_str.endswith(".NINE"):
                keys_pressed["9"] = False
            elif input_str.endswith(".0") or input_str.endswith(".ZERO"):
                keys_pressed["0"] = False
            elif input_str.endswith(".MINUS") or input_str.endswith(".-"):
                keys_pressed["-"] = False
            elif input_str.endswith(".EQUALS") or input_str.endswith(".="):
                keys_pressed["="] = False
            elif input_str.endswith(".TAB"):
                keys_pressed["tab"] = False
                toggle_lock = False

    try:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    except Exception:
        pass

    sim_dt = sim.get_physics_dt()
    base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    base_speed = 1.0
    step_size = 0.0175

    joint_data = {
        "targets": {},
        "defaults": {},
        "dirty": set(),
        "lock": threading.Lock(),
    }

    for name in all_joint_names:
        idx = joint_indices[name]
        if idx is not None:
            val = default_joint_targets[0, idx].item()
            joint_data["targets"][name] = val
            joint_data["defaults"][name] = val

    joint_limits = {name: (-3.14, 3.14) for name in all_joint_names}

    ui = JointControlUI(joint_data, joint_limits, keys_pressed, on_key_press=on_tk_key_press, on_key_release=on_tk_key_release)
    ui.set_initial_values()

    def apply_and_mark_dirty(joint_name: str, new_val: float):
        with joint_data["lock"]:
            joint_data["targets"][joint_name] = new_val
            joint_data["dirty"].add(joint_name)

    while simulation_app.is_running():
        base_velocity.zero_()

        if keys_pressed["i"]:
            base_velocity[:, 0] = base_speed
        if keys_pressed["k"]:
            base_velocity[:, 0] = -base_speed
        if keys_pressed["j"]:
            base_velocity[:, 1] = base_speed
        if keys_pressed["l"]:
            base_velocity[:, 1] = -base_speed

        if keys_pressed.get("w", False):
            name = "bow_pitch_joint_01"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("s", False):
            name = "bow_pitch_joint_01"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        if keys_pressed.get("q", False):
            name = "bow_pitch_joint_02"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("e", False):
            name = "bow_pitch_joint_02"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        if keys_pressed.get("z", False):
            name = "bow_pitch_joint_03"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("c", False):
            name = "bow_pitch_joint_03"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        if keys_pressed.get("r", False):
            name = "bow_yaw_joint"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("f", False):
            name = "bow_yaw_joint"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        if keys_pressed.get("t", False):
            name = "head_yaw_joint"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("g", False):
            name = "head_yaw_joint"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        if keys_pressed.get("y", False):
            name = "head_pitch_joint"
            v = joint_data["targets"].get(name, 0.0) + step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)
        if keys_pressed.get("h", False):
            name = "head_pitch_joint"
            v = joint_data["targets"].get(name, 0.0) - step_size
            apply_and_mark_dirty(name, v)
            if name in ui.sliders:
                ui.sliders[name].set(v)

        arm_prefix = "left_" if active_arm_left else "right_"

        if keys_pressed.get("u", False):
            name = f"{arm_prefix}shoulder_pitch_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("o", False):
            name = f"{arm_prefix}shoulder_pitch_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("1", False):
            name = f"{arm_prefix}shoulder_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("2", False):
            name = f"{arm_prefix}shoulder_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("3", False):
            name = f"{arm_prefix}shoulder_yaw_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("4", False):
            name = f"{arm_prefix}shoulder_yaw_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("5", False):
            name = f"{arm_prefix}elbow_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("6", False):
            name = f"{arm_prefix}elbow_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("7", False):
            name = f"{arm_prefix}elbow_yaw_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("8", False):
            name = f"{arm_prefix}elbow_yaw_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("9", False):
            name = f"{arm_prefix}wrist_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("0", False):
            name = f"{arm_prefix}wrist_roll_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        if keys_pressed.get("-", False):
            name = f"{arm_prefix}wrist_pitch_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) + step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)
        if keys_pressed.get("=", False):
            name = f"{arm_prefix}wrist_pitch_joint"
            if name in joint_indices and joint_indices[name] is not None:
                v = joint_data["targets"].get(name, 0.0) - step_size
                apply_and_mark_dirty(name, v)
                if name in ui.sliders:
                    ui.sliders[name].set(v)

        joint_targets = default_joint_targets.clone()
        with joint_data["lock"]:
            for name, val in joint_data["targets"].items():
                idx = joint_indices.get(name)
                if idx is not None:
                    joint_targets[:, idx] = val

        robot.set_joint_position_target(joint_targets)

        root_state = robot.data.root_state_w.clone()
        if torch.any(base_velocity != 0):
            root_state[:, 0:3] += base_velocity * sim_dt
        root_state[:, 3:7] = default_root_orientation
        robot.write_root_pose_to_sim(root_state[:, :7], torch.arange(args_cli.num_envs, device=sim.device))

        try:
            ui.root.update_idletasks()
            ui.root.update()
        except tk.TclError:
            break

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
