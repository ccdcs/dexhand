# Combined UI + keyboard teleoperation for cx002 robot.
# Keyboard control logic is the same as in teleop_keyboard.py.
# Tkinter UI adds sliders for all joints and a Reset All Joints button.

import argparse
import threading
import tkinter as tk
from tkinter import ttk

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Combined UI + keyboard teleoperation for cx002 robot.")
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
    def __init__(self, joint_data, joint_limits):
        self.joint_data = joint_data
        self.joint_limits = joint_limits

        self.root = tk.Tk()
        self.root.title("CX002 Teleoperation UI")
        self.root.geometry("1200x900")

        self.sliders = {}
        self.value_labels = {}

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0
        ttk.Label(main_frame, text="All Joints", font=("Arial", 14, "bold")).grid(
            row=row, column=0, columnspan=3, pady=10, sticky=tk.W
        )
        row += 1

        # Create sliders for all joints in deterministic order
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
        if joint_name in self.value_labels:
            self.value_labels[joint_name].config(text=f"{val:.3f}")

    def _reset_all(self):
        with self.joint_data["lock"]:
            for joint_name, default_val in self.joint_data["defaults"].items():
                self.joint_data["targets"][joint_name] = default_val
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
    print(f"[INFO]: Found {len(all_joint_names)} joints:")
    for name in all_joint_names:
        print(f"  - {name}")

    def get_joint_idx(joint_name):
        idx, _ = robot.find_joints(joint_name)
        if idx is None or len(idx) == 0:
            return None
        val = idx[0]
        return val.item() if hasattr(val, "item") else int(val)

    joint_indices = {}
    for joint_name in all_joint_names:
        joint_indices[joint_name] = get_joint_idx(joint_name)

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

    print("[INFO]: Keyboard controls (same as teleop_keyboard):")
    print("=" * 60)
    print("BASE MOVEMENT:")
    print("  I/K - Forward/Backward")
    print("  J/L - Left/Right")
    print()
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
    }

    active_arm_left = True
    keys_just_pressed = set()

    def keyboard_event_handler(event, *args, **kwargs):
        nonlocal keys_just_pressed, active_arm_left
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
            elif input_str.endswith(".U"):
                keys_pressed["u"] = True
                keys_just_pressed.add("u")
            elif input_str.endswith(".O"):
                keys_pressed["o"] = True
                keys_just_pressed.add("o")
            elif input_str.endswith(".1"):
                keys_pressed["1"] = True
                keys_just_pressed.add("1")
            elif input_str.endswith(".2"):
                keys_pressed["2"] = True
                keys_just_pressed.add("2")
            elif input_str.endswith(".3"):
                keys_pressed["3"] = True
                keys_just_pressed.add("3")
            elif input_str.endswith(".4"):
                keys_pressed["4"] = True
                keys_just_pressed.add("4")
            elif input_str.endswith(".5"):
                keys_pressed["5"] = True
                keys_just_pressed.add("5")
            elif input_str.endswith(".6"):
                keys_pressed["6"] = True
                keys_just_pressed.add("6")
            elif input_str.endswith(".7"):
                keys_pressed["7"] = True
                keys_just_pressed.add("7")
            elif input_str.endswith(".8"):
                keys_pressed["8"] = True
                keys_just_pressed.add("8")
            elif input_str.endswith(".9"):
                keys_pressed["9"] = True
                keys_just_pressed.add("9")
            elif input_str.endswith(".0"):
                keys_pressed["0"] = True
                keys_just_pressed.add("0")
            elif input_str.endswith(".MINUS"):
                keys_pressed["-"] = True
                keys_just_pressed.add("-")
            elif input_str.endswith(".EQUALS"):
                keys_pressed["="] = True
                keys_just_pressed.add("=")
            elif input_str.endswith(".TAB"):
                active_arm_left = not active_arm_left
                print(f"[INFO]: Switched to {'LEFT' if active_arm_left else 'RIGHT'} arm")

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
            elif input_str.endswith(".1"):
                keys_pressed["1"] = False
            elif input_str.endswith(".2"):
                keys_pressed["2"] = False
            elif input_str.endswith(".3"):
                keys_pressed["3"] = False
            elif input_str.endswith(".4"):
                keys_pressed["4"] = False
            elif input_str.endswith(".5"):
                keys_pressed["5"] = False
            elif input_str.endswith(".6"):
                keys_pressed["6"] = False
            elif input_str.endswith(".7"):
                keys_pressed["7"] = False
            elif input_str.endswith(".8"):
                keys_pressed["8"] = False
            elif input_str.endswith(".9"):
                keys_pressed["9"] = False
            elif input_str.endswith(".0"):
                keys_pressed["0"] = False
            elif input_str.endswith(".MINUS"):
                keys_pressed["-"] = False
            elif input_str.endswith(".EQUALS"):
                keys_pressed["="] = False

    try:
        appwindow = omni.appwindow.get_default_app_window()
        if appwindow:
            keyboard = appwindow.get_keyboard()
            if keyboard:
                input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    except Exception:
        pass

    sim_dt = sim.get_physics_dt()
    joint_targets = default_joint_targets.clone()
    joint_offsets = torch.zeros_like(default_joint_targets)
    base_velocity = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    base_speed = 10.0
    step_size = 0.1

    # Simple joint_limits for UI sliders (fallback)
    joint_limits = {}
    for name in all_joint_names:
        joint_limits[name] = (-3.14, 3.14)

    joint_data = {
        "targets": {},
        "defaults": {},
        "lock": threading.Lock(),
    }

    for name in all_joint_names:
        idx = joint_indices[name]
        if idx is not None:
            val = default_joint_targets[0, idx].item()
            joint_data["targets"][name] = val
            joint_data["defaults"][name] = val

    ui = JointControlUI(joint_data, joint_limits)
    ui.set_initial_values()

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

        if "bow_pitch_joint_01" in joint_indices and joint_indices["bow_pitch_joint_01"] is not None:
            idx = joint_indices["bow_pitch_joint_01"]
            if "w" in keys_just_pressed:
                joint_offsets[:, idx] += step_size
            if "s" in keys_just_pressed:
                joint_offsets[:, idx] -= step_size
            if "a" in keys_just_pressed:
                joint_offsets[:, idx] += step_size * 0.5
            if "d" in keys_just_pressed:
                joint_offsets[:, idx] -= step_size * 0.5

        if "bow_pitch_joint_02" in joint_indices and joint_indices["bow_pitch_joint_02"] is not None:
            idx = joint_indices["bow_pitch_joint_02"]
            if "q" in keys_just_pressed:
                joint_offsets[:, idx] += step_size
            if "e" in keys_just_pressed:
                joint_offsets[:, idx] -= step_size

        if "bow_pitch_joint_03" in joint_indices and joint_indices["bow_pitch_joint_03"] is not None:
            idx = joint_indices["bow_pitch_joint_03"]
            if "z" in keys_just_pressed:
                joint_offsets[:, idx] += step_size
            if "c" in keys_just_pressed:
                joint_offsets[:, idx] -= step_size

        if "bow_yaw_joint" in joint_indices and joint_indices["bow_yaw_joint"] is not None:
            idx = joint_indices["bow_yaw_joint"]
            if "r" in keys_just_pressed:
                joint_offsets[:, idx] += step_size
            if "f" in keys_just_pressed:
                joint_offsets[:, idx] -= step_size

        keys_just_pressed.clear()

        joint_targets = default_joint_targets + joint_offsets

        with joint_data["lock"]:
            for name, val in joint_data["targets"].items():
                idx = joint_indices.get(name)
                if idx is not None:
                    joint_targets[:, idx] = val

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

        try:
            ui.root.update_idletasks()
            ui.root.update()
        except tk.TclError:
            pass

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()