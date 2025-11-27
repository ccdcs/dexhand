# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard teleoperation script for cx002 humanoid robot.
Controls robot joints using keyboard input.

Control Scheme:

Base/Torso (WASD)
W/S: Torso pitch forward/back (bow_pitch_joint_01)
A/D: Torso yaw left/right (bow_yaw_joint)
Q/E: Torso pitch 02
Z/C: Torso pitch 03

Head:
T/G: Head yaw left/right
R/F: Head pitch up/down

Left Arm:
1/2: Shoulder pitch
3/4: Shoulder roll
5/6: Shoulder yaw
7/8: Elbow roll
9/0: Elbow yaw
-/=: Wrist roll
[/]: Wrist pitch

Right Arm:
I/O: Shoulder pitch
K/L: Shoulder roll
J/H: Shoulder yaw
M/,: Elbow roll
.//: Elbow yaw
;/': Wrist roll
P/\\: Wrist pitch

Space: Reset all joints to zero
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for cx002 humanoid robot."
)
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
    keyboard = carb.input.acquire_input_interface()
    
    joint_targets = robot.data.joint_pos.clone()
    joint_step = 0.1
    
    def get_joint_idx(joint_name):
        idx, _ = robot.find_joints(joint_name)
        if idx is not None and len(idx) > 0:
            val = idx[0]
            return val.item() if hasattr(val, 'item') else val
        return None
    
    bow_pitch_01_idx = get_joint_idx("bow_pitch_joint_01")
    bow_pitch_02_idx = get_joint_idx("bow_pitch_joint_02")
    bow_pitch_03_idx = get_joint_idx("bow_pitch_joint_03")
    bow_yaw_idx = get_joint_idx("bow_yaw_joint")
    head_yaw_idx = get_joint_idx("head_yaw_joint")
    head_pitch_idx = get_joint_idx("head_pitch_joint")
    
    left_shoulder_pitch_idx = get_joint_idx("left_shoulder_pitch_joint")
    left_shoulder_roll_idx = get_joint_idx("left_shoulder_roll_joint")
    left_shoulder_yaw_idx = get_joint_idx("left_shoulder_yaw_joint")
    left_elbow_roll_idx = get_joint_idx("left_elbow_roll_joint")
    left_elbow_yaw_idx = get_joint_idx("left_elbow_yaw_joint")
    left_wrist_roll_idx = get_joint_idx("left_wrist_roll_joint")
    left_wrist_pitch_idx = get_joint_idx("left_wrist_pitch_joint")
    
    right_shoulder_pitch_idx = get_joint_idx("right_shoulder_pitch_joint")
    right_shoulder_roll_idx = get_joint_idx("right_shoulder_roll_joint")
    right_shoulder_yaw_idx = get_joint_idx("right_shoulder_yaw_joint")
    right_elbow_roll_idx = get_joint_idx("right_elbow_roll_joint")
    right_elbow_yaw_idx = get_joint_idx("right_elbow_yaw_joint")
    right_wrist_roll_idx = get_joint_idx("right_wrist_roll_joint")
    right_wrist_pitch_idx = get_joint_idx("right_wrist_pitch_joint")
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Keyboard controls:")
    print("        Base/Torso:")
    print("          W/S: Torso pitch forward/back (bow_pitch_joint_01)")
    print("          A/D: Torso yaw left/right (bow_yaw_joint)")
    print("          Q/E: Torso pitch 02 up/down")
    print("          Z/C: Torso pitch 03 up/down")
    print("        Head:")
    print("          T/G: Head yaw left/right")
    print("          R/F: Head pitch up/down")
    print("        Left Arm:")
    print("          1/2: Shoulder pitch")
    print("          3/4: Shoulder roll")
    print("          5/6: Shoulder yaw")
    print("          7/8: Elbow roll")
    print("          9/0: Elbow yaw")
    print("          -/=: Wrist roll")
    print("          [/]: Wrist pitch")
    print("        Right Arm:")
    print("          I/O: Shoulder pitch")
    print("          K/L: Shoulder roll")
    print("          J/H: Shoulder yaw")
    print("          M/,: Elbow roll")
    print("          .//: Elbow yaw")
    print("          ;/': Wrist roll")
    print("          P/\\: Wrist pitch")
    print("        Space: Reset all joints to zero")
    print("        Ctrl+C: Exit")

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if bow_pitch_01_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_W):
                joint_targets[0, bow_pitch_01_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_S):
                joint_targets[0, bow_pitch_01_idx] -= joint_step
        
        if bow_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_A):
                joint_targets[0, bow_yaw_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_D):
                joint_targets[0, bow_yaw_idx] -= joint_step
        
        if bow_pitch_02_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_Q):
                joint_targets[0, bow_pitch_02_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_E):
                joint_targets[0, bow_pitch_02_idx] -= joint_step
        
        if bow_pitch_03_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_Z):
                joint_targets[0, bow_pitch_03_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_C):
                joint_targets[0, bow_pitch_03_idx] -= joint_step
        
        if head_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_T):
                joint_targets[0, head_yaw_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_G):
                joint_targets[0, head_yaw_idx] -= joint_step
        
        if head_pitch_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_R):
                joint_targets[0, head_pitch_idx] += joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_F):
                joint_targets[0, head_pitch_idx] -= joint_step
        
        if left_shoulder_pitch_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_1):
                joint_targets[0, left_shoulder_pitch_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_2):
                joint_targets[0, left_shoulder_pitch_idx] += joint_step
        
        if left_shoulder_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_3):
                joint_targets[0, left_shoulder_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_4):
                joint_targets[0, left_shoulder_roll_idx] += joint_step
        
        if left_shoulder_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_5):
                joint_targets[0, left_shoulder_yaw_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_6):
                joint_targets[0, left_shoulder_yaw_idx] += joint_step
        
        if left_elbow_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_7):
                joint_targets[0, left_elbow_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_8):
                joint_targets[0, left_elbow_roll_idx] += joint_step
        
        if left_elbow_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_9):
                joint_targets[0, left_elbow_yaw_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_0):
                joint_targets[0, left_elbow_yaw_idx] += joint_step
        
        if left_wrist_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_MINUS):
                joint_targets[0, left_wrist_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_EQUALS):
                joint_targets[0, left_wrist_roll_idx] += joint_step
        
        if left_wrist_pitch_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_LEFTBRACKET):
                joint_targets[0, left_wrist_pitch_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_RIGHTBRACKET):
                joint_targets[0, left_wrist_pitch_idx] += joint_step
        
        if right_shoulder_pitch_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_I):
                joint_targets[0, right_shoulder_pitch_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_O):
                joint_targets[0, right_shoulder_pitch_idx] += joint_step
        
        if right_shoulder_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_K):
                joint_targets[0, right_shoulder_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_L):
                joint_targets[0, right_shoulder_roll_idx] += joint_step
        
        if right_shoulder_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_J):
                joint_targets[0, right_shoulder_yaw_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_H):
                joint_targets[0, right_shoulder_yaw_idx] += joint_step
        
        if right_elbow_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_M):
                joint_targets[0, right_elbow_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_COMMA):
                joint_targets[0, right_elbow_roll_idx] += joint_step
        
        if right_elbow_yaw_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_PERIOD):
                joint_targets[0, right_elbow_yaw_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_SLASH):
                joint_targets[0, right_elbow_yaw_idx] += joint_step
        
        if right_wrist_roll_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_SEMICOLON):
                joint_targets[0, right_wrist_roll_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_APOSTROPHE):
                joint_targets[0, right_wrist_roll_idx] += joint_step
        
        if right_wrist_pitch_idx is not None:
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_P):
                joint_targets[0, right_wrist_pitch_idx] -= joint_step
            if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_BACKSLASH):
                joint_targets[0, right_wrist_pitch_idx] += joint_step
        
        if keyboard.get_keyboard_value(carb.input.KeyboardInput.KEY_SPACE):
            joint_targets.zero_()

        robot.set_joint_position_target(joint_targets)

        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()

