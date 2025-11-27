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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import torch
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


def main():
    world = World()
    
    add_reference_to_stage(
        usd_path="assets/cx002_description_new/cx002_robot/cx002_robot.usd",
        prim_path="/World/CX002"
    )
    
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    
    world.reset()
    
    robot = world.scene.get_object("CX002")
    
    joint_names = robot.get_joint_names()
    bow_joints = ["bow_pitch_joint_01", "bow_pitch_joint_02", "bow_pitch_joint_03", "bow_yaw_joint"]
    
    for i in range(100):
        for joint_name in bow_joints:
            if joint_name in joint_names:
                robot.set_joint_target_position(joint_name, 0.0)
        world.step(render=False)
    
    print("[INFO]: Robot stabilized to upright pose.")
    
    base_velocity = [0.0, 0.0, 0.0]
    base_speed = 0.5
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Use I/J/K/L keys to move the base:")
    print("        I: Move forward")
    print("        K: Move backward")
    print("        J: Move left")
    print("        L: Move right")
    print("        Ctrl+C: Exit")
    
    while simulation_app.is_running():
        base_velocity = [0.0, 0.0, 0.0]
        
        keyboard = world.input_interface.keyboard
        
        if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_I):
            base_velocity[0] = base_speed
            print("[DEBUG]: I key pressed - moving forward")
        if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_K):
            base_velocity[0] = -base_speed
            print("[DEBUG]: K key pressed - moving backward")
        if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_J):
            base_velocity[1] = base_speed
            print("[DEBUG]: J key pressed - moving left")
        if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_L):
            base_velocity[1] = -base_speed
            print("[DEBUG]: L key pressed - moving right")
        
        for joint_name in bow_joints:
            if joint_name in joint_names:
                robot.set_joint_target_position(joint_name, 0.0)
        
        if any(v != 0.0 for v in base_velocity):
            current_pos = robot.get_world_pose()
            new_pos = [
                current_pos[0][0] + base_velocity[0] * 0.01,
                current_pos[0][1] + base_velocity[1] * 0.01,
                current_pos[0][2]
            ]
            robot.set_world_pose(position=new_pos, orientation=current_pos[1])
        
        world.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
