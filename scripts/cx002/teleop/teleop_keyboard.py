import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom
import numpy as np

class KeyboardTeleop:
    def __init__(self, robot_prim):
        self.robot = robot_prim
        self.joint_names = self.robot.get_joint_names()
        self.step = 0.1  # rad per keypress

    def update(self, keyboard):
        for joint in self.joint_names:
            if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_1):
                self.robot.set_joint_target_position(joint, 0.2)

            if keyboard.WAS_PRESSED(carb.input.KeyboardInput.KEY_2):
                self.robot.set_joint_target_position(joint, -0.2)

def main():
    world = World()
    
    # Load your CX002 robot
    add_reference_to_stage(
        "/home/msclab/Github/dexhand/assets/cx002_description/urdf/cx002/cx002.usd",
        "/World/CX002"
    )

    world.reset()

    robot = world.scene.get_object("CX002")

    teleop = KeyboardTeleop(robot)

    while simulation_app.is_running():
        world.step(render=True)
        teleop.update(world.input_interface.keyboard)

if __name__ == "__main__":
    main()
