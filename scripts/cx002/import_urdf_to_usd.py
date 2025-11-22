# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to import URDF file and export as USD with proper articulation setup.
Run this once to create a proper USD file from the URDF.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Import URDF file and export as USD with articulation."
)
parser.add_argument(
    "--urdf_path",
    type=str,
    default="assets/cx002_description/urdf/cx002_new.urdf",
    help="Path to the URDF file to import.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="assets/cx002_description/urdf/cx002_imported.usd",
    help="Path where to save the exported USD file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from omni.isaac.urdf import _urdf
from pxr import Usd, UsdGeom, PhysxSchema

# Get the stage
stage = omni.usd.get_context().get_stage()

# Import URDF
urdf_interface = _urdf.acquire_urdf_interface()
status, import_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=args_cli.urdf_path,
    import_inertia_tensor=False,
    fix_base=True,
    make_default_prim=True,
    create_physics_scene=True,
    self_collision=False,
)

if status:
    print(f"[INFO]: Successfully imported URDF to: {import_path}")
    
    # Get the root prim (should be the robot root)
    root_prim = stage.GetPrimAtPath(import_path)
    
    if root_prim.IsValid():
        # Apply ArticulationRootAPI
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim, "ArticulationRoot")
        if articulation_api:
            print(f"[INFO]: Applied ArticulationRootAPI to {import_path}")
        else:
            print(f"[WARNING]: Failed to apply ArticulationRootAPI to {import_path}")
        
        # Set as default prim if not already set
        if not stage.GetDefaultPrim():
            stage.SetDefaultPrim(root_prim)
            print(f"[INFO]: Set {import_path} as default prim")
        
        # Save the USD file
        output_path = Path(args_cli.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stage.Export(str(output_path))
        print(f"[INFO]: Exported USD file to: {output_path}")
        print(f"[INFO]: You can now use this USD file in your environment config:")
        print(f"       usd_path=\"{args_cli.output_path}\"")
    else:
        print(f"[ERROR]: Could not find root prim at {import_path}")
else:
    print(f"[ERROR]: Failed to import URDF file: {args_cli.urdf_path}")

simulation_app.close()

