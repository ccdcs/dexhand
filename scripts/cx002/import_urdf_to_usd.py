# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from pxr import Usd, UsdGeom, PhysxSchema
import omni.kit.commands

stage = omni.usd.get_context().get_stage()

output_path = Path(args_cli.output_path)
if output_path.exists():
    print(f"[INFO]: Found existing USD file: {output_path}")
    print("[INFO]: Opening it to verify ArticulationRootAPI...")
    
    omni.usd.get_context().open_stage(str(output_path))
    stage = omni.usd.get_context().get_stage()
    
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                default_prim = prim
                break
    
    if default_prim and default_prim.IsValid():
        import_path = str(default_prim.GetPath())
        print(f"[INFO]: Found root prim: {import_path}")
        status = True
    else:
        print("[ERROR]: Could not find root prim in USD file")
        status = False
        import_path = None
else:
    print(f"[ERROR]: USD file does not exist: {output_path}")
    print(f"[INFO]: Please import the URDF manually in Isaac Sim:")
    print(f"       1. File > Import > URDF")
    print(f"       2. Select: {args_cli.urdf_path}")
    print(f"       3. Check 'Create Articulation'")
    print(f"       4. Export to: {args_cli.output_path}")
    print(f"       5. Run this script again to apply ArticulationRootAPI")
    simulation_app.close()
    exit(1)

if status:
    print(f"[INFO]: Successfully imported URDF to: {import_path}")
    
    root_prim = stage.GetPrimAtPath(import_path)
    
    if root_prim.IsValid():
        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim, "ArticulationRoot")
        if articulation_api:
            print(f"[INFO]: Applied ArticulationRootAPI to {import_path}")
        else:
            print(f"[WARNING]: Failed to apply ArticulationRootAPI to {import_path}")
        
        if not stage.GetDefaultPrim():
            stage.SetDefaultPrim(root_prim)
            print(f"[INFO]: Set {import_path} as default prim")
        
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

