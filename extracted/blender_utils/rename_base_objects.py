import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def rename_base_objects(mesh_obj: bpy.types.Object, armature_obj: bpy.types.Object) -> tuple:
    """Rename base mesh and armature to specific names."""
    # Store original names for reference
    original_mesh_name = mesh_obj.name
    original_armature_name = armature_obj.name
    
    # Rename mesh to Body.BaseAvatar
    mesh_obj.name = "Body.BaseAvatar"
    mesh_obj.data.name = "Body.BaseAvatar_Mesh"
    
    # Rename armature to Armature.BaseAvatar
    armature_obj.name = "Armature.BaseAvatar"
    armature_obj.data.name = "Armature.BaseAvatar_Data"
    
    print(f"Renamed base objects: {original_mesh_name} -> {mesh_obj.name}, {original_armature_name} -> {armature_obj.name}")
    return mesh_obj, armature_obj
