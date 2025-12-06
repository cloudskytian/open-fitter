import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from blender_utils.rename_base_objects import rename_base_objects


def cleanup_base_objects(mesh_name: str) -> tuple:
    """Delete all objects except the specified mesh and its armature."""
    
    original_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Find the mesh and its armature
    target_mesh = None
    target_armature = None
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name == mesh_name:
            target_mesh = obj
            # Find associated armature through modifiers
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE':
                    target_armature = modifier.object
                    break
    
    if not target_mesh:
        raise Exception(f"Mesh '{mesh_name}' not found")
    
    if target_armature and target_armature.parent:
        original_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = target_armature
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.context.view_layer.objects.active = original_active
    
    # Delete all other objects
    for obj in bpy.data.objects[:]:  # Create a copy of the list to avoid modification during iteration
        if obj != target_mesh and obj != target_armature:
            bpy.data.objects.remove(obj, do_unlink=True)
            
    #bpy.ops.object.mode_set(mode=original_mode)
    
    # Rename objects to specified names
    return rename_base_objects(target_mesh, target_armature)
