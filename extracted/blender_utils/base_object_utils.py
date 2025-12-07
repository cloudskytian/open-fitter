import os
import sys

import bpy
import os
import sys


# Merged from rename_base_objects.py

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

# Merged from cleanup_base_objects.py

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