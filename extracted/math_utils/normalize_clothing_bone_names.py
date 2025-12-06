import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def normalize_clothing_bone_names(clothing_armature: bpy.types.Object, clothing_avatar_data: dict, 
                                clothing_meshes: list) -> None:
    """
    Normalize bone names in clothing_avatar_data to match existing bones in clothing_armature.
    
    For each humanoidBone in clothing_avatar_data:
    1. Check if boneName exists in clothing_armature
    2. If not, convert boneName to lowercase alphabetic characters and find matching bone
    3. Update boneName in clothing_avatar_data if match found
    4. Update corresponding vertex group names in all clothing_meshes
    """
    import re
    
    # Get all bone names from clothing armature
    armature_bone_names = {bone.name for bone in clothing_armature.data.bones}
    print(f"Available bones in clothing armature: {sorted(armature_bone_names)}")
    
    # Store name changes for vertex group updates
    bone_name_changes = {}
    
    # Process each humanoid bone mapping
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if "boneName" not in bone_map:
            continue
            
        original_bone_name = bone_map["boneName"]
        
        # Check if bone exists in armature
        if original_bone_name in armature_bone_names:
            print(f"Bone '{original_bone_name}' found in armature")
            continue
            
        # Extract alphabetic characters and convert to lowercase
        normalized_pattern = re.sub(r'[^a-zA-Z]', '', original_bone_name).lower()
        if not normalized_pattern:
            print(f"Warning: No alphabetic characters found in bone name '{original_bone_name}'")
            continue
            
        print(f"Looking for bone matching pattern '{normalized_pattern}' (from '{original_bone_name}')")
        
        # Find matching bone in armature
        matching_bone = None
        for armature_bone_name in armature_bone_names:
            armature_normalized = re.sub(r'[^a-zA-Z]', '', armature_bone_name).lower()
            if armature_normalized == normalized_pattern:
                matching_bone = armature_bone_name
                break
                
        if matching_bone:
            print(f"Found matching bone: '{original_bone_name}' -> '{matching_bone}'")
            bone_name_changes[matching_bone] = original_bone_name
        else:
            print(f"Warning: No matching bone found for '{original_bone_name}' (pattern: '{normalized_pattern}')")
    
    # Update vertex group names in all clothing meshes
    if bone_name_changes:
        print(f"Updating vertex groups with bone name changes: {bone_name_changes}")
        
        for mesh_obj in clothing_meshes:
            if not mesh_obj or mesh_obj.type != 'MESH':
                continue
                
            for old_name, new_name in bone_name_changes.items():
                if old_name in mesh_obj.vertex_groups:
                    vertex_group = mesh_obj.vertex_groups[old_name]
                    vertex_group.name = new_name
                    print(f"Updated vertex group '{old_name}' -> '{new_name}' in mesh '{mesh_obj.name}'")
        
        # Update bone names in clothing armature
        print(f"Updating bone names in clothing armature: {bone_name_changes}")
        for old_name, new_name in bone_name_changes.items():
            if old_name in clothing_armature.data.bones:
                bone = clothing_armature.data.bones[old_name]
                bone.name = new_name
                print(f"Updated armature bone '{old_name}' -> '{new_name}'")
    
    print("Bone name normalization completed")
