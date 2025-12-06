import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_deformation_bone_groups(avatar_data: dict) -> list:
    """
    Get list of bone groups for deformation mask from avatar data,
    excluding Head and its auxiliary bones.
    
    Parameters:
        avatar_data: Avatar data containing bone information
        
    Returns:
        List of bone names for deformation mask
    """
    bone_groups = set()
    
    # Get mapping of humanoid bones
    for bone_map in avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            # Skip Head bone
            if bone_map["humanoidBoneName"] != "Head":
                bone_groups.add(bone_map["boneName"])
    
    # Get auxiliary bones mapping
    for aux_set in avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        # Skip Head's auxiliary bones
        if humanoid_bone != "Head":
            aux_bones = aux_set["auxiliaryBones"]
            bone_groups.update(aux_bones)
    
    return sorted(list(bone_groups))
