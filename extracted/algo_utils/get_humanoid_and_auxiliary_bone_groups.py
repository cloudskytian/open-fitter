import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_humanoid_and_auxiliary_bone_groups(base_avatar_data):
    """HumanoidボーンとAuxiliaryボーンの頂点グループを取得"""
    bone_groups = set()
    
    # Humanoidボーンを追加
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            bone_groups.add(bone_map["boneName"])
    
    # Auxiliaryボーンを追加
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set.get("auxiliaryBones", []):
            bone_groups.add(aux_bone)
            
    return bone_groups
