import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def get_humanoid_and_auxiliary_bone_groups_with_intermediate(base_armature: bpy.types.Object, base_avatar_data: dict) -> set:
    bone_groups = set()
    
    # まず基本のHumanoidボーンとAuxiliaryボーンを追加
    humanoid_bones = set()
    humanoid_name_to_bone = {}  # humanoidBoneName -> boneName のマッピング
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            bone_name = bone_map["boneName"]
            bone_groups.add(bone_name)
            humanoid_bones.add(bone_name)
            if "humanoidBoneName" in bone_map:
                humanoid_name_to_bone[bone_map["humanoidBoneName"]] = bone_name
    
    # Hipsボーンの実際のボーン名を取得
    hips_bone_name = humanoid_name_to_bone.get("Hips")
    
    # Auxiliaryボーンとその所属Humanoidボーンのマッピングを作成
    auxiliary_to_humanoid = {}
    humanoid_to_auxiliaries = {}
    
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_bone_name = aux_set.get("humanoidBoneName")
        auxiliaries = aux_set.get("auxiliaryBones", [])
        
        # Humanoidボーン名から実際のボーン名を取得
        actual_humanoid_bone = None
        for bone_map in base_avatar_data.get("humanoidBones", []):
            if bone_map.get("humanoidBoneName") == humanoid_bone_name:
                actual_humanoid_bone = bone_map.get("boneName")
                break
        
        if actual_humanoid_bone:
            humanoid_to_auxiliaries[actual_humanoid_bone] = set(auxiliaries)
            for aux_bone in auxiliaries:
                bone_groups.add(aux_bone)
                auxiliary_to_humanoid[aux_bone] = actual_humanoid_bone
    
    # 中間ボーンを検出・追加
    if base_armature and base_armature.pose:
        # Humanoidボーンの親辿り処理
        for bone in base_armature.pose.bones:
            if bone.name in humanoid_bones:
                # Hipsボーンの場合は特別処理：ルートまでのすべての親ボーンを追加
                if bone.name == hips_bone_name:
                    current_parent = bone.parent
                    while current_parent:
                        bone_groups.add(current_parent.name)
                        current_parent = current_parent.parent
                else:
                    # 通常のHumanoidボーンの処理
                    # このHumanoidボーンの親を辿る
                    current_parent = bone.parent
                    intermediate_bones = []
                    
                    while current_parent:
                        if current_parent.name in humanoid_bones:
                            # 親のHumanoidボーンに到達したら、中間ボーンをすべて追加
                            bone_groups.update(intermediate_bones)
                            break
                        else:
                            # 中間ボーンとして記録
                            intermediate_bones.append(current_parent.name)
                            current_parent = current_parent.parent
        
        # Auxiliaryボーンの親辿り処理
        for aux_bone_name in auxiliary_to_humanoid.keys():
            if aux_bone_name in base_armature.pose.bones:
                bone = base_armature.pose.bones[aux_bone_name]
                parent_humanoid_bone = auxiliary_to_humanoid[aux_bone_name]
                same_group_bones = {parent_humanoid_bone} | humanoid_to_auxiliaries.get(parent_humanoid_bone, set())
                
                # このAuxiliaryボーンの親を辿る
                current_parent = bone.parent
                intermediate_bones = []
                
                while current_parent:
                    if current_parent.name in same_group_bones:
                        # 同じグループのボーンに到達したら、中間ボーンをすべて追加
                        bone_groups.update(intermediate_bones)
                        break
                    else:
                        # 中間ボーンとして記録
                        intermediate_bones.append(current_parent.name)
                        current_parent = current_parent.parent
    
    return bone_groups
