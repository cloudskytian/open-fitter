import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def process_humanoid_vertex_groups(mesh_obj: bpy.types.Object, clothing_armature: bpy.types.Object, base_avatar_data: dict, clothing_avatar_data: dict) -> None:
    """
    衣装メッシュのHumanoidボーン頂点グループを処理
    - Humanoidボーン名を素体アバターデータのものに変換
    - 補助ボーンの頂点グループを追加
    - 条件を満たす場合はOptional Humanoidボーンの頂点グループを追加
    """

    # Get bone names from clothing armature
    clothing_bone_names = set(bone.name for bone in clothing_armature.data.bones)
    
    # Humanoidボーン名のマッピングを作成
    base_humanoid_to_bone = {bone_map["humanoidBoneName"]: bone_map["boneName"] 
                        for bone_map in base_avatar_data["humanoidBones"]}
    clothing_humanoid_to_bone = {bone_map["humanoidBoneName"]: bone_map["boneName"] 
                           for bone_map in clothing_avatar_data["humanoidBones"]}
    clothing_bone_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                           for bone_map in clothing_avatar_data["humanoidBones"]}
    
    # 補助ボーンのマッピングを作成
    auxiliary_bones = {}
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        if humanoid_bone in base_humanoid_to_bone:
            auxiliary_bones[base_humanoid_to_bone[humanoid_bone]] = aux_set["auxiliaryBones"]
    
    # 既存の頂点グループ名を取得
    existing_groups = set(vg.name for vg in mesh_obj.vertex_groups)
    
    # 名前変更が必要なグループを特定
    groups_to_rename = {}
    for group in mesh_obj.vertex_groups:
        if group.name in clothing_bone_to_humanoid:
            humanoid_name = clothing_bone_to_humanoid[group.name]
            if humanoid_name in base_humanoid_to_bone:
                base_bone_name = base_humanoid_to_bone[humanoid_name]
                groups_to_rename[group.name] = base_bone_name
    
    # グループ名を変更
    for old_name, new_name in groups_to_rename.items():
        if old_name in mesh_obj.vertex_groups:
            group = mesh_obj.vertex_groups[old_name]
            group_index = group.index
            # 頂点ごとのウェイトを保存
            weights = {}
            for vert in mesh_obj.data.vertices:
                for g in vert.groups:
                    if g.group == group_index:
                        weights[vert.index] = g.weight
                        break
            
            # グループ名を変更
            group.name = new_name
            
            # 補助ボーンの頂点グループを追加
            if new_name in auxiliary_bones:
                # 補助ボーンの頂点グループを作成
                for aux_bone in auxiliary_bones[new_name]:
                    if aux_bone not in existing_groups:
                        mesh_obj.vertex_groups.new(name=aux_bone)
    
    existing_groups = set(vg.name for vg in mesh_obj.vertex_groups)

    breast_bones_dont_exist = 'LeftBreast' not in clothing_humanoid_to_bone and 'RightBreast' not in clothing_humanoid_to_bone
    
    # Process each humanoid bone from base avatar
    for humanoid_name, bone_name in base_humanoid_to_bone.items():
        # Skip if bone exists in clothing armature
        if bone_name in existing_groups:
            continue

        should_add_optional_humanoid_bone = False
        
        # Condition 1: Chest exists in clothing, UpperChest missing in clothing but exists in base
        if (humanoid_name == "UpperChest" and 
            "Chest" in clothing_humanoid_to_bone and 
            base_humanoid_to_bone["Chest"] in existing_groups and
            "UpperChest" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 2: LeftLowerLeg exists in clothing, LeftFoot missing in clothing but exists in base
        elif (humanoid_name == "LeftFoot" and 
                "LeftLowerLeg" in clothing_humanoid_to_bone and 
                base_humanoid_to_bone["LeftLowerLeg"] in existing_groups and
                "LeftFoot" not in clothing_humanoid_to_bone and
                "LeftFoot" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 2: RightLowerLeg exists in clothing, RightFoot missing in clothing but exists in base
        elif (humanoid_name == "RightFoot" and 
                "RightLowerLeg" in clothing_humanoid_to_bone and 
                base_humanoid_to_bone["RightLowerLeg"] in existing_groups and
                "RightFoot" not in clothing_humanoid_to_bone and
                "RightFoot" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 3: LeftLowerLeg or LeftFoot exists in clothing, LeftToe missing in clothing but exists in base
        elif (humanoid_name == "LeftToe" and 
                (("LeftLowerLeg" in clothing_humanoid_to_bone and base_humanoid_to_bone["LeftLowerLeg"] in existing_groups) or
                ("LeftFoot" in clothing_humanoid_to_bone and base_humanoid_to_bone["LeftFoot"] in existing_groups)) and
                "LeftToe" not in clothing_humanoid_to_bone and
                "LeftToe" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 3: RightLowerLeg or RightFoot exists in clothing, RightToe missing in clothing but exists in base
        elif (humanoid_name == "RightToe" and 
                (("RightLowerLeg" in clothing_humanoid_to_bone and base_humanoid_to_bone["RightLowerLeg"] in existing_groups) or
                ("RightFoot" in clothing_humanoid_to_bone and base_humanoid_to_bone["RightFoot"] in existing_groups)) and
                "RightToe" not in clothing_humanoid_to_bone and
                "RightToe" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 4: LeftShoulder exists in clothing, LeftUpperArm exists in base but not in clothing
        elif (humanoid_name == "LeftUpperArm" and 
                "LeftShoulder" in clothing_humanoid_to_bone and 
                base_humanoid_to_bone["LeftShoulder"] in existing_groups and
                "LeftUpperArm" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 4: RightShoulder exists in clothing, RightUpperArm exists in base but not in clothing
        elif (humanoid_name == "RightUpperArm" and 
                "RightShoulder" in clothing_humanoid_to_bone and 
                base_humanoid_to_bone["RightShoulder"] in existing_groups and
                "RightUpperArm" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 5: LeftBreast exists in clothing, breast bones don't exist in clothing, Chest or UpperChest exists in base
        elif (humanoid_name == "LeftBreast" and breast_bones_dont_exist and
                (base_humanoid_to_bone["Chest"] in existing_groups or base_humanoid_to_bone["UpperChest"] in existing_groups) and
                "LeftBreast" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        # Condition 5: RightBreast exists in clothing, breast bones don't exist in clothing, Chest or UpperChest exists in base
        elif (humanoid_name == "RightBreast" and breast_bones_dont_exist and
                (base_humanoid_to_bone["Chest"] in existing_groups or base_humanoid_to_bone["UpperChest"] in existing_groups) and
                "RightBreast" in base_humanoid_to_bone):
            should_add_optional_humanoid_bone = True
        
        if should_add_optional_humanoid_bone:
            print(f"Adding optional humanoid bone group: {humanoid_name} ({bone_name})")
            if bone_name not in existing_groups:
                mesh_obj.vertex_groups.new(name=bone_name)
            else:
                print(f"Optional humanoid bone group already exists: {bone_name}")
            # 補助ボーンの頂点グループを追加
            if bone_name in auxiliary_bones:
                # 補助ボーンの頂点グループを作成
                for aux_bone in auxiliary_bones[bone_name]:
                    if aux_bone not in existing_groups:
                        mesh_obj.vertex_groups.new(name=aux_bone)
