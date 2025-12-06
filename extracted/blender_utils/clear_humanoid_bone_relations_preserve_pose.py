import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from io_utils.load_avatar_data import load_avatar_data


def clear_humanoid_bone_relations_preserve_pose(armature_obj, clothing_avatar_data_filepath, base_avatar_data_filepath):
    """
    Humanoidボーンの親子関係を解除しながらワールド空間でのポーズを保持する。
    ベースアバターのアバターデータにないHumanoidボーンの親子関係は保持する。
    
    Args:
        armature_obj: bpy.types.Object - アーマチュアオブジェクト
        clothing_avatar_data_filepath: str - 衣装のアバターデータのJSONファイル名
        base_avatar_data_filepath: str - ベースアバターのアバターデータのJSONファイル名
    """
    if armature_obj.type != 'ARMATURE':
        raise ValueError("Selected object must be an armature")
    
    # アバターデータを読み込む
    clothing_avatar_data = load_avatar_data(clothing_avatar_data_filepath)
    base_avatar_data = load_avatar_data(base_avatar_data_filepath)
    
    # 衣装のHumanoidボーンのセットを作成
    clothing_humanoid_bones = {bone_map['boneName'] for bone_map in clothing_avatar_data['humanoidBones']}
    
    # ベースアバターのHumanoidボーンのセットを作成
    base_humanoid_bones = {bone_map['humanoidBoneName'] for bone_map in base_avatar_data['humanoidBones']}
    
    # 衣装のHumanoidボーン名からHumanoidボーン名への変換マップを作成
    clothing_bone_to_humanoid = {bone_map['boneName']: bone_map['humanoidBoneName'] 
                                for bone_map in clothing_avatar_data['humanoidBones']}
    
    # 親子関係を解除するボーンを特定（ベースアバターにも存在するHumanoidボーンのみ）
    bones_to_unparent = set()
    for bone_name in clothing_humanoid_bones:
        humanoid_name = clothing_bone_to_humanoid.get(bone_name)
        if humanoid_name == "UpperChest" or humanoid_name == "LeftBreast" or humanoid_name == "RightBreast" or humanoid_name == "LeftToes" or humanoid_name == "RightToes":
            continue
        bones_to_unparent.add(bone_name)
        #if humanoid_name in base_humanoid_bones:
        #    bones_to_unparent.add(bone_name)
    
    # Get the armature data
    armature = armature_obj.data
    
    # Store original world space matrices for bones to unparent
    original_matrices = {}
    for bone in armature.bones:
        if bone.name in bones_to_unparent:
            pose_bone = armature_obj.pose.bones[bone.name]
            original_matrices[bone.name] = armature_obj.matrix_world @ pose_bone.matrix
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    
    # Switch to edit mode to modify bone relations
    bpy.context.view_layer.objects.active = armature_obj
    original_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Clear parent relationships for specified bones only
    for edit_bone in armature.edit_bones:
        if edit_bone.name in bones_to_unparent:
            edit_bone.parent = None
    
    # Return to pose mode
    bpy.ops.object.mode_set(mode='POSE')
    
    # Restore original world space positions for unparented bones
    for bone_name, original_matrix in original_matrices.items():
        pose_bone = armature_obj.pose.bones[bone_name]
        pose_bone.matrix = armature_obj.matrix_world.inverted() @ original_matrix
    
    # Return to original mode
    bpy.ops.object.mode_set(mode=original_mode)
