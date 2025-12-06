import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def get_deformation_bones(armature_obj: bpy.types.Object, avatar_data: dict) -> list:
    """
    アバターデータを参照し、HumanoidボーンとAuxiliaryボーン以外のボーンを取得
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
        avatar_data: アバターデータ
        
    Returns:
        変形対象のボーン名のリスト
    """
    # HumanoidボーンとAuxiliaryボーンのセットを作成
    excluded_bones = set()
    
    # Humanoidボーンを追加
    for bone_map in avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            excluded_bones.add(bone_map["boneName"])
    
    # 補助ボーンを追加
    for aux_set in avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set.get("auxiliaryBones", []):
            excluded_bones.add(aux_bone)
    
    # 除外ボーン以外のすべてのボーンを取得
    deform_bones = []
    for bone in armature_obj.data.bones:
        if bone.name not in excluded_bones:
            deform_bones.append(bone.name)
    
    return deform_bones
