import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def get_child_bones_recursive(bone_name: str, armature_obj: bpy.types.Object, clothing_avatar_data: dict = None, is_root: bool = True) -> set:
    """
    指定されたボーンのすべての子ボーンを再帰的に取得する
    最初に指定されたボーンではないHumanoidボーンとそれ以降の子ボーンは除外する
    
    Parameters:
        bone_name: 親ボーンの名前
        armature_obj: アーマチュアオブジェクト
        clothing_avatar_data: 衣装のアバターデータ（Humanoidボーンの判定に使用）
        is_root: 最初に指定されたボーンかどうか
        
    Returns:
        set: 子ボーンの名前のセット
    """
    children = set()
    if bone_name not in armature_obj.data.bones:
        return children
    
    # Humanoidボーンの判定用マッピングを作成
    humanoid_bones = set()
    if clothing_avatar_data:
        for bone_map in clothing_avatar_data.get("humanoidBones", []):
            if "boneName" in bone_map:
                humanoid_bones.add(bone_map["boneName"])
    
    bone = armature_obj.data.bones[bone_name]
    for child in bone.children:
        # 最初に指定されたボーンではないHumanoidボーンの場合、そのボーンとその子ボーンを除外
        if not is_root and child.name in humanoid_bones:
            # このボーンとその子ボーンをスキップ
            continue
        
        children.add(child.name)
        children.update(get_child_bones_recursive(child.name, armature_obj, clothing_avatar_data, False))
    
    return children
