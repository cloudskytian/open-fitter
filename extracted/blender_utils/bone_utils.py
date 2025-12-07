import os
import sys

from typing import Dict
from typing import Dict, Tuple
import bpy
import os
import sys


# Merged from build_bone_hierarchy.py

def build_bone_hierarchy(bone_node: dict, bone_parents: Dict[str, str], current_path: list):
    """
    ボーン階層から親子関係のマッピングを再帰的に構築する

    Parameters:
        bone_node (dict): 現在のボーンノード
        bone_parents (Dict[str, str]): ボーン名から親ボーン名へのマッピング
        current_path (list): 現在のパス上のボーン名のリスト
    """
    bone_name = bone_node['name']
    if current_path:
        bone_parents[bone_name] = current_path[-1]
    
    current_path.append(bone_name)
    for child in bone_node.get('children', []):
        build_bone_hierarchy(child, bone_parents, current_path)
    current_path.pop()

# Merged from get_bone_name_from_humanoid.py

def get_bone_name_from_humanoid(avatar_data: dict, humanoid_bone_name: str) -> str:
    """
    humanoidBoneNameから実際のボーン名を取得する
    
    Parameters:
        avatar_data: アバターデータ
        humanoid_bone_name: ヒューマノイドボーン名
        
    Returns:
        実際のボーン名、見つからない場合はNone
    """
    for bone_map in avatar_data.get("humanoidBones", []):
        if bone_map["humanoidBoneName"] == humanoid_bone_name:
            return bone_map["boneName"]
    return None

# Merged from get_bone_parent_map.py

def get_bone_parent_map(bone_hierarchy: dict) -> dict:
    """
    Create a map of bones to their parents from the hierarchy.
    
    Parameters:
        bone_hierarchy: Bone hierarchy data from avatar data
    
    Returns:
        Dictionary mapping bone names to their parent bone names
    """
    parent_map = {}
    
    def traverse_hierarchy(node, parent=None):
        current_bone = node["name"]
        parent_map[current_bone] = parent
        
        for child in node.get("children", []):
            traverse_hierarchy(child, current_bone)
    
    traverse_hierarchy(bone_hierarchy)
    return parent_map

# Merged from get_child_bones_recursive.py

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

# Merged from get_deformation_bones.py

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

# Merged from get_humanoid_bone_hierarchy.py

def get_humanoid_bone_hierarchy(avatar_data: dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    アバターデータからHumanoidボーンの階層関係を抽出する

    Parameters:
        avatar_data (dict): アバターデータ

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]: 
            (ボーン名から親への辞書, Humanoidボーン名からボーン名への辞書, ボーン名からHumanoidボーン名への辞書)
    """
    # ボーンの親子関係を構築
    bone_parents = {}
    build_bone_hierarchy(avatar_data['boneHierarchy'], bone_parents, [])

    # Humanoidボーン名とボーン名の対応マップを作成
    humanoid_to_bone = {bone_map['humanoidBoneName']: bone_map['boneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    bone_to_humanoid = {bone_map['boneName']: bone_map['humanoidBoneName'] 
                       for bone_map in avatar_data['humanoidBones']}
    
    return bone_parents, humanoid_to_bone, bone_to_humanoid