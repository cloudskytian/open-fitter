import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Tuple

from blender_utils.build_bone_hierarchy import build_bone_hierarchy


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
