import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict


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
