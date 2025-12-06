import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Optional


def find_nearest_parent_with_pose(bone_name: str, 
                                bone_parents: Dict[str, str], 
                                bone_to_humanoid: Dict[str, str],
                                pose_data: dict) -> Optional[str]:
    """
    指定されたボーンの親を辿り、ポーズデータを持つ最も近い親のHumanoidボーン名を返す

    Parameters:
        bone_name (str): 開始ボーン名
        bone_parents (Dict[str, str]): ボーンの親子関係辞書
        bone_to_humanoid (Dict[str, str]): ボーン名からHumanoidボーン名への変換辞書
        pose_data (dict): ポーズデータ

    Returns:
        Optional[str]: 見つかった親のHumanoidボーン名、見つからない場合はNone
    """
    current_bone = bone_name
    while current_bone in bone_parents:
        parent_bone = bone_parents[current_bone]
        if parent_bone in bone_to_humanoid:
            parent_humanoid = bone_to_humanoid[parent_bone]
            if parent_humanoid in pose_data:
                return parent_humanoid
        current_bone = parent_bone
    return None
