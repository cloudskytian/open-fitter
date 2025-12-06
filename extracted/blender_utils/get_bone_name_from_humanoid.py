import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
