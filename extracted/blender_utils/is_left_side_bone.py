import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common_utils.strip_numeric_suffix import strip_numeric_suffix


def is_left_side_bone(bone_name: str, humanoid_name: str = None) -> bool:
    """
    ボーンが左側かどうかを判定
    
    Parameters:
        bone_name: ボーン名
        humanoid_name: Humanoidボーン名（オプション）
        
    Returns:
        bool: 左側のボーンの場合True
    """
    # Humanoidボーン名での判定
    if humanoid_name and any(k in humanoid_name for k in ["Left", "left"]):
        return True
        
    # 末尾の数字を削除
    cleaned_name = strip_numeric_suffix(bone_name)
        
    # ボーン名での判定
    if any(k in cleaned_name for k in ["Left", "left"]):
        return True
        
    # 末尾での判定（スペースを含む場合も考慮）
    suffixes = ["_L", ".L", " L"]
    return any(cleaned_name.endswith(suffix) for suffix in suffixes)
