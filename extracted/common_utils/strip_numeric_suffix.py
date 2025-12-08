import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re


def strip_numeric_suffix(bone_name: str) -> str:
    """
    ボーン名から末尾の '.数字' パターンを削除
    
    Parameters:
        bone_name: ボーン名
        
    Returns:
        str: '.数字' が削除されたボーン名
    """
    return re.sub(r'\.[\d]+$', '', bone_name)
