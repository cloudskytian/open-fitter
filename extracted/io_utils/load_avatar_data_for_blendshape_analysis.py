import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json


def load_avatar_data_for_blendshape_analysis(avatar_data_path: str) -> dict:
    """
    BlendShape分析用にアバターデータを読み込む
    
    Parameters:
        avatar_data_path: アバターデータファイルのパス
        
    Returns:
        dict: アバターデータ
    """
    try:
        with open(avatar_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading avatar data {avatar_data_path}: {e}")
        return {}
