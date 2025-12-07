"""
アバターデータの読み込みユーティリティ
"""

import json


def load_avatar_data(filepath: str) -> dict:
    """Load and parse avatar data from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load avatar data: {str(e)}")


def load_avatar_data_for_blendshape_analysis(avatar_data_path: str) -> dict:
    """
    BlendShape分析用にアバターデータを読み込む
    
    Parameters:
        avatar_data_path: アバターデータファイルのパス
        
    Returns:
        dict: アバターデータ（エラー時は空辞書）
    """
    try:
        with open(avatar_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading avatar data {avatar_data_path}: {e}")
        return {}
