import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def load_deformation_field_num_steps(field_file_path: str, config_dir: str) -> int:
    """
    変形フィールドファイルからnum_stepsを読み込む
    
    Parameters:
        field_file_path: 変形フィールドファイルのパス（相対パス可）
        config_dir: 設定ファイルのディレクトリ
        
    Returns:
        int: num_stepsの値、読み込めない場合は1
    """
    try:
        # 相対パスの場合は絶対パスに変換
        if not os.path.isabs(field_file_path):
            field_file_path = os.path.join(config_dir, field_file_path)
        
        if os.path.exists(field_file_path):
            field_data = np.load(field_file_path, allow_pickle=True)
            return int(field_data.get('num_steps', 1))
        else:
            print(f"Warning: Deformation field file not found: {field_file_path}")
            return 1
    except Exception as e:
        print(f"Warning: Failed to load num_steps from {field_file_path}: {e}")
        return 1
