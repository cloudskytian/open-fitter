import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mathutils import Matrix


def list_to_matrix(matrix_list):
    """
    リストからMatrix型に変換する（JSON読み込み用）
    
    Parameters:
        matrix_list: list - 行列のデータを含む2次元リスト
        
    Returns:
        Matrix: 変換された行列
    """
    return Matrix(matrix_list)
