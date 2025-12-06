import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def apply_similarity_transform_to_points(points, s, R, t):
    """
    点群に相似変換を適用する
    
    Parameters:
        points: 変換する点群 (Nx3 のNumPy配列)
        s: スケーリング係数 (スカラー)
        R: 回転行列 (3x3)
        t: 平行移動ベクトル (3x1)
        
    Returns:
        transformed_points: 変換後の点群 (Nx3 のNumPy配列)
    """
    return s * (R @ points.T).T + t
