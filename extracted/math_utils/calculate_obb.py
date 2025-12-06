import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def calculate_obb(vertices_world):
    """
    頂点のワールド座標から最適な向きのバウンディングボックスを計算
    
    Parameters:
        vertices_world: 頂点のワールド座標のリスト
        
    Returns:
        (axes, extents): 主軸方向と、各方向の半分の長さ
    """
    if vertices_world is None or len(vertices_world) < 3:
        return None, None
    
    # 点群の重心を計算
    centroid = np.mean(vertices_world, axis=0)
    
    # 重心を原点に移動
    centered = vertices_world - centroid
    
    # 共分散行列を計算
    cov = np.cov(centered, rowvar=False)
    
    # 固有ベクトルと固有値を計算
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 固有ベクトルが主軸となる
    axes = eigenvectors
    
    # 各軸方向のextentを計算
    extents = np.zeros(3)
    for i in range(3):
        axis = axes[:, i]
        projection = np.dot(centered, axis)
        extents[i] = (np.max(projection) - np.min(projection)) / 2.0
    
    return axes, extents
