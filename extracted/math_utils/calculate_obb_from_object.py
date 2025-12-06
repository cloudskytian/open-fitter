import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np


def calculate_obb_from_object(obj):
    """
    オブジェクトのOriented Bounding Box (OBB)を計算する
    
    Parameters:
        obj: 対象のメッシュオブジェクト
        
    Returns:
        dict: OBBの情報（中心、軸、半径）
    """
    # 評価済みメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    
    # 頂点座標をワールド空間で取得
    vertices = np.array([obj.matrix_world @ v.co for v in eval_mesh.vertices])
    
    if len(vertices) == 0:
        return None
    
    # 頂点の平均位置（中心）を計算
    center = np.mean(vertices, axis=0)
    
    # 中心を原点に移動
    centered_vertices = vertices - center
    
    # 共分散行列を計算
    covariance_matrix = np.cov(centered_vertices.T)
    
    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # 固有ベクトルを正規化
    for i in range(3):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    
    # 各軸に沿った投影の最大値を計算
    min_proj = np.full(3, float('inf'))
    max_proj = np.full(3, float('-inf'))
    
    for vertex in centered_vertices:
        for i in range(3):
            proj = np.dot(vertex, eigenvectors[:, i])
            min_proj[i] = min(min_proj[i], proj)
            max_proj[i] = max(max_proj[i], proj)
    
    # 半径（各軸方向の長さの半分）を計算
    radii = (max_proj - min_proj) / 2
    
    # 中心位置を調整
    adjusted_center = center + np.sum([(min_proj[i] + max_proj[i]) / 2 * eigenvectors[:, i] for i in range(3)], axis=0)
    
    return {
        'center': adjusted_center,
        'axes': eigenvectors,
        'radii': radii
    }
