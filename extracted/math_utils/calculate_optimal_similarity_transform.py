import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def calculate_optimal_similarity_transform(source_points, target_points):
    """
    2つの点群間の最適な相似変換（スケール、回転、平行移動）を計算する
    
    Parameters:
        source_points: 変換元の点群 (Nx3 のNumPy配列)
        target_points: 変換先の点群 (Nx3 のNumPy配列)
        
    Returns:
        (s, R, t): スケーリング係数 (スカラー), 回転行列 (3x3), 平行移動ベクトル (3x1)
    """
    # 点群の重心を計算
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    # 重心を原点に移動
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    
    # ソース点群の二乗和を計算（スケーリング係数の計算用）
    source_scale = np.sum(source_centered**2)
    
    # 共分散行列を計算
    H = source_centered.T @ target_centered
    
    # 特異値分解
    U, S, Vt = np.linalg.svd(H)
    
    # 回転行列を計算
    R = Vt.T @ U.T
    
    # 反射を防ぐ（行列式が負の場合）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 最適なスケーリング係数を計算
    trace_RSH = np.sum(S)
    s = trace_RSH / source_scale if source_scale > 0 else 1.0
    
    # 平行移動ベクトルを計算
    t = centroid_target - s * (R @ centroid_source)
    
    return s, R, t
