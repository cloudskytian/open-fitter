import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def calculate_obb_from_points(points):
    """
    点群からOriented Bounding Box (OBB)を計算する
    
    Parameters:
        points: 点群のリスト（Vector型またはタプル）
        
    Returns:
        dict: OBBの情報を含む辞書
            'center': 中心座標
            'axes': 主軸（3x3の行列、各列が軸）
            'radii': 各軸方向の半径
        または None: 計算不能な場合
    """
    
    # 点群が少なすぎる場合はNoneを返す
    if len(points) < 3:
        print(f"警告: 点群が少なすぎます（{len(points)}点）。OBB計算をスキップします。")
        return None
    
    try:
        # 点群をnumpy配列に変換
        points_np = np.array([[p.x, p.y, p.z] for p in points])
        
        # 点群の中心を計算
        center = np.mean(points_np, axis=0)
        
        # 中心を原点に移動
        centered_points = points_np - center
        
        # 共分散行列を計算
        cov_matrix = np.cov(centered_points, rowvar=False)
        
        # 行列のランクをチェック
        if np.linalg.matrix_rank(cov_matrix) < 3:
            print("警告: 共分散行列のランクが不足しています。OBB計算をスキップします。")
            return None
        
        # 固有値と固有ベクトルを計算
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 固有値が非常に小さい場合はスキップ
        if np.any(np.abs(eigenvalues) < 1e-10):
            print("警告: 固有値が非常に小さいです。OBB計算をスキップします。")
            return None
        
        # 固有値の大きさでソート（降順）
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 主軸を取得（列ベクトルとして）
        axes = eigenvectors
        
        # 各軸方向の点の投影を計算
        projections = np.abs(np.dot(centered_points, axes))
        
        # 各軸方向の最大値を半径として使用
        radii = np.max(projections, axis=0)
        
        # 結果を辞書として返す
        return {
            'center': center,
            'axes': axes,
            'radii': radii
        }
    except Exception as e:
        print(f"OBB計算中にエラーが発生しました: {e}")
        return None
