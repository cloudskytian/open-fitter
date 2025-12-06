import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mathutils import Vector
from scipy.spatial import cKDTree


def batch_process_vertices_with_custom_range(vertices, all_field_points, all_delta_positions, field_weights, 
                                            field_matrix, field_matrix_inv, target_matrix, target_matrix_inv, 
                                            start_value, end_value, 
                                            deform_weights=None, rbf_epsilon=0.00001, batch_size=1000, k=8):
    """
    任意の値の範囲でフィールドによる変形を行う
    
    Parameters:
        vertices: 処理対象の頂点配列
        all_field_points: 各ステップのフィールドポイント配列
        all_delta_positions: 各ステップのデルタポジション配列
        field_weights: フィールドウェイト
        field_matrix: フィールドマトリックス
        field_matrix_inv: フィールドマトリックスの逆行列
        target_matrix: ターゲットマトリックス
        target_matrix_inv: ターゲットマトリックスの逆行列
        start_value: 開始値（シェイプキー値）
        end_value: 終了値（シェイプキー値）
        deform_weights: 変形ウェイト
        rbf_epsilon: RBF補間のイプシロン値
        batch_size: バッチサイズ
        k: 近傍点数
        
    Returns:
        変形後の頂点配列（ワールド座標）
    """
    num_vertices = len(vertices)
    num_steps = len(all_field_points)
    
    # 累積変位を初期化
    cumulative_displacements = np.zeros((num_vertices, 3))
    # 現在の頂点位置（ワールド座標）を保存
    current_world_positions = np.array([target_matrix @ Vector(v) for v in vertices])

    # もしdeform_weightsがNoneの場合は、全ての頂点のウェイトを1.0とする
    if deform_weights is None:
        deform_weights = np.ones(num_vertices)
    
    # ステップごとの値を計算
    step_size = 1.0 / num_steps
    
    # 各ステップで処理
    processed_steps = []
    for step in range(num_steps):
        step_start = step * step_size
        step_end = (step + 1) * step_size
        # start_valueからend_valueに増加（start_value < end_value）
        if step_start + 0.00001 <= end_value and step_end - 0.00001 >= start_value:
            processed_steps.append((step, step_start, step_end))
    
    print(f"処理対象ステップ: {len(processed_steps)}")
    
    # 各ステップの変位を累積的に適用
    for step_idx, (step, step_start, step_end) in enumerate(processed_steps):
        field_points = all_field_points[step].copy()
        delta_positions = all_delta_positions[step].copy()
        original_delta_positions = all_delta_positions[step].copy()
        
        print(f"ステップ {step_idx+1}/{len(processed_steps)} (step {step}) の変形を適用中...")
        print(f"ステップ値範囲: {step_start:.3f} -> {step_end:.3f}")
        print(f"使用するフィールド頂点数: {len(field_points)}")
        
        # 任意の値からの変形
        if start_value != step_start:
            if start_value >= step_start + 0.00001:
                # 開始値がステップの開始値より大きい場合
                adjustment_factor = (start_value - step_start) / step_size
                adjustment_delta = original_delta_positions * adjustment_factor
                field_points += adjustment_delta
                delta_positions -= adjustment_delta
        if end_value != step_end:
            if end_value <= step_end - 0.00001:
                # 終了値がステップの終了値より小さい場合
                adjustment_factor = (step_end - end_value) / step_size
                adjustment_delta = original_delta_positions * adjustment_factor
                delta_positions -= adjustment_delta
        
        # KDTreeを使用して近傍点を検索
        kdtree = cKDTree(field_points)
        
        # カスタムRBF補間で新しい頂点位置を計算
        step_displacements = np.zeros((num_vertices, 3))
        
        for start_idx in range(0, num_vertices, batch_size):
            end_idx = min(start_idx + batch_size, num_vertices)
            batch_weights = deform_weights[start_idx:end_idx]
            
            # バッチ内の全頂点をフィールド空間に変換
            batch_world = current_world_positions[start_idx:end_idx].copy()
            batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
            
            # 各頂点ごとに逆距離加重法で補間
            batch_displacements = np.zeros((len(batch_field), 3))
            
            for i, point in enumerate(batch_field):
                # 近傍点を検索（最大k点）
                k_use = min(k, len(field_points))
                distances, indices = kdtree.query(point, k=k_use)
                
                # 距離が0の場合（完全に一致する点がある場合）
                if distances[0] < 1e-10:
                    batch_displacements[i] = delta_positions[indices[0]]
                    continue
                
                # 逆距離の重みを計算
                weights = 1.0 / np.sqrt(distances**2 + rbf_epsilon**2)
                
                # 重みの正規化
                weights /= np.sum(weights)
                
                # 重み付き平均で変位を計算
                weighted_deltas = delta_positions[indices] * weights[:, np.newaxis]
                batch_displacements[i] = np.sum(weighted_deltas, axis=0) * batch_weights[i]
            
            # ワールド空間での変位を計算
            for i, displacement in enumerate(batch_displacements):
                world_displacement = field_matrix.to_3x3() @ Vector(displacement)
                step_displacements[start_idx + i] = world_displacement
                
                # 現在のワールド位置を更新（次のステップのために）
                current_world_positions[start_idx + i] += world_displacement
        
        # このステップの変位を累積変位に追加
        cumulative_displacements += step_displacements
        
        print(f"ステップ {step_idx+1} 完了")
    
    # 最終的な変形後の位置を返す
    final_world_positions = np.array([target_matrix @ Vector(v) for v in vertices]) + cumulative_displacements
    return final_world_positions
