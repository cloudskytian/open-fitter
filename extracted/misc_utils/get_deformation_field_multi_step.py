import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mathutils import Matrix
from misc_utils.globals import _deformation_field_cache


def get_deformation_field_multi_step(field_data_path: str) -> dict:
    """
    指定されたパスの多段階Deformation Field データを読み込み、KDTree を構築してキャッシュする。
    SaveAndApplyFieldAuto.pyのapply_field_data関数と同様の多段階データ処理をサポート。
    """
    global _deformation_field_cache
    multi_step_key = field_data_path + "_multi_step"
    if multi_step_key in _deformation_field_cache:
        return _deformation_field_cache[multi_step_key]
    
    # Deformation Field のデータ読み込み
    data = np.load(field_data_path, allow_pickle=True)
    
    # データ形式の確認と読み込み
    if 'all_field_points' in data:
        # 新形式：各ステップの座標が保存されている場合
        all_field_points = data['all_field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        print(f"複数ステップのデータ（新形式）を検出: {num_steps}ステップ")
        
        # ミラー設定を確認（データに含まれていない場合はそのまま使用）
        enable_x_mirror = data.get('enable_x_mirror', False)
        print(f"X軸ミラー設定: {'有効' if enable_x_mirror else '無効'}")
        
        if enable_x_mirror:
            # X軸ミラーリング：X座標が0より大きいデータを負に反転してミラーデータを追加
            mirrored_field_points = []
            mirrored_delta_positions = []
            
            for step in range(num_steps):
                field_points = all_field_points[step].copy()
                delta_positions = all_delta_positions[step].copy()
                
                if len(field_points) > 0:
                    # X座標が0より大きいデータを検索
                    x_positive_mask = field_points[:, 0] > 0.0
                    if np.any(x_positive_mask):
                        # ミラーデータを作成
                        mirror_field_points = field_points[x_positive_mask].copy()
                        mirror_delta_positions = delta_positions[x_positive_mask].copy()
                        
                        # X座標とX成分の変位を反転
                        mirror_field_points[:, 0] *= -1.0
                        mirror_delta_positions[:, 0] *= -1.0
                        
                        # 元のデータとミラーデータを結合
                        combined_field_points = np.vstack([field_points, mirror_field_points])
                        combined_delta_positions = np.vstack([delta_positions, mirror_delta_positions])
                        
                        mirrored_field_points.append(combined_field_points)
                        mirrored_delta_positions.append(combined_delta_positions)
                        
                        print(f"ステップ {step+1}: 元の頂点数 {len(field_points)} → ミラー適用後 {len(combined_field_points)}")
                    else:
                        mirrored_field_points.append(field_points)
                        mirrored_delta_positions.append(delta_positions)
                        print(f"ステップ {step+1}: フィールド頂点数 {len(field_points)} (ミラー対象なし)")
                else:
                    mirrored_field_points.append(field_points)
                    mirrored_delta_positions.append(delta_positions)
                    print(f"ステップ {step+1}: フィールド頂点数 0")
            
            # ミラー適用後のデータを使用
            all_field_points = mirrored_field_points
            all_delta_positions = mirrored_delta_positions
        else:
            # ミラーが無効の場合、元のデータをそのまま使用
            print("X軸ミラーリングが無効のため、元のデータをそのまま使用します")
            print("field_data_path: ", field_data_path)
            for step in range(num_steps):
                print(f"ステップ {step+1}: フィールド頂点数 {len(all_field_points[step])}")
        
    elif 'field_points' in data and 'all_delta_positions' in data:
        # 旧形式：単一の座標セットが保存されている場合
        field_points = data['field_points']
        all_delta_positions = data['all_delta_positions']
        num_steps = int(data.get('num_steps', len(all_delta_positions)))
        
        # 旧形式の場合、すべてのステップで同じ座標を使用
        all_field_points = [field_points for _ in range(num_steps)]
        print(f"複数ステップのデータ（旧形式）を検出: {num_steps}ステップ")
    else:
        # 後方互換性のため、単一ステップのデータも処理
        field_points = data.get('field_points', data.get('delta_positions', []))
        delta_positions = data.get('delta_positions', data.get('all_delta_positions', [[]])[0] if 'all_delta_positions' in data else [])
        all_field_points = [field_points]
        all_delta_positions = [delta_positions]
        num_steps = 1
        print("単一ステップのデータを検出")
    
    # weightsが存在しない場合はすべて1のものを使用
    if 'weights' in data:
        field_weights = data['weights']
    else:
        field_weights = np.ones(len(all_field_points[0]) if len(all_field_points) > 0 else 0)
        
    world_matrix = Matrix(data['world_matrix'])
    world_matrix_inv = world_matrix.inverted()

    # kdtree_query_kの値を取得（存在しない場合はデフォルト値8を使用）
    k_neighbors = 8
    # if 'kdtree_query_k' in data:
    #     try:
    #         k_value = data['kdtree_query_k']
    #         k_neighbors = int(k_value)
    #         print(f"kdtree_query_k value: {k_neighbors}")
    #     except Exception as e:
    #         print(f"Warning: Could not process kdtree_query_k value: {e}")
    
    # RBFパラメータの読み込み
    rbf_epsilon = float(data.get('rbf_epsilon', 0.00001))
    print(f"RBF補間パラメータ: 関数=multi_quadratic_biharmonic, epsilon={rbf_epsilon}")
    
    field_info = {
        'data': data,
        'all_field_points': all_field_points,
        'all_delta_positions': all_delta_positions,
        'num_steps': num_steps,
        'field_weights': field_weights,
        'world_matrix': world_matrix,
        'world_matrix_inv': world_matrix_inv,
        'kdtree_query_k': k_neighbors,
        'rbf_epsilon': rbf_epsilon,
        'is_multi_step': num_steps > 1
    }
    _deformation_field_cache[multi_step_key] = field_info
    return field_info
