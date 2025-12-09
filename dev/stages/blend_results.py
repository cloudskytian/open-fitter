import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np


def blend_results(context):
    """
    weights_a と weights_b を falloff_weight でブレンドする
    NumPy配列を使用して高速化
    """
    num_verts = len(context.target_obj.data.vertices)
    falloff_group_idx = context.distance_falloff_group.index
    
    # falloff_weight を一括取得
    falloff_weights = np.zeros(num_verts, dtype=np.float32)
    for vert in context.target_obj.data.vertices:
        for g in vert.groups:
            if g.group == falloff_group_idx:
                falloff_weights[vert.index] = g.weight
                break
    
    # 対象グループをリスト化してインデックスマップを作成
    target_groups = [name for name in context.bone_groups if name in context.target_obj.vertex_groups]
    group_objects = {name: context.target_obj.vertex_groups[name] for name in target_groups}
    
    # weights_a, weights_b を NumPy 配列に展開
    weights_a_arr = np.zeros((num_verts, len(target_groups)), dtype=np.float32)
    weights_b_arr = np.zeros((num_verts, len(target_groups)), dtype=np.float32)
    
    for g_idx, group_name in enumerate(target_groups):
        for vert_idx in range(num_verts):
            weights_a_arr[vert_idx, g_idx] = context.weights_a[vert_idx].get(group_name, 0.0)
            weights_b_arr[vert_idx, g_idx] = context.weights_b[vert_idx].get(group_name, 0.0)
    
    # ブレンド計算（ベクトル化）
    falloff_expanded = falloff_weights[:, np.newaxis]
    final_weights = weights_a_arr * falloff_expanded + weights_b_arr * (1.0 - falloff_expanded)
    
    # 頂点グループへの一括適用
    for g_idx, group_name in enumerate(target_groups):
        group = group_objects[group_name]
        weights_col = final_weights[:, g_idx]
        
        # 正のウェイトを持つ頂点のみ処理
        positive_mask = weights_col > 0
        positive_indices = np.where(positive_mask)[0]
        zero_indices = np.where(~positive_mask)[0]
        
        # 一括追加（チャンク処理で効率化）
        for vert_idx in positive_indices:
            group.add([int(vert_idx)], float(weights_col[vert_idx]), "REPLACE")
        
        # ゼロウェイトの削除
        for vert_idx in zero_indices:
            try:
                group.remove([int(vert_idx)])
            except RuntimeError:
                pass
