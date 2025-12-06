import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def custom_max_vertex_group_numpy(obj, group_name, neighbors_info, offsets, num_verts,
                                  repeat=3, weight_factor=1.0):
    """
    NumPy を用いたカスタムスムージング (MAXベース) の高速実装
    
    Parameters:
        obj: 対象のメッシュオブジェクト
        group_name: スムージング対象の頂点グループ名
        neighbors_info: create_vertex_neighbors_array で作成した近接頂点情報フラット配列
        offsets: create_vertex_neighbors_array で作成した頂点ごとのオフセット
        num_verts: 頂点数
        repeat: スムージングの繰り返し回数
        weight_factor: 周辺頂点からの最大値に掛ける係数
    """
    if group_name not in obj.vertex_groups:
        print(f"頂点グループ '{group_name}' が見つかりません")
        return
    
    group_index = obj.vertex_groups[group_name].index
    
    # 頂点ウェイトを NumPy 配列で取得
    current_weights = np.zeros(num_verts, dtype=np.float64)
    for v in obj.data.vertices:
        w = 0.0
        for g in v.groups:
            if g.group == group_index:
                w = g.weight
                break
        current_weights[v.index] = w
    
    # スムージングを繰り返し
    for _ in range(repeat):
        new_weights = np.copy(current_weights)
        
        # 各頂点ごとに近接頂点の (weight * factor) の最大値を取る
        for vert_idx in range(num_verts):
            start = offsets[vert_idx]
            end = offsets[vert_idx+1]
            if start == end:
                # 近接頂点がない場合
                continue
            
            # neighbors_info[start:end, 0] -> neighbor_idx (float なので int にキャスト)
            neighbor_idx = neighbors_info[start:end, 0].astype(np.int64)
            dist_factors = neighbors_info[start:end, 1]  # weight_dist_factor
            
            # 周囲頂点のウェイトに距離係数を掛け合わせ、その最大値を求める
            local_max = np.max(current_weights[neighbor_idx] * dist_factors)
            
            # 現在ウェイトと比較して大きい方を適用
            new_weights[vert_idx] = max(new_weights[vert_idx], local_max * weight_factor)
        
        current_weights = new_weights
    
    # 計算結果を頂点グループに反映 (まとめて書き戻し)
    vg = obj.vertex_groups[group_name]
    for vert_idx in range(num_verts):
        w = current_weights[vert_idx]
        if w > 1.0:
            w = 1.0
        vg.add([vert_idx], float(w), 'REPLACE')
