import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time

import bpy
import numpy as np
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from scipy.spatial import cKDTree


def apply_min_filter_to_vertex_group(cloth_obj, vertex_group_name, filter_radius=0.02, filter_mask=None):
    """
    頂点グループに対してMinフィルターを適用します
    各頂点から一定距離内にある頂点のウェイトの最小値を取得し、その値を新しいウェイトとして設定します
    
    Parameters:
    cloth_obj (obj): 衣装メッシュのオブジェクト
    vertex_group_name (str): 対象の頂点グループ名
    filter_radius (float): フィルター適用半径
    filter_mask (obj): フィルタリングに使用する頂点グループ
    """
    start_time = time.time()
    
    if vertex_group_name not in cloth_obj.vertex_groups:
        print(f"エラー: 頂点グループ '{vertex_group_name}' が見つかりません")
        return
    
    vertex_group = cloth_obj.vertex_groups[vertex_group_name]
    
    # 現在のモードを保存
    current_mode = bpy.context.object.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # モディファイア適用後のメッシュを取得
    cloth_bm = get_evaluated_mesh(cloth_obj)
    cloth_bm.verts.ensure_lookup_table()
    
    # 頂点座標をnumpy配列に変換
    vertex_coords = np.array([v.co for v in cloth_bm.verts])
    num_vertices = len(vertex_coords)
    
    # 現在のウェイト値を取得
    current_weights = np.zeros(num_vertices, dtype=np.float32)
    for i, vertex in enumerate(cloth_bm.verts):
        # 頂点グループのウェイトを取得
        weight = 0.0
        for group in cloth_obj.data.vertices[i].groups:
            if group.group == vertex_group.index:
                weight = group.weight
                break
        current_weights[i] = weight
    
    # cKDTreeを使用して近傍検索を効率化
    kdtree = cKDTree(vertex_coords)
    
    # 新しいウェイト配列を初期化
    new_weights = np.copy(current_weights)
    
    print(f"  Minフィルター処理開始 (半径: {filter_radius})")
    
    # 各頂点に対してMinフィルターを適用
    for i in range(num_vertices):
        # 一定半径内の近傍頂点のインデックスを取得
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], filter_radius)
        
        if neighbor_indices:
            # 近傍頂点のウェイトの最小値を取得
            neighbor_weights = current_weights[neighbor_indices]
            min_weight = np.min(neighbor_weights)
            if filter_mask is not None:
                new_weights[i] = filter_mask[i] * min_weight + (1 - filter_mask[i]) * current_weights[i]
            else:
                new_weights[i] = min_weight
    
    # 新しいウェイトを頂点グループに適用
    for i in range(num_vertices):
        vertex_group.add([i], new_weights[i], 'REPLACE')
    
    # BMeshをクリーンアップ
    cloth_bm.free()
    
    # 元のモードに戻す
    bpy.ops.object.mode_set(mode=current_mode)
    
    total_time = time.time() - start_time
    print(f"  Minフィルター完了: {total_time:.2f}秒")
