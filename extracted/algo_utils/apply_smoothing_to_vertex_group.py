import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time

import bpy
import numpy as np
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from scipy.spatial import cKDTree


def apply_smoothing_to_vertex_group(cloth_obj, vertex_group_name, smoothing_radius=0.02, iteration=1, use_distance_weighting=True, gaussian_falloff=True, neighbors_cache=None):
    """
    指定された頂点グループに対してスムージング処理を適用します
    距離による重み付きスムージングを使用して、頂点密度の偏りに対して頑健な結果を得ます
    
    Parameters:
    cloth_obj (obj): 衣装メッシュのオブジェクト
    vertex_group_name (str): 対象の頂点グループ名
    smoothing_radius (float): スムージング適用半径
    use_distance_weighting (bool): 距離による重み付けを使用するかどうか
    gaussian_falloff (bool): ガウシアン減衰を使用するかどうか
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
    for i, vertex in enumerate(cloth_obj.data.vertices):
        for group in vertex.groups:
            if group.group == vertex_group.index:
                current_weights[i] = group.weight
                break
    
    # cKDTreeを使用して近傍検索を効率化
    kdtree = cKDTree(vertex_coords)
    
    # スムージング済みウェイト配列を初期化
    smoothed_weights = np.copy(current_weights)
    
    print(f"  スムージング処理開始 (半径: {smoothing_radius}, 距離重み付け: {use_distance_weighting}, ガウシアン減衰: {gaussian_falloff})")
    
    # ガウシアン関数のシグマ値（半径の1/3程度が適切）
    sigma = smoothing_radius / 3.0
    
    # 最初のイテレーションでneighbor_indicesをキャッシュ
    if neighbors_cache is None:
        neighbors_cache = {}
    
    for iteration_idx in range(iteration):
        # 各頂点に対してスムージングを適用
        for i in range(num_vertices):
            # 最初のイテレーションでneighbor_indicesを計算・キャッシュ、二回目以降はキャッシュを使用
            if iteration_idx == 0:
                if i not in neighbors_cache:
                    neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)
                    neighbors_cache[i] = neighbor_indices
                else:
                    neighbor_indices = neighbors_cache[i]
            else:
                neighbor_indices = neighbors_cache[i]
            
            if len(neighbor_indices) > 1:  # 自分自身以外の近傍が存在する場合
                # 近傍頂点への距離を計算
                neighbor_coords = vertex_coords[neighbor_indices]
                distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
                
                # 近傍頂点のウェイト値を取得
                neighbor_weights = current_weights[neighbor_indices]
                
                if use_distance_weighting:
                    if gaussian_falloff:
                        # ガウシアン減衰による重み計算
                        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
                    else:
                        # 線形減衰による重み計算
                        weights = np.maximum(0, 1.0 - distances / smoothing_radius)
                    
                    # 自分自身の重みを少し強めに設定（オリジナル値の保持）
                    # self_index = np.where(distances == 0)[0]
                    # if len(self_index) > 0:
                    #     weights[self_index[0]] *= 2.0
                    
                    # 重み付き平均を計算
                    if np.sum(weights) > 0.001:
                        smoothed_weights[i] = np.sum(neighbor_weights * weights) / np.sum(weights)
                    else:
                        smoothed_weights[i] = current_weights[i]
                else:
                    # 従来の単純平均
                    smoothed_weights[i] = np.mean(neighbor_weights)
            else:
                # 近傍頂点が自分だけの場合は元の値を保持
                smoothed_weights[i] = current_weights[i]
        current_weights = np.copy(smoothed_weights)
    
    # 新しいウェイトを頂点グループに適用
    for i in range(num_vertices):
        vertex_group.add([i], smoothed_weights[i], 'REPLACE')
    
    # BMeshをクリーンアップ
    cloth_bm.free()
    
    # 元のモードに戻す
    bpy.ops.object.mode_set(mode=current_mode)
    
    total_time = time.time() - start_time
    print(f"  スムージング完了: {total_time:.2f}秒")

    return neighbors_cache
