import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

import bpy
import numpy as np
from scipy.spatial import cKDTree


def create_vertex_neighbors_array(obj, expand_distance=0.05, sigma=0.02):
    """
    各頂点の近接頂点情報を NumPy 配列形式で作成する
    
    Parameters:
        obj: 対象のメッシュオブジェクト
        expand_distance: 検索範囲（メートル単位）
        sigma: ガウス関数の標準偏差
        
    Returns:
        neighbors_info (np.ndarray): shape = (M, 2) のフラット配列
                                    各行は [neighbor_idx, weight_factor]
        offsets (np.ndarray): shape = (num_verts+1,)
                              頂点 i の近接データは neighbors_info[offsets[i]:offsets[i+1]] に格納
        num_verts (int): 頂点数
    """
    # 評価済みメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data

    num_verts = len(eval_mesh.vertices)
    
    # 頂点のワールド座標を取得
    world_coords = [eval_obj.matrix_world @ v.co for v in eval_mesh.vertices]
    
    # KDTreeを構築
    kdtree = cKDTree(world_coords)
    
    # ガウス関数
    def gaussian(distance, sigma):
        return math.exp(-(distance**2) / (2 * sigma**2))
    
    # 近傍頂点リストを作成
    neighbors_list = [[] for _ in range(num_verts)]
    for vert_idx, vert_world in enumerate(world_coords):
        # 範囲内の頂点を検索
        for idx in kdtree.query_ball_point(vert_world, expand_distance):
            if idx != vert_idx:
                dist = (world_coords[idx] - vert_world).length
                weight_factor = gaussian(dist, sigma)
                neighbors_list[vert_idx].append((idx, weight_factor))
    
    # フラットな配列とオフセット配列を作成
    # offsets[i] は i 番目頂点の近接配列が始まるインデックスを表す
    offsets = np.zeros(num_verts+1, dtype=np.int64)
    for i in range(num_verts):
        offsets[i+1] = offsets[i] + len(neighbors_list[i])
    
    flat_data = []
    for i in range(num_verts):
        flat_data.extend(neighbors_list[i])
    
    # (neighbor_idx, weight_factor) -> NumPy 配列化
    neighbors_info = np.array(flat_data, dtype=np.float64)  # shape = (M, 2)
    # ただし neighbor_idx は整数なので、後で int にキャストして使う
    
    return neighbors_info, offsets, num_verts
