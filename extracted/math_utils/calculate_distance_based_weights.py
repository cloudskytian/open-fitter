import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


def calculate_distance_based_weights(source_obj_name, target_obj_name, vertex_group_name="DistanceWeight", min_distance=0.0, max_distance=0.03):
    """
    指定されたオブジェクトの各頂点から別のオブジェクトまでの最近接面距離を計測し、
    距離に基づいて頂点ウェイトを設定する関数
    
    Args:
        source_obj_name (str): ウェイトを設定するオブジェクト名
        target_obj_name (str): 距離計測対象のオブジェクト名
        vertex_group_name (str): 作成する頂点グループ名
        min_distance (float): 最小距離（ウェイト1.0になる距離）
        max_distance (float): 最大距離（ウェイト0.0になる距離）
    """
    
    # オブジェクトを取得
    source_obj = bpy.data.objects.get(source_obj_name)
    target_obj = bpy.data.objects.get(target_obj_name)
    
    if not source_obj:
        print(f"エラー: オブジェクト '{source_obj_name}' が見つかりません")
        return False
    
    if not target_obj:
        print(f"エラー: オブジェクト '{target_obj_name}' が見つかりません")
        return False
    
    # メッシュデータを取得
    source_mesh = source_obj.data
    target_mesh = target_obj.data
    
    # 頂点グループを作成または取得
    if vertex_group_name not in source_obj.vertex_groups:
        vertex_group = source_obj.vertex_groups.new(name=vertex_group_name)
    else:
        vertex_group = source_obj.vertex_groups[vertex_group_name]
    
    # ターゲットオブジェクトのBVHTreeを作成
    print("BVHTreeを構築中...")
    
    # ターゲットメッシュのワールド座標での頂点とポリゴンを取得
    target_verts = []
    target_polys = []
    
    # 評価されたメッシュを取得（モディファイアが適用された状態）
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_mesh_eval = target_eval.data
    
    # ワールド座標に変換
    target_matrix = target_obj.matrix_world
    
    for vert in target_mesh_eval.vertices:
        world_co = target_matrix @ vert.co
        target_verts.append(world_co)
    
    for poly in target_mesh_eval.polygons:
        target_polys.append(poly.vertices)
    
    # BVHTreeを構築
    bvh = BVHTree.FromPolygons(target_verts, target_polys)
    
    print("距離計算とウェイト設定中...")
    
    # ソースオブジェクトの各頂点について処理
    source_matrix = source_obj.matrix_world
    source_eval = source_obj.evaluated_get(depsgraph)
    source_mesh_eval = source_eval.data
    
    weights = []
    
    for i, vert in enumerate(source_mesh_eval.vertices):
        # 頂点のワールド座標を取得
        world_co = source_matrix @ vert.co
        
        # 最近接面までの距離を計算
        location, normal, index, distance = bvh.find_nearest(world_co)
        
        if location is None:
            print(f"警告: 頂点 {i} の最近接面が見つかりません")
            distance = max_distance
        
        # 距離に基づいてウェイトを計算
        if distance <= min_distance:
            weight = 1.0
        elif distance >= max_distance:
            weight = 0.0
        else:
            # 線形補間でウェイトを計算（max_distanceに近づくほど0に近づく）
            weight = 1.0 - ((distance - min_distance) / (max_distance - min_distance))
        
        weights.append(weight)
        
        # 頂点グループにウェイトを設定
        vertex_group.add([i], weight, 'REPLACE')
    
    print(f"完了: {len(weights)} 個の頂点にウェイトを設定しました")
    print(f"最小ウェイト: {min(weights):.4f}")
    print(f"最大ウェイト: {max(weights):.4f}")
    print(f"平均ウェイト: {np.mean(weights):.4f}")
    
    return True
