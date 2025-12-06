import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Optional

import bmesh
import bpy


def propagate_bone_weights(mesh_obj: bpy.types.Object, temp_group_name: str = "PropagatedWeightsTemp", max_iterations: int = 500) -> Optional[str]:
    """
    ボーン変形に関わるボーンウェイトを持たない頂点にウェイトを伝播させる。
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        max_iterations: 最大反復回数
        
    Returns:
        Optional[str]: 伝播させた頂点を記録した頂点グループの名前。伝播が不要な場合はNone
    """
    # アーマチュアモディファイアからアーマチュアを取得
    armature_obj = None
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE':
            armature_obj = modifier.object
            break
    
    if not armature_obj:
        print(f"Warning: No armature modifier found in {mesh_obj.name}")
        return None
    
    # アーマチュアのすべてのボーン名を取得
    deform_groups = {bone.name for bone in armature_obj.data.bones}
    
    # BMeshを作成
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    # 頂点ごとのウェイト情報を取得
    vertex_weights = {}
    vertices_without_weights = set()
    
    for vert in mesh_obj.data.vertices:
        has_weight = False
        weights = {}
        
        for group in mesh_obj.vertex_groups:
            if group.name in deform_groups:
                try:
                    weight = 0.0
                    for g in vert.groups:
                        if g.group == group.index:
                            weight = g.weight
                            has_weight = True
                            break
                    if weight > 0:
                        weights[group.name] = weight
                except RuntimeError:
                    continue
                    
        vertex_weights[vert.index] = weights
        if not weights:
            vertices_without_weights.add(vert.index)
    
    # ウェイトを持たない頂点がない場合は処理を終了
    if not vertices_without_weights:
        return None
    
    print(f"Found {len(vertices_without_weights)} vertices without weights in {mesh_obj.name}")
    
    # 一時的な頂点グループを作成（既存の同名グループがあれば削除）
    if temp_group_name in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[temp_group_name])
    temp_group = mesh_obj.vertex_groups.new(name=temp_group_name)
    
    # 反復処理
    total_propagated = 0
    iteration = 0
    while iteration < max_iterations and vertices_without_weights:
        propagated_this_iteration = 0
        remaining_vertices = set()
        
        # 各ウェイトなし頂点について処理
        for vert_idx in vertices_without_weights:
            vert = bm.verts[vert_idx]
            # 隣接頂点を取得
            neighbors = set()
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if vertex_weights[other.index]:
                    neighbors.add(other)
            
            if neighbors:
                # 最も近い頂点を見つける
                closest_vert = min(neighbors, 
                                 key=lambda v: (v.co - vert.co).length)
                
                # ウェイトをコピー
                vertex_weights[vert_idx] = vertex_weights[closest_vert.index].copy()
                temp_group.add([vert_idx], 1.0, 'REPLACE')  # 伝播頂点を記録
                propagated_this_iteration += 1
            else:
                remaining_vertices.add(vert_idx)
        
        if propagated_this_iteration == 0:
            break
        
        print(f"Iteration {iteration + 1}: Propagated weights to {propagated_this_iteration} vertices in {mesh_obj.name}")
        total_propagated += propagated_this_iteration
        vertices_without_weights = remaining_vertices
        iteration += 1
    
    # 残りのウェイトなし頂点に平均ウェイトを割り当て
    if vertices_without_weights:
        total_weights = {}
        weight_count = 0
        
        # まず平均ウェイトを計算
        for vert_idx, weights in vertex_weights.items():
            if weights:
                weight_count += 1
                for group_name, weight in weights.items():
                    if group_name not in total_weights:
                        total_weights[group_name] = 0.0
                    total_weights[group_name] += weight
        
        if weight_count > 0:
            average_weights = {
                group_name: weight / weight_count
                for group_name, weight in total_weights.items()
            }
            
            # 残りの頂点に平均ウェイトを適用
            num_averaged = len(vertices_without_weights)
            print(f"Applying average weights to remaining {num_averaged} vertices in {mesh_obj.name}")
            
            for vert_idx in vertices_without_weights:
                vertex_weights[vert_idx] = average_weights.copy()
                temp_group.add([vert_idx], 1.0, 'REPLACE')  # 伝播頂点を記録
            total_propagated += num_averaged
    
    # 新しいウェイトを適用
    for vert_idx, weights in vertex_weights.items():
        for group_name, weight in weights.items():
            if group_name in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups[group_name].add([vert_idx], weight, 'REPLACE')
    
    print(f"Total: Propagated weights to {total_propagated} vertices in {mesh_obj.name}")
    
    bm.free()
    return temp_group_name
