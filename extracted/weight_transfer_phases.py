import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import bmesh
import bpy
from algo_utils.mesh_topology_utils import create_vertex_neighbors_array
from algo_utils.vertex_group_utils import custom_max_vertex_group_numpy
from algo_utils.bone_group_utils import (
    get_humanoid_and_auxiliary_bone_groups,
)
from math_utils.geometry_utils import calculate_component_size
from math_utils.geometry_utils import calculate_obb_from_points
from algo_utils.component_utils import (
    cluster_components_by_adaptive_distance,
)
from mathutils import Vector

ComponentPattern = Tuple[Tuple[Tuple[str, float], ...], Tuple[Tuple[str, float], ...]]
OBBDataEntry = Dict[str, object]


@dataclass
class WeightTransferContext:
    target_obj: object
    armature: object
    base_avatar_data: Dict[str, object]
    clothing_avatar_data: Dict[str, object]
    field_path: str
    clothing_armature: object
    blend_shape_settings: List[Dict[str, object]]
    cloth_metadata: Dict[str, object] | None = None
    base_obj: object | None = None
    left_base_obj: object | None = None
    right_base_obj: object | None = None
    existing_target_groups: Set[str] = field(default_factory=set)
    original_vertex_weights: Dict[int, Dict[str, float]] | None = None
    component_patterns: Dict[ComponentPattern, List[List[int]]] | None = None
    obb_data: List[OBBDataEntry] | None = None
    neighbors_info: object | None = None
    offsets: object | None = None
    num_verts: int | None = None


def _apply_blend_shape_settings(ctx: WeightTransferContext):
    base_obj = ctx.base_obj
    left_base_obj = ctx.left_base_obj
    right_base_obj = ctx.right_base_obj
    if not base_obj or not base_obj.data.shape_keys:
        return
    for blend_shape_setting in ctx.blend_shape_settings:
        name = blend_shape_setting['name']
        value = blend_shape_setting['value']
        if name in base_obj.data.shape_keys.key_blocks:
            base_obj.data.shape_keys.key_blocks[name].value = value
            left_base_obj.data.shape_keys.key_blocks[name].value = value
            right_base_obj.data.shape_keys.key_blocks[name].value = value


def _get_existing_target_groups(ctx: WeightTransferContext):
    target_groups = get_humanoid_and_auxiliary_bone_groups(ctx.base_avatar_data)
    ctx.existing_target_groups = {vg.name for vg in ctx.target_obj.vertex_groups if vg.name in target_groups}
    return ctx.existing_target_groups


def _store_original_vertex_weights(ctx: WeightTransferContext):
    """
    頂点ごとの元のウェイトを保存する（最適化版）
    グループインデックス→グループ名のマップを事前構築して高速化
    """
    import time
    start_time = time.time()
    
    # グループインデックス→グループ名のマップを構築（対象グループのみ）
    target_group_indices = {}
    for vg in ctx.target_obj.vertex_groups:
        if vg.name in ctx.existing_target_groups:
            target_group_indices[vg.index] = vg.name
    
    original_vertex_weights = {}
    for vert in ctx.target_obj.data.vertices:
        weights = {}
        for g in vert.groups:
            group_name = target_group_indices.get(g.group)
            if group_name is not None and g.weight > 0.0001:
                weights[group_name] = g.weight
        original_vertex_weights[vert.index] = weights
    
    ctx.original_vertex_weights = original_vertex_weights
    return original_vertex_weights


def _normalize_component_patterns(ctx: WeightTransferContext, component_patterns):
    import time
    start_time = time.time()
    new_component_patterns = {}

    for pattern, components in component_patterns.items():
        if not any(group in ctx.existing_target_groups for group in pattern):
            all_deform_groups = set(ctx.existing_target_groups)
            if ctx.clothing_armature:
                all_deform_groups.update(bone.name for bone in ctx.clothing_armature.data.bones)

            non_humanoid_difference_group = ctx.target_obj.vertex_groups.get("NonHumanoidDifference")
            is_non_humanoid_difference_group = False
            max_weight = 0.0
            if non_humanoid_difference_group:
                for component in components:
                    for vert_idx in component:
                        vert = ctx.target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group == non_humanoid_difference_group.index and g.weight > 0.0001:
                                is_non_humanoid_difference_group = True
                                if g.weight > max_weight:
                                    max_weight = g.weight
            if is_non_humanoid_difference_group:
                max_avg_pattern = {}
                count = 0
                for component in components:
                    for vert_idx in component:
                        vert = ctx.target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group == non_humanoid_difference_group.index and g.weight == max_weight:
                                for g2 in vert.groups:
                                    if ctx.target_obj.vertex_groups[g2.group].name in all_deform_groups:
                                        if g2.group not in max_avg_pattern:
                                            max_avg_pattern[g2.group] = g2.weight
                                        else:
                                            max_avg_pattern[g2.group] += g2.weight
                                count += 1
                                break
                if count > 0:
                    for group_name, weight in max_avg_pattern.items():
                        max_avg_pattern[group_name] = weight / count
                for component in components:
                    for vert_idx in component:
                        vert = ctx.target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group not in max_avg_pattern and ctx.target_obj.vertex_groups[g.group].name in all_deform_groups:
                                g.weight = 0.0
                        for max_group_id, max_weight in max_avg_pattern.items():
                            group = ctx.target_obj.vertex_groups[max_group_id]
                            group.add([vert_idx], max_weight, 'REPLACE')
            continue

        original_pattern_dict = {group_name: weight for group_name, weight in pattern}
        original_pattern = tuple(sorted((k, v) for k, v in original_pattern_dict.items() if k in ctx.existing_target_groups))

        all_weights = {group: [] for group in ctx.existing_target_groups}
        all_vertices = set()

        for component in components:
            for vert_idx in component:
                all_vertices.add(vert_idx)
                vert = ctx.target_obj.data.vertices[vert_idx]
                for group_name in ctx.existing_target_groups:
                    weight = 0.0
                    for g in vert.groups:
                        if ctx.target_obj.vertex_groups[g.group].name == group_name:
                            weight = g.weight
                            break
                    all_weights[group_name].append(weight)

        avg_weights = {}
        for group_name, weights in all_weights.items():
            avg_weights[group_name] = sum(weights) / len(weights) if weights else 0.0

        for vert_idx in all_vertices:
            for group_name, avg_weight in avg_weights.items():
                group = ctx.target_obj.vertex_groups[group_name]
                if avg_weight > 0.0001:
                    group.add([vert_idx], avg_weight, 'REPLACE')
                else:
                    group.add([vert_idx], 0.0, 'REPLACE')

        new_pattern = tuple(sorted((k, round(v, 4)) for k, v in avg_weights.items() if v > 0.0001))
        new_component_patterns[(new_pattern, original_pattern)] = components

    ctx.component_patterns = new_component_patterns
    return new_component_patterns


def _collect_obb_data(ctx: WeightTransferContext):
    import time
    obb_data = []

    start_time = time.time()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    ctx.target_obj.select_set(True)
    bpy.context.view_layer.objects.active = ctx.target_obj

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = ctx.target_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data

    if len(eval_mesh.vertices) == 0:
        return obb_data

    all_rigid_component_vertices = set()
    for (_, _), components in ctx.component_patterns.items():
        for component in components:
            all_rigid_component_vertices.update(component)

    component_count = 0
    for (new_pattern, original_pattern), components in ctx.component_patterns.items():
        pattern_weights = {group_name: weight for group_name, weight in new_pattern}
        original_pattern_weights = {group_name: weight for group_name, weight in original_pattern}

        component_coords = {}
        component_sizes = {}

        for component_idx, component in enumerate(components):
            coords = []
            for vert_idx in component:
                if vert_idx < len(eval_mesh.vertices):
                    coords.append(eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co)
            if coords:
                component_coords[component_idx] = coords
                size = calculate_component_size(coords)
                component_sizes[component_idx] = size

        if not component_coords:
            continue

        clusters = cluster_components_by_adaptive_distance(component_coords, component_sizes)

        for cluster_idx, cluster in enumerate(clusters):
            cluster_vertices = set()
            cluster_coords = []

            for comp_idx in cluster:
                for vert_idx in components[comp_idx]:
                    cluster_vertices.add(vert_idx)
                    if vert_idx < len(eval_mesh.vertices):
                        cluster_coords.append(eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co)

            if len(cluster_coords) < 3:
                continue

            obb = calculate_obb_from_points(cluster_coords)
            if obb is None:
                print(f"[Warning] OBB calculation failed for pattern {new_pattern}, cluster {cluster_idx}. Skipping.")
                continue

            obb['radii'] = [radius * 1.3 for radius in obb['radii']]

            vertices_in_obb = []
            for vert_idx, vert in enumerate(ctx.target_obj.data.vertices):
                if vert_idx in all_rigid_component_vertices or vert_idx >= len(eval_mesh.vertices):
                    continue
                try:
                    vert_world = eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co
                    relative_pos = vert_world - Vector(obb['center'])
                    projections = [abs(relative_pos.dot(Vector(obb['axes'][:, i]))) for i in range(3)]
                    if all(proj <= radius for proj, radius in zip(projections, obb['radii'])):
                        vertices_in_obb.append(vert_idx)
                except Exception as e:
                    continue

            if not vertices_in_obb:
                continue

            obb_data.append({
                'component_vertices': cluster_vertices,
                'vertices_in_obb': vertices_in_obb,
                'component_id': component_count,
                'pattern_weights': pattern_weights,
                'original_pattern_weights': original_pattern_weights
            })
            component_count += 1

    ctx.obb_data = obb_data
    return obb_data


def _process_obb_groups(ctx: WeightTransferContext):
    """
    OBB内の頂点を処理してConnected_*グループを作成する
    
    処理内容:
    1. OBB内の頂点を選択
    2. (オプション) 閉じたエッジループを検出して選択を制限
    3. select_moreで選択を拡張
    4. Connected_*グループにウェイトを設定
    5. スムージングと減衰処理
    
    注: エッジループ検出は計算コストが高いため、
    USE_EDGE_LOOP_DETECTION=Falseでスキップ可能
    """
    import time
    start_time = time.time()
    
    # エッジループ検出を使用するかどうか（Falseで大幅な高速化）
    USE_EDGE_LOOP_DETECTION = False

    neighbors_info, offsets, num_verts = create_vertex_neighbors_array(ctx.target_obj, expand_distance=0.02, sigma=0.00659)
    ctx.neighbors_info = neighbors_info
    ctx.offsets = offsets
    ctx.num_verts = num_verts
    
    bpy.ops.object.mode_set(mode='EDIT')

    for obb_idx, data in enumerate(ctx.obb_data):
        connected_group = ctx.target_obj.vertex_groups.new(name=f"Connected_{data['component_id']}")
        bpy.ops.mesh.select_all(action='DESELECT')
        bm = bmesh.from_edit_mesh(ctx.target_obj.data)
        bm.verts.ensure_lookup_table()

        for vert_idx in data['vertices_in_obb']:
            if vert_idx < len(bm.verts):
                bm.verts[vert_idx].select = True
        bmesh.update_edit_mesh(ctx.target_obj.data)
        initial_selection = {v.index for v in bm.verts if v.select}

        if initial_selection and USE_EDGE_LOOP_DETECTION:
            # エッジループ検出（計算コストが高い）
            complete_loops = _detect_closed_edge_loops_bmesh(
                bm, initial_selection, data['original_pattern_weights'], ctx.original_vertex_weights
            )
            
            if complete_loops:
                bpy.ops.mesh.select_all(action='DESELECT')
                bm = bmesh.from_edit_mesh(ctx.target_obj.data)
                bm.verts.ensure_lookup_table()
                for vert in bm.verts:
                    if vert.index in complete_loops:
                        vert.select = True
                bmesh.update_edit_mesh(ctx.target_obj.data)

        for _ in range(1):
            bpy.ops.mesh.select_more()
        bm = bmesh.from_edit_mesh(ctx.target_obj.data)
        selected_verts = [v.index for v in bm.verts if v.select]
        if len(selected_verts) == 0:
            continue

        bpy.ops.object.mode_set(mode='OBJECT')
        for vert_idx in selected_verts:
            if vert_idx not in data['component_vertices']:
                connected_group.add([vert_idx], 1.0, 'REPLACE')
        bpy.ops.object.select_all(action='DESELECT')
        ctx.target_obj.select_set(True)
        bpy.context.view_layer.objects.active = ctx.target_obj

        for i, group in enumerate(ctx.target_obj.vertex_groups):
            ctx.target_obj.vertex_groups.active_index = i
            if group.name == f"Connected_{data['component_id']}":
                break

        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=3, expand=0.5)
        custom_max_vertex_group_numpy(ctx.target_obj, f"Connected_{data['component_id']}", ctx.neighbors_info, ctx.offsets, ctx.num_verts, repeat=3, weight_factor=1.0)
        bpy.ops.object.mode_set(mode='OBJECT')
        connected_group = ctx.target_obj.vertex_groups[f"Connected_{data['component_id']}"]
        original_pattern_weights = data['original_pattern_weights']

        for vert_idx, vert in enumerate(ctx.target_obj.data.vertices):
            if vert_idx in data['component_vertices']:
                connected_group.add([vert_idx], 0.0, 'REPLACE')
                continue
            if vert_idx in ctx.original_vertex_weights:
                orig_weights = ctx.original_vertex_weights[vert_idx]
                similarity_score = 0.0
                total_weight = 0.0

                for group_name, pattern_weight in original_pattern_weights.items():
                    orig_weight = orig_weights.get(group_name, 0.0)
                    diff = abs(pattern_weight - orig_weight)
                    similarity_score += diff
                    total_weight += pattern_weight

                if total_weight > 0:
                    normalized_score = similarity_score / total_weight
                    decay_factor = 1.0 - min(normalized_score * 3.33333, 1.0)

                    connected_weight = 0.0
                    for g in ctx.target_obj.data.vertices[vert_idx].groups:
                        if g.group == connected_group.index:
                            connected_weight = g.weight
                            break

                    if normalized_score > 0.3:
                        connected_group.add([vert_idx], 0.0, 'REPLACE')
                    else:
                        connected_group.add([vert_idx], connected_weight * decay_factor, 'REPLACE')
                else:
                    connected_group.add([vert_idx], 0.0, 'REPLACE')
        if obb_idx < len(ctx.obb_data) - 1:
            bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.object.mode_set(mode='OBJECT')


def _detect_closed_edge_loops_bmesh(bm, initial_selection, pattern_weights, original_vertex_weights):
    """
    BMeshを使用して閉じたエッジループを検出する（bpy.ops不使用版）
    
    Parameters:
        bm: BMeshオブジェクト
        initial_selection: 初期選択頂点のセット
        pattern_weights: パターンウェイト辞書
        original_vertex_weights: 元の頂点ウェイト辞書
    
    Returns:
        set: 閉じたループに属する頂点インデックスのセット
    """
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    # 両端が選択されているエッジを収集
    selected_edges = [e for e in bm.edges if all(v.index in initial_selection for v in e.verts)]
    
    if not selected_edges:
        return set()
    
    complete_loops = set()
    visited_edges = set()
    
    for start_edge in selected_edges:
        if start_edge.index in visited_edges:
            continue
        
        # このエッジからループをトレース
        loop_verts, loop_edges, is_closed = _trace_edge_loop(bm, start_edge, initial_selection)
        
        visited_edges.update(e.index for e in loop_edges)
        
        if not is_closed:
            continue
        
        # 閉じたループの品質チェック（quad mesh前提）
        is_valid_loop = True
        for v_idx in loop_verts:
            v = bm.verts[v_idx]
            # ループ内エッジ数と総エッジ数をチェック
            loop_edge_count = sum(1 for e in v.link_edges if e in loop_edges)
            if loop_edge_count != 2 or len(v.link_edges) != 4:
                is_valid_loop = False
                break
        
        if not is_valid_loop:
            continue
        
        # ウェイトパターンの類似性チェック
        is_similar = _check_pattern_similarity(loop_verts, pattern_weights, original_vertex_weights)
        
        if is_similar:
            complete_loops.update(loop_verts)
    
    return complete_loops


def _trace_edge_loop(bm, start_edge, allowed_verts):
    """
    エッジからループをトレースする
    
    Returns:
        (loop_verts, loop_edges, is_closed)
    """
    loop_verts = set()
    loop_edges = [start_edge]
    
    for v in start_edge.verts:
        loop_verts.add(v.index)
    
    # 両方向にトレース
    for direction in [0, 1]:
        current_edge = start_edge
        current_vert = start_edge.verts[direction]
        
        max_iterations = 10000  # 無限ループ防止
        for _ in range(max_iterations):
            # 次のエッジを探す（同一面を共有し、allowed_verts内）
            next_edge = None
            for link_edge in current_vert.link_edges:
                if link_edge == current_edge:
                    continue
                if link_edge in loop_edges:
                    # ループが閉じた
                    return loop_verts, loop_edges, True
                
                other_vert = link_edge.other_vert(current_vert)
                if other_vert.index not in allowed_verts:
                    continue
                
                # 同一面を共有するエッジを優先（ループ継続）
                shared_faces = set(current_edge.link_faces) & set(link_edge.link_faces)
                if shared_faces:
                    next_edge = link_edge
                    break
            
            if next_edge is None:
                break
            
            loop_edges.append(next_edge)
            current_edge = next_edge
            current_vert = next_edge.other_vert(current_vert)
            loop_verts.add(current_vert.index)
    
    return loop_verts, loop_edges, False


def _check_pattern_similarity(loop_verts, pattern_weights, original_vertex_weights, threshold=0.05):
    """
    ループ頂点のウェイトパターンが元のパターンと類似しているかチェック
    """
    for vert_idx in loop_verts:
        if vert_idx not in original_vertex_weights:
            continue
        
        orig_weights = original_vertex_weights[vert_idx]
        similarity_score = 0.0
        total_weight = 0.0
        
        for group_name, pattern_weight in pattern_weights.items():
            orig_weight = orig_weights.get(group_name, 0.0)
            diff = abs(pattern_weight - orig_weight)
            similarity_score += diff
            total_weight += pattern_weight
        
        if total_weight > 0:
            normalized_score = similarity_score / total_weight
            if normalized_score > threshold:
                return False
    
    return True

def _synthesize_weights(ctx: WeightTransferContext):
    """
    Connected_* グループのウェイトを合成する（最適化版）
    事前にデータ構造を構築して検索を高速化
    """
    import time
    start_time = time.time()
    connected_groups = [vg for vg in ctx.target_obj.vertex_groups if vg.name.startswith("Connected_")]

    if not connected_groups:
        return

    # スキップ対象の頂点インデックスを事前にセット化
    skip_vertices = set()
    for (_, _), components in ctx.component_patterns.items():
        for component in components:
            skip_vertices.update(component)
    
    # Connected グループのインデックス→グループ情報のマップ
    connected_group_map = {}  # group_index -> (component_id, group_obj)
    for vg in connected_groups:
        component_id = int(vg.name.split('_')[1])
        connected_group_map[vg.index] = (component_id, vg)
    
    # component_id → pattern_weights のマップを構築
    component_pattern_map = {}  # component_id -> pattern_tuple
    for data in ctx.obb_data:
        pattern_tuple = tuple(sorted((k, v) for k, v in data['pattern_weights'].items() if v > 0.0001))
        component_pattern_map[data['component_id']] = pattern_tuple
    
    # 対象グループのインデックス→名前マップ
    target_group_indices = {}
    target_group_objects = {}
    for vg in ctx.target_obj.vertex_groups:
        if vg.name in ctx.existing_target_groups:
            target_group_indices[vg.index] = vg.name
            target_group_objects[vg.name] = vg

    for vert in ctx.target_obj.data.vertices:
        if vert.index in skip_vertices:
            continue

        connected_weights = {}
        total_weight = 0.0

        # 頂点のグループを一度走査して必要な情報を収集
        vert_group_weights = {}  # group_index -> weight
        for g in vert.groups:
            vert_group_weights[g.group] = g.weight

        for group_idx, (component_id, _) in connected_group_map.items():
            weight = vert_group_weights.get(group_idx, 0.0)
            if weight > 0:
                pattern_tuple = component_pattern_map.get(component_id)
                if pattern_tuple:
                    connected_weights[pattern_tuple] = weight
                    total_weight += weight

        if total_weight <= 0:
            continue

        factor = min(total_weight, 1.0)
        
        # 既存ウェイトを収集
        existing_weights = {}
        for group_idx, group_name in target_group_indices.items():
            existing_weights[group_name] = vert_group_weights.get(group_idx, 0.0)

        # 新しいウェイトを計算
        new_weights = {}
        for group_name, weight in existing_weights.items():
            new_weights[group_name] = weight * (1.0 - factor)

        for pattern, weight in connected_weights.items():
            normalized_weight = weight / total_weight
            if total_weight < 1.0:
                normalized_weight = weight
            for group_name, value in pattern:
                if group_name in target_group_objects:
                    component_weight = value * normalized_weight
                    new_weights[group_name] = new_weights.get(group_name, 0.0) + component_weight

        # ウェイトを適用
        for group_name, weight in new_weights.items():
            if weight > 1.0:
                weight = 1.0
            group = target_group_objects.get(group_name)
            if group:
                group.add([vert.index], weight, 'REPLACE')

