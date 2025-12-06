import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy
from algo_utils.find_connected_components import find_connected_components
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from math_utils.calculate_obb_from_points import calculate_obb_from_points
from math_utils.check_mesh_obb_intersection import check_mesh_obb_intersection


def group_components_by_weight_pattern(obj, base_avatar_data, clothing_armature):
    """
    同じウェイトパターンを持つ連結成分をグループ化する
    
    Parameters:
        obj: 処理対象のメッシュオブジェクト
        base_avatar_data: ベースアバターデータ
        
    Returns:
        dict: ウェイトパターンをキー、連結成分のリストを値とする辞書
    """
    # BMeshを作成
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    base_obj = bpy.data.objects.get("Body.BaseAvatar")
    if not base_obj:
        raise Exception("Base avatar mesh (Body.BaseAvatar) not found")

    # すべての連結成分を取得
    components = find_connected_components(obj)

    # 各コンポーネントの頂点数を表示
    # for j, comp in enumerate(components):
    #     print(f"Component {j}: {len(comp)} vertices")

    # チェック対象の頂点グループを取得
    target_groups = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)
    if clothing_armature:
        target_groups.update(bone.name for bone in clothing_armature.data.bones)
    
    # メッシュ内に存在する対象グループのみを抽出
    existing_target_groups = {vg.name for vg in obj.vertex_groups if vg.name in target_groups}
    
    # 各連結成分のウェイトパターンを計算
    component_patterns = {}
    uniform_components = []

    if "Rigid2" not in obj.vertex_groups:
        obj.vertex_groups.new(name="Rigid2")
    rigid_group = obj.vertex_groups["Rigid2"]
    
    for component in components:
        # コンポーネント内の各頂点のウェイトパターンを収集
        vertex_weights = []
        for vert_idx in component:
            vert = obj.data.vertices[vert_idx]
            weights = {group: 0.0 for group in existing_target_groups}
            
            for g in vert.groups:
                group_name = obj.vertex_groups[g.group].name
                if group_name in existing_target_groups:
                    weights[group_name] = g.weight
                    
            vertex_weights.append(weights)

        # 頂点ウェイトが空の場合は次のコンポーネントへスキップ
        if not vertex_weights:
            continue
        
        # チェック対象のすべてのグループで同じウェイトパターンかチェック
        is_uniform = True
        first_weights = vertex_weights[0]
        
        for weights in vertex_weights[1:]:
            for group_name in existing_target_groups:
                if abs(weights[group_name] - first_weights[group_name]) >= 0.0001:
                    is_uniform = False
                    break
            if not is_uniform:
                break

        # 一様なウェイトパターンを持つ連結成分のみを記録
        if is_uniform:
            # 評価済みメッシュから頂点座標を取得
            component_points = []
            for idx in component:
                if idx < len(bm.verts):
                    component_points.append(obj.matrix_world @ bm.verts[idx].co)
            
            # 素体メッシュとの交差をチェック
            if len(component_points) >= 3:
                # OBBを計算
                obb = calculate_obb_from_points(component_points)
                # OBBが計算できない場合はスキップ
                if obb is not None:
                    # 素体メッシュとの交差をチェック
                    if check_mesh_obb_intersection(base_obj, obb):
                        print(f"Component with {len(component)} vertices intersects with base mesh, excluding from rigid transfer")
                        continue

            uniform_components.append(component)
            
            # ウェイトパターンをハッシュ可能な形式に変換
            pattern_tuple = tuple(sorted((k, round(v, 4)) for k, v in first_weights.items() if v > 0))
            
            # pattern_tupleが空でない場合のみ処理を実行
            if pattern_tuple:
                # 一様なウェイトパターンを持つ連結成分の頂点すべてにRigid頂点グループのウェイトを1に設定
                for vert_idx in component:
                    rigid_group.add([vert_idx], 1.0, 'REPLACE')
                
                if pattern_tuple not in component_patterns:
                    component_patterns[pattern_tuple] = []
                component_patterns[pattern_tuple].append(component)
    
    # BMeshを解放
    bm.free()

    print(f"Found {len(components)} connected components in {obj.name}")
    print(f"Found {len(component_patterns)} uniform weight patterns in {obj.name}")

    # デバッグ用に各パターンの詳細を表示
    for i, (pattern, components_list) in enumerate(component_patterns.items()):
        total_vertices = sum(len(comp) for comp in components_list)
        print(f"Pattern {i}: {pattern}")
        print(f"  Components: {len(components_list)}, Total vertices: {total_vertices}")
        
        # 各コンポーネントの頂点数を表示
        for j, comp in enumerate(components_list):
            print(f"    Component {j}: {len(comp)} vertices")
    
    return component_patterns
