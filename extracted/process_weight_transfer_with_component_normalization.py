import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy
from algo_utils.create_vertex_neighbors_array import create_vertex_neighbors_array
from algo_utils.custom_max_vertex_group_numpy import custom_max_vertex_group_numpy
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from algo_utils.group_components_by_weight_pattern import (
    group_components_by_weight_pattern,
)
from math_utils.calculate_component_size import calculate_component_size
from math_utils.calculate_obb_from_points import calculate_obb_from_points
from math_utils.cluster_components_by_adaptive_distance import (
    cluster_components_by_adaptive_distance,
)
from mathutils import Vector
from process_weight_transfer import process_weight_transfer


def process_weight_transfer_with_component_normalization(target_obj, armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, blend_shape_settings, cloth_metadata=None):
    """
    ウェイト転送処理を行い、連結成分ごとにウェイトを正規化する
    
    Parameters:
        target_obj: 処理対象のメッシュオブジェクト
        armature: アーマチュアオブジェクト
        base_avatar_data: ベースアバターデータ
        clothing_avatar_data: 衣装アバターデータ
        field_path: フィールドパス
        clothing_armature: 衣装のアーマチュア
        cloth_metadata: クロスメタデータ
    """
    import time
    start_total = time.time()
    
    print(f"process_weight_transfer_with_component_normalization 処理開始: {target_obj.name}")
    
    # humanoid_to_boneマッピングを作成
    humanoid_to_bone = {}
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map and "humanoidBoneName" in bone_map:
            humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    
    # 素体メッシュを取得
    start_time = time.time()
    base_obj = bpy.data.objects.get("Body.BaseAvatar")
    if not base_obj:
        raise Exception("Base avatar mesh (Body.BaseAvatar) not found")
    
    left_base_obj = bpy.data.objects["Body.BaseAvatar.LeftOnly"]
    right_base_obj = bpy.data.objects["Body.BaseAvatar.RightOnly"]

    print(f"Set blend_shape_settings: {blend_shape_settings}")
    if base_obj.data.shape_keys:
        for blend_shape_setting in blend_shape_settings:
            if blend_shape_setting['name'] in base_obj.data.shape_keys.key_blocks:
                base_obj.data.shape_keys.key_blocks[blend_shape_setting['name']].value = blend_shape_setting['value']
                left_base_obj.data.shape_keys.key_blocks[blend_shape_setting['name']].value = blend_shape_setting['value']
                right_base_obj.data.shape_keys.key_blocks[blend_shape_setting['name']].value = blend_shape_setting['value']
                print(f"Set {blend_shape_setting['name']} to {blend_shape_setting['value']}")
    
    # 評価済みのメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_target_obj = target_obj.evaluated_get(depsgraph)
    eval_mesh = eval_target_obj.data
    
    # チェック対象の頂点グループを取得
    target_groups = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)
    
    # メッシュ内に存在する対象グループのみを抽出
    existing_target_groups = {vg.name for vg in target_obj.vertex_groups if vg.name in target_groups}
    print(f"準備時間: {time.time() - start_time:.2f}秒")
    
    # 処理前に同じウェイトパターンを持つ連結成分をグループ化
    start_time = time.time()
    component_patterns = group_components_by_weight_pattern(target_obj, base_avatar_data, clothing_armature)
    print(f"コンポーネントパターン抽出時間: {time.time() - start_time:.2f}秒")
    
    # 処理前の各頂点のウェイトパターンを保存
    start_time = time.time()
    original_vertex_weights = {}
    for vert_idx, vert in enumerate(target_obj.data.vertices):
        weights = {}
        for group_name in existing_target_groups:
            weight = 0.0
            for g in vert.groups:
                if target_obj.vertex_groups[g.group].name == group_name:
                    weight = g.weight
                    break
            if weight > 0.0001:
                weights[group_name] = weight
        original_vertex_weights[vert_idx] = weights
    print(f"元のウェイト保存時間: {time.time() - start_time:.2f}秒")
    
    # 通常のウェイト転送処理を実行
    start_time = time.time()
    process_weight_transfer(target_obj, armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, cloth_metadata)
    print(f"通常ウェイト転送処理時間: {time.time() - start_time:.2f}秒")

    start_time = time.time()
    new_component_patterns = {}
    
    # 各パターンのグループに対して処理
    for pattern, components in component_patterns.items():
        
        # patternにexisting_target_groupsに含まれないグループしかない場合
        if not any(group in existing_target_groups for group in pattern):
            all_deform_groups = set(existing_target_groups)
            if clothing_armature:
                all_deform_groups.update(bone.name for bone in clothing_armature.data.bones)
            # NonHumanoidDifferenceグループのウェイトが存在するかチェックしつつ、そのウェイトが最大となる頂点を取得
            non_humanoid_difference_group = target_obj.vertex_groups.get("NonHumanoidDifference")
            is_non_humanoid_difference_group = False
            max_weight = 0.0
            if non_humanoid_difference_group:
                for component in components:
                    for vert_idx in component:
                        vert = target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group == non_humanoid_difference_group.index and g.weight > 0.0001:
                                is_non_humanoid_difference_group = True
                                if g.weight > max_weight:
                                    max_weight = g.weight
            # NonHumanoidDifferenceグループのウェイトが存在する場合、そのウェイトが最大となる頂点のウェイトパターンの平均ウェイトを他のすべての頂点に適用
            if is_non_humanoid_difference_group:
                max_avg_pattern = {}
                count = 0
                for component in components:
                    for vert_idx in component:
                        vert = target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group == non_humanoid_difference_group.index and g.weight == max_weight:
                                for g2 in vert.groups:
                                    if target_obj.vertex_groups[g2.group].name in all_deform_groups:
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
                        vert = target_obj.data.vertices[vert_idx]
                        for g in vert.groups:
                            if g.group not in max_avg_pattern and target_obj.vertex_groups[g.group].name in all_deform_groups:
                                g.weight = 0.0
                        for max_group_id, max_weight in max_avg_pattern.items():
                            group = target_obj.vertex_groups[max_group_id]
                            group.add([vert_idx], max_weight, 'REPLACE')
            continue

        # patternからexisting_target_groupsに含まれるグループのみを抽出
        original_pattern_dict = {}
        for group_name, weight in pattern:
            original_pattern_dict[group_name] = weight
        original_pattern = tuple(sorted((k, v) for k, v in original_pattern_dict.items() if k in existing_target_groups))
        
        # 各グループ内のすべての頂点のウェイトを収集
        all_weights = {group: [] for group in existing_target_groups}
        all_vertices = set()
        
        for component in components:
            for vert_idx in component:
                all_vertices.add(vert_idx)
                vert = target_obj.data.vertices[vert_idx]
                
                for group_name in existing_target_groups:
                    weight = 0.0
                    for g in vert.groups:
                        if target_obj.vertex_groups[g.group].name == group_name:
                            weight = g.weight
                            break
                    all_weights[group_name].append(weight)
        
        # 各グループの平均ウェイトを計算
        avg_weights = {}
        for group_name, weights in all_weights.items():
            if weights:
                avg_weights[group_name] = sum(weights) / len(weights)
            else:
                avg_weights[group_name] = 0.0
        
        # すべての頂点に平均ウェイトを適用
        for vert_idx in all_vertices:
            for group_name, avg_weight in avg_weights.items():
                group = target_obj.vertex_groups[group_name]
                if avg_weight > 0.0001:
                    group.add([vert_idx], avg_weight, 'REPLACE')
                else:
                    group.add([vert_idx], 0.0, 'REPLACE')
        
        # component_patternsのpatternを更新
        new_pattern = tuple(sorted((k, round(v, 4)) for k, v in avg_weights.items() if v > 0.0001))
        new_component_patterns[(new_pattern, original_pattern)] = components
    
    component_patterns = new_component_patterns
    print(f"コンポーネントパターン正規化時間: {time.time() - start_time:.2f}秒")

    # コンポーネントパターンに含まれる頂点のOBBを計算し、周辺の頂点に影響を与える処理
    if component_patterns:
        # OBBデータ収集
        start_time = time.time()
        # オブジェクトモードで評価済みのメッシュを取得
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = target_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.data
        
        # 安全チェック：評価済みメッシュが空でないことを確認
        if len(eval_mesh.vertices) == 0:
            print(f"警告: {target_obj.name} の評価済みメッシュに頂点がありません。OBB計算をスキップします。")
            return
        
        # EDITモードに入る前に必要なデータを収集
        obb_data = []

        all_rigid_component_vertices = set()
        for (new_pattern, original_pattern), components in component_patterns.items():
            # コンポーネント内のすべての頂点を収集
            for component in components:
                all_rigid_component_vertices.update(component)
        
        component_count = 0
        # 各パターンのコンポーネントに対して処理
        for (new_pattern, original_pattern), components in component_patterns.items():
            # 新しいパターンのウェイト情報を辞書に変換
            pattern_weights = {}
            for group_name, weight in new_pattern:
                pattern_weights[group_name] = weight
                
            # オリジナルパターンのウェイト情報を辞書に変換
            original_pattern_weights = {}
            for group_name, weight in original_pattern:
                original_pattern_weights[group_name] = weight
                
            # 同じパターンを持つすべてのコンポーネントの頂点を収集
            all_component_vertices = set()
            for component in components:
                all_component_vertices.update(component)
            
            # 各コンポーネントの頂点座標とサイズ情報を取得
            component_coords = {}
            component_sizes = {}
            
            for component_idx, component in enumerate(components):
                coords = []
                for vert_idx in component:
                    if vert_idx < len(eval_mesh.vertices):
                        coords.append(eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co)
                
                if coords:
                    component_coords[component_idx] = coords
                    
                    # コンポーネントのサイズを計算（最大距離またはバウンディングボックスのサイズ）
                    size = calculate_component_size(coords)
                    component_sizes[component_idx] = size
            
            # 空のコンポーネントをスキップ
            if not component_coords:
                continue
            
            # コンポーネント間の距離に基づいてクラスタリング
            # サイズに基づいて適応的に閾値を決定
            clusters = cluster_components_by_adaptive_distance(component_coords, component_sizes)
            
            # 各クラスターに対してOBBを計算
            for cluster_idx, cluster in enumerate(clusters):
                # クラスター内のすべての頂点座標を収集
                cluster_vertices = set()
                cluster_coords = []
                
                for comp_idx in cluster:
                    for vert_idx in components[comp_idx]:
                        cluster_vertices.add(vert_idx)
                        if vert_idx < len(eval_mesh.vertices):
                            cluster_coords.append(eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co)
                
                # 頂点が少なすぎる場合はスキップ
                if len(cluster_coords) < 3:
                    print(f"警告: パターン {pattern} のクラスター {cluster_idx} の有効な頂点が少なすぎます（{len(cluster_coords)}点）。スキップします。")
                    continue
                
                # OBBを計算
                obb = calculate_obb_from_points(cluster_coords)
                
                # OBB計算が失敗した場合はスキップ
                if obb is None:
                    print(f"警告: パターン {pattern} のクラスター {cluster_idx} のOBB計算に失敗しました。スキップします。")
                    continue
                
                # OBBを20%膨張
                obb['radii'] = [radius * 1.3 for radius in obb['radii']]
                
                # 頂点選択用のデータを保存
                vertices_in_obb = []
                for vert_idx, vert in enumerate(target_obj.data.vertices):
                    if vert_idx not in all_rigid_component_vertices and vert_idx < len(eval_mesh.vertices):
                        try:
                            # 評価済みの頂点のワールド座標
                            vert_world = eval_obj.matrix_world @ eval_mesh.vertices[vert_idx].co
                            
                            # OBBの中心からの相対位置
                            relative_pos = vert_world - Vector(obb['center'])
                            
                            # OBBの各軸に沿った投影
                            projections = [abs(relative_pos.dot(Vector(obb['axes'][:, i]))) for i in range(3)]
                            
                            # すべての軸で投影が半径以内ならOBB内
                            if all(proj <= radius for proj, radius in zip(projections, obb['radii'])):
                                vertices_in_obb.append(vert_idx)
                        except Exception as e:
                            print(f"警告: 頂点 {vert_idx} のOBBチェック中にエラーが発生しました: {e}")
                            continue
                
                if not vertices_in_obb:
                    print(f"警告: パターン {pattern} のクラスター {cluster_idx} のOBB内に頂点が見つかりませんでした。スキップします。")
                    continue

                obb_data.append({
                    'component_vertices': cluster_vertices,
                    'vertices_in_obb': vertices_in_obb,
                    'component_id': component_count,
                    'pattern_weights': pattern_weights,
                    'original_pattern_weights': original_pattern_weights
                })

                component_count += 1
        print(f"OBBデータ収集時間: {time.time() - start_time:.2f}秒")
        
        # OBBデータがない場合は処理をスキップ
        if not obb_data:
            print("警告: 有効なOBBデータがありません。処理をスキップします。")
            return
        
        start_time = time.time()
        #vert_neighbors = create_vertex_neighbors_list(target_obj, expand_distance=0.04, sigma=0.02)
        neighbors_info, offsets, num_verts = create_vertex_neighbors_array(target_obj, expand_distance=0.02, sigma=0.00659)
        print(f"頂点近傍リスト作成時間: {time.time() - start_time:.2f}秒")

        # OBB処理開始
        start_time = time.time()
        # 編集モードに入る
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 各OBBデータに対して処理
        for obb_idx, data in enumerate(obb_data):
            obb_start = time.time()
            # "Connected"頂点グループを作成または取得
            connected_group = target_obj.vertex_groups.new(name=f"Connected_{data['component_id']}")
            print(f"    Connected頂点グループ作成: {connected_group.name}")
            
            # すべての選択を解除
            bpy.ops.mesh.select_all(action='DESELECT')
            
            # BMeshを使用して頂点を選択
            bm = bmesh.from_edit_mesh(target_obj.data)
            bm.verts.ensure_lookup_table()
            
            # OBB内の頂点を選択
            obb_vertex_select_start = time.time()
            for vert_idx in data['vertices_in_obb']:
                if vert_idx < len(bm.verts):
                    bm.verts[vert_idx].select = True
            
            # BMeshの変更をメッシュに反映
            bmesh.update_edit_mesh(target_obj.data)
            print(f"    OBB内頂点選択時間: {time.time() - obb_vertex_select_start:.2f}秒")
            
            # 選択された頂点に含まれるエッジループを検出
            # 現在の選択を保存
            edge_loop_start = time.time()
            initial_selection = {v.index for v in bm.verts if v.select}
            
            if initial_selection:
                # 選択された頂点から構成されるエッジを取得
                selected_edges = [e for e in bm.edges if all(v.select for v in e.verts)]
                
                # 完全に含まれる閉じたエッジループを記録
                complete_loops = set()
                
                # 各エッジに対してループ選択を実行
                edge_count = len(selected_edges)
                print(f"    処理対象エッジ数: {edge_count}")
                
                for edge_idx, edge in enumerate(selected_edges):
                    if edge_idx % 100 == 0 and edge_idx > 0:
                        print(f"    エッジ処理進捗: {edge_idx}/{edge_count} ({edge_idx/edge_count*100:.1f}%)")
                    
                    # 現在の選択をクリア
                    bpy.ops.mesh.select_all(action='DESELECT')
                    
                    # エッジを選択
                    edge.select = True
                    bmesh.update_edit_mesh(target_obj.data)
                    
                    # エッジループを選択
                    bpy.ops.mesh.loop_multi_select(ring=False)
                    
                    # 選択されたループの頂点とエッジを取得
                    bm = bmesh.from_edit_mesh(target_obj.data)
                    loop_verts = {v.index for v in bm.verts if v.select}
                    
                    # ループが閉じているか確認（各頂点が正確に2つの選択されたエッジに接続されている）
                    is_closed_loop = True
                    for v in bm.verts:
                        if v.select:
                            # 選択された頂点に接続する選択されたエッジの数をカウント
                            selected_edge_count = sum(1 for e in v.link_edges if e.select)
                            # 選択された頂点に接続するエッジの総数をカウント
                            total_edge_count = len(v.link_edges)
                            # ループに含まれる頂点は、ループ内の2つの頂点とループ外の2つの頂点、
                            # 合計4つの頂点とエッジでつながっている必要がある
                            if selected_edge_count != 2 or total_edge_count != 4:
                                is_closed_loop = False
                                break
                    
                    # ループが閉じていて、完全に初期選択内に含まれるか確認
                    # if is_closed_loop and loop_verts.issubset(initial_selection):
                    if is_closed_loop:
                        # ループ内の頂点の元のウェイトパターンがコンポーネントのパターンと類似しているか確認
                        pattern_check_start = time.time()
                        is_similar_pattern = True
                        pattern_weights = data['original_pattern_weights']
                        
                        for vert_idx in loop_verts:
                            if vert_idx in original_vertex_weights:
                                orig_weights = original_vertex_weights[vert_idx]
                                
                                # ウェイトパターンの類似性をチェック
                                similarity_score = 0.0
                                total_weight = 0.0
                                
                                # パターン内の各グループについて
                                for group_name, pattern_weight in pattern_weights.items():
                                    orig_weight = orig_weights.get(group_name, 0.0)
                                    diff = abs(pattern_weight - orig_weight)
                                    similarity_score += diff
                                    total_weight += pattern_weight
                                
                                # 類似性スコアを正規化（0に近いほど類似）
                                if total_weight > 0:
                                    normalized_score = similarity_score / total_weight
                                    # 閾値を超える場合は類似していないと判断
                                    if normalized_score > 0.05:  # 閾値は調整可能
                                        is_similar_pattern = False
                                        break
                        
                        if is_similar_pattern:
                            complete_loops.update(loop_verts)
                
                # すべての選択をクリア
                bpy.ops.mesh.select_all(action='DESELECT')
                
                # 閉じたループのみを選択
                bm = bmesh.from_edit_mesh(target_obj.data)
                for vert in bm.verts:
                    if vert.index in complete_loops:
                        vert.select = True
                
                bmesh.update_edit_mesh(target_obj.data)
            
            print(f"    エッジループ検出時間: {time.time() - edge_loop_start:.2f}秒")
            
            # 選択範囲を拡大
            select_more_start = time.time()
            for _ in range(1):
                bpy.ops.mesh.select_more()
            
            # 選択された頂点のインデックスを取得
            bm = bmesh.from_edit_mesh(target_obj.data)
            selected_verts = [v.index for v in bm.verts if v.select]
            print(f"    選択範囲拡大時間: {time.time() - select_more_start:.2f}秒")

            if len(selected_verts) == 0:
                print(f"警告: OBB {obb_idx} 内に頂点が見つかりませんでした。スキップします。")
                continue
            
            # オブジェクトモードに戻る
            mode_switch_start = time.time()
            bpy.ops.object.mode_set(mode='OBJECT')
            print(f"    モード切替時間: {time.time() - mode_switch_start:.2f}秒")
            
            # 選択された頂点にConnected頂点グループのウェイトを設定
            weight_assign_start = time.time()
            for vert_idx in selected_verts:
                if vert_idx not in data['component_vertices']:  # コンポーネント内の頂点は除外
                    connected_group.add([vert_idx], 1.0, 'REPLACE')
            print(f"    ウェイト割り当て時間: {time.time() - weight_assign_start:.2f}秒")

            # Connectedグループにスムージングを適用
            smoothing_start = time.time()
            bpy.ops.object.select_all(action='DESELECT')
            target_obj.select_set(True)
            bpy.context.view_layer.objects.active = target_obj
            
            # Connectedグループを選択
            for i, group in enumerate(target_obj.vertex_groups):
                target_obj.vertex_groups.active_index = i
                if group.name == f"Connected_{data['component_id']}":
                    break
            
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

            # スムージングを適用
            smooth_op_start = time.time()
            bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=3, expand=0.5)
            print(f"    標準スムージング時間: {time.time() - smooth_op_start:.2f}秒")
            
            custom_smooth_start = time.time()
            #custom_max_vertex_group(target_obj, f"Connected_{data['component_id']}", vert_neighbors, repeat=1, weight_factor=1.0)
            custom_max_vertex_group_numpy(target_obj, f"Connected_{data['component_id']}", neighbors_info, offsets, num_verts, repeat=3, weight_factor=1.0)
            print(f"    カスタムスムージング時間: {time.time() - custom_smooth_start:.2f}秒")

            bpy.ops.object.mode_set(mode='OBJECT')
            print(f"    スムージング処理時間: {time.time() - smoothing_start:.2f}秒")
            
            # スムージング後、original_patternと各頂点のoriginal_vertex_weightsの差に基づいてウェイトを減衰
            decay_start = time.time()
            connected_group = target_obj.vertex_groups[f"Connected_{data['component_id']}"]
            original_pattern_weights = data['original_pattern_weights']
            
            for vert_idx, vert in enumerate(target_obj.data.vertices):
                if vert_idx in data['component_vertices']:
                    connected_group.add([vert_idx], 0.0, 'REPLACE')
                    continue
                if vert_idx not in data['component_vertices'] and vert_idx in original_vertex_weights:  # コンポーネント内の頂点は除外
                    # 元のウェイトパターンを取得
                    orig_weights = original_vertex_weights[vert_idx]
                    
                    # パターンとの差異を計算
                    similarity_score = 0.0
                    total_weight = 0.0

                    orig_weight_dict = {}
                    
                    # パターン内の各グループについて
                    for group_name, pattern_weight in original_pattern_weights.items():
                        orig_weight = orig_weights.get(group_name, 0.0)
                        diff = abs(pattern_weight - orig_weight)
                        similarity_score += diff
                        total_weight += pattern_weight
                        orig_weight_dict[group_name] = orig_weight
                    
                    # 類似性スコアを正規化（0に近いほど類似）
                    if total_weight > 0:
                        normalized_score = similarity_score / total_weight
                        # 類似性に基づいて減衰係数を計算（類似性が低いほど減衰が強い）
                        decay_factor = 1.0 - min(normalized_score * 3.33333, 1.0)  # 最大90%まで減衰
                        
                        # Connectedグループのウェイトを取得
                        connected_weight = 0.0
                        for g in target_obj.data.vertices[vert_idx].groups:
                            if g.group == connected_group.index:
                                connected_weight = g.weight
                                break
                        
                        # 減衰したウェイトを適用
                        if normalized_score > 0.3:
                            connected_group.add([vert_idx], 0.0, 'REPLACE')
                        else:
                            connected_group.add([vert_idx], connected_weight * decay_factor, 'REPLACE')
                    else:
                        connected_group.add([vert_idx], 0.0, 'REPLACE')
            print(f"    ウェイト減衰時間: {time.time() - decay_start:.2f}秒")
            
            print(f"  OBB {obb_idx+1}/{len(obb_data)} 処理時間: {time.time() - obb_start:.2f}秒")
            
            # 編集モードに戻る（次のループのため）
            if obb_idx < len(obb_data) - 1:
                bpy.ops.object.mode_set(mode='EDIT')
        
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"OBB処理時間: {time.time() - start_time:.2f}秒")
        
        # ウェイト合成開始
        start_time = time.time()
        # 各パターンのウェイトをConnectedグループのウェイトに基づいて合成
        # 複数のConnectedグループに属する頂点の場合は加重平均を計算
        connected_groups = [vg for vg in target_obj.vertex_groups if vg.name.startswith("Connected_")]
        
        if connected_groups:
            # 各頂点に対して処理
            for vert in target_obj.data.vertices:
                # コンポーネント内の頂点はスキップ（既に処理済み）
                skip = False
                for (new_pattern, original_pattern), components in component_patterns.items():
                    for component in components:
                        if vert.index in component:
                            skip = True
                            break
                    if skip:
                        break
                
                if skip:
                    continue
                
                # 各Connectedグループのウェイトとパターンを収集
                connected_weights = {}
                total_weight = 0.0
                
                for connected_group in connected_groups:
                    weight = 0.0
                    for g in vert.groups:
                        if g.group == connected_group.index:
                            weight = g.weight
                            break
                    
                    if weight > 0:
                        # グループ名からコンポーネントIDを抽出
                        component_id = int(connected_group.name.split('_')[1])
                        # 対応するパターンを見つける
                        for i, data in enumerate(obb_data):
                            if data['component_id'] == component_id:
                                # パターンウェイトからタプル形式に変換
                                pattern_tuple = tuple(sorted((k, v) for k, v in data['pattern_weights'].items() if v > 0.0001))
                                connected_weights[pattern_tuple] = weight
                                total_weight += weight
                                break
                
                # ウェイトが0の場合はスキップ
                if total_weight <= 0:
                    continue
                
                # 各グループのウェイトを合成
                combined_weights = {}
                
                for pattern, weight in connected_weights.items():
                    # パターンの正規化
                    normalized_weight = weight / total_weight
                    
                    # パターンからグループ名とウェイト値を抽出
                    for group_name, value in pattern:
                        if group_name not in combined_weights:
                            combined_weights[group_name] = 0.0
                        combined_weights[group_name] += value * normalized_weight

                factor = total_weight
                if total_weight > 1.0:
                    factor = 1.0
                
               # 既存のウェイト値を保存
                existing_weights = {}
                for group_name in existing_target_groups:
                    if group_name in target_obj.vertex_groups:
                        group = target_obj.vertex_groups[group_name]
                        weight = 0.0
                        for g in vert.groups:
                            if g.group == group.index:
                                weight = g.weight
                                break
                        existing_weights[group_name] = weight
                
                new_weights = {}
                
                # 既存のウェイト値を更新（factor に基づいて減衰）
                for group_name, weight in existing_weights.items():
                    if group_name in target_obj.vertex_groups and group_name in existing_target_groups:
                        group = target_obj.vertex_groups[group_name]
                        # 既存のウェイトを (1-factor) 倍に減衰
                        new_weights[group_name] = weight * (1.0 - factor)

                # 各パターンのウェイトを加算
                for pattern, weight in connected_weights.items():
                    # パターンの正規化
                    normalized_weight = weight / total_weight
                    if total_weight < 1.0:
                        normalized_weight = weight
                    # パターンからグループ名とウェイト値を抽出
                    for group_name, value in pattern:
                        if group_name in target_obj.vertex_groups and group_name in existing_target_groups:
                            group = target_obj.vertex_groups[group_name]
                            # 新しいウェイト値を計算
                            compornent_weight = value * normalized_weight
                            # ウェイトを更新
                            new_weights[group_name] = new_weights[group_name] + compornent_weight
                
                for group_name, weight in new_weights.items():
                    if weight > 1.0:
                        weight = 1.0
                    group = target_obj.vertex_groups[group_name]
                    group.add([vert.index], weight, 'REPLACE')
        print(f"ウェイト合成時間: {time.time() - start_time:.2f}秒")
    
    print(f"総処理時間: {time.time() - start_total:.2f}秒")
