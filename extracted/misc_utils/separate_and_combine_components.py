import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from algo_utils.check_uniform_weights import check_uniform_weights
from algo_utils.find_connected_components import find_connected_components
from blender_utils.generate_weight_hash import generate_weight_hash
from math_utils.calculate_component_size import calculate_component_size
from math_utils.calculate_obb import calculate_obb
from math_utils.cluster_components_by_adaptive_distance import (
    cluster_components_by_adaptive_distance,
)
from mathutils import Vector


def separate_and_combine_components(mesh_obj, clothing_armature, do_not_separate_names=None, clustering=True, clothing_avatar_data=None):
    """
    メッシュオブジェクト内の接続されていないコンポーネントを検出し、
    同じボーンウェイトパターンを持つものをグループ化して分離する
    
    Parameters:
        mesh_obj: 処理対象のメッシュオブジェクト
        clothing_armature: 衣装のアーマチュアオブジェクト
        do_not_separate_names: 分離しないオブジェクト名のリスト（オプション）
        clustering: クラスタリングを実行するかどうか
        clothing_avatar_data: 衣装のアバターデータ（オプション）
        
    Returns:
        (List[bpy.types.Object], List[bpy.types.Object]): 分離されたオブジェクトと分離されなかったオブジェクトのリスト
    """
    # 分離しないオブジェクト名のリストがNoneの場合は空リストを使用
    if do_not_separate_names is None:
        do_not_separate_names = []
    
    # 指定されたhumanoidボーンとそのauxiliaryBonesを取得
    allowed_bones = set()
    if clothing_avatar_data:
        # 対象のhumanoidボーン名
        target_humanoid_bones = ["Spine", "Chest", "Neck", "LeftBreast", "RightBreast"]
        
        # humanoidBonesからマッピングを作成
        humanoid_to_bone = {}
        if "humanoidBones" in clothing_avatar_data:
            for bone_data in clothing_avatar_data["humanoidBones"]:
                humanoid_name = bone_data.get("humanoidBoneName", "")
                bone_name = bone_data.get("boneName", "")
                if humanoid_name and bone_name:
                    humanoid_to_bone[humanoid_name] = bone_name
        
        # 対象のhumanoidボーンに対応するボーン名を追加
        for humanoid_bone in target_humanoid_bones:
            if humanoid_bone in humanoid_to_bone:
                allowed_bones.add(humanoid_to_bone[humanoid_bone])
        
        # auxiliaryBonesから関連するボーンを追加
        if "auxiliaryBones" in clothing_avatar_data:
            for aux_bone_data in clothing_avatar_data["auxiliaryBones"]:
                parent_humanoid = aux_bone_data.get("parentHumanoidBoneName", "")
                if parent_humanoid in target_humanoid_bones:
                    bone_name = aux_bone_data.get("boneName", "")
                    if bone_name:
                        allowed_bones.add(bone_name)
        
        print(f"Allowed bones for separation: {sorted(allowed_bones)}")
    
    def has_allowed_bone_weights(weights):
        """ウェイトパターンが許可されたボーンのウェイトを含むかチェック"""
        if not allowed_bones:
            return True  # 制限がない場合はすべて許可
        
        for bone_name in weights.keys():
            if bone_name in allowed_bones:
                return True
        return False
    
    # 連結成分を検出
    components = find_connected_components(mesh_obj)
    
    if len(components) <= 1:
        # 単一の連結成分の場合は分離しない
        return [], [mesh_obj]
    
    print(f"Found {len(components)} connected components in {mesh_obj.name}")
    
    # 各コンポーネントのウェイトを確認
    component_data = []
    weight_hash_do_not_separate = []
    for i, component in enumerate(components):
        is_uniform, weights = check_uniform_weights(mesh_obj, component, clothing_armature)
        
        if is_uniform and weights:
            # 許可されたボーンのウェイトを持つかチェック
            # if not has_allowed_bone_weights(weights):
            #     print(f"Component {i} in {mesh_obj.name} does not have allowed bone weights, skipping separation")
            #     component_data.append((component, False, {}, "", 0.0))
            #     continue
            
            # コンポーネント内の頂点のワールド座標を取得
            vertices_world = []
            for vert_idx in component:
                vert_co = mesh_obj.data.vertices[vert_idx].co.copy()
                vert_world = mesh_obj.matrix_world @ vert_co
                vertices_world.append(np.array([vert_world.x, vert_world.y, vert_world.z]))
            
            vertices_world = np.array(vertices_world)
            
            # OBBを計算
            axes, extents = calculate_obb(vertices_world)
            
            # 最長辺の長さを計算
            if extents is not None:
                max_extent = np.max(extents) * 2.0  # 半分の長さなので2倍
                
                # 一様なウェイトを持つコンポーネント
                weight_hash = generate_weight_hash(weights)
                
                # 小さすぎるコンポーネントは除外
                if max_extent < 0.0003:
                    print(f"Component {i} in {mesh_obj.name} is too small (max extent: {max_extent:.4f}), skipping")
                    component_data.append((component, False, {}, "", max_extent))
                else:
                    # do_not_separate_namesに含まれる名前のパターンを持つコンポーネントは分離しない
                    should_separate = True
                    temp_name = f"{mesh_obj.name}_Uniform_{i}"

                    # オブジェクト名チェック
                    if should_separate:
                        for name_pattern in do_not_separate_names:
                            if name_pattern in temp_name:
                                should_separate = False
                                print(f"Component {i} in {mesh_obj.name} name matches do_not_separate pattern: {name_pattern}")
                                weight_hash_do_not_separate.append(weight_hash)
                                break
                    
                    if should_separate:
                        for hash_val in weight_hash_do_not_separate:
                            if hash_val == weight_hash:
                                should_separate = False
                                print(f"Component {i} in {mesh_obj.name} weight hash matches do_not_separate pattern: {hash_val}")
                                break
                    
                    if should_separate:
                        print(f"Component {i} in {mesh_obj.name} has uniform weights: {weight_hash} (max extent: {max_extent:.4f})")
                        # 頂点座標も保存
                        component_data.append((component, True, weights, weight_hash, max_extent, vertices_world))
                    else:
                        component_data.append((component, False, {}, "", max_extent))
            else:
                # OBBの計算に失敗した場合は分離しない
                print(f"Component {i} in {mesh_obj.name} OBB calculation failed")
                component_data.append((component, False, {}, "", 0.0))
        else:
            # 一様でないか、ウェイトを持たないコンポーネント
            print(f"Component {i} in {mesh_obj.name} does not have uniform weights")
            component_data.append((component, False, {}, "", 0.0))
    
    # ウェイトハッシュでグループ化
    weight_groups = {}
    non_uniform_components = []
    
    for component, is_uniform, weights, weight_hash, max_extent, *extra_data in component_data:
        if is_uniform:
            if weight_hash not in weight_groups:
                weight_groups[weight_hash] = []
            vertices_world = extra_data[0] if extra_data else None
            weight_groups[weight_hash].append((component, vertices_world))
        else:
            non_uniform_components.append(component)
    
    # 一様なウェイトを持つコンポーネントを分離
    uniform_objects = []
    
    if clustering:
        # 各ウェイトハッシュのコンポーネントを空間的な距離に基づいてさらにクラスタリング
        for weight_hash, components_with_coords in weight_groups.items():
            # コンポーネントの座標とサイズを計算
            component_coords = {}
            component_sizes = {}
            component_indices = {}
            
            for i, (component, vertices_world) in enumerate(components_with_coords):
                if vertices_world is not None and len(vertices_world) > 0:
                    # コンポーネントの中心を計算
                    center = np.mean(vertices_world, axis=0)
                    # NumPy配列をVectorに変換
                    vectors = [Vector(v) for v in vertices_world]
                    
                    component_coords[i] = vectors
                    component_sizes[i] = calculate_component_size(vectors)
                    component_indices[i] = component
            
            # 空間的なクラスタリングを実行
            clusters = cluster_components_by_adaptive_distance(component_coords, component_sizes)
            
            print(f"Weight hash {weight_hash} has {len(clusters)} spatial clusters")
            
            # 各クラスターごとに別々のオブジェクトを作成
            for cluster_idx, cluster in enumerate(clusters):
                # 名前を設定（最初のコンポーネントIDと空間クラスターIDを使用）
                first_component_id = -1
                for i, (component, is_uniform, weights, hash_val, _, *_) in enumerate(component_data):
                    if is_uniform and hash_val == weight_hash:
                        for comp_idx in cluster:
                            if component == component_indices[comp_idx]:
                                first_component_id = i
                                break
                        if first_component_id >= 0:
                            break
                
                if first_component_id >= 0:
                    cluster_name = f"{mesh_obj.name}_Uniform_{first_component_id}_Cluster_{cluster_idx}"
                else:
                    cluster_name = f"{mesh_obj.name}_Uniform_Hash_{len(uniform_objects)}_Cluster_{cluster_idx}"
                
                should_separate = True
                for name_pattern in do_not_separate_names:
                    if name_pattern in cluster_name:
                        print(f"Component {i} in {cluster_name} name matches do_not_separate pattern: {name_pattern}")
                        for (component, vertices_world) in components_with_coords:
                            non_uniform_components.append(component)
                        should_separate = False
                        break
                if not should_separate:
                    continue

                # アクティブオブジェクトの保存
                original_active = bpy.context.view_layer.objects.active
                
                # 元のメッシュを選択
                bpy.ops.object.select_all(action='DESELECT')
                mesh_obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_obj
                
                # オブジェクトを複製
                bpy.ops.object.duplicate(linked=False)
                new_obj = bpy.context.active_object
                new_obj.name = cluster_name
                
                # クラスター内のコンポーネントの頂点を収集
                keep_vertices = set()
                for comp_idx in cluster:
                    keep_vertices.update(component_indices[comp_idx])
                
                # このクラスターに属する頂点以外を削除
                # 編集モードに入る
                bpy.ops.object.select_all(action='DESELECT')
                new_obj.select_set(True)
                bpy.context.view_layer.objects.active = new_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_mode(type="VERT")
                
                # 全頂点の選択を解除
                bpy.ops.mesh.select_all(action='DESELECT')
                
                # 保持する頂点を選択
                bpy.ops.object.mode_set(mode='OBJECT')
                for i, vert in enumerate(new_obj.data.vertices):
                    vert.select = i in keep_vertices
                
                # 選択頂点以外を削除
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='INVERT')
                bpy.ops.mesh.delete(type='VERT')
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # オブジェクトに元のシェイプキーを保持
                if mesh_obj.data.shape_keys:
                    for key_block in mesh_obj.data.shape_keys.key_blocks:
                        if key_block.name not in new_obj.data.shape_keys.key_blocks:
                            shape_key = new_obj.shape_key_add(name=key_block.name)
                            # シェイプキーの値をコピー
                            shape_key.value = key_block.value

                uniform_objects.append(new_obj)
                
                # 元のアクティブオブジェクトに戻す
                bpy.context.view_layer.objects.active = original_active
    
    # 分離されないコンポーネントがある場合は元のメッシュを複製
    if non_uniform_components:
        # アクティブオブジェクトの保存
        original_active = bpy.context.view_layer.objects.active
        
        # 元のメッシュを選択
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        
        # オブジェクトを複製
        bpy.ops.object.duplicate(linked=False)
        non_uniform_obj = bpy.context.active_object
        non_uniform_obj.name = f"{mesh_obj.name}_NonUniform"
        
        # 分離されないコンポーネントの頂点以外を削除
        keep_vertices = set()
        for component in non_uniform_components:
            keep_vertices.update(component)

        # 編集モードに入る
        bpy.ops.object.select_all(action='DESELECT')
        non_uniform_obj.select_set(True)
        bpy.context.view_layer.objects.active = non_uniform_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type="VERT")
        
        # 全頂点の選択を解除
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # 保持する頂点を選択
        bpy.ops.object.mode_set(mode='OBJECT')
        for i, vert in enumerate(non_uniform_obj.data.vertices):
            vert.select = i in keep_vertices
        
        # 選択頂点以外を削除
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 元のアクティブオブジェクトに戻す
        bpy.context.view_layer.objects.active = original_active
    else:
        non_uniform_obj = None
    
    # 返却するオブジェクトリストを作成
    separated_objects = uniform_objects
    non_separated_objects = [non_uniform_obj] if non_uniform_obj else []

    # 分離されなかったオブジェクトの頂点数を表示
    if non_uniform_obj:
        print(f"Non-separated object '{non_uniform_obj.name}' vertex count: {len(non_uniform_obj.data.vertices)}")
    else:
        print("No non-separated object.")
    
    # 分離された各オブジェクトの頂点数を表示
    for sep_obj in uniform_objects:
        print(f"Separated object '{sep_obj.name}' vertex count: {len(sep_obj.data.vertices)}")
    
    return separated_objects, non_separated_objects
