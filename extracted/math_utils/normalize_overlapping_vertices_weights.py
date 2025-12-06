import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from blender_utils.subdivide_selected_vertices import subdivide_selected_vertices
from mathutils import Vector


def normalize_overlapping_vertices_weights(clothing_meshes, base_avatar_data, overlap_attr_name="Overlapped", world_pos_attr_name="OriginalWorldPosition"):
    """
    Overlapped属性が1となる頂点で構成される面およびエッジのみを対象に
    重なっている頂点のウェイトを正規化する
    
    Parameters:
        clothing_meshes: 処理対象の衣装メッシュのリスト
        base_avatar_data: ベースアバターデータ
        overlap_attr_name: 重なり検出フラグの属性名
        world_pos_attr_name: ワールド座標が保存された属性名
    """
    print("Normalizing weights for overlapping vertices using custom attributes...")

    original_active = bpy.context.view_layer.objects.active
    
    # チェック対象の頂点グループを取得
    target_groups = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)
    
    # 処理対象のメッシュをフィルタリング（必要な属性を持つメッシュのみ）
    valid_meshes = []
    for mesh in clothing_meshes:
        if (overlap_attr_name in mesh.data.attributes and 
            world_pos_attr_name in mesh.data.attributes and
            "InpaintMask" in mesh.vertex_groups):
            valid_meshes.append(mesh)
    
    if not valid_meshes:
        print(f"警告: {overlap_attr_name}と{world_pos_attr_name}属性を持つメッシュが見つかりません。処理をスキップします。")
        return
        
    # 各メッシュに対して処理
    for mesh_obj in valid_meshes:
        # オブジェクトを選択して編集モードに入る
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        
        # 元のメッシュを複製して処理用オブジェクトを作成
        bpy.ops.object.duplicate(linked=False)
        work_obj = bpy.context.active_object
        work_obj.name = f"{mesh_obj.name}_OverlapWork"
        
        # カスタム属性を取得
        overlap_attr = mesh_obj.data.attributes[overlap_attr_name]
        world_pos_attr = mesh_obj.data.attributes[world_pos_attr_name]
        
        # 重なっている頂点（属性値が1.0）を特定
        overlapping_verts_ids = [i for i, data in enumerate(overlap_attr.data) if data.value > 0.9999]
        
        if not overlapping_verts_ids:
            print(f"警告: {mesh_obj.name}に重なっている頂点が見つかりません。処理をスキップします。")
            bpy.data.objects.remove(work_obj, do_unlink=True)
            continue
        
        subdivide_selected_vertices(work_obj.name, overlapping_verts_ids, level=2)
        subdiv_overlap_attr = work_obj.data.attributes[overlap_attr_name]
        subdiv_overlapping_verts_ids = [i for i, data in enumerate(subdiv_overlap_attr.data) if data.value > 0.9999]
        subdiv_world_pos_attr = work_obj.data.attributes[world_pos_attr_name]
        
        # KDTreeを構築して近接頂点を効率的に検索
        subdiv_original_world_positions = []
        
        for vert_idx in subdiv_overlapping_verts_ids:
            world_pos = Vector(subdiv_world_pos_attr.data[vert_idx].vector)
            subdiv_original_world_positions.append(world_pos)
        
        # 重なっている頂点をグループ化（同じ位置の頂点をまとめる）
        distance_threshold = 0.0001  # 重なりの閾値
        overlapping_groups = {}
        
        for orig_idx, world_pos in zip(subdiv_overlapping_verts_ids, subdiv_original_world_positions):
            found_group = False
            
            for group_id, (group_pos, members) in overlapping_groups.items():
                if (world_pos - group_pos).length <= distance_threshold:
                    members.append(orig_idx)
                    found_group = True
                    break
            
            if not found_group:
                group_id = len(overlapping_groups)
                overlapping_groups[group_id] = (world_pos, [orig_idx])
        
        # 各グループで重なっている頂点の基準ウェイトを計算
        reference_weights = {}
        vert_weights = {} 
        for group_id, (group_pos, member_indices) in overlapping_groups.items():
            # InpaintMaskのウェイトを取得
            member_inpaint_weights = []
            for idx in member_indices:
                inpaint_weight = 0.0
                if "InpaintMask" in work_obj.vertex_groups:
                    inpaint_group = work_obj.vertex_groups["InpaintMask"]
                    for g in work_obj.data.vertices[idx].groups:
                        if g.group == inpaint_group.index:
                            inpaint_weight = g.weight
                            break
                member_inpaint_weights.append((idx, inpaint_weight))
            
            # InpaintMaskのウェイトでソート
            member_inpaint_weights.sort(key=lambda x: x[1])
            
            # InpaintMaskのウェイトが最小の頂点を基準にする
            if member_inpaint_weights:
                reference_idx = member_inpaint_weights[0][0]
                ref_weights = {}
                
                # 頂点グループのウェイトを取得
                for group_name in target_groups:
                    if group_name in work_obj.vertex_groups:
                        group = work_obj.vertex_groups[group_name]
                        weight = 0.0
                        for g in work_obj.data.vertices[reference_idx].groups:
                            if g.group == group.index:
                                weight = g.weight
                                break
                        ref_weights[group_name] = weight
                
                reference_weights[group_id] = ref_weights

                # 同じInpaintMaskウェイトの頂点がある場合は平均値を計算
                min_inpaint_weight = member_inpaint_weights[0][1]
                same_weight_vert_ids = [v[0] for v in member_inpaint_weights if abs(v[1] - min_inpaint_weight) < 0.0001]

                same_weight_verts = [work_obj.data.vertices[idx] for idx in same_weight_vert_ids]         
            
                if len(same_weight_verts) > 1:
                    # 平均ウェイトを計算
                    avg_weights = {}
                    for group_name in target_groups:
                        if group_name in work_obj.vertex_groups:
                            weights_sum = 0.0
                            count = 0
                            for v in same_weight_verts:
                                weight = 0.0
                                for g in v.groups:
                                    if g.group == mesh_obj.vertex_groups[group_name].index:
                                        weight = g.weight
                                        break
                                if weight > 0:
                                    weights_sum += weight
                                    count += 1
                            if count > 0:
                                avg_weights[group_name] = weights_sum / count
                    reference_weights[group_id] = avg_weights
            
                # すべての重なっている頂点に参照ウェイトを適用
                for vert_idx in member_indices:
                    vert_weights[vert_idx] = reference_weights[group_id].copy()
        
        # 元のメッシュに戻る
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        
        # 頂点グループを更新
        updated_count = 0
        
        # 細分化メッシュの頂点と元のメッシュの頂点をマッピング
        for orig_idx in overlapping_verts_ids:
            # 元の頂点のワールド座標を取得
            orig_world_pos = Vector(world_pos_attr.data[orig_idx].vector)
            
            # 最も近い細分化メッシュの頂点を見つける
            closest_idx = None
            min_dist = float('inf')
            
            for subdiv_idx, subdiv_pos in zip(subdiv_overlapping_verts_ids, subdiv_original_world_positions):
                dist = (orig_world_pos - subdiv_pos).length
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = subdiv_idx
            
            # 一定距離以内の場合、ウェイトを適用
            if closest_idx is not None and closest_idx in vert_weights and min_dist < distance_threshold:
                # 頂点グループのウェイトを更新
                for group_name, weight in vert_weights[closest_idx].items():
                    if group_name in mesh_obj.vertex_groups:
                        mesh_obj.vertex_groups[group_name].add([orig_idx], weight, 'REPLACE')
                updated_count += 1
        
        # 作業用オブジェクトを削除
        bpy.data.objects.remove(work_obj, do_unlink=True)
        
        print(f"{mesh_obj.name}の{updated_count}個の頂点のウェイトを正規化しました。")
    
    bpy.context.view_layer.objects.active = original_active
    print("重なっている頂点のウェイト正規化が完了しました。")
