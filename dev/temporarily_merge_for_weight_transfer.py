import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from mathutils.kdtree import KDTree
from process_weight_transfer_with_component_normalization import (
    process_weight_transfer_with_component_normalization,
)


def temporarily_merge_for_weight_transfer(container_obj, contained_objs, base_armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, blend_shape_settings, cloth_metadata):
    """
    オブジェクトを一時的に結合し、weight transferのみを適用した後、結果を元のオブジェクトに復元する
    
    Parameters:
        container_obj: 包含するオブジェクト
        contained_objs: 包含されるオブジェクトのリスト
        base_armature: ベースのアーマチュア
        base_avatar_data: ベースアバターデータ
        clothing_avatar_data: 衣装アバターデータ
        field_path: フィールドパス
        clothing_armature: 衣装のアーマチュア
        cloth_metadata: クロスメタデータ
    """
    # 元のデータを保存
    original_active = bpy.context.active_object
    original_mode = bpy.context.mode
    
    # すべてのオブジェクトの選択状態を保存
    original_selection = {obj: obj.select_get() for obj in bpy.data.objects}
    
    # 一時的なリストにすべてのオブジェクトを追加
    to_merge = [container_obj] + contained_objs
    
    # 頂点グループの情報を保存
    vertex_groups_data = {}
    for obj in to_merge:
        vertex_groups_data[obj.name] = {}
        for vg in obj.vertex_groups:
            vg_data = []
            for v in obj.data.vertices:
                weight = 0.0
                for g in v.groups:
                    if g.group == vg.index:
                        weight = g.weight
                        break
                if weight > 0:
                    vg_data.append((v.index, weight))
            vertex_groups_data[obj.name][vg.name] = vg_data
    
    # すべてのオブジェクトの複製を作成
    duplicated_objs = []
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in to_merge:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()
        dup_obj = bpy.context.active_object
        duplicated_objs.append(dup_obj)
        bpy.ops.object.select_all(action='DESELECT')
    
    # 複製したオブジェクトを結合
    bpy.ops.object.select_all(action='DESELECT')
    for obj in duplicated_objs:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = duplicated_objs[0]
    bpy.ops.object.join()
    
    # 結合したオブジェクト
    merged_obj = bpy.context.active_object
    merged_obj.name = f"TempMerged_{container_obj.name}"
    
    # 結合したオブジェクトに対してweight transferのみを適用
    # process_weight_transfer(merged_obj, base_armature, base_avatar_data, field_path, clothing_armature, cloth_metadata)
    process_weight_transfer_with_component_normalization(merged_obj, base_armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, blend_shape_settings, cloth_metadata)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # モディファイア適用後のソースメッシュを取得
    eval_merged_obj = merged_obj.evaluated_get(depsgraph)
    eval_merged_mesh = eval_merged_obj.data
    merged_world_coords = [merged_obj.matrix_world @ v.co for v in eval_merged_mesh.vertices]

    # KDTreeを使用して最も近い頂点を高速に検索
    kdtree = KDTree(len(merged_world_coords))
    for i, v_co in enumerate(merged_world_coords):
        kdtree.insert(v_co, i)
    kdtree.balance()
    
    # 頂点グループの情報を元のオブジェクトに復元
    for obj in to_merge:
        # 既存の頂点グループをクリア
        for vg in obj.vertex_groups[:]:
            obj.vertex_groups.remove(vg)
        
        # 結合オブジェクトから新しい頂点グループを作成
        for vg in merged_obj.vertex_groups:
            obj.vertex_groups.new(name=vg.name)
        
        # 評価済みの頂点座標を取得（現在の状態）
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.data
        obj_world_coords = [obj.matrix_world @ v.co for v in eval_mesh.vertices]
        
        # 元のオブジェクトの各頂点に対して最も近い頂点を探し、ウェイトをコピー
        for i, vert_co in enumerate(obj_world_coords):
            co, merged_vert_idx, dist = kdtree.find(vert_co)
            
            # マージされたオブジェクト内の対応する頂点からウェイト情報をコピー
            if merged_vert_idx >= 0:
                for g in merged_obj.data.vertices[merged_vert_idx].groups:
                    vg_name = merged_obj.vertex_groups[g.group].name
                    if vg_name in obj.vertex_groups:
                        obj.vertex_groups[vg_name].add([i], g.weight, 'REPLACE')
    
    # 一時的なオブジェクトを削除
    bpy.ops.object.select_all(action='DESELECT')
    merged_obj.select_set(True)
    bpy.ops.object.delete()
    
    # 元の選択状態を復元
    for obj, was_selected in original_selection.items():
        if obj.name in bpy.data.objects:  # オブジェクトが存在することを確認
            obj.select_set(was_selected)
    
    # 元のアクティブオブジェクトと元のモードを復元
    if original_active and original_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = original_active
    
    if original_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=original_mode)
