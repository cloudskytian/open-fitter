import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import bpy


def smooth_and_cleanup(context):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action="DESELECT")
    inpaint_mask_group = context.target_obj.vertex_groups.get("InpaintMask")
    if inpaint_mask_group:
        inpaint_group_idx = inpaint_mask_group.index
        for vert in context.target_obj.data.vertices:
            for g in vert.groups:
                if g.group == inpaint_group_idx and g.weight >= 0.5:
                    vert.select = True
                    break

    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.context.object.data.use_paint_mask = False
    bpy.context.object.data.use_paint_mask_vertex = True
    for group_name in context.bone_groups:
        if group_name in context.target_obj.vertex_groups:
            context.target_obj.vertex_groups.active = context.target_obj.vertex_groups[group_name]
            bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=3, expand=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    cleanup_weights_time_start = time.time()
    
    # bone_groupsに属するグループのインデックスをセット化（高速検索用）
    bone_group_indices = set()
    for group_name in context.bone_groups:
        if group_name in context.target_obj.vertex_groups:
            bone_group_indices.add(context.target_obj.vertex_groups[group_name].index)
    
    # 削除対象を一括収集してから処理（頂点ごとのremove呼び出しを最小化）
    removal_map = {}  # group_idx -> [vert_indices]
    
    for vert in context.target_obj.data.vertices:
        for g in vert.groups:
            if g.group in bone_group_indices and g.weight < 0.001:
                if g.group not in removal_map:
                    removal_map[g.group] = []
                removal_map[g.group].append(vert.index)
    
    # グループごとにまとめて削除
    for group_idx, vert_indices in removal_map.items():
        group = context.target_obj.vertex_groups[group_idx]
        try:
            group.remove(vert_indices)
        except RuntimeError:
            # 個別にフォールバック
            for vert_idx in vert_indices:
                try:
                    group.remove([vert_idx])
                except RuntimeError:
                    continue
    
    cleanup_weights_time = time.time() - cleanup_weights_time_start
