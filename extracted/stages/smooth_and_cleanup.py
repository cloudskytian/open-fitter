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
        for vert in context.target_obj.data.vertices:
            for g in vert.groups:
                if g.group == inpaint_mask_group.index and g.weight >= 0.5:
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
    for vert in context.target_obj.data.vertices:
        groups_to_remove = []
        for g in vert.groups:
            group_name = context.target_obj.vertex_groups[g.group].name
            if group_name in context.bone_groups and g.weight < 0.001:
                groups_to_remove.append(g.group)
        for group_idx in groups_to_remove:
            try:
                context.target_obj.vertex_groups[group_idx].remove([vert.index])
            except RuntimeError:
                continue
    cleanup_weights_time = time.time() - cleanup_weights_time_start
