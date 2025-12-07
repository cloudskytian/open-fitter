import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import bpy
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from blender_utils.reset_bone_weights import reset_bone_weights
from create_side_weight_groups import create_side_weight_groups
from io_utils.store_weights import store_weights


def prepare_groups_and_weights(context):
    if "InpaintMask" not in context.target_obj.vertex_groups:
        context.target_obj.vertex_groups.new(name="InpaintMask")

    side_weight_time_start = time.time()
    create_side_weight_groups(context.target_obj, context.base_avatar_data, context.clothing_armature, context.clothing_avatar_data)
    side_weight_time = time.time() - side_weight_time_start
    print(f"  側面ウェイトグループ作成: {side_weight_time:.2f}秒")

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = context.target_obj

    context.original_groups = set(vg.name for vg in context.target_obj.vertex_groups)
    context.bone_groups = set(get_humanoid_and_auxiliary_bone_groups(context.base_avatar_data))

    store_weights_time_start = time.time()
    context.original_humanoid_weights = store_weights(context.target_obj, context.bone_groups)
    store_weights_time = time.time() - store_weights_time_start
    print(f"  元のウェイト保存: {store_weights_time:.2f}秒")

    context.all_deform_groups = set(context.bone_groups)
    if context.clothing_armature:
        context.all_deform_groups.update(bone.name for bone in context.clothing_armature.data.bones)

    context.original_non_humanoid_groups = context.all_deform_groups - context.bone_groups
    context.original_non_humanoid_weights = store_weights(context.target_obj, context.original_non_humanoid_groups)
    context.all_weights = store_weights(context.target_obj, context.all_deform_groups)

    reset_weights_time_start = time.time()
    reset_bone_weights(context.target_obj, context.all_deform_groups)
    reset_weights_time = time.time() - reset_weights_time_start
    print(f"  ウェイト初期化: {reset_weights_time:.2f}秒")
