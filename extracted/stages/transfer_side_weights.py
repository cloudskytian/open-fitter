import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import bpy
from blender_utils.reset_utils import reset_bone_weights
from io_utils.weights_io import restore_weights
from stages.attempt_weight_transfer import attempt_weight_transfer


def transfer_side_weights(context):
    left_transfer_time_start = time.time()
    left_transfer_success, left_distance_used = attempt_weight_transfer(
        context, bpy.data.objects["Body.BaseAvatar.LeftOnly"], "LeftSideWeights"
    )
    left_transfer_time = time.time() - left_transfer_time_start
    print(f"  左側ウェイト転送: {left_transfer_time:.2f}秒 (成功: {left_transfer_success}, 距離: {left_distance_used})")

    if not left_transfer_success:
        print("  左側ウェイト転送失敗のため処理中断")
        reset_bone_weights(context.target_obj, context.bone_groups)
        restore_weights(context.target_obj, context.all_weights)
        return False

    right_transfer_time_start = time.time()
    right_transfer_success, right_distance_used = attempt_weight_transfer(
        context, bpy.data.objects["Body.BaseAvatar.RightOnly"], "RightSideWeights", max_distance_tried=left_distance_used
    )
    right_transfer_time = time.time() - right_transfer_time_start
    print(f"  右側ウェイト転送: {right_transfer_time:.2f}秒 (成功: {right_transfer_success}, 距離: {right_distance_used})")

    if not right_transfer_success:
        print("  右側ウェイト転送失敗のため処理中断")
        reset_bone_weights(context.target_obj, context.bone_groups)
        restore_weights(context.target_obj, context.all_weights)
        return False
    return True
