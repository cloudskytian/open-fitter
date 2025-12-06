import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_all_shapekeys(obj):
    """オブジェクトの全シェイプキーを適用する"""
    if not obj.data.shape_keys:
        return
    
    # 基底シェイプキーは常にインデックス0
    if obj.active_shape_key_index == 0 and len(obj.data.shape_keys.key_blocks) > 1:
        obj.active_shape_key_index = 1
    else:
        obj.active_shape_key_index = 0

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
