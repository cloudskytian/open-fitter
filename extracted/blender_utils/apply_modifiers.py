import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_modifiers(obj):
    """モディファイアを適用"""
    bpy.context.view_layer.objects.active = obj
    for modifier in obj.modifiers[:]:  # スライスを使用してリストのコピーを作成
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception as e:
            print(f"Failed to apply modifier {modifier.name}: {e}")
