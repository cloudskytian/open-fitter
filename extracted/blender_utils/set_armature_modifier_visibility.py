import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def set_armature_modifier_visibility(obj, show_viewport, show_render):
    """Armatureモディファイアの表示を設定"""
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            modifier.show_viewport = show_viewport
            modifier.show_render = show_render
