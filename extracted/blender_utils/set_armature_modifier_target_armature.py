import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def set_armature_modifier_target_armature(obj, target_armature):
    """Armatureモディファイアの表示を設定"""
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            modifier.object = target_armature
