import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_armature_from_modifier(mesh_obj):
    """Armatureモディファイアからアーマチュアを取得"""
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE':
            return modifier.object
    return None
