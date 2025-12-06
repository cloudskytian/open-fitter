import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def store_armature_modifier_settings(obj):
    """Armatureモディファイアの設定を保存"""
    armature_settings = []
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE':
            settings = {
                'name': modifier.name,
                'object': modifier.object,
                'vertex_group': modifier.vertex_group,
                'invert_vertex_group': modifier.invert_vertex_group,
                'use_vertex_groups': modifier.use_vertex_groups,
                'use_bone_envelopes': modifier.use_bone_envelopes,
                'use_deform_preserve_volume': modifier.use_deform_preserve_volume,
                'use_multi_modifier': modifier.use_multi_modifier,
                'show_viewport': modifier.show_viewport,
                'show_render': modifier.show_render,
            }
            armature_settings.append(settings)
    return armature_settings
