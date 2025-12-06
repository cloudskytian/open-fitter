import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def restore_armature_modifier(obj, settings):
    """Armatureモディファイアを復元"""
    for modifier_settings in settings:
        modifier = obj.modifiers.new(name=modifier_settings['name'], type='ARMATURE')
        modifier.object = modifier_settings['object']
        modifier.vertex_group = modifier_settings['vertex_group']
        modifier.invert_vertex_group = modifier_settings['invert_vertex_group']
        modifier.use_vertex_groups = modifier_settings['use_vertex_groups']
        modifier.use_bone_envelopes = modifier_settings['use_bone_envelopes']
        modifier.use_deform_preserve_volume = modifier_settings['use_deform_preserve_volume']
        modifier.use_multi_modifier = modifier_settings['use_multi_modifier']
        modifier.show_viewport = modifier_settings['show_viewport']
        modifier.show_render = modifier_settings['show_render']
