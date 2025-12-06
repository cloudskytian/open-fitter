import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def normalize_vertex_weights(obj):
    """
    指定されたメッシュオブジェクトのボーンウェイトを正規化する。
    Args:
        obj: 正規化するメッシュオブジェクト
    """
    if obj.type != 'MESH':
        print(f"Error: {obj.name} is not a mesh object")
        return

    # 頂点グループが存在するか確認
    if not obj.vertex_groups:
        print(f"Warning: {obj.name} has no vertex groups")
        return
        
    # 各頂点が少なくとも1つのグループに属しているか確認
    for vert in obj.data.vertices:
        if not vert.groups:
            print(f"Warning: Vertex {vert.index} in {obj.name} has no weights")
    
    # Armatureモディファイアの確認
    has_armature = any(mod.type == 'ARMATURE' for mod in obj.modifiers)
    if not has_armature:
        print(f"Error: {obj.name} has no Armature modifier")
        return
    
    # すべての選択を解除
    bpy.ops.object.select_all(action='DESELECT')

    # アクティブオブジェクトを設定
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # ウェイトの正規化を実行
    bpy.ops.object.vertex_group_normalize_all(
        group_select_mode='BONE_DEFORM',
        lock_active=False
    )
    print(f"Normalized weights for {obj.name}")
