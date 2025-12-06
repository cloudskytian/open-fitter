import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def remove_propagated_weights(mesh_obj: bpy.types.Object, temp_group_name: str) -> None:
    """
    伝播させたウェイトを削除する
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        temp_group_name: 伝播頂点を記録した頂点グループの名前
    """
    # 一時頂点グループが存在することを確認
    temp_group = mesh_obj.vertex_groups.get(temp_group_name)
    if not temp_group:
        return
    
    # アーマチュアモディファイアからアーマチュアを取得
    armature_obj = None
    for modifier in mesh_obj.modifiers:
        if modifier.type == 'ARMATURE':
            armature_obj = modifier.object
            break
    
    if not armature_obj:
        print(f"Warning: No armature modifier found in {mesh_obj.name}")
        return
    
    # アーマチュアのすべてのボーン名を取得
    deform_groups = {bone.name for bone in armature_obj.data.bones}
    
    # 伝播させた頂点のウェイトを削除
    for vert in mesh_obj.data.vertices:
        # 一時グループのウェイトを取得
        weight = 0.0
        for g in vert.groups:
            if g.group == temp_group.index:
                weight = g.weight
                break
        
        # ウェイトが0より大きい場合（伝播された頂点の場合）
        if weight > 0:
            for group in mesh_obj.vertex_groups:
                try:
                    group.remove([vert.index])
                except RuntimeError:
                    continue
    
    # 一時頂点グループを削除
    mesh_obj.vertex_groups.remove(temp_group)
