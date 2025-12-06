import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def merge_vertex_group_weights(mesh_obj: bpy.types.Object, source_group_name: str, target_group_name: str) -> None:
    """
    指定された頂点グループのウェイトを別のグループに統合する
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        source_group_name: 統合元のグループ名
        target_group_name: 統合先のグループ名
    """
    if source_group_name not in mesh_obj.vertex_groups or target_group_name not in mesh_obj.vertex_groups:
        return
        
    source_group = mesh_obj.vertex_groups[source_group_name]
    target_group = mesh_obj.vertex_groups[target_group_name]
    
    # 各頂点のウェイトを統合
    for vert in mesh_obj.data.vertices:
        source_weight = 0
        for group in vert.groups:
            if group.group == source_group.index:
                source_weight = group.weight
                break
        
        if source_weight > 0:
            # ターゲットグループにウェイトを加算
            target_group.add([vert.index], source_weight, 'ADD')
