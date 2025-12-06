import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def save_vertex_weights(mesh_obj: bpy.types.Object) -> dict:
    """
    オブジェクトの全頂点グループのウェイトを記録する（空のグループも含む）
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        
    Returns:
        保存されたウェイト情報のディクショナリ（vertex_weights、existing_groups、vertex_ids）
    """
    weights_data = {
        'vertex_weights': {},
        'existing_groups': set(),
        'vertex_ids': {}
    }
    
    # 全ての既存の頂点グループ名を記録
    for group in mesh_obj.vertex_groups:
        weights_data['existing_groups'].add(group.name)
    
    # 頂点に整数型のカスタム属性を作成（既に存在する場合は削除して再作成）
    mesh = mesh_obj.data
    custom_attr_name = "original_vertex_id"
    
    # 既存のカスタム属性を削除
    if custom_attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[custom_attr_name])
    
    # 新しい整数型カスタム属性を作成
    custom_attr = mesh.attributes.new(name=custom_attr_name, type='INT', domain='POINT')
    
    # 各頂点のウェイトと頂点IDを記録
    for vert in mesh.vertices:
        vertex_weights = {}
        for group in vert.groups:
            group_name = mesh_obj.vertex_groups[group.group].name
            vertex_weights[group_name] = group.weight
        
        # 頂点のウェイトを記録（空の場合も記録）
        weights_data['vertex_weights'][vert.index] = vertex_weights
        
        # カスタム属性に現在の頂点IDを設定
        custom_attr.data[vert.index].value = vert.index
        
        # weights_dataにも頂点IDを記録
        weights_data['vertex_ids'][vert.index] = vert.index
    
    print(f"Saved vertex weights for {len(mesh.vertices)} vertices with original IDs in {mesh_obj.name}")
    
    return weights_data
