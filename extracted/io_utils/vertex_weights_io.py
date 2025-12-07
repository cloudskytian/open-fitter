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


def restore_vertex_weights(mesh_obj: bpy.types.Object, weights_data: dict) -> None:
    """
    保存されたウェイト情報を使って頂点グループのウェイトを復元する
    カスタム属性を使用して頂点IDの対応を管理
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        weights_data: save_vertex_weights()で保存されたウェイト情報
    """
    vertex_weights = weights_data['vertex_weights']
    original_groups = weights_data['existing_groups']
    saved_vertex_ids = weights_data.get('vertex_ids', {})
    
    # 現在存在するグループのうち、元々存在しなかったグループを削除
    current_groups = set(group.name for group in mesh_obj.vertex_groups)
    groups_to_remove = current_groups - original_groups
    
    for group_name in groups_to_remove:
        if group_name in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[group_name])
            print(f"Removed vertex group {group_name} from {mesh_obj.name}")
    
    # 元々存在していたグループが削除されている場合は再作成
    for group_name in original_groups:
        if group_name not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=group_name)
    
    # まず全ての頂点グループから全頂点を削除
    for group in mesh_obj.vertex_groups:
        group.remove(list(range(len(mesh_obj.data.vertices))))
    
    # カスタム属性から頂点IDの対応を取得
    mesh = mesh_obj.data
    custom_attr_name = "original_vertex_id"
    
    if custom_attr_name not in mesh.attributes:
        print(f"Warning: Custom attribute '{custom_attr_name}' not found in {mesh_obj.name}. Using direct index mapping.")
        # カスタム属性がない場合は従来の方法でインデックスを直接使用
        for vert_index, vertex_weights_dict in vertex_weights.items():
            if vert_index < len(mesh.vertices):
                for group_name, weight in vertex_weights_dict.items():
                    if group_name in mesh_obj.vertex_groups:
                        group = mesh_obj.vertex_groups[group_name]
                        group.add([vert_index], weight, 'REPLACE')
        return
    
    # カスタム属性を取得
    custom_attr = mesh.attributes[custom_attr_name]
    
    # 現在の頂点の元の頂点IDを取得してマッピングを作成
    current_to_original_mapping = {}
    for current_vert in mesh.vertices:
        original_id = custom_attr.data[current_vert.index].value
        current_to_original_mapping[current_vert.index] = original_id
    
    print(f"Restoring vertex weights using custom attribute mapping for {len(mesh.vertices)} vertices in {mesh_obj.name}")
    
    # 保存されたウェイトを復元（カスタム属性を使用して対応を取る）
    restored_count = 0
    for current_vert_index, original_vert_id in current_to_original_mapping.items():
        if original_vert_id in vertex_weights:
            vertex_weights_dict = vertex_weights[original_vert_id]
            for group_name, weight in vertex_weights_dict.items():
                if group_name in mesh_obj.vertex_groups:
                    group = mesh_obj.vertex_groups[group_name]
                    group.add([current_vert_index], weight, 'REPLACE')
            restored_count += 1
    
    print(f"Successfully restored weights for {restored_count} vertices in {mesh_obj.name}")
