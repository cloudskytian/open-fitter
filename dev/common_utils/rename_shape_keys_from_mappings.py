import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def rename_shape_keys_from_mappings(meshes, blend_shape_mappings):
    """
    辞書データに基づいてメッシュのシェイプキー名を置き換える
    
    辞書の値（カスタム名）と一致するシェイプキーがあれば、
    それをキー（ラベル名）に置き換える
    
    Parameters:
        meshes: メッシュオブジェクトのリスト
        blend_shape_mappings: {label: customName} の辞書
    """
    if not blend_shape_mappings:
        return
    
    # 逆マッピングを作成（customName -> label）
    reverse_mappings = {custom_name: label for label, custom_name in blend_shape_mappings.items()}
    
    for obj in meshes:
        if not obj.data.shape_keys:
            continue
        
        # 名前を変更する必要があるシェイプキーを収集
        keys_to_rename = []
        for shape_key in obj.data.shape_keys.key_blocks:
            if shape_key.name in reverse_mappings:
                new_name = reverse_mappings[shape_key.name]
                keys_to_rename.append((shape_key, new_name))
        
        # 名前を変更
        for shape_key, new_name in keys_to_rename:
            old_name = shape_key.name
            shape_key.name = new_name
