import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def save_shape_key_state(mesh_obj: bpy.types.Object) -> dict:
    """
    メッシュオブジェクトのシェイプキー状態を保存する
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        
    Returns:
        保存されたシェイプキー状態のディクショナリ
    """
    if not mesh_obj or not mesh_obj.data.shape_keys:
        return {}
    
    shape_key_state = {}
    for key_block in mesh_obj.data.shape_keys.key_blocks:
        shape_key_state[key_block.name] = key_block.value
    
    return shape_key_state
