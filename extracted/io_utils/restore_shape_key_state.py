import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def restore_shape_key_state(mesh_obj: bpy.types.Object, shape_key_state: dict) -> None:
    """
    メッシュオブジェクトのシェイプキー状態を復元する
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        shape_key_state: 復元するシェイプキー状態のディクショナリ
    """
    if not mesh_obj or not mesh_obj.data.shape_keys or not shape_key_state:
        return
    
    for key_name, value in shape_key_state.items():
        if key_name in mesh_obj.data.shape_keys.key_blocks:
            mesh_obj.data.shape_keys.key_blocks[key_name].value = value
