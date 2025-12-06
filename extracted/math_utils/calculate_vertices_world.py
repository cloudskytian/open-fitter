import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np


def calculate_vertices_world(mesh_obj):
    """
    変形後のメッシュの頂点のワールド座標を取得
    
    Args:
        mesh_obj: メッシュオブジェクト
    Returns:
        vertices_world: ワールド座標のnumpy配列
    """
    # 変形後のメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = mesh_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.data
    
    # ワールド座標に変換（変形後の頂点位置を使用）
    vertices_world = np.array([evaluated_obj.matrix_world @ v.co for v in evaluated_mesh.vertices])
    
    return vertices_world
