import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from mathutils import Vector


def check_mesh_obb_intersection(mesh_obj, obb):
    """
    メッシュとOBBの交差をチェックする
    
    Parameters:
        mesh_obj: チェック対象のメッシュオブジェクト
        obb: OBB情報（中心、軸、半径）
        
    Returns:
        bool: 交差する場合はTrue
    """
    if obb is None:
        return False
    
    # 評価済みメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    
    # メッシュの頂点をOBB空間に変換して交差チェック
    for v in eval_mesh.vertices:
        # 頂点のワールド座標
        vertex_world = mesh_obj.matrix_world @ v.co
        
        # OBBの中心からの相対位置
        relative_pos = vertex_world - Vector(obb['center'])
        
        # OBBの各軸に沿った投影
        projections = [abs(relative_pos.dot(Vector(obb['axes'][:, i]))) for i in range(3)]
        
        # すべての軸で投影が半径以内なら交差
        if all(proj <= radius for proj, radius in zip(projections, obb['radii'])):
            return True
    
    return False
