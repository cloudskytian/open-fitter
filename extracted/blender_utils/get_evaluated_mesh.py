import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy


def get_evaluated_mesh(obj):
    """モディファイア適用後のメッシュを取得"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.data
    
    # BMeshを作成して評価済みメッシュの情報を取得
    bm = bmesh.new()
    bm.from_mesh(evaluated_mesh)
    bm.transform(obj.matrix_world)
    return bm
