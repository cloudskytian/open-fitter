import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy


def subdivide_selected_vertices(obj_name, vertex_indices, level=2):
    """
    特定のメッシュの選択された頂点を細分化する
    
    引数:
        obj_name (str): 操作対象のオブジェクト名
        vertex_indices (list): 選択する頂点のインデックスリスト
        cuts (int): 細分化の分割数
    """
    # アクティブオブジェクトの設定
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = bpy.data.objects.get(obj_name)
    
    if obj is None:
        print(f"オブジェクト '{obj_name}' が見つかりません")
        return
    
    # オブジェクトを選択してアクティブに
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # 編集モードに切り替え
    bpy.ops.object.mode_set(mode='EDIT')
    
    # bmeshを取得
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # すべての選択を解除
    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False
    
    # 指定された頂点を選択
    for idx in vertex_indices:
        if idx < len(bm.verts):
            bm.verts[idx].select = True
    
    # 選択された頂点「同士」で構成されるエッジのみを選択
    # つまり、エッジの両端の頂点が両方とも選択された頂点リストに含まれる場合のみ選択
    selected_verts = set(bm.verts[idx] for idx in vertex_indices if idx < len(bm.verts))
    connected_edges = []
    
    for e in bm.edges:
        # エッジの両端の頂点が両方とも選択された頂点セットに含まれる場合のみ選択
        if e.verts[0] in selected_verts and e.verts[1] in selected_verts:
            e.select = True
            connected_edges.append(e)
    
    # 変更を適用
    bmesh.update_edit_mesh(me)
    
    # 細分化操作
    if connected_edges:
        for _ in range(level):
            bpy.ops.mesh.subdivide(number_cuts=1)
        print(f"{len(connected_edges)} 個のエッジが細分化されました")
    else:
        print("選択された頂点間にエッジが見つかりませんでした")
    
    # オブジェクトモードに戻る
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.data.update()
