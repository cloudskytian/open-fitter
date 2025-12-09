import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy
from intersect_triangle_triangle import intersect_triangle_triangle
from mathutils.bvhtree import BVHTree


def find_intersecting_faces_bvh(obj):
    """
    BVHを用いてメッシュ内の自己交差を検出する。
    各面（３角形または４角形）はまず三角形に分割し、
    それぞれの三角形のバウンディングボックスに基づき候補ペアを
    BVHで絞り込んだ上で、詳細な三角形交差判定を行う。
    隣接面（頂点共有）は除外しています。
    """
    # 評価済みメッシュを取得
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.data
    
    # 作業用のBMeshを作成
    bm = bmesh.new()
    bm.from_mesh(evaluated_mesh)
    bm.faces.ensure_lookup_table()
    bm.transform(obj.matrix_world)
    
    # 各面から「三角形」リストを作成
    triangles = []         # 各要素は [Vector, Vector, Vector]
    face_map = []          # 各三角形が元々属していた面のインデックス
    face_vertex_sets = []  # 各三角形の元の面の頂点インデックス集合（隣接面チェック用）
    
    for face in bm.faces:
        if len(face.verts) not in [3, 4]:
            continue
        vertex_set = {v.index for v in face.verts}
        if len(face.verts) == 3:
            tri = [v.co.copy() for v in face.verts]
            triangles.append(tri)
            face_map.append(face.index)
            face_vertex_sets.append(vertex_set)
        elif len(face.verts) == 4:
            # 対角線の長さにより分割方法を選択
            v = [v.co.copy() for v in face.verts]
            diag1 = (v[2] - v[0]).length_squared
            diag2 = (v[3] - v[1]).length_squared
            if diag1 < diag2:
                tri1 = [v[0], v[1], v[2]]
                tri2 = [v[0], v[2], v[3]]
            else:
                tri1 = [v[0], v[1], v[3]]
                tri2 = [v[1], v[2], v[3]]
            triangles.append(tri1)
            face_map.append(face.index)
            face_vertex_sets.append(vertex_set)
            triangles.append(tri2)
            face_map.append(face.index)
            face_vertex_sets.append(vertex_set)
    
    # BVHツリー作成用の頂点リストと三角形（ポリゴン）リストを構築
    bvh_verts = []
    bvh_polys = []
    offset = 0
    for tri in triangles:
        bvh_verts.extend(tri)           # 各三角形は独立の頂点集合として追加（同じ頂点でも複製）
        bvh_polys.append((offset, offset+1, offset+2))
        offset += 3
    
    # BVHツリーを作成
    epsilon = 1e-6
    bvh_tree = BVHTree.FromPolygons(bvh_verts, bvh_polys, epsilon=epsilon)
    
    # BVH同士のオーバーラップから候補ペアを取得
    candidate_pairs = bvh_tree.overlap(bvh_tree)
    
    intersecting_face_indices = set()
    for i, j in candidate_pairs:
        # 重複判定を避けるため i < j の組のみ処理
        if i >= j:
            continue
        face_i = face_map[i]
        face_j = face_map[j]
        # 同じ面の場合は除外
        if face_i == face_j:
            continue
        # 隣接面（頂点を共有している）は除外
        if face_vertex_sets[i].intersection(face_vertex_sets[j]):
            continue
        
        tri1 = triangles[i]
        tri2 = triangles[j]
        if intersect_triangle_triangle(tri1, tri2):
            intersecting_face_indices.add(face_i)
            intersecting_face_indices.add(face_j)
    
    # BMeshをクリーンアップ
    bm.free()
    
    return intersecting_face_indices
