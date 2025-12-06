import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree


def subdivide_faces(obj, face_indices, cuts=1, max_distance=0.005):
    """
    指定された面（face_indices）からワールド座標系で一定距離以内にある面を細分化します。
    BVHTreeを使用して高速化を行います。
    ※Custom Split Normalsがある場合、細分化前に各頂点の平均カスタム法線を保存し、
      細分化後に各ループの最寄り元法線を補間して再設定します。
    """
    mesh = obj.data
    had_custom_normals = mesh.has_custom_normals

    if not obj or obj.type != 'MESH':
        print("無効なオブジェクトです")
        return

    if len(obj.data.vertices) == 0:
        print("メッシュに頂点がありません")
        return

    # --- 細分化前にCustom Split Normalsを保存（cKDTree版） ---
    orig_normals_per_vertex = {}
    kd = None
    if had_custom_normals:
        # 各ループを1度の走査で、頂点ごとに法線リストを作成
        temp_normals = {i: [] for i in range(len(mesh.vertices))}
        for loop in mesh.loops:
            temp_normals[loop.vertex_index].append(loop.normal)
        for v_idx, normals in temp_normals.items():
            if normals:
                avg = Vector((0.0, 0.0, 0.0))
                for n in normals:
                    avg += n
                if avg.length > 1e-8:
                    avg.normalize()
                orig_normals_per_vertex[v_idx] = avg.copy()
        # 各頂点の座標をNumPy配列にまとめ、cKDTreeを構築
        points = np.array([v.co[:] for v in mesh.vertices])
        kd = cKDTree(points)

    try:
        # --- BMeshを用いた細分化処理 ---
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        
        # ワールド座標系に変換
        bm.transform(obj.matrix_world)
        
        # BVHTreeを構築
        bvh_tree = BVHTree.FromBMesh(bm)

        # 初期対象の面を取得
        initial_faces = {f for f in bm.faces if f.index in face_indices}
        
        # 距離内の面を検索するための対象面のセット
        faces_within_distance = set(initial_faces)
        
        # 各初期対象面からdistance_threshold以内の面を検索
        for f in initial_faces:
            # 面の中心点を計算
            face_center = f.calc_center_median()
            
            # 面の法線とサイズを考慮した検索範囲を設定
            # 面の最大エッジ長を計算してサーチ半径に加算
            max_edge_length = max([e.calc_length() for e in f.edges])
            search_radius = max_edge_length
            if max_edge_length > max_distance:
                search_radius = max_distance
            
            # BVHTreeで近傍の面を検索
            for (location, normal, index, distance) in bvh_tree.find_nearest_range(face_center, search_radius):
                if index is not None and index < len(bm.faces):
                    candidate_face = bm.faces[index]
                    faces_within_distance.add(candidate_face)

        # ワールド座標から元の座標系に戻す
        bm.transform(obj.matrix_world.inverted())

        # 細分化対象のエッジは、距離内の対象面に属するエッジのみ
        all_edges_candidates = {edge for f in faces_within_distance for edge in f.edges}
        
        # エッジの長さが0.004より短いものを除外
        min_edge_length = 0.004
        edges_to_subdivide = []
        
        for edge in all_edges_candidates:
            edge_length = edge.calc_length()
            if edge_length >= min_edge_length:
                edges_to_subdivide.append(edge)

        if edges_to_subdivide:
            bmesh.ops.subdivide_edges(
                bm,
                edges=edges_to_subdivide,
                cuts=cuts,
                use_grid_fill=True,
                use_single_edge=False,
                use_only_quads=True
            )

        # 対象面とその隣接面、さらにその隣接面を一度の走査で取得
        faces_to_check = set(faces_within_distance)
        # 1次隣接面を取得
        first_level_adjacent = set()
        for f in faces_within_distance:
            for edge in f.edges:
                first_level_adjacent.update(edge.link_faces)
        faces_to_check.update(first_level_adjacent)
        
        # 2次隣接面を取得
        for f in first_level_adjacent:
            for edge in f.edges:
                faces_to_check.update(edge.link_faces)

        # 五角形以上のポリゴンを三角形化
        ngons = [f for f in faces_to_check if len(f.verts) > 4]
        if ngons:
            bmesh.ops.triangulate(
                bm,
                faces=ngons,
                quad_method='BEAUTY',
                ngon_method='BEAUTY'
            )

        # BMeshの内容をメッシュに反映
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()
    except Exception as e:
        print(f"細分化中にエラーが発生しました: {e}")

    # --- 細分化後、Custom Split Normalsを再設定（cKDTree使用） ---
    if had_custom_normals and kd is not None:
        new_loop_normals = [None] * len(mesh.loops)
        for i, loop in enumerate(mesh.loops):
            v_index = loop.vertex_index
            v_co = mesh.vertices[v_index].co
            # cKDTreeで最寄りの頂点を検索（距離, インデックスを返す）
            dist, orig_index = kd.query(v_co)
            # 保存しておいた元の法線を取得（なければ現状の法線を使用）
            new_loop_normals[i] = orig_normals_per_vertex.get(orig_index, mesh.vertices[v_index].normal)
        mesh.use_auto_smooth = True
        mesh.normals_split_custom_set(new_loop_normals)
        mesh.update()
