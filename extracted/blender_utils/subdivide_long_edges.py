import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import numpy as np
from mathutils import Vector
from scipy.spatial import cKDTree


def subdivide_long_edges(obj, min_edge_length=0.005, max_edge_length_ratio=2.0, cuts=1):
    """
    指定されたオブジェクトの中央値エッジ長より指定された倍率以上のエッジを細分化します。
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
        bm.edges.ensure_lookup_table()
        
        # 全エッジの長さを計算して中央値を求める
        edge_lengths = []
        for edge in bm.edges:
            if edge.calc_length() >= min_edge_length:
                edge_lengths.append(edge.calc_length())
        
        if not edge_lengths:
            print("エッジが見つかりません")
            bm.free()
            return
            
        # エッジ長をソートして中央値を計算
        edge_lengths.sort()
        n = len(edge_lengths)
        if n % 2 == 0:
            # 偶数個の場合は中央2つの値の平均
            median_edge_length = (edge_lengths[n//2 - 1] + edge_lengths[n//2]) / 2
        else:
            # 奇数個の場合は中央の値
            median_edge_length = edge_lengths[n//2]
            
        threshold_length = median_edge_length * max_edge_length_ratio
        
        print(f"中央値エッジ長: {median_edge_length:.6f}")
        print(f"細分化閾値: {threshold_length:.6f} (中央値の{max_edge_length_ratio}倍)")
        
        # 閾値以上の長さのエッジを特定
        edges_to_subdivide = []
        for edge in bm.edges:
            if edge.calc_length() >= threshold_length:
                edges_to_subdivide.append(edge)
        
        print(f"細分化対象エッジ数: {len(edges_to_subdivide)} / {len(bm.edges)}")

        if edges_to_subdivide:
            bmesh.ops.subdivide_edges(
                bm,
                edges=edges_to_subdivide,
                cuts=cuts,
                use_grid_fill=True,
                use_single_edge=False,
                use_only_quads=False
            )
            print(f"エッジを{cuts}回細分化しました")

        # BMeshの内容をメッシュに反映
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()
    except Exception as e:
        print(f"細分化中にエラーが発生しました: {e}")
        if 'bm' in locals():
            bm.free()

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
