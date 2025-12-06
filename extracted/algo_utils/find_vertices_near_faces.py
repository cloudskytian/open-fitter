import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

import bmesh
import bpy
from math_utils.barycentric_coords_from_point import barycentric_coords_from_point
from mathutils.bvhtree import BVHTree


def find_vertices_near_faces(base_mesh, target_mesh, vertex_group_name, max_distance=1.0, max_angle_degrees=None, use_all_faces=False,  smooth_repeat=3):
    """
    ベースメッシュの特定の頂点グループに属する面から指定距離内にあるターゲットメッシュの頂点を見つける、法線の方向を考慮する
    
    Args:
        base_mesh: ベースメッシュオブジェクト（面を構成する頂点が属する頂点グループを持つ）
        target_mesh: ターゲットメッシュオブジェクト（検索対象の頂点を持つ）
        vertex_group_name (str): 検索対象の頂点グループ名（両メッシュで共通）
        max_distance (float): 最大距離
        max_angle_degrees (float): 最大角度 (度)、Noneの場合は法線の方向を考慮しない
        use_all_faces (bool): すべての面を使用するかどうか
        smooth_repeat (int): スムージングの繰り返し回数
    """
    
    # オブジェクトの検証
    if not base_mesh or base_mesh.type != 'MESH':
        print("エラー: ベースメッシュが指定されていないか、メッシュではありません")
        return
    
    if not target_mesh or target_mesh.type != 'MESH':
        print("エラー: ターゲットメッシュが指定されていないか、メッシュではありません")
        return
    
    # ベースメッシュの頂点グループを取得
    base_vertex_group = None
    for vg in base_mesh.vertex_groups:
        if vg.name == vertex_group_name:
            base_vertex_group = vg
            break
    
    if not base_vertex_group:
        print(f"エラー: ベースメッシュに頂点グループ '{vertex_group_name}' が見つかりません")
        return
    
    # 現在のアクティブオブジェクトと選択状態を保存
    original_active = bpy.context.active_object
    original_selected = bpy.context.selected_objects
    original_mode = bpy.context.mode
    
    print(f"ベースメッシュ '{base_mesh.name}' の頂点グループ '{vertex_group_name}' に属する面を分析中...")
    
    # ベースメッシュから対象となる面を抽出
    bpy.ops.object.select_all(action='DESELECT')
    base_mesh.select_set(True)
    bpy.context.view_layer.objects.active = base_mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # ベースメッシュを複製して三角面化
    print("ベースメッシュを複製して三角面化中...")
    bpy.ops.object.duplicate()
    temp_base_mesh = bpy.context.active_object
    temp_base_mesh.name = f"{base_mesh.name}_temp_triangulated"
    
    # 複製したメッシュを三角面化
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 評価後のメッシュデータを取得（三角面化されたベースメッシュ）
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_base_mesh = temp_base_mesh.evaluated_get(depsgraph)
    base_mesh_data = evaluated_base_mesh.data
    base_world_matrix = evaluated_base_mesh.matrix_world
    
    # 元のベースメッシュの頂点グループインデックスを取得
    base_vertex_group_idx = base_vertex_group.index
    
    # 複製メッシュでも同じ頂点グループが存在することを確認
    temp_base_vertex_group = None
    for vg in temp_base_mesh.vertex_groups:
        if vg.name == vertex_group_name:
            temp_base_vertex_group = vg
            break
    
    if not temp_base_vertex_group:
        print(f"エラー: 複製メッシュに頂点グループ '{vertex_group_name}' が見つかりません")
        # 一時メッシュを削除
        bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
        return
    
    # ベースメッシュの頂点グループに属する頂点を取得（評価後のメッシュデータを使用）
    base_vertices_in_group = set()
    for vertex_idx, vertex in enumerate(base_mesh_data.vertices):
        for group_elem in vertex.groups:
            if group_elem.group == temp_base_vertex_group.index and group_elem.weight > 0.001:
                base_vertices_in_group.add(vertex_idx)
                break
    
    print(f"頂点グループに属する頂点数: {len(base_vertices_in_group)}")
    
    # 構成する頂点がすべて頂点グループに属する面を見つける（評価後のメッシュデータを使用）
    target_face_indices = []
    if use_all_faces:
        target_face_indices = [face.index for face in base_mesh_data.polygons]
    else:
        for face in base_mesh_data.polygons:
            if all(vertex_idx in base_vertices_in_group for vertex_idx in face.vertices):
                target_face_indices.append(face.index)
    
    print(f"条件を満たす面数: {len(target_face_indices)} (すべて三角形)")
    
    if not target_face_indices:
        print("警告: 条件を満たす面が見つかりません")
        # 一時メッシュを削除
        bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
        # 元の状態に復元
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selected:
            obj.select_set(True)
        if original_active:
            bpy.context.view_layer.objects.active = original_active
        return
    
    # ターゲットメッシュの頂点グループを作成または取得
    target_vertex_group = None
    if vertex_group_name in target_mesh.vertex_groups:
        target_mesh.vertex_groups.remove(target_mesh.vertex_groups[vertex_group_name])
    target_vertex_group = target_mesh.vertex_groups.new(name=vertex_group_name)
    
    # ターゲットメッシュの各頂点について距離をチェック
    found_vertices = []
    
    # ターゲットメッシュの評価後データも取得
    evaluated_target_mesh = target_mesh.evaluated_get(depsgraph)
    target_mesh_data = evaluated_target_mesh.data
    target_world_matrix = evaluated_target_mesh.matrix_world
    target_normal_matrix = evaluated_target_mesh.matrix_world.inverted().transposed()


    # BVHTreeを使った高速化
    print("BVHTreeを使用して高速検索を実行中...")
    import time
    start_time = time.time()
    
    # 三角面化されたベースメッシュからBVHTreeを構築
    temp_bm = bmesh.new()
    temp_bm.from_mesh(base_mesh_data)
    temp_bm.faces.ensure_lookup_table()
    temp_bm.verts.ensure_lookup_table()
    
    # 対象面の頂点座標と面インデックスを準備
    vertices = []
    faces = []
    
    # すべての頂点を追加（ワールド座標）
    for vert in temp_bm.verts:
        world_vert = base_world_matrix @ vert.co
        vertices.append(world_vert)
    
    # 対象面のみを追加（すべて三角形）
    for face_idx in target_face_indices:
        face = temp_bm.faces[face_idx]
        face_indices = [v.index for v in face.verts]
        faces.append(face_indices)
    
    # BVHTreeを構築
    if faces:  # 面が存在する場合のみ
        bvh = BVHTree.FromPolygons(vertices, faces)
        
        # 各頂点の補間ウェイトを保存する辞書
        vertex_interpolated_weights = {}
        
        for vertex_idx, vertex in enumerate(target_mesh_data.vertices):
            # 頂点のワールド座標（評価後のターゲットメッシュデータを使用）
            world_vertex_pos = target_world_matrix @ vertex.co
            
            nearest_point, normal, face_idx, distance = bvh.find_nearest(world_vertex_pos)

            if max_angle_degrees is not None:
                v = (world_vertex_pos - nearest_point).normalized()
                angle = math.degrees(math.acos(min(1.0, max(-1.0, v.dot(normal)))))
                if angle > max_angle_degrees:
                    vertex_interpolated_weights[vertex_idx] = 0.0
                    continue
            
            # 最も近い面までの距離を取得
            if nearest_point is not None and distance <= max_distance and face_idx is not None:
                found_vertices.append(vertex_idx)
                
                # 面を構成する頂点のインデックスを取得（すべて三角形）
                face_vertex_indices = faces[face_idx]
                
                # 面を構成する頂点のワールド座標を取得
                face_vertices = [vertices[vi] for vi in face_vertex_indices]
                
                # 三角形の重心座標を計算
                bary_coords = barycentric_coords_from_point(nearest_point, face_vertices[0], face_vertices[1], face_vertices[2])
                
                # 各頂点のベースメッシュ頂点グループでのウェイトを取得
                weights = []
                for vi in face_vertex_indices:
                    base_vert = base_mesh_data.vertices[vi]
                    vert_weight = 0.0
                    for group_elem in base_vert.groups:
                        if group_elem.group == temp_base_vertex_group.index:
                            vert_weight = group_elem.weight
                            break
                    weights.append(vert_weight)
                
                # 重心座標で補間
                interpolated_weight = (bary_coords[0] * weights[0] + 
                                     bary_coords[1] * weights[1] + 
                                     bary_coords[2] * weights[2])
                vertex_interpolated_weights[vertex_idx] = max(0.0, min(1.0, interpolated_weight))
            else:
                vertex_interpolated_weights[vertex_idx] = 0.0
    else:
        print("警告: 対象となる面が見つかりません")
        # 一時メッシュを削除
        bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
        return
    
    # 一時的なbmeshを解放
    temp_bm.free()
    
    end_time = time.time()
    print(f"BVHTree検索完了: {end_time - start_time:.3f}秒")
    
    # ターゲットメッシュの頂点グループにウェイトを設定
    for vertex_idx in range(len(target_mesh_data.vertices)):
        weight = vertex_interpolated_weights.get(vertex_idx, 0.0)
        target_vertex_group.add([vertex_idx], weight, 'REPLACE')
    
    bpy.ops.object.select_all(action='DESELECT')
    target_mesh.select_set(True)
    bpy.context.view_layer.objects.active = target_mesh
    
    # Editモードに切り替えて全頂点を選択
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # グループを選択
    for i, group in enumerate(target_mesh.vertex_groups):
        target_mesh.vertex_groups.active_index = i
        if group.name == vertex_group_name:
            break
    
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

    # スムージングを適用
    if smooth_repeat > 0:
        bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=smooth_repeat, expand=0.5)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 一時的に作成した三角面化メッシュを削除
    print(f"一時メッシュ '{temp_base_mesh.name}' を削除中...")
    bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
    
    # 元の状態に復元
    bpy.ops.object.select_all(action='DESELECT')
    for obj in original_selected:
        obj.select_set(True)
    if original_active:
        bpy.context.view_layer.objects.active = original_active
        if original_mode.startswith('EDIT'):
            bpy.ops.object.mode_set(mode='EDIT')
    
    print(f"作成された頂点グループ: {vertex_group_name}")
    print(f"条件を満たした頂点数: {len(found_vertices)}")
    print(f"最大距離: {max_distance}")
