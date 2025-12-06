import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import mathutils


def find_material_index_from_faces(mesh_obj, faces_data):
    """
    面の頂点座標に基づいて該当する面を特定し、マッチした全ての面のマテリアルインデックスの中で
    最も頻度が高いものを返す
    
    Args:
        mesh_obj: Blenderのメッシュオブジェクト
        faces_data: Unityから来た面データのリスト
    
    Returns:
        int: 最も頻度が高いマテリアルインデックス（見つからない場合はNone）
    """
    from collections import Counter
    
    # オブジェクトモードであることを確認
    bpy.context.view_layer.objects.active = mesh_obj
    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # シーンの評価を最新の状態に更新
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    mesh = mesh_obj.data
    
    # ワールド変換行列を取得
    world_matrix = mesh_obj.matrix_world
    
    tolerance = 0.00001  # 座標の許容誤差
    
    # マッチした面のマテリアルインデックスを記録
    matched_material_indices = []
    
    for face_data in faces_data:
        # Unity座標をBlender座標に変換
        unity_vertices = face_data['vertices']
        blender_vertices = []
        
        for unity_vertex in unity_vertices:
            # Unity → Blender座標変換
            blender_vertex = mathutils.Vector((
                -unity_vertex['x'],  # X軸反転
                -unity_vertex['z'],  # Y → Z
                unity_vertex['y']    # Z → Y
            ))
            blender_vertices.append(blender_vertex)
        
        # Blenderの面を検索して一致するものを探す
        for polygon in mesh.polygons:
            if len(polygon.vertices) == 3:  # 三角形面処理
                # 面の頂点のワールド座標を取得
                face_world_verts = []
                for vert_idx in polygon.vertices:
                    vertex = mesh.vertices[vert_idx]
                    world_vert = world_matrix @ vertex.co
                    face_world_verts.append(world_vert)
                
                # 3つの頂点すべてが近い位置にあるかチェック
                match = True
                for i in range(3):
                    closest_dist = min(
                        (face_world_verts[j] - blender_vertices[i]).length 
                        for j in range(3)
                    )
                    if closest_dist > tolerance:
                        match = False
                        break
                
                if match:
                    # マッチした面のマテリアルインデックスを記録
                    material_index = polygon.material_index
                    matched_material_indices.append(material_index)
                    print(f"Found matching triangular face with material index: {material_index}")
                    
            elif len(polygon.vertices) >= 4:  # 多角形面処理
                num_vertices = len(polygon.vertices)
                # 面の頂点のワールド座標を取得
                face_world_verts = []
                for vert_idx in polygon.vertices:
                    vertex = mesh.vertices[vert_idx]
                    world_vert = world_matrix @ vertex.co
                    face_world_verts.append(world_vert)
                
                # 4つの頂点から3つを選ぶ全ての組み合わせをチェック
                from itertools import combinations
                
                for face_vert_combo in combinations(range(num_vertices), 3):
                    # この組み合わせでマッチするかチェック
                    match = True
                    for i in range(3):
                        closest_dist = min(
                            (face_world_verts[face_vert_combo[j]] - blender_vertices[i]).length 
                            for j in range(3)
                        )
                        if closest_dist > tolerance:
                            match = False
                            break
                    
                    if match:
                        # マッチした組み合わせが見つかった
                        material_index = polygon.material_index
                        matched_material_indices.append(material_index)
                        print(f"Found matching face (num_vertices: {num_vertices}) with material index: {material_index}")
                        break  # 同じ面の複数の組み合わせを重複カウントしないように
    
    # マッチした面が見つからない場合
    if not matched_material_indices:
        return None
    
    # 最も頻度が高いマテリアルインデックスを取得
    material_counter = Counter(matched_material_indices)
    most_common_material = material_counter.most_common(1)[0]
    most_common_index = most_common_material[0]
    most_common_count = most_common_material[1]
    
    print(f"Material index frequencies: {dict(material_counter)}")
    print(f"Most common material index: {most_common_index} (appears {most_common_count} times)")
    
    return most_common_index
