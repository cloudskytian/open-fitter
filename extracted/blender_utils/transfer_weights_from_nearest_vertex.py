import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
import time

import bpy
import mathutils
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree


def transfer_weights_from_nearest_vertex(base_mesh, target_obj, vertex_group_name, angle_min=-1.0, angle_max=-1.0, normal_radius=0.0):
    """
    base_meshの指定された頂点グループのウェイトをtarget_objに転写する
    
    target_objの各頂点において、最も近いbase_meshの頂点を取得し、そのウェイト値を設定する
    法線のなす角に基づいてウェイトを調整する
    
    Args:
        base_mesh: ベースメッシュオブジェクト（ウェイトのソース）
        target_obj: ターゲットメッシュオブジェクト（ウェイトの転写先）
        vertex_group_name (str): 転写する頂点グループ名
        angle_min (float): 角度の最小値、この値以下では ウェイト係数0.0（度単位）
        angle_max (float): 角度の最大値、この値以上では ウェイト係数1.0（度単位）
        normal_radius (float): 法線の加重平均を計算する際に考慮する球体の半径
    """
    
    # オブジェクトの検証
    if not base_mesh or base_mesh.type != 'MESH':
        print("エラー: ベースメッシュが指定されていないか、メッシュではありません")
        return
    
    if not target_obj or target_obj.type != 'MESH':
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
    
    print(f"ベースメッシュ '{base_mesh.name}' からターゲットメッシュ '{target_obj.name}' へ頂点グループ '{vertex_group_name}' のウェイトを転写中...")
    
    # モードを確認してオブジェクトモードに切り替え
    original_mode = bpy.context.mode
    if original_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    angle_min_rad = math.radians(angle_min)
    angle_max_rad = math.radians(angle_max)
    
    # BVHツリーを作成（高速な最近傍点検索のため）
    # モディファイア適用後のターゲットメッシュを取得
    body_bm = get_evaluated_mesh(base_mesh)
    body_bm.faces.ensure_lookup_table()

    # ターゲットメッシュのBVHツリーを作成
    bvh_time_start = time.time()
    bvh_tree = BVHTree.FromBMesh(body_bm)
    bvh_time = time.time() - bvh_time_start
    print(f"  BVHツリー作成: {bvh_time:.2f}秒")
    
    # 頂点グループがまだ存在しない場合は作成
    if vertex_group_name not in target_obj.vertex_groups:
        target_obj.vertex_groups.new(name=vertex_group_name)
    target_vertex_group = target_obj.vertex_groups[vertex_group_name]
    
    # モディファイア適用後のソースメッシュを取得
    cloth_bm = get_evaluated_mesh(target_obj)
    cloth_bm.verts.ensure_lookup_table()
    cloth_bm.faces.ensure_lookup_table()
    
    # トランスフォームマトリックスをキャッシュ（繰り返しの計算を避けるため）
    body_normal_matrix = base_mesh.matrix_world.inverted().transposed()
    cloth_normal_matrix = target_obj.matrix_world.inverted().transposed()
    
    # 修正した法線を格納する辞書
    adjusted_normals_time_start = time.time()
    adjusted_normals = {}
    
    # 衣装メッシュの各頂点の法線処理（逆転の必要があるかチェック）
    for i, vertex in enumerate(cloth_bm.verts):
        # ワールド座標系での頂点位置と法線
        cloth_vert_world = vertex.co
        original_normal_world = (cloth_normal_matrix @ Vector((vertex.normal[0], vertex.normal[1], vertex.normal[2], 0))).xyz.normalized()
        
        # 素体メッシュ上の最近傍面を検索
        nearest_result = bvh_tree.find_nearest(cloth_vert_world)
        if nearest_result:
            # BVHTree.find_nearest() は (co, normal, index, distance) を返す
            nearest_point, nearest_normal, nearest_face_index, _ = nearest_result
            
            # 最近傍面を取得
            face = body_bm.faces[nearest_face_index]
            face_normal = face.normal
            
            # 面の法線をワールド座標系に変換
            face_normal_world = (body_normal_matrix @ Vector((face_normal[0], face_normal[1], face_normal[2], 0))).xyz.normalized()
            
            # 内積が負の場合、法線を反転
            dot_product = original_normal_world.dot(face_normal_world)
            if dot_product < 0:
                adjusted_normal = -original_normal_world
            else:
                adjusted_normal = original_normal_world
                 
            # 調整済み法線を辞書に保存
            adjusted_normals[i] = adjusted_normal
        else:
            # 最近傍点が見つからない場合は元の法線を使用
            adjusted_normals[i] = original_normal_world
    adjusted_normals_time = time.time() - adjusted_normals_time_start
    print(f"  法線調整: {adjusted_normals_time:.2f}秒")
    
    # 面の中心点と面積を事前計算してキャッシュ
    face_cache_time_start = time.time()
    face_centers = []
    face_areas = {}
    face_adjusted_normals = {}
    face_indices = []
    
    for face in cloth_bm.faces:
        # 面の中心点を計算
        center = Vector((0, 0, 0))
        for v in face.verts:
            center += v.co
        center /= len(face.verts)
        face_centers.append(center)
        face_indices.append(face.index)
        
        # 面積を計算
        face_areas[face.index] = face.calc_area()
        
        # 面の調整済み法線を計算
        face_normal = Vector((0, 0, 0))
        for v in face.verts:
            face_normal += adjusted_normals[v.index]
        face_adjusted_normals[face.index] = face_normal.normalized()
    face_cache_time = time.time() - face_cache_time_start
    print(f"  面キャッシュ作成: {face_cache_time:.2f}秒")
    
    # 衣装メッシュの面に対してKDTreeを構築
    kdtree_time_start = time.time()
    
    # kd.balance()
    kd = cKDTree(face_centers)
    kdtree_time = time.time() - kdtree_time_start
    print(f"  KDTree構築: {kdtree_time:.2f}秒")
    
    # 各頂点の法線を近傍面の法線の加重平均で更新
    normal_avg_time_start = time.time()
    for i, vertex in enumerate(cloth_bm.verts):
        # 一定の半径内の面を検索
        co = vertex.co
        weighted_normal = Vector((0, 0, 0))
        total_weight = 0
        
        # KDTreeを使用して近傍の面を効率的に検索
        for index in kd.query_ball_point(co, normal_radius):
            # 距離に応じた重みを計算（距離が近いほど影響が大きい）
            face_index = face_indices[index]
            area = face_areas[face_index]
            dist = (co - face_centers[index]).length
            # 距離に基づく減衰係数
            distance_factor = 1.0 - (dist / normal_radius) if dist < normal_radius else 0.0
            weight = area * distance_factor
            
            weighted_normal += face_adjusted_normals[face_index] * weight
            total_weight += weight
        
        # 重みの合計が0でない場合は正規化
        if total_weight > 0:
            weighted_normal /= total_weight
            weighted_normal.normalize()
            # 調整済み法線を更新
            adjusted_normals[i] = weighted_normal
    normal_avg_time = time.time() - normal_avg_time_start
    print(f"  法線加重平均計算: {normal_avg_time:.2f}秒")
    
    # 衣装メッシュの各頂点に対して処理
    weight_calc_time_start = time.time()
    for i, vertex in enumerate(cloth_bm.verts):
        # ワールド座標系での頂点位置
        cloth_vert_world = vertex.co
        
        # 調整済みの法線を使用
        cloth_normal_world = adjusted_normals[i]
        
        # 素体メッシュ上の最近傍面を検索
        nearest_result = bvh_tree.find_nearest(cloth_vert_world)
        distance = float('inf')  # 初期値として無限大を設定

        # 頂点ウェイトの初期値
        weight = 0.0
        
        if nearest_result:
            # BVHTree.find_nearest() は (co, normal, index, distance) を返す
            nearest_point, nearest_normal, nearest_face_index, _ = nearest_result
            
            # 最近傍面を取得
            face = body_bm.faces[nearest_face_index]
            face_normal = face.normal
            
            # 面上の最近接点を計算
            closest_point_on_face = mathutils.geometry.closest_point_on_tri(
                cloth_vert_world,
                face.verts[0].co,
                face.verts[1].co,
                face.verts[2].co
            )

            # base_mesh面上の最近接点の{vertex_group_name}頂点グループのウェイトを線形補完によって計算する
            # 面の3つの頂点を取得
            v0, v1, v2 = face.verts[0], face.verts[1], face.verts[2]
            
            # 各頂点のウェイトを取得
            vg_index = base_vertex_group.index
            w0 = 0.0
            w1 = 0.0
            w2 = 0.0
            
            # base_meshの元のメッシュデータから頂点ウェイトを取得
            base_mesh_data = base_mesh.data
            try:
                for group in base_mesh_data.vertices[v0.index].groups:
                    if group.group == vg_index:
                        w0 = group.weight
                        break
            except (IndexError, KeyError):
                pass
            
            try:
                for group in base_mesh_data.vertices[v1.index].groups:
                    if group.group == vg_index:
                        w1 = group.weight
                        break
            except (IndexError, KeyError):
                pass
            
            try:
                for group in base_mesh_data.vertices[v2.index].groups:
                    if group.group == vg_index:
                        w2 = group.weight
                        break
            except (IndexError, KeyError):
                pass
            
            # 重心座標を計算
            # 三角形の3つの頂点と面上の点から重心座標を求める
            p0 = v0.co
            p1 = v1.co
            p2 = v2.co
            p = closest_point_on_face
            
            v0v1 = p1 - p0
            v0v2 = p2 - p0
            v0p = p - p0
            
            d00 = v0v1.dot(v0v1)
            d01 = v0v1.dot(v0v2)
            d11 = v0v2.dot(v0v2)
            d20 = v0p.dot(v0v1)
            d21 = v0p.dot(v0v2)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) > 1e-8:
                # 重心座標 (u, v, w) を計算
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                
                # 重心座標を使ってウェイトを線形補間
                weight = u * w0 + v * w1 + w * w2
                # ウェイトを0～1の範囲にクランプ
                weight = max(0.0, min(1.0, weight))
            else:
                # 退化した三角形の場合は最も近い頂点のウェイトを使用
                dist0 = (p - p0).length
                dist1 = (p - p1).length
                dist2 = (p - p2).length
                
                if dist0 <= dist1 and dist0 <= dist2:
                    weight = w0
                elif dist1 <= dist2:
                    weight = w1
                else:
                    weight = w2
            
            # 面の法線をワールド座標系に変換
            face_normal_world = (body_normal_matrix @ Vector((face_normal[0], face_normal[1], face_normal[2], 0))).xyz.normalized()
            
            # 距離を計算
            distance = (cloth_vert_world - closest_point_on_face).length
            
            # 最近傍点と法線を設定
            nearest_point = closest_point_on_face
            nearest_normal = face_normal_world
        else:
            # 最近傍点が見つからない場合は初期値をNoneに設定
            nearest_point = None
            nearest_normal = None
        
        if nearest_point:
            # 法線角度に基づくウェイト（線形補間）
            angle_weight = 0.0
            if nearest_normal:
                # 法線の角度を計算
                angle = math.acos(min(1.0, max(-1.0, cloth_normal_world.dot(nearest_normal))))
                
                # 90度以上の場合は法線を反転して再計算
                if angle > math.pi / 2:
                    inverted_normal = -nearest_normal
                    angle = math.acos(min(1.0, max(-1.0, cloth_normal_world.dot(inverted_normal))))
                
                # 角度の線形補間
                if angle <= angle_min_rad:
                    angle_weight = 0.0
                elif angle >= angle_max_rad:
                    angle_weight = 1.0
                else:
                    # 線形補間
                    angle_weight = (angle - angle_min_rad) / (angle_max_rad - angle_min_rad)
            
            weight = weight * angle_weight
        
        # 頂点グループにウェイトを設定
        target_vertex_group.add([i], weight, 'REPLACE')
    weight_calc_time = time.time() - weight_calc_time_start
    print(f"  ウェイト計算: {weight_calc_time:.2f}秒")
    
    # 元のモードに戻す
    if original_mode != 'OBJECT':
        if original_mode.startswith('EDIT'):
            bpy.ops.object.mode_set(mode='EDIT')
