import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from mathutils.bvhtree import BVHTree


def create_distance_falloff_transfer_mask(obj: bpy.types.Object, 
                                        base_avatar_data: dict,
                                        group_name: str = "DistanceFalloffMask",
                                        max_distance: float = 0.025,
                                        min_distance: float = 0.002) -> bpy.types.VertexGroup:
    """
    距離に基づいて減衰するTransferMask頂点グループを作成
    
    Parameters:
        obj: 対象のメッシュオブジェクト
        base_avatar_data: ベースアバターのデータ
        group_name: 生成する頂点グループの名前（デフォルト: "DistanceFalloffMask"）
        max_distance: ウェイトが0になる最大距離（デフォルト: 0.025）
        min_distance: ウェイトが1になる最小距離（デフォルト: 0.002）
        
    Returns:
        bpy.types.VertexGroup: 生成された頂点グループ
    """
    # 入力チェック
    if obj.type != 'MESH':
        print(f"Error: {obj.name} is not a mesh object")
        return None

    # ソースオブジェクト(Body.BaseAvatar)の取得
    source_obj = bpy.data.objects.get("Body.BaseAvatar")
    if not source_obj:
        print("Error: Body.BaseAvatar not found")
        return None

    # モディファイア適用後のターゲットメッシュを取得
    target_bm = get_evaluated_mesh(source_obj)
    target_bm.faces.ensure_lookup_table()

    # ターゲットメッシュのBVHツリーを作成
    bvh = BVHTree.FromBMesh(target_bm)

    # モディファイア適用後のソースメッシュを取得
    source_bm = get_evaluated_mesh(obj)
    source_bm.verts.ensure_lookup_table()

    # 新しい頂点グループを作成
    transfer_mask = obj.vertex_groups.new(name=group_name)

    # 各頂点を処理
    for vert_idx, vert in enumerate(obj.data.vertices):

        # モディファイア適用後の頂点位置を使用
        evaluated_vertex_co = source_bm.verts[vert_idx].co

        # 最近接点と法線を取得
        location, normal, index, distance = bvh.find_nearest(evaluated_vertex_co)

        if location is not None:
            # 距離に基づいてベースウェイトを計算
            if distance > max_distance:
                weight = 0.0
            else:
                d = distance - min_distance
                if d < 0.0:
                    d = 0.0
                weight = 1.0 - d / (max_distance - min_distance)

        # 頂点グループに追加
        transfer_mask.add([vert_idx], weight, 'REPLACE')

    # BMeshをクリーンアップ
    source_bm.free()
    target_bm.free()

    return transfer_mask
