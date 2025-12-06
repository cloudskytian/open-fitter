import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from algo_utils.get_deformation_bone_groups import get_deformation_bone_groups


def create_deformation_mask(obj: bpy.types.Object, avatar_data: dict) -> None:
    """
    Create deformation mask vertex group based on avatar data.
    
    Parameters:
        obj: Mesh object to process
        avatar_data: Avatar data containing bone information
    """
    # 入力チェック
    if obj.type != 'MESH':
        print(f"Error: {obj.name} is not a mesh object")
        return
    
    # Get bone groups from avatar data
    group_names = get_deformation_bone_groups(avatar_data)
    
    # TransferMaskという名前の頂点グループがすでに存在する場合は削除
    if "DeformationMask" in obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups["DeformationMask"])
    
    # 新しい頂点グループを作成
    deformation_mask = obj.vertex_groups.new(name="DeformationMask")
    
    # 各頂点をチェック
    for vert in obj.data.vertices:
        should_add = False
        weight_sum = 0.0
        # 指定された頂点グループのウェイトをチェック
        for group_name in group_names:
            try:
                group = obj.vertex_groups[group_name]
                # その頂点のウェイト値を取得
                weight = 0
                for g in vert.groups:
                    if g.group == group.index:
                        weight = g.weight
                # ウェイトが0より大きければフラグを立てる
                if weight > 0:
                    should_add = True
                    weight_sum += weight
            except KeyError:
                # 頂点グループが存在しない場合はスキップ
                continue
        
        # フラグが立っている場合、DeformationMaskグループに頂点を追加
        if should_add:
            deformation_mask.add([vert.index], weight_sum, 'REPLACE')
