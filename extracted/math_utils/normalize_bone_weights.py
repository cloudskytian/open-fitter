import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def normalize_bone_weights(obj: bpy.types.Object, avatar_data: dict) -> None:
    """
    メッシュのボーン変形に関わる頂点ウェイトを正規化する。
    
    Parameters:
        obj: メッシュオブジェクト
        avatar_data: アバターデータ
    """
    if obj.type != 'MESH':
        return
        
    # 正規化対象のボーングループを取得
    target_groups = set()
    # Humanoidボーンを追加
    for bone_map in avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            target_groups.add(bone_map["boneName"])
    
    # 補助ボーンを追加
    for aux_set in avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set.get("auxiliaryBones", []):
            target_groups.add(aux_bone)
    
    # 各頂点について処理
    for vert in obj.data.vertices:
        # ターゲットグループのウェイト合計を計算
        total_weight = 0.0
        weights = {}
        
        for g in vert.groups:
            group_name = obj.vertex_groups[g.group].name
            if group_name in target_groups:
                total_weight += g.weight
                weights[group_name] = g.weight
        
        # ウェイトの正規化
        for group_name, weight in weights.items():
            normalized_weight = weight / total_weight
            obj.vertex_groups[group_name].add([vert.index], normalized_weight, 'REPLACE')
