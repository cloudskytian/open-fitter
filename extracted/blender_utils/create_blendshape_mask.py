import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from blender_utils.bone_utils import get_child_bones_recursive


def create_blendshape_mask(target_obj, mask_bones, clothing_avatar_data, field_name="", store_debug_mask=True):
    """
    指定されたボーンとその子ボーンのウェイトを合算したマスクを作成する

    Parameters:
        target_obj: 対象のメッシュオブジェクト
        mask_bones: マスクに使用するHumanoidボーンのリスト
        clothing_avatar_data: 衣装アバターのデータ（Humanoidボーン名の変換に使用）
        field_name: フィールド名（デバッグ用の頂点グループ名に使用）
        store_debug_mask: デバッグ用のマスク頂点グループを保存するかどうか

    Returns:
        numpy.ndarray: 各頂点のマスクウェイト値の配列
    """
    #print(f"mask_bones: {mask_bones}")
    
    mask_weights = np.zeros(len(target_obj.data.vertices))

    # アーマチュアを取得
    armature_obj = None
    for modifier in target_obj.modifiers:
        if modifier.type == 'ARMATURE':
            armature_obj = modifier.object
            break
            
    if not armature_obj:
        print(f"Warning: No armature found for {target_obj.name}")
        return mask_weights

    # Humanoidボーン名からボーン名への変換マップを作成
    humanoid_to_bone = {}
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    
    # 補助ボーンのマッピングを作成
    auxiliary_bones = {}
    for aux_set in clothing_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        auxiliary_bones[humanoid_bone] = aux_set["auxiliaryBones"]
    
    # デバッグ用に処理したボーンの情報を収集
    processed_bones = set()
    
    # 対象となるすべてのボーンを収集（Humanoidボーン、補助ボーン、それらの子ボーン）
    target_bones = set()
    
    # 各Humanoidボーンに対して処理
    for humanoid_bone in mask_bones:
        # メインのボーンを追加
        bone_name = humanoid_to_bone.get(humanoid_bone)
        if bone_name:
            target_bones.add(bone_name)
            processed_bones.add(bone_name)
            # 子ボーンを追加
            target_bones.update(get_child_bones_recursive(bone_name, armature_obj, clothing_avatar_data))
        
        # 補助ボーンとその子ボーンを追加
        if humanoid_bone in auxiliary_bones:
            for aux_bone in auxiliary_bones[humanoid_bone]:
                target_bones.add(aux_bone)
                processed_bones.add(aux_bone)
                # 補助ボーンの子ボーンを追加
                target_bones.update(get_child_bones_recursive(aux_bone, armature_obj, clothing_avatar_data))
    
    #print(f"target_bones: {target_bones}")
    
    # 各頂点のウェイトを計算
    for vert in target_obj.data.vertices:
        for bone_name in target_bones:
            if bone_name in target_obj.vertex_groups:
                group = target_obj.vertex_groups[bone_name]
                for g in vert.groups:
                    if g.group == group.index:
                        mask_weights[vert.index] += g.weight
                        break
    
    # ウェイトを0-1の範囲にクランプ
    mask_weights = np.clip(mask_weights, 0.0, 1.0)
    
    # デバッグ用の頂点グループを作成
    if store_debug_mask:
        # 頂点グループ名を生成
        group_name = f"DEBUG_Mask_{field_name}" if field_name else "DEBUG_Mask"
        
        # 既存のグループがあれば削除
        if group_name in target_obj.vertex_groups:
            target_obj.vertex_groups.remove(target_obj.vertex_groups[group_name])
        
        # 新しいグループを作成
        debug_group = target_obj.vertex_groups.new(name=group_name)
        
        # ウェイトを設定
        for vert_idx, weight in enumerate(mask_weights):
            if weight > 0:
                debug_group.add([vert_idx], weight, 'REPLACE')
        
        print(f"Created debug mask group '{group_name}' using bones: {sorted(processed_bones)}")
    
    return mask_weights
