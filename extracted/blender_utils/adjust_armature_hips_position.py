import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from mathutils import Vector


def adjust_armature_hips_position(armature_obj: bpy.types.Object, target_position: Vector, clothing_avatar_data: dict) -> None:
    """
    アーマチュアのHipsボーンを指定された位置に移動させる。
    子オブジェクトのワールド空間での位置を維持する。
    目標位置と現在位置が同じ場合は処理をスキップする。
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
        target_position: 目標とするHipsボーンのワールド座標
        clothing_avatar_data: 衣装のアバターデータ
    """
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return
        
    # Hipsボーンの名前を取得
    hips_bone_name = None
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if bone_map["humanoidBoneName"] == "Hips":
            hips_bone_name = bone_map["boneName"]
            break
            
    if not hips_bone_name:
        print("Warning: Hips bone not found in avatar data")
        return
        
    # 現在のHipsボーンのワールド座標を取得
    pose_bone = armature_obj.pose.bones.get(hips_bone_name)
    if not pose_bone:
        print(f"Warning: Bone {hips_bone_name} not found in armature")
        return
        
    current_position = armature_obj.matrix_world @ pose_bone.head
    
    # 現在位置と目標位置の差を計算
    offset = target_position - current_position
    print(f"Hip Offset: {offset}")
    
    # 位置の差が十分小さい場合は処理をスキップ
    if offset.length < 0.0001:  # 0.1mm未満の差は無視
        print("Hips position is already at target position, skipping adjustment")
        return

    # 現在のアクティブオブジェクトとモードを保存
    current_active = bpy.context.active_object
    current_mode = current_active.mode if current_active else 'OBJECT'
    
    # オブジェクトモードに切り替え
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # アーマチュアの子オブジェクトを取得
    children = []
    for child in bpy.data.objects:
        if child.parent == armature_obj:
            # 子オブジェクトの情報を保存
            children.append(child)
            
    # 親子関係を解除
    for child in children:
        # 他のオブジェクトの選択を解除
        bpy.ops.object.select_all(action='DESELECT')
        
        # 子オブジェクトを選択してアクティブに
        child.select_set(True)
        bpy.context.view_layer.objects.active = child
        
        # 親子関係を解除
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    
    # アーマチュアを移動
    armature_obj.location += offset
    
    # 子オブジェクトの親子関係を復元
    for child in children:
        # 他のオブジェクトの選択を解除
        bpy.ops.object.select_all(action='DESELECT')
        
        # アーマチュアと子オブジェクトを選択
        armature_obj.select_set(True)
        child.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        
        # 親子関係を設定
        bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    
    # 元のアクティブオブジェクトと選択状態を復元
    bpy.ops.object.select_all(action='DESELECT')
    if current_active:
        current_active.select_set(True)
        bpy.context.view_layer.objects.active = current_active
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=current_mode)
    
    # ビューを更新
    bpy.context.view_layer.update()
