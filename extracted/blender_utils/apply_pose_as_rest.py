import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_pose_as_rest(armature):
    # アクティブなオブジェクトを保存
    original_active = bpy.context.active_object
    
    # 指定されたアーマチュアを取得
    if not armature or armature.type != 'ARMATURE':
        print(f"Error: {armature.name} is not a valid armature object")
        return
    
    # アーマチュアをアクティブに設定
    bpy.context.view_layer.objects.active = armature
    
    # 編集モードに入る
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    
    # 現在のポーズをレストポーズとして適用
    bpy.ops.pose.armature_apply()
    
    # 元のモードに戻る
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 元のアクティブオブジェクトを復元
    bpy.context.view_layer.objects.active = original_active
