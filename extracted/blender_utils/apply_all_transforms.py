import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_all_transforms():
    """Apply transforms to all objects while maintaining world space positions"""
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 選択状態を保存
    original_selection = {obj: obj.select_get() for obj in bpy.data.objects}
    original_active = bpy.context.view_layer.objects.active
    
    # すべてのオブジェクトを取得し、親子関係の深さでソート
    def get_object_depth(obj):
        depth = 0
        parent = obj.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth
    
    # 深い階層から順番に処理するためにソート
    all_objects = sorted(bpy.data.objects, key=get_object_depth, reverse=True)
    
    # 親子関係情報を保存するリスト
    parent_info_list = []
    
    # 第1段階: すべてのオブジェクトで親子関係を解除してTransformを適用
    for obj in all_objects:
        if obj.type not in {'MESH', 'EMPTY', 'ARMATURE', 'CURVE', 'SURFACE', 'FONT'}:
            continue
        
        # すべての選択を解除
        bpy.ops.object.select_all(action='DESELECT')
        
        # 現在のオブジェクトを選択してアクティブに
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # 親子関係情報を保存
        parent = obj.parent
        parent_type = obj.parent_type
        parent_bone = obj.parent_bone if parent_type == 'BONE' else None
        
        if parent:
            parent_info_list.append({
                'obj': obj,
                'parent': parent,
                'parent_type': parent_type,
                'parent_bone': parent_bone
            })
        
        # 親子関係を一時的に解除（位置は保持）
        if parent:
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        
        # Armatureオブジェクトまたは Armature モディファイアを持つMeshオブジェクトの場合
        has_armature = obj.type == 'ARMATURE' or \
                      (obj.type == 'MESH' and any(mod.type == 'ARMATURE' for mod in obj.modifiers))
        
        if has_armature:
            # すべての Transform を適用
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        else:
            # スケールのみ適用
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 第2段階: すべての親子関係をまとめて復元
    for parent_info in parent_info_list:
        obj = parent_info['obj']
        parent = parent_info['parent']
        parent_type = parent_info['parent_type']
        parent_bone = parent_info['parent_bone']
        
        # すべての選択を解除
        bpy.ops.object.select_all(action='DESELECT')
        
        if parent_type == 'BONE' and parent_bone:
            # ボーン親だった場合
            obj.select_set(True)
            bpy.context.view_layer.objects.active = parent
            parent.select_set(True)
            
            # ポーズモードに切り替えてボーンをアクティブに設定
            bpy.ops.object.mode_set(mode='POSE')
            parent.data.bones.active = parent.data.bones[parent_bone]
            
            # オブジェクトモードに戻る
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # ボーンペアレントを設定
            bpy.ops.object.parent_set(type='BONE', keep_transform=True)
            print(f"Restored bone parent '{parent_bone}' for object '{obj.name}'")
        else:
            # オブジェクト親だった場合
            obj.select_set(True)
            parent.select_set(True)
            bpy.context.view_layer.objects.active = parent
            bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    
    # 元の選択状態を復元
    for obj, was_selected in original_selection.items():
        obj.select_set(was_selected)
    bpy.context.view_layer.objects.active = original_active
