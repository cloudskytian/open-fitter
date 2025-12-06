import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json

import bpy
from blender_utils.adjust_armature_hips_position import adjust_armature_hips_position


def process_clothing_avatar(input_fbx, clothing_avatar_data_path, hips_position=None, target_meshes=None, mesh_renderers=None):
    """Process clothing avatar."""
    
    original_active = bpy.context.view_layer.objects.active
    
    # Import clothing FBX
    bpy.ops.import_scene.fbx(filepath=input_fbx, use_anim=False)
    
    # 非アクティブなオブジェクトとその子を削除
    def remove_inactive_objects():
        """非アクティブなオブジェクトとそのすべての子を削除する"""
        objects_to_remove = []
        
        def is_object_inactive(obj):
            """オブジェクトが非アクティブかどうかを判定"""
            # hide_viewport または hide_render が True の場合、非アクティブと判定
            return obj.hide_viewport or obj.hide_render or obj.hide_get()
        
        def collect_children_recursive(obj, collected_list):
            """オブジェクトのすべての子を再帰的に収集"""
            for child in obj.children:
                collected_list.append(child)
                collect_children_recursive(child, collected_list)
        
        # 非アクティブなオブジェクトを探す
        for obj in bpy.data.objects:
            if is_object_inactive(obj) and obj not in objects_to_remove:
                objects_to_remove.append(obj)
                # すべての子も収集
                collect_children_recursive(obj, objects_to_remove)
        
        # 重複を削除
        objects_to_remove = list(set(objects_to_remove))
        
        # オブジェクトを削除
        for obj in objects_to_remove:
            obj_name = obj.name
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                print(f"Removed inactive object: {obj_name}")
            except Exception as e:
                print(f"Failed to remove object {obj_name}: {e}")
    
    remove_inactive_objects()
    
    # Load clothing avatar data
    print(f"Loading clothing avatar data from {clothing_avatar_data_path}")
    with open(clothing_avatar_data_path, 'r', encoding='utf-8') as f:
        clothing_avatar_data = json.load(f)
    
    # Find clothing armature
    clothing_armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj.name != "Armature.BaseAvatar":
            clothing_armature = obj
            break
    
    if not clothing_armature:
        raise Exception("Clothing armature not found")
    
    # Find clothing meshes
    clothing_meshes = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != "Body.BaseAvatar" and obj.name != "Body.BaseAvatar.RightOnly" and obj.name != "Body.BaseAvatar.LeftOnly":
            # Check if this mesh has an armature modifier
            has_armature = False
            for modifier in obj.modifiers:
                if modifier.type == 'ARMATURE':
                    has_armature = True
                    break
            
            # 頂点数が0のメッシュを除外
            if has_armature and len(obj.data.vertices) > 0:
                clothing_meshes.append(obj)
            elif has_armature and len(obj.data.vertices) == 0:
                print(f"Skipping mesh '{obj.name}': vertex count is 0")
    
    # フィルタリング: target_meshesが指定されている場合、それに含まれるメッシュのみを保持
    if target_meshes:
        target_mesh_list = [name for name in target_meshes.split(';')]
        print(f"Target mesh list: {target_mesh_list}")
        filtered_meshes = []
        for obj in clothing_meshes:
            if obj.name in target_mesh_list:
                filtered_meshes.append(obj)
            else:
                # 対象外のメッシュを削除
                obj_name = obj.name
                bpy.data.objects.remove(obj, do_unlink=True)
                print(f"Removed non-target mesh: {obj_name}")
        
        if not filtered_meshes:
            raise Exception(f"No target meshes found. Specified: {target_meshes}")
        
        clothing_meshes = filtered_meshes
    
    # Set hips position if provided
    if hips_position:
        adjust_armature_hips_position(clothing_armature, hips_position, clothing_avatar_data)
    
    # Process mesh renderers if provided
    if mesh_renderers:
        print(f"Processing mesh renderers: {mesh_renderers}")
        for mesh_name, parent_name in mesh_renderers.items():
            # MeshRendererを持っていたオブジェクトと同じ名前を持つメッシュオブジェクトを探す
            mesh_obj = None
            for obj in bpy.data.objects:
                if obj.type == 'MESH' and obj.name == mesh_name:
                    mesh_obj = obj
                    break
            
            if mesh_obj:
                # Armatureモディファイアを持たず、親の名前がデータ内の親オブジェクトの名前と異なるかチェック
                has_armature = False
                for modifier in mesh_obj.modifiers:
                    if modifier.type == 'ARMATURE':
                        has_armature = True
                        break
                
                current_parent_name = mesh_obj.parent.name if mesh_obj.parent else None
                
                if not has_armature and current_parent_name != parent_name:
                    # データ内の親オブジェクトの名前と同じ名前を持つボーンをclothing_armatureから探す
                    bone_found = False
                    if parent_name in clothing_armature.data.bones:
                        # ボーンが見つかった場合、そのボーンをメッシュオブジェクトの親にする
                        # すべての選択を解除
                        bpy.ops.object.select_all(action='DESELECT')
                        
                        # メッシュを選択
                        mesh_obj.select_set(True)
                        
                        # アーマチュアをアクティブに設定
                        bpy.context.view_layer.objects.active = clothing_armature
                        clothing_armature.select_set(True)
                        
                        # ポーズモードに切り替えてボーンをアクティブに設定
                        bpy.ops.object.mode_set(mode='POSE')
                        clothing_armature.data.bones.active = clothing_armature.data.bones[parent_name]
                        
                        # オブジェクトモードに戻る
                        bpy.ops.object.mode_set(mode='OBJECT')
                        
                        # ボーンペアレントを設定（keep_transformでワールド座標を保持）
                        bpy.ops.object.parent_set(type='BONE', keep_transform=True)
                        
                        print(f"Set parent bone '{parent_name}' for mesh '{mesh_name}' (world transform preserved)")
                        bone_found = True
                        
                        # 選択を解除
                        bpy.ops.object.select_all(action='DESELECT')
                    
                    if not bone_found:
                        print(f"Warning: Bone '{parent_name}' not found in clothing_armature for mesh '{mesh_name}'")
                else:
                    if has_armature:
                        print(f"Skipping mesh '{mesh_name}': already has Armature modifier")
                    else:
                        print(f"Skipping mesh '{mesh_name}': parent already matches ('{current_parent_name}')")
            else:
                print(f"Warning: Mesh object '{mesh_name}' not found")
    
    bpy.context.view_layer.objects.active = original_active
    
    return clothing_meshes, clothing_armature, clothing_avatar_data
