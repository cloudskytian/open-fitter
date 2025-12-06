import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_bone_name_conversion(clothing_armature: bpy.types.Object, clothing_meshes: list, name_conv_data: dict) -> None:
    """
    JSONファイルで指定されたボーンの名前変更マッピングに従って、
    clothing_armatureのボーンとclothing_meshesの頂点グループの名前を変更する
    
    Parameters:
        clothing_armature: 服のアーマチュアオブジェクト
        clothing_meshes: 服のメッシュオブジェクトのリスト
        name_conv_data: ボーン名前変更マッピングのJSONデータ
    """
    if not name_conv_data or 'boneMapping' not in name_conv_data:
        print("ボーン名前変更データが見つかりません")
        return
    
    bone_mappings = name_conv_data['boneMapping']
    renamed_bones = {}
    
    print(f"ボーン名前変更処理を開始: {len(bone_mappings)}個のマッピング")
    
    # 1. アーマチュアのボーン名を変更
    if clothing_armature and clothing_armature.type == 'ARMATURE':
        # Edit modeに入ってボーン名を変更
        bpy.context.view_layer.objects.active = clothing_armature
        bpy.ops.object.mode_set(mode='EDIT')
        
        for mapping in bone_mappings:
            fbx_bone = mapping.get('fbxBone')
            prefab_bone = mapping.get('prefabBone')
            
            if not fbx_bone or not prefab_bone or fbx_bone == prefab_bone:
                continue
                
            # アーマチュア内でfbxBoneに対応するボーンを探す
            if fbx_bone in clothing_armature.data.edit_bones:
                edit_bone = clothing_armature.data.edit_bones[fbx_bone]
                edit_bone.name = prefab_bone
                renamed_bones[fbx_bone] = prefab_bone
                print(f"アーマチュアのボーン名を変更: {fbx_bone} -> {prefab_bone}")
        
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 2. メッシュの頂点グループ名を変更
    for mesh_obj in clothing_meshes:
        if not mesh_obj or mesh_obj.type != 'MESH':
            continue
            
        for mapping in bone_mappings:
            fbx_bone = mapping.get('fbxBone')
            prefab_bone = mapping.get('prefabBone')
            
            if not fbx_bone or not prefab_bone or fbx_bone == prefab_bone:
                continue
                
            # 頂点グループの名前を変更
            if fbx_bone in mesh_obj.vertex_groups:
                vertex_group = mesh_obj.vertex_groups[fbx_bone]
                vertex_group.name = prefab_bone
                print(f"メッシュ {mesh_obj.name} の頂点グループ名を変更: {fbx_bone} -> {prefab_bone}")
    
    print(f"ボーン名前変更処理完了: {len(renamed_bones)}個のボーンが変更されました")
