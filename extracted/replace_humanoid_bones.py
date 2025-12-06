import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Optional

import bmesh
import bpy
from add_pose_from_json import add_pose_from_json
from algo_utils.find_humanoid_parent_in_hierarchy import (
    find_humanoid_parent_in_hierarchy,
)
from algo_utils.get_humanoid_and_auxiliary_bone_groups_with_intermediate import (
    get_humanoid_and_auxiliary_bone_groups_with_intermediate,
)
from blender_utils.apply_pose_as_rest import apply_pose_as_rest
from blender_utils.inverse_bone_deform_all_vertices import (
    inverse_bone_deform_all_vertices,
)
from math_utils.copy_bone_transform import copy_bone_transform
from mathutils.bvhtree import BVHTree


def replace_humanoid_bones(base_armature: bpy.types.Object, clothing_armature: bpy.types.Object, 
                        base_avatar_data: dict, clothing_avatar_data: dict, preserve_humanoid_bones: bool, base_pose_filepath: Optional[str], clothing_meshes: list, process_upper_chest: bool) -> None:
   
    current_active = bpy.context.active_object
    current_mode = current_active.mode if current_active else 'OBJECT'

    # Create mappings
    base_humanoid_map = {bone_map["humanoidBoneName"]: bone_map["boneName"] 
                        for bone_map in base_avatar_data["humanoidBones"]}
    clothing_humanoid_map = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                            for bone_map in clothing_avatar_data["humanoidBones"]}
    clothing_bones_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                                for bone_map in clothing_avatar_data["humanoidBones"]}
    
    # Create reverse mapping for finding bones by humanoid names
    base_bone_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                            for bone_map in base_avatar_data["humanoidBones"]}

    # Humanoidボーンの照合
    clothing_humanoid_bones = {bone_map["humanoidBoneName"] for bone_map in clothing_avatar_data["humanoidBones"]}
    base_humanoid_bones = {bone_map["humanoidBoneName"] for bone_map in base_avatar_data["humanoidBones"]}
    
    # base_avatar_dataに存在しないHumanoidボーンを特定
    # missing_humanoid_bones = clothing_humanoid_bones - base_humanoid_bones
    missing_humanoid_bones = {}
    
    # Map auxiliary bones to humanoid bones
    aux_to_humanoid = {}
    for aux_set in clothing_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        # base_avatar_dataに存在しないHumanoidボーンの補助ボーンは除外
        if humanoid_bone not in missing_humanoid_bones:
            for aux_bone in aux_set["auxiliaryBones"]:
                aux_to_humanoid[aux_bone] = humanoid_bone

    # Map humanoid bones to auxiliary bones
    humanoid_to_aux = {}
    for aux_set in clothing_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        # base_avatar_dataに存在しないHumanoidボーンの補助ボーンは除外
        if humanoid_bone not in missing_humanoid_bones:
            humanoid_to_aux[humanoid_bone] = aux_set["auxiliaryBones"]
    
    humanoid_to_aux_base = {}
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_to_aux_base[aux_set["humanoidBoneName"]] = aux_set["auxiliaryBones"]

    # bones_to_replaceからbase_avatar_dataに存在しないHumanoidボーンとその補助ボーンを除外
    bones_to_replace = set()
    for bone_map in clothing_avatar_data["humanoidBones"]:
        if bone_map["humanoidBoneName"] not in missing_humanoid_bones:
            bones_to_replace.add(bone_map["boneName"])
    
    for aux_set in clothing_avatar_data.get("auxiliaryBones", []):
        if aux_set["humanoidBoneName"] not in missing_humanoid_bones:
            bones_to_replace.update(aux_set["auxiliaryBones"])

    print(f"bones_to_replace: {bones_to_replace}")
    
    base_bones = get_humanoid_and_auxiliary_bone_groups_with_intermediate(base_armature, base_avatar_data)

    # Get humanoid bones that should be preserved
    if preserve_humanoid_bones:
        humanoid_bones_to_preserve = {bone_name for bone_name, humanoid_name 
                                    in clothing_bones_to_humanoid.items() 
                                    if humanoid_name not in missing_humanoid_bones}
    else:
        humanoid_bones_to_preserve = set()

    # Get base mesh and create BVH tree
    base_mesh = bpy.data.objects.get("Body.BaseAvatar")
    if not base_mesh:
        raise Exception("Body.BaseAvatar not found")
        
    bm = bmesh.new()
    bm.from_mesh(base_mesh.data)
    bm.faces.ensure_lookup_table()
    bm.transform(base_mesh.matrix_world)
    bvh = BVHTree.FromBMesh(bm)

    # Armatureモディファイアの設定を保存して一時的に削除
    armature_modifiers = []
    clothing_obj_list = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for modifier in obj.modifiers[:]:  # リストのコピーを使用
                if modifier.type == 'ARMATURE' and modifier.object == clothing_armature:
                    mod_settings = {
                        'object': obj,
                        'name': modifier.name,
                        'target': modifier.object,
                        'vertex_group': modifier.vertex_group,
                        'invert_vertex_group': modifier.invert_vertex_group,
                        'use_vertex_groups': modifier.use_vertex_groups,
                        'use_bone_envelopes': modifier.use_bone_envelopes,
                        'use_deform_preserve_volume': modifier.use_deform_preserve_volume
                    }
                    armature_modifiers.append(mod_settings)
                    obj.modifiers.remove(modifier)
                    clothing_obj_list.append(obj)

    if base_pose_filepath:
        print(f"Applying clothing base pose from {base_pose_filepath}")
        add_pose_from_json(clothing_armature, base_pose_filepath, clothing_avatar_data, invert=True)
        apply_pose_as_rest(clothing_armature)
    
    # Get clothing bone positions and their original parents
    clothing_bone_data = {}
    clothing_matrix_world = clothing_armature.matrix_world
    
    for bone in clothing_armature.pose.bones:
        if bone.parent and bone.parent.name in bones_to_replace and bone.name not in bones_to_replace:
            head_pos = clothing_matrix_world @ bone.head
            
            # まずbone.parentのhumanoid名を取得
            parent_humanoid = None
            if bone.parent.name in clothing_humanoid_map:
                parent_humanoid = clothing_humanoid_map[bone.parent.name]
            elif bone.parent.name in aux_to_humanoid:
                parent_humanoid = aux_to_humanoid[bone.parent.name]
            
            # もしparent_humanoidがbase_humanoid_mapに存在しない場合は
            # clothing_avatar_dataで親を辿ってbase_avatar_dataにも存在する最初のhumanoidボーンを探す
            if parent_humanoid and parent_humanoid not in base_humanoid_map:
                # parent_humanoid = find_humanoid_parent_in_clothing(bone.parent.name, clothing_bones_to_humanoid, clothing_armature)
                parent_humanoid = find_humanoid_parent_in_hierarchy(bone.parent.name, clothing_avatar_data, base_avatar_data)

            if parent_humanoid and parent_humanoid in base_humanoid_map:
                # 候補ボーンを取得
                candidate_bones = {base_humanoid_map[parent_humanoid]}
                if parent_humanoid in humanoid_to_aux_base:
                    candidate_bones.update(humanoid_to_aux_base[parent_humanoid])
                
                # ChestでUpperChestが存在する場合の処理
                sub_parent_humanoid = None
                if parent_humanoid == 'Chest' and 'UpperChest' in base_humanoid_map and process_upper_chest:
                    sub_parent_humanoid = base_humanoid_map['UpperChest']
                    candidate_bones.add(sub_parent_humanoid)
                    if 'UpperChest' in humanoid_to_aux_base:
                        candidate_bones.update(humanoid_to_aux_base['UpperChest'])
                
                clothing_bone_data[bone.name] = {
                    'head_pos': head_pos,
                    'candidate_bones': candidate_bones,
                    'parent_humanoid': base_humanoid_map[parent_humanoid],
                    'sub_parent_humanoid': sub_parent_humanoid
                }

    base_group_index_to_name = {group.index: group.name for group in base_mesh.vertex_groups}

    # Find parent bones using only the candidate bones
    parent_bones = {}
    for bone_name, data in clothing_bone_data.items():
        head_pos = data['head_pos']
        candidate_bones = data['candidate_bones']
        parent_humanoid = data['parent_humanoid']
        sub_parent_humanoid = data.get('sub_parent_humanoid', None)

        # 追加手法: clothing_meshesから対象ボーンのウェイトが一定以上の頂点を取得し、それらに近いbase_mesh頂点から
        # candidate_bonesのウェイトスコアを集計する
        bone_scores = defaultdict(float)

        weighted_vertices = []
        for mesh_obj in clothing_meshes:
            if mesh_obj.type != 'MESH':
                continue

            vg_lookup = {vg.name: vg.index for vg in mesh_obj.vertex_groups}
            if bone_name not in vg_lookup:
                continue

            target_group_index = vg_lookup[bone_name]
            mesh_data = mesh_obj.data

            mesh_world_matrix = mesh_obj.matrix_world

            for vertex in mesh_data.vertices:
                weight = 0.0
                for g in vertex.groups:
                    if g.group == target_group_index:
                        weight = g.weight
                        break
                if weight >= 0.001:
                    vertex_world_co = mesh_world_matrix @ vertex.co
                    weighted_vertices.append((vertex_world_co, weight))
            print(f"bone_name: {bone_name}, weighted_vertices: {len(weighted_vertices)}")

        if weighted_vertices:
            weighted_vertices.sort(key=lambda item: item[1], reverse=True)
            top_vertices = weighted_vertices[:100]

            for vertex_world_co, _ in top_vertices:
                closest_point, _, face_idx, _ = bvh.find_nearest(vertex_world_co)
                if closest_point is None or face_idx is None:
                    continue

                face = bm.faces[face_idx]
                vertex_indices = [v.index for v in face.verts]
                closest_vert_idx = min(
                    vertex_indices,
                    key=lambda idx: (base_mesh.data.vertices[idx].co - closest_point).length
                )

                vertex = base_mesh.data.vertices[closest_vert_idx]
                for group_element in vertex.groups:
                    group_name = base_group_index_to_name.get(group_element.group)
                    if group_name in candidate_bones:
                        bone_scores[group_name] += group_element.weight

        chosen_parent = None
        if bone_scores:
            print(f"bone_scores: {bone_scores}")
            chosen_parent = max(bone_scores.items(), key=lambda item: item[1])[0]

        # if not chosen_parent:
        #     # 既存手法: 頂点距離とボーン距離を使用する
        #     closest_point, normal, face_idx, vertex_distance = bvh.find_nearest(head_pos)
        #     closest_bone = None
        #     min_vertex_weight_distance = float('inf')

        #     if closest_point and face_idx is not None:
        #         face = bm.faces[face_idx]
        #         vertex_indices = [v.index for v in face.verts]
        #         closest_vert_idx = min(vertex_indices,
        #                             key=lambda idx: (base_mesh.data.vertices[idx].co - closest_point).length)

        #         max_weight = 0
        #         vertex = base_mesh.data.vertices[closest_vert_idx]
        #         for group_element in vertex.groups:
        #             group_name = base_group_index_to_name.get(group_element.group)
        #             if group_name in candidate_bones:
        #                 weight = group_element.weight
        #                 if weight > max_weight:
        #                     max_weight = weight
        #                     closest_bone = group_name
        #                     min_vertex_weight_distance = vertex_distance

        #     min_bone_distance = float('inf')
        #     closest_bone_by_distance = None

        #     for bone in base_armature.pose.bones:
        #         if bone.name in candidate_bones:
        #             bone_head_world = base_armature.matrix_world @ bone.head
        #             distance = (head_pos - bone_head_world).length
        #             if distance < min_bone_distance:
        #                 min_bone_distance = distance
        #                 closest_bone_by_distance = bone.name

        #     if closest_bone_by_distance and min_bone_distance < min_vertex_weight_distance:
        #         chosen_parent = closest_bone_by_distance
        #     elif closest_bone:
        #         chosen_parent = closest_bone

        if chosen_parent and chosen_parent == bone_name and bone_name in clothing_armature.data.bones and clothing_armature.data.bones.get(bone_name).parent:
            chosen_parent = clothing_armature.data.bones.get(bone_name).parent.name
            if chosen_parent not in candidate_bones:
                chosen_parent = None
        
        if chosen_parent:
            parent_bones[bone_name] = chosen_parent
            print(f"bone_name: {bone_name}, chosen_parent: {chosen_parent}")
        else:
            # chosen_parentが見つからない場合、sub_parent_humanoidがあれば距離を比較
            if sub_parent_humanoid:
                parent_humanoid_bone = base_armature.pose.bones.get(parent_humanoid)
                sub_parent_humanoid_bone = base_armature.pose.bones.get(sub_parent_humanoid)
                
                if parent_humanoid_bone and sub_parent_humanoid_bone:
                    parent_distance = (head_pos - (base_armature.matrix_world @ parent_humanoid_bone.head)).length
                    sub_parent_distance = (head_pos - (base_armature.matrix_world @ sub_parent_humanoid_bone.head)).length
                    
                    if sub_parent_distance < parent_distance:
                        parent_bones[bone_name] = sub_parent_humanoid
                        print(f"bone_name: {bone_name}, chosen_parent: {sub_parent_humanoid} (sub_parent, distance: {sub_parent_distance:.4f})")
                    else:
                        parent_bones[bone_name] = parent_humanoid
                        print(f"bone_name: {bone_name}, chosen_parent: {parent_humanoid} (fallback, distance: {parent_distance:.4f})")
                else:
                    parent_bones[bone_name] = parent_humanoid
                    print(f"bone_name: {bone_name}, chosen_parent: {parent_humanoid} (fallback)")
            else:
                parent_bones[bone_name] = parent_humanoid
                print(f"bone_name: {bone_name}, chosen_parent: {parent_humanoid} (fallback)")

    bm.free()

    # Replace bones
    bpy.context.view_layer.objects.active = clothing_armature
    bpy.ops.object.mode_set(mode='EDIT')
    clothing_edit_bones = clothing_armature.data.edit_bones

    # Store children to update
    children_to_update = []
    for bone in clothing_edit_bones:
        if bone.parent and bone.parent.name in bones_to_replace and bone.name not in bones_to_replace:
            children_to_update.append(bone.name)

    # Store base bone parents
    base_bone_parents = {}
    bpy.context.view_layer.objects.active = base_armature
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in base_armature.data.edit_bones:
        if bone.name in base_bones:
            base_bone_parents[bone.name] = bone.parent.name if bone.parent and bone.parent.name in base_bones else None

    print(base_bone_parents)

    bpy.context.view_layer.objects.active = clothing_armature
    bpy.ops.object.mode_set(mode='EDIT')

    # Process bones to preserve or delete
    original_bone_data = {}
    for bone_name in bones_to_replace:
        if bone_name in clothing_edit_bones:
            if bone_name in humanoid_bones_to_preserve:
                # Preserve and rename Humanoid bones
                orig_bone = clothing_edit_bones[bone_name]
                new_name = f"origORS_{bone_name}"
                bone_data = {
                    'head': orig_bone.head.copy(),
                    'tail': orig_bone.tail.copy(),
                    'roll': orig_bone.roll,
                    'matrix': orig_bone.matrix.copy(),
                    'new_name': new_name,
                    'humanoid_name': clothing_bones_to_humanoid[bone_name]  # Store the humanoid name
                }
                
                original_bone_data[bone_name] = bone_data
                orig_bone.name = new_name
            else:
                # Delete non-Humanoid bones
                clothing_edit_bones.remove(clothing_edit_bones[bone_name])

    # Create new bones
    new_bones = {}
    for bone_name in base_bones:
        source_bone = base_armature.data.edit_bones.get(bone_name)
        if source_bone:
            new_bone = clothing_edit_bones.new(name=bone_name)
            copy_bone_transform(source_bone, new_bone)
            new_bones[bone_name] = new_bone

    # Set parent relationships for new bones
    for bone_name, new_bone in new_bones.items():
        parent_name = base_bone_parents.get(bone_name)
        if parent_name and parent_name in new_bones:
            new_bone.parent = new_bones[parent_name]

    # Make original humanoid bones children of new bones based on boneHierarchy
    for orig_bone_name, data in original_bone_data.items():
        orig_bone = clothing_edit_bones[data['new_name']]
        humanoid_name = data['humanoid_name']  # Get the humanoid name for matching
        
        # Find parent using boneHierarchy
        parent_humanoid_name = find_humanoid_parent_in_hierarchy(orig_bone_name, clothing_avatar_data, base_avatar_data)
        
        if parent_humanoid_name:
            # Find the new bone with matching parent humanoid name
            matched_new_bone = None
            for new_bone_name, new_bone in new_bones.items():
                if new_bone_name in base_bone_to_humanoid:
                    if base_bone_to_humanoid[new_bone_name] == parent_humanoid_name:
                        matched_new_bone = new_bone
                        break
            
            if matched_new_bone:
                orig_bone.parent = matched_new_bone
            else:
                print(f"Warning: No matching new bone found for parent humanoid bone {parent_humanoid_name}")
        else:
            # Fallback to original matching logic
            matched_new_bone = None
            for new_bone_name, new_bone in new_bones.items():
                if new_bone_name in base_bone_to_humanoid:
                    if base_bone_to_humanoid[new_bone_name] == humanoid_name:
                        matched_new_bone = new_bone
                        break
            
            if matched_new_bone:
                orig_bone.parent = matched_new_bone
            else:
                print(f"Warning: No matching new bone found for humanoid bone {humanoid_name}")

    # parent_boneがHumanoidBoneであり、subHumanoidBonesのHumanoidBoneNameに一致するものが存在する場合、subHumanoidBoneの方に入れ替える
    if "subHumanoidBones" in base_avatar_data:
        sub_humanoid_bones = {}
        for sub_humanoid_bone in base_avatar_data["subHumanoidBones"]:
            sub_humanoid_bones[sub_humanoid_bone["humanoidBoneName"]] = sub_humanoid_bone["boneName"]
        for bone_name, parent_name in parent_bones.items():
            if parent_name in base_bone_to_humanoid:
                if base_bone_to_humanoid[parent_name] in sub_humanoid_bones.keys():
                    parent_bones[bone_name] = sub_humanoid_bones[base_bone_to_humanoid[parent_name]]
    
    # Update children parents
    for child_name in children_to_update:
        child_bone = clothing_edit_bones.get(child_name)
        print(f"child_name: {child_name}, child_bone: {child_bone.name if child_bone else None}")
        if child_bone:
            new_parent_name = parent_bones.get(child_name)
            print(f"child_name: {child_name}, new_parent_name: {new_parent_name}")
            if new_parent_name and new_parent_name in clothing_edit_bones:
                child_bone.parent = clothing_edit_bones[new_parent_name]
                print(f"child_name: {child_name}, new_parent_name: {new_parent_name}")

    bpy.ops.object.mode_set(mode='OBJECT')

    if base_pose_filepath:
        print(f"Applying base pose from {base_pose_filepath}")
        add_pose_from_json(clothing_armature, base_pose_filepath, base_avatar_data, invert=False)
        for obj in clothing_obj_list:
            inverse_bone_deform_all_vertices(clothing_armature, obj)
        add_pose_from_json(clothing_armature, base_pose_filepath, base_avatar_data, invert=True)
        apply_pose_as_rest(clothing_armature)
    
    # Armatureモディファイアを復元
    for mod_settings in armature_modifiers:
        obj = mod_settings['object']
        modifier = obj.modifiers.new(name=mod_settings['name'], type='ARMATURE')
        modifier.object = mod_settings['target']
        modifier.vertex_group = mod_settings['vertex_group']
        modifier.invert_vertex_group = mod_settings['invert_vertex_group']
        modifier.use_vertex_groups = mod_settings['use_vertex_groups']
        modifier.use_bone_envelopes = mod_settings['use_bone_envelopes']
        modifier.use_deform_preserve_volume = mod_settings['use_deform_preserve_volume']

    bpy.context.view_layer.objects.active = current_active
    if current_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=current_mode)
