import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
import time
from collections import defaultdict, deque

import bmesh
import bpy
import mathutils
import numpy as np
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from apply_distance_normal_based_smoothing import apply_distance_normal_based_smoothing
from blender_utils.adjust_hand_weights import adjust_hand_weights
from blender_utils.create_blendshape_mask import create_blendshape_mask
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from blender_utils.merge_weights_to_parent import merge_weights_to_parent
from blender_utils.reset_bone_weights import reset_bone_weights
from create_distance_normal_based_vertex_group import (
    create_distance_normal_based_vertex_group,
)
from create_side_weight_groups import create_side_weight_groups
from io_utils.restore_shape_key_state import restore_shape_key_state
from io_utils.restore_weights import restore_weights
from io_utils.save_shape_key_state import save_shape_key_state
from io_utils.store_weights import store_weights
from math_utils.create_distance_falloff_transfer_mask import (
    create_distance_falloff_transfer_mask,
)
from scipy.spatial import cKDTree


def process_weight_transfer(target_obj, armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, cloth_metadata=None):
    """Process weight transfer for the target object."""
    start_time = time.time()

    # Humanoidボーン名からボーン名への変換マップを作成
    humanoid_to_bone = {}
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    
    # 補助ボーンのマッピングを作成
    auxiliary_bones = {}
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        auxiliary_bones[humanoid_bone] = aux_set["auxiliaryBones"]

    auxiliary_bones_to_humanoid = {}
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set["auxiliaryBones"]:
            auxiliary_bones_to_humanoid[aux_bone] = aux_set["humanoidBoneName"]

    finger_humanoid_bones = [
        "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
        "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
        "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
        "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
        "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
        "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
        "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
        "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
        "LeftHand", "RightHand"
    ]

    left_foot_finger_humanoid_bones = [
        "LeftFootThumbProximal",
        "LeftFootThumbIntermediate", 
        "LeftFootThumbDistal",
        "LeftFootIndexProximal",
        "LeftFootIndexIntermediate",
        "LeftFootIndexDistal",
        "LeftFootMiddleProximal",
        "LeftFootMiddleIntermediate",
        "LeftFootMiddleDistal",
        "LeftFootRingProximal",
        "LeftFootRingIntermediate",
        "LeftFootRingDistal",
        "LeftFootLittleProximal",
        "LeftFootLittleIntermediate",
        "LeftFootLittleDistal",
    ]
    right_foot_finger_humanoid_bones = [
        "RightFootThumbProximal",
        "RightFootThumbIntermediate",
        "RightFootThumbDistal", 
        "RightFootIndexProximal",
        "RightFootIndexIntermediate",
        "RightFootIndexDistal",
        "RightFootMiddleProximal",
        "RightFootMiddleIntermediate",
        "RightFootMiddleDistal",
        "RightFootRingProximal",
        "RightFootRingIntermediate",
        "RightFootRingDistal",
        "RightFootLittleProximal",
        "RightFootLittleIntermediate",
        "RightFootLittleDistal"
    ]
    
    # 指のボーンの実際のボーン名を取得
    finger_bone_names = set()
    for humanoid_bone in finger_humanoid_bones:
        if humanoid_bone in humanoid_to_bone:
            bone_name = humanoid_to_bone[humanoid_bone]
            finger_bone_names.add(bone_name)
            
            # 関連する補助ボーンも追加
            if humanoid_bone in auxiliary_bones:
                for aux_bone in auxiliary_bones[humanoid_bone]:
                    finger_bone_names.add(aux_bone)
    
    print(f"finger_bone_names: {finger_bone_names}")
    
    # 指のボーンウェイトを持つ頂点を特定
    finger_vertices = set()
    if finger_bone_names:
        mesh = target_obj.data
        
        # 各指のボーン名に対応する頂点グループをチェック
        for bone_name in finger_bone_names:
            if bone_name in target_obj.vertex_groups:
                for vert in mesh.vertices:
                    weight = 0.0
                    for g in vert.groups:
                        if target_obj.vertex_groups[g.group].name == bone_name:
                            weight = g.weight
                            break
                    if weight > 0.001:  # 閾値以上のウェイトを持つ頂点
                        finger_vertices.add(vert.index)
        
        print(f"finger_vertices: {len(finger_vertices)}")
    
    closing_filter_mask_weights = create_blendshape_mask(target_obj, ["LeftUpperLeg", "RightUpperLeg", "Hips", "Chest", "Spine", "LeftShoulder", "RightShoulder", "LeftBreast", "RightBreast"], base_avatar_data)

    def attempt_weight_transfer(source_obj, vertex_group, max_distance_try=0.2, max_distance_tried=0.0):
        """ウェイト転送を試行"""
        bone_groups_tmp = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)
        prev_weights = store_weights(target_obj, bone_groups_tmp)
        initial_max_distance = max_distance_try
        
        while max_distance_try <= 1.0:
            if max_distance_tried + 0.0001 < max_distance_try:
                create_distance_normal_based_vertex_group(bpy.data.objects["Body.BaseAvatar"], target_obj, max_distance_try, 0.005, 20.0, "InpaintMask", normal_radius=0.003, filter_mask=closing_filter_mask_weights)
                
                #デバッグ用にbpy.data.objects["Body.BaseAvatar"]をコピーしておく
                # body_base_avatar_copy = bpy.data.objects["Body.BaseAvatar"].copy()
                # body_base_avatar_copy.data = bpy.data.objects["Body.BaseAvatar"].data.copy()
                # body_base_avatar_copy.name = "Body.BaseAvatar.Copy"
                # bpy.context.scene.collection.objects.link(body_base_avatar_copy)

                # target_obj_copy = target_obj.copy()
                # target_obj_copy.data = target_obj.data.copy()
                # target_obj_copy.name = target_obj.name + ".Copy"
                # bpy.context.scene.collection.objects.link(target_obj_copy)

                # current_mode = bpy.context.object.mode
                # bpy.ops.object.mode_set(mode='OBJECT')
                # current_active = bpy.context.active_object
                # bpy.context.view_layer.objects.active = body_base_avatar_copy
                # selection = bpy.context.selected_objects
                # bpy.ops.object.select_all(action='DESELECT')
                
                # body_base_avatar_copy.select_set(True)
                # target_obj_copy.select_set(True)
                # bpy.ops.object.convert(target='MESH')

                # bpy.ops.object.select_all(action='DESELECT')
                # for obj in selection:
                #     obj.select_set(True)
                # bpy.context.view_layer.objects.active = current_active
                # bpy.ops.object.mode_set(mode=current_mode)

                # 指のボーンウェイトを持つ頂点がある場合、より精密なInpaintMaskを作成
                # if finger_vertices and len(finger_vertices) > 0:
                #     # normal_radius=0.001で精密なマスクを作成（一時的な名前で）
                #     temp_mask_name = "TempFingerInpaintMask"
                #     create_distance_normal_based_vertex_group(bpy.data.objects["Body.BaseAvatar"], target_obj, max_distance_try, 0.003, 30.0, temp_mask_name, normal_radius=0.001)
                    
                #     # 指の頂点のみ、精密なマスクの値で元のInpaintMaskを上書き
                #     if temp_mask_name in target_obj.vertex_groups and "InpaintMask" in target_obj.vertex_groups:
                #         temp_group = target_obj.vertex_groups[temp_mask_name]
                #         inpaint_group = target_obj.vertex_groups["InpaintMask"]
                        
                #         for vert_idx in finger_vertices:
                #             vert = target_obj.data.vertices[vert_idx]
                #             weight = 0.0
                #             for g in vert.groups:
                #                 if target_obj.vertex_groups[g.group].name == temp_mask_name:
                #                     weight = g.weight
                #                     break
                #             inpaint_group.add([vert_idx], weight, 'REPLACE')
                        
                #         # 一時的なグループを削除
                #         # target_obj.vertex_groups.remove(temp_group)
                
                if finger_vertices and len(finger_vertices) > 0:
                    # 指の頂点でInpaintMaskの値を0にする
                    for vert_idx in finger_vertices:
                        target_obj.vertex_groups["InpaintMask"].add([vert_idx], 0.0, 'REPLACE')
                
                #MF_InpaintのウェイトをInpaintMaskのウェイトにかける
                if "MF_Inpaint" in target_obj.vertex_groups and "InpaintMask" in target_obj.vertex_groups:
                    inpaint_group = target_obj.vertex_groups["InpaintMask"]
                    source_group = target_obj.vertex_groups["MF_Inpaint"]
                    
                    for vert in target_obj.data.vertices:
                        source_weight = 0.0
                        for g in vert.groups:
                            if g.group == source_group.index:
                                source_weight = g.weight
                                break
                        inpaint_weight = 0.0
                        for g in vert.groups:
                            if g.group == inpaint_group.index:
                                inpaint_weight = g.weight
                                break
                        inpaint_group.add([vert.index], source_weight * inpaint_weight, 'REPLACE')
                
                # vertex_groupのウェイトが0である頂点のInpaintMaskウェイトを0に設定
                if "InpaintMask" in target_obj.vertex_groups and vertex_group in target_obj.vertex_groups:
                    inpaint_group = target_obj.vertex_groups["InpaintMask"]
                    source_group = target_obj.vertex_groups[vertex_group]
                    
                    for vert in target_obj.data.vertices:
                        source_weight = 0.0
                        # vertex_groupのウェイトを取得
                        for g in vert.groups:
                            if g.group == source_group.index:
                                source_weight = g.weight
                                break
                        
                        # ウェイトが0の場合、InpaintMaskも0に設定
                        if source_weight == 0.0:
                            inpaint_group.add([vert.index], 0.0, 'REPLACE')
                            
            try:
                bpy.context.scene.robust_weight_transfer_settings.source_object = source_obj
                bpy.context.object.robust_weight_transfer_settings.vertex_group = vertex_group
                bpy.context.scene.robust_weight_transfer_settings.inpaint_mode = 'POINT'
                bpy.context.scene.robust_weight_transfer_settings.max_distance = max_distance_try
                bpy.context.scene.robust_weight_transfer_settings.use_deformed_target = True
                bpy.context.scene.robust_weight_transfer_settings.use_deformed_source = True
                bpy.context.scene.robust_weight_transfer_settings.enforce_four_bone_limit = True
                bpy.context.scene.robust_weight_transfer_settings.max_normal_angle_difference = 1.5708
                #bpy.context.scene.robust_weight_transfer_settings.max_normal_angle_difference = 0.349066
                bpy.context.scene.robust_weight_transfer_settings.flip_vertex_normal = True
                bpy.context.scene.robust_weight_transfer_settings.smoothing_enable = False
                bpy.context.scene.robust_weight_transfer_settings.smoothing_repeat = 4
                bpy.context.scene.robust_weight_transfer_settings.smoothing_factor = 0.5
                bpy.context.object.robust_weight_transfer_settings.inpaint_group = "InpaintMask"
                bpy.context.object.robust_weight_transfer_settings.inpaint_threshold = 0.5
                bpy.context.object.robust_weight_transfer_settings.inpaint_group_invert = False
                bpy.context.object.robust_weight_transfer_settings.vertex_group_invert = False
                bpy.context.scene.robust_weight_transfer_settings.group_selection = 'DEFORM_POSE_BONES'
                bpy.ops.object.skin_weight_transfer()
                print(f"Weight transfered with max_distance {max_distance_try}")
                return True, max_distance_try
            except RuntimeError as e:
                print(f"Weight transfer failed with max_distance {max_distance_try}: {str(e)}")
                restore_weights(target_obj, prev_weights)
                max_distance_try += 0.05
                if max_distance_try > 1.0:
                    print("Max distance exceeded 1.0, stopping weight transfer attempts")
                    return False, initial_max_distance
        return False, initial_max_distance
    
    def get_vertex_weight_safe(group, vertex_index):
        """頂点グループからウェイトを安全に取得"""
        if not group:
            return 0.0
        try:
            for g in target_obj.data.vertices[vertex_index].groups:
                if g.group == group.index:
                    return g.weight
        except Exception:
            pass
        return 0.0
    
    def propagate_weights_to_side_vertices(target_obj, bone_groups, original_humanoid_weights, clothing_armature, max_iterations=100):
        """
        側面ウェイトを持つがボーンウェイトを持たない頂点にウェイトを伝播
        """
        # BMeshを作成
        bm = bmesh.new()
        bm.from_mesh(target_obj.data)
        bm.verts.ensure_lookup_table()
        
        # 側面ウェイトグループのインデックスを取得
        left_group = target_obj.vertex_groups.get("LeftSideWeights")
        right_group = target_obj.vertex_groups.get("RightSideWeights")
        
        # 衣装アーマチュアのボーングループも含めた対象グループを作成
        all_deform_groups = set(bone_groups)
        if clothing_armature:
            all_deform_groups.update(bone.name for bone in clothing_armature.data.bones)
        
        def get_side_weight(vert_idx, group):
            """頂点の側面ウェイトを取得"""
            if not group:
                return 0.0
            try:
                for g in target_obj.data.vertices[vert_idx].groups:
                    if g.group == group.index:
                        return g.weight
            except Exception:
                pass
            return 0.0
        
        def has_bone_weights(vert_idx):
            """頂点がボーンウェイトを持つかチェック（衣装のボーングループも含む）"""
            for g in target_obj.data.vertices[vert_idx].groups:
                if target_obj.vertex_groups[g.group].name in all_deform_groups:
                    return True
            return False
        
        # 処理対象の頂点を特定
        vertices_to_process = set()
        for vert in target_obj.data.vertices:
            # 側面ウェイトがあり、ボーンウェイトを持たない頂点を特定
            if (get_side_weight(vert.index, left_group) > 0 or 
                get_side_weight(vert.index, right_group) > 0) and not has_bone_weights(vert.index):
                vertices_to_process.add(vert.index)
        
        if not vertices_to_process:
            bm.free()
            return
            
        print(f"Found {len(vertices_to_process)} vertices without bone weights but with side weights")
        
        # ウェイト伝播の反復処理
        iteration = 0
        while vertices_to_process and iteration < max_iterations:
            propagated_this_iteration = set()
            
            for vert_idx in vertices_to_process:
                vert = bm.verts[vert_idx]
                # 隣接頂点を取得
                neighbors_with_weights = []
                
                for edge in vert.link_edges:
                    other = edge.other_vert(vert)
                    if has_bone_weights(other.index):
                        # 頂点間の距離を計算
                        distance = (vert.co - other.co).length
                        neighbors_with_weights.append((other.index, distance))
                
                if neighbors_with_weights:
                    # 最も近い頂点を選択
                    closest_vert_idx = min(neighbors_with_weights, key=lambda x: x[1])[0]
                    
                    # ウェイトをコピー
                    for group in target_obj.vertex_groups:
                        if group.name in all_deform_groups:
                            weight = 0.0
                            for g in target_obj.data.vertices[closest_vert_idx].groups:
                                if g.group == group.index:
                                    weight = g.weight
                                    break
                            if weight > 0:
                                group.add([vert_idx], weight, 'REPLACE')
                    
                    propagated_this_iteration.add(vert_idx)
            
            if not propagated_this_iteration:
                break
                
            print(f"Iteration {iteration + 1}: Propagated weights to {len(propagated_this_iteration)} vertices")
            vertices_to_process -= propagated_this_iteration
            iteration += 1
        
        # 残りの頂点に元のウェイトを割り当て
        if vertices_to_process:
            print(f"Restoring original weights for {len(vertices_to_process)} remaining vertices")
            for vert_idx in vertices_to_process:
                if vert_idx in original_humanoid_weights:
                    # 現在のウェイトを削除
                    for group in target_obj.vertex_groups:
                        if group.name in all_deform_groups:
                            try:
                                group.remove([vert_idx])
                            except RuntimeError:
                                continue
                                
                    # 元のウェイトを復元
                    for group_name, weight in original_humanoid_weights[vert_idx].items():
                        if group_name in target_obj.vertex_groups:
                            target_obj.vertex_groups[group_name].add([vert_idx], weight, 'REPLACE')
        
        bm.free()

    print(f"処理開始: {target_obj.name}")
    if "InpaintMask" not in target_obj.vertex_groups:
        target_obj.vertex_groups.new(name="InpaintMask")
    
    # 側面ウェイトグループ作成
    side_weight_time_start = time.time()
    create_side_weight_groups(target_obj, base_avatar_data, clothing_armature, clothing_avatar_data)
    side_weight_time = time.time() - side_weight_time_start
    print(f"  側面ウェイトグループ作成: {side_weight_time:.2f}秒")

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = target_obj

    # 転送前の頂点グループ名を保存
    original_groups = set(vg.name for vg in target_obj.vertex_groups)

    # 対象のボーングループを取得
    bone_groups = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)

    # 元のHumanoidウェイトを保存
    store_weights_time_start = time.time()
    original_humanoid_weights = store_weights(target_obj, bone_groups)
    store_weights_time = time.time() - store_weights_time_start
    print(f"  元のウェイト保存: {store_weights_time:.2f}秒")

    # 衣装アーマチュアのボーングループも含めた対象グループを作成
    all_deform_groups = set(bone_groups)
    if clothing_armature:
        all_deform_groups.update(bone.name for bone in clothing_armature.data.bones)

    # original_groupsからbone_groupsを除いたグループのウェイトを保存
    original_non_humanoid_groups = all_deform_groups - bone_groups
    original_non_humanoid_weights = store_weights(target_obj, original_non_humanoid_groups)

    # 全てのグループのウェイトを保存
    all_weights = store_weights(target_obj, all_deform_groups)

    # ウェイト初期化
    reset_weights_time_start = time.time()
    reset_bone_weights(target_obj, all_deform_groups)
    reset_weights_time = time.time() - reset_weights_time_start
    print(f"  ウェイト初期化: {reset_weights_time:.2f}秒")

    # 左側のウェイト転送
    left_transfer_time_start = time.time()
    left_transfer_success, left_distance_used = attempt_weight_transfer(bpy.data.objects["Body.BaseAvatar.LeftOnly"], "LeftSideWeights")
    left_transfer_time = time.time() - left_transfer_time_start
    print(f"  左側ウェイト転送: {left_transfer_time:.2f}秒 (成功: {left_transfer_success}, 距離: {left_distance_used})")
    
    failed = False
    
    if not left_transfer_success:
        print("  左側ウェイト転送失敗のため処理中断")
        failed = True
    
    
    if not failed:
        # 右側のウェイト転送
        right_transfer_time_start = time.time()
        right_transfer_success, right_distance_used = attempt_weight_transfer(bpy.data.objects["Body.BaseAvatar.RightOnly"], "RightSideWeights", max_distance_tried=left_distance_used)
        right_transfer_time = time.time() - right_transfer_time_start
        print(f"  右側ウェイト転送: {right_transfer_time:.2f}秒 (成功: {right_transfer_success}, 距離: {right_distance_used})")
        
        if not right_transfer_success:
            print("  右側ウェイト転送失敗のため処理中断")
            failed = True
    
    if failed:
        reset_bone_weights(target_obj, bone_groups)
        restore_weights(target_obj, all_weights)
        return
    
    # MF_Armpitグループが存在し、0.001より大きいウェイトを持つ頂点があるかチェック
    mf_armpit_group = target_obj.vertex_groups.get("MF_Armpit")
    should_armpit_process = False
    if mf_armpit_group:
        for vert in target_obj.data.vertices:
            for g in vert.groups:
                if g.group == mf_armpit_group.index and g.weight > 0.001:
                    should_armpit_process = True
                    break
            if should_armpit_process:
                break
    
    if should_armpit_process:
        if armature and armature.type == 'ARMATURE':
            print("  MF_Armpitグループが存在し、有効なウェイトを持つため処理を実行")
            base_humanoid_weights = store_weights(target_obj, bone_groups)
            reset_bone_weights(target_obj, bone_groups)
            restore_weights(target_obj, all_weights)

            # LeftUpperArmとRightUpperArmボーンにY軸回転を適用
            print("  LeftUpperArmとRightUpperArmボーンにY軸回転を適用")
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # humanoidBonesからLeftUpperArmとRightUpperArmのboneNameを取得
            left_upper_arm_bone = None
            right_upper_arm_bone = None
            
            for bone_map in base_avatar_data.get("humanoidBones", []):
                if bone_map.get("humanoidBoneName") == "LeftUpperArm":
                    left_upper_arm_bone = bone_map.get("boneName")
                elif bone_map.get("humanoidBoneName") == "RightUpperArm":
                    right_upper_arm_bone = bone_map.get("boneName")
            
            # LeftUpperLegボーンに-45度のY軸回転を適用
            if left_upper_arm_bone and left_upper_arm_bone in armature.pose.bones:
                bone = armature.pose.bones[left_upper_arm_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での-45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(-45), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            # RightUpperLegボーンに45度のY軸回転を適用
            if right_upper_arm_bone and right_upper_arm_bone in armature.pose.bones:
                bone = armature.pose.bones[right_upper_arm_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(45), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = target_obj
            bpy.context.view_layer.update()

            shape_key_state = save_shape_key_state(target_obj)
            for key_block in target_obj.data.shape_keys.key_blocks:
                key_block.value = 0.0
            
            # 一時シェイプキーを作成
            temp_shape_name = "WT_shape_forA.MFTemp"
            if target_obj.data.shape_keys and temp_shape_name in target_obj.data.shape_keys.key_blocks:
                temp_shape_key = target_obj.data.shape_keys.key_blocks[temp_shape_name]
            temp_shape_key.value = 1.0

            # ウェイト初期化
            reset_bone_weights(target_obj, bone_groups)

            # ウェイト転送
            print("  ウェイト転送開始")
            transfer_success, distance_used = attempt_weight_transfer(bpy.data.objects["Body.BaseAvatar"], "BothSideWeights")

            restore_shape_key_state(target_obj, shape_key_state)
            temp_shape_key.value = 0.0

            # LeftUpperArmとRightUpperArmボーンにY軸逆回転を適用
            print("  LeftUpperArmとRightUpperArmボーンにY軸逆回転を適用")
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # humanoidBonesからLeftUpperArmとRightUpperArmのboneNameを取得
            left_upper_arm_bone = None
            right_upper_arm_bone = None
            
            for bone_map in base_avatar_data.get("humanoidBones", []):
                if bone_map.get("humanoidBoneName") == "LeftUpperArm":
                    left_upper_arm_bone = bone_map.get("boneName")
                elif bone_map.get("humanoidBoneName") == "RightUpperArm":
                    right_upper_arm_bone = bone_map.get("boneName")
            
            # LeftUpperLegボーンに-45度のY軸回転を適用
            if left_upper_arm_bone and left_upper_arm_bone in armature.pose.bones:
                bone = armature.pose.bones[left_upper_arm_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での-45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(45), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            # RightUpperLegボーンに45度のY軸回転を適用
            if right_upper_arm_bone and right_upper_arm_bone in armature.pose.bones:
                bone = armature.pose.bones[right_upper_arm_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(-45), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = target_obj
            bpy.context.view_layer.update()

            # bone_groupsのウェイトとbase_humanoid_weightsを合成
            mf_armpit_group = target_obj.vertex_groups.get("MF_Armpit")
            if mf_armpit_group and base_humanoid_weights:
                print("  ウェイト合成処理開始")
                
                for vert in target_obj.data.vertices:
                    vert_idx = vert.index
                    
                    # MF_Armpitグループのウェイトを取得
                    mf_armpit_weight = 0.0
                    for g in vert.groups:
                        if g.group == mf_armpit_group.index:
                            mf_armpit_weight = g.weight
                            break
                    
                    # 合成係数を計算
                    current_factor = mf_armpit_weight
                    base_factor = 1.0 - mf_armpit_weight
                    
                    # bone_groupsに属するグループのウェイトを合成
                    for group_name in bone_groups:
                        if group_name in target_obj.vertex_groups:
                            group = target_obj.vertex_groups[group_name]
                            
                            # 現在のウェイトを取得
                            current_weight = 0.0
                            for g in vert.groups:
                                if g.group == group.index:
                                    current_weight = g.weight
                                    break
                            
                            # base_humanoid_weightsからのウェイトを取得
                            base_weight = 0.0
                            if vert_idx in base_humanoid_weights and group_name in base_humanoid_weights[vert_idx]:
                                base_weight = base_humanoid_weights[vert_idx][group_name]
                            
                            # ウェイトを合成：(現在のウェイト) * (MF_crotchのウェイト) + (base_humanoid_weightsでのウェイト) * (1.0 - MF_crotchのウェイト)
                            blended_weight = current_weight * current_factor + base_weight * base_factor
                            
                            # 合成されたウェイトを適用
                            if blended_weight > 0.0001:  # 微小値は無視
                                group.add([vert_idx], blended_weight, 'REPLACE')
                                base_humanoid_weights[vert_idx][group_name] = blended_weight
                            else:
                                try:
                                    group.remove([vert_idx])
                                    base_humanoid_weights[vert_idx][group_name] = 0.0
                                except RuntimeError:
                                    pass
            print("  ウェイト合成処理完了")
        else:
            print("  MF_Armpitグループが存在しないか、アーマチュアが存在しないため処理をスキップ")
    else:
        print("  MF_Armpitグループが存在しないか、有効なウェイトがないため処理をスキップ")
    

    
    # MF_crotchグループが存在し、0.001より大きいウェイトを持つ頂点があるかチェック
    mf_crotch_group = target_obj.vertex_groups.get("MF_crotch")
    should_process = False
    if mf_crotch_group:
        for vert in target_obj.data.vertices:
            for g in vert.groups:
                if g.group == mf_crotch_group.index and g.weight > 0.001:
                    should_process = True
                    break
            if should_process:
                break
    
    if should_process:
        if armature and armature.type == 'ARMATURE':
            print("  MF_crotchグループが存在し、有効なウェイトを持つため処理を実行")
            base_humanoid_weights = store_weights(target_obj, bone_groups)
            reset_bone_weights(target_obj, bone_groups)
            restore_weights(target_obj, all_weights)

            # LeftUpperLegとRightUpperLegボーンにY軸回転を適用
            print("  LeftUpperLegとRightUpperLegボーンにY軸回転を適用")
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # humanoidBonesからLeftUpperLegとRightUpperLegのboneNameを取得
            left_upper_leg_bone = None
            right_upper_leg_bone = None
            
            for bone_map in base_avatar_data.get("humanoidBones", []):
                if bone_map.get("humanoidBoneName") == "LeftUpperLeg":
                    left_upper_leg_bone = bone_map.get("boneName")
                elif bone_map.get("humanoidBoneName") == "RightUpperLeg":
                    right_upper_leg_bone = bone_map.get("boneName")
            
            # LeftUpperLegボーンに-45度のY軸回転を適用
            if left_upper_leg_bone and left_upper_leg_bone in armature.pose.bones:
                bone = armature.pose.bones[left_upper_leg_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での-45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(-70), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            # RightUpperLegボーンに45度のY軸回転を適用
            if right_upper_leg_bone and right_upper_leg_bone in armature.pose.bones:
                bone = armature.pose.bones[right_upper_leg_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(70), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = target_obj
            bpy.context.view_layer.update()

            shape_key_state = save_shape_key_state(target_obj)
            for key_block in target_obj.data.shape_keys.key_blocks:
                key_block.value = 0.0
            
            # 一時シェイプキーを作成
            temp_shape_name = "WT_shape_forCrotch.MFTemp"
            if target_obj.data.shape_keys and temp_shape_name in target_obj.data.shape_keys.key_blocks:
                temp_shape_key = target_obj.data.shape_keys.key_blocks[temp_shape_name]
            temp_shape_key.value = 1.0

            # ウェイト初期化
            reset_bone_weights(target_obj, bone_groups)

            # ウェイト転送
            print("  ウェイト転送開始")
            transfer_success, distance_used = attempt_weight_transfer(bpy.data.objects["Body.BaseAvatar"], "BothSideWeights")

            restore_shape_key_state(target_obj, shape_key_state)
            temp_shape_key.value = 0.0
            
            # LeftUpperLegとRightUpperLegボーンにY軸逆回転を適用
            print("  LeftUpperLegとRightUpperLegボーンにY軸逆回転を適用")
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # humanoidBonesからLeftUpperLegとRightUpperLegのboneNameを取得
            left_upper_leg_bone = None
            right_upper_leg_bone = None
            
            for bone_map in base_avatar_data.get("humanoidBones", []):
                if bone_map.get("humanoidBoneName") == "LeftUpperLeg":
                    left_upper_leg_bone = bone_map.get("boneName")
                elif bone_map.get("humanoidBoneName") == "RightUpperLeg":
                    right_upper_leg_bone = bone_map.get("boneName")
            
            # LeftUpperLegボーンに-45度のY軸回転を適用
            if left_upper_leg_bone and left_upper_leg_bone in armature.pose.bones:
                bone = armature.pose.bones[left_upper_leg_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での-45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(70), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            # RightUpperLegボーンに45度のY軸回転を適用
            if right_upper_leg_bone and right_upper_leg_bone in armature.pose.bones:
                bone = armature.pose.bones[right_upper_leg_bone]
                current_world_matrix = armature.matrix_world @ bone.matrix
                # グローバル座標系での45度Y軸回転を適用
                head_world_transformed = armature.matrix_world @ bone.head
                offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
                rotation_matrix = mathutils.Matrix.Rotation(math.radians(-70), 4, 'Y')
                bone.matrix = armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix
            
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.context.view_layer.objects.active = target_obj
            bpy.context.view_layer.update()

            # bone_groupsのウェイトとbase_humanoid_weightsを合成
            mf_crotch_group = target_obj.vertex_groups.get("MF_crotch")
            if mf_crotch_group and base_humanoid_weights:
                print("  ウェイト合成処理開始")
                
                for vert in target_obj.data.vertices:
                    vert_idx = vert.index
                    
                    # MF_crotchグループのウェイトを取得
                    mf_crotch_weight = 0.0
                    for g in vert.groups:
                        if g.group == mf_crotch_group.index:
                            mf_crotch_weight = g.weight
                            break
                    
                    # 合成係数を計算
                    current_factor = mf_crotch_weight
                    base_factor = 1.0 - mf_crotch_weight
                    
                    # bone_groupsに属するグループのウェイトを合成
                    for group_name in bone_groups:
                        if group_name in target_obj.vertex_groups:
                            group = target_obj.vertex_groups[group_name]
                            
                            # 現在のウェイトを取得
                            current_weight = 0.0
                            for g in vert.groups:
                                if g.group == group.index:
                                    current_weight = g.weight
                                    break
                            
                            # base_humanoid_weightsからのウェイトを取得
                            base_weight = 0.0
                            if vert_idx in base_humanoid_weights and group_name in base_humanoid_weights[vert_idx]:
                                base_weight = base_humanoid_weights[vert_idx][group_name]
                            
                            # ウェイトを合成：(現在のウェイト) * (MF_crotchのウェイト) + (base_humanoid_weightsでのウェイト) * (1.0 - MF_crotchのウェイト)
                            blended_weight = current_weight * current_factor + base_weight * base_factor
                            
                            # 合成されたウェイトを適用
                            if blended_weight > 0.0001:  # 微小値は無視
                                group.add([vert_idx], blended_weight, 'REPLACE')
                            else:
                                try:
                                    group.remove([vert_idx])
                                except RuntimeError:
                                    pass
            print("  ウェイト合成処理完了")
        else:
            print("  MF_crotchグループが存在しないか、アーマチュアが存在しないため処理をスキップ")
    else:
        print("  MF_crotchグループが存在しないか、有効なウェイトがないため処理をスキップ")

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action='DESELECT')
    # InpaintMaskグループのウェイトが0.5以上の頂点を選択
    inpaint_mask_group = target_obj.vertex_groups.get("InpaintMask")
    if inpaint_mask_group:
        for vert in target_obj.data.vertices:
            for g in vert.groups:
                if g.group == inpaint_mask_group.index and g.weight >= 0.5:
                    vert.select = True
                    break
    
    # bone_groupsに含まれるすべての頂点グループに対してスムージングを実行
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    bpy.context.object.data.use_paint_mask = False
    bpy.context.object.data.use_paint_mask_vertex = True
    for group_name in bone_groups:
        if group_name in target_obj.vertex_groups:
            target_obj.vertex_groups.active = target_obj.vertex_groups[group_name]
            bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=3, expand=0.0)

    bpy.ops.object.mode_set(mode='OBJECT')

    # 微小なウェイトを除外
    cleanup_weights_time_start = time.time()
    for vert in target_obj.data.vertices:
        groups_to_remove = []
        for g in vert.groups:
            group_name = target_obj.vertex_groups[g.group].name
            if group_name in bone_groups and g.weight < 0.001:
                groups_to_remove.append(g.group)
        
        # 微小なウェイトを持つグループからその頂点を削除
        for group_idx in groups_to_remove:
            try:
                target_obj.vertex_groups[group_idx].remove([vert.index])
            except RuntimeError:
                continue
    cleanup_weights_time = time.time() - cleanup_weights_time_start
    print(f"  微小ウェイト除外: {cleanup_weights_time:.2f}秒")

    # Create mappings
    humanoid_to_bone = {bone_map["humanoidBoneName"]: bone_map["boneName"] 
                        for bone_map in base_avatar_data["humanoidBones"]}
    bone_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                        for bone_map in base_avatar_data["humanoidBones"]}

    # 転送後の新しい頂点グループを特定
    new_groups = set(vg.name for vg in target_obj.vertex_groups)
    added_groups = new_groups - original_groups

    print(f"  ボーングループ: {bone_groups}")
    print(f"  オリジナルグループ: {original_groups}")
    print(f"  新規グループ: {new_groups}")
    print(f"  追加グループ: {added_groups}")
    


    # 現時点での全てのグループのウェイトを保存
    num_vertices = len(target_obj.data.vertices)
    all_transferred_weights = store_weights(target_obj, all_deform_groups)

    clothing_bone_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                           for bone_map in clothing_avatar_data["humanoidBones"]}
    clothing_bone_to_parent_humanoid = {}
    for clothing_bone in clothing_armature.data.bones:
        current_bone = clothing_bone
        current_bone_name = current_bone.name
        parent_humanoid_name = None
        while current_bone:
            if current_bone.name in clothing_bone_to_humanoid.keys():
                parent_humanoid_name = clothing_bone_to_humanoid[current_bone.name]
                break
            current_bone = current_bone.parent
        print(f"current_bone_name: {current_bone_name}, parent_humanoid_name: {parent_humanoid_name}")
        if parent_humanoid_name:
            clothing_bone_to_parent_humanoid[current_bone_name] = parent_humanoid_name
    
    non_humanoid_parts_mask = np.zeros(num_vertices)
    non_humanoid_total_weights = np.zeros(num_vertices)
    for vert_idx, groups in original_non_humanoid_weights.items():
        total_weight = 0.0
        for group_name, weight in groups.items():
            total_weight += weight
        if total_weight > 1.0:
            total_weight = 1.0
        non_humanoid_total_weights[vert_idx] = total_weight
        if total_weight > 0.999:
            non_humanoid_parts_mask[vert_idx] = 1.0

    transferred_weight_patterns = [None] * num_vertices
    for vert_idx in range(num_vertices):
        groups = all_transferred_weights.get(vert_idx, {})
        converted_weights = defaultdict(float)

        for group_name, weight in groups.items():
            if weight <= 0.0:
                continue
            if group_name in auxiliary_bones_to_humanoid:
                humanoid_name = auxiliary_bones_to_humanoid[group_name]
                if humanoid_name:
                    converted_weights[humanoid_name] += weight
            else:
                humanoid_name = bone_to_humanoid.get(group_name)
                if humanoid_name:
                    converted_weights[humanoid_name] += weight
                else:
                    converted_weights[group_name] += weight
        transferred_weight_patterns[vert_idx] = dict(converted_weights)

    original_non_humanoid_weight_patterns = [None] * num_vertices
    for vert_idx in range(num_vertices):
        groups = original_non_humanoid_weights.get(vert_idx, {})
        converted_weights = defaultdict(float)

        for group_name, weight in groups.items():
            if weight <= 0.0:
                continue

            parent_humanoid = clothing_bone_to_parent_humanoid.get(group_name)
            if parent_humanoid:
                converted_weights[parent_humanoid] += weight
            else:
                converted_weights[group_name] += weight

        original_non_humanoid_weight_patterns[vert_idx] = dict(converted_weights)

    cloth_bm = get_evaluated_mesh(target_obj)
    cloth_bm.verts.ensure_lookup_table()
    cloth_bm.faces.ensure_lookup_table()
    vertex_coords = np.array([v.co for v in cloth_bm.verts])

    pattern_difference_threshold = 0.2
    neighbor_search_radius = 0.005
    non_humanoid_difference_mask = np.zeros_like(non_humanoid_parts_mask)

    hinge_bone_mask = np.zeros_like(non_humanoid_parts_mask)
    hinge_group = target_obj.vertex_groups.get("HingeBone")
    if hinge_group:
        for vert_idx in range(num_vertices):
            for g in target_obj.data.vertices[vert_idx].groups:
                if g.group == hinge_group.index and g.weight > 0.001:
                    hinge_bone_mask[vert_idx] = 1.0
                    break

    if num_vertices > 0:
        kd_tree = cKDTree(vertex_coords)

        def calculate_pattern_difference(weights_a, weights_b):
            if not weights_a and not weights_b:
                return 0.0
            keys = set(weights_a.keys()) | set(weights_b.keys())
            difference = 0.0
            for key in keys:
                difference += abs(weights_a.get(key, 0.0) - weights_b.get(key, 0.0))
            return difference

        for vert_idx, mask_value in enumerate(non_humanoid_parts_mask):
            if mask_value <= 0.0:
                continue

            base_pattern = original_non_humanoid_weight_patterns[vert_idx]
            neighbor_indices = kd_tree.query_ball_point(vertex_coords[vert_idx], neighbor_search_radius)

            for neighbor_idx in neighbor_indices:
                if neighbor_idx == vert_idx:
                    continue

                if non_humanoid_parts_mask[neighbor_idx] > 0.001:
                    continue

                neighbor_pattern = transferred_weight_patterns[neighbor_idx]
                if not neighbor_pattern:
                    continue

                difference = calculate_pattern_difference(base_pattern, neighbor_pattern)
                if difference > pattern_difference_threshold:
                    non_humanoid_difference_mask[vert_idx] = 1.0 * hinge_bone_mask[vert_idx] * hinge_bone_mask[vert_idx]
                    break

    # non_humanoid_difference_maskを頂点グループとして追加
    non_humanoid_difference_group = target_obj.vertex_groups.new(name="NonHumanoidDifference")
    for vert_idx, mask_value in enumerate(non_humanoid_difference_mask):
        if mask_value > 0.0:
            non_humanoid_difference_group.add([vert_idx], 1.0, 'REPLACE')
    
    # 現在のモードを保存
    current_mode = bpy.context.object.mode
    
    # Weight Paintモードに切り替え
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    target_obj.vertex_groups.active_index = non_humanoid_difference_group.index
    bpy.ops.paint.vert_select_all(action='SELECT')
    
    # vertex_group_smoothを使用してスムージング
    bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=5, expand=0.5)

    # DistanceFalloffMaskグループを作成
    falloff_mask_time_start = time.time()
    # CommonSwaySettingsから距離パラメータを取得
    sway_settings = base_avatar_data.get("commonSwaySettings", {"startDistance": 0.025, "endDistance": 0.050})
    distance_falloff_group = create_distance_falloff_transfer_mask(target_obj, base_avatar_data, 'DistanceFalloffMask', 
                                                                 max_distance=sway_settings["endDistance"], 
                                                                 min_distance=sway_settings["startDistance"])
    target_obj.vertex_groups.active_index = distance_falloff_group.index
    
    # vertex_group_smoothを使用してスムージング
    bpy.ops.object.vertex_group_smooth(factor=1, repeat=3, expand=0.1)
    falloff_mask_time = time.time() - falloff_mask_time_start
    print(f"  距離フォールオフマスク作成: {falloff_mask_time:.2f}秒")

    distance_falloff_group2 = create_distance_falloff_transfer_mask(target_obj, base_avatar_data, 'DistanceFalloffMask2', 
                                                                 max_distance=0.1, 
                                                                 min_distance=0.04)
    target_obj.vertex_groups.active_index = distance_falloff_group2.index
    
    # vertex_group_smoothを使用してスムージング
    bpy.ops.object.vertex_group_smooth(factor=1, repeat=3, expand=0.1)

    print(f"  distance_falloff_group2: {distance_falloff_group2.index}")

    # 元のモードに戻す
    bpy.ops.object.mode_set(mode=current_mode)

    non_humanoid_difference_weights = np.zeros(num_vertices)
    distance_falloff_weights = np.zeros(num_vertices)
    for vert_idx in range(num_vertices):
        for g in target_obj.data.vertices[vert_idx].groups:
            if g.group == non_humanoid_difference_group.index:
                non_humanoid_difference_weights[vert_idx] = g.weight
            if g.group == distance_falloff_group2.index:
                distance_falloff_weights[vert_idx] = g.weight

    for vert_idx, groups in original_non_humanoid_weights.items():
        for group_name, weight in groups.items():
            if group_name in target_obj.vertex_groups:
                result_weight = weight * ( 1.0 - non_humanoid_difference_weights[vert_idx] * distance_falloff_weights[vert_idx] )
                target_obj.vertex_groups[group_name].add([vert_idx], result_weight, 'REPLACE')
    
    current_humanoid_weights = store_weights(target_obj, bone_groups)
    for vert_idx, groups in current_humanoid_weights.items():
        for group_name, weight in groups.items():
            if group_name in target_obj.vertex_groups:
                factor = ( 1.0 - non_humanoid_total_weights[vert_idx] * (1.0 - non_humanoid_difference_weights[vert_idx] * distance_falloff_weights[vert_idx]) )
                result_weight = weight * factor
                target_obj.vertex_groups[group_name].add([vert_idx], result_weight, 'REPLACE')
    
    for vert_idx in range(len(non_humanoid_total_weights)):
        non_humanoid_total_weights[vert_idx] = non_humanoid_total_weights[vert_idx] * (1.0 - non_humanoid_difference_weights[vert_idx] * distance_falloff_weights[vert_idx])
    
    cloth_bm.free()


    # 各新規グループに対して親ボーンを見つけてウェイトを統合
    group_merge_time_start = time.time()
    max_iterations = 5
    iteration = 0
    while added_groups and iteration < max_iterations:
        changed = False
        remaining_groups = set()

        print(f"  反復処理: {iteration}")

        for group_name in added_groups:

            print(f"  グループ名: {group_name}")

            if group_name not in target_obj.vertex_groups:
                print(f"  {group_name} は削除されています。スキップします")
                continue

            # 新規グループのウェイトが0より大きい頂点を取得
            group = target_obj.vertex_groups[group_name]
            verts_with_weight = []
            for v in target_obj.data.vertices:
                weight = get_vertex_weight_safe(group, v.index)
                if weight > 0:
                    verts_with_weight.append(v)
            
            print(f"  ウェイトを持つ頂点数: {len(verts_with_weight)}")

            if len(verts_with_weight) == 0:
                print(f"  {group_name} は空: スキップします")
                continue

            if group_name in bone_to_humanoid:
                humanoid_group_name = bone_to_humanoid[group_name]
                if "LeftToes" in humanoid_to_bone and humanoid_to_bone["LeftToes"] in original_groups:
                    if humanoid_group_name in left_foot_finger_humanoid_bones:
                        merge_weights_to_parent(target_obj, group_name, humanoid_to_bone["LeftToes"])
                        changed = True
                        continue
                if "RightToes" in humanoid_to_bone and humanoid_to_bone["RightToes"] in original_groups:
                    if humanoid_group_name in right_foot_finger_humanoid_bones:
                        merge_weights_to_parent(target_obj, group_name, humanoid_to_bone["RightToes"])
                        changed = True
                        continue
            
            # 該当する既存グループを探す
            existing_groups = set()
            for vert in verts_with_weight:
                for g in vert.groups:
                    g_name = target_obj.vertex_groups[g.group].name
                    if g_name in bone_groups and g_name in original_groups and g.weight > 0:
                        existing_groups.add(g_name)
            
            print(f"  既存グループ: {existing_groups}")
            
            if len(existing_groups) == 1:
                # 一つだけ該当する既存グループがある場合はそれに統合
                merge_weights_to_parent(target_obj, group_name, list(existing_groups)[0])
                changed = True
            elif len(existing_groups) == 0:
                # 該当する既存グループがない場合は隣接頂点も探索
                bm = bmesh.new()
                bm.from_mesh(target_obj.data)
                bm.verts.ensure_lookup_table()

                visited_verts = set(vert.index for vert in verts_with_weight)
                queue = deque(verts_with_weight)
                
                while queue:
                    vert = queue.popleft()
                    for edge in bm.verts[vert.index].link_edges:
                        other_vert = edge.other_vert(bm.verts[vert.index])
                        if other_vert.index not in visited_verts:
                            visited_verts.add(other_vert.index)
                            for g in target_obj.data.vertices[other_vert.index].groups:
                                if target_obj.vertex_groups[g.group].name in bone_groups and g.weight > 0:
                                    existing_groups.add(target_obj.vertex_groups[g.group].name)
                                    if len(existing_groups) > 1:
                                        break
                            if len(existing_groups) == 1:
                                merge_weights_to_parent(target_obj, group_name, existing_groups.pop())
                                changed = True
                                break
                            queue.append(target_obj.data.vertices[other_vert.index])
                
                bm.free()

                print(f"  隣接探索後の既存グループ: {existing_groups}")

            if len(existing_groups) != 1:
                remaining_groups.add(group_name)
        
        if not changed:
            break

        added_groups = remaining_groups
        iteration += 1
    group_merge_time = time.time() - group_merge_time_start
    print(f"  グループ統合処理: {group_merge_time:.2f}秒")

    # 統合できなかった新規グループについて補助ボーンの処理を行う
    aux_bone_time_start = time.time()
    for group_name in list(added_groups):  # Setのコピーを作成してイテレーション
        for aux_set in base_avatar_data.get("auxiliaryBones", []):
            if group_name in aux_set["auxiliaryBones"]:
                humanoid_bone = aux_set["humanoidBoneName"]
                if humanoid_bone in humanoid_to_bone and humanoid_to_bone[humanoid_bone] in bone_groups:
                    merge_weights_to_parent(target_obj, group_name, humanoid_to_bone[humanoid_bone])
                    try:
                        added_groups.remove(group_name)
                    except KeyError:
                        pass  # group_nameが既に削除されている場合を無視
                    break
    
    # それでも統合できなかった新規グループについてオリジナルのウェイトを加算
    for group_name in added_groups:
        if group_name not in target_obj.vertex_groups:
            continue
        group = target_obj.vertex_groups[group_name]
        for vert in target_obj.data.vertices:
            weight = get_vertex_weight_safe(group, vert.index)
            if weight > 0:
                for orig_group_name, orig_weight in original_humanoid_weights[vert.index].items():
                    if orig_group_name in target_obj.vertex_groups:
                        target_obj.vertex_groups[orig_group_name].add([vert.index], orig_weight * weight, 'ADD')
        
    # 新規グループを削除
    for group_name in added_groups:
        if group_name in target_obj.vertex_groups:
            target_obj.vertex_groups.remove(target_obj.vertex_groups[group_name])
    aux_bone_time = time.time() - aux_bone_time_start
    print(f"  補助ボーン処理: {aux_bone_time:.2f}秒")

    # 現在のウェイトを結果Aとして保存
    store_result_a_time_start = time.time()
    weights_a = {}
    for vert_idx in range(len(target_obj.data.vertices)):
        weights_a[vert_idx] = {}
        for group in target_obj.vertex_groups:
            if group.name in bone_groups:
                try:
                    weight = 0.0
                    for g in target_obj.data.vertices[vert_idx].groups:
                        if g.group == group.index:
                            weight = g.weight
                            break
                    weights_a[vert_idx][group.name] = weight
                except Exception:
                    continue
    store_result_a_time = time.time() - store_result_a_time_start
    print(f"  結果A保存: {store_result_a_time:.2f}秒")

    # 現在のウェイトをコピーして結果Bを作成
    store_result_b_time_start = time.time()
    weights_b = {}
    for vert_idx in range(len(target_obj.data.vertices)):
        weights_b[vert_idx] = {}
        for group in target_obj.vertex_groups:
            if group.name in bone_groups:
                try:
                    weight = 0.0
                    for g in target_obj.data.vertices[vert_idx].groups:
                        if g.group == group.index:
                            weight = g.weight
                            break
                    weights_b[vert_idx][group.name] = weight
                except Exception:
                    continue
    store_result_b_time = time.time() - store_result_b_time_start
    print(f"  結果B保存: {store_result_b_time:.2f}秒")

    # swayBonesの統合処理
    sway_bones_time_start = time.time()
    for sway_bone in base_avatar_data.get("swayBones", []):
        parent_bone = sway_bone["parentBoneName"]
        for affected_bone in sway_bone["affectedBones"]:
            # 各頂点について処理
            for vert_idx in weights_b:
                if affected_bone in weights_b[vert_idx]:
                    affected_weight = weights_b[vert_idx][affected_bone]
                    # 親ボーンのウェイトに加算
                    if parent_bone not in weights_b[vert_idx]:
                        weights_b[vert_idx][parent_bone] = 0.0
                    weights_b[vert_idx][parent_bone] += affected_weight
                    # affected_boneのウェイトを削除
                    del weights_b[vert_idx][affected_bone]
    sway_bones_time = time.time() - sway_bones_time_start
    print(f"  SwayBones処理: {sway_bones_time:.2f}秒")

    # 結果AとBを合成
    weight_blend_time_start = time.time()
    for vert_idx in range(len(target_obj.data.vertices)):
        # DistanceFalloffMaskのウェイトを取得
        falloff_weight = 0.0
        for g in target_obj.data.vertices[vert_idx].groups:
            if g.group == distance_falloff_group.index:
                falloff_weight = g.weight
                break

        # 各頂点グループについて処理
        for group_name in bone_groups:
            if group_name in target_obj.vertex_groups:
                weight_a = weights_a[vert_idx].get(group_name, 0.0)
                weight_b = weights_b[vert_idx].get(group_name, 0.0)

                # ウェイトを合成
                final_weight = (weight_a * falloff_weight) + (weight_b * (1.0 - falloff_weight))

                # 新しいウェイトを設定
                group = target_obj.vertex_groups[group_name]
                if final_weight > 0:
                    group.add([vert_idx], final_weight, 'REPLACE')
                else:
                    try:
                        group.remove([vert_idx])
                    except RuntimeError:
                        pass
    weight_blend_time = time.time() - weight_blend_time_start
    print(f"  ウェイト合成: {weight_blend_time:.2f}秒")
    
    # 手のウェイト調整
    hand_weights_time_start = time.time()
    adjust_hand_weights(target_obj, armature, base_avatar_data)
    hand_weights_time = time.time() - hand_weights_time_start
    print(f"  手のウェイト調整: {hand_weights_time:.2f}秒")

    #normalize_connected_components_weights(target_obj, base_avatar_data)

    # 側面頂点へのウェイト伝播
    propagate_time_start = time.time()
    propagate_weights_to_side_vertices(target_obj, bone_groups, original_humanoid_weights, clothing_armature)
    propagate_time = time.time() - propagate_time_start
    print(f"  側面頂点へのウェイト伝播: {propagate_time:.2f}秒")

    # サイドウェイトとボーンウェイトの比較と調整
    comparison_time_start = time.time()
    side_left_group = target_obj.vertex_groups.get("LeftSideWeights")
    side_right_group = target_obj.vertex_groups.get("RightSideWeights")

    failed_vertices_count = 0
    if side_left_group and side_right_group:
        for vert in target_obj.data.vertices:
            # サイドウェイトの合計を計算
            total_side_weight = 0.0
            for g in vert.groups:
                if g.group == side_left_group.index or g.group == side_right_group.index:
                    total_side_weight += g.weight
            total_side_weight = min(total_side_weight, 1.0)  # 0-1にクランプ

            total_side_weight = total_side_weight - non_humanoid_total_weights[vert.index]
            total_side_weight = max(total_side_weight, 0.0)

            # bone_groupsの合計ウェイトを計算
            total_bone_weight = 0.0
            for g in vert.groups:
                group_name = target_obj.vertex_groups[g.group].name
                if group_name in bone_groups:
                    total_bone_weight += g.weight

            # サイドウェイトがボーンウェイトより0.5以上大きい場合
            if total_side_weight > total_bone_weight + 0.5:
                # 現在のbone_groupsのウェイトを消去
                for group in target_obj.vertex_groups:
                    if group.name in bone_groups:
                        try:
                            group.remove([vert.index])
                        except RuntimeError:
                            continue

                # 元のウェイトを復元
                if vert.index in original_humanoid_weights:
                    for group_name, weight in original_humanoid_weights[vert.index].items():
                        if group_name in target_obj.vertex_groups:
                            target_obj.vertex_groups[group_name].add([vert.index], weight, 'REPLACE')
                failed_vertices_count += 1
    if failed_vertices_count > 0:
        print(f"  ウェイト転送失敗: {failed_vertices_count}頂点 -> オリジナルウェイトにフォールバック")
    comparison_time = time.time() - comparison_time_start
    print(f"  サイドウェイト比較調整: {comparison_time:.2f}秒")

    # apply_distance_normal_based_smoothingを実行
    smoothing_time_start = time.time()
    
    # target_vertex_groupsを構築（Chest, LeftBreast, RightBreastとそれらのauxiliaryBones）
    target_vertex_groups = []
    smoothing_mask_groups = []
    target_humanoid_bones = [
        "Chest", "LeftBreast", "RightBreast", "Neck", "Head", "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
        "LeftHand", 
        "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
        "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
        "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
        "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
        "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
        "RightHand", 
        "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
        "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
        "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal", 
        "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
        "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal"
        ]
    smoothing_mask_humanoid_bones = [
        "Chest", "LeftBreast", "RightBreast", "Neck", "Head", "LeftShoulder", "RightShoulder",
        "LeftHand",
        "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
        "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal", 
        "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal", 
        "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal", 
        "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
        "RightHand",
        "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
        "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal", 
        "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal", 
        "RightRingProximal", "RightRingIntermediate", "RightRingDistal", 
        "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal"
        ]
    humanoid_to_bone = {bone_map["humanoidBoneName"]: bone_map["boneName"] 
                        for bone_map in base_avatar_data["humanoidBones"]}
    
    for humanoid_bone in target_humanoid_bones:
        if humanoid_bone in humanoid_to_bone:
            target_vertex_groups.append(humanoid_to_bone[humanoid_bone])
    
    # auxiliaryBonesを追加
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        if aux_set["humanoidBoneName"] in target_humanoid_bones:
            target_vertex_groups.extend(aux_set["auxiliaryBones"])
    
    for humanoid_bone in smoothing_mask_humanoid_bones:
        if humanoid_bone in humanoid_to_bone:
            smoothing_mask_groups.append(humanoid_to_bone[humanoid_bone])
    
    # auxiliaryBonesを追加
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        if aux_set["humanoidBoneName"] in smoothing_mask_humanoid_bones:
            smoothing_mask_groups.extend(aux_set["auxiliaryBones"])
    
    # Body.BaseAvatarオブジェクトを取得
    body_obj = bpy.data.objects.get("Body.BaseAvatar")
    
    # LeftBreastまたはRightBreastのボーンウェイトが0でない頂点があるかチェック
    breast_bone_groups = []
    breast_humanoid_bones = ["Hips", "LeftBreast", "RightBreast", "Neck", "Head", "LeftHand", "RightHand"]
    
    for humanoid_bone in breast_humanoid_bones:
        if humanoid_bone in humanoid_to_bone:
            breast_bone_groups.append(humanoid_to_bone[humanoid_bone])
    
    # LeftBreastとRightBreastのauxiliaryBonesも追加
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        if aux_set["humanoidBoneName"] in breast_humanoid_bones:
            breast_bone_groups.extend(aux_set["auxiliaryBones"])
    
    # target_objにbreast_bone_groupsのウェイトが0でない頂点があるかチェック
    has_breast_weights = False
    if breast_bone_groups:
        for group_name in breast_bone_groups:
            if group_name in target_obj.vertex_groups:
                group = target_obj.vertex_groups[group_name]
                # 頂点グループに実際にウェイトがあるかチェック
                for vert in target_obj.data.vertices:
                    try:
                        weight = 0.0
                        for g in vert.groups:
                            if g.group == group.index:
                                weight = g.weight
                                break
                        if weight > 0:
                            has_breast_weights = True
                            break
                    except RuntimeError:
                        continue
                if has_breast_weights:
                    break
    
    if body_obj and target_vertex_groups and has_breast_weights:
        print(f"  距離・法線ベースのスムージングを実行: {len(target_vertex_groups)}個のターゲットグループ (LeftBreast/RightBreastウェイト検出)")
        apply_distance_normal_based_smoothing(
            body_obj=body_obj,
            cloth_obj=target_obj,
            distance_min=0.005,
            distance_max=0.015,
            angle_min=15.0,
            angle_max=30.0,
            new_group_name="SmoothMask",
            normal_radius=0.01,
            smoothing_mask_groups=smoothing_mask_groups,
            target_vertex_groups=target_vertex_groups,
            smoothing_radius=0.05,
            mask_group_name="MF_Blur"
        )
    else:
        print("  Body.BaseAvatarオブジェクトが見つからないか、ターゲットグループが空です")
    
    smoothing_time = time.time() - smoothing_time_start
    print(f"  距離・法線ベースのスムージング: {smoothing_time:.2f}秒")

    # 距離が大きいほどオリジナルのウェイトの比率を高めて合成するように調整
    current_mode = bpy.context.object.mode
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

    target_obj.vertex_groups.active_index = distance_falloff_group2.index

    print(f"  distance_falloff_group2: {distance_falloff_group2.index}")
    print(f"  distance_falloff_group2_index: {target_obj.vertex_groups[distance_falloff_group2.name].index}")

    # LeftBreastまたはRightBreastのボーンウェイトが0でない頂点があるかチェック
    exclude_bone_groups = []
    exclude_humanoid_bones = ["LeftBreast", "RightBreast"]
    
    for humanoid_bone in exclude_humanoid_bones:
        if humanoid_bone in humanoid_to_bone:
            exclude_bone_groups.append(humanoid_to_bone[humanoid_bone])
    
    # LeftBreastとRightBreastのauxiliaryBonesも追加
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        if aux_set["humanoidBoneName"] in exclude_humanoid_bones:
            exclude_bone_groups.extend(aux_set["auxiliaryBones"])

    # 胸部分は合成処理から除外する
    if exclude_bone_groups:
        new_group_weights = np.zeros(len(target_obj.data.vertices), dtype=np.float32)
        for i, vertex in enumerate(target_obj.data.vertices):
            for group in vertex.groups:
                if group.group == distance_falloff_group2.index:
                    new_group_weights[i] = group.weight
                    break
        total_target_weights = np.zeros(len(target_obj.data.vertices), dtype=np.float32)
        
        for target_group_name in exclude_bone_groups:
            if target_group_name in target_obj.vertex_groups:
                target_group = target_obj.vertex_groups[target_group_name]
                print(f"    頂点グループ '{target_group_name}' のウェイトを取得中...")
                
                for i, vertex in enumerate(target_obj.data.vertices):
                    for group in vertex.groups:
                        if group.group == target_group.index:
                            total_target_weights[i] += group.weight
                            break
            else:
                print(f"    警告: 頂点グループ '{target_group_name}' が見つかりません")
        
        masked_weights = np.maximum(new_group_weights, total_target_weights)
        
        # 結果を新しい頂点グループに適用
        for i in range(len(target_obj.data.vertices)):
            distance_falloff_group2.add([i], masked_weights[i], 'REPLACE')

    for vert_idx in range(len(target_obj.data.vertices)):
        if vert_idx in original_humanoid_weights and non_humanoid_parts_mask[vert_idx] < 0.0001:
            falloff_weight = 0.0
            for g in target_obj.data.vertices[vert_idx].groups:
                if g.group == distance_falloff_group2.index:
                    falloff_weight = g.weight
                    break
            
            for g in target_obj.data.vertices[vert_idx].groups:
                if target_obj.vertex_groups[g.group].name in bone_groups:
                    weight = g.weight
                    group_name = target_obj.vertex_groups[g.group].name
                    target_obj.vertex_groups[group_name].add([vert_idx], weight * falloff_weight, 'REPLACE')

            for group_name, weight in original_humanoid_weights[vert_idx].items():
                if group_name in target_obj.vertex_groups:
                    target_obj.vertex_groups[group_name].add([vert_idx], weight * (1.0 - falloff_weight), 'ADD')

    bpy.ops.object.mode_set(mode=current_mode)


    # Headボーンのウェイトをオリジナルに戻す処理
    head_time_start = time.time()
    head_bone_name = None
    # base_avatar_dataからHeadラベルを持つボーンを検索
    if base_avatar_data:
        if "humanoidBones" in base_avatar_data:
            for bone_data in base_avatar_data["humanoidBones"]:
                if bone_data.get("humanoidBoneName", "") == "Head":
                    head_bone_name = bone_data.get("boneName", "")
                    break
    
    if head_bone_name and head_bone_name in target_obj.vertex_groups:
        print(f"  Headボーンウェイトを処理中: {head_bone_name}")
        head_vertices_count = 0
        
        for vert_idx in range(len(target_obj.data.vertices)):
            # オリジナルのHeadウェイトを取得
            original_head_weight = 0.0
            if vert_idx in original_humanoid_weights:
                original_head_weight = original_humanoid_weights[vert_idx].get(head_bone_name, 0.0)
            
            # 現在のHeadウェイトを取得
            current_head_weight = 0.0
            for g in target_obj.data.vertices[vert_idx].groups:
                if g.group == target_obj.vertex_groups[head_bone_name].index:
                    current_head_weight = g.weight
                    break
            
            # Headウェイトの差分を計算
            head_weight_diff = original_head_weight - current_head_weight
            
            # Headウェイトをオリジナルの値に設定
            if original_head_weight > 0.0:
                target_obj.vertex_groups[head_bone_name].add([vert_idx], original_head_weight, 'REPLACE')
            else:
                # オリジナルが0の場合は削除
                try:
                    target_obj.vertex_groups[head_bone_name].remove([vert_idx])
                except RuntimeError:
                    pass
            
            # 差分がある場合、他のボーンのオリジナルウェイトに差分を掛けて加算
            if abs(head_weight_diff) > 0.0001 and vert_idx in original_humanoid_weights:
                for group in target_obj.vertex_groups:
                    if group.name in bone_groups and group.name != head_bone_name:
                        # オリジナルウェイトを取得
                        original_weight = original_humanoid_weights[vert_idx].get(group.name, 0.0)
                        
                        if original_weight > 0.0:
                            # 現在のウェイトを取得
                            current_weight = 0.0
                            for g in target_obj.data.vertices[vert_idx].groups:
                                if g.group == group.index:
                                    current_weight = g.weight
                                    break
                            
                            # 差分に基づいて加算
                            new_weight = current_weight + (original_weight * head_weight_diff)
                            if new_weight > 0.0:
                                group.add([vert_idx], new_weight, 'REPLACE')
                            else:
                                try:
                                    group.remove([vert_idx])
                                except RuntimeError:
                                    pass
            
            # 最終的にall_deform_groupsのウェイト合計が1未満の場合、埋め合わせる
            total_weight = 0.0
            for g in target_obj.data.vertices[vert_idx].groups:
                group_name = target_obj.vertex_groups[g.group].name
                if group_name in all_deform_groups:
                    total_weight += g.weight
            
            # ウェイト合計が1未満の場合、不足分を埋め合わせる
            if total_weight < 0.9999 and vert_idx in original_humanoid_weights:
                weight_shortage = 1.0 - total_weight
                
                for group in target_obj.vertex_groups:
                    if group.name in bone_groups:
                        # オリジナルウェイトを取得
                        original_weight = original_humanoid_weights[vert_idx].get(group.name, 0.0)
                        
                        if original_weight > 0.0:
                            # 現在のウェイトを取得
                            current_weight = 0.0
                            for g in target_obj.data.vertices[vert_idx].groups:
                                if g.group == group.index:
                                    current_weight = g.weight
                                    break
                            
                            # 不足分をオリジナルウェイトに基づいて加算
                            additional_weight = original_weight * weight_shortage
                            new_weight = current_weight + additional_weight
                            group.add([vert_idx], new_weight, 'REPLACE')
            
            head_vertices_count += 1
        
        if head_vertices_count > 0:
            print(f"  Headウェイト処理完了: {head_vertices_count}頂点")
    
    head_time = time.time() - head_time_start
    print(f"  Headウェイト処理: {head_time:.2f}秒")

    # clothMetadataに基づいてウェイトを選択的に元に戻す
    metadata_time_start = time.time()
    if cloth_metadata:
        mesh_name = target_obj.name
        if mesh_name in cloth_metadata:
            vertex_max_distances = cloth_metadata[mesh_name]
            print(f"  メッシュのクロスメタデータを処理: {mesh_name}")
            
            count = 0
            # 各頂点について処理
            for vert_idx in range(len(target_obj.data.vertices)):
                # maxDistanceを取得（ない場合は10.0を使用）
                max_distance = float(vertex_max_distances.get(str(vert_idx), 10.0))
                
                # maxDistanceが1.0より大きい場合、元のウェイトを復元
                if max_distance > 1.0:
                    if vert_idx in original_humanoid_weights:
                        # 現在のグループをすべて削除
                        for group in target_obj.vertex_groups:
                            if group.name in bone_groups:
                                try:
                                    group.remove([vert_idx])
                                except RuntimeError:
                                    continue
                        
                        # 元のウェイトを復元
                        for group_name, weight in original_humanoid_weights[vert_idx].items():
                            if group_name in target_obj.vertex_groups:
                                target_obj.vertex_groups[group_name].add([vert_idx], weight, 'REPLACE')
                        count += 1
            print(f"  処理された頂点数: {count}")
    metadata_time = time.time() - metadata_time_start
    print(f"  クロスメタデータ処理: {metadata_time:.2f}秒")

    total_time = time.time() - start_time
    print(f"処理完了: {target_obj.name} - 合計時間: {total_time:.2f}秒")
