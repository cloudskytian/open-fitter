import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mathutils import Vector


def adjust_hand_weights(target_obj, armature, base_avatar_data):

    def get_bone_name(humanoid_bone_name):
        """Humanoidボーン名から実際のボーン名を取得"""
        for bone_data in base_avatar_data.get("humanoidBones", []):
            if bone_data.get("humanoidBoneName") == humanoid_bone_name:
                return bone_data.get("boneName")
        return None

    def get_finger_bones(side_prefix):
        """指のボーン名を取得（足の指は除外）"""
        finger_bones = []
        finger_types = ["Thumb", "Index", "Middle", "Ring", "Little"]
        positions = ["Proximal", "Intermediate", "Distal"]
        
        for finger in finger_types:
            for pos in positions:
                humanoid_name = f"{side_prefix}{finger}{pos}"
                # "Foot"を含まないHumanoidボーン名のみを処理
                if "Foot" not in humanoid_name:
                    bone_name = get_bone_name(humanoid_name)
                    if bone_name:
                        finger_bones.append(bone_name)
        
        return finger_bones

    def get_bone_head_world(bone_name):
        """ボーンのhead位置をワールド座標で取得"""
        bone = armature.pose.bones[bone_name]
        return armature.matrix_world @ bone.head

    def get_lowerarm_and_auxiliary_bones(side_prefix):
        """LowerArmとその補助ボーンを取得"""
        lower_arm_bones = []
        
        # LowerArmボーンを追加
        lower_arm_name = get_bone_name(f"{side_prefix}LowerArm")
        if lower_arm_name:
            lower_arm_bones.append(lower_arm_name)
        
        # 補助ボーンを追加
        for aux_set in base_avatar_data.get("auxiliaryBones", []):
            if aux_set["humanoidBoneName"] == f"{side_prefix}LowerArm":
                lower_arm_bones.extend(aux_set["auxiliaryBones"])
                
        return lower_arm_bones

    def find_closest_lower_arm_bone(hand_head_pos, lower_arm_bones):
        """手のボーンのHeadに最も近いLowerArmまたは補助ボーンを見つける"""
        closest_bone = None
        min_distance = float('inf')
        
        for bone_name in lower_arm_bones:
            if bone_name in armature.pose.bones:
                bone_head = get_bone_head_world(bone_name)
                distance = (Vector(bone_head) - hand_head_pos).length
                if distance < min_distance:
                    min_distance = distance
                    closest_bone = bone_name
                    
        return closest_bone

    def process_hand(is_right):
        # 手の種類に応じてHumanoidボーン名を設定
        side = "Right" if is_right else "Left"
        hand_bone_name = get_bone_name(f"{side}Hand")
        lower_arm_bone_name = get_bone_name(f"{side}LowerArm")

        if not hand_bone_name or not lower_arm_bone_name:
            return

        # 手と指のボーン名を収集
        vertex_groups = [hand_bone_name] + get_finger_bones(side)

        # ボーンの位置をワールド座標で取得
        hand_head = Vector(get_bone_head_world(hand_bone_name))
        lower_arm_head = Vector(get_bone_head_world(lower_arm_bone_name))

        # 先端方向ベクトルを計算
        tip_direction = (hand_head - lower_arm_head).normalized()

        # 最小角度を探す
        min_angle = float('inf')
        has_weight = False

        # 各頂点について処理
        for v in target_obj.data.vertices:
            has_vertex_weight = False
            for group_name in vertex_groups:
                if group_name not in target_obj.vertex_groups:
                    continue
                weight = 0
                try:
                    for g in v.groups:
                        if g.group == target_obj.vertex_groups[group_name].index:
                            weight = g.weight
                            break
                    if weight > 0:
                        has_weight = True
                        has_vertex_weight = True
                except RuntimeError:
                    continue
            
            # この頂点が手または指のウェイトを持っている場合
            if has_vertex_weight:
                # 頂点のワールド座標を計算
                vertex_world = target_obj.matrix_world @ Vector(v.co)
                # 頂点からhandボーンへのベクトル
                vertex_vector = (vertex_world - hand_head).normalized()
                # 角度を計算 (0-180度の範囲に収める)
                # dot productを使用して角度を計算
                dot_product = vertex_vector.dot(tip_direction)
                # -1.0から1.0の範囲にクランプ
                dot_product = max(min(dot_product, 1.0), -1.0)
                angle = np.degrees(np.arccos(dot_product))
                min_angle = min(min_angle, angle)

        if not has_weight:
            return

        # 70度以上の場合の処理
        if min_angle >= 70:
            print(f"- Minimum angle exceeds 70 degrees ({min_angle} degrees), transferring weights for {side} hand")
            
            # LowerArmとその補助ボーンを取得
            lower_arm_bones = get_lowerarm_and_auxiliary_bones(side)
            
            # 手のボーンのHeadに最も近いLowerArmボーンを見つける
            closest_bone = find_closest_lower_arm_bone(hand_head, lower_arm_bones)
            
            if closest_bone:
                print(f"- Transferring weights to {closest_bone}")
                
                # 各頂点について処理
                for v in target_obj.data.vertices:
                    total_weight = 0.0
                    
                    # 手と指のボーンのウェイトを合計
                    for group_name in vertex_groups:
                        if group_name in target_obj.vertex_groups:
                            group = target_obj.vertex_groups[group_name]
                            try:
                                for g in v.groups:
                                    if g.group == group.index:
                                        total_weight += g.weight
                                        break
                            except RuntimeError:
                                continue
                    
                    # ウェイトを最も近いLowerArmボーンに転送
                    if total_weight > 0:
                        if closest_bone not in target_obj.vertex_groups:
                            target_obj.vertex_groups.new(name=closest_bone)
                        target_obj.vertex_groups[closest_bone].add([v.index], total_weight, 'ADD')
                    
                    # 元のウェイトを削除
                    for group_name in vertex_groups:
                        if group_name in target_obj.vertex_groups:
                            try:
                                target_obj.vertex_groups[group_name].remove([v.index])
                            except RuntimeError:
                                continue
            else:
                print(f"Warning: No suitable LowerArm bone found for {side} hand")
        else:
            print(f"- Minimum angle is within acceptable range ({min_angle} degrees), keeping weights for {side} hand")

    # 両手の処理を実行
    process_hand(is_right=True)
    process_hand(is_right=False)
