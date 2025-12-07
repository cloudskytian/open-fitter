import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def build_bone_maps(base_avatar_data):
    """
    ヒューマノイドボーンと補助ボーンのマッピングを構築する。

    Args:
        base_avatar_data: ベースアバターのデータ（humanoidBones, auxiliaryBones を含む）

    Returns:
        tuple: (humanoid_to_bone, bone_to_humanoid, auxiliary_bones, auxiliary_bones_to_humanoid)
            - humanoid_to_bone: ヒューマノイドボーン名 -> 実際のボーン名
            - bone_to_humanoid: 実際のボーン名 -> ヒューマノイドボーン名
            - auxiliary_bones: ヒューマノイドボーン名 -> 補助ボーンリスト
            - auxiliary_bones_to_humanoid: 補助ボーン名 -> ヒューマノイドボーン名
    """
    humanoid_to_bone = {}
    bone_to_humanoid = {}
    auxiliary_bones = {}
    auxiliary_bones_to_humanoid = {}

    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
            bone_to_humanoid[bone_map["boneName"]] = bone_map["humanoidBoneName"]

    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        auxiliary_bones[humanoid_bone] = aux_set["auxiliaryBones"]
        for aux_bone in aux_set["auxiliaryBones"]:
            auxiliary_bones_to_humanoid[aux_bone] = humanoid_bone

    return humanoid_to_bone, bone_to_humanoid, auxiliary_bones, auxiliary_bones_to_humanoid
