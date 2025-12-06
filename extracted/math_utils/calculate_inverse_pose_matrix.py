import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algo_utils.get_vertex_groups_and_weights import get_vertex_groups_and_weights
from mathutils import Matrix


def calculate_inverse_pose_matrix(mesh_obj, armature_obj, vertex_index):
    """指定された頂点のポーズ逆行列を計算"""

    # 頂点グループとウェイトの取得
    weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
    if not weights:
        print(f"頂点 {vertex_index} にウェイトが割り当てられていません")
        return None

    # 最終的な変換行列の初期化
    final_matrix = Matrix.Identity(4)
    final_matrix.zero()
    total_weight = 0

    # 各ボーンの影響を計算
    for bone_name, weight in weights.items():
        if weight > 0 and bone_name in armature_obj.data.bones:
            bone = armature_obj.data.bones[bone_name]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if bone and pose_bone:
                # ボーンの最終的な行列を計算
                mat = armature_obj.matrix_world @ \
                      pose_bone.matrix @ \
                      bone.matrix_local.inverted() @ \
                      armature_obj.matrix_world.inverted()
                
                # ウェイトを考慮して行列を加算
                final_matrix += mat * weight
                total_weight += weight

    # ウェイトの合計で正規化
    if total_weight > 0:
        final_matrix = final_matrix * (1.0 / total_weight)

    # 逆行列を計算して返す
    try:
        return final_matrix.inverted()
    except Exception as e:
        print(f"error: {e}")
        return Matrix.Identity(4)
