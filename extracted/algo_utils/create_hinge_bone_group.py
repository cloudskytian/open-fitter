import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from blender_utils.get_evaluated_mesh import get_evaluated_mesh
from scipy.spatial import cKDTree


def create_hinge_bone_group(obj: bpy.types.Object, armature: bpy.types.Object, avatar_data: dict) -> None:
    """
    Create a hinge bone group.
    """
    bone_groups = get_humanoid_and_auxiliary_bone_groups(avatar_data)

    # 衣装アーマチュアのボーングループも含めた対象グループを作成
    all_deform_groups = set(bone_groups)
    if armature:
        all_deform_groups.update(bone.name for bone in armature.data.bones)

    # original_groupsからbone_groupsを除いたグループのウェイトを保存
    original_non_humanoid_groups = all_deform_groups - bone_groups

    cloth_bm = get_evaluated_mesh(obj)
    cloth_bm.verts.ensure_lookup_table()
    cloth_bm.faces.ensure_lookup_table()
    vertex_coords = np.array([v.co for v in cloth_bm.verts])
    kdtree = cKDTree(vertex_coords)

    hinge_bone_group = obj.vertex_groups.new(name="HingeBone")
    for bone_name in original_non_humanoid_groups:
        bone = armature.pose.bones.get(bone_name)
        if bone.parent and bone.parent.name in bone_groups:
            group_index = obj.vertex_groups.find(bone_name)
            print(f"Processing hinge bone: {bone_name}")
            print(f"Bone parent: {bone.parent.name}")
            print(f"Group index: {group_index}")
            if group_index != -1:
                bone_head = armature.matrix_world @ bone.head
                neighbor_indices = kdtree.query_ball_point(bone_head, 0.01)
                for index in neighbor_indices:
                    for g in obj.data.vertices[index].groups:
                        if g.group == group_index:
                            weight = g.weight
                            hinge_bone_group.add([index], weight, 'REPLACE')
                            print(f"Added weight to {index}")
                            break
