import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from blender_utils.get_deformation_bones import get_deformation_bones
from mathutils import Matrix, Vector
from misc_utils.get_deformation_field_multi_step import get_deformation_field_multi_step
from scipy.spatial import cKDTree


def apply_bone_field_delta(armature_obj: bpy.types.Object, field_data_path: str, avatar_data: dict) -> None:
    """
    ボーンにDeformation Fieldを適用
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
        field_data_path: Deformation Fieldデータのパス
        avatar_data: アバターデータ
    """
    # データの読み込み
    field_info = get_deformation_field_multi_step(field_data_path)
    all_field_points = field_info['all_field_points']
    all_delta_positions = field_info['all_delta_positions']
    all_field_weights = field_info['field_weights']
    field_matrix = field_info['world_matrix']
    field_matrix_inv = field_info['world_matrix_inv']
    k_neighbors = field_info['kdtree_query_k']
    
    # 変形対象のボーンを取得
    deform_bones = get_deformation_bones(armature_obj, avatar_data)

    bpy.ops.object.mode_set(mode='OBJECT')
    
    # すべての選択を解除
    bpy.ops.object.select_all(action='DESELECT')

    # アクティブオブジェクトを設定
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # ------------------------------------------------------------------
    # 【追加処理：処理前の親子Head位置の記録】
    # deform_bones内で「子が１つのみ」のボーンについて、親ボーンとその子ボーンの
    # ワールド空間でのHead位置を記録しておく。
    # ------------------------------------------------------------------
    original_heads = {}
    for bone in armature_obj.pose.bones:
        if bone.name in deform_bones and len(bone.children) == 1:
            child = bone.children[0]
            parent_head_world = armature_obj.matrix_world @ (bone.matrix @ Vector((0, 0, 0)))
            child_head_world = armature_obj.matrix_world @ (child.matrix @ Vector((0, 0, 0)))
            # コピーして記録（後で参照するため）
            original_heads[bone.name] = (parent_head_world.copy(), child_head_world.copy())
    
    def process_bone_hierarchy(bone_name, parent_world_displacement, kdtree, delta_positions):
        """ボーン階層を再帰的に処理"""
        
        bone = armature_obj.pose.bones[bone_name]
        ret_displacement = parent_world_displacement

        if bone_name in deform_bones:
            base_matrix = armature_obj.data.bones[bone.name].matrix_local
            current_world_matrix = armature_obj.matrix_world @ (bone.matrix @ base_matrix.inverted())
               
            # ヘッドの位置を取得
            head_world = (armature_obj.matrix_world @ bone.matrix @ Vector((0, 0, 0))) - parent_world_displacement
            
            # ヘッドのフィールド空間での座標を計算
            head_field = field_matrix_inv @ head_world
        
            # ヘッドの最近接点の検索
            head_distances, head_indices = kdtree.query(head_field, k=k_neighbors)
        
            # ヘッドの変位を計算
            weights = 1.0 / (head_distances + 0.0001)
            weights /= weights.sum()
            deltas = delta_positions[head_indices]
            head_displacement = (deltas * weights[:, np.newaxis]).sum(axis=0)

            # ワールド空間での変位を計算
            world_displacement = (field_matrix.to_3x3() @ Vector(head_displacement)) - parent_world_displacement

            new_matrix = Matrix.Translation(world_displacement)
            combined_matrix = new_matrix @ current_world_matrix
            bone.matrix = armature_obj.matrix_world.inverted() @ combined_matrix @ base_matrix

            ret_displacement = world_displacement + parent_world_displacement
        
        # 子ボーンを処理
        for child in bone.children:
            process_bone_hierarchy(child.name, ret_displacement, kdtree, delta_positions)
    
    # 各ステップの変位を累積的に適用
    num_steps = len(all_field_points)
    for step in range(num_steps):
        field_points = all_field_points[step]
        delta_positions = all_delta_positions[step]
        # KDTreeを使用して近傍点を検索（各ステップで新しいKDTreeを構築）
        kdtree = cKDTree(field_points)

        # ルートボーンから処理を開始
        root_displacement = Vector((0, 0, 0))
        root_bones = [bone.name for bone in armature_obj.pose.bones if not bone.parent]
        for root_bone in root_bones:
            process_bone_hierarchy(root_bone, root_displacement, kdtree, delta_positions)
        
        bpy.context.view_layer.update()


    # ------------------------------------------------------------------
    # 【追加処理：回転補正の適用】
    # 対象のdeform_bone（子が１つのみ）について、処理前と処理後の
    # 親子のHead間の方向ベクトルの変化から回転差分を求め、その回転を
    # 親ボーンに適用するとともに、子ボーンにはその影響を打ち消す補正をかける。
    # ------------------------------------------------------------------
    # for parent_name, (old_parent_head, old_child_head) in original_heads.items():
    #     parent_bone = armature_obj.pose.bones.get(parent_name)
    #     if not parent_bone or len(parent_bone.children) != 1:
    #         continue
    #     child_bone = parent_bone.children[0]

    #     # 【処理後】の親・子のHead位置を計算（ワールド座標）
    #     new_parent_head = armature_obj.matrix_world @ (parent_bone.matrix @ Vector((0, 0, 0)))
    #     new_child_head = armature_obj.matrix_world @ (child_bone.matrix @ Vector((0, 0, 0)))

    #     # 処理前と処理後の方向ベクトルを計算（子Head - 親Head）
    #     old_dir = old_child_head - old_parent_head
    #     new_dir = new_child_head - new_parent_head
    #     # もしどちらかのベクトルがゼロ長の場合はスキップ
    #     if old_dir.length == 0.001 or new_dir.length == 0.001:
    #         continue
    #     old_dir.normalize()
    #     new_dir.normalize()

    #     # 「old_dir」から「new_dir」へ回転させる回転差分を求める
    #     rot_diff = old_dir.rotation_difference(new_dir)

    #     # 親ボーンに対して、親のHeadを中心にrot_diffを適用する
    #     parent_world_matrix = armature_obj.matrix_world @ parent_bone.matrix
    #     T = Matrix.Translation(new_parent_head)
    #     T_inv = Matrix.Translation(-new_parent_head)
    #     rot_matrix = rot_diff.to_matrix().to_4x4()
    #     R = T @ rot_matrix @ T_inv
    #     new_parent_world_matrix = R @ parent_world_matrix
    #     parent_bone.matrix = armature_obj.matrix_world.inverted() @ new_parent_world_matrix

    #     # 子ボーンには、親の回転変化の影響が及ばないよう、逆の補正を適用する
    #     child_world_matrix = armature_obj.matrix_world @ child_bone.matrix
    #     compensation = T @ rot_matrix.inverted() @ T_inv
    #     new_child_world_matrix = compensation @ child_world_matrix
    #     child_bone.matrix = armature_obj.matrix_world.inverted() @ new_child_world_matrix

    #     bpy.context.view_layer.update()

    bpy.context.view_layer.update()
    
    # オブジェクトモードに戻る
    bpy.ops.object.mode_set(mode='OBJECT')
