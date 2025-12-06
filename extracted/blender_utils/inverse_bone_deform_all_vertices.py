import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from algo_utils.get_vertex_groups_and_weights import get_vertex_groups_and_weights
from mathutils import Matrix


def inverse_bone_deform_all_vertices(armature_obj, mesh_obj):
    """
    メッシュオブジェクトの評価後の頂点のワールド座標から、
    現在のArmatureオブジェクトのポーズの逆変換をすべての頂点に対して行う
    
    Parameters:
        armature_obj: Armatureオブジェクト
        mesh_obj: メッシュオブジェクト
        
    Returns:
        np.ndarray: すべての頂点の逆変換後の座標（ローカル座標）
        

        通常のボーン変形: 変形後 = Σ(weight_i × bone_matrix_i) × 変形前
        この関数の逆変換: 変形前 = [Σ(weight_i × bone_matrix_i)]^(-1) × 変形後
    """
    
    if not armature_obj or armature_obj.type != 'ARMATURE':
        raise ValueError("有効なArmatureオブジェクトを指定してください")
    
    if not mesh_obj or mesh_obj.type != 'MESH':
        raise ValueError("有効なメッシュオブジェクトを指定してください")
    
    # ワールド座標に変換（変形後の頂点位置）
    vertices = [v.co.copy() for v in mesh_obj.data.vertices]
    
    # 結果を格納するリスト
    inverse_transformed_vertices = []
    
    print(f"ボーン変形の逆変換を開始: {len(vertices)}頂点")
    
    # 各頂点に対して逆変換を適用
    for vertex_index in range(len(vertices)):
        pos = vertices[vertex_index]
        
        # 頂点のボーンウェイトを取得
        weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
        
        if not weights:
            # ウェイトがない場合はそのまま追加
            print(f"警告: 頂点 {vertex_index} のウェイトがないため、単位行列を使用します")
            inverse_transformed_vertices.append(pos)
            continue
        
        # ウェイト付き合成変形行列を計算
        combined_matrix = Matrix.Identity(4)
        combined_matrix.zero()
        total_weight = 0.0
        
        for bone_name, weight in weights.items():
            if weight > 0 and bone_name in armature_obj.data.bones:
                bone = armature_obj.data.bones[bone_name]
                pose_bone = armature_obj.pose.bones.get(bone_name)
                if bone and pose_bone:
                    # ボーンの変形行列を計算
                    # この行列は、レストポーズからポーズ後への変形を表す
                    bone_matrix = pose_bone.matrix @ \
                                  bone.matrix_local.inverted()
                    
                    # ウェイトを考慮して行列を加算
                    combined_matrix += bone_matrix * weight
                    total_weight += weight
        
        # ウェイトの合計で正規化
        if total_weight > 0:
            combined_matrix = combined_matrix * (1.0 / total_weight)
        else:
            # ウェイトがない場合は単位行列
            print(f"警告: 頂点 {vertex_index} のウェイトがないため、単位行列を使用します")
            combined_matrix = Matrix.Identity(4)
        
        # 合成行列の逆行列を計算
        try:
            inverse_matrix = combined_matrix.inverted()
        except:
            # 逆行列が計算できない場合は単位行列を使用
            inverse_matrix = Matrix.Identity(4)
            print(f"警告: 頂点 {vertex_index} の逆行列を計算できませんでした")
        
        # 逆変換を適用
        # inverse_matrix を適用して「レストポーズのローカル座標」を取得
        rest_pose_pos = inverse_matrix @ pos
        
        inverse_transformed_vertices.append(rest_pose_pos)
        
        # 進捗表示（1000頂点ごと）
        if (vertex_index + 1) % 1000 == 0:
            print(f"進捗: {vertex_index + 1}/{len(vertices)} 頂点処理完了")
    
    print(f"ボーン変形の逆変換が完了しました")

    # 変形後の頂点をメッシュに適用
    if mesh_obj.data.shape_keys:
        for shape_key in mesh_obj.data.shape_keys.key_blocks:
            if shape_key.name != "Basis":
                for i, vert in enumerate(shape_key.data):
                    vert.co += inverse_transformed_vertices[i] - vertices[i]
        basis_shape_key = mesh_obj.data.shape_keys.key_blocks["Basis"]
        for i, vert in enumerate(basis_shape_key.data):
            vert.co = inverse_transformed_vertices[i]

    for vertex_index, pos in enumerate(inverse_transformed_vertices):
        mesh_obj.data.vertices[vertex_index].co = pos
    
    # numpy配列に変換して返す（Vector型からnumpy配列へ）
    result = np.array([[v[0], v[1], v[2]] for v in inverse_transformed_vertices])
    
    return result
