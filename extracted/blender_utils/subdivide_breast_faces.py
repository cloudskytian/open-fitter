import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blender_utils.subdivide_faces import subdivide_faces


def subdivide_breast_faces(target_obj, clothing_avatar_data):
    # subdivisionがTrueの場合、胸のボーンに関連する面を事前に細分化
    if clothing_avatar_data:
        breast_related_faces = set()
        
        # LeftBreastとRightBreastのボーン名を取得
        breast_bone_names = []
        for bone_mapping in clothing_avatar_data.get("humanoidBones", []):
            if bone_mapping["humanoidBoneName"] in ["LeftBreast", "RightBreast"]:
                breast_bone_names.append(bone_mapping["boneName"])
        
        # 補助ボーンも取得
        for aux_bone_group in clothing_avatar_data.get("auxiliaryBones", []):
            if aux_bone_group["humanoidBoneName"] in ["LeftBreast", "RightBreast"]:
                breast_bone_names.extend(aux_bone_group["auxiliaryBones"])
        
        # 胸のボーンに関連する頂点を特定
        breast_vertices = set()
        for bone_name in breast_bone_names:
            if bone_name in target_obj.vertex_groups:
                vertex_group = target_obj.vertex_groups[bone_name]
                for vertex in target_obj.data.vertices:
                    for group in vertex.groups:
                        if group.group == vertex_group.index and group.weight > 0.001:
                            breast_vertices.add(vertex.index)
        
        # 胸の頂点を含む面を特定
        if breast_vertices:
            for face in target_obj.data.polygons:
                if any(vertex_idx in breast_vertices for vertex_idx in face.vertices):
                    breast_related_faces.add(face.index)
            
            if breast_related_faces:
                print(f"Subdividing {len(breast_related_faces)} breast-related faces...")
                subdivide_faces(target_obj, list(breast_related_faces), cuts=1)
