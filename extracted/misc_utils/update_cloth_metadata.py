import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json

import bpy


def update_cloth_metadata(metadata_dict: dict, output_path: str, vertex_index_mapping: dict) -> None:
    """
    ClothMetadataの頂点位置を更新し、指定されたパスに保存する
    
    Parameters:
        metadata_dict: 元のClothMetadataの辞書
        output_path: 保存先のパス
        vertex_index_mapping: Unity頂点インデックスからBlender頂点インデックスへのマッピング
    """
    # 各メッシュについて処理
    for cloth_data in metadata_dict.get("clothMetadata", []):
        mesh_name = cloth_data["meshName"]
        mesh_obj = bpy.data.objects.get(mesh_name)
        
        if not mesh_obj or mesh_obj.type != 'MESH':
            print(f"Warning: Mesh {mesh_name} not found")
            continue
            
        # このメッシュのマッピング情報を取得
        mesh_mapping = vertex_index_mapping.get(mesh_name, {})
        if not mesh_mapping:
            print(f"Warning: No vertex mappings found for {mesh_name}")
            continue
            
        # 評価済みメッシュを取得（モディファイア適用後の状態）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = mesh_obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated_obj.data
        
        # vertexDataを更新
        for i, data in enumerate(cloth_data.get("vertexData", [])):
            # Unity頂点インデックスに対応するBlender頂点インデックスを取得
            blender_vert_idx = mesh_mapping.get(i)
            
            if blender_vert_idx is not None and blender_vert_idx < len(evaluated_mesh.vertices):
                # ワールド座標を取得
                world_pos = evaluated_obj.matrix_world @ evaluated_mesh.vertices[blender_vert_idx].co
                
                # Blender座標系からUnity座標系に変換
                data["position"]["x"] = -world_pos.x
                data["position"]["y"] = world_pos.z
                data["position"]["z"] = -world_pos.y
            else:
                print(f"Warning: No mapping found for Unity vertex {i} in {mesh_name}")

    # 更新したデータを保存
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=4)
        print(f"Updated cloth metadata saved to {output_path}")
    except Exception as e:
        print(f"Error saving cloth metadata: {e}")
