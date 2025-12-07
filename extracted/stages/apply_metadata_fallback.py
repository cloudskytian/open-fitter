import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time


def apply_metadata_fallback(context):
    metadata_time_start = time.time()
    if context.cloth_metadata:
        mesh_name = context.target_obj.name
        if mesh_name in context.cloth_metadata:
            vertex_max_distances = context.cloth_metadata[mesh_name]
            print(f"  メッシュのクロスメタデータを処理: {mesh_name}")
            count = 0
            for vert_idx in range(len(context.target_obj.data.vertices)):
                max_distance = float(vertex_max_distances.get(str(vert_idx), 10.0))
                if max_distance > 1.0:
                    if vert_idx in context.original_humanoid_weights:
                        for group in context.target_obj.vertex_groups:
                            if group.name in context.bone_groups:
                                try:
                                    group.remove([vert_idx])
                                except RuntimeError:
                                    continue
                        for group_name, weight in context.original_humanoid_weights[vert_idx].items():
                            if group_name in context.target_obj.vertex_groups:
                                context.target_obj.vertex_groups[group_name].add([vert_idx], weight, "REPLACE")
                        count += 1
            print(f"  処理された頂点数: {count}")
    metadata_time = time.time() - metadata_time_start
    print(f"  クロスメタデータ処理: {metadata_time:.2f}秒")
