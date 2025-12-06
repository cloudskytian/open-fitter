import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def remove_empty_vertex_groups(mesh_obj: bpy.types.Object) -> None:
    """Remove vertex groups that are empty or have zero weights for all vertices."""
    if not mesh_obj.type == 'MESH' or not mesh_obj.vertex_groups:
        return
        
    groups_to_remove = []
    for vgroup in mesh_obj.vertex_groups:
        has_weights = False
        for vert in mesh_obj.data.vertices:
            weight_index = vgroup.index
            for g in vert.groups:
                if g.group == weight_index and g.weight > 0.0005:
                    has_weights = True
                    break
            if has_weights:
                break
        if not has_weights:
            groups_to_remove.append(vgroup.name)

    for group_name in groups_to_remove:
        if group_name in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[group_name])
            print(f"Removed empty vertex group: {group_name}")
