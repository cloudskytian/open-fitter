import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def merge_weights_to_parent(mesh_obj: bpy.types.Object, source_bone: str, target_bone: str) -> None:
    """
    Merge weights from source bone to target bone and remove source bone vertex group.
    
    Parameters:
        mesh_obj: Mesh object to process
        source_bone: Name of the source bone (whose weights will be moved)
        target_bone: Name of the target bone (that will receive the weights)
    """
    source_group = mesh_obj.vertex_groups.get(source_bone)
    target_group = mesh_obj.vertex_groups.get(target_bone)
    
    if not source_group:
        return
        
    if not target_group:
        # Create target group if it doesn't exist
        target_group = mesh_obj.vertex_groups.new(name=target_bone)
    
    # Transfer weights
    for vert in mesh_obj.data.vertices:
        source_weight = 0
        for group in vert.groups:
            if group.group == source_group.index:
                source_weight = group.weight
                break
                
        if source_weight > 0:
            target_group.add([vert.index], source_weight, 'ADD')
    
    # Remove source group
    mesh_obj.vertex_groups.remove(source_group)
    print(f"Merged weights from {source_bone} to {target_bone} in {mesh_obj.name}")
