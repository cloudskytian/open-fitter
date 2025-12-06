import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def merge_auxiliary_to_humanoid_weights(mesh_obj: bpy.types.Object, avatar_data: dict) -> None:
    """Create missing Humanoid bone vertex groups and merge auxiliary weights."""
    # Map auxiliary bones to their Humanoid bones
    aux_to_humanoid = {}
    for aux_set in avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        bone_name = None
        # Get the actual bone name for the Humanoid bone
        for bone_map in avatar_data.get("humanoidBones", []):
            if bone_map["humanoidBoneName"] == humanoid_bone:
                bone_name = bone_map["boneName"]
                break
        if bone_name:
            for aux_bone in aux_set["auxiliaryBones"]:
                aux_to_humanoid[aux_bone] = bone_name

    # Check each auxiliary bone vertex group
    for aux_bone in aux_to_humanoid:
        if aux_bone in mesh_obj.vertex_groups:
            humanoid_bone = aux_to_humanoid[aux_bone]
            # Create Humanoid bone group if it doesn't exist
            if humanoid_bone not in mesh_obj.vertex_groups:
                print(f"Creating missing Humanoid bone group {humanoid_bone} for {mesh_obj.name}")
                mesh_obj.vertex_groups.new(name=humanoid_bone)

            # Get the vertex groups
            aux_group = mesh_obj.vertex_groups[aux_bone]
            humanoid_group = mesh_obj.vertex_groups[humanoid_bone]

            # Transfer weights from auxiliary to humanoid group
            for vert in mesh_obj.data.vertices:
                aux_weight = 0
                for group in vert.groups:
                    if group.group == aux_group.index:
                        aux_weight = group.weight
                        break
                
                if aux_weight > 0:
                    # Add weight to humanoid bone group
                    humanoid_group.add([vert.index], aux_weight, 'ADD')

            # Remove auxiliary bone vertex group
            mesh_obj.vertex_groups.remove(aux_group)
            print(f"Merged weights from {aux_bone} to {humanoid_bone} in {mesh_obj.name}")
