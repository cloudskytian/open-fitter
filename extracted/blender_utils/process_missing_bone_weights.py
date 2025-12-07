import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from blender_utils.bone_utils import get_bone_parent_map
from blender_utils.merge_weights_to_parent import merge_weights_to_parent


def process_missing_bone_weights(base_mesh: bpy.types.Object, clothing_armature: bpy.types.Object, 
                               base_avatar_data: dict, clothing_avatar_data: dict, preserve_optional_humanoid_bones: bool) -> None:
    """
    Process weights for humanoid bones that exist in base avatar but not in clothing.
    """
    # Get bone names from clothing armature
    clothing_bone_names = set(bone.name for bone in clothing_armature.data.bones)

    # Create mappings for base avatar
    base_humanoid_to_bone = {}
    base_bone_to_humanoid = {}
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            base_humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
            base_bone_to_humanoid[bone_map["boneName"]] = bone_map["humanoidBoneName"]

    # Create mappings for clothing
    clothing_humanoid_to_bone = {}
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            clothing_humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]

    # Create auxiliary bones mapping
    aux_bones_map = {}
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        bone_name = base_humanoid_to_bone.get(humanoid_bone)
        if bone_name:
            aux_bones_map[bone_name] = aux_set["auxiliaryBones"]

    # Create parent map from bone hierarchy
    parent_map = get_bone_parent_map(base_avatar_data["boneHierarchy"])

    # Process each humanoid bone from base avatar
    for humanoid_name, bone_name in base_humanoid_to_bone.items():
        # Skip if bone exists in clothing armature
        if clothing_humanoid_to_bone.get(humanoid_name) in clothing_bone_names:
            continue

        # Check if this bone should be preserved when preserve_optional_humanoid_bones is True
        if preserve_optional_humanoid_bones:
            should_preserve = False
            
            # Condition 1: Chest exists in clothing, UpperChest missing in clothing but exists in base
            if (humanoid_name == "UpperChest" and 
                "Chest" in clothing_humanoid_to_bone and 
                clothing_humanoid_to_bone["Chest"] in clothing_bone_names and
                "UpperChest" not in clothing_humanoid_to_bone and
                "UpperChest" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving UpperChest bone weights due to Chest condition")
            
            # Condition 2: LeftLowerLeg exists in clothing, LeftFoot missing in clothing but exists in base
            elif (humanoid_name == "LeftFoot" and 
                  "LeftLowerLeg" in clothing_humanoid_to_bone and 
                  clothing_humanoid_to_bone["LeftLowerLeg"] in clothing_bone_names and
                  "LeftFoot" not in clothing_humanoid_to_bone and
                  "LeftFoot" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving LeftFoot bone weights due to LeftLowerLeg condition")
            
            # Condition 2: RightLowerLeg exists in clothing, RightFoot missing in clothing but exists in base
            elif (humanoid_name == "RightFoot" and 
                  "RightLowerLeg" in clothing_humanoid_to_bone and 
                  clothing_humanoid_to_bone["RightLowerLeg"] in clothing_bone_names and
                  "RightFoot" not in clothing_humanoid_to_bone and
                  "RightFoot" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving RightFoot bone weights due to RightLowerLeg condition")
            
            # Condition 3: LeftLowerLeg or LeftFoot exists in clothing, LeftToe missing in clothing but exists in base
            elif (humanoid_name == "LeftToe" and 
                  (("LeftLowerLeg" in clothing_humanoid_to_bone and clothing_humanoid_to_bone["LeftLowerLeg"] in clothing_bone_names) or
                   ("LeftFoot" in clothing_humanoid_to_bone and clothing_humanoid_to_bone["LeftFoot"] in clothing_bone_names)) and
                  "LeftToe" not in clothing_humanoid_to_bone and
                  "LeftToe" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving LeftToe bone weights due to LeftLowerLeg/LeftFoot condition")
            
            # Condition 3: RightLowerLeg or RightFoot exists in clothing, RightToe missing in clothing but exists in base
            elif (humanoid_name == "RightToe" and 
                  (("RightLowerLeg" in clothing_humanoid_to_bone and clothing_humanoid_to_bone["RightLowerLeg"] in clothing_bone_names) or
                   ("RightFoot" in clothing_humanoid_to_bone and clothing_humanoid_to_bone["RightFoot"] in clothing_bone_names)) and
                  "RightToe" not in clothing_humanoid_to_bone and
                  "RightToe" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving RightToe bone weights due to RightLowerLeg/RightFoot condition")

            elif (humanoid_name == "LeftBreast" and 
                  "LeftBreast" not in clothing_humanoid_to_bone and
                  ("Chest" in clothing_humanoid_to_bone or "UpperChest" in clothing_humanoid_to_bone) and 
                  (clothing_humanoid_to_bone["Chest"] in clothing_bone_names or clothing_humanoid_to_bone["UpperChest"] in clothing_bone_names) and
                  "LeftBreast" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving LeftBreast bone weights due to Chest condition")
            
            elif (humanoid_name == "RightBreast" and 
                  "RightBreast" not in clothing_humanoid_to_bone and
                  ("Chest" in clothing_humanoid_to_bone or "UpperChest" in clothing_humanoid_to_bone) and 
                  (clothing_humanoid_to_bone["Chest"] in clothing_bone_names or clothing_humanoid_to_bone["UpperChest"] in clothing_bone_names) and
                  "RightBreast" in base_humanoid_to_bone):
                should_preserve = True
                print(f"Preserving RightBreast bone weights due to Chest condition")
            
            if should_preserve:
                print(f"Skipping processing for preserved bone: {humanoid_name} ({bone_name})")
                continue

        print(f"Processing missing humanoid bone: {humanoid_name} ({bone_name})")
        
        # Find parent that exists in clothing armature
        current_bone = bone_name
        target_bone = None

        while current_bone and not target_bone:
            parent_bone = parent_map.get(current_bone)
            if not parent_bone:
                break

            parent_humanoid = base_bone_to_humanoid.get(parent_bone)
            if parent_humanoid and clothing_humanoid_to_bone.get(parent_humanoid) in clothing_bone_names:
                target_bone = base_humanoid_to_bone[parent_humanoid]
                break

            current_bone = parent_bone

        if target_bone:
            # Transfer main bone weights
            source_group = base_mesh.vertex_groups.get(bone_name)
            if source_group:
                merge_weights_to_parent(base_mesh, bone_name, target_bone)

                # Transfer auxiliary bone weights
                for aux_bone in aux_bones_map.get(bone_name, []):
                    if aux_bone in base_mesh.vertex_groups:
                        merge_weights_to_parent(base_mesh, aux_bone, target_bone)

                # Remove source groups
                if bone_name in base_mesh.vertex_groups:
                    base_mesh.vertex_groups.remove(base_mesh.vertex_groups[bone_name])
                for aux_bone in aux_bones_map.get(bone_name, []):
                    if aux_bone in base_mesh.vertex_groups:
                        base_mesh.vertex_groups.remove(base_mesh.vertex_groups[aux_bone])
