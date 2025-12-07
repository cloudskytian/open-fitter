import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Optional

import bmesh
import bpy
from add_pose_from_json import add_pose_from_json
from algo_utils.find_humanoid_parent_in_hierarchy import (
    find_humanoid_parent_in_hierarchy,
)
from algo_utils.get_humanoid_and_auxiliary_bone_groups_with_intermediate import (
    get_humanoid_and_auxiliary_bone_groups_with_intermediate,
)
from blender_utils.apply_pose_as_rest import apply_pose_as_rest
from blender_utils.inverse_bone_deform_all_vertices import (
    inverse_bone_deform_all_vertices,
)
from math_utils.copy_bone_transform import copy_bone_transform
from mathutils.bvhtree import BVHTree


class _ReplaceHumanoidBonesContext:
    """State holder for humanoid bone replacement steps."""

    def __init__(
        self,
        base_armature: bpy.types.Object,
        clothing_armature: bpy.types.Object,
        base_avatar_data: dict,
        clothing_avatar_data: dict,
        preserve_humanoid_bones: bool,
        base_pose_filepath: Optional[str],
        clothing_meshes: list,
        process_upper_chest: bool,
    ):
        # Input parameters
        self.base_armature = base_armature
        self.clothing_armature = clothing_armature
        self.base_avatar_data = base_avatar_data
        self.clothing_avatar_data = clothing_avatar_data
        self.preserve_humanoid_bones = preserve_humanoid_bones
        self.base_pose_filepath = base_pose_filepath
        self.clothing_meshes = clothing_meshes
        self.process_upper_chest = process_upper_chest

        # Original state
        self.original_active = bpy.context.active_object
        self.original_mode = self.original_active.mode if self.original_active else 'OBJECT'

        # Bone mappings
        self.base_humanoid_map: dict[str, str] = {}  # humanoidName -> boneName
        self.clothing_humanoid_map: dict[str, str] = {}  # boneName -> humanoidName
        self.clothing_bones_to_humanoid: dict[str, str] = {}
        self.base_bone_to_humanoid: dict[str, str] = {}  # boneName -> humanoidName
        self.aux_to_humanoid: dict[str, str] = {}
        self.humanoid_to_aux: dict[str, list[str]] = {}
        self.humanoid_to_aux_base: dict[str, list[str]] = {}
        self.missing_humanoid_bones: set[str] = set()

        # Derived data
        self.bones_to_replace: set[str] = set()
        self.base_bones: set[str] = set()
        self.humanoid_bones_to_preserve: set[str] = set()

        # BVH data
        self.base_mesh: Optional[bpy.types.Object] = None
        self.bmesh_data: Optional[bmesh.types.BMesh] = None
        self.bvh: Optional[BVHTree] = None
        self.base_group_index_to_name: dict[int, str] = {}

        # Modifier backup
        self.armature_modifiers: list[dict] = []
        self.clothing_obj_list: list[bpy.types.Object] = []

        # Intermediate data for bone replacement
        self.clothing_bone_data: dict[str, dict] = {}
        self.parent_bones: dict[str, str] = {}
        self.children_to_update: list[str] = []
        self.base_bone_parents: dict[str, Optional[str]] = {}
        self.original_bone_data: dict[str, dict] = {}
        self.new_bones: dict[str, bpy.types.EditBone] = {}

    # =========================================================================
    # Step 1: Create bone mappings
    # =========================================================================
    def create_bone_mappings(self):
        """Create all bone name mappings from avatar data."""
        # Base avatar mappings
        self.base_humanoid_map = {
            bone_map["humanoidBoneName"]: bone_map["boneName"]
            for bone_map in self.base_avatar_data["humanoidBones"]
        }
        self.base_bone_to_humanoid = {
            bone_map["boneName"]: bone_map["humanoidBoneName"]
            for bone_map in self.base_avatar_data["humanoidBones"]
        }

        # Clothing avatar mappings
        self.clothing_humanoid_map = {
            bone_map["boneName"]: bone_map["humanoidBoneName"]
            for bone_map in self.clothing_avatar_data["humanoidBones"]
        }
        self.clothing_bones_to_humanoid = self.clothing_humanoid_map.copy()

        # Note: missing_humanoid_bones is intentionally empty for now
        # missing_humanoid_bones = clothing_humanoid_bones - base_humanoid_bones
        self.missing_humanoid_bones = set()

    def build_auxiliary_mappings(self):
        """Build auxiliary bone mappings."""
        # Map auxiliary bones to humanoid bones (clothing)
        for aux_set in self.clothing_avatar_data.get("auxiliaryBones", []):
            humanoid_bone = aux_set["humanoidBoneName"]
            if humanoid_bone not in self.missing_humanoid_bones:
                for aux_bone in aux_set["auxiliaryBones"]:
                    self.aux_to_humanoid[aux_bone] = humanoid_bone
                self.humanoid_to_aux[humanoid_bone] = aux_set["auxiliaryBones"]

        # Map humanoid bones to auxiliary bones (base)
        for aux_set in self.base_avatar_data.get("auxiliaryBones", []):
            self.humanoid_to_aux_base[aux_set["humanoidBoneName"]] = aux_set["auxiliaryBones"]

    def build_bones_to_replace(self):
        """Build the set of bones to be replaced."""
        for bone_map in self.clothing_avatar_data["humanoidBones"]:
            if bone_map["humanoidBoneName"] not in self.missing_humanoid_bones:
                self.bones_to_replace.add(bone_map["boneName"])

        for aux_set in self.clothing_avatar_data.get("auxiliaryBones", []):
            if aux_set["humanoidBoneName"] not in self.missing_humanoid_bones:
                self.bones_to_replace.update(aux_set["auxiliaryBones"])

        print(f"bones_to_replace: {self.bones_to_replace}")

    def get_base_bones(self):
        """Get humanoid and auxiliary bone groups from base armature."""
        self.base_bones = get_humanoid_and_auxiliary_bone_groups_with_intermediate(
            self.base_armature, self.base_avatar_data
        )

    def build_humanoid_bones_to_preserve(self):
        """Build the set of humanoid bones to preserve."""
        if self.preserve_humanoid_bones:
            self.humanoid_bones_to_preserve = {
                bone_name
                for bone_name, humanoid_name in self.clothing_bones_to_humanoid.items()
                if humanoid_name not in self.missing_humanoid_bones
            }
        else:
            self.humanoid_bones_to_preserve = set()

    # =========================================================================
    # Step 2: Build BVH tree
    # =========================================================================
    def build_bvh_tree(self):
        """Build BVH tree from base mesh for spatial queries."""
        self.base_mesh = bpy.data.objects.get("Body.BaseAvatar")
        if not self.base_mesh:
            raise Exception("Body.BaseAvatar not found")

        self.bmesh_data = bmesh.new()
        self.bmesh_data.from_mesh(self.base_mesh.data)
        self.bmesh_data.faces.ensure_lookup_table()
        self.bmesh_data.transform(self.base_mesh.matrix_world)
        self.bvh = BVHTree.FromBMesh(self.bmesh_data)

        self.base_group_index_to_name = {
            group.index: group.name for group in self.base_mesh.vertex_groups
        }

    def free_bvh_resources(self):
        """Free BVH and bmesh resources."""
        if self.bmesh_data:
            self.bmesh_data.free()
            self.bmesh_data = None

    # =========================================================================
    # Step 3: Backup armature modifiers
    # =========================================================================
    def backup_armature_modifiers(self):
        """Backup and temporarily remove armature modifiers."""
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                for modifier in obj.modifiers[:]:
                    if modifier.type == 'ARMATURE' and modifier.object == self.clothing_armature:
                        mod_settings = {
                            'object': obj,
                            'name': modifier.name,
                            'target': modifier.object,
                            'vertex_group': modifier.vertex_group,
                            'invert_vertex_group': modifier.invert_vertex_group,
                            'use_vertex_groups': modifier.use_vertex_groups,
                            'use_bone_envelopes': modifier.use_bone_envelopes,
                            'use_deform_preserve_volume': modifier.use_deform_preserve_volume,
                        }
                        self.armature_modifiers.append(mod_settings)
                        obj.modifiers.remove(modifier)
                        self.clothing_obj_list.append(obj)

    def restore_armature_modifiers(self):
        """Restore armature modifiers from backup."""
        for mod_settings in self.armature_modifiers:
            obj = mod_settings['object']
            modifier = obj.modifiers.new(name=mod_settings['name'], type='ARMATURE')
            modifier.object = mod_settings['target']
            modifier.vertex_group = mod_settings['vertex_group']
            modifier.invert_vertex_group = mod_settings['invert_vertex_group']
            modifier.use_vertex_groups = mod_settings['use_vertex_groups']
            modifier.use_bone_envelopes = mod_settings['use_bone_envelopes']
            modifier.use_deform_preserve_volume = mod_settings['use_deform_preserve_volume']

    # =========================================================================
    # Step 4: Apply initial pose
    # =========================================================================
    def apply_initial_pose(self):
        """Apply initial pose to clothing armature if base_pose_filepath is provided."""
        if self.base_pose_filepath:
            print(f"Applying clothing base pose from {self.base_pose_filepath}")
            add_pose_from_json(
                self.clothing_armature, self.base_pose_filepath, self.clothing_avatar_data, invert=True
            )
            apply_pose_as_rest(self.clothing_armature)

    # =========================================================================
    # Step 5: Collect clothing bone data
    # =========================================================================
    def collect_clothing_bone_data(self):
        """Collect clothing bone positions and their candidate parent bones."""
        clothing_matrix_world = self.clothing_armature.matrix_world

        for bone in self.clothing_armature.pose.bones:
            if bone.parent and bone.parent.name in self.bones_to_replace and bone.name not in self.bones_to_replace:
                head_pos = clothing_matrix_world @ bone.head

                # Get humanoid name of parent bone
                parent_humanoid = self._get_parent_humanoid_name(bone.parent.name)

                # If parent_humanoid is not in base_humanoid_map, find valid parent in hierarchy
                if parent_humanoid and parent_humanoid not in self.base_humanoid_map:
                    parent_humanoid = find_humanoid_parent_in_hierarchy(
                        bone.parent.name, self.clothing_avatar_data, self.base_avatar_data
                    )

                if parent_humanoid and parent_humanoid in self.base_humanoid_map:
                    candidate_bones, sub_parent_humanoid = self._get_candidate_bones(parent_humanoid)

                    self.clothing_bone_data[bone.name] = {
                        'head_pos': head_pos,
                        'candidate_bones': candidate_bones,
                        'parent_humanoid': self.base_humanoid_map[parent_humanoid],
                        'sub_parent_humanoid': sub_parent_humanoid,
                    }

    def _get_parent_humanoid_name(self, bone_name: str) -> Optional[str]:
        """Get humanoid name for a bone."""
        if bone_name in self.clothing_humanoid_map:
            return self.clothing_humanoid_map[bone_name]
        elif bone_name in self.aux_to_humanoid:
            return self.aux_to_humanoid[bone_name]
        return None

    def _get_candidate_bones(self, parent_humanoid: str) -> tuple[set[str], Optional[str]]:
        """Get candidate bones for parent matching."""
        candidate_bones = {self.base_humanoid_map[parent_humanoid]}
        if parent_humanoid in self.humanoid_to_aux_base:
            candidate_bones.update(self.humanoid_to_aux_base[parent_humanoid])

        sub_parent_humanoid = None
        if parent_humanoid == 'Chest' and 'UpperChest' in self.base_humanoid_map and self.process_upper_chest:
            sub_parent_humanoid = self.base_humanoid_map['UpperChest']
            candidate_bones.add(sub_parent_humanoid)
            if 'UpperChest' in self.humanoid_to_aux_base:
                candidate_bones.update(self.humanoid_to_aux_base['UpperChest'])

        return candidate_bones, sub_parent_humanoid

    # =========================================================================
    # Step 6: Find parent bones
    # =========================================================================
    def find_parent_bones(self):
        """Find parent bones for each clothing bone using weight-based scoring."""
        for bone_name, data in self.clothing_bone_data.items():
            chosen_parent = self._find_parent_for_bone(bone_name, data)
            self.parent_bones[bone_name] = chosen_parent
            print(f"bone_name: {bone_name}, chosen_parent: {chosen_parent}")

    def _find_parent_for_bone(self, bone_name: str, data: dict) -> str:
        """Find the best parent bone for a single clothing bone."""
        head_pos = data['head_pos']
        candidate_bones = data['candidate_bones']
        parent_humanoid = data['parent_humanoid']
        sub_parent_humanoid = data.get('sub_parent_humanoid', None)

        # Score candidate bones based on vertex weights
        bone_scores = self._calculate_bone_scores(bone_name, candidate_bones)

        chosen_parent = None
        if bone_scores:
            print(f"bone_scores: {bone_scores}")
            chosen_parent = max(bone_scores.items(), key=lambda item: item[1])[0]

        # Avoid self-reference
        if chosen_parent and chosen_parent == bone_name:
            if bone_name in self.clothing_armature.data.bones:
                parent_bone = self.clothing_armature.data.bones.get(bone_name).parent
                if parent_bone:
                    chosen_parent = parent_bone.name
                    if chosen_parent not in candidate_bones:
                        chosen_parent = None

        if chosen_parent:
            return chosen_parent

        # Fallback: use distance comparison if sub_parent_humanoid exists
        return self._fallback_parent_selection(head_pos, parent_humanoid, sub_parent_humanoid)

    def _calculate_bone_scores(self, bone_name: str, candidate_bones: set[str]) -> dict[str, float]:
        """Calculate weight scores for candidate bones based on clothing mesh vertices."""
        bone_scores = defaultdict(float)
        weighted_vertices = []

        for mesh_obj in self.clothing_meshes:
            if mesh_obj.type != 'MESH':
                continue

            vg_lookup = {vg.name: vg.index for vg in mesh_obj.vertex_groups}
            if bone_name not in vg_lookup:
                continue

            target_group_index = vg_lookup[bone_name]
            mesh_data = mesh_obj.data
            mesh_world_matrix = mesh_obj.matrix_world

            for vertex in mesh_data.vertices:
                weight = 0.0
                for g in vertex.groups:
                    if g.group == target_group_index:
                        weight = g.weight
                        break
                if weight >= 0.001:
                    vertex_world_co = mesh_world_matrix @ vertex.co
                    weighted_vertices.append((vertex_world_co, weight))

            print(f"bone_name: {bone_name}, weighted_vertices: {len(weighted_vertices)}")

        if weighted_vertices:
            weighted_vertices.sort(key=lambda item: item[1], reverse=True)
            top_vertices = weighted_vertices[:100]

            for vertex_world_co, _ in top_vertices:
                closest_point, _, face_idx, _ = self.bvh.find_nearest(vertex_world_co)
                if closest_point is None or face_idx is None:
                    continue

                face = self.bmesh_data.faces[face_idx]
                vertex_indices = [v.index for v in face.verts]
                closest_vert_idx = min(
                    vertex_indices,
                    key=lambda idx: (self.base_mesh.data.vertices[idx].co - closest_point).length,
                )

                vertex = self.base_mesh.data.vertices[closest_vert_idx]
                for group_element in vertex.groups:
                    group_name = self.base_group_index_to_name.get(group_element.group)
                    if group_name in candidate_bones:
                        bone_scores[group_name] += group_element.weight

        return bone_scores

    def _fallback_parent_selection(
        self, head_pos, parent_humanoid: str, sub_parent_humanoid: Optional[str]
    ) -> str:
        """Fallback parent selection using distance comparison."""
        if sub_parent_humanoid:
            parent_humanoid_bone = self.base_armature.pose.bones.get(parent_humanoid)
            sub_parent_humanoid_bone = self.base_armature.pose.bones.get(sub_parent_humanoid)

            if parent_humanoid_bone and sub_parent_humanoid_bone:
                parent_distance = (head_pos - (self.base_armature.matrix_world @ parent_humanoid_bone.head)).length
                sub_parent_distance = (
                    head_pos - (self.base_armature.matrix_world @ sub_parent_humanoid_bone.head)
                ).length

                if sub_parent_distance < parent_distance:
                    print(
                        f"chosen_parent: {sub_parent_humanoid} (sub_parent, distance: {sub_parent_distance:.4f})"
                    )
                    return sub_parent_humanoid
                else:
                    print(f"chosen_parent: {parent_humanoid} (fallback, distance: {parent_distance:.4f})")
                    return parent_humanoid
            else:
                print(f"chosen_parent: {parent_humanoid} (fallback)")
                return parent_humanoid
        else:
            print(f"chosen_parent: {parent_humanoid} (fallback)")
            return parent_humanoid

    # =========================================================================
    # Step 7: Replace bones in armature
    # =========================================================================
    def collect_children_to_update(self):
        """Collect children bones that need parent updates."""
        bpy.context.view_layer.objects.active = self.clothing_armature
        bpy.ops.object.mode_set(mode='EDIT')
        clothing_edit_bones = self.clothing_armature.data.edit_bones

        for bone in clothing_edit_bones:
            if bone.parent and bone.parent.name in self.bones_to_replace and bone.name not in self.bones_to_replace:
                self.children_to_update.append(bone.name)

    def store_base_bone_parents(self):
        """Store parent relationships from base armature."""
        bpy.context.view_layer.objects.active = self.base_armature
        bpy.ops.object.mode_set(mode='EDIT')

        for bone in self.base_armature.data.edit_bones:
            if bone.name in self.base_bones:
                self.base_bone_parents[bone.name] = (
                    bone.parent.name if bone.parent and bone.parent.name in self.base_bones else None
                )

        print(self.base_bone_parents)

    def process_bones_to_preserve_or_delete(self):
        """Process bones: preserve humanoid bones or delete others."""
        bpy.context.view_layer.objects.active = self.clothing_armature
        bpy.ops.object.mode_set(mode='EDIT')
        clothing_edit_bones = self.clothing_armature.data.edit_bones

        for bone_name in self.bones_to_replace:
            if bone_name in clothing_edit_bones:
                if bone_name in self.humanoid_bones_to_preserve:
                    # Preserve and rename Humanoid bones
                    orig_bone = clothing_edit_bones[bone_name]
                    new_name = f"origORS_{bone_name}"
                    bone_data = {
                        'head': orig_bone.head.copy(),
                        'tail': orig_bone.tail.copy(),
                        'roll': orig_bone.roll,
                        'matrix': orig_bone.matrix.copy(),
                        'new_name': new_name,
                        'humanoid_name': self.clothing_bones_to_humanoid[bone_name],
                    }
                    self.original_bone_data[bone_name] = bone_data
                    orig_bone.name = new_name
                else:
                    # Delete non-Humanoid bones
                    clothing_edit_bones.remove(clothing_edit_bones[bone_name])

    def create_new_bones(self):
        """Create new bones from base armature."""
        clothing_edit_bones = self.clothing_armature.data.edit_bones

        for bone_name in self.base_bones:
            source_bone = self.base_armature.data.edit_bones.get(bone_name)
            if source_bone:
                new_bone = clothing_edit_bones.new(name=bone_name)
                copy_bone_transform(source_bone, new_bone)
                self.new_bones[bone_name] = new_bone

    def set_new_bone_parents(self):
        """Set parent relationships for newly created bones."""
        for bone_name, new_bone in self.new_bones.items():
            parent_name = self.base_bone_parents.get(bone_name)
            if parent_name and parent_name in self.new_bones:
                new_bone.parent = self.new_bones[parent_name]

    def parent_original_bones_to_new(self):
        """Make original humanoid bones children of new bones based on hierarchy."""
        clothing_edit_bones = self.clothing_armature.data.edit_bones

        for orig_bone_name, data in self.original_bone_data.items():
            orig_bone = clothing_edit_bones[data['new_name']]
            humanoid_name = data['humanoid_name']

            # Find parent using boneHierarchy
            parent_humanoid_name = find_humanoid_parent_in_hierarchy(
                orig_bone_name, self.clothing_avatar_data, self.base_avatar_data
            )

            if parent_humanoid_name:
                matched_new_bone = self._find_new_bone_by_humanoid_name(parent_humanoid_name)
                if matched_new_bone:
                    orig_bone.parent = matched_new_bone
                else:
                    print(f"Warning: No matching new bone found for parent humanoid bone {parent_humanoid_name}")
            else:
                # Fallback to original matching logic
                matched_new_bone = self._find_new_bone_by_humanoid_name(humanoid_name)
                if matched_new_bone:
                    orig_bone.parent = matched_new_bone
                else:
                    print(f"Warning: No matching new bone found for humanoid bone {humanoid_name}")

    def _find_new_bone_by_humanoid_name(self, humanoid_name: str) -> Optional[bpy.types.EditBone]:
        """Find a new bone by its humanoid name."""
        for new_bone_name, new_bone in self.new_bones.items():
            if new_bone_name in self.base_bone_to_humanoid:
                if self.base_bone_to_humanoid[new_bone_name] == humanoid_name:
                    return new_bone
        return None

    def apply_sub_humanoid_bone_substitution(self):
        """Replace parent with subHumanoidBone if applicable."""
        if "subHumanoidBones" not in self.base_avatar_data:
            return

        sub_humanoid_bones = {}
        for sub_humanoid_bone in self.base_avatar_data["subHumanoidBones"]:
            sub_humanoid_bones[sub_humanoid_bone["humanoidBoneName"]] = sub_humanoid_bone["boneName"]

        for bone_name, parent_name in self.parent_bones.items():
            if parent_name in self.base_bone_to_humanoid:
                if self.base_bone_to_humanoid[parent_name] in sub_humanoid_bones.keys():
                    self.parent_bones[bone_name] = sub_humanoid_bones[self.base_bone_to_humanoid[parent_name]]

    def update_children_parents(self):
        """Update parent relationships for children bones."""
        clothing_edit_bones = self.clothing_armature.data.edit_bones

        for child_name in self.children_to_update:
            child_bone = clothing_edit_bones.get(child_name)
            print(f"child_name: {child_name}, child_bone: {child_bone.name if child_bone else None}")
            if child_bone:
                new_parent_name = self.parent_bones.get(child_name)
                print(f"child_name: {child_name}, new_parent_name: {new_parent_name}")
                if new_parent_name and new_parent_name in clothing_edit_bones:
                    child_bone.parent = clothing_edit_bones[new_parent_name]
                    print(f"child_name: {child_name}, new_parent_name: {new_parent_name}")

    def finish_edit_mode(self):
        """Exit edit mode."""
        bpy.ops.object.mode_set(mode='OBJECT')

    # =========================================================================
    # Step 8: Apply final pose
    # =========================================================================
    def apply_final_pose(self):
        """Apply final pose transformations if base_pose_filepath is provided."""
        if self.base_pose_filepath:
            print(f"Applying base pose from {self.base_pose_filepath}")
            add_pose_from_json(
                self.clothing_armature, self.base_pose_filepath, self.base_avatar_data, invert=False
            )
            for obj in self.clothing_obj_list:
                inverse_bone_deform_all_vertices(self.clothing_armature, obj)
            add_pose_from_json(
                self.clothing_armature, self.base_pose_filepath, self.base_avatar_data, invert=True
            )
            apply_pose_as_rest(self.clothing_armature)

    # =========================================================================
    # Step 9: Restore original state
    # =========================================================================
    def restore_original_state(self):
        """Restore original active object and mode."""
        bpy.context.view_layer.objects.active = self.original_active
        if self.original_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=self.original_mode)


def replace_humanoid_bones(
    base_armature: bpy.types.Object,
    clothing_armature: bpy.types.Object,
    base_avatar_data: dict,
    clothing_avatar_data: dict,
    preserve_humanoid_bones: bool,
    base_pose_filepath: Optional[str],
    clothing_meshes: list,
    process_upper_chest: bool,
) -> None:
    """Replace humanoid bones in clothing armature with bones from base armature."""

    ctx = _ReplaceHumanoidBonesContext(
        base_armature,
        clothing_armature,
        base_avatar_data,
        clothing_avatar_data,
        preserve_humanoid_bones,
        base_pose_filepath,
        clothing_meshes,
        process_upper_chest,
    )

    # Step 1: Create bone mappings
    ctx.create_bone_mappings()
    ctx.build_auxiliary_mappings()
    ctx.build_bones_to_replace()
    ctx.get_base_bones()
    ctx.build_humanoid_bones_to_preserve()

    # Step 2: Build BVH tree for spatial queries
    ctx.build_bvh_tree()

    try:
        # Step 3: Backup and remove armature modifiers
        ctx.backup_armature_modifiers()

        # Step 4: Apply initial pose
        ctx.apply_initial_pose()

        # Step 5: Collect clothing bone data
        ctx.collect_clothing_bone_data()

        # Step 6: Find parent bones using weight-based scoring
        ctx.find_parent_bones()

        # Step 7: Replace bones in armature
        ctx.collect_children_to_update()
        ctx.store_base_bone_parents()
        ctx.process_bones_to_preserve_or_delete()
        ctx.create_new_bones()
        ctx.set_new_bone_parents()
        ctx.parent_original_bones_to_new()
        ctx.apply_sub_humanoid_bone_substitution()
        ctx.update_children_parents()
        ctx.finish_edit_mode()

        # Step 8: Apply final pose
        ctx.apply_final_pose()

        # Step 9: Restore armature modifiers
        ctx.restore_armature_modifiers()

    finally:
        # Cleanup: Free BVH resources
        ctx.free_bvh_resources()

        # Restore original state
        ctx.restore_original_state()
