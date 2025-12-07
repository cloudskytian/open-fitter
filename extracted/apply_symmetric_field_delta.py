import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from blender_utils.batch_process_vertices_multi_step import (
    batch_process_vertices_multi_step,
)
from blender_utils.create_blendshape_mask import create_blendshape_mask
from blender_utils.get_armature_from_modifier import get_armature_from_modifier
from common_utils.get_source_label import get_source_label
from execute_transitions_with_cache import execute_transitions_with_cache
from find_intersecting_faces_bvh import find_intersecting_faces_bvh
from io_utils.restore_shape_key_state import restore_shape_key_state
from io_utils.save_shape_key_state import save_shape_key_state
from math_utils.calculate_inverse_pose_matrix import calculate_inverse_pose_matrix
from mathutils import Matrix, Vector
from misc_utils.get_deformation_field_multi_step import get_deformation_field_multi_step
from misc_utils.TransitionCache import TransitionCache
from process_field_deformation import process_field_deformation



class SymmetricFieldDeformer:
    """
    保存された対称Deformation Field差分データを読み込みメッシュに適用するクラス（最適化版、多段階対応）。
    """
    def __init__(self, target_obj, field_data_path, blend_shape_labels=None, clothing_avatar_data=None, base_avatar_data=None, subdivision=True, shape_key_name="SymmetricDeformed", skip_blend_shape_generation=False, config_data=None, ignore_blendshape=None):
        self.target_obj = target_obj
        self.field_data_path = field_data_path
        self.blend_shape_labels = blend_shape_labels
        self.clothing_avatar_data = clothing_avatar_data
        self.base_avatar_data = base_avatar_data
        self.subdivision = subdivision
        self.shape_key_name = shape_key_name
        self.skip_blend_shape_generation = skip_blend_shape_generation
        self.config_data = config_data
        self.ignore_blendshape = ignore_blendshape

        # Shared state
        self.transition_cache = TransitionCache()
        self.deferred_transitions = []
        self.config_blend_shape_labels = set()
        self.config_generated_shape_keys = {}
        self.additional_shape_keys = set()
        self.non_relative_shape_keys = set()
        self.label_to_target_shape_key_name = {'Basis': shape_key_name}
        self.shape_key = None
        self.non_transitioned_shape_vertices = None
        self.created_shape_key_mask_weights = {}
        self.shape_keys_to_remove = []


    def process_basis_loop(self):
        MAX_ITERATIONS = 0
        iteration = 0
        basis_field_path = os.path.join(os.path.dirname(self.field_data_path), self.field_data_path)
        
        while iteration <= MAX_ITERATIONS:
            original_shape_key_state = save_shape_key_state(self.target_obj)
            
            print(f"selected field_data_path: {basis_field_path}")
            
            if self.shape_key:
                self.target_obj.shape_key_remove(self.shape_key)
            self.shape_key = process_field_deformation(self.target_obj, basis_field_path, self.blend_shape_labels, self.clothing_avatar_data, self.shape_key_name, self.ignore_blendshape)
            
            restore_shape_key_state(self.target_obj, original_shape_key_state)
            
            if self.config_data:
                self.deferred_transitions.append({
                    'target_obj': self.target_obj,
                    'config_data': self.config_data,
                    'target_label': 'Basis',
                    'target_shape_key_name': self.shape_key_name,
                    'base_avatar_data': self.base_avatar_data,
                    'clothing_avatar_data': self.clothing_avatar_data,
                    'save_original_shape_key': False
                })
            
            intersections = find_intersecting_faces_bvh(self.target_obj)
            print(f"Iteration {iteration + 1}: Intersecting faces: {len(intersections)}")
            
            if not self.subdivision:
                print("Subdivision skipped")
                break

            if not intersections:
                print("No intersections detected")
                break

            if iteration == MAX_ITERATIONS:
                print("Maximum iterations reached")
                break
            
            iteration += 1

    def process_config_blendshapes(self):
        if not (self.config_data and "blendShapeFields" in self.config_data):
            return

        print("Processing config blendShapeFields...")
        
        for blend_field in self.config_data["blendShapeFields"]:
            label = blend_field["label"]
            source_label = blend_field["sourceLabel"]
            field_path = os.path.join(os.path.dirname(self.field_data_path), blend_field["path"])

            print(f"selected field_path: {field_path}")
            source_blend_shape_settings = blend_field.get("sourceBlendShapeSettings", [])

            if (self.blend_shape_labels is None or source_label not in self.blend_shape_labels) and source_label not in self.target_obj.data.shape_keys.key_blocks:
                print(f"Skipping {label} - source label {source_label} not in shape keys")
                continue
            
            mask_bones = blend_field.get("maskBones", [])
            mask_weights = None
            if mask_bones:
                mask_weights = create_blendshape_mask(self.target_obj, mask_bones, self.clothing_avatar_data, field_name=label, store_debug_mask=True)
            
            if mask_weights is not None and np.all(mask_weights == 0):
                print(f"Skipping {label} - all mask weights are zero")
                continue
            
            original_shape_key_state = save_shape_key_state(self.target_obj)
            
            if self.target_obj.data.shape_keys:
                for key_block in self.target_obj.data.shape_keys.key_blocks:
                    key_block.value = 0.0
            
            if self.clothing_avatar_data["name"] == "Template":
                if self.target_obj.data.shape_keys:
                    if source_label in self.target_obj.data.shape_keys.key_blocks:
                        source_shape_key = self.target_obj.data.shape_keys.key_blocks.get(source_label)
                        source_shape_key.value = 1.0
                        print(f"source_label: {source_label} is found in shape keys")
                    else:
                        temp_shape_key_name = f"{source_label}_temp"
                        if temp_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                            self.target_obj.data.shape_keys.key_blocks[temp_shape_key_name].value = 1.0
                            print(f"temp_shape_key_name: {temp_shape_key_name} is found in shape keys")
            else:
                for source_blend_shape_setting in source_blend_shape_settings:
                    source_blend_shape_name = source_blend_shape_setting.get("name", "")
                    source_blend_shape_value = source_blend_shape_setting.get("value", 0.0)
                    if source_blend_shape_name in self.target_obj.data.shape_keys.key_blocks:
                        source_blend_shape_key = self.target_obj.data.shape_keys.key_blocks.get(source_blend_shape_name)
                        source_blend_shape_key.value = source_blend_shape_value
                        print(f"source_blend_shape_name: {source_blend_shape_name} is found in shape keys")
                    else:
                        temp_blend_shape_key_name = f"{source_blend_shape_name}_temp"
                        if temp_blend_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                            self.target_obj.data.shape_keys.key_blocks[temp_blend_shape_key_name].value = source_blend_shape_value
                            print(f"temp_blend_shape_key_name: {temp_blend_shape_key_name} is found in shape keys")
            
            blend_shape_key_name = label
            if self.target_obj.data.shape_keys and label in self.target_obj.data.shape_keys.key_blocks:
                blend_shape_key_name = f"{label}_generated"
            
            if os.path.exists(field_path):
                print(f"Processing config blend shape field: {label} -> {blend_shape_key_name}")
                generated_shape_key = process_field_deformation(self.target_obj, field_path, self.blend_shape_labels, self.clothing_avatar_data, blend_shape_key_name, self.ignore_blendshape)
                
                if self.config_data and generated_shape_key:
                    self.deferred_transitions.append({
                        'target_obj': self.target_obj,
                        'config_data': self.config_data,
                        'target_label': label,
                        'target_shape_key_name': generated_shape_key.name,
                        'base_avatar_data': self.base_avatar_data,
                        'clothing_avatar_data': self.clothing_avatar_data,
                        'save_original_shape_key': False
                    })
                
                if generated_shape_key:
                    generated_shape_key.value = 0.0
                    self.config_generated_shape_keys[generated_shape_key.name] = mask_weights
                    self.non_relative_shape_keys.add(generated_shape_key.name)
                
                self.config_blend_shape_labels.add(label)
                self.label_to_target_shape_key_name[label] = generated_shape_key.name
            else:
                print(f"Warning: Config blend shape field file not found: {field_path}")
            
            restore_shape_key_state(self.target_obj, original_shape_key_state)

    def process_skipped_transitions(self):
        if not (self.config_data and self.config_data.get('blend_shape_transition_sets', [])):
            return

        transition_sets = self.config_data.get('blend_shape_transition_sets', [])
        print("Processing skipped config blendShapeFields...")
        
        for transition_set in transition_sets:
            label = transition_set["label"]
            if label in self.config_blend_shape_labels or label == 'Basis':
                continue

            source_label = get_source_label(label, self.config_data)
            if source_label not in self.label_to_target_shape_key_name:
                print(f"Skipping {label} - source label {source_label} not in label_to_target_shape_key_name")
                continue

            print(f"Processing skipped config blendShapeField: {label}")
            
            mask_bones = transition_set.get("mask_bones", [])
            print(f"mask_bones: {mask_bones}")
            mask_weights = None
            if mask_bones:
                mask_weights = create_blendshape_mask(self.target_obj, mask_bones, self.clothing_avatar_data, field_name=label, store_debug_mask=True)
            
            if mask_weights is not None and np.all(mask_weights == 0):
                print(f"Skipping {label} - all mask weights are zero")
                continue
            
            target_shape_key_name = self.label_to_target_shape_key_name[source_label]
            target_shape_key = self.target_obj.data.shape_keys.key_blocks.get(target_shape_key_name)

            if not target_shape_key:
                print(f"Skipping {label} - target shape key {target_shape_key_name} not found")
                continue

            blend_shape_key_name = label
            if self.target_obj.data.shape_keys and label in self.target_obj.data.shape_keys.key_blocks:
                blend_shape_key_name = f"{label}_generated"
            
            skipped_blend_shape_key = self.target_obj.shape_key_add(name=blend_shape_key_name)
        
            for i in range(len(skipped_blend_shape_key.data)):
                skipped_blend_shape_key.data[i].co = target_shape_key.data[i].co.copy()

            print(f"skipped_blend_shape_key: {skipped_blend_shape_key.name}")
            
            if self.config_data and skipped_blend_shape_key:
                self.deferred_transitions.append({
                    'target_obj': self.target_obj,
                    'config_data': self.config_data,
                    'target_label': label,
                    'target_shape_key_name': skipped_blend_shape_key.name,
                    'base_avatar_data': self.base_avatar_data,
                    'clothing_avatar_data': self.clothing_avatar_data,
                    'save_original_shape_key': False
                })

                print(f"Added deferred transition: {label} -> {skipped_blend_shape_key.name}")

                self.config_generated_shape_keys[skipped_blend_shape_key.name] = mask_weights
                self.non_relative_shape_keys.add(skipped_blend_shape_key.name)
                self.config_blend_shape_labels.add(label)
                self.label_to_target_shape_key_name[label] = skipped_blend_shape_key.name

    def process_clothing_blendshapes(self):
        if not self.target_obj.data.shape_keys:
            return

        clothing_blendshapes = set()
        if self.clothing_avatar_data and "blendshapes" in self.clothing_avatar_data:
            for blendshape in self.clothing_avatar_data["blendshapes"]:
                clothing_blendshapes.add(blendshape["name"])
        
        current_shape_key_blocks = [key_block for key_block in self.target_obj.data.shape_keys.key_blocks]

        for key_block in current_shape_key_blocks:
            if (key_block.name == "Basis" or 
                key_block.name in clothing_blendshapes or 
                key_block == self.shape_key or 
                key_block.name.endswith("_BaseShape") or
                key_block.name in self.config_generated_shape_keys.keys() or
                key_block.name in self.config_blend_shape_labels or
                key_block.name.endswith("_original") or 
                key_block.name.endswith("_generated") or
                key_block.name.endswith("_temp")):
                continue
            
            print(f"Processing additional shape key: {key_block.name}")

            original_shape_key_state = save_shape_key_state(self.target_obj)
            
            for sk in self.target_obj.data.shape_keys.key_blocks:
                sk.value = 0.0
            
            basis_field_path2 = os.path.join(os.path.dirname(self.field_data_path), self.field_data_path)
            source_label = get_source_label('Basis', self.config_data)
            if source_label is not None and source_label != 'Basis' and self.target_obj.data.shape_keys:
                source_field_path = None
                source_shape_name = None
                if self.config_data and "blendShapeFields" in self.config_data:
                    for blend_field in self.config_data["blendShapeFields"]:
                        if blend_field["label"] == source_label:
                            source_field_path = os.path.join(os.path.dirname(self.field_data_path), blend_field["path"])
                            source_shape_name = blend_field["sourceLabel"]
                            break
                if source_field_path is not None and source_shape_name is not None:
                    if source_shape_name in self.target_obj.data.shape_keys.key_blocks:
                        source_shape_key = self.target_obj.data.shape_keys.key_blocks.get(source_shape_name)
                        source_shape_key.value = 1.0
                        basis_field_path2 = source_field_path
                        print(f"source_label: {source_shape_name} is found in shape keys")
                    else:
                        temp_shape_key_name = f"{source_shape_name}_temp"
                        if temp_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                            self.target_obj.data.shape_keys.key_blocks[temp_shape_key_name].value = 1.0
                            basis_field_path2 = source_field_path
                            print(f"temp_shape_key_name: {temp_shape_key_name} is found in shape keys")

            print(f"basis_field_path2: {basis_field_path2}")
            
            key_block.value = 1.0

            temp_blend_shape_key_name = f"{key_block.name}_generated"

            temp_shape_key = process_field_deformation(self.target_obj, basis_field_path2, self.blend_shape_labels, self.clothing_avatar_data, temp_blend_shape_key_name, self.ignore_blendshape)

            self.additional_shape_keys.add(temp_shape_key.name)
            self.non_relative_shape_keys.add(temp_shape_key.name)

            key_block.value = 0.0

            restore_shape_key_state(self.target_obj, original_shape_key_state)

    def execute_deferred_transitions(self):
        if not self.deferred_transitions:
            return

        transition_operations, created_shape_key_mask_weights, used_shape_key_names = execute_transitions_with_cache(self.deferred_transitions, self.transition_cache, self.target_obj)
        
        for transition_operation in transition_operations:
            if transition_operation['transition_data']['target_label'] == 'Basis':
                self.non_transitioned_shape_vertices = [Vector(v) for v in transition_operation['initial_vertices']]
                break
        
        if used_shape_key_names:
            for config_shape_key_name in self.config_generated_shape_keys:
                if config_shape_key_name not in used_shape_key_names and config_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                    self.shape_keys_to_remove.append(config_shape_key_name)
        
        for created_shape_key_name, mask_weights in created_shape_key_mask_weights.items():
            if created_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                self.config_generated_shape_keys[created_shape_key_name] = mask_weights
                self.non_relative_shape_keys.add(created_shape_key_name)
                self.config_blend_shape_labels.add(created_shape_key_name)
                self.label_to_target_shape_key_name[created_shape_key_name] = created_shape_key_name
                print(f"Added created shape key: {created_shape_key_name}")

    def apply_masks_and_cleanup(self):
        self.shape_key.value = 1.0
        
        basis_name = 'Basis'
        basis_index = self.target_obj.data.shape_keys.key_blocks.find(basis_name)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.target_obj
        self.target_obj.select_set(True)

        if self.non_transitioned_shape_vertices:
            for additionalshape_key_name in self.additional_shape_keys:
                if additionalshape_key_name in self.target_obj.data.shape_keys.key_blocks:
                    additional_shape_key = self.target_obj.data.shape_keys.key_blocks.get(additionalshape_key_name)
                    for i, vert in enumerate(additional_shape_key.data):
                        shape_diff = self.shape_key.data[i].co - self.non_transitioned_shape_vertices[i]
                        additional_shape_key.data[i].co += shape_diff
                else:
                    print(f"Warning: {additionalshape_key_name} is not found in shape keys")
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        print(f"Shape keys in {self.target_obj.name}:")
        for key_block in self.target_obj.data.shape_keys.key_blocks:
            print(f"- {key_block.name} (value: {key_block.value})")
        
        original_shape_key_name = f"{self.shape_key_name}_original"
        for sk in self.target_obj.data.shape_keys.key_blocks:
            if sk.name in self.non_relative_shape_keys and sk.name != basis_name:
                if self.shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                    self.target_obj.active_shape_key_index = self.target_obj.data.shape_keys.key_blocks.find(sk.name)
                    bpy.ops.mesh.blend_from_shape(shape=self.shape_key_name, blend=-1, add=True)
                else:
                    print(f"Warning: {self.shape_key_name} or {self.shape_key_name}_original is not found in shape keys")

        bpy.context.object.active_shape_key_index = basis_index
        bpy.ops.mesh.blend_from_shape(shape=self.shape_key_name, blend=1, add=True)

        bpy.ops.object.mode_set(mode='OBJECT')

        if original_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
            original_shape_key = self.target_obj.data.shape_keys.key_blocks.get(original_shape_key_name)
            self.target_obj.shape_key_remove(original_shape_key)
            print(f"Removed shape key: {original_shape_key_name} from {self.target_obj.name}")
        
        if self.shape_key:
           self.target_obj.shape_key_remove(self.shape_key)
        
        for unused_shape_key_name in self.shape_keys_to_remove:
            if unused_shape_key_name in self.target_obj.data.shape_keys.key_blocks:
                unused_shape_key = self.target_obj.data.shape_keys.key_blocks.get(unused_shape_key_name)
                if unused_shape_key:
                    self.target_obj.shape_key_remove(unused_shape_key)
                    print(f"Removed shape key: {unused_shape_key_name} from {self.target_obj.name}")
                else:
                    print(f"Warning: {unused_shape_key_name} is not found in shape keys")
            else:
                print(f"Warning: {unused_shape_key_name} is not found in shape keys")

        if self.config_generated_shape_keys:
            print(f"Applying mask weights to generated shape keys: {list(self.config_generated_shape_keys.keys())}")
            
            basis_shape_key = self.target_obj.data.shape_keys.key_blocks.get(basis_name)
            if basis_shape_key:
                basis_positions = np.array([v.co for v in basis_shape_key.data])
                
                for shape_key_name_to_mask, mask_weights in self.config_generated_shape_keys.items():
                    if shape_key_name_to_mask == basis_name:
                        continue
                        
                    shape_key_to_mask = self.target_obj.data.shape_keys.key_blocks.get(shape_key_name_to_mask)
                    if shape_key_to_mask:
                        shape_positions = np.array([v.co for v in shape_key_to_mask.data])
                        displacement = shape_positions - basis_positions
                        
                        if mask_weights is not None:
                            masked_displacement = displacement * mask_weights[:, np.newaxis]
                        else:
                            masked_displacement = displacement
                        
                        new_positions = basis_positions + masked_displacement
                        
                        for i, vertex in enumerate(shape_key_to_mask.data):
                            vertex.co = new_positions[i]
                        
                        print(f"Applied mask weights to shape key: {shape_key_name_to_mask}")

    def process_base_avatar_blendshapes(self):
        if not (self.base_avatar_data and "blendShapeFields" in self.base_avatar_data and not self.skip_blend_shape_generation):
            return

        armature_obj = get_armature_from_modifier(self.target_obj)
        if not armature_obj:
            raise ValueError("Armatureモディファイアが見つかりません")
        
        original_shape_key_state = save_shape_key_state(self.target_obj)
        
        if self.target_obj.data.shape_keys:
            for key_block in self.target_obj.data.shape_keys.key_blocks:
                key_block.value = 0.0

        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        eval_obj = self.target_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.data
        vertices = np.array([v.co for v in self.target_obj.data.vertices])
        deformed_vertices = np.array([v.co for v in eval_mesh.vertices])

        for blend_field in self.base_avatar_data["blendShapeFields"]:
            label = blend_field["label"]
            
            if label in self.config_blend_shape_labels:
                print(f"Skipping base avatar blend shape field '{label}' (already processed from config)")
                continue
                
            field_path = os.path.join(os.path.dirname(self.field_data_path), blend_field["filePath"])
            
            if os.path.exists(field_path):
                print(f"Applying blend shape field for {label}")
                field_info_blend = get_deformation_field_multi_step(field_path)
                blend_points = field_info_blend['all_field_points']
                blend_deltas = field_info_blend['all_delta_positions']
                blend_field_weights = field_info_blend['field_weights']
                blend_matrix = field_info_blend['world_matrix']
                blend_matrix_inv = field_info_blend['world_matrix_inv']
                blend_k_neighbors = field_info_blend['kdtree_query_k']
                
                mask_weights = None
                if "maskBones" in blend_field:
                    mask_weights = create_blendshape_mask(self.target_obj, blend_field["maskBones"], self.clothing_avatar_data, field_name=label, store_debug_mask=True)
                
                deformed_positions = batch_process_vertices_multi_step(
                    deformed_vertices,
                    blend_points,
                    blend_deltas,
                    blend_field_weights,
                    blend_matrix,
                    blend_matrix_inv,
                    self.target_obj.matrix_world,
                    self.target_obj.matrix_world.inverted(),
                    mask_weights,
                    batch_size=1000,
                    k=blend_k_neighbors
                )

                has_displacement = False
                for i in range(len(deformed_vertices)):
                    displacement = deformed_positions[i] - (self.target_obj.matrix_world @ Vector(deformed_vertices[i]))
                    if np.any(np.abs(displacement) > 1e-5):
                        print(f"blendShapeFields {label} world_displacement: {displacement}")
                        has_displacement = True
                        break

                if has_displacement:
                    blend_shape_key_name = label
                    if self.target_obj.data.shape_keys and label in self.target_obj.data.shape_keys.key_blocks:
                        blend_shape_key_name = f"{label}_generated"
                    
                    shape_key_b = self.target_obj.shape_key_add(name=blend_shape_key_name)
                    shape_key_b.value = 0.0

                    matrix_armature_inv_fallback = Matrix.Identity(4)
                    for i in range(len(vertices)):
                        matrix_armature_inv = calculate_inverse_pose_matrix(self.target_obj, armature_obj, i)
                        if matrix_armature_inv is None:
                            matrix_armature_inv = matrix_armature_inv_fallback
                        deformed_world_pos = matrix_armature_inv @ Vector(deformed_positions[i])
                        deformed_local_pos = self.target_obj.matrix_world.inverted() @ deformed_world_pos
                        shape_key_b.data[i].co = deformed_local_pos
                        matrix_armature_inv_fallback = matrix_armature_inv
                else:
                    print(f"Skipping creation of shape key '{label}' as it has no displacement")

            else:
                print(f"Warning: Field file not found for blend shape {label}")
        
        restore_shape_key_state(self.target_obj, original_shape_key_state)

    def finalize(self):
        for sk in self.target_obj.data.shape_keys.key_blocks:
            sk.value = 0.0


def apply_symmetric_field_delta(target_obj, field_data_path, blend_shape_labels=None, clothing_avatar_data=None, base_avatar_data=None, subdivision=True, shape_key_name="SymmetricDeformed", skip_blend_shape_generation=False, config_data=None, ignore_blendshape=None):
    """
    保存された対称Deformation Field差分データを読み込みメッシュに適用する（最適化版、多段階対応）。
    ※BlendShape用のDeformation Fieldを先に適用した場合と、メインのみ適用した場合の交差面の割合を
      比較し、所定の条件下ではBlendShapeの変位を無視する処理を行います。
    """
    ctx = SymmetricFieldDeformer(target_obj, field_data_path, blend_shape_labels, clothing_avatar_data, base_avatar_data, subdivision, shape_key_name, skip_blend_shape_generation, config_data, ignore_blendshape)
    
    ctx.process_basis_loop()
    ctx.process_config_blendshapes()
    ctx.process_skipped_transitions()
    ctx.process_clothing_blendshapes()
    ctx.execute_deferred_transitions()
    ctx.apply_masks_and_cleanup()
    ctx.process_base_avatar_blendshapes()
    ctx.finalize()
    
    return ctx.shape_key

