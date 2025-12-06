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


def apply_symmetric_field_delta(target_obj, field_data_path, blend_shape_labels=None, clothing_avatar_data=None, base_avatar_data=None, subdivision=True, shape_key_name="SymmetricDeformed", skip_blend_shape_generation=False, config_data=None, ignore_blendshape=None):
    """
    保存された対称Deformation Field差分データを読み込みメッシュに適用する（最適化版、多段階対応）。
    ※BlendShape用のDeformation Fieldを先に適用した場合と、メインのみ適用した場合の交差面の割合を
      比較し、所定の条件下ではBlendShapeの変位を無視する処理を行います。
    """
    
    # Transitionキャッシュを初期化
    transition_cache = TransitionCache()
    deferred_transitions = []  # 遅延実行するTransitionのリスト
    
    MAX_ITERATIONS = 0  # 最大繰り返し回数

    # メインの処理ループ（従来の単一ステップ処理）
    iteration = 0
    shape_key = None
    basis_field_path = os.path.join(os.path.dirname(field_data_path), field_data_path)
    while iteration <= MAX_ITERATIONS:
        
        original_shape_key_state = save_shape_key_state(target_obj)
        
        print(f"selected field_data_path: {basis_field_path}")
        
        # シェイプキーを作成して変形を適用
        if shape_key:
            target_obj.shape_key_remove(shape_key)
        shape_key = process_field_deformation(target_obj, basis_field_path, blend_shape_labels, clothing_avatar_data, shape_key_name, ignore_blendshape)
        
        restore_shape_key_state(target_obj, original_shape_key_state)
        
        # Basis遷移を遅延実行リストに追加
        if config_data:
            deferred_transitions.append({
                'target_obj': target_obj,
                'config_data': config_data,
                'target_label': 'Basis',
                'target_shape_key_name': shape_key_name,
                'base_avatar_data': base_avatar_data,
                'clothing_avatar_data': clothing_avatar_data,
                'save_original_shape_key': False
            })
        
        # 新たな交差を検出
        intersections = find_intersecting_faces_bvh(target_obj)
        print(f"Iteration {iteration + 1}: Intersecting faces: {len(intersections)}")
        
        if not subdivision:
            print("Subdivision skipped")
            break

        if not intersections:
            print("No intersections detected")
            break

        if iteration == MAX_ITERATIONS:
            print("Maximum iterations reached")
            break
        # 新たな交差が検出された場合、それらの面を細分化
        # subdivide_faces(target_obj, intersections)

        iteration += 1
    
    # configファイルのblendShapeFieldsを処理するためのラベルセットを作成
    config_blend_shape_labels = set()
    config_generated_shape_keys = {}  # 後続処理の対象外にするシェイプキー名を保存
    additional_shape_keys = set() # 追加で処理するシェイプキー名を保存
    non_relative_shape_keys = set() # 相対的な変位を持たないシェイプキー名を保存

    skipped_shape_keys = set()
    label_to_target_shape_key_name = {'Basis': shape_key_name}

    # 1. configファイルのblendShapeFieldsを先に処理
    if config_data and "blendShapeFields" in config_data:
        print("Processing config blendShapeFields...")
        
        for blend_field in config_data["blendShapeFields"]:
            label = blend_field["label"]
            source_label = blend_field["sourceLabel"]
            field_path = os.path.join(os.path.dirname(field_data_path), blend_field["path"])

            print(f"selected field_path: {field_path}")
            source_blend_shape_settings = blend_field.get("sourceBlendShapeSettings", [])

            if (blend_shape_labels is None or source_label not in blend_shape_labels) and source_label not in target_obj.data.shape_keys.key_blocks:
                print(f"Skipping {label} - source label {source_label} not in shape keys")
                skipped_shape_keys.add(label)
                continue
            
            # マスクウェイトを取得
            mask_bones = blend_field.get("maskBones", [])
            mask_weights = None
            if mask_bones:
                mask_weights = create_blendshape_mask(target_obj, mask_bones, clothing_avatar_data, field_name=label, store_debug_mask=True)
            
            if mask_weights is not None and np.all(mask_weights == 0):
                print(f"Skipping {label} - all mask weights are zero")
                continue
            
            # 対象メッシュオブジェクトの元のシェイプキー設定を保存
            original_shape_key_state = save_shape_key_state(target_obj)
            
            # すべてのシェイプキーの値を0にする
            if target_obj.data.shape_keys:
                for key_block in target_obj.data.shape_keys.key_blocks:
                    key_block.value = 0.0
            
            # 最初のConfig Pairでの対象シェイプキー（1が前提）もしくは前のConfig PairでTransition後のシェイプキーの値を1にする
            if clothing_avatar_data["name"] == "Template":
                if target_obj.data.shape_keys:
                    if source_label in target_obj.data.shape_keys.key_blocks:
                        source_shape_key = target_obj.data.shape_keys.key_blocks.get(source_label)
                        source_shape_key.value = 1.0
                        print(f"source_label: {source_label} is found in shape keys")
                    else:
                        temp_shape_key_name = f"{source_label}_temp"
                        if temp_shape_key_name in target_obj.data.shape_keys.key_blocks:
                            target_obj.data.shape_keys.key_blocks[temp_shape_key_name].value = 1.0
                            print(f"temp_shape_key_name: {temp_shape_key_name} is found in shape keys")
            else:
                # source_blend_shape_settingsを適用
                for source_blend_shape_setting in source_blend_shape_settings:
                    source_blend_shape_name = source_blend_shape_setting.get("name", "")
                    source_blend_shape_value = source_blend_shape_setting.get("value", 0.0)
                    if source_blend_shape_name in target_obj.data.shape_keys.key_blocks:
                        source_blend_shape_key = target_obj.data.shape_keys.key_blocks.get(source_blend_shape_name)
                        source_blend_shape_key.value = source_blend_shape_value
                        print(f"source_blend_shape_name: {source_blend_shape_name} is found in shape keys")
                    else:
                        temp_blend_shape_key_name = f"{source_blend_shape_name}_temp"
                        if temp_blend_shape_key_name in target_obj.data.shape_keys.key_blocks:
                            target_obj.data.shape_keys.key_blocks[temp_blend_shape_key_name].value = source_blend_shape_value
                            print(f"temp_blend_shape_key_name: {temp_blend_shape_key_name} is found in shape keys")
            
            # blend_shape_key_nameを設定（同名のシェイプキーがある場合は_generatedを付ける）
            blend_shape_key_name = label
            if target_obj.data.shape_keys and label in target_obj.data.shape_keys.key_blocks:
                blend_shape_key_name = f"{label}_generated"
            
            # process_field_deformationを実行
            if os.path.exists(field_path):
                print(f"Processing config blend shape field: {label} -> {blend_shape_key_name}")
                generated_shape_key = process_field_deformation(target_obj, field_path, blend_shape_labels, clothing_avatar_data, blend_shape_key_name, ignore_blendshape)
                
                # 該当するラベルの遷移を遅延実行リストに追加
                if config_data and generated_shape_key:
                    deferred_transitions.append({
                        'target_obj': target_obj,
                        'config_data': config_data,
                        'target_label': label,
                        'target_shape_key_name': generated_shape_key.name,
                        'base_avatar_data': base_avatar_data,
                        'clothing_avatar_data': clothing_avatar_data,
                        'save_original_shape_key': False
                    })
                
                # 生成されたシェイプキーの値を0にする
                if generated_shape_key:
                    generated_shape_key.value = 0.0
                    config_generated_shape_keys[generated_shape_key.name] = mask_weights
                    non_relative_shape_keys.add(generated_shape_key.name)
                
                config_blend_shape_labels.add(label)

                label_to_target_shape_key_name[label] = generated_shape_key.name
            else:
                print(f"Warning: Config blend shape field file not found: {field_path}")
            
            # 元のシェイプキー設定を復元
            restore_shape_key_state(target_obj, original_shape_key_state)
    
    # transition_setsに含まれるがconfig_blend_shape_labelsに含まれないシェイプキーに対して処理
    if config_data and config_data.get('blend_shape_transition_sets', []):
        transition_sets = config_data.get('blend_shape_transition_sets', [])
        print("Processing skipped config blendShapeFields...")
        
        for transition_set in transition_sets:
            label = transition_set["label"]
            if label in config_blend_shape_labels or label == 'Basis':
                continue

            source_label = get_source_label(label, config_data)
            if source_label not in label_to_target_shape_key_name:
                print(f"Skipping {label} - source label {source_label} not in label_to_target_shape_key_name")
                continue

            print(f"Processing skipped config blendShapeField: {label}")
            
            # マスクウェイトを取得
            mask_bones = transition_set.get("mask_bones", [])
            print(f"mask_bones: {mask_bones}")
            mask_weights = None
            if mask_bones:
                mask_weights = create_blendshape_mask(target_obj, mask_bones, clothing_avatar_data, field_name=label, store_debug_mask=True)
            
            if mask_weights is not None and np.all(mask_weights == 0):
                print(f"Skipping {label} - all mask weights are zero")
                continue
            
            target_shape_key_name = label_to_target_shape_key_name[source_label]
            target_shape_key = target_obj.data.shape_keys.key_blocks.get(target_shape_key_name)

            if not target_shape_key:
                print(f"Skipping {label} - target shape key {target_shape_key_name} not found")
                continue

            # target_shape_key_nameで指定されるシェイプキーのコピーを作成
            blend_shape_key_name = label
            if target_obj.data.shape_keys and label in target_obj.data.shape_keys.key_blocks:
                blend_shape_key_name = f"{label}_generated"
            
            skipped_blend_shape_key = target_obj.shape_key_add(name=blend_shape_key_name)
        
            for i in range(len(skipped_blend_shape_key.data)):
                skipped_blend_shape_key.data[i].co = target_shape_key.data[i].co.copy()

            print(f"skipped_blend_shape_key: {skipped_blend_shape_key.name}")
            
            if config_data and skipped_blend_shape_key:
                deferred_transitions.append({
                    'target_obj': target_obj,
                    'config_data': config_data,
                    'target_label': label,
                    'target_shape_key_name': skipped_blend_shape_key.name,
                    'base_avatar_data': base_avatar_data,
                    'clothing_avatar_data': clothing_avatar_data,
                    'save_original_shape_key': False
                })

                print(f"Added deferred transition: {label} -> {skipped_blend_shape_key.name}")

                config_generated_shape_keys[skipped_blend_shape_key.name] = mask_weights
                non_relative_shape_keys.add(skipped_blend_shape_key.name)
                config_blend_shape_labels.add(label)
                label_to_target_shape_key_name[label] = skipped_blend_shape_key.name
            
    
    # 2. clothing_avatar_dataのblendshapesに含まれないシェイプキーに対して処理
    if target_obj.data.shape_keys:
        # clothing_avatar_dataからblendshapeのリストを作成
        clothing_blendshapes = set()
        if clothing_avatar_data and "blendshapes" in clothing_avatar_data:
            for blendshape in clothing_avatar_data["blendshapes"]:
                clothing_blendshapes.add(blendshape["name"])
        
        # 各シェイプキーについて処理

        #無限ループ回避のため事前にシェイプキーのリストを取得
        current_shape_key_blocks = [key_block for key_block in target_obj.data.shape_keys.key_blocks]

        for key_block in current_shape_key_blocks:
            if (key_block.name == "Basis" or 
                key_block.name in clothing_blendshapes or 
                key_block == shape_key or 
                key_block.name.endswith("_BaseShape") or
                key_block.name in config_generated_shape_keys.keys() or
                key_block.name in config_blend_shape_labels or
                key_block.name.endswith("_original") or 
                key_block.name.endswith("_generated") or
                key_block.name.endswith("_temp")):
                continue  # Basisまたはclothing_avatar_dataのblendshapesに含まれるもの、または_BaseShapeで終わるもの、またはconfigで生成されたものはスキップ
            
            print(f"Processing additional shape key: {key_block.name}")

            original_shape_key_state = save_shape_key_state(target_obj)
            
            # すべてのシェイプキーの値を0に設定
            for sk in target_obj.data.shape_keys.key_blocks:
                sk.value = 0.0
            
            basis_field_path2 = os.path.join(os.path.dirname(field_data_path), field_data_path)
            source_label = get_source_label('Basis', config_data)
            if source_label is not None and source_label != 'Basis' and target_obj.data.shape_keys:
                source_field_path = None
                source_shape_name = None
                if config_data and "blendShapeFields" in config_data:
                    for blend_field in config_data["blendShapeFields"]:
                        if blend_field["label"] == source_label:
                            source_field_path = os.path.join(os.path.dirname(field_data_path), blend_field["path"])
                            source_shape_name = blend_field["sourceLabel"]
                            break
                if source_field_path is not None and source_shape_name is not None:
                    if source_shape_name in target_obj.data.shape_keys.key_blocks:
                        source_shape_key = target_obj.data.shape_keys.key_blocks.get(source_shape_name)
                        source_shape_key.value = 1.0
                        basis_field_path2 = source_field_path
                        print(f"source_label: {source_shape_name} is found in shape keys")
                    else:
                        temp_shape_key_name = f"{source_shape_name}_temp"
                        if temp_shape_key_name in target_obj.data.shape_keys.key_blocks:
                            target_obj.data.shape_keys.key_blocks[temp_shape_key_name].value = 1.0
                            basis_field_path2 = source_field_path
                            print(f"temp_shape_key_name: {temp_shape_key_name} is found in shape keys")

            print(f"basis_field_path2: {basis_field_path2}")
            
            # 対象のシェイプキーの値を1に設定
            key_block.value = 1.0

            temp_blend_shape_key_name = f"{key_block.name}_generated"

            temp_shape_key = process_field_deformation(target_obj, basis_field_path2, blend_shape_labels, clothing_avatar_data, temp_blend_shape_key_name, ignore_blendshape)

            additional_shape_keys.add(temp_shape_key.name)
            non_relative_shape_keys.add(temp_shape_key.name)

            # シェイプキーの値を元に戻す
            key_block.value = 0.0

            restore_shape_key_state(target_obj, original_shape_key_state)

    # 遅延されたTransitionをキャッシュシステムと共に実行
    non_transitioned_shape_vertices = None
    created_shape_key_mask_weights = {}
    shape_keys_to_remove = []
    if deferred_transitions:
        transition_operations, created_shape_key_mask_weights, used_shape_key_names = execute_transitions_with_cache(deferred_transitions, transition_cache, target_obj)
        for transition_operation in transition_operations:
            if transition_operation['transition_data']['target_label'] == 'Basis':
                non_transitioned_shape_vertices = [Vector(v) for v in transition_operation['initial_vertices']]
                break
        if used_shape_key_names:
            for config_shape_key_name in config_generated_shape_keys:
                if config_shape_key_name not in used_shape_key_names and config_shape_key_name in target_obj.data.shape_keys.key_blocks:
                    shape_keys_to_remove.append(config_shape_key_name)
    
    for created_shape_key_name, mask_weights in created_shape_key_mask_weights.items():
        if created_shape_key_name in target_obj.data.shape_keys.key_blocks:
            config_generated_shape_keys[created_shape_key_name] = mask_weights
            non_relative_shape_keys.add(created_shape_key_name)
            config_blend_shape_labels.add(created_shape_key_name)
            label_to_target_shape_key_name[created_shape_key_name] = created_shape_key_name
            print(f"Added created shape key: {created_shape_key_name}")
    
    shape_key.value = 1.0
    
    # base_avatar_dataのblendShapeFieldsを処理する前の準備
    basis_name = 'Basis'
    basis_index = target_obj.data.shape_keys.key_blocks.find(basis_name)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = target_obj
    target_obj.select_set(True)

    if non_transitioned_shape_vertices:
        for additionalshape_key_name in additional_shape_keys:
            if additionalshape_key_name in target_obj.data.shape_keys.key_blocks:
                additional_shape_key = target_obj.data.shape_keys.key_blocks.get(additionalshape_key_name)
                # shape_keyとtransition前のBasisのシェイプの差をadditional_shape_keyの各頂点に追加
                for i, vert in enumerate(additional_shape_key.data):
                    # shape_keyとBasisの差分を計算
                    shape_diff = shape_key.data[i].co - non_transitioned_shape_vertices[i]
                    # additional_shape_keyの頂点座標に差分を追加
                    additional_shape_key.data[i].co += shape_diff
                
            else:
                print(f"Warning: {additionalshape_key_name} is not found in shape keys")
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    print(f"Shape keys in {target_obj.name}:")
    for key_block in target_obj.data.shape_keys.key_blocks:
        print(f"- {key_block.name} (value: {key_block.value})")
    
    original_shape_key_name = f"{shape_key_name}_original"
    for sk in target_obj.data.shape_keys.key_blocks:
        if sk.name in non_relative_shape_keys and sk.name != basis_name:
            if shape_key_name in target_obj.data.shape_keys.key_blocks:
                target_obj.active_shape_key_index = target_obj.data.shape_keys.key_blocks.find(sk.name)
                bpy.ops.mesh.blend_from_shape(shape=shape_key_name, blend=-1, add=True)
            else:
                print(f"Warning: {shape_key_name} or {shape_key_name}_original is not found in shape keys")

    bpy.context.object.active_shape_key_index = basis_index
    bpy.ops.mesh.blend_from_shape(shape=shape_key_name, blend=1, add=True)

    bpy.ops.object.mode_set(mode='OBJECT')

    if original_shape_key_name in target_obj.data.shape_keys.key_blocks:
        original_shape_key = target_obj.data.shape_keys.key_blocks.get(original_shape_key_name)
        target_obj.shape_key_remove(original_shape_key)
        print(f"Removed shape key: {original_shape_key_name} from {target_obj.name}")
    
    # # 不要なシェイプキーを削除
    if shape_key:
       target_obj.shape_key_remove(shape_key)
    
    for unused_shape_key_name in shape_keys_to_remove:
        if unused_shape_key_name in target_obj.data.shape_keys.key_blocks:
            unused_shape_key = target_obj.data.shape_keys.key_blocks.get(unused_shape_key_name)
            if unused_shape_key:
                target_obj.shape_key_remove(unused_shape_key)
                print(f"Removed shape key: {unused_shape_key_name} from {target_obj.name}")
            else:
                print(f"Warning: {unused_shape_key_name} is not found in shape keys")
        else:
            print(f"Warning: {unused_shape_key_name} is not found in shape keys")

    # configファイルのblendShapeFieldsで生成されたシェイプキーの変位にmask_weightsを適用
    if config_generated_shape_keys:
        print(f"Applying mask weights to generated shape keys: {list(config_generated_shape_keys.keys())}")
        
        # ベースシェイプの頂点位置を取得
        basis_shape_key = target_obj.data.shape_keys.key_blocks.get(basis_name)
        if basis_shape_key:
            basis_positions = np.array([v.co for v in basis_shape_key.data])
            
            # 各生成されたシェイプキーに対してマスクを適用
            for shape_key_name_to_mask, mask_weights in config_generated_shape_keys.items():
                if shape_key_name_to_mask == basis_name:
                    continue
                    
                shape_key_to_mask = target_obj.data.shape_keys.key_blocks.get(shape_key_name_to_mask)
                if shape_key_to_mask:
                    # 現在のシェイプキーの頂点位置を取得
                    shape_positions = np.array([v.co for v in shape_key_to_mask.data])
                    
                    # 変位を計算
                    displacement = shape_positions - basis_positions
                    
                    # マスクを適用（変位にmask_weightsを掛ける）
                    if mask_weights is not None:
                        masked_displacement = displacement * mask_weights[:, np.newaxis]
                    else:
                        masked_displacement = displacement
                    
                    # マスク適用後の位置を計算
                    new_positions = basis_positions + masked_displacement
                    
                    # シェイプキーの頂点位置を更新
                    for i, vertex in enumerate(shape_key_to_mask.data):
                        vertex.co = new_positions[i]
                    
                    print(f"Applied mask weights to shape key: {shape_key_name_to_mask}")
    
    # 4. base_avatar_dataのblendShapeFieldsを処理（configのlabelと一致するものはスキップ）
    if base_avatar_data and "blendShapeFields" in base_avatar_data and not skip_blend_shape_generation:
        # アーマチュアの取得
        armature_obj = get_armature_from_modifier(target_obj)
        if not armature_obj:
            raise ValueError("Armatureモディファイアが見つかりません")
        
        # 対象メッシュオブジェクトの元のシェイプキー設定を保存
        original_shape_key_state = save_shape_key_state(target_obj)
        
        # すべてのシェイプキーの値を0にする
        if target_obj.data.shape_keys:
            for key_block in target_obj.data.shape_keys.key_blocks:
                key_block.value = 0.0

        # 評価されたメッシュの頂点位置を取得（シェイプキーA適用後）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        eval_obj = target_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.data
        vertices = np.array([v.co for v in target_obj.data.vertices])  # オリジナルの頂点配列
        deformed_vertices = np.array([v.co for v in eval_mesh.vertices])

        # 各blendShapeFieldを処理
        for blend_field in base_avatar_data["blendShapeFields"]:
            label = blend_field["label"]
            
            # configファイルのblendShapeFieldsのlabelと一致する場合はスキップ
            if label in config_blend_shape_labels:
                print(f"Skipping base avatar blend shape field '{label}' (already processed from config)")
                continue
                
            field_path = os.path.join(os.path.dirname(field_data_path), blend_field["filePath"])
            
            if os.path.exists(field_path):
                print(f"Applying blend shape field for {label}")
                # フィールドデータの読み込み
                field_info_blend = get_deformation_field_multi_step(field_path)
                blend_points = field_info_blend['all_field_points']
                blend_deltas = field_info_blend['all_delta_positions']
                blend_field_weights = field_info_blend['field_weights']
                blend_matrix = field_info_blend['world_matrix']
                blend_matrix_inv = field_info_blend['world_matrix_inv']
                blend_k_neighbors = field_info_blend['kdtree_query_k']
                
                # マスクウェイトを取得
                mask_weights = None
                if "maskBones" in blend_field:
                    mask_weights = create_blendshape_mask(target_obj, blend_field["maskBones"], clothing_avatar_data, field_name=label, store_debug_mask=True)
                
                # 変形後の位置を計算
                deformed_positions = batch_process_vertices_multi_step(
                    deformed_vertices,
                    blend_points,
                    blend_deltas,
                    blend_field_weights,
                    blend_matrix,
                    blend_matrix_inv,
                    target_obj.matrix_world,
                    target_obj.matrix_world.inverted(),
                    mask_weights,
                    batch_size=1000,
                    k=blend_k_neighbors
                )

                # 変位が0かどうかをワールド座標でチェック
                has_displacement = False
                for i in range(len(deformed_vertices)):
                    displacement = deformed_positions[i] - (target_obj.matrix_world @ Vector(deformed_vertices[i]))
                    if np.any(np.abs(displacement) > 1e-5):  # 微小な変位は無視
                        print(f"blendShapeFields {label} world_displacement: {displacement}")
                        has_displacement = True
                        break

                # 変位が存在する場合のみシェイプキーを作成
                if has_displacement:
                    
                    blend_shape_key_name = label
                    if target_obj.data.shape_keys and label in target_obj.data.shape_keys.key_blocks:
                        blend_shape_key_name = f"{label}_generated"
                    
                    # シェイプキーを作成
                    shape_key_b = target_obj.shape_key_add(name=blend_shape_key_name)
                    shape_key_b.value = 0.0  # 初期値は0

                    # シェイプキーに頂点位置を保存
                    matrix_armature_inv_fallback = Matrix.Identity(4)
                    for i in range(len(vertices)):
                        matrix_armature_inv = calculate_inverse_pose_matrix(target_obj, armature_obj, i)
                        if matrix_armature_inv is None:
                            matrix_armature_inv = matrix_armature_inv_fallback
                        # 変形後の位置をローカル座標に変換
                        deformed_world_pos = matrix_armature_inv @ Vector(deformed_positions[i])
                        deformed_local_pos = target_obj.matrix_world.inverted() @ deformed_world_pos
                        shape_key_b.data[i].co = deformed_local_pos
                        matrix_armature_inv_fallback = matrix_armature_inv
                else:
                    print(f"Skipping creation of shape key '{label}' as it has no displacement")

            else:
                print(f"Warning: Field file not found for blend shape {label}")
        # 元のシェイプキー設定を復元
        restore_shape_key_state(target_obj, original_shape_key_state)

    # すべてのシェイプキーの値を元に戻す
    for sk in target_obj.data.shape_keys.key_blocks:
        sk.value = 0.0
