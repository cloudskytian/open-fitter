import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from io_utils.load_deformation_field_num_steps import load_deformation_field_num_steps


def process_single_blendshape_transition_set(current_settings: list, next_settings: list, 
                                           label: str, source_label: str, blend_shape_groups: dict, 
                                           blend_shape_fields: dict, inverted_blend_shape_fields: dict,
                                           current_config_dir: str, mask_bones: list = None) -> dict:
    """
    単一のBlendShape設定セット間の遷移を処理する
    
    Parameters:
        current_settings: 現在の設定リスト
        next_settings: 次の設定リスト
        label: ラベル名（'Basis'または具体的なblendShapeFieldsラベル）
        blend_shape_groups: BlendShapeGroupsの辞書
        blend_shape_fields: BlendShapeFieldsの辞書
        inverted_blend_shape_fields: invertedBlendShapeFieldsの辞書
        current_config_dir: 現在の設定ファイルのディレクトリ
        
    Returns:
        list: 遷移データのリスト
    """
    # 設定を辞書形式に変換
    current_dict = {item['name']: item['value'] for item in current_settings}
    next_dict = {item['name']: item['value'] for item in next_settings}
    
    # すべてのBlendShape名を収集
    all_blend_shapes = set(current_dict.keys()) | set(next_dict.keys())
    
    transitions = []

    processed_blend_shapes = set()
    
    for blend_shape_name in all_blend_shapes:

        if blend_shape_name in processed_blend_shapes:
            continue

        current_value = current_dict.get(blend_shape_name, 0.0)
        next_value = next_dict.get(blend_shape_name, 0.0)
        
        # 値に変化がある場合のみ処理
        if current_value != next_value:
            transition = {
                'label': label,
                'blend_shape_name': blend_shape_name,
                'from_value': current_value,
                'to_value': next_value,
                'operations': [],
            }
            
            # BlendShapeGroupsでの特別処理
            group_processed = False
            for group_name, group_blend_shapes in blend_shape_groups.items():
                if blend_shape_name in group_blend_shapes:
                    # グループ内の現在の非ゼロ値を探す
                    current_non_zero = None
                    for group_blend_shape in group_blend_shapes:
                        if current_dict.get(group_blend_shape, 0.0) != 0.0:
                            current_non_zero = group_blend_shape
                            break
                    
                    # グループ内の次の非ゼロ値を探す
                    next_non_zero = None
                    for group_blend_shape in group_blend_shapes:
                        if next_dict.get(group_blend_shape, 0.0) != 0.0:
                            next_non_zero = group_blend_shape
                            break
                    
                    # グループ内で異なるBlendShapeが正の値をとる場合
                    if current_non_zero and next_non_zero and current_non_zero != next_non_zero:
                        # 最初に前の値を0にする操作
                        field_file_path = inverted_blend_shape_fields[current_non_zero]['filePath']
                        num_steps = load_deformation_field_num_steps(field_file_path, current_config_dir)
                        current_value = current_dict.get(current_non_zero, 0.0)
                        from_step = int((1.0 - current_value) * num_steps + 0.5)
                        to_step = num_steps
                        transition['operations'].append({
                            'type': 'set_to_zero',
                            'blend_shape': current_non_zero,
                            'from_value': current_value,
                            'to_value': 0.0,
                            'file_path': os.path.join(current_config_dir, field_file_path),
                            'mask_bones': inverted_blend_shape_fields[current_non_zero]['maskBones'],
                            'num_steps': num_steps,
                            'from_step': from_step,
                            'to_step': to_step,
                            'field_type': 'inverted'
                        })
                        # 次に新しい値を設定する操作
                        field_file_path = blend_shape_fields[next_non_zero]['filePath']
                        num_steps = load_deformation_field_num_steps(field_file_path, current_config_dir)
                        next_value = next_dict.get(next_non_zero, 0.0)
                        from_step = 0
                        to_step = int(next_value * num_steps + 0.5)
                        transition['operations'].append({
                            'type': 'set_value',
                            'blend_shape': next_non_zero,
                            'from_value': 0.0,
                            'to_value': next_value,
                            'file_path': os.path.join(current_config_dir, field_file_path),
                            'mask_bones': blend_shape_fields[next_non_zero]['maskBones'],
                            'num_steps': num_steps,
                            'from_step': from_step,
                            'to_step': to_step,
                            'field_type': 'normal'
                        })
                        group_processed = True

                        processed_blend_shapes.add(current_non_zero)
                        processed_blend_shapes.add(next_non_zero)

                        break
            
            # グループ処理がされなかった場合は単純な値の変更として記録
            if not group_processed:
                if current_value > next_value:
                    # 値の減少
                    field_file_path = inverted_blend_shape_fields[blend_shape_name]['filePath']
                    num_steps = load_deformation_field_num_steps(field_file_path, current_config_dir)
                    from_step = int((1.0 - current_value) * num_steps + 0.5)
                    to_step = int((1.0 - next_value) * num_steps + 0.5)
                    transition['operations'].append({
                        'type': 'decrease',
                        'blend_shape': blend_shape_name,
                        'from_value': current_value,
                        'to_value': next_value,
                        'file_path': os.path.join(current_config_dir, field_file_path),
                        'mask_bones': inverted_blend_shape_fields[blend_shape_name]['maskBones'],
                        'num_steps': num_steps,
                        'from_step': from_step,
                        'to_step': to_step,
                        'field_type': 'inverted'
                    })
                else:
                    # 値の増加
                    field_file_path = blend_shape_fields[blend_shape_name]['filePath']
                    num_steps = load_deformation_field_num_steps(field_file_path, current_config_dir)
                    from_step = int(current_value * num_steps + 0.5)
                    to_step = int(next_value * num_steps + 0.5)
                    transition['operations'].append({
                        'type': 'increase',
                        'blend_shape': blend_shape_name,
                        'from_value': current_value,
                        'to_value': next_value,
                        'file_path': os.path.join(current_config_dir, field_file_path),
                        'mask_bones': blend_shape_fields[blend_shape_name]['maskBones'],
                        'num_steps': num_steps,
                        'from_step': from_step,
                        'to_step': to_step,
                        'field_type': 'normal'
                    })
            
            processed_blend_shapes.add(blend_shape_name)
            
            transitions.append(transition)
            print(f"  Transition detected [{label}]: {blend_shape_name} {current_value} -> {next_value}")
    
    transition_set = {
        'label': label,
        'source_label': source_label,  # 選ばれたtargetBlendShapeSettingsのlabelを記録
        'mask_bones': mask_bones,
        'current_settings': current_settings,
        'next_settings': next_settings,
        'transitions': transitions
    }
    
    return transition_set
