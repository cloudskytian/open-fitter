import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from math_utils.calculate_blendshape_settings_difference import (
    calculate_blendshape_settings_difference,
)


def find_best_matching_target_settings(source_label: str, 
                                     all_target_settings: dict, 
                                     all_target_mask_bones: dict,
                                     source_settings: list,
                                     blend_shape_fields: dict,
                                     config_dir: str,
                                     mask_bones: list = None) -> tuple:
    """
    sourceBlendShapeSettingsに最も近いtargetBlendShapeSettingsを見つける
    
    Parameters:
        all_target_settings: ラベルごとのtargetBlendShapeSettingsの辞書
        all_target_mask_bones: ラベルごとのmaskBonesの辞書
        source_settings: sourceBlendShapeSettings
        blend_shape_fields: BlendShapeFieldsの辞書
        config_dir: 設定ファイルのディレクトリ
        mask_bones: 比較対象のmaskBones
        
    Returns:
        tuple: (best_label, best_target_settings)
    """
    best_label = None
    best_target_settings = None
    min_difference = float('inf')
    
    for label, target_settings in all_target_settings.items():
        # mask_bonesとall_target_mask_bones[label]の間に共通要素があるかチェック
        if mask_bones is not None and label in all_target_mask_bones:
            target_mask_bones = all_target_mask_bones[label]
            if target_mask_bones is not None:
                # setに変換して共通要素をチェック
                mask_bones_set = set(mask_bones)
                target_mask_bones_set = set(target_mask_bones)
                
                # 共通要素がない場合はスキップ
                if not mask_bones_set.intersection(target_mask_bones_set):
                    print(f"label: {label} - skip: no common mask_bones")
                    continue
        
        difference = calculate_blendshape_settings_difference(
            target_settings, source_settings, blend_shape_fields, config_dir
        )

        # labelとsource_labelから___idを取り除いて比較
        label_without_id = label.split('___')[0] if '___' in label else label
        source_label_without_id = source_label.split('___')[0] if '___' in source_label else source_label
        
        # labelがsource_labelの場合は、差異を1.5で割り優先度を上げる
        if label_without_id == source_label_without_id:
            difference = difference / 1.5
        else:
            difference = difference + 0.00001
        
        print(f"label: {label} difference: {difference}")
        
        if difference < min_difference:
            min_difference = difference
            best_label = label
            best_target_settings = target_settings
    
    return best_label, best_target_settings
