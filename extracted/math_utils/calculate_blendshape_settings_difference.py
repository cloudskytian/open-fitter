import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def calculate_blendshape_settings_difference(settings1: list, settings2: list, 
                                           blend_shape_fields: dict, 
                                           config_dir: str) -> float:
    """
    BlendShapeSettings間の状態差異を計算する
    
    Parameters:
        settings1: 最初のBlendShapeSettings
        settings2: 次のBlendShapeSettings  
        blend_shape_fields: BlendShapeFieldsの辞書
        config_dir: 設定ファイルのディレクトリ
        
    Returns:
        float: 差異の量
    """
    # 設定を辞書形式に変換
    dict1 = {item['name']: item['value'] for item in settings1}
    dict2 = {item['name']: item['value'] for item in settings2}
    
    # すべてのBlendShape名を収集
    all_blend_shapes = set(dict1.keys()) | set(dict2.keys())
    
    total_difference = 0.0
    
    for blend_shape_name in all_blend_shapes:
        value1 = dict1.get(blend_shape_name, 0.0)
        value2 = dict2.get(blend_shape_name, 0.0)
        
        # 値の差の絶対値
        value_diff = abs(value1 - value2)
        
        if value_diff > 0.0 and blend_shape_name in blend_shape_fields:
            # 変形データのファイルパスを取得
            field_file_path = blend_shape_fields[blend_shape_name]['filePath']
            full_field_path = os.path.join(config_dir, field_file_path)
            
            try:
                # 変形データを読み込み
                data = np.load(full_field_path, allow_pickle=True)
                delta_positions = data['all_delta_positions']
                
                total_max_displacement = 0.0
                for i in range(len(delta_positions)):
                    max_displacement = np.max(np.linalg.norm(delta_positions[i], axis=1))
                    total_max_displacement += max_displacement
                
                # if len(delta_positions) > 0:
                #     first_step_deltas = delta_positions[0]
                #     max_displacement = np.max(np.linalg.norm(first_step_deltas, axis=1))
                
                # 差の絶対値に最大変位を掛けて加算
                total_difference += value_diff * total_max_displacement
                
            except Exception as e:
                print(f"Warning: Could not load deformation data for {blend_shape_name}: {e}")
                # データを読み込めない場合は値の差をそのまま使用
                total_difference += value_diff
    
    return total_difference
