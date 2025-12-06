import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_deformation_fields_mapping(avatar_data: dict) -> tuple:
    """
    アバターデータからBlendShapeの変形フィールドマッピングを取得する
    
    Parameters:
        avatar_data: アバターデータ
        
    Returns:
        tuple: (blendShapeFields, invertedBlendShapeFields) のマッピング辞書のタプル
    """
    blend_shape_fields = {}
    inverted_fields = {}
    
    # blendShapeFieldsから取得
    for field in avatar_data.get('blendShapeFields', []):
        label = field.get('label', '')
        if label:
            blend_shape_fields[label] = field
    
    # invertedBlendShapeFieldsから取得
    for field in avatar_data.get('invertedBlendShapeFields', []):
        label = field.get('label', '')
        if label:
            inverted_fields[label] = field
    
    return blend_shape_fields, inverted_fields
