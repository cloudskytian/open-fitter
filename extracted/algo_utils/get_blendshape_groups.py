import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_blendshape_groups(avatar_data: dict) -> dict:
    """
    アバターデータからBlendShapeGroupsを取得する
    
    Parameters:
        avatar_data: アバターデータ
        
    Returns:
        dict: BlendShapeGroup名をキーとし、そのグループに含まれるBlendShape名のリストを値とする辞書
    """
    groups = {}
    blend_shape_groups = avatar_data.get('blendShapeGroups', [])
    for group in blend_shape_groups:
        group_name = group.get('name', '')
        blend_shape_fields = group.get('blendShapeFields', [])
        groups[group_name] = blend_shape_fields
    return groups
