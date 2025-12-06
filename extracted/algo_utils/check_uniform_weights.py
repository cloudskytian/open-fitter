import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_uniform_weights(mesh_obj, component_verts, armature_obj):
    """
    指定されたコンポーネント内の頂点が一様なボーンウェイトを持つか確認する
    
    Parameters:
        mesh_obj: メッシュオブジェクト
        component_verts: コンポーネントに含まれる頂点インデックスのセット
        armature_obj: ウェイト確認対象のアーマチュア
        
    Returns:
        (bool, dict): 一様なウェイトを持つかどうかのフラグと、ボーン名:ウェイト値の辞書
    """
    if not armature_obj:
        return False, {}
    
    # アーマチュアの全ボーン名を取得
    target_bones = {bone.name for bone in armature_obj.data.bones}
    
    # 最初の頂点のウェイトパターンを取得
    first_vert_idx = next(iter(component_verts))
    first_weights = {}
    
    for group in mesh_obj.vertex_groups:
        if group.name in target_bones:
            weight = 0.0
            try:
                for g in mesh_obj.data.vertices[first_vert_idx].groups:
                    if g.group == group.index:
                        weight = g.weight
                        break
            except RuntimeError:
                pass
            
            if weight > 0:
                first_weights[group.name] = weight
    
    # 他の全頂点が同じウェイトパターンを持つか確認
    for vert_idx in component_verts:
        if vert_idx == first_vert_idx:
            continue
        
        for bone_name, weight in first_weights.items():
            group = mesh_obj.vertex_groups.get(bone_name)
            if not group:
                return False, {}
            
            current_weight = 0.0
            try:
                for g in mesh_obj.data.vertices[vert_idx].groups:
                    if g.group == group.index:
                        current_weight = g.weight
                        break
            except RuntimeError:
                pass
            
            # ウェイト値が異なる場合は一様でない
            if abs(current_weight - weight) >= 0.001:
                return False, {}
        
        # 追加のボーングループがないか確認
        for group in mesh_obj.vertex_groups:
            if group.name in target_bones and group.name not in first_weights:
                weight = 0.0
                try:
                    for g in mesh_obj.data.vertices[vert_idx].groups:
                        if g.group == group.index:
                            weight = g.weight
                            break
                except RuntimeError:
                    pass
                
                if weight > 0:
                    return False, {}
    
    return True, first_weights
