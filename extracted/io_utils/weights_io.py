"""
頂点ウェイトの保存・復元ユーティリティ
"""


def store_weights(target_obj, bone_groups_to_store):
    """頂点グループのウェイトを保存"""
    weights = {}
    for vert in target_obj.data.vertices:
        weights[vert.index] = {}
        for group in target_obj.vertex_groups:
            if group.name in bone_groups_to_store:
                try:
                    for g in vert.groups:
                        if g.group == group.index:
                            weights[vert.index][group.name] = g.weight
                            break
                except RuntimeError:
                    continue
    return weights


def restore_weights(target_obj, stored_weights):
    """保存したウェイトを復元"""
    for vert_idx, groups in stored_weights.items():
        for group_name, weight in groups.items():
            if group_name in target_obj.vertex_groups:
                target_obj.vertex_groups[group_name].add([vert_idx], weight, 'REPLACE')
