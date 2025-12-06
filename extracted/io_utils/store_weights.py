import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
