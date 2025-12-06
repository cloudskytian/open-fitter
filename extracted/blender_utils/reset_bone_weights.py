import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def reset_bone_weights(target_obj, bone_groups):
    """指定された頂点グループのウェイトを0に設定"""
    for vert in target_obj.data.vertices:
        for group in target_obj.vertex_groups:
            if group.name in bone_groups:
                try:
                    group.add([vert.index], 0, 'REPLACE')
                except RuntimeError:
                    continue
