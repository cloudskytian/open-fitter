import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def restore_weights(target_obj, stored_weights):
    """保存したウェイトを復元"""
    for vert_idx, groups in stored_weights.items():
        for group_name, weight in groups.items():
            if group_name in target_obj.vertex_groups:
                target_obj.vertex_groups[group_name].add([vert_idx], weight, 'REPLACE')
