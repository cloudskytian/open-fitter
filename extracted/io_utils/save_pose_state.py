import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def save_pose_state(armature_obj: bpy.types.Object) -> dict:
    """
    アーマチュアの現在のポーズ状態を保存する
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
        
    Returns:
        保存されたポーズ状態のディクショナリ
    """
    if not armature_obj or armature_obj.type != 'ARMATURE':
        return None
    
    pose_state = {}
    for bone in armature_obj.pose.bones:
        pose_state[bone.name] = {
            'matrix': bone.matrix.copy(),
            'location': bone.location.copy(),
            'rotation_euler': bone.rotation_euler.copy(),
            'rotation_quaternion': bone.rotation_quaternion.copy(),
            'scale': bone.scale.copy()
        }
    
    return pose_state
