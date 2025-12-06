import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from io_utils.save_pose_state import save_pose_state


def store_pose_globally(armature_obj: bpy.types.Object) -> None:
    """
    グローバル変数にポーズ状態を保存する
    
    Parameters:
        armature_obj: アーマチュアオブジェクト
    """
    global _saved_pose_state
    _saved_pose_state = save_pose_state(armature_obj)
