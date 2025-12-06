import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def copy_bone_transform(source_bone: bpy.types.EditBone, target_bone: bpy.types.EditBone) -> None:
    """
    Copy transformation data from source bone to target bone.
    
    Parameters:
        source_bone: Source edit bone
        target_bone: Target edit bone
    """
    target_bone.head = source_bone.head.copy()
    target_bone.tail = source_bone.tail.copy()
    target_bone.roll = source_bone.roll
    target_bone.matrix = source_bone.matrix.copy()
    target_bone.length = source_bone.length
