import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def import_base_fbx(filepath: str, automatic_bone_orientation: bool = False) -> None:
    """Import base avatar FBX file."""
    try:
        bpy.ops.import_scene.fbx(
            filepath=filepath,
            use_anim=False,  # アニメーションの読み込みを無効化
            automatic_bone_orientation=automatic_bone_orientation
        )
    except Exception as e:
        raise Exception(f"Failed to import base FBX: {str(e)}")
