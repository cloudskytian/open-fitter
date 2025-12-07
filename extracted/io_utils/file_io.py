import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def export_fbx(filepath: str, selected_only: bool = True) -> None:
    """Export selected objects to FBX."""
    try:
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            use_selection=selected_only,
            apply_scale_options='FBX_SCALE_ALL',
            apply_unit_scale=True,
            add_leaf_bones=False,
            axis_forward='-Z', axis_up='Y'
        )
    except Exception as e:
        raise Exception(f"Failed to export FBX: {str(e)}")


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


def load_base_file(filepath: str) -> None:
    """Load the base Blender file containing the character model."""
    try:
        bpy.ops.wm.open_mainfile(filepath=filepath)
    except Exception as e:
        raise Exception(f"Failed to load base file: {str(e)}")
