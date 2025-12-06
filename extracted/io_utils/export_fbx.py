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
