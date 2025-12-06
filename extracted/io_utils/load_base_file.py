import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def load_base_file(filepath: str) -> None:
    """Load the base Blender file containing the character model."""
    try:
        bpy.ops.wm.open_mainfile(filepath=filepath)
    except Exception as e:
        raise Exception(f"Failed to load base file: {str(e)}")
