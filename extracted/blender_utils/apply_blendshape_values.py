import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def apply_blendshape_values(mesh_obj: bpy.types.Object, blendshapes: list) -> None:
    """Apply blendshape values from avatar data."""
    if not mesh_obj.data.shape_keys:
        return
        
    # Create a mapping of shape key names
    shape_keys = mesh_obj.data.shape_keys.key_blocks
    
    # Apply values
    for blendshape in blendshapes:
        shape_key_name = blendshape["name"]
        if shape_key_name in shape_keys:
            # Set value to 1% of the specified value
            shape_keys[shape_key_name].value = blendshape["value"] * 0.01
