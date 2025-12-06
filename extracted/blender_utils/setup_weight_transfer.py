import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def setup_weight_transfer() -> None:
    """Setup the Robust Weight Transfer plugin settings."""
    try:
        bpy.context.scene.robust_weight_transfer_settings.source_object = bpy.data.objects["Body.BaseAvatar"]
    except Exception as e:
        raise Exception(f"Failed to setup weight transfer: {str(e)}")
