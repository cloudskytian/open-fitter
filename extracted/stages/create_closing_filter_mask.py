import os
import sys

# Add the parent directory (extracted/) to sys.path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blender_utils.create_blendshape_mask import create_blendshape_mask


def create_closing_filter_mask(context):
    context.closing_filter_mask_weights = create_blendshape_mask(
        context.target_obj,
        ["LeftUpperLeg", "RightUpperLeg", "Hips", "Chest", "Spine", "LeftShoulder", "RightShoulder", "LeftBreast", "RightBreast"],
        context.base_avatar_data,
    )
