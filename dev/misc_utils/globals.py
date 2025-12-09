import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_mesh_cache = {}
_saved_pose_state = None
_previous_pose_state = None
_is_A_pose = False
_deformation_field_cache = {}
