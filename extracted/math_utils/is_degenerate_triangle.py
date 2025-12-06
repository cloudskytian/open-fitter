import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from math_utils.triangle_area import triangle_area
from mathutils import Vector


def is_degenerate_triangle(triangle: list[Vector], epsilon: float = 1e-6) -> bool:
    """三角形が縮退しているかチェック"""
    area = triangle_area(triangle)
    return area < epsilon
