import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mathutils import Vector


def calc_triangle_normal(triangle: list[Vector]) -> Vector:
    """三角形の法線を計算（面積で重み付け）"""
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = v1.cross(v2)
    length = normal.length
    if length > 1e-8:  # 数値的な安定性のため
        return normal / length
    return Vector((0, 0, 0))
