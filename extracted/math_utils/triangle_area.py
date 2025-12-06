import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

from mathutils import Vector


def triangle_area(triangle: list[Vector]) -> float:
    a = (triangle[1] - triangle[0]).length
    b = (triangle[2] - triangle[1]).length
    c = (triangle[0] - triangle[2]).length
    s = (a + b + c) / 2  # 半周長
    # 浮動小数点の誤差による負の値を防ぐため max(..., 0) とする
    area_val = max(s * (s - a) * (s - b) * (s - c), 0)
    area = math.sqrt(area_val)
    return area
