import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def barycentric_coords_from_point(p, a, b, c):
    """
    三角形上の点pの重心座標を計算する
    
    Args:
        p: 点の座標（Vector）
        a, b, c: 三角形の頂点座標（Vector）
    
    Returns:
        (u, v, w): 重心座標のタプル（u + v + w = 1）
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    
    denom = d00 * d11 - d01 * d01
    
    if abs(denom) < 1e-10:
        # 退化した三角形の場合は最も近い頂点のウェイトを1にする
        dist_a = (p - a).length
        dist_b = (p - b).length
        dist_c = (p - c).length
        min_dist = min(dist_a, dist_b, dist_c)
        if min_dist == dist_a:
            return (1.0, 0.0, 0.0)
        elif min_dist == dist_b:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return (u, v, w)
