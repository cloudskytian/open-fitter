import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math


def check_edge_direction_similarity(directions1, directions2, angle_threshold=3.0):
    """
    2つの頂点のエッジ方向セットが類似しているかをチェックする
    
    Parameters:
        directions1: 1つ目の頂点のエッジ方向ベクトルのリスト
        directions2: 2つ目の頂点のエッジ方向ベクトルのリスト
        angle_threshold: 類似と判断する角度の閾値（度）
        
    Returns:
        bool: 少なくとも1つのエッジ方向が類似している場合はTrue
    """
    # 孤立頂点（エッジがない）の場合はFalseを返す
    if not directions1 or not directions2:
        return False
    
    # 角度の閾値をラジアンに変換
    angle_threshold_rad = math.radians(angle_threshold)
    
    # 各方向の組み合わせをチェック
    for dir1 in directions1:
        for dir2 in directions2:
            # 2つの方向ベクトル間の角度を計算
            dot_product = dir1.dot(dir2)
            # 内積が1を超えることがあるため、クランプする
            dot_product = max(min(dot_product, 1.0), -1.0)
            angle = math.acos(dot_product)
            
            # 角度が閾値以下、または180度から閾値を引いた値以上（逆方向も考慮）
            if angle <= angle_threshold_rad or angle >= (math.pi - angle_threshold_rad):
                return True
    
    return False
