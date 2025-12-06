import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def calculate_component_size(coords):
    """
    コンポーネントのサイズを計算する
    
    Parameters:
        coords: 頂点座標のリスト
        
    Returns:
        float: コンポーネントのサイズ（直径または最大の辺の長さ）
    """
    if len(coords) < 2:
        return 0.0
    
    # バウンディングボックスを計算
    min_x = min(co.x for co in coords)
    max_x = max(co.x for co in coords)
    min_y = min(co.y for co in coords)
    max_y = max(co.y for co in coords)
    min_z = min(co.z for co in coords)
    max_z = max(co.z for co in coords)
    
    # バウンディングボックスの対角線の長さを計算
    diagonal = ((max_x - min_x)**2 + (max_y - min_y)**2 + (max_z - min_z)**2)**0.5
    
    return diagonal
