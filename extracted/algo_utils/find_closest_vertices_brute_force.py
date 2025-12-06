import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_closest_vertices_brute_force(positions, vertices_world, max_distance=0.0001):
    """
    複数の位置に対して最も近い頂点を総当たりで探索
    
    Args:
        positions: 検索する位置のリスト（ワールド座標）
        vertices_world: メッシュの頂点のワールド座標のリスト
        max_distance: 許容する最大距離
    Returns:
        Dict[int, float]: 頂点インデックスをキーとし、距離を値とする辞書
    """
    valid_mappings = {}
    
    # 各検索位置について
    for i, search_pos in enumerate(positions):
        min_distance = float('inf')
        closest_idx = None
        
        # すべてのメッシュ頂点と距離を計算
        for vertex_idx, vertex_pos in enumerate(vertices_world):
            # ユークリッド距離を計算
            distance = ((search_pos[0] - vertex_pos[0])**2 + 
                       (search_pos[1] - vertex_pos[1])**2 + 
                       (search_pos[2] - vertex_pos[2])**2)**0.5
            
            # より近い頂点が見つかった場合は更新
            if distance < min_distance:
                min_distance = distance
                closest_idx = vertex_idx
        
        # 最大距離以内の場合のみマッピングを追加
        if closest_idx is not None and min_distance < max_distance:
            valid_mappings[i] = closest_idx
    
    return valid_mappings
