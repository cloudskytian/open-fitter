import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mathutils import Vector


def cluster_components_by_adaptive_distance(component_coords, component_sizes):
    """
    コンポーネント間の距離に基づいてクラスタリングする（サイズに応じた適応的な閾値を使用）
    
    Parameters:
        component_coords: コンポーネントインデックスをキー、頂点座標のリストを値とする辞書
        component_sizes: コンポーネントインデックスをキー、サイズを値とする辞書
        
    Returns:
        list: クラスターのリスト（各クラスターはコンポーネントインデックスのリスト）
    """
    if not component_coords:
        return []
    
    # 各コンポーネントの中心点を計算
    centers = {}
    for comp_idx, coords in component_coords.items():
        if coords:
            center = Vector((0, 0, 0))
            for co in coords:
                center += co
            center /= len(coords)
            centers[comp_idx] = center
    
    # クラスターのリスト（初期状態では各コンポーネントが独立したクラスター）
    clusters = [[comp_idx] for comp_idx in centers.keys()]
    
    # コンポーネントの平均サイズを計算
    if component_sizes:
        average_size = sum(component_sizes.values()) / len(component_sizes)
    else:
        average_size = 0.1  # デフォルト値
    
    # 最小閾値と最大閾値を設定
    min_threshold = 0.1
    max_threshold = 1.0
    
    # クラスターをマージする
    merged = True
    while merged:
        merged = False
        
        # 各クラスターペアをチェック
        for i in range(len(clusters)):
            if i >= len(clusters):  # クラスター数が変わった場合の安全チェック
                break
                
            for j in range(i + 1, len(clusters)):
                if j >= len(clusters):  # クラスター数が変わった場合の安全チェック
                    break
                    
                # 各クラスター内のコンポーネント間の最小距離と関連するサイズを計算
                min_distance = float('inf')
                comp_i_size = 0.0
                comp_j_size = 0.0
                
                for comp_i in clusters[i]:
                    for comp_j in clusters[j]:
                        if comp_i in centers and comp_j in centers:
                            dist = (centers[comp_i] - centers[comp_j]).length
                            if dist < min_distance:
                                min_distance = dist
                                comp_i_size = component_sizes.get(comp_i, average_size)
                                comp_j_size = component_sizes.get(comp_j, average_size)
                
                # 2つのコンポーネントのサイズに基づいて適応的な閾値を計算
                # より大きいコンポーネントのサイズの一定割合を使用
                adaptive_threshold = max(comp_i_size, comp_j_size) * 0.5
                
                # 閾値の範囲を制限
                adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))
                
                # 距離が閾値以下ならクラスターをマージ
                if min_distance <= adaptive_threshold:
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merged = True
                    break
            
            if merged:
                break
    
    return clusters
