import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def calculate_weight_pattern_similarity(weights1, weights2):
    """
    2つのウェイトパターン間の類似性を計算する
    
    Parameters:
        weights1: 1つ目のウェイトパターン {group_name: weight}
        weights2: 2つ目のウェイトパターン {group_name: weight}
        
    Returns:
        float: 類似度（0.0〜1.0、1.0が完全一致）
    """
    # 両方のパターンに存在するグループを取得
    all_groups = set(weights1.keys()) | set(weights2.keys())
    
    if not all_groups:
        return 0.0
    
    # 各グループのウェイト差の合計を計算
    total_diff = 0.0
    for group in all_groups:
        w1 = weights1.get(group, 0.0)
        w2 = weights2.get(group, 0.0)
        total_diff += abs(w1 - w2)
    
    # 正規化（グループ数で割る）
    normalized_diff = total_diff / len(all_groups)
    
    # 類似度に変換（差が小さいほど類似度が高い）
    similarity = 1.0 - min(normalized_diff, 1.0)
    
    return similarity
