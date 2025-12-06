import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_weight_hash(weights):
    """ウェイト辞書からハッシュ値を生成する（0.001より小さい部分を四捨五入）"""
    sorted_items = sorted(weights.items())
    # ウェイト値を0.001の精度で四捨五入
    hash_str = "_".join([f"{name}:{round(weight, 3):.3f}" for name, weight in sorted_items])
    return hash_str
