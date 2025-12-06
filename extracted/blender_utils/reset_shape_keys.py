import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def reset_shape_keys(obj):
    # オブジェクトにシェイプキーがあるか確認
    if obj.data.shape_keys is not None:
        # シェイプキーのキーブロックをループ
        for kb in obj.data.shape_keys.key_blocks:
            # ベースシェイプ（Basis）以外の値を0にする
            if kb.name != "Basis":
                kb.value = 0.0
