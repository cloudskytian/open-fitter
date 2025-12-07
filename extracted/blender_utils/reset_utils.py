"""
シェイプキー・ボーンウェイトのリセットユーティリティ
"""


def reset_shape_keys(obj):
    """全シェイプキーの値を0にリセット（Basis以外）"""
    if obj.data.shape_keys is not None:
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 0.0


def reset_bone_weights(target_obj, bone_groups):
    """指定された頂点グループのウェイトを0に設定"""
    for vert in target_obj.data.vertices:
        for group in target_obj.vertex_groups:
            if group.name in bone_groups:
                try:
                    group.add([vert.index], 0, 'REPLACE')
                except RuntimeError:
                    continue
