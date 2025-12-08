"""AssetLoadingStageV2: Phase対応版のアセットローディングステージ

新アーキテクチャでは、このステージはPhase 2でのみ使用され、
ベースアバターのFBXロードのみを担当する。

衣装アセットのロードはOutfitRetargetPipelineV2._load_clothing_assets()で行う。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURR_DIR)
for _p in (_PARENT_DIR,):
    if _p not in sys.path:
        sys.path.append(_p)

from process_base_avatar import process_base_avatar_preserve_clothing


class AssetLoadingStageV2:
    """Phase 2用: ベースアバターFBXをロードするステージ
    
    責務:
        - ベースアバターFBXのインポートと処理
        - 衣装オブジェクトの保持
    
    前提:
        - Phase 2でのみ呼び出される
        - 衣装アセットは既にロード済み（clothing_meshes, clothing_armature）
    
    成果物:
        - base_mesh, base_armature, base_avatar_data
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        p = self.pipeline
        time = p.time_module

        print("Status: ベースアバター処理中")
        base_start_time = time.time()

        # 衣装オブジェクトを保持しながらベースアバターをロード
        (
            p.base_mesh,
            p.base_armature,
            p.base_avatar_data,
        ) = process_base_avatar_preserve_clothing(
            p.config_pair['base_fbx'],
            p.config_pair['base_avatar_data'],
            p.clothing_meshes,
            p.clothing_armature,
        )

        base_load_time = time.time()
        print(f"ベースアバター処理: {base_load_time - base_start_time:.2f}秒")
