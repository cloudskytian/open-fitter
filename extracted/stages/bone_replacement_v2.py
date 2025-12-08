"""BoneReplacementStageV2: Phase対応版のボーン置換ステージ

新アーキテクチャでは、このステージはPhase 2でのみ使用される。
is_final_pairチェックを削除し、base_armatureが必ず存在することを前提とする。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURR_DIR)
for _p in (_PARENT_DIR,):
    if _p not in sys.path:
        sys.path.append(_p)

from replace_humanoid_bones import replace_humanoid_bones


class BoneReplacementStageV2:
    """Phase 2用: ヒューマノイドボーン置換を担当するステージ
    
    責務:
        - ベースアーマチュアから衣装アーマチュアへのヒューマノイドボーン置換
    
    前提:
        - Phase 2でのみ呼び出される
        - base_armatureが存在
        - PoseFinalizationStage が完了していること
    
    成果物:
        - ヒューマノイドボーンが置換された衣装アーマチュア
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        p = self.pipeline
        time = p.time_module

        print("Status: ヒューマノイドボーン置換中")

        # ベースポーズファイルパスの取得
        base_pose_filepath = None
        if p.config_pair.get('do_not_use_base_pose', 0) == 0:
            base_pose_filepath = p.base_avatar_data.get('basePose', None)
            if base_pose_filepath:
                pose_dir = os.path.dirname(
                    os.path.abspath(p.config_pair['base_avatar_data'])
                )
                base_pose_filepath = os.path.join(pose_dir, base_pose_filepath)

        # ヒューマノイドボーン置換
        replace_humanoid_bones(
            p.base_armature,
            p.clothing_armature,
            p.base_avatar_data,
            p.clothing_avatar_data,
            True,
            base_pose_filepath,
            p.clothing_meshes,
            False,
        )

        p.bones_replace_time = time.time()
        print(f"ボーン置換完了")
