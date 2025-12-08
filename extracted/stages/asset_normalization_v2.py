"""AssetNormalizationStageV2: Phase対応版のアセット正規化ステージ

新アーキテクチャでは、このステージはPhase 2でのみ使用され、
ベースアバターの正規化のみを担当する。

衣装の正規化はOutfitRetargetPipelineV2._normalize_clothing_assets()で行う。
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURR_DIR)
for _p in (_PARENT_DIR,):
    if _p not in sys.path:
        sys.path.append(_p)

from add_pose_from_json import add_pose_from_json
from algo_utils.vertex_group_utils import remove_empty_vertex_groups
from blender_utils.weight_transfer_utils import setup_weight_transfer
from math_utils.weight_utils import normalize_bone_weights
from update_base_avatar_weights import update_base_avatar_weights


class AssetNormalizationStageV2:
    """Phase 2用: ベースアバターの正規化を行うステージ
    
    責務:
        - ベースポーズ適用
        - ウェイト転送のセットアップ
        - ベースアバターのウェイト更新・正規化
    
    前提:
        - Phase 2でのみ呼び出される
        - base_mesh, base_armature, base_avatar_dataが存在
    
    成果物:
        - base_weights_time タイムスタンプ
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self):
        p = self.pipeline
        time = p.time_module
        stage_start_time = time.time()

        # Aポーズの場合、Aポーズ用ベースポーズを使用（既にPhase 1で設定済みのはず）
        if (
            p.is_A_pose
            and p.base_avatar_data
            and p.base_avatar_data.get('basePoseA', None)
        ):
            print("AポーズのためAポーズ用ベースポーズを使用")
            p.base_avatar_data['basePose'] = p.base_avatar_data['basePoseA']

        # ベースポーズ適用
        base_pose_filepath = p.base_avatar_data.get('basePose', None)
        if (
            base_pose_filepath
            and p.config_pair.get('do_not_use_base_pose', 0) == 0
        ):
            pose_dir = os.path.dirname(
                os.path.abspath(p.config_pair['base_avatar_data'])
            )
            base_pose_filepath = os.path.join(pose_dir, base_pose_filepath)
            print(f"Applying target avatar base pose from {base_pose_filepath}")
            add_pose_from_json(
                p.base_armature,
                base_pose_filepath,
                p.base_avatar_data,
                invert=False,
            )
        base_pose_time = time.time()
        print(f"ベースポーズ適用: {base_pose_time - stage_start_time:.2f}秒")

        # ウェイト転送セットアップ
        print("Status: ウェイト転送セットアップ中")
        setup_weight_transfer()
        setup_time = time.time()
        print(f"ウェイト転送セットアップ: {setup_time - base_pose_time:.2f}秒")

        # ベースメッシュの空頂点グループを削除
        print("Status: ベースアバターウェイト更新中")
        remove_empty_vertex_groups(p.base_mesh)

        # ベースアバターのウェイト更新
        update_base_avatar_weights(
            p.base_mesh,
            p.clothing_armature,
            p.base_avatar_data,
            p.clothing_avatar_data,
            preserve_optional_humanoid_bones=True,
        )

        # ボーンウェイトの正規化
        normalize_bone_weights(p.base_mesh, p.base_avatar_data)

        p.base_weights_time = time.time()
        print(f"ベースアバターウェイト更新: {p.base_weights_time - setup_time:.2f}秒")
