"""
apply_symmetric_field_delta - 対称Deformation Field差分データをメッシュに適用するモジュール

このモジュールは symmetric_field_deformer パッケージの各処理を
オーケストレーションするエントリーポイントです。
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symmetric_field_deformer.basis_processor import process_basis_loop
from symmetric_field_deformer.blendshape_processor import (
    process_base_avatar_blendshapes,
    process_clothing_blendshapes,
    process_config_blendshapes,
    process_skipped_transitions,
)
from symmetric_field_deformer.context import SymmetricFieldDeformerContext
from symmetric_field_deformer.post_processor import (
    apply_masks_and_cleanup,
    execute_deferred_transitions,
    finalize,
)


def apply_symmetric_field_delta(
    target_obj,
    field_data_path,
    blend_shape_labels=None,
    clothing_avatar_data=None,
    base_avatar_data=None,
    subdivision=True,
    shape_key_name="SymmetricDeformed",
    skip_blend_shape_generation=False,
    config_data=None,
    ignore_blendshape=None,
):
    """
    保存された対称Deformation Field差分データを読み込みメッシュに適用する（最適化版、多段階対応）。

    ※BlendShape用のDeformation Fieldを先に適用した場合と、メインのみ適用した場合の交差面の割合を
      比較し、所定の条件下ではBlendShapeの変位を無視する処理を行います。

    Args:
        target_obj: 対象のBlenderメッシュオブジェクト
        field_data_path: Deformation Fieldデータのパス
        blend_shape_labels: BlendShapeラベルのリスト
        clothing_avatar_data: 衣装アバターデータ
        base_avatar_data: ベースアバターデータ
        subdivision: サブディビジョンを使用するかどうか
        shape_key_name: 作成するシェイプキーの名前
        skip_blend_shape_generation: BlendShape生成をスキップするかどうか
        config_data: 設定データ
        ignore_blendshape: 無視するBlendShapeのリスト

    Returns:
        作成されたシェイプキー
    """
    ctx = SymmetricFieldDeformerContext(
        target_obj,
        field_data_path,
        blend_shape_labels,
        clothing_avatar_data,
        base_avatar_data,
        subdivision,
        shape_key_name,
        skip_blend_shape_generation,
        config_data,
        ignore_blendshape,
    )

    process_basis_loop(ctx)
    process_config_blendshapes(ctx)
    process_skipped_transitions(ctx)
    process_clothing_blendshapes(ctx)
    execute_deferred_transitions(ctx)
    apply_masks_and_cleanup(ctx)
    process_base_avatar_blendshapes(ctx)
    finalize(ctx)

    return ctx.shape_key

