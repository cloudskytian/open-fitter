"""OutfitRetargetPipeline v2: Template依存を排除した新アーキテクチャ

AvatarA -> Template -> AvatarB の変換を、forループではなく
明示的な2フェーズ構造で処理する。

Phase 1 (MeshTransformPhase): メッシュ変形
    - ベースメッシュ不要
    - 複数のfield_dataを順次適用可能
    
Phase 2 (WeightTransferPhase): ウェイト転送・ボーン統合
    - 最終ターゲットのベースメッシュが必要
    - 1回のみ実行
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bpy
from processing_context import ProcessingContext

# Phase 1: ベースメッシュ不要なステージ
from stages.blendshape_application import BlendShapeApplicationStage
from stages.mesh_deformation import MeshDeformationStage
from stages.pose_application import PoseApplicationStage
from stages.pose_finalization import PoseFinalizationStage
from stages.template_adjustment import TemplateAdjustmentStage
from stages.weight_transfer_postprocess import WeightTransferPostProcessStage

# Phase 2: ベースメッシュ必要なステージ（V2版を使用）
from stages.asset_loading_v2 import AssetLoadingStageV2
from stages.asset_normalization_v2 import AssetNormalizationStageV2
from stages.bone_replacement_v2 import BoneReplacementStageV2
from stages.export_preparation import ExportPreparationStage
from stages.weight_transfer_execution import WeightTransferExecutionStage
from stages.weight_transfer_preparation_v2 import WeightTransferPreparationStageV2


class OutfitRetargetPipelineV2:
    """衣装リターゲティングパイプライン v2
    
    AvatarA -> Template -> AvatarB の変換を明示的な2フェーズで処理。
    forループを排除し、責任を明確に分離。
    
    Phase 1 (MeshTransformPhase):
        メッシュ変形処理。ベースメッシュ不要。
        複数のfield_dataを順次適用する。
        
    Phase 2 (WeightTransferPhase):
        ウェイト転送とボーン統合。最終ターゲットのベースメッシュが必要。
        1回のみ実行。
    """
    
    _CTX_ATTRS = frozenset({
        'base_mesh', 'base_armature', 'base_avatar_data',
        'clothing_meshes', 'clothing_armature', 'clothing_avatar_data',
        'cloth_metadata', 'vertex_index_mapping',
        'use_subdivision', 'use_triangulation', 'propagated_groups_map',
        'original_humanoid_bones', 'original_auxiliary_bones',
        'is_A_pose', 'blend_shape_labels',
        'base_weights_time', 'blendshape_time', 'pose_time',
        'cycle1_end_time', 'cycle2_post_end',
        'propagated_end_time', 'bones_replace_time',
        'containing_objects', 'armature_settings_dict',
        'time_module', 'start_time'
    })

    def __init__(self, args, config_pairs):
        """
        Args:
            args: コマンドライン引数
            config_pairs: 設定ペアのリスト
                - config_pairs[:-1]: 中間変換（field_dataのみ使用）
                - config_pairs[-1]: 最終変換（base_fbxも使用）
        """
        object.__setattr__(self, 'args', args)
        object.__setattr__(self, 'config_pairs', config_pairs)
        object.__setattr__(self, 'ctx', ProcessingContext())
        
        # 現在処理中のconfig_pair（Phase 1で更新される）
        object.__setattr__(self, 'config_pair', None)
        object.__setattr__(self, 'pair_index', 0)
        object.__setattr__(self, 'total_pairs', len(config_pairs))

    def __getattr__(self, name):
        if name in OutfitRetargetPipelineV2._CTX_ATTRS:
            return getattr(self.ctx, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in OutfitRetargetPipelineV2._CTX_ATTRS:
            setattr(self.ctx, name, value)
        else:
            object.__setattr__(self, name, value)

    def execute(self):
        """パイプライン全体を実行"""
        try:
            import time
            self.time_module = time
            self.start_time = time.time()
            
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Phase 1: メッシュ変形（全config_pairのfield_dataを適用）
            print("=" * 60)
            print("Phase 1: メッシュ変形")
            print("=" * 60)
            self._execute_phase1_mesh_transform()
            
            # Phase 2: ウェイト転送・ボーン統合（最終config_pairのbase_fbxを使用）
            print("=" * 60)
            print("Phase 2: ウェイト転送・ボーン統合")
            print("=" * 60)
            self._execute_phase2_weight_transfer()
            
            total_time = time.time() - self.start_time
            print(f"処理完了: 合計 {total_time:.2f}秒")
            return True
            
        except Exception as e:
            import traceback
            print("============= Error Details =============")
            print(f"Error message: {str(e)}")
            print("\n============= Full Stack Trace =============")
            print(traceback.format_exc())
            print("==========================================")
            
            output_blend = self.args.output.rsplit('.', 1)[0] + '.blend'
            bpy.ops.wm.save_as_mainfile(filepath=output_blend)
            return False

    def _execute_phase1_mesh_transform(self):
        """Phase 1: メッシュ変形処理
        
        ベースメッシュ不要。衣装メッシュに対してfield_dataを順次適用。
        """
        # 最初のconfig_pairで衣装をロード
        first_pair = self.config_pairs[0]
        object.__setattr__(self, 'config_pair', first_pair)
        object.__setattr__(self, 'pair_index', 0)
        
        self.use_subdivision = not self.args.no_subdivision
        self.use_triangulation = not self.args.no_triangle
        
        # 衣装アセットのロード（ベースアバターはロードしない）
        print("--- 衣装アセットロード ---")
        self._load_clothing_assets()
        
        # 衣装の正規化（ベースアバター関連処理はスキップ）
        print("--- 衣装正規化 ---")
        self._normalize_clothing_assets()
        
        # 各config_pairのfield_dataを順次適用
        for i, config_pair in enumerate(self.config_pairs):
            object.__setattr__(self, 'config_pair', config_pair)
            object.__setattr__(self, 'pair_index', i)
            
            print(f"--- Field Data適用 ({i+1}/{len(self.config_pairs)}) ---")
            
            # 2回目以降のpairではclothing_avatar_dataを更新
            # （各pairのinvertedBlendShapeFields/blendShapeFieldsが異なる可能性がある）
            if i > 0:
                from io_utils.io_utils import load_avatar_data
                self.clothing_avatar_data = load_avatar_data(config_pair['clothing_avatar_data'])
                self.base_avatar_data = load_avatar_data(config_pair['base_avatar_data'])
                print(f"  clothing_avatar_data更新: {config_pair['clothing_avatar_data']}")
            
            # Template調整（条件付き）
            if not TemplateAdjustmentStage(self).run():
                return
            
            # BlendShape変形フィールド適用
            BlendShapeApplicationStage(self).run()
            
            # ポーズ適用（最初のpairのみ）
            if i == 0:
                PoseApplicationStage(self).run()
            
            # メッシュ変形（最初のpairのみ）
            if i == 0:
                MeshDeformationStage(self).run()
        
        # Phase 1終了時刻を記録（Phase 2のタイミング計算用）
        self.cycle1_end_time = self.time_module.time()
        print(f"Phase 1完了: {self.cycle1_end_time - self.start_time:.2f}秒")

    def _execute_phase2_weight_transfer(self):
        """Phase 2: ウェイト転送・ボーン統合
        
        最終ターゲットのベースメッシュが必要。
        """
        # 最終config_pairを使用
        final_pair = self.config_pairs[-1]
        object.__setattr__(self, 'config_pair', final_pair)
        object.__setattr__(self, 'pair_index', len(self.config_pairs) - 1)
        
        # ベースアバターのロード（V2ステージ使用）
        print("--- ベースアバターロード ---")
        AssetLoadingStageV2(self).run()
        
        # ベースアバターの正規化（V2ステージ使用）
        print("--- ベースアバター正規化 ---")
        AssetNormalizationStageV2(self).run()
        
        # ウェイト転送準備
        print("--- ウェイト転送準備 ---")
        WeightTransferPreparationStageV2(self).run()
        
        # ウェイト転送実行
        print("--- ウェイト転送実行 ---")
        WeightTransferExecutionStage(self).run()
        
        # ウェイト転送後処理
        print("--- ウェイト転送後処理 ---")
        WeightTransferPostProcessStage(self).run()
        
        # ポーズ最終化
        print("--- ポーズ最終化 ---")
        PoseFinalizationStage(self).run()
        
        # ボーン置換
        print("--- ボーン置換 ---")
        BoneReplacementStageV2(self).run()
        
        # エクスポート準備
        print("--- エクスポート準備 ---")
        ExportPreparationStage(self).run()

    def _load_clothing_assets(self):
        """衣装アセットのみをロード（ベースアバターはロードしない）"""
        from blender_utils.process_clothing_avatar import process_clothing_avatar
        from common_utils.rename_shape_keys_from_mappings import rename_shape_keys_from_mappings
        from common_utils.truncate_long_shape_key_names import truncate_long_shape_key_names
        from io_utils.io_utils import load_base_file, load_cloth_metadata, load_mesh_material_data, load_avatar_data
        
        p = self
        time = p.time_module
        
        # ベースBlendファイル読み込み
        load_base_file(p.args.base)
        
        # ベースアバターのavatar_dataのみロード（FBXはロードしない）
        p.base_avatar_data = load_avatar_data(p.config_pair['base_avatar_data'])
        p.base_mesh = None
        p.base_armature = None
        
        # 衣装アバター処理
        (
            p.clothing_meshes,
            p.clothing_armature,
            p.clothing_avatar_data,
        ) = process_clothing_avatar(
            p.config_pair['input_clothing_fbx_path'],
            p.config_pair['clothing_avatar_data'],
            p.config_pair['hips_position'],
            p.config_pair['target_meshes'],
            p.config_pair['mesh_renderers'],
        )
        
        # シェイプキーのリネーム
        if p.config_pair.get('blend_shape_mappings'):
            rename_shape_keys_from_mappings(
                p.clothing_meshes, p.config_pair['blend_shape_mappings']
            )
        
        truncate_long_shape_key_names(p.clothing_meshes, p.clothing_avatar_data)
        
        # メタデータ読み込み
        (
            p.cloth_metadata,
            p.vertex_index_mapping,
        ) = load_cloth_metadata(p.args.cloth_metadata)
        
        # マテリアルデータ読み込み
        load_mesh_material_data(p.args.mesh_material_data)

    def _normalize_clothing_assets(self):
        """衣装アセットの正規化（ベースアバター関連処理はスキップ）"""
        import json
        from is_A_pose import is_A_pose
        from blender_utils.armature_utils import normalize_clothing_bone_names
        from blender_utils.bone_utils import apply_bone_name_conversion
        
        p = self
        
        # Aポーズ判定
        p.is_A_pose = is_A_pose(
            p.clothing_avatar_data,
            p.clothing_armature,
            init_pose_filepath=p.config_pair['init_pose'],
            pose_filepath=p.config_pair['pose_data'],
            clothing_avatar_data_filepath=p.config_pair['clothing_avatar_data'],
        )
        print(f"is_A_pose: {p.is_A_pose}")
        
        # Aポーズの場合、Aポーズ用ベースポーズを使用
        if (
            p.is_A_pose
            and p.base_avatar_data
            and p.base_avatar_data.get('basePoseA', None)
        ):
            print("AポーズのためAポーズ用ベースポーズを使用")
            p.base_avatar_data['basePose'] = p.base_avatar_data['basePoseA']
        
        # ボーン名変換
        if hasattr(p.args, 'name_conv') and p.args.name_conv:
            try:
                with open(p.args.name_conv, 'r', encoding='utf-8') as f:
                    name_conv_data = json.load(f)
                apply_bone_name_conversion(
                    p.clothing_armature, p.clothing_meshes, name_conv_data
                )
                print(f"ボーン名前変更処理完了: {p.args.name_conv}")
            except Exception as e:
                print(f"Warning: ボーン名前変更処理でエラーが発生しました: {e}")
        
        # 衣装ボーン名の正規化
        normalize_clothing_bone_names(
            p.clothing_armature,
            p.clothing_avatar_data,
            p.clothing_meshes,
        )


# 後方互換性のためのエイリアス
OutfitRetargetPipeline = OutfitRetargetPipelineV2
