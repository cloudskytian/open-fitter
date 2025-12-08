"""main_v2.py: Template依存を排除した新アーキテクチャのエントリポイント

従来のforループ構造を排除し、OutfitRetargetPipelineV2を使用する。

Usage:
    blender --background --python main_v2.py -- <args>
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from outfit_retarget_pipeline_v2 import OutfitRetargetPipelineV2
from parse_args import parse_args


def main():
    """新アーキテクチャのメインエントリポイント
    
    従来のmain.pyとの違い:
    1. forループを排除
    2. 中間config_pairへのベースFBXロードを完全に排除
    3. Phase 1 (メッシュ変形) とPhase 2 (ウェイト転送) を明確に分離
    
    処理フロー:
    - Phase 1: 全てのfield_dataを順次適用してメッシュを変形
    - Phase 2: 最終ターゲットのベースアバターを使用してウェイト転送
    """
    import time
    start_time = time.time()
    
    args, config_pairs = parse_args()
    
    print("=" * 60)
    print("OutfitRetargetPipeline V2 - Template依存排除版")
    print("=" * 60)
    print(f"config_pairs数: {len(config_pairs)}")
    print(f"最終ターゲット: {config_pairs[-1].get('base_fbx', 'N/A')}")
    print("=" * 60)
    
    # 新パイプラインを実行
    pipeline = OutfitRetargetPipelineV2(args, config_pairs)
    success = pipeline.execute()
    
    total_time = time.time() - start_time
    print("=" * 60)
    if success:
        print(f"処理成功: 合計 {total_time:.2f}秒")
    else:
        print(f"処理失敗: 合計 {total_time:.2f}秒")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    main()
