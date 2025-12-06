import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def triangulate_mesh(obj: bpy.types.Object) -> None:
    """
    現在の3DView上でのレンダリングにおける三角面への分割をそのまま利用して、
    メッシュのすべての面を三角面に変換する。
    
    Args:
        obj: 三角分割するメッシュオブジェクト
    """
    if obj is None or obj.type != 'MESH':
        return
    
    # 元のアクティブオブジェクトを保存
    original_active = bpy.context.view_layer.objects.active
    
    try:
        # オブジェクトをアクティブに設定
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # エディットモードに切り替え
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 全ての面を選択
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 三角分割を実行（bmeshの三角分割を使用）
        bpy.ops.mesh.quads_convert_to_tris(quad_method='FIXED', ngon_method='BEAUTY')
        
        # オブジェクトモードに戻る
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print(f"Triangulated mesh: {obj.name}")
        
    except Exception as e:
        print(f"Error triangulating mesh {obj.name}: {e}")
        # エラーが発生した場合もオブジェクトモードに戻る
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass
    
    finally:
        # 元のアクティブオブジェクトを復元
        if original_active:
            bpy.context.view_layer.objects.active = original_active
        obj.select_set(False)
