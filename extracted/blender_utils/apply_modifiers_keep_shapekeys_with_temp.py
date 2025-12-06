import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy
from blender_utils.apply_all_shapekeys import apply_all_shapekeys
from blender_utils.apply_modifiers import apply_modifiers


def apply_modifiers_keep_shapekeys_with_temp(obj):
    """一時オブジェクトを使用してシェイプキーを維持しながらモディファイアを適用する"""
    if obj.type != 'MESH':
        return
    
    if not obj.data.shape_keys:
        # シェイプキーがない場合は通常のモディファイア適用
        bpy.context.view_layer.objects.active = obj
        for modifier in obj.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            except Exception as e:
                print(f"Failed to apply modifier {modifier.name}: {e}")
        return

    # グローバルカウンタの初期化（存在しない場合）
    if not hasattr(apply_modifiers_keep_shapekeys_with_temp, 'counter'):
        apply_modifiers_keep_shapekeys_with_temp.counter = 0

    shape_keys = obj.data.shape_keys.key_blocks
    temp_objects = []
    shape_key_names = []  # 元のシェイプキー名を保存するリスト
    
    # 各シェイプキーに対して一時オブジェクトを作成
    for i, shape_key in enumerate(shape_keys):
        if i == 0:  # Basis は飛ばす
            continue
        
        # すべてのオブジェクトの選択を解除
        bpy.ops.object.select_all(action='DESELECT')
        # オブジェクトを複製
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.duplicate(linked=False)
        temp_obj = bpy.context.active_object

        temp_obj.name = f"t{apply_modifiers_keep_shapekeys_with_temp.counter}"
        apply_modifiers_keep_shapekeys_with_temp.counter += 1

        temp_objects.append(temp_obj)
        shape_key_names.append(shape_key.name)  # 元の名前を保存
        
        # 他のシェイプキーの値を0に、対象のシェイプキーの値を1に設定
        for sk in temp_obj.data.shape_keys.key_blocks:
            if sk.name == shape_key.name:
                sk.value = 1.0
            else:
                sk.value = 0.0
        
        # シェイプキーを適用
        apply_all_shapekeys(temp_obj)
        
        # Armature以外のモディファイアを適用
        apply_modifiers(temp_obj)

    # 元のオブジェクトの処理
    bpy.context.view_layer.objects.active = obj
    # まず全てのシェイプキーの値を0に設定
    for sk in obj.data.shape_keys.key_blocks:
        sk.value = 0.0
    # シェイプキーを適用
    apply_all_shapekeys(obj)
    # モディファイアを適用
    apply_modifiers(obj)

    # 一時オブジェクトの形状を元のオブジェクトのシェイプキーとして追加
    obj.shape_key_add(name="Basis")
    for temp_obj, original_name in zip(temp_objects, shape_key_names):
        # シェイプキーを追加（元の名前を使用）
        shape_key = obj.shape_key_add(name=original_name)
        shape_key.interpolation = 'KEY_LINEAR'
        if shape_key.name == "SymmetricDeformed":
            shape_key.value = 1.0
        
        # 頂点座標を設定
        for i, vert in enumerate(temp_obj.data.vertices):
            shape_key.data[i].co = vert.co.copy()
        
        # 一時オブジェクトを削除
        bpy.data.objects.remove(temp_obj, do_unlink=True)
