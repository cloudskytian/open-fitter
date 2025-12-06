import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bpy


def round_bone_coordinates(armature: bpy.types.Object, decimal_places: int = 6) -> None:
    """
    アーマチュアのすべてのボーンのhead、tail座標およびRoll値を指定された小数点位置で四捨五入する。
    
    Args:
        armature: 対象のアーマチュアオブジェクト
        decimal_places: 四捨五入する小数点以下の桁数 (デフォルト: 6)
    """
    if not armature or armature.type != 'ARMATURE':
        print(f"Warning: Invalid armature object for rounding bone coordinates")
        return
    
    # エディットモードに切り替え
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    
    try:
        edit_bones = armature.data.edit_bones
        rounded_count = 0
        
        for bone in edit_bones:
            # headの座標を四捨五入
            bone.head.x = round(bone.head.x, decimal_places)
            bone.head.y = round(bone.head.y, decimal_places)
            bone.head.z = round(bone.head.z, decimal_places)
            
            # tailの座標を四捨五入
            bone.tail.x = round(bone.tail.x, decimal_places)
            bone.tail.y = round(bone.tail.y, decimal_places)
            bone.tail.z = round(bone.tail.z, decimal_places)
            
            # Roll値を四捨五入
            bone.roll = round(bone.roll, decimal_places - 3)
            
            rounded_count += 1
        
        print(f"ボーン座標の四捨五入完了: {rounded_count}個のボーン（小数点以下{decimal_places}桁）")
        
    finally:
        # 元のモードに戻す
        bpy.ops.object.mode_set(mode='OBJECT')
