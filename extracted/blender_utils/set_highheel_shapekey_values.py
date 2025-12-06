import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def set_highheel_shapekey_values(clothing_meshes, blend_shape_labels=None, base_avatar_data=None):
    """
    Highheelを含むシェイプキーの値を1にする
    
    Parameters:
        clothing_meshes: 衣装メッシュのリスト
        blend_shape_labels: ブレンドシェイプラベルのリスト
        base_avatar_data: ベースアバターデータ
    """
    if not blend_shape_labels or not base_avatar_data:
        return
    
    # base_avatar_dataのblendShapeFieldsの存在確認
    if "blendShapeFields" not in base_avatar_data:
        return
    
    # まずHighheelを含むラベルを検索
    highheel_labels = [label for label in blend_shape_labels if "highheel" in label.lower() and "off" not in label.lower()]
    base_highheel_fields = [field for field in base_avatar_data["blendShapeFields"] 
                          if "highheel" in field.get("label", "").lower() and "off" not in field.get("label", "").lower()]
    
    # Highheelを含むラベルが無い場合は、Heelを含むラベルを検索
    if not highheel_labels:
        highheel_labels = [label for label in blend_shape_labels if "heel" in label.lower() and "off" not in label.lower()]
        base_highheel_fields = [field for field in base_avatar_data["blendShapeFields"] 
                              if "heel" in field.get("label", "").lower() and "off" not in field.get("label", "").lower()]
    
    # 条件：blend_shape_labelsに該当ラベルが一つだけ、かつbase_avatar_dataに該当フィールドが一つだけ
    if len(highheel_labels) != 1 or len(base_highheel_fields) != 1:
        return
    
    # 唯一のラベルとフィールドを取得
    target_label = highheel_labels[0]
    base_field = base_highheel_fields[0]
    base_label = base_field.get("label", "")
    
    # 各メッシュのシェイプキーをチェック
    for obj in clothing_meshes:
        if not obj.data.shape_keys:
            continue
        
        # base_avatar_dataのラベルでシェイプキーを探す
        if base_label in obj.data.shape_keys.key_blocks:
            shape_key = obj.data.shape_keys.key_blocks[base_label]
            shape_key.value = 1.0
            print(f"Set shape key '{base_label}' value to 1.0 on {obj.name}")
