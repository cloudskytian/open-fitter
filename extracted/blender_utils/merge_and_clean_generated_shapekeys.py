import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def merge_and_clean_generated_shapekeys(clothing_meshes, blend_shape_labels=None):
    """
    apply_blendshape_deformation_fieldsで作成されたシェイプキーを削除し、
    _generatedサフィックス付きシェイプキーを処理する
    
    _generatedで終わるシェイプキー名から_generatedを除いた名前のシェイプキーが存在する場合、
    そのシェイプキーを_generatedシェイプキーの内容で上書きして、_generatedシェイプキーを削除する
    
    Parameters:
        clothing_meshes: 衣装メッシュのリスト
        blend_shape_labels: ブレンドシェイプラベルのリスト
    """
    for obj in clothing_meshes:
        if not obj.data.shape_keys:
            continue
        
        # _generatedサフィックス付きシェイプキーの処理
        generated_shape_keys = []
        for shape_key in obj.data.shape_keys.key_blocks:
            if shape_key.name.endswith("_generated"):
                generated_shape_keys.append(shape_key.name)
        
        # _generatedシェイプキーを対応するベースシェイプキーに統合
        for generated_name in generated_shape_keys:
            base_name = generated_name[:-10]  # "_generated"を除去
            
            generated_key = obj.data.shape_keys.key_blocks.get(generated_name)
            base_key = obj.data.shape_keys.key_blocks.get(base_name)
            
            if generated_key and base_key:
                # generatedシェイプキーの内容でベースシェイプキーを上書き
                for i, point in enumerate(generated_key.data):
                    base_key.data[i].co = point.co
                print(f"Merged {generated_name} into {base_name} for {obj.name}")
                
                # generatedシェイプキーを削除
                obj.shape_key_remove(generated_key)
                print(f"Removed generated shape key: {generated_name} from {obj.name}")
        
        # 従来の機能: blend_shape_labelsで指定されたシェイプキーの削除
        if blend_shape_labels:
            shape_keys_to_remove = []
            for label in blend_shape_labels:
                shape_key_name = f"{label}_BaseShape"
                if shape_key_name in obj.data.shape_keys.key_blocks:
                    shape_keys_to_remove.append(shape_key_name)
            
            for label in blend_shape_labels:
                shape_key_name = f"{label}_temp"
                if shape_key_name in obj.data.shape_keys.key_blocks:
                    shape_keys_to_remove.append(shape_key_name)
            
            # シェイプキーを削除
            for shape_key_name in shape_keys_to_remove:
                shape_key = obj.data.shape_keys.key_blocks.get(shape_key_name)
                if shape_key:
                    obj.shape_key_remove(shape_key)
                    print(f"Removed shape key: {shape_key_name} from {obj.name}")

        # 不要なシェイプキーを削除
        shape_keys_to_remove = []
        for shape_key in obj.data.shape_keys.key_blocks:
            if shape_key.name.endswith(".MFTemp"):
                shape_keys_to_remove.append(shape_key.name)
        for shape_key_name in shape_keys_to_remove:
            shape_key = obj.data.shape_keys.key_blocks.get(shape_key_name)
            if shape_key:
                obj.shape_key_remove(shape_key)
                print(f"Removed shape key: {shape_key_name} from {obj.name}")
