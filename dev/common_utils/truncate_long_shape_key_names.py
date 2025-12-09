import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def truncate_long_shape_key_names(clothing_meshes, clothing_avatar_data):
    """
    clothing_avatar_data["blendshapes"]に含まれないシェイプキーで、
    名前が48バイト以上のものを48バイトに切り詰める（UTF-8エンコード）
    多バイト文字が途中で切断されないように処理する
    
    Parameters:
        clothing_meshes: メッシュオブジェクトのリスト
        clothing_avatar_data: 衣装アバターデータ（blendshapes情報を含む）
    """
    if not clothing_avatar_data or 'blendshapes' not in clothing_avatar_data:
        return
    
    # clothing_avatar_dataのblendshapesに含まれる名前のセットを作成
    blendshape_names = set()
    for blendshape in clothing_avatar_data.get('blendshapes', []):
        if isinstance(blendshape, dict) and 'name' in blendshape:
            blendshape_names.add(blendshape['name'])
    
    for obj in clothing_meshes:
        if not obj.data.shape_keys:
            continue
        
        # 名前を変更する必要があるシェイプキーを収集
        keys_to_rename = []
        for shape_key in obj.data.shape_keys.key_blocks:
            # clothing_avatar_data["blendshapes"]に含まれず、かつ48バイト以上の場合
            if shape_key.name not in blendshape_names:
                # UTF-8バイト長をチェック
                name_bytes = shape_key.name.encode('utf-8')
                if len(name_bytes) >= 48:
                    # 多バイト文字を考慮して48バイトに切り詰める
                    truncated_bytes = name_bytes[:48]
                    # 不完全なマルチバイト文字を削除
                    try:
                        truncated_name = truncated_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        # 不完全なバイト列の場合、1バイトずつ削って再試行
                        for i in range(1, 4):  # UTF-8は最大4バイト
                            try:
                                truncated_name = truncated_bytes[:-i].decode('utf-8')
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            # それでも失敗した場合は元の名前を使用（通常発生しない）
                            truncated_name = shape_key.name
                    
                    keys_to_rename.append((shape_key, truncated_name))
        
        # 名前を変更
        for shape_key, truncated_name in keys_to_rename:
            old_name = shape_key.name
            shape_key.name = truncated_name
