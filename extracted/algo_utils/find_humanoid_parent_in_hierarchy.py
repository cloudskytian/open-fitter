import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Optional


def find_humanoid_parent_in_hierarchy(bone_name: str, clothing_avatar_data: dict, base_avatar_data: dict) -> Optional[str]:
    """
    clothing_avatar_dataのboneHierarchyでbone_nameから親を辿り、base_armatureにも存在する最初のhumanoidボーンを返す
    
    Parameters:
        bone_name: 開始ボーン名
        clothing_avatar_data: 衣装のアバターデータ
        base_avatar_data: ベースのアバターデータ
        
    Returns:
        Optional[str]: 見つかった親のHumanoidボーン名、見つからない場合はNone
    """
    # clothing_avatar_dataのhumanoidBonesからbone_nameのhumanoidBoneNameを取得
    clothing_bones_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] 
                                for bone_map in clothing_avatar_data["humanoidBones"]}
    base_humanoid_bones = {bone_map["humanoidBoneName"] for bone_map in base_avatar_data["humanoidBones"]}
    
    def find_bone_in_hierarchy(hierarchy_node, target_name):
        """階層内でボーンを探す再帰関数"""
        if hierarchy_node["name"] == target_name:
            return hierarchy_node
        for child in hierarchy_node.get("children", []):
            result = find_bone_in_hierarchy(child, target_name)
            if result:
                return result
        return None
    
    def find_parent_path(hierarchy_node, target_name, path=[]):
        """ターゲットボーンまでのパスを見つける再帰関数"""
        current_path = path + [hierarchy_node["name"]]
        if hierarchy_node["name"] == target_name:
            return current_path
        for child in hierarchy_node.get("children", []):
            result = find_parent_path(child, target_name, current_path)
            if result:
                return result
        return None
    
    # boneHierarchyでbone_nameまでのパスを取得
    bone_hierarchy = clothing_avatar_data.get("boneHierarchy")
    if not bone_hierarchy:
        return None
    
    path = find_parent_path(bone_hierarchy, bone_name)
    if not path:
        return None
    
    # パスを逆順にして親から辿る
    path.reverse()
    
    # 自分から親に向かってhumanoidボーンを探す
    for parent_bone_name in path:
        if parent_bone_name in clothing_bones_to_humanoid:
            humanoid_name = clothing_bones_to_humanoid[parent_bone_name]
            if humanoid_name in base_humanoid_bones:
                return humanoid_name
    
    return None
