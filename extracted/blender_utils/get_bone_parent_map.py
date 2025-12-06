import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_bone_parent_map(bone_hierarchy: dict) -> dict:
    """
    Create a map of bones to their parents from the hierarchy.
    
    Parameters:
        bone_hierarchy: Bone hierarchy data from avatar data
    
    Returns:
        Dictionary mapping bone names to their parent bone names
    """
    parent_map = {}
    
    def traverse_hierarchy(node, parent=None):
        current_bone = node["name"]
        parent_map[current_bone] = parent
        
        for child in node.get("children", []):
            traverse_hierarchy(child, current_bone)
    
    traverse_hierarchy(bone_hierarchy)
    return parent_map
