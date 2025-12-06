import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_vertex_groups_and_weights(mesh_obj, vertex_index):
    """頂点の所属する頂点グループとウェイトを取得"""
    groups = {}
    vertex = mesh_obj.data.vertices[vertex_index]

    for g in vertex.groups:
        group_name = mesh_obj.vertex_groups[g.group].name
        groups[group_name] = g.weight
        
    return groups
