import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json


def load_vertex_group(obj, filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    group_name = payload.get("vertex_group_name")
    weights = payload.get("weights", [])
    if not group_name:
        print("JSON に頂点グループ名が含まれていません。")
        return group_name

    vg = obj.vertex_groups.get(group_name)
    if vg is None:
        vg = obj.vertex_groups.new(name=group_name)
    else:
        indices = [v.index for v in obj.data.vertices]
        vg.remove(indices)

    missing_vertices = []
    for record in weights:
        vidx = record.get("vertex_index")
        weight = record.get("weight")
        if vidx is None or weight is None:
            continue
        if vidx >= len(obj.data.vertices):
            missing_vertices.append(vidx)
            continue
        vg.add([vidx], weight, 'REPLACE')

    obj.vertex_groups.active = vg
    print(f"{group_name} を {filepath} から復元しました。")
    if missing_vertices:
        print(f"存在しない頂点インデックス: {missing_vertices}")
    return group_name
