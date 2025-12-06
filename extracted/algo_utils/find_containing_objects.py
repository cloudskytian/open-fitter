import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy
from mathutils.bvhtree import BVHTree


def find_containing_objects(clothing_meshes, threshold=0.02):
    """
    あるオブジェクトが他のオブジェクト全体を包含するペアを見つける
    複数のオブジェクトに包含される場合は平均距離が最も小さいものにのみ包含される
    
    Parameters:
        clothing_meshes: チェック対象のメッシュオブジェクトのリスト
        threshold: 距離の閾値
        
    Returns:
        dict: 包含するオブジェクトをキー、包含されるオブジェクトのリストを値とする辞書
    """
    # 頂点間の平均距離を追跡する辞書
    average_distances = {}  # {(container, contained): average_distance}
    
    # 各オブジェクトペアについてチェック
    for i, obj1 in enumerate(clothing_meshes):
        for j, obj2 in enumerate(clothing_meshes):
            if i == j:  # 同じオブジェクトはスキップ
                continue
            
            # 距離計算のため評価済みメッシュを取得
            depsgraph = bpy.context.evaluated_depsgraph_get()
            
            eval_obj1 = obj1.evaluated_get(depsgraph)
            eval_mesh1 = eval_obj1.data
            
            eval_obj2 = obj2.evaluated_get(depsgraph)
            eval_mesh2 = eval_obj2.data
            
            # BVHツリーを構築
            bm1 = bmesh.new()
            bm1.from_mesh(eval_mesh1)
            bm1.transform(obj1.matrix_world)
            bvh_tree1 = BVHTree.FromBMesh(bm1)
            
            # すべての頂点が閾値内かどうかのフラグと距離の合計
            all_within_threshold = True
            total_distance = 0.0
            vertex_count = 0
            
            # 2つ目のオブジェクトの各頂点について最近接面までの距離を探索
            for vert in eval_mesh2.vertices:
                # 頂点のワールド座標を計算
                vert_world = obj2.matrix_world @ vert.co
                
                # 最近接点と距離を探索
                nearest = bvh_tree1.find_nearest(vert_world)
                
                if nearest is None:
                    all_within_threshold = False
                    break
                    
                # 距離は4番目の要素（インデックス3）
                distance = nearest[3]
                total_distance += distance
                vertex_count += 1
                
                if distance > threshold:
                    all_within_threshold = False
                    break
            
            # すべての頂点が閾値内であれば、平均距離を記録
            if all_within_threshold and vertex_count > 0:
                average_distance = total_distance / vertex_count
                average_distances[(obj1, obj2)] = average_distance
            
            bm1.free()
    
    # 最も平均距離が小さいコンテナを選択
    best_containers = {}  # {contained: (container, avg_distance)}
    
    for (container, contained), avg_distance in average_distances.items():
        if contained not in best_containers or avg_distance < best_containers[contained][1]:
            best_containers[contained] = (container, avg_distance)
    
    # 結果の辞書を構築
    containing_objects = {}
    
    for contained, (container, _) in best_containers.items():
        if container not in containing_objects:
            containing_objects[container] = []
        containing_objects[container].append(contained)

    if not containing_objects:
        return {}

    # 多重包有関係を統合し、各オブジェクトが一度だけ出現するようにする
    parent_map = {}
    for container, contained_list in containing_objects.items():
        for child in contained_list:
            parent_map[child] = container

    def get_bounding_box_volume(obj):
        try:
            dims = getattr(obj, "dimensions", None)
            if dims is None:
                return 0.0
            return float(dims[0]) * float(dims[1]) * float(dims[2])
        except Exception:
            return 0.0

    def find_root(obj):
        visited_list = []
        visited_set = set()
        current = obj

        while current in parent_map and current not in visited_set:
            visited_list.append(current)
            visited_set.add(current)
            current = parent_map[current]

        if current in visited_set:
            cycle_start = visited_list.index(current)
            cycle_nodes = visited_list[cycle_start:]
            root = max(
                cycle_nodes,
                key=lambda o: (
                    get_bounding_box_volume(o),
                    getattr(o, "name", str(id(o)))
                )
            )
        else:
            root = current

        for node in visited_list:
            parent_map[node] = root

        return root

    def collect_descendants(obj, visited):
        result = []
        for child in containing_objects.get(obj, []):
            if child in visited:
                continue
            visited.add(child)
            result.append(child)
            result.extend(collect_descendants(child, visited))
        return result

    merged_containing_objects = {}
    roots_in_order = []

    for container in containing_objects.keys():
        root = find_root(container)
        if root not in merged_containing_objects:
            merged_containing_objects[root] = []
            roots_in_order.append(root)

    assigned_objects = set()
    for root in roots_in_order:
        visited = {root}
        descendants = collect_descendants(root, visited)
        for child in descendants:
            if child in assigned_objects:
                continue
            merged_containing_objects[root].append(child)
            assigned_objects.add(child)

    for contained, (container, _) in best_containers.items():
        if contained in assigned_objects:
            continue
        root = find_root(container)
        if root not in merged_containing_objects:
            merged_containing_objects[root] = []
            roots_in_order.append(root)
        if contained == root:
            continue
        merged_containing_objects[root].append(contained)
        assigned_objects.add(contained)

    final_result = {root: merged_containing_objects[root] for root in roots_in_order if merged_containing_objects[root]}

    if final_result:
        seen_objects = set()
        duplicate_objects = set()

        for container, contained_list in final_result.items():
            if container in seen_objects:
                duplicate_objects.add(container)
            else:
                seen_objects.add(container)

            for obj in contained_list:
                if obj in seen_objects:
                    duplicate_objects.add(obj)
                else:
                    seen_objects.add(obj)

        if duplicate_objects:
            duplicate_names = sorted(
                {getattr(obj, "name", str(id(obj))) for obj in duplicate_objects}
            )
            print(
                "find_containing_objects: 同じオブジェクトが複数回検出されました -> "
                + ", ".join(duplicate_names)
            )

    return final_result
