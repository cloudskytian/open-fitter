import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import bmesh


def find_connected_components(mesh_obj):
    """
    メッシュオブジェクト内で接続していないコンポーネントを検出する
    
    Parameters:
        mesh_obj: 検出対象のメッシュオブジェクト
        
    Returns:
        List[Set[int]]: 各コンポーネントに含まれる頂点インデックスのセットのリスト
    """
    # BMeshを作成し、元のメッシュからデータをコピー
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    
    # 頂点インデックスのマッピングを作成（BMesh内のインデックス → 元のメッシュのインデックス）
    vert_indices = {v.index: i for i, v in enumerate(bm.verts)}
    
    # 未訪問の頂点を追跡
    unvisited = set(vert_indices.keys())
    components = []
    
    while unvisited:
        # 未訪問の頂点から開始
        start_idx = next(iter(unvisited))
        
        # 幅優先探索で連結成分を検出
        component = set()
        queue = [start_idx]
        
        while queue:
            current = queue.pop(0)
            if current in unvisited:
                unvisited.remove(current)
                component.add(vert_indices[current])  # 元のメッシュのインデックスに変換して追加
                
                # 隣接頂点をキューに追加（エッジで接続されている頂点のみ）
                for edge in bm.verts[current].link_edges:
                    other = edge.other_vert(bm.verts[current]).index
                    if other in unvisited:
                        queue.append(other)
        
        # 頂点数が1のコンポーネント（孤立頂点）は除外
        if len(component) > 1:
            components.append(component)
    
    bm.free()
    return components
