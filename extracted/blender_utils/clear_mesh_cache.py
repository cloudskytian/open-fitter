import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from misc_utils.globals import _mesh_cache


def clear_mesh_cache():
    """
    メッシュキャッシュをクリアします
    """
    global _mesh_cache
    
    # BMeshオブジェクトを解放
    for cache_data in _mesh_cache.values():
        if 'bmesh' in cache_data and cache_data['bmesh']:
            cache_data['bmesh'].free()
    
    _mesh_cache.clear()
    print("メッシュキャッシュをクリアしました")
