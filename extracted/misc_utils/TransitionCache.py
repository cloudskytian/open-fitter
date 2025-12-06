import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TransitionCache:
    """Transitionの実行結果をキャッシュするクラス"""
    def __init__(self):
        self.cache = {}  # {blend_shape_combination_hash: {vertices: np.array, blendshape_values: dict}}
        
    def get_cache_key(self, blendshape_values):
        """BlendShapeの値からキャッシュキーを生成"""
        sorted_items = sorted(blendshape_values.items())
        return hash(tuple(sorted_items))
    
    def store_result(self, blendshape_values, vertices, all_blendshape_values):
        """実行結果をキャッシュに保存"""
        cache_key = self.get_cache_key(blendshape_values)
        
        # 同じキーが既に存在する場合は上書きしない
        if cache_key in self.cache:
            print(f"Cache key already exists, keeping existing entry: {cache_key}")
            return
        
        self.cache[cache_key] = {
            'vertices': vertices.copy(),
            'blendshape_values': all_blendshape_values.copy()
        }
        print(f"Stored new cache entry: {cache_key}")
    
    def find_interpolation_candidates(self, target_blendshape_values, changing_blendshape, blendshape_groups=None):
        """線形補間候補を検索"""
        candidates = []
        
        # changing_blendshapeが属するBlendShapeGroupを特定
        changing_blendshape_group = None
        group_blendshapes = set()
        
        if blendshape_groups:
            for group in blendshape_groups:
                blendshapes_in_group = group.get('blendShapeFields', [])
                if changing_blendshape in blendshapes_in_group:
                    changing_blendshape_group = group
                    group_blendshapes = set(blendshapes_in_group)
                    break
        
        print(f"target_blendshape_values: {target_blendshape_values}")
        
        for cache_key, cached_data in self.cache.items():
            cached_values = cached_data['blendshape_values']

            print(f"cached_values: {cached_values}")
            
            values_match = True
            
            # BlendShapeGroupに属する場合：同じグループの他のBlendShapeの値が同じかチェック
            if changing_blendshape_group:
                for name in group_blendshapes:
                    if name != changing_blendshape and abs(cached_values.get(name, 0.0) - target_blendshape_values.get(name, 0.0)) > 1e-6:
                        values_match = False
                        break
            
            if values_match:
                cached_changing_value = cached_values.get(changing_blendshape, 0.0)
                target_changing_value = target_blendshape_values.get(changing_blendshape, 0.0)
                
                print(f"cached_changing_value: {cached_changing_value}, target_changing_value: {target_changing_value}")
                
                candidates.append({
                    'cached_value': cached_changing_value,
                    'target_value': target_changing_value,
                    'vertices': cached_data['vertices'],
                    'distance': abs(cached_changing_value - target_changing_value)
                })
        
        return candidates
    
    def interpolate_result(self, target_blendshape_values, changing_blendshape, blendshape_groups=None):
        """線形補間で結果を計算"""
        candidates = self.find_interpolation_candidates(target_blendshape_values, changing_blendshape, blendshape_groups)
        
        if len(candidates) < 2:
            return None
            
        target_value = target_blendshape_values.get(changing_blendshape, 0.0)
        
        # ターゲット値を挟む候補ペアを全て見つける
        valid_pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                val1, val2 = candidates[i]['cached_value'], candidates[j]['cached_value']
                
                if (val1 <= target_value <= val2) or (val2 <= target_value <= val1):
                    if abs(val2 - val1) < 1e-6:
                        continue  # 同じ値の場合はスキップ
                    
                    interval_size = abs(val2 - val1)
                    valid_pairs.append({
                        'interval_size': interval_size,
                        'candidate1': candidates[i],
                        'candidate2': candidates[j],
                        'val1': val1,
                        'val2': val2
                    })
        
        if not valid_pairs:
            return None
        
        # 区間が最も小さいペアを選択
        best_pair = min(valid_pairs, key=lambda x: x['interval_size'])
        
        # 線形補間を実行
        val1, val2 = best_pair['val1'], best_pair['val2']
        t = (target_value - val1) / (val2 - val1)
        vertices1 = best_pair['candidate1']['vertices']
        vertices2 = best_pair['candidate2']['vertices']
        
        interpolated_vertices = vertices1 + t * (vertices2 - vertices1)
        print(f"Using linear interpolation with interval size {best_pair['interval_size']:.6f} for {changing_blendshape}")
        return interpolated_vertices
