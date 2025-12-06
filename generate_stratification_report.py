import os
import ast
import sys
from collections import defaultdict


def get_definitions_and_lines(file_path):
    defs = {} # name -> line_count
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
            tree = ast.parse(source, filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, 'end_lineno', start)
                defs[node.name] = end - start + 1
    except Exception:
        pass
    return defs

def get_calls(file_path):
    calls = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read(), filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    except Exception:
        pass
    return calls

def calculate_levels(graph, nodes):
    """
    再帰的にレベルを計算する
    Level 0 = 依存なし
    Level N = 1 + max(Level of children)
    """
    levels = {}
    
    def get_level(node, path=set()):
        if node in levels:
            return levels[node]
        
        children = graph.get(node, set())
        real_children = [c for c in children if c != node and c in nodes]
        
        if not real_children:
            levels[node] = 0
            return 0
        
        if node in path:
            return float('inf') 

        path.add(node)
        child_levels = [get_level(child, path) for child in real_children]
        path.remove(node)
        
        my_level = 1 + max(child_levels)
        levels[node] = my_level
        return my_level

    for node in nodes:
        get_level(node)
        
    return levels

def main():
    target_dir = os.path.join(os.getcwd(), "extracted")
    if not os.path.exists(target_dir):
         print(f"Directory '{target_dir}' not found.")
         return

    files = []
    for root, _, filenames in os.walk(target_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                files.append(os.path.join(root, filename))

    module_map = {}
    for path in files:
        rel_path = os.path.relpath(path, target_dir)
        module_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
        module_map[module_name] = path
    
    symbol_info = {} # symbol -> {'file': file, 'lines': n}
    symbol_to_module = {} # symbol -> module_name
    
    print("Parsing files...")
    for module_name, path in module_map.items():
        defs = get_definitions_and_lines(path)
        
        rel_display = os.path.relpath(path, target_dir)
        symbol_info[module_name] = {'file': rel_display, 'lines': sum(defs.values()) if defs else 0}
        symbol_to_module[module_name] = module_name
        
        for d, lines in defs.items():
            symbol_info[d] = {'file': rel_display, 'lines': lines}
            symbol_to_module[d] = module_name

    graph = defaultdict(set)
    
    for module_name, path in module_map.items():
        calls = get_calls(path)
        
        for call in calls:
            if call in symbol_to_module:
                target_mod = symbol_to_module[call]
                if target_mod != module_name: # 自分自身への呼び出しは除外
                    graph[module_name].add(target_mod)

    all_modules = list(module_map.keys())
    levels = calculate_levels(graph, all_modules)

    sorted_nodes = sorted(levels.items(), key=lambda x: (x[1], x[0]))
    
    level_groups = defaultdict(list)
    for node, level in sorted_nodes:
        if level == float('inf'):
            level_groups["Circular / Error"].append(node)
        else:
            level_groups[level].append(node)

    print("\n" + "="*60)
    print("REFACTORING STRATEGY REPORT (By Level)")
    print("="*60)
    
    total_levels = max(k for k in level_groups.keys() if isinstance(k, int))
    
    for lvl in range(total_levels + 1):
        nodes = level_groups[lvl]
        print(f"\n[ Level {lvl} ] - {len(nodes)} items")
        print(f"  (Functions that only depend on Level {lvl-1} or lower)")
        print("-" * 60)
        
        nodes_with_info = []
        for node in nodes:
            info = symbol_info.get(node, {'lines': '?'})
            nodes_with_info.append((node, info['lines']))
        
        nodes_with_info.sort(key=lambda x: x[1] if isinstance(x[1], int) else 99999)
        
        for node, lines in nodes_with_info:
            deps = graph.get(node, [])
            deps_str = ", ".join(list(deps)[:3]) + ("..." if len(deps)>3 else "")
            print(f"  - {node:<40} | Lines: {lines:<5} | Depends on: [{deps_str}]")

if __name__ == "__main__":
    main()