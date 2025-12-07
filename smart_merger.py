import os
import sys
import ast
import re
import glob

# Add current directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_module_path(file_path):
    """Convert file path to module path (e.g. extracted/foo/bar.py -> foo.bar)"""
    rel_path = os.path.relpath(file_path, os.path.join(os.getcwd(), 'extracted'))
    if rel_path.startswith('..'):
        # Fallback for files not in extracted
        rel_path = os.path.relpath(file_path, os.getcwd())
    
    return os.path.splitext(rel_path)[0].replace(os.sep, '.')

def get_defined_symbols(code):
    """Get top-level function and class names defined in the code"""
    try:
        tree = ast.parse(code)
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        return symbols
    except SyntaxError:
        print("Warning: Syntax error parsing code for symbols")
        return []

def split_imports_and_body(code):
    """Split code into imports and body"""
    lines = code.splitlines()
    imports = []
    body = []
    
    in_docstring = False
    in_import = False
    paren_depth = 0
    current_import = []
    
    for line in lines:
        stripped = line.strip()
        
        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                pass 
            else:
                in_docstring = not in_docstring
            
            if not imports and not body and not in_import: # File docstring
                continue 
            
            body.append(line)
            continue
            
        if in_docstring:
            body.append(line)
            continue
            
        # Skip empty lines if we haven't started body yet
        if not stripped and not in_import:
            if imports and not body:
                continue
            if body:
                body.append(line)
            continue

        # Check for sys.path hacks
        if 'sys.path.append' in line or 'os.path.dirname' in line:
            continue 

        # Import detection logic
        if in_import:
            current_import.append(line)
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0 and not stripped.endswith('\\'):
                in_import = False
                paren_depth = 0
                imports.append("\n".join(current_import))
                current_import = []
        else:
            # Start of new import?
            if (stripped.startswith('import ') or stripped.startswith('from ')) and not line.startswith(' '):
                current_import = [line]
                paren_depth += line.count('(') - line.count(')')
                if paren_depth > 0 or stripped.endswith('\\'):
                    in_import = True
                else:
                    imports.append(line)
                    current_import = []
            else:
                body.append(line)
            
    return imports, "\n".join(body)

def update_imports_in_file(file_path, child_module, parent_module, symbols):
    """Update imports in a file to point to parent instead of child"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return # Skip binary files
        
    original_content = content
    
    # Pattern 1: from child_module import Symbol
    # Replace with: from parent_module import Symbol
    
    # We need to handle relative imports too if we are in the same package
    # But for now let's assume absolute imports from 'extracted' root or similar
    
    # Regex for "from child_module import ..."
    # We want to match "from child_module" but not "from child_module_foo"
    
    # Case A: Exact match of module name
    # from child.module import X -> from parent.module import X
    
    # Escape dots for regex
    child_mod_regex = child_module.replace('.', r'\.')
    parent_mod_str = parent_module
    
    # 1. Replace "from child_module import"
    pattern = r'from\s+' + child_mod_regex + r'\s+import'
    replacement = f'from {parent_mod_str} import'
    content = re.sub(pattern, replacement, content)
    
    # 2. Replace "import child_module"
    # This is trickier because usage needs update: child_module.Symbol -> parent_module.Symbol
    # But in this project, most imports are "from ... import ..."
    # We will log a warning if we see "import child_module"
    if re.search(r'^\s*import\s+' + child_mod_regex + r'\b', content, re.MULTILINE):
        print(f"Warning: Direct import of {child_module} found in {file_path}. Manual check recommended.")
        
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated imports in {file_path}")

def merge_modules(parent_path, child_paths):
    """Merge child modules into parent module"""
    
    # 1. Read Parent
    if not os.path.exists(parent_path):
        # Create parent if it doesn't exist
        print(f"Creating new parent file: {parent_path}")
        with open(parent_path, 'w', encoding='utf-8') as f:
            f.write('import os\nimport sys\n\n')
            f.write('sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n\n')
    
    with open(parent_path, 'r', encoding='utf-8') as f:
        parent_code = f.read()
        
    parent_imports, parent_body = split_imports_and_body(parent_code)
    
    # 2. Process Children
    all_new_body = []
    all_new_imports = set(parent_imports)
    
    parent_module = get_module_path(parent_path)
    
    for child_path in child_paths:
        if not os.path.exists(child_path):
            print(f"Error: Child file not found: {child_path}")
            continue
            
        print(f"Processing {child_path}...")
        child_module = get_module_path(child_path)
        
        with open(child_path, 'r', encoding='utf-8') as f:
            child_code = f.read()
            
        child_imports, child_body = split_imports_and_body(child_code)
        child_symbols = get_defined_symbols(child_code)
        
        # Add imports
        for imp in child_imports:
            all_new_imports.add(imp)
            
        # Add body
        all_new_body.append(f"\n# Merged from {os.path.basename(child_path)}\n")
        all_new_body.append(child_body)
        
        # 3. Update references in codebase
        # Scan all .py files in extracted/ and root
        search_files = glob.glob('extracted/**/*.py', recursive=True) + glob.glob('*.py')
        
        for file_path in search_files:
            if os.path.abspath(file_path) == os.path.abspath(child_path):
                continue
            if os.path.abspath(file_path) == os.path.abspath(parent_path):
                continue
                
            update_imports_in_file(file_path, child_module, parent_module, child_symbols)
            
    # 4. Write Parent
    # Sort imports (simple sort)
    sorted_imports = sorted(list(all_new_imports))
    
    # Filter out imports of the child modules themselves (if any)
    # and imports of the parent module itself
    final_imports = []
    for imp in sorted_imports:
        # Check if this import line imports any of the child modules
        # This is a bit rough, but prevents "from parent import child" if we merged child into parent
        skip = False
        for cp in child_paths:
            cm = get_module_path(cp)
            if cm in imp:
                skip = True
                break
        if parent_module in imp:
            skip = True
            
        if not skip:
            final_imports.append(imp)
            
    new_content = "import os\nimport sys\n\n"
    # Ensure sys.path setup is present
    if "sys.path.append" not in parent_code:
         new_content += "sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n\n"
         
    new_content += "\n".join(final_imports) + "\n\n"
    if parent_body.strip():
        new_content += parent_body + "\n"
    new_content += "\n".join(all_new_body)
    
    with open(parent_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Successfully merged into {parent_path}")
    
    # 5. Delete Children
    for child_path in child_paths:
        os.remove(child_path)
        print(f"Deleted {child_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python smart_merger.py <parent_path> <child_path1> [child_path2 ...]")
        sys.exit(1)
        
    parent = sys.argv[1]
    children = sys.argv[2:]
    
    merge_modules(parent, children)
