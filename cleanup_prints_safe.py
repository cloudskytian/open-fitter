"""
安全なprint文削除スクリプト
- 削除後に空ブロックが残らないかチェック
- 構文エラーを事前検出
"""
import ast
import re
import os
from pathlib import Path
from typing import List, Tuple, Set

# 保持するパターン（正規表現）
KEEP_PATTERNS = [
    r'print\s*\(\s*f?"Status:',           # Status メッセージ
    r'print\s*\(\s*f?"Progress:',         # Progress メッセージ  
    r'print\s*\(\s*f?"Error:',            # Error メッセージ
    r'print\s*\(\s*f?"Warning:',          # Warning メッセージ
    r'print\s*\(\s*f?"警告:',             # 警告メッセージ
    r'print\s*\(\s*f?"エラー:',           # エラーメッセージ
    r'print\s*\(\s*"={10,}',              # セパレータ（エラー表示用）
    r'print\s*\(\s*f?"\\n={10,}',         # セパレータ
    r'print\s*\(traceback',               # トレースバック
    r'print\s*\(\s*f?"============',      # エラー詳細
]

# 削除するパターン（これらを含むprint文を削除）
DELETE_PATTERNS = [
    r'print\s*\(\s*f?"Found ',
    r'print\s*\(\s*f?"Processing ',
    r'print\s*\(\s*f?"Merged ',
    r'print\s*\(\s*f?"Removed ',
    r'print\s*\(\s*f?"Created ',
    r'print\s*\(\s*f?"Component ',
    r'print\s*\(\s*f?"Skipping ',
    r'print\s*\(\s*f?"Using ',
    r'print\s*\(\s*f?"Set ',
    r'print\s*\(\s*f?"Updated ',
    r'print\s*\(\s*f?"Renamed ',
    r'print\s*\(\s*f?"Restored ',
    r'print\s*\(\s*f?"Saved ',
    r'print\s*\(\s*f?"Target shape key',
    r'print\s*\(\s*f?"Shape key',
    r'print\s*\(\s*f?"Initialize:',
    r'print\s*\(\s*f?"Generated ',
    r'print\s*\(\s*f?"Normalized ',
    r'print\s*\(\s*f?"Truncated ',
    r'print\s*\(\s*f?"Added ',
    r'print\s*\(\s*f?"Cached ',
    r'print\s*\(\s*f?"Executing ',
    r'print\s*\(\s*f?"Looking ',
    r'print\s*\(\s*f?"Iteration ',
    r'print\s*\(\s*f?"Total: ',
    r'print\s*\(\s*f?"Pose ',
    r'print\s*\(\s*f?"Bone ',
    r'print\s*\(\s*f?"Weight ',
    r'print\s*\(\s*f?"Subdivision',
    r'print\s*\(\s*f?"Preserving ',
    r'print\s*\(\s*f?"RBF',
    r'print\s*\(\s*f?"Most common',
    r'print\s*\(\s*f?"No ',
    r'print\s*\(\s*f?"label:',
    r'print\s*\(\s*f?"source_',
    r'print\s*\(\s*f?"temp_',
    r'print\s*\(\s*f?"mask_weights',
    r'print\s*\(\s*f?"chosen_parent',
    r'print\s*\(\s*f?"- ',           # リスト項目
    r'print\s*\(\s*f?"  ',           # インデント付きメッセージ
    r'print\s*\(\s*f?"\'',           # クォートで始まる
    r'print\s*\(\s*f?"ステップ',
    r'print\s*\(\s*f?"細分化',
    r'print\s*\(\s*f?"条件',
    r'print\s*\(\s*f?"全体処理',
    r'print\s*\(\s*f?"エッジを',
    r'print\s*\(\s*f?"オブジェクト',
    r'print\s*\(\s*"cycle1 "',
    r'print\s*\(\s*"cycle2 ',
    r'print\s*\(\s*"BVH',
    r'print\s*\(\s*"Normalizing',
    r'print\s*\(\s*"Finished ',
    r'print\s*\(\s*"Hips position',
    r'print\s*\(\s*"Max distance',
    r'print\s*\(\s*"Maximum iter',
    r'print\s*\(\s*"No intersect',
    r'print\s*\(\s*"No non-separated',
    r'print\s*\(\s*"No step change',
    r'print\s*\(\s*"set_humanoid',
    r'print\s*\(\s*"Subdivision skip',
    r'print\s*\(\s*"subHumanoid',
    r'print\s*\(\s*"Template',
    r'print\s*\(\s*"Aポーズ',
    r'print\s*\(\s*"元の',
    r'print\s*\(\s*"指定され',
    r'print\s*\(\s*"無効な',
    r'print\s*\(\s*base_bone',
    r'print\s*\(\s*sys\.argv',
    r'print\s*\(\s*\)$',             # 空のprint()
    r'print\s*\(\s*"エッジが見つかりません',
]


def should_keep_print(line: str) -> bool:
    """このprint文を保持すべきか判定"""
    for pattern in KEEP_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def should_delete_print(line: str) -> bool:
    """このprint文を削除すべきか判定"""
    for pattern in DELETE_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def find_empty_blocks(content: str) -> List[Tuple[int, str]]:
    """空のブロック（if/else/for/except/try後にボディがない）を検出"""
    lines = content.split('\n')
    issues = []
    
    block_pattern = re.compile(r'^(\s*)(if\s+.+:|elif\s+.+:|else:|for\s+.+:|while\s+.+:|try:|except.*:|finally:|with\s+.+:)\s*$')
    
    for i, line in enumerate(lines):
        match = block_pattern.match(line)
        if match:
            indent = match.group(1)
            expected_indent = indent + '    '
            
            # 次の非空行を探す
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            
            if j < len(lines):
                next_line = lines[j]
                # 次の行が期待するインデントより浅い場合、空ブロック
                if next_line.strip() and not next_line.startswith(expected_indent):
                    # コメント行でもない
                    if not next_line.strip().startswith('#'):
                        issues.append((i + 1, line.strip()))
    
    return issues


def check_syntax(filepath: str) -> bool:
    """ファイルの構文をチェック"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"  Syntax error at line {e.lineno}: {e.msg}")
        return False


def process_file(filepath: str, dry_run: bool = False) -> Tuple[int, int, List[Tuple[int, str]]]:
    """
    ファイルを処理してprint文を削除
    
    Returns:
        (削除数, 保持数, 空ブロック問題リスト)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    deleted_count = 0
    kept_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # print文かチェック
        if stripped.startswith('print(') or stripped.startswith('print ('):
            if should_keep_print(stripped):
                new_lines.append(line)
                kept_count += 1
            elif should_delete_print(stripped):
                # 複数行にまたがるprint文の処理
                full_line = line
                while full_line.count('(') > full_line.count(')') and i + 1 < len(lines):
                    i += 1
                    full_line += lines[i]
                deleted_count += 1
            else:
                # どちらにも該当しない場合は保持
                new_lines.append(line)
                kept_count += 1
        else:
            new_lines.append(line)
        i += 1
    
    # 空ブロックをチェック
    new_content = ''.join(new_lines)
    empty_blocks = find_empty_blocks(new_content)
    
    if not dry_run and deleted_count > 0 and not empty_blocks:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return deleted_count, kept_count, empty_blocks


def fix_empty_block(lines: List[str], line_num: int) -> List[str]:
    """空ブロックを修正（passを挿入または不要なブロックを削除）"""
    idx = line_num - 1
    line = lines[idx]
    indent_match = re.match(r'^(\s*)', line)
    indent = indent_match.group(1) if indent_match else ''
    
    stripped = line.strip()
    
    # else:/except:/finally: は削除
    if stripped in ['else:', 'finally:'] or stripped.startswith('except'):
        lines[idx] = ''
        return lines
    
    # その他はpassを挿入
    pass_line = indent + '    pass  # Auto-inserted\n'
    lines.insert(idx + 1, pass_line)
    return lines


def main():
    extracted_dir = Path(r'c:\Users\tallcat\Documents\OpenFitter-Core\extracted')
    
    total_deleted = 0
    total_kept = 0
    files_with_issues = []
    
    print("=== Phase 1: Analyzing print statements ===\n")
    
    py_files = list(extracted_dir.rglob('*.py'))
    
    for filepath in py_files:
        rel_path = filepath.relative_to(extracted_dir)
        deleted, kept, issues = process_file(str(filepath), dry_run=True)
        
        if deleted > 0:
            print(f"{rel_path}: {deleted} to delete, {kept} to keep")
            if issues:
                files_with_issues.append((filepath, issues))
                print(f"  WARNING: {len(issues)} empty blocks would be created:")
                for line_num, line_content in issues[:3]:
                    print(f"    Line {line_num}: {line_content[:50]}...")
        
        total_deleted += deleted
        total_kept += kept
    
    print(f"\n=== Summary ===")
    print(f"Total to delete: {total_deleted}")
    print(f"Total to keep: {total_kept}")
    print(f"Files with potential empty block issues: {len(files_with_issues)}")
    
    if files_with_issues:
        print("\n=== Phase 2: Fixing empty blocks and deleting prints ===\n")
        
        for filepath, issues in files_with_issues:
            rel_path = filepath.relative_to(extracted_dir)
            print(f"Fixing {rel_path}...")
            
            # まずprint削除
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                
                if stripped.startswith('print(') or stripped.startswith('print ('):
                    if should_keep_print(stripped):
                        new_lines.append(line)
                    elif should_delete_print(stripped):
                        full_line = line
                        while full_line.count('(') > full_line.count(')') and i + 1 < len(lines):
                            i += 1
                            full_line += lines[i]
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
                i += 1
            
            # 空ブロックを検出して修正
            content = ''.join(new_lines)
            empty_blocks = find_empty_blocks(content)
            
            while empty_blocks:
                # 後ろから修正（行番号がずれないように）
                empty_blocks.sort(key=lambda x: x[0], reverse=True)
                for line_num, line_content in empty_blocks:
                    new_lines = fix_empty_block(new_lines, line_num)
                
                content = ''.join(new_lines)
                empty_blocks = find_empty_blocks(content)
            
            # 書き込み
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # 構文チェック
            if not check_syntax(str(filepath)):
                print(f"  ERROR: Syntax error after processing {rel_path}")
    
    # 問題なかったファイルを処理
    print("\n=== Phase 3: Processing remaining files ===\n")
    
    for filepath in py_files:
        if filepath not in [f for f, _ in files_with_issues]:
            deleted, kept, issues = process_file(str(filepath), dry_run=True)
            if deleted > 0 and not issues:
                process_file(str(filepath), dry_run=False)
                rel_path = filepath.relative_to(extracted_dir)
                print(f"Processed {rel_path}: {deleted} deleted")
    
    # 最終構文チェック
    print("\n=== Phase 4: Final syntax check ===\n")
    
    errors = []
    for filepath in py_files:
        if not check_syntax(str(filepath)):
            errors.append(filepath)
    
    if errors:
        print(f"ERROR: {len(errors)} files have syntax errors!")
        for f in errors:
            print(f"  {f}")
    else:
        print("All files passed syntax check!")


if __name__ == '__main__':
    main()
