#!/usr/bin/env python3
"""
Script to add noindex robots meta tag to Markdown and reStructuredText files.
This prevents Google and other search engines from indexing the content.

Supports exception lists to exclude specific files from being updated.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Set
import fnmatch


def update_rst_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Update RST file to include noindex meta tag.
    Replaces the entire meta directive with a clean noindex-only version.
    
    Args:
        file_path: Path to the RST file
        dry_run: If True, don't write changes, just report what would be done
        
    Returns:
        Tuple of (was_modified, message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Check if noindex is already present
        if re.search(r':robots:\s+noindex', content):
            return False, f"Already has noindex: {file_path}"
        
        # Pattern to match existing meta directive
        # This pattern correctly handles multi-line field values:
        # - Starts with ".. meta::"
        # - Followed by any number of lines that start with whitespace
        # - Stops when hitting a line that doesn't start with whitespace (or end of string)
        meta_pattern = r'^\.\.\s+meta::[^\n]*\n((?:[ \t]+[^\n]*\n)*)'
        meta_match = re.search(meta_pattern, content, re.MULTILINE)
        
        # The new meta directive (clean, noindex only)
        new_meta = '.. meta::\n  :robots: noindex\n'
        
        if meta_match:
            # Replace existing meta directive with new one
            content = content[:meta_match.start()] + new_meta + content[meta_match.end():]
        else:
            # No meta directive exists, add one at the beginning
            # Try to add after the title if present (line followed by === or ---)
            title_pattern = r'^(.+)\n([=\-~]+)\n'
            title_match = re.match(title_pattern, content, re.MULTILINE)
            
            if title_match:
                # Insert after title
                insert_pos = title_match.end()
                content = content[:insert_pos] + '\n' + new_meta + '\n' + content[insert_pos:]
            else:
                # Insert at the very beginning
                content = new_meta + '\n' + content
        
        if content != original_content:
            if not dry_run:
                file_path.write_text(content, encoding='utf-8')
            return True, f"Updated: {file_path}"
        else:
            return False, f"No changes needed: {file_path}"
            
    except Exception as e:
        return False, f"Error processing {file_path}: {e}"


def update_md_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Update Markdown file to include noindex meta tag in YAML front matter.
    Replaces the entire frontmatter with a clean noindex-only version.
    
    Args:
        file_path: Path to the Markdown file
        dry_run: If True, don't write changes, just report what would be done
        
    Returns:
        Tuple of (was_modified, message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Check if noindex is already present
        if re.search(r'"robots":\s*"noindex"', content):
            return False, f"Already has noindex: {file_path}"
        
        # The new frontmatter (clean, noindex only)
        new_frontmatter = '''---
myst:
  html_meta:
    "robots": "noindex"
---

'''
        
        # Pattern to match any YAML front matter
        frontmatter_pattern = r'^---\s*\n.*?^---\s*\n'
        match = re.search(frontmatter_pattern, content, re.MULTILINE | re.DOTALL)
        
        if match:
            # Replace existing frontmatter with new one
            content = content[:match.start()] + new_frontmatter + content[match.end():]
        else:
            # No frontmatter exists, add new one at the beginning
            content = new_frontmatter + content
        
        if content != original_content:
            if not dry_run:
                file_path.write_text(content, encoding='utf-8')
            return True, f"Updated: {file_path}"
        else:
            return False, f"No changes needed: {file_path}"
            
    except Exception as e:
        return False, f"Error processing {file_path}: {e}"


def find_files(directory: Path, extensions: List[str], recursive: bool = True) -> List[Path]:
    """
    Find all files with specified extensions in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.md', '.rst'])
        recursive: If True, search recursively
        
    Returns:
        List of Path objects
    """
    files = []
    pattern = '**/*' if recursive else '*'
    
    for ext in extensions:
        files.extend(directory.glob(f'{pattern}{ext}'))
    
    return sorted(files)


def load_exception_list(exception_file: Path, base_path: Path) -> Set[Path]:
    """
    Load exception list from a file.
    
    The exception file should contain one pattern per line. Patterns can be:
    - Exact file paths (relative or absolute)
    - Glob patterns (e.g., docs/api/*.rst)
    - File names (e.g., index.md - will match any file with this name)
    
    Lines starting with # are treated as comments.
    Empty lines are ignored.
    
    Args:
        exception_file: Path to the exception list file
        base_path: Base path for resolving relative paths
        
    Returns:
        Set of Path objects to exclude
    """
    if not exception_file.exists():
        print(f"Warning: Exception file not found: {exception_file}")
        return set()
    
    exceptions = set()
    
    try:
        content = exception_file.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Store the pattern as-is for matching
            exceptions.add(line)
            
    except Exception as e:
        print(f"Error reading exception file {exception_file}: {e}")
    
    return exceptions


def is_file_excepted(file_path: Path, exception_patterns: Set[str], base_path: Path) -> bool:
    """
    Check if a file matches any exception pattern.
    
    Args:
        file_path: Path to check
        exception_patterns: Set of exception patterns
        base_path: Base path for resolving relative paths
        
    Returns:
        True if file should be excluded, False otherwise
    """
    if not exception_patterns:
        return False
    
    # Get both absolute and relative (to base_path) versions
    abs_path = file_path.resolve()
    
    try:
        rel_path = file_path.relative_to(base_path)
        rel_path_str = str(rel_path)
    except ValueError:
        # file_path is not relative to base_path
        rel_path_str = str(file_path)
    
    file_name = file_path.name
    
    for pattern in exception_patterns:
        # Check exact match against relative path
        if rel_path_str == pattern:
            return True
        
        # Check exact match against absolute path
        if str(abs_path) == pattern:
            return True
        
        # Check exact match against file name
        if file_name == pattern:
            return True
        
        # Check glob pattern against relative path
        if fnmatch.fnmatch(rel_path_str, pattern):
            return True
        
        # Check glob pattern against absolute path
        if fnmatch.fnmatch(str(abs_path), pattern):
            return True
        
        # Check glob pattern with forward slashes (cross-platform)
        rel_path_posix = rel_path_str.replace('\\', '/')
        pattern_posix = pattern.replace('\\', '/')
        if fnmatch.fnmatch(rel_path_posix, pattern_posix):
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Add noindex meta tag to Markdown and reStructuredText files to prevent search engine indexing.'
    )
    parser.add_argument(
        'path',
        type=Path,
        help='Directory containing files to update, or path to a single file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search recursively in subdirectories'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.md', '.rst'],
        help='File extensions to process (default: .md .rst)'
    )
    parser.add_argument(
        '--exceptions',
        type=Path,
        help='Path to file containing exception patterns (one per line)'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help='Patterns to exclude from processing (can be used multiple times)'
    )
    
    args = parser.parse_args()
    
    # Validate path
    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        return 1
    
    # Determine base path for relative path resolution
    if args.path.is_file():
        base_path = args.path.parent
    else:
        base_path = args.path
    
    # Load exception patterns
    exception_patterns = set(args.exclude)
    
    if args.exceptions:
        file_exceptions = load_exception_list(args.exceptions, base_path)
        exception_patterns.update(file_exceptions)
    
    if exception_patterns:
        print(f"Loaded {len(exception_patterns)} exception pattern(s)")
    
    # Get list of files to process
    if args.path.is_file():
        files = [args.path]
    else:
        files = find_files(args.path, args.extensions, recursive=not args.no_recursive)
    
    if not files:
        print(f"No files found with extensions {args.extensions} in {args.path}")
        return 0
    
    print(f"Found {len(files)} file(s) to process")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    else:
        print()
    
    # Process files
    modified_count = 0
    skipped_count = 0
    excepted_count = 0
    error_count = 0
    
    for file_path in files:
        # Check if file is in exception list
        if is_file_excepted(file_path, exception_patterns, base_path):
            print(f"Excepted (in exception list): {file_path}")
            excepted_count += 1
            continue
        
        if file_path.suffix == '.rst':
            was_modified, message = update_rst_file(file_path, args.dry_run)
        elif file_path.suffix == '.md':
            was_modified, message = update_md_file(file_path, args.dry_run)
        else:
            continue
        
        print(message)
        
        if was_modified:
            modified_count += 1
        elif 'Error' in message:
            error_count += 1
        else:
            skipped_count += 1
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Modified:  {modified_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Excepted:  {excepted_count}")
    print(f"  Errors:    {error_count}")
    print(f"{'=' * 60}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
