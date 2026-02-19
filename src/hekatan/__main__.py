"""
Hekatan preprocessor — run .hkpy or .py files with natural math syntax.

Usage:
    python -m hekatan archivo.hkpy
    python -m hekatan archivo.py

Each line is treated as calc("...") UNLESS it looks like Python code:
  - Lines starting with known hekatan functions: title(), text(), show(), etc.
  - Lines starting with Python keywords: if, for, import, from, def, class, etc.
  - Lines starting with # (comments)
  - Lines starting with ' or " (strings)
  - Empty lines
  - Lines containing ( at the start of an identifier (function calls)
"""

import sys
import os
import re


# Python keywords that should NOT be processed as calc()
_PYTHON_KEYWORDS = {
    'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
    'import', 'from', 'try', 'except', 'finally', 'with', 'as',
    'raise', 'pass', 'break', 'continue', 'yield', 'assert',
    'del', 'global', 'nonlocal', 'lambda', 'and', 'or', 'not',
    'in', 'is', 'True', 'False', 'None', 'print',
}

# Hekatan display functions that are Python calls (not math)
_HEKATAN_FUNCS = {
    'title', 'heading', 'text', 'markdown', 'note', 'code',
    'hr', 'show', 'clear', 'set_mode',
    'matrix', 'table', 'eq', 'var', 'fraction', 'formula',
    'integral', 'derivative', 'partial', 'summation', 'product_op',
    'double_integral', 'limit_op', 'eq_num', 'eq_block',
    'columns', 'column', 'end_columns',
    'paper', 'header', 'footer', 'author', 'abstract_block',
    'image', 'figure', 'check', 'page_break', 'html_raw',
    'calc', 'calc_clear', 'run',
    # Common Python builtins people might use
    'sys', 'os', 'math', 'numpy', 'np', 'plt',
}

# Regex: line starts with identifier followed by ( — it's a function call
_FUNC_CALL_RE = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')

# Regex: line is a Python-style assignment with complex RHS (lists, dicts, strings, etc.)
_PYTHON_ASSIGN_RE = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*[\[\{"\']')


def _is_python_line(line: str) -> bool:
    """Check if a line should be treated as Python (not math)."""
    stripped = line.strip()

    # Empty line or comment
    if not stripped or stripped.startswith('#'):
        return True

    # String literal
    if stripped.startswith(("'", '"', "b'", 'b"', "f'", 'f"')):
        return True

    # Decorator
    if stripped.startswith('@'):
        return True

    # Python keyword at start
    first_word = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
    if first_word and first_word.group(1) in _PYTHON_KEYWORDS:
        return True

    # Known hekatan function call: title("..."), show(), etc.
    m = _FUNC_CALL_RE.match(stripped)
    if m and m.group(1) in _HEKATAN_FUNCS:
        return True

    # Python-style assignment to list/dict/string: K = [[1,2],[3,4]]
    if _PYTHON_ASSIGN_RE.match(stripped):
        return True

    # Multi-line continuation
    if stripped.endswith('\\'):
        return True

    # Indented lines (inside if/for/def blocks)
    if line and line[0] in (' ', '\t'):
        return True

    return False


def _convert_line(line: str) -> str:
    """Convert a math line to calc() call, preserving indentation."""
    stripped = line.strip()
    if not stripped:
        return line

    # Escape quotes in the expression
    escaped = stripped.replace('\\', '\\\\').replace('"', '\\"')
    return f'calc("{escaped}")\n'


def preprocess(source: str) -> str:
    """Convert a .hkpy source file to executable Python.

    - Math lines become calc("...") calls
    - Python lines pass through unchanged
    - Adds automatic imports at the top
    """
    lines = source.split('\n')
    output = []

    # Add imports
    output.append('from hekatan import *\n')
    output.append('clear()\n')
    output.append('')

    in_block = False  # Track if we're inside a Python block (def, if, for...)

    for line in lines:
        stripped = line.strip()

        # Track Python block context
        if stripped and not stripped.startswith('#'):
            # Entering a block
            if re.match(r'^(if|for|while|def|class|try|with|elif|else|except|finally)\b', stripped):
                in_block = True
            # Exiting block: non-indented, non-empty line
            elif line and line[0] not in (' ', '\t') and not stripped.startswith('#'):
                in_block = False

        # Inside a Python block — pass through
        if in_block and line and line[0] in (' ', '\t'):
            output.append(line)
            continue

        # Check if this is a Python line
        if _is_python_line(line):
            output.append(line)
        else:
            output.append(_convert_line(line))

    return '\n'.join(output)


def main():
    """Entry point: python -m hekatan file.hkpy"""
    if len(sys.argv) < 2:
        print("Uso: python -m hekatan <archivo.hkpy>")
        print()
        print("Ejemplo:")
        print("  python -m hekatan calculo.hkpy")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Error: archivo no encontrado: {filepath}")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    # Preprocess
    python_code = preprocess(source)

    # Execute in a clean namespace
    namespace = {'__file__': os.path.abspath(filepath), '__name__': '__main__'}

    # Add src to path for development
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if os.path.isdir(os.path.join(src_dir, 'src')):
        sys.path.insert(0, os.path.join(src_dir, 'src'))

    try:
        exec(compile(python_code, filepath, 'exec'), namespace)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
