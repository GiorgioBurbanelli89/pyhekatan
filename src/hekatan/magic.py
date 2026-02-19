#  Hekatan cell magic for Jupyter — %%hekatan
#
#  Works like handcalcs %%render but generates HTML instead of LaTeX.
#  Write normal Python and get: formula → substitution → result
#
#  Usage:
#      %%hekatan
#      a = 3
#      b = 4
#      c = math.sqrt(a**2 + b**2)
#
#  Produces (HTML rendered):
#      a = 3
#      b = 4
#      c = √(a² + b²) = √(3² + 4²) = 5.0

import ast
import re
import math
from typing import Optional

try:
    from IPython.core.magic import register_cell_magic
    from IPython import get_ipython
    from IPython.display import HTML, display
    # Only proceed if we're actually inside an IPython session
    if get_ipython() is None:
        raise ImportError("Not inside an active IPython session")
except ImportError:
    raise ImportError("hekatan.magic requires a Jupyter/IPython environment.")


def _py_to_hekatan(expr_str: str) -> str:
    """Convert Python math syntax to Hekatan syntax.

    Python:   math.sqrt(a**2 + b**2)
    Hekatan:  sqrt(a^2 + b^2)
    """
    s = expr_str
    # math.func → func
    s = re.sub(r'math\.', '', s)
    # numpy shortcuts
    s = re.sub(r'np\.', '', s)
    # ** → ^
    s = s.replace('**', '^')
    return s


def _parse_cell(cell_source: str, user_ns: dict):
    """Parse a cell of Python code and generate Hekatan display calls.

    For each line:
      - Simple assignment (a = 3) → var("a", 3)
      - Computed assignment (c = math.sqrt(a**2 + b**2))
          → formula + substitution + result (3 steps)
      - Comments (#) → skip
      - Import statements → skip
      - Blank lines → skip

    Returns list of HTML fragments.
    """
    from hekatan.calc_engine import (
        _expr_to_html, _format_number, tokenize, Parser,
        AssignNode, _evaluate, _SYMBOL_TABLE, _BUILTINS, _FUNCTIONS,
    )
    from hekatan.display import _format_subscript, _format_unit, _get_css

    html_parts = []

    # Inject CSS once
    css = _get_css()
    if css:
        html_parts.append(css)

    lines = cell_source.strip().split('\n')

    for line in lines:
        stripped = line.strip()

        # Skip empty, comments, imports
        if not stripped or stripped.startswith('#') or stripped.startswith('import ') \
                or stripped.startswith('from '):
            continue

        # Must be an assignment: name = expression
        if '=' not in stripped or stripped.startswith('if ') \
                or stripped.startswith('for ') or '==' in stripped:
            continue

        # Parse: left = right
        eq_idx = stripped.index('=')
        # Skip !=, <=, >=
        if eq_idx > 0 and stripped[eq_idx - 1] in ('!', '<', '>'):
            continue

        lhs = stripped[:eq_idx].strip()
        rhs_py = stripped[eq_idx + 1:].strip()

        # Skip augmented assignments (+=, -=, etc.)
        if lhs.endswith(('+', '-', '*', '/')):
            continue

        # Get the computed value from user namespace
        value = user_ns.get(lhs)
        if value is None:
            continue

        # Convert Python syntax to Hekatan syntax
        rhs_hek = _py_to_hekatan(rhs_py)

        # Format name
        name_html = f'<var>{_format_subscript(lhs)}</var>'

        # Check if it's a simple literal (a = 3, b = 4)
        try:
            literal = ast.literal_eval(rhs_py)
            # Simple assignment: just show name = value
            if isinstance(literal, (int, float)):
                val_str = _format_number(float(literal))
                html_parts.append(
                    f'<div class="eq">{name_html} = <b>{val_str}</b></div>'
                )
                # Also register in SYMBOL_TABLE for substitution
                _SYMBOL_TABLE[lhs] = float(literal)
                continue
        except (ValueError, SyntaxError):
            pass

        # Computed assignment: generate 3 steps
        # Step 1: formula (symbolic) — c = √(a² + b²)
        try:
            rhs_html = _expr_to_html(rhs_hek)
        except Exception:
            rhs_html = rhs_hek

        # Step 2: substitution — replace variables with values
        substituted = rhs_hek
        # Collect variables to substitute (sorted by length desc to avoid partial match)
        subs = []
        for var_name, var_val in _SYMBOL_TABLE.items():
            if var_name in ('pi', 'e'):
                continue
            if var_name in _BUILTINS or var_name in _FUNCTIONS:
                continue
            subs.append((var_name, var_val))
        subs.sort(key=lambda x: len(x[0]), reverse=True)

        for var_name, var_val in subs:
            substituted = re.sub(
                r'\b' + re.escape(var_name) + r'\b',
                _format_number(var_val),
                substituted
            )

        try:
            sub_html = _expr_to_html(substituted)
        except Exception:
            sub_html = substituted

        # Step 3: result
        if isinstance(value, float):
            val_str = _format_number(value)
        elif isinstance(value, int):
            val_str = str(value)
        else:
            val_str = str(value)

        # Build the line: name = formula = substitution = result
        # Only show substitution if different from formula and result
        if substituted != rhs_hek and substituted != val_str:
            html_parts.append(
                f'<div class="eq">{name_html} = {rhs_html}'
                f' = {sub_html}'
                f' = <b>{val_str}</b></div>'
            )
        elif rhs_hek != val_str:
            html_parts.append(
                f'<div class="eq">{name_html} = {rhs_html}'
                f' = <b>{val_str}</b></div>'
            )
        else:
            html_parts.append(
                f'<div class="eq">{name_html} = <b>{val_str}</b></div>'
            )

        # Register value for next substitutions
        if isinstance(value, (int, float)):
            _SYMBOL_TABLE[lhs] = float(value)

    return html_parts


@register_cell_magic
def hekatan(line, cell):
    """%%hekatan cell magic — write Python, get formatted HTML.

    Usage:
        %%hekatan
        a = 3
        b = 4
        c = math.sqrt(a**2 + b**2)

    Options (on the %%hekatan line):
        params  — show as parameter list (name = value only)
        short   — show formula = result (no substitution)
    """
    ip = get_ipython()

    # Execute the cell code in user namespace
    result = ip.run_cell(cell, silent=True)
    if not result.success:
        # If execution failed, show the error
        result.raise_error()
        return

    # Get user namespace with computed results
    user_ns = ip.user_ns

    # Parse line args
    show_params_only = 'params' in line.lower()
    show_short = 'short' in line.lower()

    if show_params_only:
        # Simple: just show name = value for each assignment
        html_parts = _parse_params(cell, user_ns)
    else:
        # Full: formula → substitution → result
        html_parts = _parse_cell(cell, user_ns)

    if html_parts:
        full_html = f'<div class="hekatan-doc">{"".join(html_parts)}</div>'
        display(HTML(full_html))


def _parse_params(cell_source: str, user_ns: dict):
    """Simple parameter display: name = value for each line."""
    from hekatan.calc_engine import _format_number, _SYMBOL_TABLE
    from hekatan.display import _format_subscript, _get_css

    html_parts = []
    css = _get_css()
    if css:
        html_parts.append(css)

    for line in cell_source.strip().split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('import '):
            continue
        if '=' not in stripped or '==' in stripped:
            continue

        eq_idx = stripped.index('=')
        if eq_idx > 0 and stripped[eq_idx - 1] in ('!', '<', '>'):
            continue

        lhs = stripped[:eq_idx].strip()
        if lhs.endswith(('+', '-', '*', '/')):
            continue

        value = user_ns.get(lhs)
        if value is None:
            continue

        name_html = f'<var>{_format_subscript(lhs)}</var>'
        if isinstance(value, (int, float)):
            val_str = _format_number(float(value))
            _SYMBOL_TABLE[lhs] = float(value)
        else:
            val_str = str(value)

        html_parts.append(
            f'<div class="eq">{name_html} = <b>{val_str}</b></div>'
        )

    return html_parts
