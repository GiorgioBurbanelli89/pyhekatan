"""
Core display functions for Hekatan.

Each function works in 3 modes:
  - HEKATAN mode: emits @@DSL commands to stdout (C# parses → HTML)
  - STANDALONE mode: accumulates HTML, show() opens in browser
  - CONSOLE mode: ASCII formatted output

v0.7.0 - DSL protocol: hekatan mode emits @@ commands, no HTML tags
"""

import os
import sys
import tempfile
import webbrowser
from typing import Any, List, Optional, Union

# ============================================================
# Mode detection
# ============================================================

_MODE = None  # 'hekatan', 'standalone', 'console'
_BUFFER = []  # Accumulated elements for standalone mode


def _in_jupyter() -> bool:
    """Check if running inside Jupyter (notebook or lab)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ('ZMQInteractiveShell', 'Shell')
    except (ImportError, NameError):
        return False


def _detect_mode() -> str:
    """Auto-detect rendering mode.

    Jupyter is treated as 'standalone' — _emit() handles the display difference.
    """
    if os.environ.get("HEKATAN_RENDER") == "1":
        return "hekatan"
    return "standalone"


def _get_mode() -> str:
    global _MODE
    if _MODE is None:
        _MODE = _detect_mode()
    return _MODE


_JUPYTER_CSS_INJECTED = False


def _get_css() -> str:
    """Return CSS <style> tag if not yet injected in Jupyter. Empty string if already done."""
    global _JUPYTER_CSS_INJECTED
    if not _JUPYTER_CSS_INJECTED:
        _JUPYTER_CSS_INJECTED = True
        return f'<style>{_CSS}</style>'
    return ''


def _emit(html: str):
    """Emit an HTML element.

    In Jupyter: display(HTML(...)) directly per element.
    In standalone: buffer for later show().
    """
    if _in_jupyter():
        from IPython.display import display, HTML as _HTML
        css = _get_css()
        if css:
            display(_HTML(css))
        display(_HTML(f'<div class="hekatan-doc">{html}</div>'))
    else:
        _BUFFER.append(html)


def _dsl(cmd: str):
    """Emit a DSL command to stdout for Hekatan Calc."""
    print(cmd)


def set_mode(mode: str):
    """Force a specific mode: 'hekatan', 'standalone', or 'console'.

    Note: Jupyter is auto-detected inside _emit(). In Jupyter, use 'standalone'.
    """
    global _MODE
    if mode == "jupyter":
        mode = "standalone"  # Jupyter is handled by _emit() auto-detection
    if mode not in ("hekatan", "standalone", "console"):
        raise ValueError(f"Invalid mode: {mode}. Use 'hekatan', 'standalone', or 'console'.")
    _MODE = mode


def clear():
    """Clear the accumulated buffer and calc symbol table."""
    _BUFFER.clear()
    try:
        from hekatan.calc_engine import calc_clear as _cc
        _cc()
    except ImportError:
        pass


# ============================================================
# Helper: escape pipe for DSL fields
# ============================================================

def _esc(val: str) -> str:
    """Escape pipe characters in DSL field values."""
    return str(val).replace("|", "\\|")


# ============================================================
# Matrix display
# ============================================================

def matrix(data: List[List[Any]], name: Optional[str] = None):
    """
    Display a matrix with formatted brackets.

    Args:
        data: 2D list [[row1], [row2], ...]
        name: Optional name (e.g. "A", "K")

    Example:
        matrix([[1, 2], [3, 4]], "A")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@matrix name|row1_c1,row1_c2;row2_c1,row2_c2
        rows_str = ";".join(",".join(_esc(str(x)) for x in row) for row in data)
        name_str = _esc(name) if name else ""
        _dsl(f"@@matrix {name_str}|{rows_str}")

    elif mode == "standalone":
        html = _matrix_to_html(data, name)
        _emit(html)

    else:  # console
        _matrix_to_console(data, name)


def _matrix_to_html(data: List[List[Any]], name: Optional[str] = None) -> str:
    """Generate HTML for a matrix using CSS grid (avoids Jupyter table conflicts)."""
    ncols = max(len(row) for row in data) if data else 1
    rows_html = []
    for row in data:
        cells = "".join(
            f'<span class="mc">{_format_subscript(str(x))}</span>' for x in row
        )
        rows_html.append(f'<span class="mr">{cells}</span>')

    grid = (
        f'<span class="matrix" style="grid-template-columns:repeat({ncols},auto);">'
        f'{"".join(rows_html)}'
        f'</span>'
    )

    if name:
        display_name = _format_subscript(name)
        return f'<div class="eq"><var>{display_name}</var> = {grid}</div>'
    return f'<div class="eq">{grid}</div>'


def _matrix_to_console(data: List[List[Any]], name: Optional[str] = None):
    """Print matrix in ASCII format."""
    if not data:
        print("[]")
        return

    col_widths = []
    for j in range(len(data[0])):
        w = max(len(str(data[i][j])) for i in range(len(data)))
        col_widths.append(w)

    prefix = f"{name} = " if name else ""
    pad = " " * len(prefix)

    for i, row in enumerate(data):
        cells = "  ".join(str(row[j]).rjust(col_widths[j]) for j in range(len(row)))
        line_prefix = prefix if i == len(data) // 2 else pad
        print(f"{line_prefix}| {cells} |")


# ============================================================
# Equation display
# ============================================================

def eq(name: str, value: Any, unit: str = ""):
    """
    Display a formatted equation: name = value [unit]

    Args:
        name: Variable name (e.g. "F", "M_u")
        value: Numeric value
        unit: Optional unit string (e.g. "kN", "mm")

    Example:
        eq("F", 25.5, "kN")
        eq("A_s", 1256.64, "mm^2")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@eq name|value|unit
        _dsl(f"@@eq {_esc(name)}|{_esc(str(value))}|{_esc(unit)}")

    elif mode == "standalone":
        display_name = _format_subscript(name)
        value_html = _format_expr(str(value))
        unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""
        html = f'<div class="eq"><var>{display_name}</var> = <span class="val">{value_html}{unit_html}</span></div>'
        _emit(html)

    else:  # console
        unit_str = f" {unit}" if unit else ""
        print(f"{name} = {value}{unit_str}")


# ============================================================
# Variable display (with description)
# ============================================================

def var(name: str, value: Any, unit: str = "", desc: str = ""):
    """
    Display a variable with optional description.

    Args:
        name: Variable name
        value: Value
        unit: Optional unit
        desc: Optional description

    Example:
        var("b", 300, "mm", "ancho de la viga")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@var name|value|unit|desc
        _dsl(f"@@var {_esc(name)}|{_esc(str(value))}|{_esc(unit)}|{_esc(desc)}")

    elif mode == "standalone":
        display_name = _format_subscript(name)
        value_html = _format_expr(str(value))
        unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""
        desc_html = f' <span class="desc">{desc}</span>' if desc else ""
        html = f'<div class="eq"><var>{display_name}</var> = <span class="val">{value_html}{unit_html}</span>{desc_html}</div>'
        _emit(html)

    else:  # console
        unit_str = f" {unit}" if unit else ""
        desc_str = f"  - {desc}" if desc else ""
        print(f"{name} = {value}{unit_str}{desc_str}")


# ============================================================
# Fraction display
# ============================================================

def fraction(numerator: Any, denominator: Any, name: Optional[str] = None):
    """
    Display a fraction.

    Args:
        numerator: Top value
        denominator: Bottom value
        name: Optional name for the result

    Example:
        fraction("M_u", "phi * b * d^2", "R_n")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@fraction name|numerator|denominator
        name_str = _esc(name) if name else ""
        _dsl(f"@@fraction {name_str}|{_esc(str(numerator))}|{_esc(str(denominator))}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        num_html = _format_subscript(str(numerator))
        den_html = _format_subscript(str(denominator))
        html = (
            f'<div class="eq">{name_html}'
            f'<span class="dvc">'
            f'<span class="dvl">{num_html}</span>'
            f'<span class="dvl">{den_html}</span>'
            f'</span></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        num_str = str(numerator)
        den_str = str(denominator)
        width = max(len(num_str), len(den_str))
        print(f"{prefix}{num_str.center(width)}")
        print(f"{' ' * len(prefix)}{'-' * width}")
        print(f"{' ' * len(prefix)}{den_str.center(width)}")


# ============================================================
# Integral display
# ============================================================

def integral(integrand: str, variable: str = "x",
             lower: Optional[str] = None, upper: Optional[str] = None,
             name: Optional[str] = None):
    """
    Display an integral expression.

    Args:
        integrand: The expression to integrate (e.g. "sigma_x")
        variable: Integration variable (e.g. "z")
        lower: Lower limit (e.g. "-h/2")
        upper: Upper limit (e.g. "h/2")
        name: Optional result name (e.g. "N_x")

    Example:
        integral("sigma_x", "z", "-h/2", "h/2", "N_x")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@integral name|integrand|variable|lower|upper
        name_str = _esc(name) if name else ""
        lower_str = _esc(lower) if lower else ""
        upper_str = _esc(upper) if upper else ""
        _dsl(f"@@integral {name_str}|{_esc(integrand)}|{_esc(variable)}|{lower_str}|{upper_str}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""

        # Use AST renderer for the integrand (converts *, ^, fractions properly)
        try:
            from hekatan.calc_engine import _expr_to_html
            integrand_html = _expr_to_html(integrand)
        except (ImportError, Exception):
            integrand_html = _format_expr(integrand)

        var_html = _format_subscript(variable)

        # Build the integral symbol with limits
        if lower and upper:
            # Format limits using AST renderer too
            try:
                from hekatan.calc_engine import _expr_to_html
                lower_html = _expr_to_html(lower)
                upper_html = _expr_to_html(upper)
            except (ImportError, Exception):
                lower_html = _format_subscript(lower)
                upper_html = _format_subscript(upper)

            integral_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary-sup">{upper_html}</span>'
                f'<span class="nary">&#8747;</span>'
                f'<span class="nary-sub">{lower_html}</span>'
                f'</span>'
            )
        elif lower:
            try:
                from hekatan.calc_engine import _expr_to_html
                lower_html = _expr_to_html(lower)
            except (ImportError, Exception):
                lower_html = _format_subscript(lower)
            integral_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary">&#8747;</span>'
                f'<span class="nary-sub">{lower_html}</span>'
                f'</span>'
            )
        else:
            integral_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary">&#8747;</span>'
                f'</span>'
            )

        # Build: name = ∫ (integrand) dx  — parentheses for compound expressions
        has_additive = any(c in integrand for c in ('+', '-')) and integrand.strip()[0] != '-'
        if has_additive:
            body = f'({integrand_html})'
        else:
            body = integrand_html

        html = (
            f'<div class="eq">{name_html}'
            f'{integral_sym}'
            f'<span>{body}\u2009'
            f'<var>d</var><var>{var_html}</var></span></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            print(f"{prefix}integral({integrand}, {variable}, {lower}, {upper})")
        else:
            print(f"{prefix}integral({integrand}, {variable})")


# ============================================================
# Derivative display
# ============================================================

def derivative(func: str, variable: str = "x", order: int = 1,
               name: Optional[str] = None):
    """
    Display an ordinary derivative: df/dx or d²f/dx².

    Args:
        func: Function name (e.g. "f", "y", "u")
        variable: Differentiation variable (e.g. "x", "t")
        order: Derivative order (1, 2, 3...)
        name: Optional name for the result

    Examples:
        derivative("y", "x")           # dy/dx
        derivative("f", "t", order=2)  # d²f/dt²
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@derivative name|func|variable|order
        name_str = _esc(name) if name else ""
        _dsl(f"@@derivative {name_str}|{_esc(func)}|{_esc(variable)}|{order}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        var_html = _format_subscript(variable)

        # Detect if func is a compound expression (operators, long)
        _is_compound = any(c in func for c in ('+', '-', '*', '/')) or len(func) > 6

        if _is_compound:
            # Compound: d/dx (expr) — operator fraction + argument outside
            try:
                from hekatan.calc_engine import _expr_to_html
                func_display = _expr_to_html(func)
            except (ImportError, Exception):
                func_display = _format_expr(func)

            if order == 1:
                num = 'd'
                den = f'd<var>{var_html}</var>'
            else:
                num = f'd<sup>{order}</sup>'
                den = f'd<var>{var_html}</var><sup>{order}</sup>'

            html = (
                f'<div class="eq">{name_html}'
                f'<span class="dvc">'
                f'<span class="dvl">{num}</span>'
                f'<span class="dvl">{den}</span>'
                f'</span>'
                f'\u2009({func_display})'
                f'</div>'
            )
        else:
            # Simple: df/dx — all inside the fraction
            func_html = _format_subscript(func)
            if order == 1:
                num = f'd<var>{func_html}</var>'
                den = f'd<var>{var_html}</var>'
            else:
                num = f'd<sup>{order}</sup><var>{func_html}</var>'
                den = f'd<var>{var_html}</var><sup>{order}</sup>'

            html = (
                f'<div class="eq">{name_html}'
                f'<span class="dvc">'
                f'<span class="dvl">{num}</span>'
                f'<span class="dvl">{den}</span>'
                f'</span></div>'
            )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if order == 1:
            print(f"{prefix}d{func}/d{variable}")
        else:
            print(f"{prefix}d^{order}{func}/d{variable}^{order}")


# ============================================================
# Partial derivative display
# ============================================================

def partial(func: str, variable: Union[str, List[str]] = "x",
            order: int = 1, name: Optional[str] = None):
    """
    Display a partial derivative: ∂f/∂x or ∂²f/∂x².
    Also supports mixed partials: ∂²f/∂x∂y.

    Args:
        func: Function name (e.g. "u", "phi")
        variable: Variable(s) - str for single, list for mixed
        order: Derivative order (auto-calculated for mixed)
        name: Optional name for the result

    Examples:
        partial("u", "x")              # ∂u/∂x
        partial("u", "t", order=2)     # ∂²u/∂t²
        partial("u", ["x", "y"])       # ∂²u/∂x∂y
    """
    mode = _get_mode()

    # Mixed partial
    if isinstance(variable, list):
        total_order = len(variable)
        vars_str = variable
    else:
        total_order = order
        vars_str = [variable]

    if mode == "hekatan":
        # @@partial name|func|var1,var2,...|order
        name_str = _esc(name) if name else ""
        vars_joined = ",".join(_esc(v) for v in vars_str)
        _dsl(f"@@partial {name_str}|{_esc(func)}|{vars_joined}|{total_order}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        pd = "\u2202"  # ∂

        # Detect if func is a compound expression
        _is_compound = any(c in func for c in ('+', '-', '*', '/')) or len(func) > 6

        if _is_compound:
            # Compound: ∂/∂x (expr) — operator fraction + argument outside
            try:
                from hekatan.calc_engine import _expr_to_html
                func_display = _expr_to_html(func)
            except (ImportError, Exception):
                func_display = _format_expr(func)

            if total_order == 1:
                num = pd
                den = f'{pd}<var>{_format_subscript(vars_str[0])}</var>'
            elif len(vars_str) == 1:
                num = f'{pd}<sup>{total_order}</sup>'
                den = f'{pd}<var>{_format_subscript(vars_str[0])}</var><sup>{total_order}</sup>'
            else:
                num = f'{pd}<sup>{total_order}</sup>'
                den_parts = "".join(f'{pd}<var>{_format_subscript(v)}</var>' for v in vars_str)
                den = den_parts

            html = (
                f'<div class="eq">{name_html}'
                f'<span class="dvc">'
                f'<span class="dvl">{num}</span>'
                f'<span class="dvl">{den}</span>'
                f'</span>'
                f'\u2009({func_display})'
                f'</div>'
            )
        else:
            # Simple: ∂f/∂x — all inside the fraction
            func_html = _format_subscript(func)
            if total_order == 1:
                num = f'{pd}<var>{func_html}</var>'
                den = f'{pd}<var>{_format_subscript(vars_str[0])}</var>'
            elif len(vars_str) == 1:
                num = f'{pd}<sup>{total_order}</sup><var>{func_html}</var>'
                den = f'{pd}<var>{_format_subscript(vars_str[0])}</var><sup>{total_order}</sup>'
            else:
                num = f'{pd}<sup>{total_order}</sup><var>{func_html}</var>'
                den_parts = "".join(f'{pd}<var>{_format_subscript(v)}</var>' for v in vars_str)
                den = den_parts

            html = (
                f'<div class="eq">{name_html}'
                f'<span class="dvc">'
                f'<span class="dvl">{num}</span>'
                f'<span class="dvl">{den}</span>'
                f'</span></div>'
            )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        pd = "d" if total_order == 1 else f"d^{total_order}"
        if len(vars_str) == 1:
            den = f"d{vars_str[0]}" if total_order == 1 else f"d{vars_str[0]}^{total_order}"
        else:
            den = "".join(f"d{v}" for v in vars_str)
        print(f"{prefix}{pd}{func}/{den}")


# ============================================================
# Summation display
# ============================================================

def summation(expr: str, variable: str = "i",
              lower: Optional[str] = None, upper: Optional[str] = None,
              name: Optional[str] = None):
    """
    Display a summation: Σ expr.

    Args:
        expr: Expression to sum (e.g. "a_i", "x_i^2")
        variable: Index variable (e.g. "i", "k")
        lower: Lower limit (e.g. "1", "i=1")
        upper: Upper limit (e.g. "n", "N")
        name: Optional result name

    Examples:
        summation("a_i", "i", "1", "n")
        summation("k^2", "k", "0", "N", "S")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@summation name|expr|variable|lower|upper
        name_str = _esc(name) if name else ""
        lower_str = _esc(lower) if lower else ""
        upper_str = _esc(upper) if upper else ""
        _dsl(f"@@summation {name_str}|{_esc(expr)}|{_esc(variable)}|{lower_str}|{upper_str}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_subscript(expr)

        lower_html = ""
        if lower:
            if "=" in lower:
                lower_html = _format_subscript(lower)
            else:
                lower_html = f'{_format_subscript(variable)}={_format_subscript(lower)}'

        if lower and upper:
            sum_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary-sup">{_format_subscript(upper)}</span>'
                f'<span class="nary">&sum;</span>'
                f'<span class="nary-sub">{lower_html}</span>'
                f'</span>'
            )
        else:
            sum_sym = '<span class="nary">&sum;</span>'

        html = (
            f'<div class="eq">{name_html}'
            f'{sum_sym}'
            f'<var>{expr_html}</var></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            low_str = f"{variable}={lower}" if "=" not in (lower or "") else lower
            print(f"{prefix}sum({expr}, {low_str}..{upper})")
        else:
            print(f"{prefix}sum({expr})")


# ============================================================
# Product display
# ============================================================

def product_op(expr: str, variable: str = "i",
               lower: Optional[str] = None, upper: Optional[str] = None,
               name: Optional[str] = None):
    """
    Display a product: Π expr.

    Args:
        expr: Expression (e.g. "a_i")
        variable: Index variable
        lower: Lower limit
        upper: Upper limit
        name: Optional result name

    Example:
        product_op("a_i", "i", "1", "n", "P")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@product name|expr|variable|lower|upper
        name_str = _esc(name) if name else ""
        lower_str = _esc(lower) if lower else ""
        upper_str = _esc(upper) if upper else ""
        _dsl(f"@@product {name_str}|{_esc(expr)}|{_esc(variable)}|{lower_str}|{upper_str}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_subscript(expr)

        lower_html = ""
        if lower:
            if "=" in lower:
                lower_html = _format_subscript(lower)
            else:
                lower_html = f'{_format_subscript(variable)}={_format_subscript(lower)}'

        if lower and upper:
            prod_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary-sup">{_format_subscript(upper)}</span>'
                f'<span class="nary">&prod;</span>'
                f'<span class="nary-sub">{lower_html}</span>'
                f'</span>'
            )
        else:
            prod_sym = '<span class="nary">&prod;</span>'

        html = (
            f'<div class="eq">{name_html}'
            f'{prod_sym}'
            f'<var>{expr_html}</var></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            low_str = f"{variable}={lower}" if "=" not in (lower or "") else lower
            print(f"{prefix}prod({expr}, {low_str}..{upper})")
        else:
            print(f"{prefix}prod({expr})")


# ============================================================
# Square root display
# ============================================================

def sqrt(expr: str, name: Optional[str] = None, index: Optional[int] = None):
    """
    Display a square root (or nth root).

    Args:
        expr: Expression under the radical
        name: Optional result name
        index: Root index (None for square root, 3 for cube root, etc.)

    Examples:
        sqrt("a^2 + b^2", "c")
        sqrt("x", index=3)
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@sqrt name|expr|index
        name_str = _esc(name) if name else ""
        index_str = str(index) if index else ""
        _dsl(f"@@sqrt {name_str}|{_esc(expr)}|{index_str}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_expr(expr)

        if index:
            pad = f'&hairsp;<sup class="nth">{index}</sup>&hairsp;&hairsp;'
        else:
            pad = '&ensp;&hairsp;&hairsp;'

        html = (
            f'<div class="eq">{name_html}'
            f'{pad}'
            f'<span class="o0">'
            f'<span class="r">\u221A</span>&hairsp;'
            f'{expr_html}'
            f'</span>'
            f'</div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if index:
            print(f"{prefix}{index}-root({expr})")
        else:
            print(f"{prefix}sqrt({expr})")


# ============================================================
# Double integral display
# ============================================================

def double_integral(integrand: str,
                    var1: str = "x", lower1: Optional[str] = None, upper1: Optional[str] = None,
                    var2: str = "y", lower2: Optional[str] = None, upper2: Optional[str] = None,
                    name: Optional[str] = None):
    """
    Display a double integral.

    Args:
        integrand: Expression to integrate
        var1: First variable
        lower1, upper1: Limits for first integral
        var2: Second variable
        lower2, upper2: Limits for second integral
        name: Optional result name

    Example:
        double_integral("f(x,y)", "x", "0", "a", "y", "0", "b", "I")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@double_integral name|integrand|var1|lower1|upper1|var2|lower2|upper2
        name_str = _esc(name) if name else ""
        _dsl(f"@@double_integral {name_str}|{_esc(integrand)}|{_esc(var1)}|{_esc(lower1 or '')}|{_esc(upper1 or '')}|{_esc(var2)}|{_esc(lower2 or '')}|{_esc(upper2 or '')}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        integrand_html = _format_subscript(integrand)

        def _make_int_sym(lower, upper):
            if lower and upper:
                return (
                    f'<span class="nary-wrap">'
                    f'<span class="nary-sup">{_format_subscript(upper)}</span>'
                    f'<span class="nary">&#8747;</span>'
                    f'<span class="nary-sub">{_format_subscript(lower)}</span>'
                    f'</span>'
                )
            return '<span class="nary">&#8747;</span>'

        int1 = _make_int_sym(lower2, upper2)
        int2 = _make_int_sym(lower1, upper1)
        var1_html = _format_subscript(var1)
        var2_html = _format_subscript(var2)

        html = (
            f'<div class="eq">{name_html}'
            f'{int1}{int2}'
            f'<var>{integrand_html}</var>'
            f'<span class="dot-sep">&middot;</span>'
            f'd<var>{var1_html}</var>'
            f'<span class="dot-sep">&middot;</span>'
            f'd<var>{var2_html}</var></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        lims1 = f", {lower1}..{upper1}" if lower1 and upper1 else ""
        lims2 = f", {lower2}..{upper2}" if lower2 and upper2 else ""
        print(f"{prefix}dblint({integrand}, d{var1}{lims1}, d{var2}{lims2})")


# ============================================================
# Limit display
# ============================================================

def limit_op(expr: str, variable: str = "x", to: str = "0",
             direction: Optional[str] = None, name: Optional[str] = None):
    """
    Display a limit expression.

    Args:
        expr: Expression (e.g. "sin(x)/x")
        variable: Variable approaching the limit
        to: Value being approached
        direction: "+" for right, "-" for left, None for both
        name: Optional result name

    Examples:
        limit_op("sin(x)/x", "x", "0")
        limit_op("1/x", "x", "0", direction="+")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@limit name|expr|variable|to|direction
        name_str = _esc(name) if name else ""
        dir_str = _esc(direction) if direction else ""
        _dsl(f"@@limit {name_str}|{_esc(expr)}|{_esc(variable)}|{_esc(to)}|{dir_str}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_subscript(expr)
        var_html = _format_subscript(variable)
        to_html = _format_subscript(to)
        dir_html = f'<sup>{direction}</sup>' if direction else ""

        lim_sym = (
            f'<span class="nary-wrap">'
            f'<span class="nary" style="font-size:120%;color:#333;">lim</span>'
            f'<span class="nary-sub"><var>{var_html}</var>&rarr;{to_html}{dir_html}</span>'
            f'</span>'
        )

        html = (
            f'<div class="eq">{name_html}'
            f'{lim_sym}'
            f'<var>{expr_html}</var></div>'
        )
        _emit(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        d = f"{direction}" if direction else ""
        print(f"{prefix}lim({variable}->{to}{d}) {expr}")


# ============================================================
# Equation number (reference tag)
# ============================================================

def eq_num(tag: str):
    """Add an equation number reference like (1.2) to the last equation."""
    mode = _get_mode()

    if mode == "hekatan":
        _dsl(f"@@eq_num {_esc(tag)}")

    elif mode == "standalone":
        html = f'<span class="eq-num">({tag})</span>'
        if _BUFFER:
            last = _BUFFER[-1]
            if last.endswith("</div>"):
                _BUFFER[-1] = last[:-6] + html + "</div>"
            else:
                _BUFFER.append(html)
        else:
            _BUFFER.append(html)

    else:
        print(f"  ({tag})")


# ============================================================
# Text elements
# ============================================================

def title(text_content: str, level: int = 1):
    """Display a heading."""
    mode = _get_mode()

    if mode == "hekatan":
        _dsl(f"@@title {level}|{_esc(text_content)}")

    elif mode == "standalone":
        tag = f"h{min(level, 6)}"
        _emit(f"<{tag}>{text_content}</{tag}>")

    else:
        marker = "#" * level
        print(f"\n{marker} {text_content}\n")


def heading(text_content: str, level: int = 2):
    """Alias for title()."""
    title(text_content, level)


def table(data: List[List[Any]], header: bool = True):
    """
    Display a formatted table.

    Args:
        data: 2D list [[row1], [row2], ...]
              First row is treated as header if header=True.
        header: If True, first row is rendered as header (bold, shaded).

    Example:
        table([
            ["Method", "w (mm)", "Error %"],
            ["FEM", "-6.635", "0.13"],
            ["SAP2000", "-6.529", "ref"],
        ])
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@table header|row1_c1,row1_c2;row2_c1,row2_c2
        header_flag = "1" if header else "0"
        rows_str = ";".join(",".join(_esc(str(c)) for c in row) for row in data)
        _dsl(f"@@table {header_flag}|{rows_str}")

    elif mode == "standalone":
        html_parts = ['<table class="hekatan-table">']

        for r_idx, row in enumerate(data):
            html_parts.append("<tr>")
            tag = "th" if (r_idx == 0 and header) else "td"
            for cell in row:
                cell_html = _format_subscript(str(cell))
                html_parts.append(f"<{tag}>{cell_html}</{tag}>")
            html_parts.append("</tr>")

        html_parts.append("</table>")
        _emit("".join(html_parts))

    else:  # console
        if not data:
            return
        col_widths = []
        for j in range(len(data[0])):
            w = max(len(str(data[i][j])) for i in range(len(data)))
            col_widths.append(w)

        for i, row in enumerate(data):
            cells = " | ".join(str(row[j]).ljust(col_widths[j]) for j in range(len(row)))
            print(f"| {cells} |")
            if i == 0 and header:
                sep = "-+-".join("-" * w for w in col_widths)
                print(f"+-{sep}-+")


def text(content: str):
    """Display plain text or markdown."""
    mode = _get_mode()

    if mode == "hekatan":
        _dsl(f"@@text {_esc(content)}")

    elif mode == "standalone":
        _emit(f"<p>{_greek(content)}</p>")

    else:
        print(content)


# ============================================================
# Columns layout
# ============================================================

_COLUMNS_ACTIVE = False
_COLUMNS_COUNT = 0
_COLUMNS_PROPS = []
_COLUMNS_CSS = False


def columns(n: int = 2, proportions: str = "", css_columns: bool = False):
    """Start a multi-column layout.

    Args:
        n: Number of columns (2-4 recommended).
        proportions: Column width ratios, e.g. "32:68" or "1:2:1".
            If empty, columns are equal width.
        css_columns: If True, use CSS column-count (auto-flowing text).
            If False (default), use flex layout (explicit column() breaks).

    Examples:
        columns(2)                        # Equal 2-column flex
        columns(2, "32:68")               # 32% / 68% flex
        columns(2, css_columns=True)      # CSS multi-column flow
    """
    global _COLUMNS_ACTIVE, _COLUMNS_COUNT, _COLUMNS_PROPS, _COLUMNS_CSS
    mode = _get_mode()
    _COLUMNS_ACTIVE = True
    _COLUMNS_COUNT = n
    _COLUMNS_CSS = css_columns

    # Parse proportions
    if proportions:
        parts = [float(p) for p in proportions.split(":")]
        total = sum(parts)
        _COLUMNS_PROPS = [p / total * 100 for p in parts]
    else:
        _COLUMNS_PROPS = [100 / n] * n

    if mode == "hekatan":
        prop_str = f" {proportions}" if proportions else ""
        _dsl(f"@@columns {n}{prop_str}")

    elif mode == "standalone":
        if css_columns:
            gap = _PAPER_CONFIG.get("columngap", "8mm") if _PAPER_CONFIG else "1em"
            html = f'<div style="column-count:{n};column-gap:{gap};orphans:3;widows:3;">'
        else:
            gap = _PAPER_CONFIG.get("columngap", "1em") if _PAPER_CONFIG else "1em"
            w = _COLUMNS_PROPS[0]
            html = (
                f'<div class="columns-container" '
                f'style="display:flex;gap:{gap};flex-wrap:wrap;">'
                f'<div class="column" style="flex:0 0 {w:.1f}%;max-width:{w:.1f}%;">'
            )
        _emit(html)
    else:
        print(f"--- Columns ({n}) ---")


def column():
    """Start the next column in a multi-column layout.

    In CSS column mode, this inserts a column break.
    In flex mode, this closes the current column and opens the next.
    """
    mode = _get_mode()

    if mode == "hekatan":
        _dsl("@@column")

    elif mode == "standalone":
        if _COLUMNS_CSS:
            _emit('<div style="break-before:column;"></div>')
        else:
            # Determine which column we're opening (track via buffer)
            # Simple approach: count existing column divs
            col_idx = 1
            for item in _BUFFER:
                if 'class="column"' in item and '</div>' not in item:
                    col_idx += 1
            if col_idx >= len(_COLUMNS_PROPS):
                col_idx = len(_COLUMNS_PROPS) - 1
            w = _COLUMNS_PROPS[col_idx] if col_idx < len(_COLUMNS_PROPS) else _COLUMNS_PROPS[-1]
            html = (
                f'</div>'
                f'<div class="column" style="flex:0 0 {w:.1f}%;max-width:{w:.1f}%;">'
            )
            _emit(html)
    else:
        print("---")


def end_columns():
    """End the multi-column layout."""
    global _COLUMNS_ACTIVE, _COLUMNS_COUNT, _COLUMNS_PROPS, _COLUMNS_CSS
    mode = _get_mode()
    _COLUMNS_ACTIVE = False
    _COLUMNS_COUNT = 0
    _COLUMNS_PROPS = []
    _COLUMNS_CSS = False

    if mode == "hekatan":
        _dsl("@@end_columns")

    elif mode == "standalone":
        _emit('</div></div>')
    else:
        print("--- End Columns ---")


# ============================================================
# Design check
# ============================================================

def check(name: str, value: Any, limit: Any, unit: str = "",
          condition: str = "<=", desc: str = ""):
    """
    Display a design verification check: value <= limit → ✓ or ✗

    Args:
        name: Variable name (e.g. "sigma")
        value: Computed value
        limit: Allowable limit
        unit: Unit string
        condition: Comparison operator ("<=", ">=", "<", ">", "==")
        desc: Optional description

    Example:
        check("sigma", 150, 250, "MPa")           # 150 ≤ 250 → ✓
        check("d", 500, 450, "mm", condition="<=") # 500 ≤ 450 → ✗
    """
    mode = _get_mode()

    # Evaluate condition
    ops = {
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "==": lambda a, b: a == b,
    }
    op_symbols = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "==": "="}
    passed = ops.get(condition, ops["<="])(float(value), float(limit))
    symbol = op_symbols.get(condition, "≤")

    if mode == "hekatan":
        # @@check name|value|limit|unit|condition|desc
        _dsl(f"@@check {_esc(name)}|{_esc(str(value))}|{_esc(str(limit))}|{_esc(unit)}|{_esc(condition)}|{_esc(desc)}")

    elif mode == "standalone":
        display_name = _format_subscript(name)
        unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""
        status_class = "ok" if passed else "err"
        status_mark = "✓" if passed else "✗"
        desc_html = f' <span class="desc">{desc}</span>' if desc else ""

        html = (
            f'<div class="eq">'
            f'<var>{display_name}</var> = '
            f'<span class="val">{_format_expr(str(value))}{unit_html}</span>'
            f' {symbol} '
            f'<span class="val">{_format_expr(str(limit))}{unit_html}</span>'
            f' <span class="{status_class}"><b>{status_mark}</b></span>'
            f'{desc_html}'
            f'</div>'
        )
        _emit(html)

    else:
        mark = "OK" if passed else "FAIL"
        unit_str = f" {unit}" if unit else ""
        desc_str = f"  ({desc})" if desc else ""
        print(f"{name} = {value}{unit_str} {symbol} {limit}{unit_str} → [{mark}]{desc_str}")


# ============================================================
# Image
# ============================================================

def image(src: str, alt: str = "", width: Optional[str] = None, caption: Optional[str] = None):
    """
    Display an image.

    Args:
        src: Image URL or file path
        alt: Alt text
        width: Optional width (e.g. "400px", "80%")
        caption: Optional caption text below image

    Example:
        image("diagram.png", width="60%", caption="Figure 1: Beam diagram")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@image src|alt|width|caption
        width_str = _esc(width) if width else ""
        caption_str = _esc(caption) if caption else ""
        _dsl(f"@@image {_esc(src)}|{_esc(alt)}|{width_str}|{caption_str}")

    elif mode == "standalone":
        style = f' style="max-width:{width};"' if width else ' style="max-width:100%;"'
        img_html = f'<img src="{src}" alt="{alt}"{style}>'
        if caption:
            html = (
                f'<figure style="margin:12px 0;">'
                f'{img_html}'
                f'<figcaption style="font-size:9pt;color:#666;margin-top:4px;font-style:italic;">'
                f'{caption}</figcaption></figure>'
            )
        else:
            html = f'<div style="margin:8px 0;">{img_html}</div>'
        _emit(html)

    else:
        cap = f" ({caption})" if caption else ""
        print(f"[IMAGE: {src}{cap}]")


# ============================================================
# Note / callout
# ============================================================

def note(content: str, kind: str = "info"):
    """
    Display a styled callout/note.

    Args:
        content: Note text
        kind: "info", "warning", "error", "success"

    Example:
        note("Check concrete cover requirements", "warning")
    """
    mode = _get_mode()

    colors = {
        "info":    ("#e3f2fd", "#1565c0", "#bbdefb", "ℹ"),
        "warning": ("#fff3e0", "#e65100", "#ffcc80", "⚠"),
        "error":   ("#fce4ec", "#c62828", "#ef9a9a", "✗"),
        "success": ("#e8f5e9", "#2e7d32", "#a5d6a7", "✓"),
    }
    bg, fg, border_color, icon = colors.get(kind, colors["info"])

    if mode == "hekatan":
        # @@note kind|content
        _dsl(f"@@note {_esc(kind)}|{_esc(content)}")

    elif mode == "standalone":
        html = (
            f'<div style="background:{bg};color:{fg};border-left:4px solid {border_color};'
            f'padding:8px 12px;margin:8px 0;border-radius:4px;font-size:10pt;">'
            f'<b>{icon}</b> {_greek(content)}</div>'
        )
        _emit(html)

    else:
        prefix = {"info": "INFO", "warning": "WARN", "error": "ERR", "success": "OK"}.get(kind, "NOTE")
        print(f"[{prefix}] {content}")


# ============================================================
# Code block
# ============================================================

def code(content: str, lang: str = ""):
    """
    Display a formatted code block.

    Args:
        content: Code text
        lang: Optional language hint for styling

    Example:
        code("import numpy as np\\na = np.array([1,2,3])", "python")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@code lang|content (use base64 for multiline)
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        _dsl(f"@@code {_esc(lang)}|{encoded}")

    elif mode == "standalone":
        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lang_class = f" code-{lang}" if lang else ""
        html = f'<pre class="code-block{lang_class}" style="background:#f8f8f8;border-left:3px solid #3776ab;border-radius:3px;padding:6px 8px;margin:6px 0;font-family:Consolas,monospace;font-size:9pt;line-height:1.4;white-space:pre;overflow-x:auto;">{escaped}</pre>'
        _emit(html)

    else:
        print(f"```{lang}")
        print(content)
        print("```")


# ============================================================
# Utility elements
# ============================================================

def hr():
    """Display a horizontal rule."""
    mode = _get_mode()
    if mode == "hekatan":
        _dsl("@@hr")
    elif mode == "standalone":
        _emit('<hr style="margin:12px 0;border:none;border-top:1px solid #ddd;">')
    else:
        print("-" * 60)


def page_break(left: str = "", right: str = "", linecolor: str = "",
               textcolor: str = ""):
    """Insert a page break with optional running header.

    When called with text arguments, renders a header/footer line at the
    bottom of the page before the break (useful for running headers in
    multi-page documents).

    Args:
        left: Left-aligned running header text.
        right: Right-aligned running header text (e.g. page number).
        linecolor: Color of the separator line.
        textcolor: Color of the header text.
    """
    mode = _get_mode()
    if mode == "hekatan":
        if left or right:
            parts = []
            if left:
                parts.append(f"left={_esc(left)}")
            if right:
                parts.append(f"right={_esc(right)}")
            if linecolor:
                parts.append(f"linecolor={_esc(linecolor)}")
            if textcolor:
                parts.append(f"textcolor={_esc(textcolor)}")
            _dsl(f"@@pagebreak {';'.join(parts)}")
        else:
            _dsl("@@page_break")
    elif mode == "standalone":
        if left or right:
            lc = linecolor or "#999"
            tc = textcolor or "#666"
            _emit(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:8.5pt;color:{tc};border-top:1px solid {lc};'
                f'padding:6px 0;margin-top:20px;">'
                f'<span>{_greek(left)}</span>'
                f'<span>{_greek(right)}</span>'
                f'</div>'
            )
        _emit('<div style="page-break-before:always;"></div>')
    else:
        if left or right:
            print(f"--- {left}  |  {right} ---")
        print("\n" + "=" * 60 + " [PAGE BREAK] " + "=" * 60 + "\n")


def html_raw(content: str):
    """
    Emit raw HTML directly into the output.

    Args:
        content: Raw HTML string

    Example:
        html_raw('<div style="color:red;font-size:24px;">Custom HTML</div>')
    """
    mode = _get_mode()
    if mode == "hekatan":
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        _dsl(f"@@html_raw {encoded}")
    elif mode == "standalone":
        _emit(content)
    else:
        print(f"[HTML] {content[:80]}...")


# ============================================================
# Formula display
# ============================================================

def formula(expression: str, name: Optional[str] = None, unit: str = ""):
    """
    Display a math formula with rich formatting.
    Supports: subscripts (_), superscripts (^), Greek letters, operators.
    This is for showing the formula expression itself (not computed values).

    Args:
        expression: Math expression string (e.g. "M_u / (phi * b * d^2)")
        name: Optional result variable name
        unit: Optional unit

    Example:
        formula("sigma_x = N / A + M * y / I_z")
        formula("A_s * f_y / (0.85 * f_c * b)", "a", "mm")
    """
    mode = _get_mode()

    if mode == "hekatan":
        # @@formula name|expression|unit
        name_str = _esc(name) if name else ""
        _dsl(f"@@formula {name_str}|{_esc(expression)}|{_esc(unit)}")

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        # Use AST renderer for proper sqrt, fractions, superscripts
        try:
            from hekatan.calc_engine import _expr_to_html
            expr_html = _expr_to_html(expression)
        except (ImportError, Exception):
            expr_html = _format_expr(expression)
        unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""
        html = f'<div class="eq">{name_html}{expr_html}{unit_html}</div>'
        _emit(html)

    else:
        prefix = f"{name} = " if name else ""
        unit_str = f" [{unit}]" if unit else ""
        print(f"{prefix}{expression}{unit_str}")


# ============================================================
# Show (standalone mode - generates HTML)
# ============================================================

def show(filename: Optional[str] = None):
    """
    Generate HTML document and open in browser.
    Only works in standalone mode. In hekatan mode, DSL commands already sent to stdout.

    Args:
        filename: Optional output file path. If None, uses temp file.
    """
    if _get_mode() == "hekatan":
        return  # DSL already sent to stdout

    if not _BUFFER:
        print("hekatan: nothing to show")
        return

    html = _generate_html()

    if filename:
        path = filename
    else:
        fd, path = tempfile.mkstemp(suffix=".html", prefix="hekatan_")
        os.close(fd)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open(f"file://{os.path.abspath(path)}")
    print(f"hekatan: opened {path}")


def _generate_html() -> str:
    """Generate complete HTML document with Hekatan CSS.

    If paper() was called, applies paper-specific overrides (font, size, margin, accent).
    """
    body = "\n".join(_BUFFER)

    # Paper-specific CSS overrides
    paper_css = ""
    if _PAPER_CONFIG:
        p = _PAPER_CONFIG
        font = p.get("font", '"Georgia", "Times New Roman", Times, serif')
        fontsize = p.get("fontsize", "10pt")
        color = p.get("color", "#333")
        accent = p.get("accent", "#e0c060")
        bg = p.get("background", "#fff")
        lh = p.get("lineheight", 1.45)
        margin = p.get("margin", "20mm 18mm 25mm 18mm")
        size = p.get("size", "A4")

        paper_css = f"""
/* Paper configuration overrides */
@page {{
    size: {size};
    margin: {margin};
}}
body {{
    font-family: {font};
    font-size: {fontsize};
    color: {color};
    background: {bg};
    line-height: {lh};
}}
h1 {{ border-bottom-color: {accent}; }}
h2 {{ color: {color}; }}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hekatan Output</title>
<style>
{_CSS}
{paper_css}
</style>
</head>
<body>
<div class="hekatan-doc">
{body}
</div>
</body>
</html>"""


# ============================================================
# Helpers
# ============================================================

def _format_subscript(name: str) -> str:
    """Convert A_s to A<sub>s</sub>, handle superscripts, and Greek letters.

    Supports both simple notation (A_s, x^2) and braced notation (A_{steel}, x^{2n}).

    Rules:
      - Braced: _{...} and ^{...} consume everything inside braces
      - Simple superscript ^N: only consume DIGITS (so ∂^2M → ∂²M, not ∂^{2M})
      - Simple subscript _x: consume ONE letter or Greek char, or digits
      - All occurrences are processed (global replacement, not just the first)
    """
    if not name:
        return name
    name = _greek(name)
    # 1) Braced subscripts: _{...}  (global)
    name = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', name)
    # 2) Braced superscripts: ^{...}  (global)
    name = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', name)
    # 3) Simple superscript: ^2, ^42 — only consume DIGITS (global)
    #    This prevents ∂^2M from becoming ∂<sup>2M</sup>
    name = re.sub(r'\^(\d+)', r'<sup>\1</sup>', name)
    # 4) Simple subscript: N_x, σ_x, τ_{xy} — consume one letter/Greek char or digits (global)
    #    Match: letter/Greek NOT preceded by < (to avoid matching inside HTML tags)
    _GREEK_CHARS = 'αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΣΦΨΩ∇∂∞'
    name = re.sub(
        r'_([a-zA-Z' + _GREEK_CHARS + r'])',
        lambda m: f'<sub>{m.group(1)}</sub>',
        name
    )
    # Also handle _N where N is digits: M_1, σ_12
    name = re.sub(r'_(\d+)', r'<sub>\1</sub>', name)
    return name


def _format_expr(expr: str) -> str:
    """Format a math expression: split by operators, apply subscript/Greek to each token."""
    if not expr:
        return expr
    stripped = expr.strip()
    import re as _re
    if _re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', stripped):
        return stripped
    tokens = _re.split(r'(\s*[+\-*/=<>]\s*|\s*\*\*\s*)', expr)
    result = []
    for token in tokens:
        tok_stripped = token.strip()
        if tok_stripped in ('+', '-', '*', '/', '=', '<', '>', '**'):
            result.append(f' {tok_stripped} ' if tok_stripped != '*' else ' &middot; ')
        elif not tok_stripped:
            result.append(token)
        else:
            result.append(_format_subscript(tok_stripped))
    return ''.join(result)


# ============================================================
# Greek letter mapping
# ============================================================

_GREEK_MAP = {
    # Lowercase
    "alpha": "\u03B1",    "beta": "\u03B2",     "gamma": "\u03B3",
    "delta": "\u03B4",    "epsilon": "\u03B5",   "zeta": "\u03B6",
    "eta": "\u03B7",      "theta": "\u03B8",     "iota": "\u03B9",
    "kappa": "\u03BA",    "lambda": "\u03BB",    "mu": "\u03BC",
    "nu": "\u03BD",       "xi": "\u03BE",        "omicron": "\u03BF",
    "pi": "\u03C0",       "rho": "\u03C1",       "sigma": "\u03C3",
    "tau": "\u03C4",      "upsilon": "\u03C5",   "phi": "\u03C6",
    "chi": "\u03C7",      "psi": "\u03C8",       "omega": "\u03C9",
    # Uppercase
    "Alpha": "\u0391",    "Beta": "\u0392",      "Gamma": "\u0393",
    "Delta": "\u0394",    "Epsilon": "\u0395",   "Zeta": "\u0396",
    "Eta": "\u0397",      "Theta": "\u0398",     "Iota": "\u0399",
    "Kappa": "\u039A",    "Lambda": "\u039B",    "Mu": "\u039C",
    "Nu": "\u039D",       "Xi": "\u039E",        "Omicron": "\u039F",
    "Pi": "\u03A0",       "Rho": "\u03A1",       "Sigma": "\u03A3",
    "Tau": "\u03A4",      "Upsilon": "\u03A5",   "Phi": "\u03A6",
    "Chi": "\u03A7",      "Psi": "\u03A8",       "Omega": "\u03A9",
    # Common variants
    "varepsilon": "\u03B5", "varphi": "\u03C6",
    "infty": "\u221E",     "infinity": "\u221E",
    # Math symbols
    "nabla": "\u2207",     "partial": "\u2202",
}

import re
_GREEK_NAMES = '|'.join(sorted(_GREEK_MAP.keys(), key=len, reverse=True))
# Word-boundary that works with Spanish characters — prevents "mu" matching inside "comunidad"
_GREEK_PATTERN = re.compile(
    r'(?<![a-zA-Z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc\u00c1\u00c9\u00cd\u00d3\u00da\u00d1\u00dc])'
    r'(' + _GREEK_NAMES + r')'
    r'(?![a-zA-Z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc\u00c1\u00c9\u00cd\u00d3\u00da\u00d1\u00dc])'
)


def _greek(text: str) -> str:
    """Replace Greek letter names with Unicode symbols.

    Uses word-boundary regex that respects Spanish characters,
    so 'mu' won't match inside 'comunidad', 'nu' won't match inside 'continuidad'.
    """
    if not text:
        return text
    if text in _GREEK_MAP:
        return _GREEK_MAP[text]
    return _GREEK_PATTERN.sub(lambda m: _GREEK_MAP[m.group(1)], text)


def _format_unit(unit: str) -> str:
    """Format unit strings like Hekatan Calc."""
    if not unit:
        return unit
    result = unit
    if "/" in result:
        parts = result.split("/", 1)
        left = _format_unit_part(parts[0])
        right = _format_unit_part(parts[1])
        return f"{left}\u2009\u2215\u2009{right}"
    if "*" in result:
        parts = result.split("*")
        formatted = [_format_unit_part(p) for p in parts]
        return ("\u200A\u00B7\u200A").join(formatted)
    return _format_unit_part(result)


def _format_unit_part(part: str) -> str:
    """Format a single unit part: mm^2 -> mm<sup>2</sup>."""
    part = part.strip()
    if "^" in part:
        base, exp = part.split("^", 1)
        return f"{_greek(base)}<sup>{exp}</sup>"
    return _greek(part)


# ============================================================
# Recursive fraction parser (for eq_block)
# ============================================================

def _parse_fraction(expr: str) -> str:
    """Recursively parse (numerator)/(denominator) into HTML fraction divs.

    Handles nested parentheses properly:
        (a + b)/(c + d)  →  fraction bar
        ((a)/(b))/(c)    →  nested fractions

    Returns HTML string with <span class="dvc">/<span class="dvl"> for each fraction.
    """
    result = []
    i = 0
    while i < len(expr):
        if expr[i] == '(':
            # Find matching closing paren
            depth = 1
            j = i + 1
            while j < len(expr) and depth > 0:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                j += 1
            num_content = expr[i + 1:j - 1]

            # Check if followed by /(
            if j < len(expr) and expr[j] == '/' and j + 1 < len(expr) and expr[j + 1] == '(':
                depth = 1
                k = j + 2
                while k < len(expr) and depth > 0:
                    if expr[k] == '(':
                        depth += 1
                    elif expr[k] == ')':
                        depth -= 1
                    k += 1
                den_content = expr[j + 2:k - 1]

                num_html = _parse_fraction(num_content)
                den_html = _parse_fraction(den_content)

                result.append(
                    f'<span class="dvc">'
                    f'<span class="dvl">{_format_subscript(num_html)}</span>'
                    f'<span class="dvl">{_format_subscript(den_html)}</span>'
                    f'</span>'
                )
                i = k
            else:
                inner = _parse_fraction(num_content)
                result.append(f'({inner})')
                i = j
        else:
            result.append(expr[i])
            i += 1
    return ''.join(result)


def _parse_calc_eq(expr: str) -> str:
    """Parse Hekatan Calc-style equation syntax into HTML.

    Handles:
      - Fractions: (a)/(b) with recursive nesting
      - Integrals: ∫_{lower}^{upper}
      - Summation: Σ_{lower}^{upper}
      - Nabla: ∇^2, ∇^4
      - Equation numbers: (N) at end of line
      - Subscripts/superscripts: _{...}, ^{...}
      - Greek letters: alpha, beta, etc.
      - Multiplication dot: ·
    """
    expr = _greek(expr)

    # Equation number: (N), (1.2), (1.5a) at the end
    eq_num_html = ""
    num_match = re.search(r'\s{2,}\((\d+[\.\d]*[a-z]?)\)\s*$', expr)
    if num_match:
        eq_num_html = f'<span class="eq-num">({num_match.group(1)})</span>'
        expr = expr[:num_match.start()]

    # Fractions with recursive parser
    expr = _parse_fraction(expr)

    # Integrals: ∫_{lower}^{upper}
    def _replace_integral(m):
        lower = m.group(1) if m.group(1) else ""
        upper = m.group(2) if m.group(2) else ""
        lower_html = _format_subscript(lower) if lower else ""
        upper_html = _format_subscript(upper) if upper else ""
        if lower_html or upper_html:
            return (f'<span class="nary-wrap">'
                    f'<span class="nary-sup">{upper_html}</span>'
                    f'<span class="nary">&#8747;</span>'
                    f'<span class="nary-sub">{lower_html}</span>'
                    f'</span>')
        return '<span class="nary">&#8747;</span>'

    expr = re.sub(r'\u222B_\{([^}]*)\}\^\{([^}]*)\}', _replace_integral, expr)
    expr = re.sub(
        r'\u222B_\{([^}]*)\}',
        lambda m: (f'<span class="nary-wrap">'
                   f'<span class="nary-sup"></span>'
                   f'<span class="nary">&#8747;</span>'
                   f'<span class="nary-sub">{_format_subscript(m.group(1))}</span>'
                   f'</span>'),
        expr
    )
    expr = re.sub(r'\u222B', lambda m: '<span class="nary">&#8747;</span>', expr)

    # Summation: Σ_{lower}^{upper}
    def _replace_sum(m):
        lower = m.group(1) if m.group(1) else ""
        upper = m.group(2) if m.group(2) else ""
        return (f'<span class="nary-wrap">'
                f'<span class="nary-sup">{_format_subscript(upper)}</span>'
                f'<span class="nary" style="font-size:180%;">\u03A3</span>'
                f'<span class="nary-sub">{_format_subscript(lower)}</span>'
                f'</span>')

    expr = re.sub(r'\u03A3_\{([^}]*)\}\^\{([^}]*)\}', _replace_sum, expr)

    # ∇^N — use lambda to avoid bad escape in replacement string
    expr = re.sub(r'\u2207\^(\d+)', lambda m: f'\u2207<sup>{m.group(1)}</sup>', expr)

    # Subscripts/superscripts
    expr = _format_subscript(expr)

    # Multiplication dot spacing
    expr = expr.replace('\u00B7', ' \u00B7 ')

    return f'<div class="eq" style="justify-content:center;">{expr}{eq_num_html}</div>'


# ============================================================
# eq_block — rich equation display (Hekatan Calc style)
# ============================================================

def eq_block(*equations: str):
    """Display one or more equations using Hekatan Calc expression syntax.

    Each equation string is parsed for:
      - Fractions: (num)/(den) with recursive nesting
      - Integrals: ∫_{lower}^{upper}
      - Summation: Σ_{lower}^{upper}
      - Equation numbers: (N) at end → right-aligned
      - Greek letters, subscripts, superscripts

    Args:
        *equations: Equation strings in Hekatan Calc notation.

    Examples:
        eq_block("k = (E · A)/(L)  (1)")
        eq_block(
            "sigma = (N)/(A) + (M · y)/(I_z)  (2)",
            "epsilon = (partial u)/(partial x)  (3)",
        )
    """
    mode = _get_mode()

    if mode == "hekatan":
        for equation in equations:
            _dsl(f"@@eq_block {_esc(equation)}")

    elif mode == "standalone":
        for equation in equations:
            html = _parse_calc_eq(equation)
            _emit(html)

    else:  # console
        for equation in equations:
            print(f"  {equation}")


# ============================================================
# Markdown text block
# ============================================================

def markdown(content: str):
    """Display markdown-formatted text.

    Supports: headings (#, ##, ###), bold (**text**), italic (*text*),
    unordered lists (- item), and paragraphs.
    Greek letter names are auto-replaced.

    Args:
        content: Markdown text string.

    Example:
        markdown('''
        ## Introduction
        The **finite element method** is based on:
        - Weak formulation
        - Shape functions
        - Assembly procedure
        ''')
    """
    mode = _get_mode()

    if mode == "hekatan":
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        _dsl(f"@@markdown {encoded}")

    elif mode == "standalone":
        lines = content.strip().split('\n')
        in_list = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_list:
                    _emit("</ul>")
                    in_list = False
                _emit("<p>&nbsp;</p>")
                continue
            if stripped.startswith('### '):
                if in_list:
                    _emit("</ul>")
                    in_list = False
                _emit(f"<h3>{_greek(stripped[4:])}</h3>")
            elif stripped.startswith('## '):
                if in_list:
                    _emit("</ul>")
                    in_list = False
                _emit(f"<h2>{_greek(stripped[3:])}</h2>")
            elif stripped.startswith('# '):
                if in_list:
                    _emit("</ul>")
                    in_list = False
                _emit(f"<h1>{_greek(stripped[2:])}</h1>")
            elif stripped.startswith('- '):
                if not in_list:
                    _emit("<ul>")
                    in_list = True
                item = _greek(stripped[2:])
                item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
                item = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', item)
                _emit(f"<li>{item}</li>")
            else:
                if in_list:
                    _emit("</ul>")
                    in_list = False
                txt = _greek(stripped)
                txt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', txt)
                txt = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', txt)
                _emit(f"<p>{txt}</p>")
        if in_list:
            _emit("</ul>")

    else:  # console
        print(content)


# ============================================================
# Figure display (image or SVG with caption)
# ============================================================

def figure(content: str, caption: str = "", number: str = "",
           width: str = "", alt: str = ""):
    """Display a figure with optional caption and numbering.

    Content can be an SVG string, an image URL/path, or raw HTML.

    Args:
        content: SVG string, image path/URL, or HTML content.
        caption: Caption text below the figure.
        number: Figure number (e.g. "1", "2a").
        width: CSS width for the figure (e.g. "80%", "400px").
        alt: Alt text for images.

    Examples:
        figure('<svg ...>...</svg>', "Beam element", "1", width="60%")
        figure("diagram.png", "Cross section", "2", width="400px")
    """
    mode = _get_mode()

    if mode == "hekatan":
        import base64
        # For SVG/HTML send encoded, for file paths send as-is
        if content.strip().startswith('<'):
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            _dsl(f"@@figure html|{encoded}|{_esc(caption)}|{_esc(number)}|{_esc(width)}")
        else:
            _dsl(f"@@figure src|{_esc(content)}|{_esc(caption)}|{_esc(number)}|{_esc(width)}")

    elif mode == "standalone":
        style = f' style="max-width:{width};"' if width else ""

        if content.strip().startswith('<'):
            # SVG or HTML content
            fig_content = content
            if width:
                fig_content = f'<div style="max-width:{width};margin:0 auto;">{content}</div>'
        else:
            # Image file/URL
            fig_content = f'<img src="{content}" alt="{alt}"{style}>'

        cap_html = ""
        if caption:
            num_str = f"<strong>Figura {number}.</strong> " if number else ""
            cap_html = (
                f'<p style="font-size:9pt;text-align:center;color:#444;'
                f'margin:4px 0 12px;font-style:italic;">{num_str}{_greek(caption)}</p>'
            )

        html = f'<div style="text-align:center;margin:12px 0;">{fig_content}</div>{cap_html}'
        _emit(html)

    else:  # console
        num_str = f"Figure {number}: " if number else ""
        cap_str = f" - {caption}" if caption else ""
        if content.strip().startswith('<'):
            print(f"[{num_str}SVG figure{cap_str}]")
        else:
            print(f"[{num_str}{content}{cap_str}]")


# ============================================================
# Paper configuration (for academic/publication layouts)
# ============================================================

_PAPER_CONFIG = {}


def paper(size: str = "A4", margin: str = "20mm 18mm 25mm 18mm",
          font: Optional[str] = None, fontsize: str = "10pt",
          color: str = "#000", accent: str = "#e0c060",
          background: str = "#fff", lineheight: float = 1.45,
          columngap: str = "8mm", **kwargs):
    """Configure paper layout for standalone HTML output.

    Sets page dimensions, fonts, colors, and other print-ready styling.
    Call this BEFORE any content functions.

    Args:
        size: Page size ("A4", "letter", or "WxH" custom).
        margin: CSS margin (e.g. "20mm 18mm 25mm 18mm").
        font: Font family (default: Georgia/serif).
        fontsize: Base font size (e.g. "10pt", "9pt").
        color: Text color.
        accent: Accent color for headings, borders.
        background: Background color.
        lineheight: Line height multiplier.
        columngap: Gap between columns.
        **kwargs: Additional config (startpage, pagenumber, etc.)
    """
    _PAPER_CONFIG.update({
        "size": size,
        "margin": margin,
        "font": font or '"Georgia", "Times New Roman", Times, serif',
        "fontsize": fontsize,
        "color": color,
        "accent": accent,
        "background": background,
        "lineheight": lineheight,
        "columngap": columngap,
        **kwargs,
    })

    mode = _get_mode()
    if mode == "hekatan":
        parts = [f"{k}={_esc(str(v))}" for k, v in _PAPER_CONFIG.items()]
        _dsl(f"@@paper {';'.join(parts)}")
    elif mode == "standalone":
        # Paper config is applied in _generate_html()
        pass


# ============================================================
# Header / Footer (for academic paper layout)
# ============================================================

def header(left: str = "", right: str = "", barside: str = "left",
           color: str = "", textcolor: str = "", **kwargs):
    """Add a page header bar.

    Args:
        left: Left-aligned text (e.g. journal name).
        right: Right-aligned text (e.g. volume info).
        barside: Side of accent bar ("left" or "right").
        color: Background color of the header bar.
        textcolor: Text color.
    """
    mode = _get_mode()

    if mode == "hekatan":
        parts = [f"left={_esc(left)}", f"right={_esc(right)}",
                 f"barside={_esc(barside)}"]
        if color:
            parts.append(f"color={_esc(color)}")
        if textcolor:
            parts.append(f"textcolor={_esc(textcolor)}")
        _dsl(f"@@header {';'.join(parts)}")

    elif mode == "standalone":
        accent = color or _PAPER_CONFIG.get("accent", "#e0c060")
        txt_color = textcolor or "#333"
        bar_css = f"border-left: 4px solid {accent};" if barside == "left" else f"border-right: 4px solid {accent};"
        html = (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 12px;margin-bottom:16px;font-size:8.5pt;color:{txt_color};{bar_css}'
            f'background:linear-gradient(90deg,{accent}10,transparent);">'
            f'<span>{_greek(left)}</span>'
            f'<span>{_greek(right)}</span>'
            f'</div>'
        )
        _emit(html)

    else:
        if left or right:
            print(f"[HEADER] {left}  |  {right}")


def footer(left: str = "", right: str = ""):
    """Add a page footer.

    Args:
        left: Left text (e.g. page number).
        right: Right text (e.g. DOI).
    """
    mode = _get_mode()

    if mode == "hekatan":
        _dsl(f"@@footer left={_esc(left)};right={_esc(right)}")

    elif mode == "standalone":
        html = (
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:8pt;color:#666;border-top:1px solid #ccc;'
            f'padding:6px 0;margin-top:20px;">'
            f'<span>{_greek(left)}</span>'
            f'<span>{_greek(right)}</span>'
            f'</div>'
        )
        _emit(html)

    else:
        if left or right:
            print(f"[FOOTER] {left}  |  {right}")


# ============================================================
# Author block (for academic papers)
# ============================================================

def author(name: str, affiliation: str = "", email: str = "",
           photo: str = ""):
    """Display an author block for academic papers.

    Args:
        name: Author full name.
        affiliation: Institution / university.
        email: Email address.
        photo: Photo URL or path (optional).

    Example:
        author("Dr. Juan Pérez", "Universidad Nacional", "jperez@univ.edu")
    """
    mode = _get_mode()

    if mode == "hekatan":
        _dsl(f"@@author {_esc(name)}|{_esc(affiliation)}|{_esc(email)}|{_esc(photo)}")

    elif mode == "standalone":
        photo_html = ""
        if photo:
            photo_html = (
                f'<img src="{photo}" alt="{name}" '
                f'style="width:60px;height:60px;border-radius:50%;object-fit:cover;margin-right:10px;">'
            )
        html = (
            f'<div style="display:flex;align-items:center;margin:8px 0;font-size:9.5pt;">'
            f'{photo_html}'
            f'<div>'
            f'<div style="font-weight:600;">{_greek(name)}</div>'
        )
        if affiliation:
            html += f'<div style="color:#666;font-size:8.5pt;">{_greek(affiliation)}</div>'
        if email:
            html += f'<div style="color:#06d;font-size:8.5pt;">{email}</div>'
        html += '</div></div>'
        _emit(html)

    else:
        print(f"Author: {name}")
        if affiliation:
            print(f"  {affiliation}")
        if email:
            print(f"  {email}")


# ============================================================
# Abstract block (for academic papers)
# ============================================================

def abstract_block(content: str, keywords: Optional[List[str]] = None,
                   lang: str = ""):
    """Display an abstract block with optional keywords.

    Args:
        content: Abstract text.
        keywords: List of keyword strings.
        lang: Language label (e.g. "english", "español").

    Example:
        abstract_block(
            "This paper presents the method of incompatible modes...",
            keywords=["finite elements", "incompatible modes", "localization"],
            lang="english"
        )
    """
    mode = _get_mode()

    if mode == "hekatan":
        kw_str = ",".join(keywords) if keywords else ""
        _dsl(f"@@abstract {_esc(lang)}|{_esc(content)}|{kw_str}")

    elif mode == "standalone":
        lang_label = f" ({lang})" if lang else ""
        html = (
            f'<div style="margin:12px 0;padding:10px 16px;'
            f'border-left:3px solid {_PAPER_CONFIG.get("accent", "#e0c060")};'
            f'background:#fafafa;font-size:9.5pt;">'
            f'<p style="font-weight:700;margin-bottom:4px;">Abstract{lang_label}</p>'
            f'<p style="text-align:justify;line-height:1.5;">{_greek(content)}</p>'
        )
        if keywords:
            kw_html = "; ".join(f"<em>{_greek(k)}</em>" for k in keywords)
            html += f'<p style="margin-top:6px;font-size:9pt;"><strong>Keywords:</strong> {kw_html}</p>'
        html += '</div>'
        _emit(html)

    else:
        print(f"\n--- Abstract{f' ({lang})' if lang else ''} ---")
        print(content)
        if keywords:
            print(f"Keywords: {', '.join(keywords)}")
        print("---\n")


# ============================================================
# Embedded CSS (matches Hekatan Calc real template)
# Only used by standalone mode (show())
# ============================================================

_CSS = """
/* Hekatan Calc CSS - Real template styles */
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    background: #fff;
    padding: 24px 40px;
    max-width: 960px;
    margin: 0 auto;
}

.hekatan-doc { padding: 16px 0; }

h1 {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.8em;
    margin: 20px 0 10px;
    color: #1a1a1a;
    border-bottom: 2px solid #e0c060;
    padding-bottom: 4px;
}

h2 {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.4em;
    margin: 18px 0 8px;
    color: #2a2a2a;
}

h3 {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.15em;
    margin: 14px 0 6px;
    color: #333;
}

p { margin: 6px 0; }

/* ============================================
   Equation line (.eq) - LaTeX-like math font
   ============================================ */
.eq, .matrix {
    font-family: 'MJXTEX', 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
}

.eq {
    font-size: 12pt;
    line-height: 2;
    margin: 4px 0;
    color: #000;
}

.eq var {
    font-family: 'MJXTEX-I', 'MJXTEX', 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    color: #000;
    font-size: 105%;
    font-style: italic;
}

.eq i {
    color: #000;
    font-style: italic;
    font-size: 100%;
    font-weight: normal;
}

.eq b {
    font-weight: 600;
    color: #000;
}

.eq sub {
    font-family: 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    font-size: 75%;
    vertical-align: -20%;
    margin-left: 1pt;
}

.eq sup {
    display: inline-block;
    margin-left: 1pt;
    margin-top: -3pt;
    font-size: 70%;
}

.eq small {
    font-family: Calibri, Candara, Corbel, sans-serif;
    font-size: 70%;
}

.unit {
    font-size: 0.85em;
    color: #666;
    font-family: Calibri, Candara, Corbel, sans-serif;
}

.val {
    font-weight: 600;
    color: #1a1a1a;
}

.desc {
    font-size: 0.85em;
    color: #888;
    font-style: italic;
    margin-left: 4px;
}

/* ============================================
   Matrix with bracket borders (CSS grid)
   ============================================ */
.matrix {
    display: inline-grid !important;
    gap: 0;
    margin: 4px 2px;
    vertical-align: middle;
    border-left: solid 1.5pt black;
    border-right: solid 1.5pt black;
    border-radius: 3px;
    padding: 2pt 5pt;
}

.matrix .mr {
    display: contents;
}

.matrix .mc {
    white-space: nowrap;
    padding: 1pt 5pt;
    min-width: 10pt;
    text-align: center;
    font-size: 10pt;
}

/* ============================================
   Fraction / Division (.dvc, .dvl)
   ============================================ */
.dvc, .dvr, .dvs {
    display: inline-block;
    vertical-align: middle;
    white-space: nowrap;
}

.dvc {
    padding-left: 2pt;
    padding-right: 2pt;
    text-align: center;
    line-height: 110%;
}

.dvr {
    text-align: center;
    line-height: 110%;
    margin-bottom: 4pt;
}

.dvs {
    text-align: left;
    line-height: 110%;
}

.dvl {
    display: block;
    border-bottom: solid 1pt black;
    margin-top: 1pt;
    margin-bottom: 1pt;
    padding: 1px 6px;
}

.dvl:last-child {
    border-bottom: none;
}

/* ============================================
   Integral / Nary operators
   ============================================ */
.nary {
    color: #000;
    font-size: 100%;
    font-family: 'MJXTEX-S2', 'MJXTEX-S1', 'MJXTEX', 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', 'Times New Roman', serif;
    font-weight: normal;
    -webkit-text-stroke: 0.2px white;
    paint-order: stroke fill;
    transform: translateX(-5px);
    line-height: 1;
    display: block;
    margin: 0;
    padding: 0;
}

.nary-wrap {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 -2px;
    padding-left: 8px;
    line-height: 1;
}

/* Tighten consecutive integrals (double/triple) */
.nary-wrap + .nary-wrap {
    margin-left: -4px;
}

.nary-sup {
    font-size: 75%;
    line-height: 1;
    order: -1;
    color: #000;
    font-family: 'MJXTEX', 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    min-height: 1.1em;
    padding-bottom: 10px;
    align-self: center;
    transform: translateX(6px);
}

.nary-sub {
    font-size: 75%;
    line-height: 1;
    color: #000;
    font-family: 'MJXTEX', 'STIX Two Math', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', serif;
    min-height: 1.1em;
    padding-top: 10px;
    align-self: center;
    transform: translateX(-8px);
}

.dot-sep {
    margin: 0 1px;
}

/* ============================================
   Equation number
   ============================================ */
.eq-num {
    margin-left: auto;
    padding-right: 8px;
    font-size: 11pt;
    color: #000;
}

/* ============================================
   Code output
   ============================================ */
.lang-output-text {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 10.5pt;
    line-height: 1.5;
    white-space: pre-wrap;
    color: #333;
    padding: 4px 0;
    margin: 4px 0;
}

/* ============================================
   Block (nested expressions)
   ============================================ */
.block {
    display: inline-block;
    vertical-align: middle;
    padding-left: 4pt;
    margin-left: -1pt;
    border-left: solid 1pt #80b0e8;
    background: linear-gradient(to right, rgba(0, 192, 255, 0.06), rgba(0, 192, 255, 0.03));
}

/* ============================================
   Bracket sizes (.b0-.b3, .c1-.c8)
   ============================================ */
.b0, .b1, .b2, .b3,
.c1, .c2, .c3, .c4,
.c5, .c6, .c7, .c8 {
    display: inline-block;
    vertical-align: middle;
    font-weight: 100;
    font-stretch: ultra-condensed;
}

.b0 { font-size: 120%; font-weight: 600; padding: 0 1pt; }
.b1 { font-size: 240%; margin-top: -3pt; margin-left: -1pt; margin-right: -1pt; }
.b2 { font-size: 370%; margin-top: -5pt; margin-left: -3pt; margin-right: -3pt; }
.b3 { font-size: 520%; margin-top: -8pt; margin-left: -5pt; margin-right: -5pt; }
.c1 { font-size: 240%; margin-top: -4pt; }
.c2 { font-size: 360%; margin-top: -6pt; margin-left: -2.5pt; margin-right: -0.5pt; }
.c3 { font-size: 480%; margin-top: -8pt; margin-left: -3pt; margin-right: -1pt; }
.c4 { font-size: 600%; margin-top: -10pt; margin-left: -4pt; margin-right: -2pt; transform: scaleX(0.9); }

/* Design check */
.ok {
    color: green;
    background-color: #F0FFF0;
    padding: 1px 4px;
    font-size: 9pt;
}

.err {
    color: red;
    background-color: #FFF0F0;
    padding: 1px 4px;
    font-size: 9pt;
}

/* ============================================
   Square root (matches Hekatan Calc template)
   ============================================ */
.o0 {
    display: inline-block;
    border-top: solid 0.75pt;
    line-height: 130%;
    vertical-align: middle;
    margin-top: 0.75pt;
    padding-top: 1.25pt;
    padding-left: 1pt;
    padding-right: 1pt;
}

.r {
    font-family: 'Times New Roman', Times, serif;
    font-size: 150%;
    display: inline-block;
    vertical-align: top;
    margin-left: -9.5pt;
    position: relative;
    top: 1pt;
}

.nth {
    position: relative;
    bottom: 1pt;
}

.eq small.nth { font-size: 70%; }

/* ============================================
   Table
   ============================================ */
.hekatan-table {
    border-collapse: collapse;
    margin: 8px 0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10.5pt;
}

.hekatan-table th,
.hekatan-table td {
    border: 1px solid #ccc;
    padding: 4px 10px;
    text-align: center;
}

.hekatan-table th {
    background: #f0f0f0;
    font-weight: 600;
    color: #333;
}

.hekatan-table tr:nth-child(even) {
    background: #fafafa;
}

/* ============================================
   Columns layout
   ============================================ */
.columns-container {
    display: flex;
    gap: 1em;
    flex-wrap: wrap;
    margin: 8px 0;
}

.columns-container .column {
    flex: 1;
    min-width: 20%;
}

@media (max-width: 700px) {
    .columns-container {
        flex-direction: column !important;
    }
    .columns-container .column {
        min-width: 100% !important;
        max-width: 100% !important;
    }
}

@media print {
    .columns-container {
        break-inside: avoid;
    }
}
"""
