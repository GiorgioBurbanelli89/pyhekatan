"""
Core display functions for Hekatan.

Each function works in 3 modes:
  - HEKATAN mode: prints @@HEKATAN markers (detected by C# side)
  - STANDALONE mode: accumulates HTML, show() opens in browser
  - CONSOLE mode: ASCII formatted output
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


def _detect_mode() -> str:
    """Auto-detect rendering mode."""
    if os.environ.get("HEKATAN_RENDER") == "1":
        return "hekatan"
    return "standalone"


def _get_mode() -> str:
    global _MODE
    if _MODE is None:
        _MODE = _detect_mode()
    return _MODE


def set_mode(mode: str):
    """Force a specific mode: 'hekatan', 'standalone', or 'console'."""
    global _MODE
    if mode not in ("hekatan", "standalone", "console"):
        raise ValueError(f"Invalid mode: {mode}. Use 'hekatan', 'standalone', or 'console'.")
    _MODE = mode


def clear():
    """Clear the accumulated buffer."""
    _BUFFER.clear()


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
        rows = ";".join(",".join(str(x) for x in row) for row in data)
        marker = f"@@HEKATAN:MATRIX:{name or ''}[{rows}]"
        print(marker)

    elif mode == "standalone":
        html = _matrix_to_html(data, name)
        _BUFFER.append(html)

    else:  # console
        _matrix_to_console(data, name)


def _matrix_to_html(data: List[List[Any]], name: Optional[str] = None) -> str:
    """Generate HTML for a matrix using Hekatan CSS classes."""
    rows_html = []
    for row in data:
        cells = "".join(f'<td class="td">{_format_subscript(str(x))}</td>' for x in row)
        # Empty first/last cells create bracket effect
        rows_html.append(f'<tr class="tr"><td class="td"></td>{cells}<td class="td"></td></tr>')

    table = f'<table class="matrix">{"".join(rows_html)}</table>'

    if name:
        display_name = _format_subscript(name)
        return f'<div class="eq"><var>{display_name}</var> = {table}</div>'
    return f'<div class="eq">{table}</div>'


def _matrix_to_console(data: List[List[Any]], name: Optional[str] = None):
    """Print matrix in ASCII format."""
    if not data:
        print("[]")
        return

    # Calculate column widths
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
        marker = f"@@HEKATAN:EQ:{name}={value}"
        if unit:
            marker += f"|{unit}"
        print(marker)

    elif mode == "standalone":
        unit_html = f' <span class="unit">{_format_unit(unit)}</span>' if unit else ""
        display_name = _format_subscript(name)
        html = f'<div class="eq"><var>{display_name}</var> = <b>{value}</b>{unit_html}</div>'
        _BUFFER.append(html)

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
        parts = [f"@@HEKATAN:VAR:{name}={value}"]
        if unit:
            parts[0] += f"|{unit}"
        if desc:
            parts[0] += f"|{desc}"
        print(parts[0])

    elif mode == "standalone":
        display_name = _format_subscript(name)
        unit_html = f' <span class="unit">{_format_unit(unit)}</span>' if unit else ""
        desc_html = f' <span class="desc">{desc}</span>' if desc else ""
        html = f'<div class="eq"><var>{display_name}</var> = <b>{value}</b>{unit_html}{desc_html}</div>'
        _BUFFER.append(html)

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
        marker = f"@@HEKATAN:FRAC:{name or ''}={numerator}/{denominator}"
        print(marker)

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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        num_str = str(numerator)
        den_str = str(denominator)
        width = max(len(num_str), len(den_str))
        print(f"{prefix}{num_str.center(width)}")
        print(f"{' ' * len(prefix)}{'-' * width}")
        print(f"{' ' * len(prefix)}{den_str.center(width)}")


# ============================================================
# Integral display (new in 0.2.0)
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
        marker = f"@@HEKATAN:INTEGRAL:{name or ''}={integrand}|{variable}"
        if lower and upper:
            marker += f"|{lower}|{upper}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        integrand_html = _format_subscript(integrand)
        var_html = _format_subscript(variable)

        if lower and upper:
            integral_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary-sup">{_format_subscript(upper)}</span>'
                f'<span class="nary">&#8747;</span>'
                f'<span class="nary-sub">{_format_subscript(lower)}</span>'
                f'</span>'
            )
        else:
            integral_sym = '<span class="nary">&#8747;</span>'

        html = (
            f'<div class="eq">{name_html}'
            f'{integral_sym}'
            f'<var>{integrand_html}</var>'
            f'<span class="dot-sep">&middot;</span>'
            f'd<var>{var_html}</var></div>'
        )
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            print(f"{prefix}integral({integrand}, {variable}, {lower}, {upper})")
        else:
            print(f"{prefix}integral({integrand}, {variable})")


# ============================================================
# Derivative display (new in 0.3.0)
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
        marker = f"@@HEKATAN:DERIV:{name or ''}={func}|{variable}|{order}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        func_html = _format_subscript(func)
        var_html = _format_subscript(variable)

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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if order == 1:
            print(f"{prefix}d{func}/d{variable}")
        else:
            print(f"{prefix}d^{order}{func}/d{variable}^{order}")


# ============================================================
# Partial derivative display (new in 0.3.0)
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
        v_list = ",".join(vars_str)
        marker = f"@@HEKATAN:PARTIAL:{name or ''}={func}|{v_list}|{total_order}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        func_html = _format_subscript(func)
        pd = "\u2202"  # ∂

        if total_order == 1:
            num = f'{pd}<var>{func_html}</var>'
            den = f'{pd}<var>{_format_subscript(vars_str[0])}</var>'
        elif len(vars_str) == 1:
            num = f'{pd}<sup>{total_order}</sup><var>{func_html}</var>'
            den = f'{pd}<var>{_format_subscript(vars_str[0])}</var><sup>{total_order}</sup>'
        else:
            # Mixed: ∂²f / ∂x∂y
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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        pd = "d" if total_order == 1 else f"d^{total_order}"
        if len(vars_str) == 1:
            den = f"d{vars_str[0]}" if total_order == 1 else f"d{vars_str[0]}^{total_order}"
        else:
            den = "".join(f"d{v}" for v in vars_str)
        print(f"{prefix}{pd}{func}/{den}")


# ============================================================
# Summation display (new in 0.3.0)
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
        marker = f"@@HEKATAN:SUM:{name or ''}={expr}|{variable}"
        if lower and upper:
            marker += f"|{lower}|{upper}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_subscript(expr)

        # Format lower limit: if just a number, add "var ="
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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            low_str = f"{variable}={lower}" if "=" not in (lower or "") else lower
            print(f"{prefix}sum({expr}, {low_str}..{upper})")
        else:
            print(f"{prefix}sum({expr})")


# ============================================================
# Product display (new in 0.3.0)
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
        marker = f"@@HEKATAN:PROD:{name or ''}={expr}|{variable}"
        if lower and upper:
            marker += f"|{lower}|{upper}"
        print(marker)

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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if lower and upper:
            low_str = f"{variable}={lower}" if "=" not in (lower or "") else lower
            print(f"{prefix}prod({expr}, {low_str}..{upper})")
        else:
            print(f"{prefix}prod({expr})")


# ============================================================
# Square root display (new in 0.3.0)
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
        idx = f"|{index}" if index else ""
        marker = f"@@HEKATAN:SQRT:{name or ''}={expr}{idx}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        expr_html = _format_subscript(expr)
        index_html = f'<sup class="sqrt-index">{index}</sup>' if index else ""

        html = (
            f'<div class="eq">{name_html}'
            f'{index_html}'
            f'<span class="sqrt-sym">&radic;</span>'
            f'<span class="sqrt-body">{expr_html}</span>'
            f'</div>'
        )
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        if index:
            print(f"{prefix}{index}-root({expr})")
        else:
            print(f"{prefix}sqrt({expr})")


# ============================================================
# Double integral display (new in 0.3.0)
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
        marker = (f"@@HEKATAN:DBLINT:{name or ''}={integrand}"
                  f"|{var1}|{lower1 or ''}|{upper1 or ''}"
                  f"|{var2}|{lower2 or ''}|{upper2 or ''}")
        print(marker)

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
        _BUFFER.append(html)

    else:  # console
        prefix = f"{name} = " if name else ""
        lims1 = f", {lower1}..{upper1}" if lower1 and upper1 else ""
        lims2 = f", {lower2}..{upper2}" if lower2 and upper2 else ""
        print(f"{prefix}dblint({integrand}, d{var1}{lims1}, d{var2}{lims2})")


# ============================================================
# Limit display (new in 0.3.0)
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
        d = direction or ""
        marker = f"@@HEKATAN:LIMIT:{name or ''}={expr}|{variable}|{to}|{d}"
        print(marker)

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
        _BUFFER.append(html)

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
        print(f"@@HEKATAN:EQNUM:{tag}")

    elif mode == "standalone":
        html = f'<span class="eq-num">({tag})</span>'
        # Modify last buffer entry to append the eq number
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
    tag = f"h{min(level, 6)}"

    if mode == "hekatan":
        print(f"@@HEKATAN:TITLE:{level}|{text_content}")

    elif mode == "standalone":
        _BUFFER.append(f"<{tag}>{text_content}</{tag}>")

    else:
        marker = "#" * level
        print(f"\n{marker} {text_content}\n")


def heading(text_content: str, level: int = 2):
    """Alias for title()."""
    title(text_content, level)


def text(content: str):
    """Display plain text or markdown."""
    mode = _get_mode()

    if mode == "hekatan":
        print(f"@@HEKATAN:TEXT:{content}")

    elif mode == "standalone":
        _BUFFER.append(f"<p>{content}</p>")

    else:
        print(content)


# ============================================================
# Show (standalone mode - generates HTML)
# ============================================================

def show(filename: Optional[str] = None):
    """
    Generate HTML document and open in browser.
    Only works in standalone mode.

    Args:
        filename: Optional output file path. If None, uses temp file.
    """
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
    """Generate complete HTML document with Hekatan CSS."""
    body = "\n".join(_BUFFER)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hekatan Output</title>
<style>
{_CSS}
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
    """Convert A_s to A<sub>s</sub>, handle superscripts, and Greek letters."""
    if not name:
        return name
    # First apply Greek letter conversion
    name = _greek(name)
    # Handle superscripts: x^2 -> x<sup>2</sup>
    if "^" in name:
        parts = name.split("^", 1)
        base = _format_subscript(parts[0])  # recurse for subscripts in base
        return f"{base}<sup>{parts[1]}</sup>"
    # Handle subscripts: A_s -> A<sub>s</sub>
    if "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]}<sub>{_greek(parts[1])}</sub>"
    return name


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
}

import re
# Build regex: match longest names first to avoid partial matches (e.g. "epsilon" before "nu")
# Use word boundaries OR underscore/start/end as delimiters
_GREEK_NAMES = '|'.join(sorted(_GREEK_MAP.keys(), key=len, reverse=True))
_GREEK_PATTERN = re.compile(
    r'(?:^|(?<=[\s_*()/,]))(' + _GREEK_NAMES + r')(?=[\s_^*()/,]|$)'
)


def _greek(text: str) -> str:
    """Replace Greek letter names with Unicode symbols."""
    if not text:
        return text
    # Also handle standalone exact matches (single word)
    if text in _GREEK_MAP:
        return _GREEK_MAP[text]
    return _GREEK_PATTERN.sub(lambda m: _GREEK_MAP[m.group(1)], text)


def _format_unit(unit: str) -> str:
    """Format unit strings: mm^2 -> mm<sup>2</sup>, kN*m -> kN&middot;m."""
    if not unit:
        return unit
    # Handle powers: mm^2 -> mm<sup>2</sup>
    result = unit
    if "^" in result:
        parts = result.split("^", 1)
        result = f"{parts[0]}<sup>{parts[1]}</sup>"
    # Handle multiplication: kN*m -> kN&middot;m
    result = result.replace("*", "&middot;")
    # Also convert Greek letters in units
    result = _greek(result)
    return result


# ============================================================
# Embedded CSS (matches Hekatan Calc real template)
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
   Equation line (.eq) - Georgia Pro serif font
   ============================================ */
.eq, table.matrix {
    font-family: 'Georgia Pro', 'Century Schoolbook', 'Times New Roman', Times, serif;
}

.eq {
    font-size: 11pt;
    line-height: 2;
    margin: 2px 0;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}

.eq var {
    color: #06d;
    font-size: 105%;
    font-style: italic;
}

.eq i {
    color: #086;
    font-style: normal;
    font-size: 90%;
}

.eq b {
    font-weight: 600;
    color: #1a1a1a;
}

.eq sub {
    font-family: Calibri, Candara, Corbel, sans-serif;
    font-size: 80%;
    vertical-align: -18%;
    margin-left: 1pt;
}

.eq sup {
    display: inline-block;
    margin-left: 1pt;
    margin-top: -3pt;
    font-size: 75%;
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

.desc {
    font-size: 0.85em;
    color: #888;
    font-style: italic;
    margin-left: 4px;
}

/* ============================================
   Matrix with bracket borders
   ============================================ */
.matrix {
    display: inline-table;
    border-collapse: collapse;
    margin: 4px 2px;
    vertical-align: middle;
}

.matrix .tr {
    display: table-row;
}

.matrix .td {
    white-space: nowrap;
    padding: 0 2pt 0 2pt;
    min-width: 10pt;
    display: table-cell;
    text-align: center;
    font-size: 10pt;
}

/* Left and right bracket columns (empty cells) */
.matrix .td:first-child,
.matrix .td:last-child {
    width: 0.75pt;
    min-width: 0.75pt;
    max-width: 0.75pt;
    padding: 0 1pt 0 1pt;
}

/* Left bracket border */
.matrix .td:first-child {
    border-left: solid 1pt black;
}

/* Right bracket border */
.matrix .td:last-child {
    border-right: solid 1pt black;
}

/* Top bracket borders */
.matrix .tr:first-child .td:first-child,
.matrix .tr:first-child .td:last-child {
    border-top: solid 1pt black;
}

/* Bottom bracket borders */
.matrix .tr:last-child .td:first-child,
.matrix .tr:last-child .td:last-child {
    border-bottom: solid 1pt black;
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
    color: #C080F0;
    font-size: 240%;
    font-family: 'Georgia Pro Light', 'Georgia Pro', Georgia, serif;
    font-weight: 200;
    line-height: 80%;
    display: block;
    margin: -1pt 1pt 3pt 1pt;
}

.nary-wrap {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 2px;
}

.nary-sup {
    font-size: 70%;
    line-height: 1;
    order: -1;
}

.nary-sub {
    font-size: 70%;
    line-height: 1;
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
    font-size: 10pt;
    color: #666;
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
   Square root
   ============================================ */
.sqrt-sym {
    font-size: 130%;
    vertical-align: middle;
    color: #333;
}

.sqrt-body {
    display: inline-block;
    border-top: solid 1pt black;
    padding: 0 4px 0 2px;
    vertical-align: middle;
}

.sqrt-index {
    font-size: 65%;
    margin-right: -4px;
    vertical-align: super;
}
"""
