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
    """Convert A_s to A<sub>s</sub> and handle superscripts like x^2."""
    if not name:
        return name
    # Handle superscripts: x^2 -> x<sup>2</sup>
    if "^" in name:
        parts = name.split("^", 1)
        base = _format_subscript(parts[0])  # recurse for subscripts in base
        return f"{base}<sup>{parts[1]}</sup>"
    # Handle subscripts: A_s -> A<sub>s</sub>
    if "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]}<sub>{parts[1]}</sub>"
    return name


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
"""
