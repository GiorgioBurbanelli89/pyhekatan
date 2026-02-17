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
        cells = "".join(f'<td class="td">{x}</td>' for x in row)
        # Empty first/last cells create bracket effect
        rows_html.append(f'<tr class="tr"><td class="td"></td>{cells}<td class="td"></td></tr>')

    table = f'<table class="matrix">{"".join(rows_html)}</table>'

    if name:
        return f'<div class="eq"><var>{name}</var> = {table}</div>'
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
        eq("A_s", 1256.64, "mm²")
    """
    mode = _get_mode()

    if mode == "hekatan":
        marker = f"@@HEKATAN:EQ:{name}={value}"
        if unit:
            marker += f"|{unit}"
        print(marker)

    elif mode == "standalone":
        unit_html = f' <span class="unit">{unit}</span>' if unit else ""
        # Handle subscripts: A_s -> A<sub>s</sub>
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
        unit_html = f' <span class="unit">{unit}</span>' if unit else ""
        desc_html = f' <span class="desc">— {desc}</span>' if desc else ""
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
        fraction("M_u", "phi * b * d²", "R_n")
    """
    mode = _get_mode()

    if mode == "hekatan":
        marker = f"@@HEKATAN:FRAC:{name or ''}={numerator}/{denominator}"
        print(marker)

    elif mode == "standalone":
        name_html = f'<var>{_format_subscript(name)}</var> = ' if name else ""
        html = (
            f'<div class="eq">{name_html}'
            f'<span class="dvc">'
            f'<span class="dvl">{numerator}</span>'
            f'<span class="dvl">{denominator}</span>'
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
    """Convert A_s to A<sub>s</sub>."""
    if "_" in name:
        parts = name.split("_", 1)
        return f"{parts[0]}<sub>{parts[1]}</sub>"
    return name


# ============================================================
# Embedded CSS (matches Hekatan Calc rendering)
# ============================================================

_CSS = """
/* Hekatan Display CSS */
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    background: #fff;
    padding: 24px 32px;
    max-width: 900px;
    margin: 0 auto;
}

.hekatan-doc { padding: 16px 0; }

h1 { font-size: 1.8em; margin: 16px 0 8px; color: #1a1a1a; border-bottom: 2px solid #e0c060; padding-bottom: 4px; }
h2 { font-size: 1.4em; margin: 14px 0 6px; color: #2a2a2a; }
h3 { font-size: 1.15em; margin: 12px 0 4px; color: #333; }

p { margin: 6px 0; }

/* Equation line */
.eq {
    font-family: 'Cambria Math', 'Latin Modern Math', 'STIX Two Math', serif;
    font-size: 11pt;
    line-height: 2;
    margin: 2px 0;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}

.eq var {
    font-style: italic;
    color: #333;
}

.eq b {
    font-weight: 600;
    color: #1a1a1a;
}

.unit {
    font-size: 0.9em;
    color: #666;
}

.desc {
    font-size: 0.9em;
    color: #888;
    font-style: italic;
}

/* Matrix */
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
    display: table-cell;
    text-align: center;
    padding: 2px 8px;
    font-size: 10.5pt;
}

/* Bracket effect: first and last cells have borders */
.matrix .tr .td:first-child {
    border-left: 2px solid #333;
    border-top: 2px solid #333;
    border-bottom: 2px solid #333;
    padding: 2px 4px;
    width: 4px;
}

.matrix .tr:first-child .td:first-child { border-bottom: none; }
.matrix .tr:last-child .td:first-child { border-top: none; }
.matrix .tr:not(:first-child):not(:last-child) .td:first-child { border-top: none; border-bottom: none; }

.matrix .tr .td:last-child {
    border-right: 2px solid #333;
    border-top: 2px solid #333;
    border-bottom: 2px solid #333;
    padding: 2px 4px;
    width: 4px;
}

.matrix .tr:first-child .td:last-child { border-bottom: none; }
.matrix .tr:last-child .td:last-child { border-top: none; }
.matrix .tr:not(:first-child):not(:last-child) .td:last-child { border-top: none; border-bottom: none; }

/* Fraction */
.dvc {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 4px;
}

.dvc .dvl {
    display: block;
    text-align: center;
    padding: 1px 6px;
}

.dvc .dvl:first-child {
    border-bottom: 1.5px solid #333;
}

/* Code output */
.lang-output-text {
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 10.5pt;
    line-height: 1.5;
    white-space: pre-wrap;
    color: #333;
    padding: 4px 0;
    margin: 4px 0;
}
"""
