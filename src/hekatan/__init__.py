"""
Hekatan - Python display library for engineering calculations.

Works in 3 modes:
  1. Inside Hekatan Calc (WPF/CLI) - emits @@DSL commands to stdout (C# parses to HTML)
  2. Standalone Python - show() generates HTML and opens in browser
  3. Console fallback - ASCII formatted output

Usage:
    from hekatan import matrix, eq, var, fraction, title, text, show

    A = [[1, 2], [3, 4]]
    matrix(A, "A")
    eq("F", 25.5, "kN")
    var("b", 300, "mm")
    show()  # Opens HTML in browser (standalone mode)
"""

__version__ = "0.8.0"

from hekatan.display import (
    # Core math
    matrix,
    table,
    eq,
    var,
    fraction,
    integral,
    derivative,
    partial,
    summation,
    product_op,
    sqrt,
    double_integral,
    limit_op,
    eq_num,
    formula,
    # Rich equation blocks (Hekatan Calc style)
    eq_block,
    # Layout
    columns,
    column,
    end_columns,
    # Paper / academic layout
    paper,
    header,
    footer,
    author,
    abstract_block,
    # Text & headings
    title,
    text,
    heading,
    markdown,
    # Verification
    check,
    # Media & content
    image,
    figure,
    note,
    code,
    hr,
    page_break,
    html_raw,
    # Control
    show,
    clear,
    set_mode,
)
