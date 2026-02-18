# hekatan

> Python display library for engineering calculations — equations, matrices, figures, academic papers.

[![PyPI version](https://img.shields.io/pypi/v/hekatan.svg)](https://pypi.org/project/hekatan/)
[![Python](https://img.shields.io/pypi/pyversions/hekatan.svg)](https://pypi.org/project/hekatan/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/hekatan.svg)](https://pypi.org/project/hekatan/)

**hekatan** turns your Python engineering scripts into publication-quality HTML documents — with proper math notation, matrices with brackets, fraction bars, integrals, summations, design checks, and academic paper layouts. Zero dependencies. No LaTeX required.

---

## Installation

```bash
pip install hekatan
```

Requires Python 3.8+. No external dependencies.

## Quick Start

```python
from hekatan import matrix, eq, var, fraction, title, text, check, show

title("Beam Design")
text("Rectangular section properties:")

var("b", 300, "mm", "beam width")
var("h", 500, "mm", "beam height")
eq("A", 300 * 500, "mm^2")

title("Stiffness Matrix", level=2)
K = [[12, 6, -12, 6],
     [6, 4, -6, 2],
     [-12, -6, 12, -6],
     [6, 2, -6, 4]]
matrix(K, "K")

title("Design Check", level=2)
fraction("M_u", "phi * b * d^2", "R_n")
check("sigma", 150, 250, "MPa")  # 150 <= 250 -> OK

show()  # Opens formatted HTML in your browser
```

## Rich Equations

The `eq_block()` function renders equations with fractions, integrals, summations, and equation numbers — using Hekatan Calc notation:

```python
from hekatan import eq_block, show

# Fractions: (numerator)/(denominator) with recursive nesting
eq_block("k = (E * A)/(L)  (1)")

# Integrals with limits
eq_block("∫_{a}^{b} f(x)*dx = F(b) - F(a)  (5.5)")

# Summation
eq_block("A = Σ_{i=1}^{N} f_i * Delta*x  (5.2)")

# Nested fractions + integrals
eq_block("I = ∫_{0}^{1} (1)/(e^{3x})*dx ≈ 0.3167  (5.13)")

# Partial derivatives as fractions
eq_block("(∂^2M_x)/(∂x^2) + (∂^2M_y)/(∂y^2) + q = 0  (1.1)")

# Multiple equations at once
eq_block(
    "sigma = (N)/(A) + (M * y)/(I_z)  (2)",
    "epsilon = (partial u)/(partial x)  (3)",
)

show()
```

### Equation Number Syntax

Equation numbers are placed at the end with **2+ spaces** before the parenthesized number:

```python
eq_block("F = m*a  (1)")              # Simple: (1)
eq_block("D = (E*h^3)/(12)  (1.3)")   # Dotted: (1.3)
eq_block("N_x = ∫ sigma_x*dz  (1.5a)")  # With suffix: (1.5a)
```

### Subscript & Superscript Rules

| Notation | Renders as | Notes |
|----------|-----------|-------|
| `x^2` | x² | Simple superscript (digits only) |
| `x^{2n}` | x²ⁿ | Braced superscript (any content) |
| `N_x` | N with subscript x | Simple subscript (one letter) |
| `N_{xy}` | N with subscript xy | Braced subscript (any content) |
| `∂^2M_x` | ∂²Mₓ | `^` takes digits, `_` takes letter |

## Academic Paper Layout

Create publication-quality documents with paper configuration, headers, footers, author blocks, abstracts, and multi-column layouts:

```python
from hekatan import (
    paper, header, footer, author, abstract_block,
    title, heading, text, markdown, eq_block, figure,
    columns, column, end_columns, table, show, clear,
)

clear()

# Paper config: page size, fonts, accent color
paper(
    size="A4",
    margin="20mm 18mm 25mm 18mm",
    fontsize="10pt",
    accent="#F27835",
)

# Header bar
header(left="Journal of Civil Engineering", right="Vol 70, 2018")

# Title and authors
title("Method of Incompatible Modes")
author("Ivo Kozar", "University of Rijeka, Croatia")

# Abstract with keywords
abstract_block(
    "This paper presents the method of incompatible modes...",
    keywords=["finite elements", "incompatible modes"],
    lang="english",
)

# Two-column layout (CSS multi-column flow)
columns(2, css_columns=True)
heading("1. Introduction", 2)
markdown("""
The **finite element method** is based on:
- Weak formulation
- Shape functions
- Assembly procedure
""")

# Rich equations
eq_block("u(x) = N(x) * d + M(x) * alpha  (1)")

# SVG figures with captions
figure('<svg width="400" height="200">...</svg>',
       caption="Beam element with shape functions",
       number="1", width="90%")
end_columns()

show("paper_output.html")
```

## How It Works

Each function works in **3 modes** (auto-detected):

| Mode | When | Behavior |
|------|------|----------|
| **Hekatan** | Inside Hekatan Calc (WPF/CLI) | Emits `@@DSL` commands to stdout |
| **Standalone** | Regular Python script | `show()` generates HTML, opens in browser |
| **Console** | Fallback | ASCII formatted output |

Mode is auto-detected via `HEKATAN_RENDER=1` environment variable. You can force a mode with `set_mode("standalone")`, `set_mode("hekatan")`, or `set_mode("console")`.

## API Reference

### Core Math

| Function | Description | Example |
|----------|-------------|---------|
| `eq(name, value, unit)` | Equation: name = value | `eq("F", 25.5, "kN")` |
| `var(name, value, unit, desc)` | Variable with description | `var("b", 300, "mm", "width")` |
| `eq_block(*equations)` | Rich equations with fractions/integrals | `eq_block("k = (E*A)/(L)  (1)")` |
| `formula(expr, name, unit)` | Math formula display | `formula("A_s * f_y / (0.85 * f_c)")` |
| `fraction(num, den, name)` | Formatted fraction | `fraction("M", "S", "sigma")` |
| `matrix(data, name)` | Matrix with brackets | `matrix([[1,2],[3,4]], "A")` |
| `table(data, header)` | Data table | `table([["x","y"],["1","2"]])` |

### Calculus Operators

| Function | Description | Example |
|----------|-------------|---------|
| `integral(expr, var, lo, hi)` | Integral display | `integral("f(x)", "x", "0", "L")` |
| `double_integral(...)` | Double integral | `double_integral("f", "x", "0", "a", "y", "0", "b")` |
| `derivative(func, var, order)` | Derivative df/dx | `derivative("y", "x")` |
| `partial(func, var, order)` | Partial derivative | `partial("u", "x")` |
| `summation(expr, var, lo, hi)` | Summation operator | `summation("a_i", "i", "1", "n")` |
| `product_op(expr, var, lo, hi)` | Product operator | `product_op("a_i", "i", "1", "n")` |
| `sqrt(expr, name, index)` | Square/nth root | `sqrt("a^2 + b^2", "c")` |
| `limit_op(expr, var, to)` | Limit expression | `limit_op("sin(x)/x", "x", "0")` |

### Paper Layout

| Function | Description | Example |
|----------|-------------|---------|
| `paper(size, margin, ...)` | Page configuration | `paper(size="A4", accent="#F27835")` |
| `header(left, right, ...)` | Page header bar | `header(left="Journal", right="Vol 1")` |
| `footer(left, right)` | Page footer | `footer(left="Page 1")` |
| `author(name, affil, email)` | Author block | `author("Dr. Smith", "MIT")` |
| `abstract_block(text, kw)` | Abstract + keywords | `abstract_block("...", keywords=[...])` |

### Text & Content

| Function | Description | Example |
|----------|-------------|---------|
| `title(text, level)` | Heading (h1-h6) | `title("Results", 2)` |
| `heading(text, level)` | Alias for title() | `heading("Section", 3)` |
| `text(content)` | Paragraph text | `text("The beam is safe.")` |
| `markdown(content)` | Markdown text block | `markdown("**bold** and *italic*")` |
| `figure(content, caption, num)` | Figure with caption | `figure("img.png", "Fig 1", "1")` |
| `image(src, alt, width)` | Simple image | `image("photo.jpg", width="60%")` |
| `check(name, val, limit, unit)` | Design verification | `check("sigma", 150, 250, "MPa")` |
| `note(content, kind)` | Callout/note box | `note("Check cover", "warning")` |
| `code(content, lang)` | Code block | `code("import numpy", "python")` |

### Layout

| Function | Description | Example |
|----------|-------------|---------|
| `columns(n, proportions)` | Start multi-column | `columns(2, "32:68")` |
| `column()` | Next column | `column()` |
| `end_columns()` | End columns | `end_columns()` |
| `hr()` | Horizontal rule | `hr()` |
| `page_break(left, right, ...)` | Page break with running header | `page_break(left="Title", right="15")` |
| `html_raw(content)` | Raw HTML | `html_raw("<div>...</div>")` |
| `eq_num(tag)` | Equation number | `eq_num("1.2")` |

### Control

| Function | Description | Example |
|----------|-------------|---------|
| `show(filename)` | Generate HTML + open browser | `show()` or `show("out.html")` |
| `clear()` | Clear accumulated buffer | `clear()` |
| `set_mode(mode)` | Force rendering mode | `set_mode("console")` |

## Features

- **Zero dependencies** — pure Python, no NumPy/LaTeX/MathJax required
- **Greek letters** — auto-converts `alpha`, `sigma`, `phi`, `Delta`, etc. to symbols
- **Subscripts/superscripts** — `A_s` renders as subscript, `x^2` as superscript
- **Braced notation** — `A_{steel}`, `x^{2n}` for multi-character sub/superscripts
- **Smart superscript** — `^` only consumes digits (`∂^2M` stays correct), use `^{x}` for letters
- **Word-boundary safe** — Greek replacement won't corrupt Spanish/Portuguese words
- **Recursive fractions** — `(a + (b)/(c))/(d)` renders nested fraction bars
- **Integrals** — `∫_{a}^{b}` renders with proper limits above/below
- **Summation** — `Σ_{i=1}^{N}` renders with limits above/below
- **Equation numbers** — `(1.1)`, `(1.5a)`, `(2.3b)` all supported
- **CSS columns** — flex-based (manual breaks) and CSS multi-column (auto-flow) layouts
- **Column proportions** — `columns(2, "32:68")` for asymmetric layouts
- **Page breaks with headers** — running headers on each page
- **Print-ready** — `@page` CSS rules for PDF export via browser print
- **Design checks** — pass/fail verification with color-coded output

## Integration with Hekatan Calc

When used inside a [Hekatan Calc](https://github.com/nickkuijpers/Hekatan-Calc-1.0.0) `.hcalc` document:

```
# My Calculation

@{python}
from hekatan import matrix, eq, eq_block

K = [[12, 6], [6, 4]]
matrix(K, "K")
eq("det_K", 12*4 - 6*6)
eq_block("sigma = (M * y)/(I_z)  (1)")
@{end python}
```

The output is automatically formatted with Hekatan Calc's CSS — matrices with brackets, equations with proper serif typography, fraction bars, integral symbols, and more.

## Example: FEA Slab Analysis

The `examples/` directory includes a full finite element analysis of a rectangular slab using the BFS plate bending element (585 lines). It demonstrates variables, equations, matrices, tables, partial derivatives, double integrals, and design checks — all rendered as a single HTML document.

```bash
cd examples
python rectangular_slab_fea.py
```

## License

[MIT](LICENSE)
