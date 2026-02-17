# hekatan

> Python display library for engineering calculations — matrices, equations, formatted output

[![PyPI](https://img.shields.io/pypi/v/hekatan.svg)](https://pypi.org/project/hekatan/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

```bash
pip install hekatan
```

## Quick Start

```python
from hekatan import matrix, eq, var, fraction, title, text, show

title("Beam Design")
text("Rectangular section properties:")

var("b", 300, "mm", "beam width")
var("h", 500, "mm", "beam height")

eq("A", 300 * 500, "mm²")

title("Stiffness Matrix", level=2)
K = [[12, 6, -12, 6],
     [6, 4, -6, 2],
     [-12, -6, 12, -6],
     [6, 2, -6, 4]]
matrix(K, "K")

title("Design Ratio", level=2)
fraction("M_u", "φ · b · d²", "R_n")

show()  # Opens formatted HTML in your browser
```

## How It Works

Each function (`matrix()`, `eq()`, `var()`, etc.) works in **3 modes**:

| Mode | When | Behavior |
|------|------|----------|
| **Hekatan** | Inside Hekatan Calc (WPF/CLI) | Emits `@@HEKATAN` markers → rendered as formatted HTML |
| **Standalone** | Regular Python script | `show()` generates HTML, opens in browser |
| **Console** | Fallback | ASCII formatted output |

Mode is auto-detected via `HEKATAN_RENDER=1` environment variable.

## Functions

| Function | Description | Example |
|----------|-------------|---------|
| `matrix(data, name)` | Display formatted matrix | `matrix([[1,2],[3,4]], "A")` |
| `eq(name, value, unit)` | Equation: name = value unit | `eq("F", 25.5, "kN")` |
| `var(name, value, unit, desc)` | Variable with description | `var("b", 300, "mm", "width")` |
| `fraction(num, den, name)` | Formatted fraction | `fraction("M", "S", "σ")` |
| `title(text, level)` | Heading (h1-h6) | `title("Results", 2)` |
| `text(content)` | Paragraph text | `text("Design is OK.")` |
| `show(filename)` | Generate HTML + open browser | `show()` or `show("out.html")` |
| `clear()` | Clear accumulated buffer | `clear()` |
| `set_mode(mode)` | Force mode | `set_mode("console")` |

## Integration with Hekatan Calc

When used inside a Hekatan Calc `.hcalc` document:

```
# My Calculation

@{python}
from hekatan import matrix, eq

K = [[12, 6], [6, 4]]
matrix(K, "K")
eq("det_K", 12*4 - 6*6)
@{end python}
```

The output is automatically formatted with Hekatan's CSS (matrices with brackets, equations with proper typography).

## License

MIT — Giorgio Burbanelli
