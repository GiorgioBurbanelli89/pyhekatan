"""
SymPy bridge for Hekatan — symbolic math with HTML rendering.

Combines SymPy's symbolic computation with Hekatan's HTML display engine.
No LaTeX needed — everything renders as pure HTML/CSS.

Usage:
    from hekatan.sympy_bridge import *

    x, y, z = symbols('x y z')

    # Symbolic integral → HTML
    sym_integral(sin(x), x)                    # ∫ sin(x) dx = -cos(x) + C
    sym_integral(x**2, x, 0, 1)               # ∫₀¹ x² dx = 1/3

    # Symbolic derivative → HTML
    sym_diff(x**3 + 2*x, x)                   # d/dx(x³ + 2x) = 3x² + 2

    # Symbolic equation solving
    sym_solve(x**2 - 4, x)                    # x² - 4 = 0 → x = {-2, 2}

    # Simplify & display
    sym_simplify((x**2 - 1)/(x - 1))          # = x + 1

    # Factor / Expand
    sym_factor(x**3 - 8)                       # = (x - 2)(x² + 2x + 4)
    sym_expand((x + 1)**3)                     # = x³ + 3x² + 3x + 1

    # Limits
    sym_limit(sin(x)/x, x, 0)                 # lim x→0 sin(x)/x = 1

    # Series expansion
    sym_series(exp(x), x, n=5)                 # eˣ = 1 + x + x²/2 + ...

    # Matrix operations (symbolic)
    sym_matrix([[1, 2], [3, 4]], name="A", det=True, inv=True)

    # Generic symbolic display
    sym_eq("result", expr)                     # result = <formatted expr>

Requires: pip install sympy
"""

from typing import Optional, Union, List, Tuple
from hekatan.display import _emit, _get_mode, _dsl, _format_subscript, _greek, _get_css, _in_jupyter


# ============================================================
# SymPy to HTML converter
# ============================================================

def _sympy_to_html(expr, compact: bool = False) -> str:
    """Convert a SymPy expression to Hekatan HTML.

    This is the core converter — no LaTeX, pure HTML/CSS.
    Handles: Add, Mul, Pow, sin, cos, sqrt, Rational, Integral, etc.
    """
    import sympy as sp

    if expr is None:
        return ""

    # Handle SymPy special types
    if isinstance(expr, sp.Rational):
        if expr.q == 1:
            return str(expr.p)
        return (
            f'<span class="frac">'
            f'<span class="frac-num">{_sympy_to_html(sp.Integer(expr.p))}</span>'
            f'<span class="frac-den">{_sympy_to_html(sp.Integer(expr.q))}</span>'
            f'</span>'
        )

    if isinstance(expr, sp.Integer):
        return str(int(expr))

    if isinstance(expr, sp.Float):
        val = float(expr)
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"

    if expr is sp.oo:
        return "&#8734;"  # ∞
    if expr is sp.zoo:
        return "&#8734;&#771;"  # complex infinity
    if expr is sp.nan:
        return "NaN"
    if expr == -sp.oo:
        return "&minus;&#8734;"

    if isinstance(expr, sp.Symbol):
        name = str(expr)
        return f'<var>{_format_subscript(_greek(name))}</var>'

    if isinstance(expr, sp.NumberSymbol):
        # pi, E, etc.
        if expr is sp.pi:
            return '<var>&pi;</var>'
        if expr is sp.E:
            return '<var>e</var>'
        if expr is sp.I:
            return '<var>i</var>'
        return str(expr)

    if expr is sp.I:
        return '<var>i</var>'

    # ---- Functions ----
    if isinstance(expr, sp.Function):
        return _sympy_func_to_html(expr)

    # ---- Pow ----
    if isinstance(expr, sp.Pow):
        base, exp_val = expr.args
        # sqrt: x^(1/2)
        if exp_val == sp.Rational(1, 2):
            return (
                f'<span class="sqrt-wrap">'
                f'<span class="sqrt-sym">&#8730;</span>'
                f'<span class="sqrt-body">{_sympy_to_html(base)}</span>'
                f'</span>'
            )
        # cbrt: x^(1/3)
        if exp_val == sp.Rational(1, 3):
            return (
                f'<span class="sqrt-wrap">'
                f'<sup class="sqrt-idx">3</sup>'
                f'<span class="sqrt-sym">&#8730;</span>'
                f'<span class="sqrt-body">{_sympy_to_html(base)}</span>'
                f'</span>'
            )
        # n-th root: x^(1/n)
        if isinstance(exp_val, sp.Rational) and exp_val.p == 1 and exp_val.q > 1:
            return (
                f'<span class="sqrt-wrap">'
                f'<sup class="sqrt-idx">{exp_val.q}</sup>'
                f'<span class="sqrt-sym">&#8730;</span>'
                f'<span class="sqrt-body">{_sympy_to_html(base)}</span>'
                f'</span>'
            )
        # x^(-1) → 1/x
        if exp_val == -1 and not compact:
            return (
                f'<span class="frac">'
                f'<span class="frac-num">1</span>'
                f'<span class="frac-den">{_sympy_to_html(base)}</span>'
                f'</span>'
            )
        # Rational exponent: x^(p/q) → x^(p/q) as fraction superscript
        if isinstance(exp_val, sp.Rational) and exp_val.q != 1:
            base_html = _sympy_to_html(base)
            if _needs_parens(base, 'pow'):
                base_html = f'({base_html})'
            frac_sup = (
                f'<span class="frac" style="font-size:0.7em">'
                f'<span class="frac-num">{exp_val.p}</span>'
                f'<span class="frac-den">{exp_val.q}</span>'
                f'</span>'
            )
            return f'{base_html}<sup>{frac_sup}</sup>'

        # Negative exponent: x^(-n) — keep as power
        base_html = _sympy_to_html(base)
        if _needs_parens(base, 'pow'):
            base_html = f'({base_html})'
        exp_html = _sympy_to_html(exp_val)
        return f'{base_html}<sup>{exp_html}</sup>'

    # ---- Mul ----
    if isinstance(expr, sp.Mul):
        return _sympy_mul_to_html(expr)

    # ---- Add ----
    if isinstance(expr, sp.Add):
        return _sympy_add_to_html(expr)

    # ---- Equality ----
    if isinstance(expr, sp.Eq):
        lhs = _sympy_to_html(expr.lhs)
        rhs = _sympy_to_html(expr.rhs)
        return f'{lhs} = {rhs}'

    # ---- Matrix ----
    if isinstance(expr, sp.MatrixBase):
        return _sympy_matrix_to_html(expr)

    # ---- Set / FiniteSet / list ----
    if isinstance(expr, (sp.FiniteSet, set)):
        items = sorted(expr, key=lambda x: (float(x) if x.is_real else 0))
        items_html = ', '.join(_sympy_to_html(i) for i in items)
        return f'{{{items_html}}}'

    if isinstance(expr, (list, tuple)):
        items_html = ', '.join(_sympy_to_html(sp.sympify(i)) for i in expr)
        return f'[{items_html}]'

    # Fallback
    return str(expr)


def _needs_parens(expr, context: str) -> bool:
    """Check if an expression needs parentheses in a given context."""
    import sympy as sp
    if context == 'pow':
        return isinstance(expr, (sp.Add, sp.Mul))
    if context == 'mul':
        return isinstance(expr, sp.Add)
    return False


def _sympy_func_to_html(expr) -> str:
    """Convert a SymPy function call to HTML."""
    import sympy as sp

    func_name = expr.func.__name__
    args = expr.args

    # Trig functions
    trig_map = {
        'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
        'cot': 'cot', 'sec': 'sec', 'csc': 'csc',
        'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
        'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
    }
    if func_name in trig_map:
        arg_html = _sympy_to_html(args[0])
        return f'{trig_map[func_name]}({arg_html})'

    # sqrt
    if func_name == 'sqrt' or (isinstance(expr, sp.Pow) and expr.args[1] == sp.Rational(1, 2)):
        arg_html = _sympy_to_html(args[0])
        return (
            f'<span class="sqrt-wrap">'
            f'<span class="sqrt-sym">&#8730;</span>'
            f'<span class="sqrt-body">{arg_html}</span>'
            f'</span>'
        )

    # Abs
    if func_name == 'Abs':
        arg_html = _sympy_to_html(args[0])
        return f'|{arg_html}|'

    # exp
    if func_name == 'exp':
        arg_html = _sympy_to_html(args[0])
        return f'<var>e</var><sup>{arg_html}</sup>'

    # log / ln
    if func_name == 'log':
        if len(args) == 2:
            # log(x, base)
            base_html = _sympy_to_html(args[1])
            arg_html = _sympy_to_html(args[0])
            return f'log<sub>{base_html}</sub>({arg_html})'
        arg_html = _sympy_to_html(args[0])
        return f'ln({arg_html})'

    # factorial
    if func_name == 'factorial':
        arg_html = _sympy_to_html(args[0])
        if isinstance(args[0], (sp.Symbol, sp.Integer)):
            return f'{arg_html}!'
        return f'({arg_html})!'

    # Piecewise
    if func_name == 'Piecewise':
        pieces = []
        for expr_i, cond_i in args:
            e = _sympy_to_html(expr_i)
            c = _sympy_to_html(cond_i) if cond_i is not sp.true else "otherwise"
            pieces.append(f'{e}, &nbsp; {c}')
        body = '<br>'.join(pieces)
        return f'<span style="border-left:2px solid;padding-left:6px">{body}</span>'

    # Generic function
    args_html = ', '.join(_sympy_to_html(a) for a in args)
    return f'{_greek(func_name)}({args_html})'


def _sympy_mul_to_html(expr) -> str:
    """Convert Mul to HTML, handling coefficients and fractions."""
    import sympy as sp

    # SymPy stores a*b^(-1) as Mul(a, Pow(b, -1))
    # Separate numerator and denominator
    numer, denom = expr.as_numer_denom()

    if denom != 1:
        n_html = _sympy_to_html(numer)
        d_html = _sympy_to_html(denom)
        return (
            f'<span class="frac">'
            f'<span class="frac-num">{n_html}</span>'
            f'<span class="frac-den">{d_html}</span>'
            f'</span>'
        )

    # Regular multiplication
    coeff, rest = expr.as_coeff_Mul()
    terms = sp.Mul.make_args(expr)

    parts = []
    for i, term in enumerate(terms):
        h = _sympy_to_html(term, compact=True)
        if _needs_parens(term, 'mul'):
            h = f'({h})'
        parts.append(h)

    # Join with middot or implicit multiplication
    result_parts = []
    for i, p in enumerate(parts):
        if i > 0:
            # Use implicit multiplication (no dot) between coefficient and variable
            prev = terms[i - 1]
            curr = terms[i]
            if isinstance(prev, sp.Number) and not isinstance(curr, sp.Number):
                result_parts.append(p)  # 3x, not 3·x
            elif isinstance(prev, sp.Pow) and isinstance(curr, sp.Pow):
                result_parts.append(f' &middot; {p}')
            else:
                result_parts.append(f' &middot; {p}')
        else:
            # Handle negative coefficient: -1 * x → -x
            if term == sp.Integer(-1) and len(terms) > 1:
                result_parts.append('&minus;')
            else:
                result_parts.append(p)

    return ''.join(result_parts)


def _sympy_add_to_html(expr) -> str:
    """Convert Add to HTML with proper + and - signs."""
    import sympy as sp

    terms = sp.Add.make_args(expr)
    # Sort: positive terms first, then negatives (SymPy default order is fine)
    ordered = list(expr.as_ordered_terms())

    parts = []
    for i, term in enumerate(ordered):
        h = _sympy_to_html(term)
        if i == 0:
            parts.append(h)
        else:
            # Check if term is negative
            if _is_negative(term):
                # Already has minus sign from converter
                neg_term = -term
                neg_html = _sympy_to_html(neg_term)
                parts.append(f' &minus; {neg_html}')
            else:
                parts.append(f' + {h}')

    return ''.join(parts)


def _is_negative(expr) -> bool:
    """Check if a SymPy expression is negative (for display purposes)."""
    import sympy as sp
    if isinstance(expr, sp.Mul):
        coeff = expr.as_coeff_Mul()[0]
        return coeff < 0
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return expr < 0
    return False


def _sympy_matrix_to_html(mat, name: str = None) -> str:
    """Convert SymPy Matrix to Hekatan grid HTML."""
    rows, cols = mat.shape
    cells = []
    for i in range(rows):
        for j in range(cols):
            cells.append(f'<span class="m-cell">{_sympy_to_html(mat[i, j])}</span>')
    grid = ''.join(cells)
    style = f'grid-template-columns: repeat({cols}, auto);'
    name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
    return (
        f'{name_html}'
        f'<span class="m-bracket">[</span>'
        f'<span class="m-grid" style="{style}">{grid}</span>'
        f'<span class="m-bracket">]</span>'
    )


# ============================================================
# Public API — Symbolic operations + HTML display
# ============================================================

def sym_integral(expr, var, lower=None, upper=None, name: str = None):
    """Compute and display a symbolic integral.

    Args:
        expr: SymPy expression to integrate
        var: Integration variable (SymPy Symbol)
        lower: Lower limit (optional, for definite integrals)
        upper: Upper limit (optional, for definite integrals)
        name: Optional label for the result

    Examples:
        x = symbols('x')
        sym_integral(sin(x), x)                  # ∫ sin(x) dx = -cos(x) + C
        sym_integral(x**2, x, 0, 1)              # ∫₀¹ x² dx = 1/3
    """
    import sympy as sp
    mode = _get_mode()

    # Compute
    if lower is not None and upper is not None:
        result = sp.integrate(expr, (var, lower, upper))
        is_definite = True
    else:
        result = sp.integrate(expr, var)
        is_definite = False

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        integrand_html = _sympy_to_html(expr)
        var_html = _sympy_to_html(var)
        result_html = _sympy_to_html(result)

        # Build integral symbol
        if lower is not None and upper is not None:
            lower_html = _sympy_to_html(sp.sympify(lower))
            upper_html = _sympy_to_html(sp.sympify(upper))
            int_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary-sup">{upper_html}</span>'
                f'<span class="nary">&#8747;</span>'
                f'<span class="nary-sub">{lower_html}</span>'
                f'</span>'
            )
        else:
            int_sym = (
                f'<span class="nary-wrap">'
                f'<span class="nary">&#8747;</span>'
                f'</span>'
            )

        # Add parentheses for compound integrand
        if isinstance(expr, sp.Add):
            body = f'({integrand_html})'
        else:
            body = integrand_html

        constant = '' if is_definite else ' + <var>C</var>'
        html = (
            f'<div class="eq">{name_html}'
            f'{int_sym}'
            f'<span>{body}\u2009<var>d</var>{var_html}</span>'
            f' = {result_html}{constant}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        if lower is not None:
            print(f"{prefix}integral({expr}, ({var}, {lower}, {upper})) = {result}")
        else:
            print(f"{prefix}integral({expr}, {var}) = {result} + C")

    return result


def sym_diff(expr, var, order: int = 1, name: str = None):
    """Compute and display a symbolic derivative.

    Args:
        expr: SymPy expression to differentiate
        var: Differentiation variable
        order: Derivative order (1, 2, 3...)
        name: Optional label

    Examples:
        sym_diff(x**3 + 2*x, x)             # d/dx(x³ + 2x) = 3x² + 2
        sym_diff(sin(x)*cos(x), x)          # d/dx(sin(x)cos(x)) = cos(2x)
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.diff(expr, var, order)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        var_html = _sympy_to_html(var)
        result_html = _sympy_to_html(result)

        # Build d/dx or d²/dx² notation
        if order == 1:
            deriv_sym = (
                f'<span class="frac">'
                f'<span class="frac-num"><var>d</var></span>'
                f'<span class="frac-den"><var>d</var>{var_html}</span>'
                f'</span>'
            )
        else:
            deriv_sym = (
                f'<span class="frac">'
                f'<span class="frac-num"><var>d</var><sup>{order}</sup></span>'
                f'<span class="frac-den"><var>d</var>{var_html}<sup>{order}</sup></span>'
                f'</span>'
            )

        html = (
            f'<div class="eq">{name_html}'
            f'{deriv_sym}'
            f'({expr_html})'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}d{'²' if order==2 else '³' if order==3 else ''}/d{var}{'²' if order==2 else '³' if order==3 else ''}({expr}) = {result}")

    return result


def sym_partial(expr, *vars_orders, name: str = None):
    """Compute and display partial derivatives.

    Args:
        expr: SymPy expression
        *vars_orders: Variables and orders, e.g., (x, 1, y, 1) or just (x, y)
        name: Optional label

    Examples:
        sym_partial(x**2*y + y**3, x)              # ∂/∂x(x²y + y³) = 2xy
        sym_partial(x**2*y + y**3, x, y)           # ∂²/∂x∂y(...) = 2x
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.diff(expr, *vars_orders)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result)

        # Determine variable list
        vars_list = [v for v in vars_orders if isinstance(v, sp.Symbol)]
        total_order = len(vars_list) if vars_list else 1
        vars_str = ''.join(f'&#8706;{_sympy_to_html(v)}' for v in vars_list)

        if total_order == 1:
            var_html = _sympy_to_html(vars_list[0]) if vars_list else '?'
            deriv_sym = (
                f'<span class="frac">'
                f'<span class="frac-num">&#8706;</span>'
                f'<span class="frac-den">&#8706;{var_html}</span>'
                f'</span>'
            )
        else:
            deriv_sym = (
                f'<span class="frac">'
                f'<span class="frac-num">&#8706;<sup>{total_order}</sup></span>'
                f'<span class="frac-den">{vars_str}</span>'
                f'</span>'
            )

        html = (
            f'<div class="eq">{name_html}'
            f'{deriv_sym}'
            f'({expr_html})'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}partial({expr}, {vars_orders}) = {result}")

    return result


def sym_limit(expr, var, to, direction: str = None, name: str = None):
    """Compute and display a symbolic limit.

    Args:
        expr: SymPy expression
        var: Variable approaching the limit
        to: The value the variable approaches
        direction: '+' for right, '-' for left, None for both
        name: Optional label

    Examples:
        sym_limit(sin(x)/x, x, 0)              # lim x→0 sin(x)/x = 1
        sym_limit(1/x, x, 0, '+')              # lim x→0⁺ 1/x = ∞
    """
    import sympy as sp
    mode = _get_mode()

    kwargs = {}
    if direction == '+':
        kwargs['dir'] = '+'
    elif direction == '-':
        kwargs['dir'] = '-'

    result = sp.limit(expr, var, to, **kwargs)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        var_html = _sympy_to_html(var)
        to_html = _sympy_to_html(sp.sympify(to))
        result_html = _sympy_to_html(result)

        dir_html = ''
        if direction == '+':
            dir_html = '<sup>+</sup>'
        elif direction == '-':
            dir_html = '<sup>&minus;</sup>'

        lim_sym = (
            f'<span class="nary-wrap">'
            f'<span class="nary" style="font-size:1em">lim</span>'
            f'<span class="nary-sub">{var_html} &#8594; {to_html}{dir_html}</span>'
            f'</span>'
        )

        html = (
            f'<div class="eq">{name_html}'
            f'{lim_sym}\u2009'
            f'{expr_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        d = f"({direction})" if direction else ""
        print(f"{prefix}lim({var}->{to}{d}) {expr} = {result}")

    return result


def sym_solve(expr, var, name: str = None):
    """Solve an equation symbolically and display the result.

    Args:
        expr: SymPy expression (assumed = 0) or Eq object
        var: Variable to solve for
        name: Optional label

    Examples:
        sym_solve(x**2 - 4, x)                 # x² - 4 = 0 → x = {-2, 2}
        sym_solve(Eq(x**2, 9), x)              # x² = 9 → x = {-3, 3}
    """
    import sympy as sp
    mode = _get_mode()

    solutions = sp.solve(expr, var)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''

        if isinstance(expr, sp.Eq):
            lhs_html = _sympy_to_html(expr.lhs)
            rhs_html = _sympy_to_html(expr.rhs)
            eq_html = f'{lhs_html} = {rhs_html}'
        else:
            eq_html = f'{_sympy_to_html(expr)} = 0'

        var_html = _sympy_to_html(var)

        if len(solutions) == 1:
            sol_html = _sympy_to_html(solutions[0])
            result_str = f'{var_html} = {sol_html}'
        elif len(solutions) > 1:
            sols = ', '.join(_sympy_to_html(s) for s in solutions)
            result_str = f'{var_html} &#8712; {{{sols}}}'
        else:
            result_str = 'sin soluci&oacute;n'

        html = (
            f'<div class="eq">{name_html}'
            f'{eq_html}'
            f' &nbsp;&#8594;&nbsp; '
            f'{result_str}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}{expr} = 0 => {var} = {solutions}")

    return solutions


def sym_simplify(expr, name: str = None):
    """Simplify and display a symbolic expression.

    Args:
        expr: SymPy expression to simplify
        name: Optional label

    Examples:
        sym_simplify((x**2 - 1)/(x - 1))       # (x²-1)/(x-1) = x + 1
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.simplify(expr)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var>: ' if name else ''
        orig_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result)

        html = (
            f'<div class="eq">{name_html}'
            f'{orig_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name}: " if name else ""
        print(f"{prefix}{expr} = {result}")

    return result


def sym_factor(expr, name: str = None):
    """Factor and display a symbolic expression.

    Examples:
        sym_factor(x**3 - 8)                   # x³ - 8 = (x - 2)(x² + 2x + 4)
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.factor(expr)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var>: ' if name else ''
        orig_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result)

        html = (
            f'<div class="eq">{name_html}'
            f'{orig_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name}: " if name else ""
        print(f"{prefix}{expr} = {result}")

    return result


def sym_expand(expr, name: str = None):
    """Expand and display a symbolic expression.

    Examples:
        sym_expand((x + 1)**3)                 # (x+1)³ = x³ + 3x² + 3x + 1
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.expand(expr)

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var>: ' if name else ''
        orig_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result)

        html = (
            f'<div class="eq">{name_html}'
            f'{orig_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name}: " if name else ""
        print(f"{prefix}{expr} = {result}")

    return result


def sym_series(expr, var, x0=0, n: int = 6, name: str = None):
    """Compute and display a Taylor series expansion.

    Args:
        expr: SymPy expression
        var: Variable
        x0: Point of expansion (default: 0)
        n: Number of terms
        name: Optional label

    Examples:
        sym_series(exp(x), x, n=5)            # eˣ = 1 + x + x²/2 + x³/6 + ...
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.series(expr, var, x0, n)
    # Remove O() term for display
    result_no_O = result.removeO()

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result_no_O)

        around = f' (around {_sympy_to_html(sp.sympify(x0))})' if x0 != 0 else ''

        html = (
            f'<div class="eq">{name_html}'
            f'{expr_html}{around}'
            f' &#8776; {result_html} + &#8943;</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}{expr} ~= {result_no_O} + ...")

    return result


def sym_matrix(data, name: str = None, det: bool = False,
               inv: bool = False, eigenvals: bool = False, rank: bool = False):
    """Display a symbolic matrix with optional computed properties.

    Args:
        data: 2D list or SymPy Matrix
        name: Matrix name
        det: Show determinant
        inv: Show inverse
        eigenvals: Show eigenvalues
        rank: Show rank

    Examples:
        sym_matrix([[1, 2], [3, 4]], name="A", det=True, inv=True)
    """
    import sympy as sp
    mode = _get_mode()

    if isinstance(data, sp.MatrixBase):
        mat = data
    else:
        mat = sp.Matrix(data)

    if mode == "standalone":
        # Main matrix
        mat_html = _sympy_matrix_to_html(mat, name)
        _emit(f'<div class="eq">{mat_html}</div>')

        # Determinant
        if det:
            d = mat.det()
            d_html = _sympy_to_html(sp.simplify(d))
            name_prefix = f'det({_format_subscript(_greek(name))})' if name else 'det'
            _emit(f'<div class="eq"><var>{name_prefix}</var> = {d_html}</div>')

        # Inverse
        if inv:
            try:
                m_inv = mat.inv()
                m_inv_simplified = m_inv.applyfunc(sp.simplify)
                inv_html = _sympy_matrix_to_html(m_inv_simplified)
                name_prefix = f'{_format_subscript(_greek(name))}<sup>&minus;1</sup>' if name else 'M<sup>&minus;1</sup>'
                _emit(f'<div class="eq"><var>{name_prefix}</var> = {inv_html}</div>')
            except Exception:
                _emit('<div class="eq" style="color:#c62828">Matriz singular — no invertible</div>')

        # Eigenvalues
        if eigenvals:
            evals = mat.eigenvals()
            evals_parts = []
            for val, mult in evals.items():
                v_html = _sympy_to_html(sp.simplify(val))
                if mult > 1:
                    evals_parts.append(f'{v_html} <small>(mult. {mult})</small>')
                else:
                    evals_parts.append(v_html)
            evals_str = ', '.join(evals_parts)
            _emit(f'<div class="eq">&lambda; = {{{evals_str}}}</div>')

        # Rank
        if rank:
            r = mat.rank()
            _emit(f'<div class="eq">rango = {r}</div>')

    elif mode == "console":
        print(f"{name} = " if name else "", mat)
        if det:
            print(f"det = {mat.det()}")
        if inv:
            try:
                print(f"inv = {mat.inv()}")
            except Exception:
                print("Singular — no inverse")

    return mat


def sym_eq(name: str, expr, unit: str = None):
    """Display a named symbolic expression.

    Args:
        name: Variable/result name
        expr: SymPy expression
        unit: Optional unit string

    Examples:
        sym_eq("sigma_max", M*c/I, "MPa")
    """
    import sympy as sp
    mode = _get_mode()

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var>'
        expr_html = _sympy_to_html(expr)
        unit_html = ''
        if unit:
            from hekatan.display import _format_unit
            unit_html = f'\u2009<span class="unit">{_format_unit(unit)}</span>'

        html = f'<div class="eq">{name_html} = {expr_html}{unit_html}</div>'
        _emit(html)
    elif mode == "console":
        u = f" [{unit}]" if unit else ""
        print(f"{name} = {expr}{u}")


def sym_subs(expr, subs_dict: dict, name: str = None, unit: str = None):
    """Substitute values into a symbolic expression and display step-by-step.

    Shows: expression → substituted → numeric result

    Args:
        expr: SymPy expression
        subs_dict: Dict of {symbol: value} substitutions
        name: Optional label
        unit: Optional unit

    Examples:
        sym_subs(a**2 + b**2, {a: 3, b: 4}, name="c_sq")
        # c² = a² + b² = 3² + 4² = 25
    """
    import sympy as sp
    mode = _get_mode()

    # Compute substituted expression
    substituted = expr.subs(subs_dict)
    # Try to evaluate numerically
    try:
        numeric = float(substituted)
        if numeric == int(numeric):
            numeric = int(numeric)
    except (TypeError, ValueError):
        numeric = None

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)

        # Build substitution step: replace symbols with their numeric values
        subs_expr = expr
        for sym, val in subs_dict.items():
            subs_expr = subs_expr.subs(sym, val)
        subs_html = _sympy_to_html(subs_expr)

        unit_html = ''
        if unit:
            from hekatan.display import _format_unit
            unit_html = f'\u2009<span class="unit">{_format_unit(unit)}</span>'

        if numeric is not None and numeric != substituted:
            result_html = f'{numeric:g}' if isinstance(numeric, float) else str(numeric)
            html = (
                f'<div class="eq">{name_html}'
                f'{expr_html} = {subs_html} = {result_html}{unit_html}</div>'
            )
        else:
            result_html = _sympy_to_html(sp.simplify(substituted))
            html = (
                f'<div class="eq">{name_html}'
                f'{expr_html} = {result_html}{unit_html}</div>'
            )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        u = f" [{unit}]" if unit else ""
        print(f"{prefix}{expr} = {substituted}{u}")

    return substituted


def sym_summation(expr, var, lower, upper, name: str = None):
    """Compute and display a symbolic summation.

    Args:
        expr: SymPy expression (summand)
        var: Index variable
        lower: Lower bound
        upper: Upper bound
        name: Optional label

    Examples:
        sym_summation(k**2, k, 1, n)           # Σ k² from 1 to n = n(n+1)(2n+1)/6
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.summation(expr, (var, lower, upper))

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        var_html = _sympy_to_html(var)
        lower_html = _sympy_to_html(sp.sympify(lower))
        upper_html = _sympy_to_html(sp.sympify(upper))
        result_html = _sympy_to_html(result)

        sum_sym = (
            f'<span class="nary-wrap">'
            f'<span class="nary-sup">{upper_html}</span>'
            f'<span class="nary">&#8721;</span>'
            f'<span class="nary-sub">{var_html} = {lower_html}</span>'
            f'</span>'
        )

        html = (
            f'<div class="eq">{name_html}'
            f'{sum_sym}\u2009'
            f'{expr_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}sum({expr}, {var}={lower}..{upper}) = {result}")

    return result


def sym_product(expr, var, lower, upper, name: str = None):
    """Compute and display a symbolic product.

    Args:
        expr: SymPy expression (factor)
        var: Index variable
        lower: Lower bound
        upper: Upper bound
        name: Optional label
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.product(expr, (var, lower, upper))

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        var_html = _sympy_to_html(var)
        lower_html = _sympy_to_html(sp.sympify(lower))
        upper_html = _sympy_to_html(sp.sympify(upper))
        result_html = _sympy_to_html(result)

        prod_sym = (
            f'<span class="nary-wrap">'
            f'<span class="nary-sup">{upper_html}</span>'
            f'<span class="nary">&#8719;</span>'
            f'<span class="nary-sub">{var_html} = {lower_html}</span>'
            f'</span>'
        )

        html = (
            f'<div class="eq">{name_html}'
            f'{prod_sym}\u2009'
            f'{expr_html}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}product({expr}, {var}={lower}..{upper}) = {result}")

    return result


def sym_double_integral(expr, var1, lower1, upper1, var2, lower2, upper2, name: str = None):
    """Compute and display a symbolic double integral.

    Args:
        expr: Integrand
        var1: Inner variable
        lower1, upper1: Inner limits
        var2: Outer variable
        lower2, upper2: Outer limits
        name: Optional label
    """
    import sympy as sp
    mode = _get_mode()

    result = sp.integrate(expr, (var1, lower1, upper1), (var2, lower2, upper2))

    if mode == "standalone":
        name_html = f'<var>{_format_subscript(_greek(name))}</var> = ' if name else ''
        expr_html = _sympy_to_html(expr)
        result_html = _sympy_to_html(result)

        # Outer integral
        outer = (
            f'<span class="nary-wrap">'
            f'<span class="nary-sup">{_sympy_to_html(sp.sympify(upper2))}</span>'
            f'<span class="nary">&#8747;</span>'
            f'<span class="nary-sub">{_sympy_to_html(sp.sympify(lower2))}</span>'
            f'</span>'
        )
        # Inner integral
        inner = (
            f'<span class="nary-wrap">'
            f'<span class="nary-sup">{_sympy_to_html(sp.sympify(upper1))}</span>'
            f'<span class="nary">&#8747;</span>'
            f'<span class="nary-sub">{_sympy_to_html(sp.sympify(lower1))}</span>'
            f'</span>'
        )

        html = (
            f'<div class="eq">{name_html}'
            f'{outer}{inner}\u2009'
            f'{expr_html}\u2009'
            f'<var>d</var>{_sympy_to_html(var1)}\u2009'
            f'<var>d</var>{_sympy_to_html(var2)}'
            f' = {result_html}</div>'
        )
        _emit(html)
    elif mode == "console":
        prefix = f"{name} = " if name else ""
        print(f"{prefix}double_integral({expr}, d{var1}d{var2}) = {result}")

    return result


# ============================================================
# Convenience: re-export sympy symbols for easier imports
# ============================================================

def symbols(names: str):
    """Re-export sympy.symbols for convenience."""
    import sympy as sp
    return sp.symbols(names)
