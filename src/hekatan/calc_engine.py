"""
Hekatan calc engine — natural math expression evaluator.

Provides the calc() function for evaluating math expressions naturally:

    from hekatan import calc, show

    calc("a = 3")
    calc("b = 4")
    calc("c = sqrt(a^2 + b^2)")   # → c = sqrt(3² + 4²) = 5
    calc("f(x) = x^2 + 1")
    calc("f(3)")                   # → f(3) = 10

    show()

Zero dependencies — uses only Python's math module.
"""

import math
import re
import enum
from typing import List, Optional, Tuple


# ============================================================
# Token types and tokenizer
# ============================================================

class TokenType(enum.Enum):
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    CARET = "CARET"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    EQUALS = "EQUALS"
    EOF = "EOF"


class Token:
    __slots__ = ("type", "value")

    def __init__(self, type_: TokenType, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


_NUM_RE = re.compile(r'\d+(\.\d+)?([eE][+-]?\d+)?')
_IDENT_RE = re.compile(r'[a-zA-Z_\u03B1-\u03C9\u0391-\u03A9][a-zA-Z0-9_]*')


def tokenize(expr: str) -> List[Token]:
    """Tokenize a math expression string."""
    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        # Skip whitespace
        if ch in ' \t':
            i += 1
            continue
        # Number
        m = _NUM_RE.match(expr, i)
        if m:
            tokens.append(Token(TokenType.NUMBER, m.group()))
            i = m.end()
            continue
        # Identifier
        m = _IDENT_RE.match(expr, i)
        if m:
            tokens.append(Token(TokenType.IDENT, m.group()))
            i = m.end()
            continue
        # Operators
        if ch == '+':
            tokens.append(Token(TokenType.PLUS, '+'))
        elif ch == '-':
            tokens.append(Token(TokenType.MINUS, '-'))
        elif ch == '*':
            tokens.append(Token(TokenType.STAR, '*'))
        elif ch == '/':
            tokens.append(Token(TokenType.SLASH, '/'))
        elif ch == '^':
            tokens.append(Token(TokenType.CARET, '^'))
        elif ch == '(':
            tokens.append(Token(TokenType.LPAREN, '('))
        elif ch == ')':
            tokens.append(Token(TokenType.RPAREN, ')'))
        elif ch == ',':
            tokens.append(Token(TokenType.COMMA, ','))
        elif ch == '=':
            tokens.append(Token(TokenType.EQUALS, '='))
        else:
            raise CalcError(f"Caracter inesperado: '{ch}'")
        i += 1
    tokens.append(Token(TokenType.EOF, None))
    return tokens


# ============================================================
# AST Nodes
# ============================================================

class Node:
    pass


class NumberNode(Node):
    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = value


class IdentNode(Node):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class BinOpNode(Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right


class UnaryNode(Node):
    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Node):
        self.op = op
        self.operand = operand


class CallNode(Node):
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: List[Node]):
        self.name = name
        self.args = args


class AssignNode(Node):
    __slots__ = ("name", "rhs_str", "expr")

    def __init__(self, name: str, rhs_str: str, expr: Node):
        self.name = name
        self.rhs_str = rhs_str  # original right-hand side string
        self.expr = expr


class FuncDefNode(Node):
    __slots__ = ("name", "params", "body_str", "body")

    def __init__(self, name: str, params: List[str], body_str: str, body: Node):
        self.name = name
        self.params = params
        self.body_str = body_str  # original body string
        self.body = body


# ============================================================
# Parser — recursive descent
# ============================================================

class Parser:
    """Recursive descent parser for math expressions."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: TokenType) -> Token:
        tok = self.advance()
        if tok.type != type_:
            raise CalcError(f"Se esperaba {type_.name}, se encontro {tok.type.name}")
        return tok

    def save(self) -> int:
        return self.pos

    def restore(self, pos: int):
        self.pos = pos

    # ---- Statement parsing ----

    def parse_statement(self, original_expr: str) -> Node:
        """Parse a complete statement — funcdef, assignment, or expression."""

        # Try function definition: ident(ident, ...) = expr
        if (self.peek().type == TokenType.IDENT and
                self.pos + 1 < len(self.tokens) and
                self.tokens[self.pos + 1].type == TokenType.LPAREN):
            saved = self.save()
            try:
                node = self._try_funcdef(original_expr)
                if node is not None:
                    return node
            except CalcError:
                pass
            self.restore(saved)

        # Try assignment: ident = expr
        if (self.peek().type == TokenType.IDENT and
                self.pos + 1 < len(self.tokens) and
                self.tokens[self.pos + 1].type == TokenType.EQUALS):
            name = self.advance().value  # ident
            self.advance()  # =
            # rhs string = everything after the first '='
            eq_idx = original_expr.index('=')
            rhs_str = original_expr[eq_idx + 1:].strip()
            expr = self.parse_expr()
            self.expect(TokenType.EOF)
            return AssignNode(name, rhs_str, expr)

        # Expression evaluation
        expr = self.parse_expr()
        self.expect(TokenType.EOF)
        return expr

    def _try_funcdef(self, original_expr: str) -> Optional[FuncDefNode]:
        """Try to parse function definition: f(x, y) = expr"""
        name = self.advance().value  # function name
        self.expect(TokenType.LPAREN)

        # Check if all items inside parens are plain IDENTs (not expressions)
        params = []
        if self.peek().type == TokenType.IDENT:
            params.append(self.advance().value)
            while self.peek().type == TokenType.COMMA:
                self.advance()  # skip comma
                params.append(self.expect(TokenType.IDENT).value)

        if self.peek().type != TokenType.RPAREN:
            return None
        self.advance()  # )

        if self.peek().type != TokenType.EQUALS:
            return None
        self.advance()  # =

        # Body string = everything after the second '=' occurrence
        # Actually: after "f(x) ="
        pattern = re.escape(name) + r'\s*\([^)]*\)\s*=\s*'
        m = re.match(pattern, original_expr)
        if m:
            body_str = original_expr[m.end():].strip()
        else:
            eq_idx = original_expr.index('=')
            body_str = original_expr[eq_idx + 1:].strip()

        body = self.parse_expr()
        self.expect(TokenType.EOF)
        return FuncDefNode(name, params, body_str, body)

    # ---- Expression parsing (precedence climbing) ----

    def parse_expr(self) -> Node:
        """expr := term (('+' | '-') term)*"""
        left = self.parse_term()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_term()
            left = BinOpNode(op, left, right)
        return left

    def parse_term(self) -> Node:
        """term := power (('*' | '/') power)*"""
        left = self.parse_power()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH):
            op = self.advance().value
            right = self.parse_power()
            left = BinOpNode(op, left, right)
        return left

    def parse_power(self) -> Node:
        """power := unary ('^' power)?  (right-associative)"""
        base = self.parse_unary()
        if self.peek().type == TokenType.CARET:
            self.advance()  # ^
            exp = self.parse_power()  # right-associative
            return BinOpNode('^', base, exp)
        return base

    def parse_unary(self) -> Node:
        """unary := '-' unary | atom"""
        if self.peek().type == TokenType.MINUS:
            self.advance()
            operand = self.parse_unary()
            return UnaryNode('-', operand)
        return self.parse_atom()

    def parse_atom(self) -> Node:
        """atom := NUMBER | ident '(' args ')' | ident | '(' expr ')'"""
        tok = self.peek()

        # Number
        if tok.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(float(tok.value))

        # Identifier (variable or function call)
        if tok.type == TokenType.IDENT:
            self.advance()
            name = tok.value
            # Check for function call
            if self.peek().type == TokenType.LPAREN:
                self.advance()  # (
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.parse_expr())
                    while self.peek().type == TokenType.COMMA:
                        self.advance()  # ,
                        args.append(self.parse_expr())
                self.expect(TokenType.RPAREN)
                return CallNode(name, args)
            return IdentNode(name)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.advance()  # (
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        raise CalcError(f"Expresion inesperada: {tok}")


# ============================================================
# Symbol table, builtins, user functions
# ============================================================

_SYMBOL_TABLE = {
    "pi": math.pi,
    "e": math.e,
}

_FUNCTIONS = {}  # name -> FuncDef

_BUILTINS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log10,
    "ln": math.log,
    "abs": abs,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}


class FuncDef:
    """User-defined function."""
    __slots__ = ("name", "params", "body_str", "body_ast")

    def __init__(self, name: str, params: List[str], body_str: str, body_ast: Node):
        self.name = name
        self.params = params
        self.body_str = body_str
        self.body_ast = body_ast


# ============================================================
# Evaluator
# ============================================================

def _evaluate(node: Node, local_scope: Optional[dict] = None) -> float:
    """Evaluate an AST node to produce a numeric result."""

    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, IdentNode):
        if local_scope and node.name in local_scope:
            return local_scope[node.name]
        if node.name in _SYMBOL_TABLE:
            return _SYMBOL_TABLE[node.name]
        raise CalcError(f"Variable no definida: {node.name}")

    if isinstance(node, BinOpNode):
        left = _evaluate(node.left, local_scope)
        right = _evaluate(node.right, local_scope)
        if node.op == '+':
            return left + right
        if node.op == '-':
            return left - right
        if node.op == '*':
            return left * right
        if node.op == '/':
            if right == 0:
                raise CalcError("Division por cero")
            return left / right
        if node.op == '^':
            return left ** right

    if isinstance(node, UnaryNode):
        val = _evaluate(node.operand, local_scope)
        if node.op == '-':
            return -val

    if isinstance(node, CallNode):
        args = [_evaluate(a, local_scope) for a in node.args]
        # Builtins first
        if node.name in _BUILTINS:
            try:
                return float(_BUILTINS[node.name](*args))
            except (ValueError, OverflowError) as ex:
                raise CalcError(f"Error en {node.name}(): {ex}")
        # User functions
        if node.name in _FUNCTIONS:
            func_def = _FUNCTIONS[node.name]
            if len(args) != len(func_def.params):
                raise CalcError(
                    f"{node.name}() espera {len(func_def.params)} arg(s), recibio {len(args)}"
                )
            local = dict(zip(func_def.params, args))
            return _evaluate(func_def.body_ast, local)
        raise CalcError(f"Funcion no definida: {node.name}()")

    raise CalcError(f"Nodo desconocido: {type(node).__name__}")


# ============================================================
# Display helpers
# ============================================================

def _format_number(value: float) -> str:
    """Format a number for display — clean integers, reasonable decimals."""
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if math.isnan(value):
        return "NaN"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


def _substitute_display(rhs_str: str, local_scope: Optional[dict] = None) -> str:
    """Replace known variable names with their numeric values in a string.

    Used to show intermediate step: sqrt(a^2 + b^2) → sqrt(3^2 + 4^2)
    """
    result = rhs_str
    # Collect all replacements (variable name → formatted value)
    replacements = []
    for tok in tokenize(rhs_str):
        if tok.type == TokenType.IDENT:
            name = tok.value
            # Check local scope first (function params), then global
            if local_scope and name in local_scope:
                replacements.append((name, _format_number(local_scope[name])))
            elif name in _SYMBOL_TABLE and name not in ('pi', 'e'):
                replacements.append((name, _format_number(_SYMBOL_TABLE[name])))
    # Skip builtins and function names
    skip = set(_BUILTINS.keys()) | set(_FUNCTIONS.keys())
    replacements = [(n, v) for n, v in replacements if n not in skip]
    # Sort by length descending to avoid partial replacement
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    # Deduplicate
    seen = set()
    unique = []
    for n, v in replacements:
        if n not in seen:
            seen.add(n)
            unique.append((n, v))
    for name, val_str in unique:
        result = re.sub(r'\b' + re.escape(name) + r'\b', val_str, result)
    return result


# ============================================================
# AST → HTML renderer (fractions, superscripts, etc.)
# ============================================================

def _ast_to_html(node: Node, parent_op: str = "") -> str:
    """Render an AST node to formatted HTML with fractions, superscripts, etc."""
    from hekatan.display import _format_subscript, _greek

    if isinstance(node, NumberNode):
        v = node.value
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"

    if isinstance(node, IdentNode):
        return f"<var>{_format_subscript(_greek(node.name))}</var>"

    if isinstance(node, UnaryNode):
        if node.op == '-':
            inner = _ast_to_html(node.operand, '-')
            return f"−{inner}"

    if isinstance(node, BinOpNode):
        # Division → fraction with dvc/dvl
        if node.op == '/':
            num_html = _ast_to_html(node.left, '/')
            den_html = _ast_to_html(node.right, '/')
            return (
                f'<span class="dvc">'
                f'<span class="dvl">{num_html}</span>'
                f'<span class="dvl">{den_html}</span>'
                f'</span>'
            )

        # Power → superscript
        if node.op == '^':
            base_html = _ast_to_html(node.left, '^')
            exp_html = _ast_to_html(node.right, '^')
            # If base needs parentheses (e.g. (a+b)^2)
            if isinstance(node.left, BinOpNode) and node.left.op in ('+', '-', '*', '/'):
                base_html = f"({base_html})"
            return f"{base_html}<sup>{exp_html}</sup>"

        # Multiplication
        if node.op == '*':
            left_html = _ast_to_html(node.left, '*')
            right_html = _ast_to_html(node.right, '*')
            # Parentheses for lower-precedence ops inside *
            if isinstance(node.left, BinOpNode) and node.left.op in ('+', '-'):
                left_html = f"({left_html})"
            if isinstance(node.right, BinOpNode) and node.right.op in ('+', '-'):
                right_html = f"({right_html})"

            # Implicit multiplication (no dot): number × variable/call/power
            # Examples: 3x² → "3x²", 2sin(x) → "2sin(x)"
            def _is_var_like(n):
                """Check if node is a variable, call, or power of variable."""
                if isinstance(n, IdentNode):
                    return True
                if isinstance(n, CallNode):
                    return True
                if isinstance(n, BinOpNode) and n.op == '^':
                    return _is_var_like(n.left)
                if isinstance(n, BinOpNode) and n.op == '*':
                    return _is_var_like(n.left)
                return False

            if isinstance(node.left, NumberNode) and _is_var_like(node.right):
                return f"{left_html}{right_html}"
            # Variable × Variable: use thin space (a b → a b, not a·b)
            if _is_var_like(node.left) and _is_var_like(node.right):
                return f"{left_html}\u2009{right_html}"
            return f"{left_html} · {right_html}"

        # Addition / subtraction
        if node.op in ('+', '-'):
            left_html = _ast_to_html(node.left, node.op)
            right_html = _ast_to_html(node.right, node.op)
            op_str = ' + ' if node.op == '+' else ' − '
            # Parentheses if inside * or / or ^
            needs_parens = parent_op in ('*', '^')
            if needs_parens:
                return f"({left_html}{op_str}{right_html})"
            return f"{left_html}{op_str}{right_html}"

    if isinstance(node, CallNode):
        from hekatan.display import _format_subscript, _greek
        args_html = ", ".join(_ast_to_html(a) for a in node.args)
        fname = _greek(node.name)
        # sqrt → special rendering with overline
        if node.name == "sqrt" and len(node.args) == 1:
            inner = _ast_to_html(node.args[0])
            return f'√<span class="o0">{inner}</span>'
        return f"{fname}({args_html})"

    return "?"


def _expr_to_html(expr_str: str) -> str:
    """Parse a math expression string and render it as formatted HTML."""
    try:
        tokens = tokenize(expr_str)
        parser = Parser(tokens)
        ast = parser.parse_expr()
        return _ast_to_html(ast)
    except CalcError:
        # Fallback to simple formatting
        from hekatan.display import _format_expr
        return _format_expr(expr_str)


# ============================================================
# Error class
# ============================================================

class CalcError(Exception):
    """Error in calc() — undefined variable, syntax error, etc."""
    pass


# ============================================================
# Main calc() function
# ============================================================

def calc(expression: str, unit: str = "", desc: str = "") -> Optional[float]:
    """Evaluate a math expression and display it formatted.

    Usage:
        calc("a = 3")                    # assignment → a = 3
        calc("b = 4")                    # assignment → b = 4
        calc("c = sqrt(a^2 + b^2)")     # → c = sqrt(3² + 4²) = 5
        calc("f(x) = x^2 + 1")          # function def → f(x) = x² + 1
        calc("f(3)")                     # evaluation → f(3) = 10
        calc("a + b")                    # expression → a + b = 3 + 4 = 7

    Args:
        expression: Math expression string.
        unit: Optional unit for display (e.g. "kN", "mm^2").
        desc: Optional description text.

    Returns:
        Numeric result (float), or None for function definitions.
    """
    from hekatan.display import (
        _get_mode, _emit, _dsl, _esc,
        _format_subscript, _format_expr, _greek, _format_unit,
    )

    expr_str = expression.strip()
    if not expr_str:
        return None

    try:
        tokens = tokenize(expr_str)
        parser = Parser(tokens)
        ast_node = parser.parse_statement(expr_str)

        # --- Function definition ---
        if isinstance(ast_node, FuncDefNode):
            func_def = FuncDef(
                ast_node.name, ast_node.params,
                ast_node.body_str, ast_node.body
            )
            _FUNCTIONS[ast_node.name] = func_def
            mode = _get_mode()
            # Display: f(x) = x^2 + 1
            params_str = ", ".join(ast_node.params)
            lhs = f"{ast_node.name}({params_str})"
            rhs = ast_node.body_str
            if mode == "hekatan":
                _dsl(f"@@calc_def {_esc(lhs)}|{_esc(rhs)}")
            elif mode == "standalone":
                lhs_html = _format_subscript(lhs)
                rhs_html = _expr_to_html(rhs)
                _emit(
                    f'<div class="eq"><var>{lhs_html}</var>'
                    f' = {rhs_html}</div>'
                )
            else:
                print(f"{lhs} = {rhs}")
            return None

        # --- Assignment ---
        if isinstance(ast_node, AssignNode):
            value = _evaluate(ast_node.expr)
            _SYMBOL_TABLE[ast_node.name] = value
            mode = _get_mode()
            rhs = ast_node.rhs_str
            substituted = _substitute_display(rhs)
            val_str = _format_number(value)
            name = ast_node.name

            if mode == "hekatan":
                _dsl(f"@@calc {_esc(name)}|{_esc(rhs)}|{_esc(substituted)}|{_esc(val_str)}|{_esc(unit)}")
            elif mode == "standalone":
                name_html = f'<var>{_format_subscript(name)}</var>'
                rhs_html = _expr_to_html(rhs)
                val_html = val_str
                unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""
                desc_html = f'<span class="desc">{desc}</span> ' if desc else ""

                # Show intermediate step only if substitution differs
                if substituted != rhs and substituted != val_str:
                    sub_html = _expr_to_html(substituted)
                    _emit(
                        f'<div class="eq">{desc_html}'
                        f'{name_html} = {rhs_html} = {sub_html}'
                        f' = <b>{val_html}</b>{unit_html}</div>'
                    )
                elif rhs != val_str:
                    _emit(
                        f'<div class="eq">{desc_html}'
                        f'{name_html} = {rhs_html}'
                        f' = <b>{val_html}</b>{unit_html}</div>'
                    )
                else:
                    _emit(
                        f'<div class="eq">{desc_html}'
                        f'{name_html} = <b>{val_html}</b>{unit_html}</div>'
                    )
            else:
                u = f" {unit}" if unit else ""
                d = f" - {desc}" if desc else ""
                if substituted != rhs and substituted != val_str:
                    print(f"{name} = {rhs} = {substituted} = {val_str}{u}{d}")
                elif rhs != val_str:
                    print(f"{name} = {rhs} = {val_str}{u}{d}")
                else:
                    print(f"{name} = {val_str}{u}{d}")
            return value

        # --- Expression evaluation ---
        value = _evaluate(ast_node)
        mode = _get_mode()
        val_str = _format_number(value)
        substituted = _substitute_display(expr_str)

        if mode == "hekatan":
            _dsl(f"@@calc_eval {_esc(expr_str)}|{_esc(val_str)}|{_esc(unit)}")
        elif mode == "standalone":
            expr_html = _expr_to_html(expr_str)
            unit_html = f'\u2009<i>{_format_unit(unit)}</i>' if unit else ""

            if substituted != expr_str and substituted != val_str:
                sub_html = _expr_to_html(substituted)
                _emit(
                    f'<div class="eq">{expr_html} = {sub_html}'
                    f' = <b>{val_str}</b>{unit_html}</div>'
                )
            elif expr_str != val_str:
                _emit(
                    f'<div class="eq">{expr_html}'
                    f' = <b>{val_str}</b>{unit_html}</div>'
                )
            else:
                _emit(f'<div class="eq"><b>{val_str}</b>{unit_html}</div>')
        else:
            u = f" {unit}" if unit else ""
            if substituted != expr_str and substituted != val_str:
                print(f"{expr_str} = {substituted} = {val_str}{u}")
            elif expr_str != val_str:
                print(f"{expr_str} = {val_str}{u}")
            else:
                print(f"{val_str}{u}")
        return value

    except CalcError as ex:
        mode = _get_mode()
        if mode == "hekatan":
            _dsl(f"@@calc_error {_esc(expr_str)}|{_esc(str(ex))}")
        elif mode == "standalone":
            _emit(
                f'<div class="eq" style="color:#c00">'
                f'<b>Error:</b> {expr_str} — {ex}</div>'
            )
        else:
            print(f"Error: {expr_str} — {ex}")
        return None


def calc_clear():
    """Clear the symbol table and user functions."""
    _SYMBOL_TABLE.clear()
    _FUNCTIONS.clear()
    _SYMBOL_TABLE["pi"] = math.pi
    _SYMBOL_TABLE["e"] = math.e


# ============================================================
# run() — preprocessor for IDLE / scripts
# ============================================================

def run(source: Optional[str] = None):
    """Run a hekatan script with natural math syntax.

    Can be used in three ways:

    1. Auto-detect calling file (IDLE / script):
        from hekatan import run
        title("Mi Calculo")
        b = 350      # ← this line is ignored by Python,
        h = 560      #   run() re-reads the file and processes it
        A = b*h
        run()        # ← reads THIS file, preprocesses, and executes

    2. Inline multiline string:
        from hekatan import run
        run('''
        title("Mi Calculo")
        b = 350
        h = 560
        A = b*h
        I = b*h^3/12
        show()
        ''')

    3. File path:
        from hekatan import run
        run("calculo.hkpy")
    """
    import inspect
    import os

    # Prevent recursion: if run() is called inside a preprocessed execution, skip
    frame = inspect.currentframe()
    try:
        caller_globals = frame.f_back.f_globals
        if caller_globals.get('__run_active__'):
            return  # Already inside a run() execution — skip
    finally:
        del frame

    # Import the preprocessor
    from hekatan.__main__ import preprocess

    if source is None:
        # Auto-detect: read the file that called run()
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            caller_file = caller_frame.f_globals.get('__file__')
            if caller_file is None:
                # Running from interactive interpreter (IDLE shell, python -i)
                # Can't auto-detect — tell the user
                print("Error: run() sin argumentos solo funciona en un archivo .py")
                print("Uso: run('archivo.hkpy') o run('''codigo...''')")
                return
            caller_file = os.path.abspath(caller_file)
        finally:
            del frame

        if not os.path.isfile(caller_file):
            print(f"Error: no se encontro el archivo: {caller_file}")
            return

        with open(caller_file, 'r', encoding='utf-8') as f:
            file_source = f.read()

        # Preprocess and execute
        python_code = preprocess(file_source)
        namespace = {
            '__file__': caller_file,
            '__name__': '__run__',  # Prevent re-triggering if __name__ == '__main__'
            '__run_active__': True,
        }
        try:
            exec(compile(python_code, caller_file, 'exec'), namespace)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error: {e}")
        # Exit after run to prevent the rest of the original file from executing again
        raise SystemExit(0)

    elif os.path.isfile(source):
        # File path provided
        with open(source, 'r', encoding='utf-8') as f:
            file_source = f.read()

        python_code = preprocess(file_source)
        namespace = {
            '__file__': os.path.abspath(source),
            '__name__': '__run__',
            '__run_active__': True,
        }
        try:
            exec(compile(python_code, source, 'exec'), namespace)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error: {e}")

    else:
        # Inline source string
        python_code = preprocess(source)
        namespace = {
            '__name__': '__run__',
            '__run_active__': True,
        }
        try:
            exec(compile(python_code, '<run>', 'exec'), namespace)
        except SystemExit:
            pass
        except Exception as e:
            print(f"Error: {e}")
