"""
Réplica del paper JCE-70-2018-1-3-2078-EN
"Method of incompatible modes – overview and application"
Kožar, Rukavina, Ibrahimbegović — GRAĐEVINAR 70 (2018) 1, 19-29

Traducido al español, usando pyhekatan.
Enfoque: ecuaciones (1)-(69), tablas, figuras esquemáticas.
"""
import sys, os, tempfile, webbrowser, re
from typing import Optional, List, Any

# ============================================================
# HTML Buffer
# ============================================================
_BUFFER = []
_PAPER = {}

def _emit(html: str):
    _BUFFER.append(html)

# ============================================================
# Greek letters & formatting
# ============================================================
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "nabla": "∇", "partial": "∂", "infty": "∞", "infinity": "∞",
}

def _greek(text):
    """Replace Greek letter names with symbols — only standalone words, not inside other words."""
    if not text: return text
    for name, sym in sorted(_GREEK.items(), key=lambda x: -len(x[0])):
        # Use word-boundary regex so "mu" doesn't match inside "comunidad"
        text = re.sub(r'(?<![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])' + re.escape(name) + r'(?![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])', sym, text)
    return text

def _fmt_sub(name):
    if not name: return name
    name = _greek(name)
    name = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', name)
    name = re.sub(r'_([a-zA-Z0-9αβγδεζηθικλμνξπρστυφχψω])', r'<sub>\1</sub>', name)
    name = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', name)
    name = re.sub(r'\^([0-9]+)', r'<sup>\1</sup>', name)
    return name

# ============================================================
# Paper configuration
# ============================================================
def paper(size="A4", margin="20mm 18mm 25mm 18mm", font=None, fontsize="10pt",
          color="#000", accent="#000", background="#fff", lineheight=1.45,
          columngap="8mm", num_columns=1, startpage=1, pagenumber="right"):
    _PAPER.update({
        "size": size, "margin": margin, "font": font or '"Georgia", "Times New Roman", Times, serif',
        "fontsize": fontsize, "color": color, "accent": accent, "background": background,
        "lineheight": lineheight, "columngap": columngap, "columns": num_columns,
        "startpage": startpage, "pagenumber": pagenumber,
    })

# ============================================================
# Text & Headings
# ============================================================
def title(text_content, level=1):
    tag = f"h{min(level, 6)}"
    _emit(f"<{tag}>{_greek(text_content)}</{tag}>")

def heading(text_content, level=2):
    title(text_content, level)

def text(content):
    content = _greek(content)
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', content)
    _emit(f"<p>{content}</p>")

def markdown(content):
    lines = content.strip().split('\n')
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list: _emit("</ul>"); in_list = False
            _emit("<p>&nbsp;</p>")
            continue
        if stripped.startswith('# '):
            if in_list: _emit("</ul>"); in_list = False
            title(stripped[2:], 1)
        elif stripped.startswith('## '):
            if in_list: _emit("</ul>"); in_list = False
            title(stripped[3:], 2)
        elif stripped.startswith('### '):
            if in_list: _emit("</ul>"); in_list = False
            title(stripped[4:], 3)
        elif stripped.startswith('- '):
            if not in_list: _emit("<ul>"); in_list = True
            item = _greek(stripped[2:])
            item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
            item = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', item)
            _emit(f"<li>{item}</li>")
        else:
            if in_list: _emit("</ul>"); in_list = False
            text(stripped)
    if in_list: _emit("</ul>")


# ============================================================
# Equation parsing — FIXED fraction handling
# ============================================================

def _parse_fraction(expr):
    """
    Recursively parse (numerator)/(denominator) into HTML fraction divs.
    Handles nested parentheses properly.
    """
    result = []
    i = 0
    while i < len(expr):
        # Look for pattern: (...)/(...)
        if expr[i] == '(':
            # Find matching closing paren
            depth = 1
            j = i + 1
            while j < len(expr) and depth > 0:
                if expr[j] == '(': depth += 1
                elif expr[j] == ')': depth -= 1
                j += 1
            # j is now past the closing paren
            num_content = expr[i+1:j-1]

            # Check if followed by /(
            if j < len(expr) and expr[j] == '/' and j+1 < len(expr) and expr[j+1] == '(':
                # Find matching closing paren for denominator
                depth = 1
                k = j + 2
                while k < len(expr) and depth > 0:
                    if expr[k] == '(': depth += 1
                    elif expr[k] == ')': depth -= 1
                    k += 1
                den_content = expr[j+2:k-1]

                # Recursively parse both parts
                num_html = _parse_fraction(num_content)
                den_html = _parse_fraction(den_content)

                result.append(
                    f'<span class="dvc">'
                    f'<span class="dvl">{_fmt_sub(num_html)}</span>'
                    f'<span class="dvl">{_fmt_sub(den_html)}</span>'
                    f'</span>'
                )
                i = k
            else:
                # Not a fraction, just parenthesized group
                inner = _parse_fraction(num_content)
                result.append(f'({inner})')
                i = j
        else:
            result.append(expr[i])
            i += 1
    return ''.join(result)


def _parse_calc_eq(expr):
    """
    Parse Hekatan Calc-style equation syntax into HTML.
    Handles: fractions (a)/(b), integrals ∫, ∂, ∇, subscripts, superscripts, etc.
    """
    expr = _greek(expr)

    # Handle equation number: (N) at end
    eq_num = ""
    num_match = re.search(r'\s+\((\d+[a-z]?)\)\s*$', expr)
    if num_match:
        eq_num = f'<span class="eq-num">({num_match.group(1)})</span>'
        expr = expr[:num_match.start()]

    # Handle fractions with recursive parser
    expr = _parse_fraction(expr)

    # Handle integrals: ∫_{lower}^{upper}
    def replace_integral(m):
        lower = m.group(1) if m.group(1) else ""
        upper = m.group(2) if m.group(2) else ""
        lower_html = _fmt_sub(lower) if lower else ""
        upper_html = _fmt_sub(upper) if upper else ""
        if lower_html or upper_html:
            return (f'<span class="nary-wrap">'
                    f'<span class="nary-sup">{upper_html}</span>'
                    f'<span class="nary">∫</span>'
                    f'<span class="nary-sub">{lower_html}</span>'
                    f'</span>')
        return '<span class="nary">∫</span>'

    expr = re.sub(r'∫_\{([^}]*)\}\^\{([^}]*)\}', replace_integral, expr)
    expr = re.sub(r'∫_\{([^}]*)\}', lambda m: f'<span class="nary-wrap"><span class="nary-sup"></span><span class="nary">∫</span><span class="nary-sub">{_fmt_sub(m.group(1))}</span></span>', expr)
    expr = re.sub(r'∫', '<span class="nary">∫</span>', expr)

    # Summation: Σ_{lower}^{upper}
    def replace_sum(m):
        lower = m.group(1) if m.group(1) else ""
        upper = m.group(2) if m.group(2) else ""
        return (f'<span class="nary-wrap">'
                f'<span class="nary-sup">{_fmt_sub(upper)}</span>'
                f'<span class="nary" style="font-size:180%;">Σ</span>'
                f'<span class="nary-sub">{_fmt_sub(lower)}</span>'
                f'</span>')
    expr = re.sub(r'Σ_\{([^}]*)\}\^\{([^}]*)\}', replace_sum, expr)

    # ∇^4, ∇^2
    expr = re.sub(r'∇\^(\d+)', r'∇<sup>\1</sup>', expr)

    # Handle subscripts/superscripts
    expr = _fmt_sub(expr)

    # Multiplication dot
    expr = expr.replace('·', ' · ')

    return f'<div class="eq" style="justify-content:center;">{expr}{eq_num}</div>'


def eq_block(*equations):
    for equation in equations:
        html = _parse_calc_eq(equation)
        _emit(html)


# ============================================================
# Matrix display
# ============================================================
def matrix(data, name=None, pre="", post=""):
    """Display a matrix with brackets."""
    rows_html = []
    for row in data:
        cells = '<span class="td"></span>'  # left bracket
        for cell in row:
            cells += f'<span class="td">{_fmt_sub(_greek(str(cell)))}</span>'
        cells += '<span class="td"></span>'  # right bracket
        rows_html.append(f'<span class="tr">{cells}</span>')

    mat_html = f'<span class="matrix">{"".join(rows_html)}</span>'

    parts = []
    if pre:
        parts.append(_fmt_sub(_greek(pre)))
    if name:
        parts.append(f'<var>{_fmt_sub(_greek(name))}</var> = ')
    parts.append(mat_html)
    if post:
        parts.append(_fmt_sub(_greek(post)))

    _emit(f'<div class="eq" style="justify-content:center;">{"".join(parts)}</div>')


# ============================================================
# Table
# ============================================================
def table(data=None, headers=None, rows=None, caption=None):
    html = []
    if caption:
        html.append(f'<p style="font-size:9.5pt;font-weight:600;margin:10px 0 4px;">{_greek(caption)}</p>')
    html.append('<table class="hekatan-table">')
    if headers and rows:
        html.append("<tr>")
        for h in headers:
            html.append(f"<th>{_fmt_sub(_greek(str(h)))}</th>")
        html.append("</tr>")
        for row in rows:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{_fmt_sub(_greek(str(cell)))}</td>")
            html.append("</tr>")
    elif data:
        for r_idx, row in enumerate(data):
            html.append("<tr>")
            tag = "th" if r_idx == 0 else "td"
            for cell in row:
                html.append(f"<{tag}>{_fmt_sub(_greek(str(cell)))}</{tag}>")
            html.append("</tr>")
    html.append("</table>")
    _emit("".join(html))

# ============================================================
# Figure placeholder (SVG-based schematic)
# ============================================================
def figure(svg_content, caption="", number=""):
    """Display an SVG figure with caption."""
    cap = ""
    if caption:
        num_str = f"<strong>Figura {number}.</strong> " if number else ""
        cap = f'<p style="font-size:9pt;text-align:center;color:#444;margin:4px 0 12px;">{num_str}{_greek(caption)}</p>'
    _emit(f'<div style="text-align:center;margin:12px 0;">{svg_content}</div>{cap}')


# ============================================================
# Utility
# ============================================================
def hr_line():
    _emit('<hr style="margin:12px 0;border:none;border-top:1px solid #ddd;">')

def page_break(left="", right=""):
    _emit('<div style="page-break-before:always;"></div>')
    if left or right:
        _emit(f'<div style="display:flex;justify-content:space-between;font-size:8pt;'
              f'color:#666;border-bottom:1px solid #999;padding:4px 0;margin-bottom:12px;">'
              f'<span>{_greek(left)}</span><span>{_greek(right)}</span></div>')

def columns_start(n=2):
    gap = _PAPER.get("columngap", "8mm")
    _emit(f'<div style="column-count:{n};column-gap:{gap};orphans:3;widows:3;">')

def columns_end():
    _emit('</div>')

def note(content, kind="info"):
    colors = {
        "info":    ("#e3f2fd", "#1565c0", "#bbdefb", "ℹ"),
        "warning": ("#fff3e0", "#e65100", "#ffcc80", "⚠"),
        "success": ("#e8f5e9", "#2e7d32", "#a5d6a7", "✓"),
    }
    bg, fg, border, icon = colors.get(kind, colors["info"])
    _emit(f'<div style="background:{bg};color:{fg};border-left:4px solid {border};'
          f'padding:8px 12px;margin:8px 0;border-radius:4px;font-size:10pt;">'
          f'<b>{icon}</b> {_greek(content)}</div>')


# ============================================================
# SVG diagrams for figures
# ============================================================

def _svg_truss_bar():
    """Figure 1: 2-node truss bar element with shape functions."""
    return '''<svg width="500" height="200" viewBox="0 0 500 200" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Bar element -->
  <line x1="60" y1="60" x2="420" y2="60" stroke="#333" stroke-width="2"/>
  <circle cx="60" cy="60" r="5" fill="#333"/>
  <circle cx="420" cy="60" r="5" fill="#333"/>
  <text x="50" y="50" text-anchor="middle" font-style="italic">u<tspan dy="3" font-size="8">i</tspan></text>
  <text x="430" y="50" text-anchor="middle" font-style="italic">u<tspan dy="3" font-size="8">j</tspan></text>
  <text x="240" y="45" text-anchor="middle" font-style="italic">L</text>
  <!-- N1 shape function -->
  <polyline points="60,130 60,170 420,130" fill="none" stroke="#0066cc" stroke-width="1.5"/>
  <text x="40" y="175" fill="#0066cc" font-style="italic">N<tspan dy="3" font-size="8">1</tspan></text>
  <text x="55" y="128" fill="#0066cc" font-size="9">1</text>
  <!-- N2 shape function -->
  <polyline points="60,130 420,170 420,130" fill="none" stroke="#cc3300" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="430" y="175" fill="#cc3300" font-style="italic">N<tspan dy="3" font-size="8">2</tspan></text>
  <text x="425" y="128" fill="#cc3300" font-size="9">1</text>
  <!-- Axes -->
  <line x1="60" y1="130" x2="440" y2="130" stroke="#999" stroke-width="0.5"/>
  <text x="450" y="134" fill="#999" font-size="9" font-style="italic">x</text>
</svg>'''

def _svg_heterogeneous_bar():
    """Figure 2: Heterogeneous bar."""
    return '''<svg width="460" height="120" viewBox="0 0 460 120" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Support -->
  <line x1="40" y1="30" x2="40" y2="90" stroke="#333" stroke-width="2"/>
  <line x1="30" y1="30" x2="40" y2="40" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="45" x2="40" y2="55" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="60" x2="40" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="75" x2="40" y2="85" stroke="#333" stroke-width="1"/>
  <!-- E1 block -->
  <rect x="40" y="40" width="180" height="40" fill="#dde8f0" stroke="#333" stroke-width="1.5"/>
  <text x="130" y="65" text-anchor="middle" font-style="italic">E<tspan dy="3" font-size="8">1</tspan></text>
  <!-- E2 block -->
  <rect x="220" y="40" width="180" height="40" fill="#f0e8dd" stroke="#333" stroke-width="1.5"/>
  <text x="310" y="65" text-anchor="middle" font-style="italic">E<tspan dy="3" font-size="8">2</tspan></text>
  <!-- Interface line -->
  <line x1="220" y1="35" x2="220" y2="85" stroke="#c00" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="220" y="105" text-anchor="middle" fill="#c00" font-size="9">x̄ = L/2</text>
  <!-- Force arrow -->
  <line x1="400" y1="60" x2="440" y2="60" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="445" y="64" font-style="italic" font-weight="bold">F</text>
  <!-- L dimension -->
  <line x1="40" y1="20" x2="400" y2="20" stroke="#666" stroke-width="0.5"/>
  <text x="220" y="16" text-anchor="middle" font-style="italic">L</text>
  <defs><marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#333"/></marker></defs>
</svg>'''

def _svg_incomp_mode_strain():
    """Figure 3: Incompatible mode function M(x) and G(x) for strain discontinuity."""
    return '''<svg width="500" height="180" viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- M(x) -->
  <text x="20" y="15" font-style="italic" font-weight="600">M(x)</text>
  <line x1="50" y1="80" x2="450" y2="80" stroke="#999" stroke-width="0.5"/>
  <polyline points="50,80 250,30 450,80" fill="none" stroke="#0066cc" stroke-width="2"/>
  <text x="45" y="30" fill="#0066cc" font-size="9">+</text>
  <text x="245" y="25" fill="#0066cc" font-size="9">max</text>
  <!-- G(x) = dM/dx -->
  <text x="20" y="105" font-style="italic" font-weight="600">G(x)</text>
  <line x1="50" y1="140" x2="450" y2="140" stroke="#999" stroke-width="0.5"/>
  <line x1="50" y1="120" x2="250" y2="120" stroke="#cc3300" stroke-width="2"/>
  <line x1="250" y1="160" x2="450" y2="160" stroke="#cc3300" stroke-width="2"/>
  <line x1="250" y1="120" x2="250" y2="160" stroke="#cc3300" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="140" y="115" fill="#cc3300" font-size="9">+1/x̄</text>
  <text x="340" y="175" fill="#cc3300" font-size="9">−1/(L−x̄)</text>
  <!-- x axis label -->
  <text x="460" y="84" fill="#999" font-size="9" font-style="italic">x</text>
  <text x="460" y="144" fill="#999" font-size="9" font-style="italic">x</text>
</svg>'''

def _svg_incomp_mode_disp():
    """Figure 4: Incompatible mode function for displacement discontinuity."""
    return '''<svg width="500" height="180" viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- M(x) - Heaviside-like -->
  <text x="20" y="15" font-style="italic" font-weight="600">M(x)</text>
  <line x1="50" y1="80" x2="450" y2="80" stroke="#999" stroke-width="0.5"/>
  <line x1="50" y1="80" x2="250" y2="80" stroke="#0066cc" stroke-width="2"/>
  <line x1="250" y1="40" x2="450" y2="80" stroke="#0066cc" stroke-width="2"/>
  <line x1="250" y1="80" x2="250" y2="40" stroke="#0066cc" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="350" y="55" fill="#0066cc" font-size="9">Heaviside</text>
  <!-- G(x) = dM/dx -->
  <text x="20" y="105" font-style="italic" font-weight="600">G(x)</text>
  <line x1="50" y1="140" x2="450" y2="140" stroke="#999" stroke-width="0.5"/>
  <line x1="50" y1="140" x2="240" y2="140" stroke="#cc3300" stroke-width="2"/>
  <line x1="260" y1="155" x2="450" y2="155" stroke="#cc3300" stroke-width="2"/>
  <!-- Delta function -->
  <line x1="250" y1="140" x2="250" y2="110" stroke="#cc3300" stroke-width="2"/>
  <polygon points="247,112 253,112 250,108" fill="#cc3300"/>
  <text x="255" y="108" fill="#cc3300" font-size="9">δ(x−x̄)</text>
  <text x="340" y="170" fill="#cc3300" font-size="9">Ḡ = −1/L</text>
  <text x="460" y="84" fill="#999" font-size="9" font-style="italic">x</text>
  <text x="460" y="144" fill="#999" font-size="9" font-style="italic">x</text>
</svg>'''

def _svg_cohesive_law():
    """Figure 5: Cohesive law at the discontinuity (traction-separation)."""
    return '''<svg width="300" height="200" viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Axes -->
  <line x1="40" y1="170" x2="280" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="40" y1="170" x2="40" y2="20" stroke="#333" stroke-width="1"/>
  <text x="285" y="175" font-style="italic">ū</text>
  <text x="25" y="20" font-style="italic">t</text>
  <!-- Softening curve -->
  <polyline points="40,40 120,40 250,170" fill="none" stroke="#0066cc" stroke-width="2"/>
  <!-- σ_f label -->
  <line x1="35" y1="40" x2="40" y2="40" stroke="#333" stroke-width="1"/>
  <text x="10" y="44" font-style="italic" font-size="10">σ<tspan dy="3" font-size="7">f</tspan></text>
  <!-- Slope label -->
  <text x="170" y="95" fill="#c00" font-size="10" transform="rotate(-42,170,95)">pendiente = −K̄</text>
</svg>'''

def _svg_force_disp():
    """Figure 8: Force-displacement diagram with loading-unloading cycles."""
    return '''<svg width="350" height="220" viewBox="0 0 350 220" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Axes -->
  <line x1="40" y1="190" x2="330" y2="190" stroke="#333" stroke-width="1"/>
  <line x1="40" y1="190" x2="40" y2="20" stroke="#333" stroke-width="1"/>
  <text x="335" y="195" font-style="italic">ū</text>
  <text x="25" y="20" font-style="italic">F</text>
  <!-- Loading curve -->
  <polyline points="40,190 100,50 140,80" fill="none" stroke="#0066cc" stroke-width="2"/>
  <!-- Unloading 1 -->
  <polyline points="140,80 80,190" fill="none" stroke="#0066cc" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Reloading -->
  <polyline points="80,190 160,80 200,120" fill="none" stroke="#0066cc" stroke-width="2"/>
  <!-- Unloading 2 -->
  <polyline points="200,120 110,190" fill="none" stroke="#0066cc" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Continue to failure -->
  <polyline points="110,190 220,120 300,190" fill="none" stroke="#0066cc" stroke-width="2"/>
  <!-- Labels -->
  <text x="85" y="42" fill="#0066cc" font-size="9">σ<tspan dy="3" font-size="7">f</tspan></text>
  <text x="255" y="145" fill="#666" font-size="9">ablandamiento</text>
</svg>'''

def _svg_q4_element():
    """Figure 9: Isoparametric Q4 element."""
    return '''<svg width="300" height="260" viewBox="0 0 300 260" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Element -->
  <polygon points="60,200 240,210 250,50 80,40" fill="#f0f4f8" stroke="#333" stroke-width="1.5"/>
  <!-- Nodes -->
  <circle cx="60" cy="200" r="4" fill="#333"/>
  <circle cx="240" cy="210" r="4" fill="#333"/>
  <circle cx="250" cy="50" r="4" fill="#333"/>
  <circle cx="80" cy="40" r="4" fill="#333"/>
  <text x="42" y="215" font-size="10">1</text>
  <text x="248" y="228" font-size="10">2</text>
  <text x="258" y="48" font-size="10">3</text>
  <text x="62" y="35" font-size="10">4</text>
  <!-- Isoparametric coordinates -->
  <line x1="150" y1="125" x2="210" y2="125" stroke="#c00" stroke-width="1" marker-end="url(#arrowR)"/>
  <line x1="150" y1="125" x2="150" y2="65" stroke="#c00" stroke-width="1" marker-end="url(#arrowR)"/>
  <text x="215" y="130" fill="#c00" font-style="italic">ξ</text>
  <text x="140" y="60" fill="#c00" font-style="italic">η</text>
  <!-- DOF labels -->
  <text x="25" y="180" font-style="italic" font-size="10">u<tspan dy="3" font-size="7">1</tspan>,v<tspan dy="3" font-size="7">1</tspan></text>
  <defs><marker id="arrowR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c00"/></marker></defs>
</svg>'''

def _svg_cantilever_2d():
    """Figure 12: Cantilever beam with force and moment loading."""
    return '''<svg width="450" height="140" viewBox="0 0 450 140" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;font-size:11px;">
  <!-- Support -->
  <line x1="40" y1="40" x2="40" y2="100" stroke="#333" stroke-width="2"/>
  <line x1="30" y1="42" x2="40" y2="52" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="55" x2="40" y2="65" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="68" x2="40" y2="78" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="81" x2="40" y2="91" stroke="#333" stroke-width="1"/>
  <!-- Beam -->
  <rect x="40" y="55" width="350" height="30" fill="#e8e8e8" stroke="#333" stroke-width="1.5"/>
  <!-- Mesh lines -->
  <line x1="110" y1="55" x2="110" y2="85" stroke="#999" stroke-width="0.5" stroke-dasharray="2,2"/>
  <line x1="180" y1="55" x2="180" y2="85" stroke="#999" stroke-width="0.5" stroke-dasharray="2,2"/>
  <line x1="250" y1="55" x2="250" y2="85" stroke="#999" stroke-width="0.5" stroke-dasharray="2,2"/>
  <line x1="320" y1="55" x2="320" y2="85" stroke="#999" stroke-width="0.5" stroke-dasharray="2,2"/>
  <!-- Force P -->
  <line x1="400" y1="30" x2="400" y2="55" stroke="#c00" stroke-width="2" marker-end="url(#arrowF)"/>
  <text x="408" y="35" fill="#c00" font-weight="bold" font-style="italic">P</text>
  <!-- Moment M -->
  <path d="M405,100 A15,15 0 1,1 395,100" fill="none" stroke="#06c" stroke-width="1.5" marker-end="url(#arrowM)"/>
  <text x="415" y="108" fill="#06c" font-weight="bold" font-style="italic">M</text>
  <!-- Dimensions -->
  <text x="215" y="130" text-anchor="middle" font-style="italic">L = 3.0</text>
  <text x="420" y="75" font-style="italic" font-size="9">h = 0.2</text>
  <defs>
    <marker id="arrowF" markerWidth="8" markerHeight="6" refX="4" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#c00"/></marker>
    <marker id="arrowM" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#06c"/></marker>
  </defs>
</svg>'''


# ============================================================
# CSS
# ============================================================
def _build_css():
    p = _PAPER
    font = p.get("font", '"Georgia", "Times New Roman", Times, serif')
    fontsize = p.get("fontsize", "10pt")
    color = p.get("color", "#000")
    bg = p.get("background", "#fff")
    lh = p.get("lineheight", 1.45)
    margin = p.get("margin", "20mm 18mm 25mm 18mm")
    size = p.get("size", "A4")

    return f"""
@page {{ size: {size}; margin: {margin}; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: {font};
    font-size: {fontsize};
    line-height: {lh};
    color: {color};
    background: {bg};
    padding: 24px 36px;
    max-width: 960px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.5em; margin: 20px 0 8px; color: #000; border-bottom: 1.5px solid #000; padding-bottom: 4px; }}
h2 {{ font-size: 1.2em; margin: 16px 0 6px; color: #000; }}
h3 {{ font-size: 1.05em; margin: 12px 0 4px; color: #222; }}
p {{ margin: 5px 0; text-align: justify; }}
ul, ol {{ margin: 6px 0 6px 24px; }}
li {{ margin: 3px 0; }}
strong {{ font-weight: 700; }}
em {{ font-style: italic; }}

.eq, table.matrix {{ font-family: 'Georgia Pro', 'Century Schoolbook', 'Times New Roman', Times, serif; }}
.eq {{
    font-size: {fontsize}; line-height: 2.2; margin: 4px 0;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}}
.eq var {{ color: #000; font-size: 105%; font-style: italic; }}
.eq i {{ color: #000; font-style: normal; font-size: 95%; }}
.eq sub {{ font-family: {font}; font-size: 75%; vertical-align: -20%; margin-left: 1pt; }}
.eq sup {{ display: inline-block; margin-left: 1pt; font-size: 70%; }}
.eq-num {{ margin-left: auto; padding-right: 8px; font-size: 10pt; color: #444; }}
.val {{ font-weight: 600; color: #000; }}
.desc {{ font-size: 0.85em; color: #666; font-style: italic; margin-left: 4px; }}

.dvc {{ display: inline-block; vertical-align: middle; white-space: nowrap; padding: 0 2pt; text-align: center; line-height: 110%; }}
.dvl {{ display: block; border-bottom: solid 1pt black; margin: 1pt 0; padding: 1px 6px; }}
.dvl:last-child {{ border-bottom: none; }}

.nary {{ color: #000; font-size: 220%; font-family: 'Georgia Pro Light', Georgia, serif; font-weight: 200; line-height: 80%; display: block; margin: -1pt 1pt 3pt 1pt; }}
.nary-wrap {{ display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; margin: 0 2px; }}
.nary-sup {{ font-size: 70%; line-height: 1; order: -1; }}
.nary-sub {{ font-size: 70%; line-height: 1; }}

.matrix {{ display: inline-table; border-collapse: collapse; margin: 4px 2px; vertical-align: middle; }}
.matrix .tr {{ display: table-row; }}
.matrix .td {{ white-space: nowrap; padding: 0 3pt; min-width: 12pt; display: table-cell; text-align: center; font-size: 10pt; }}
.matrix .td:first-child, .matrix .td:last-child {{ width: 0.75pt; min-width: 0.75pt; max-width: 0.75pt; padding: 0 1pt; }}
.matrix .td:first-child {{ border-left: solid 1pt black; }}
.matrix .td:last-child {{ border-right: solid 1pt black; }}
.matrix .tr:first-child .td:first-child, .matrix .tr:first-child .td:last-child {{ border-top: solid 1pt black; }}
.matrix .tr:last-child .td:first-child, .matrix .tr:last-child .td:last-child {{ border-bottom: solid 1pt black; }}

.hekatan-table {{ border-collapse: collapse; margin: 10px auto; font-family: {font}; font-size: 9.5pt; }}
.hekatan-table th, .hekatan-table td {{ border: 1px solid #999; padding: 4px 10px; text-align: center; }}
.hekatan-table th {{ background: #f0f0f0; font-weight: 600; }}
.hekatan-table tr:nth-child(even) {{ background: #fafafa; }}

@media print {{
    body {{ padding: 0; max-width: none; }}
}}
"""


# ============================================================
# Show
# ============================================================
def show(filename=None):
    body = "\n".join(_BUFFER)
    css = _build_css()
    html_doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hekatan — JCE-70-2018 Modos Incompatibles</title>
<style>{css}</style>
</head>
<body>
<div class="hekatan-doc">
{body}
</div>
</body>
</html>"""
    if filename:
        path = filename
    else:
        fd, path = tempfile.mkstemp(suffix=".html", prefix="hekatan_jce_")
        os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    webbrowser.open(f"file://{os.path.abspath(path)}")
    print(f"Abierto: {path}")
    return path


# ################################################################
# DOCUMENTO: JCE-70-2018 — Método de modos incompatibles
# ################################################################

if __name__ == "__main__":

    paper(
        size="A4",
        margin="20mm 20mm 25mm 20mm",
        font='"Georgia", "Times New Roman", Times, serif',
        fontsize="10pt",
        lineheight=1.5,
        columngap="8mm",
    )

    # =========================================================
    # TÍTULO Y AUTORES
    # =========================================================
    _emit('<div style="text-align:center;margin:20px 0 16px;">')
    _emit('<p style="font-size:9pt;color:#666;">GRAĐEVINAR 70 (2018) 1, 19-29 &nbsp;·&nbsp; DOI: 10.14256/JCE.2078.2017</p>')
    _emit('<h1 style="border:none;font-size:1.6em;margin:12px 0;">Método de modos incompatibles — resumen y aplicación</h1>')
    _emit('<p style="font-size:10pt;color:#444;font-style:italic;">Ivica Kožar, Tea Rukavina, Adnan Ibrahimbegović</p>')
    _emit('<p style="font-size:9pt;color:#888;">Universidad de Rijeka · Sorbonne Universités, Compiègne</p>')
    _emit('</div>')

    hr_line()

    # Abstract
    _emit('<div style="background:#f8f8f8;padding:12px 16px;margin:8px 0;border-left:3px solid #999;font-size:9.5pt;">')
    text("**Resumen:** El método de elementos finitos se ha utilizado en la comunidad de ingeniería "
         "durante más de 50 años, período en el cual ha sido mejorado constantemente. Una mejora "
         "importante es la adición de modos de desplazamiento requeridos (\"modos incompatibles\") "
         "en las funciones de forma del elemento. Tal adición viola la condición de continuidad y "
         "debe realizarse según ciertas reglas para lograr convergencia. Los beneficios se muestran "
         "en el comportamiento del elemento bajo condiciones de carga desfavorables y en la posibilidad "
         "de un tratamiento simplificado de discontinuidades en deformación o desplazamiento.")
    _emit('</div>')

    text("**Palabras clave:** elemento finito, función de forma, análisis de flexión, análisis de discontinuidad, condensación estática")

    hr_line()

    # =========================================================
    # 1. INTRODUCCIÓN
    # =========================================================
    title("1. Introducción")

    columns_start(2)
    text("Se observó ya en las etapas iniciales de implementación que el mayor problema con los "
         "elementos finitos con bajo número de nodos es su sensibilidad al \"bloqueo\" (*locking*). "
         "El remedio se ha buscado mediante el enriquecimiento del campo de desplazamientos o deformaciones "
         "del elemento, y el método se denominó \"método de modos incompatibles\".")

    text("El método fue introducido en 1973 por Wilson et al. como una mejora de los elementos planos "
         "y sólidos de bajo orden en problemas de flexión. Se observó mediante análisis de autovectores "
         "que los elementos de bajo orden no pueden describir el comportamiento a flexión simplemente "
         "porque carecen de las funciones de forma necesarias, por lo que las formas de desplazamiento "
         "faltantes se añadieron artificialmente.")

    text("Aunque la simple adición de funciones de forma viola la condición de continuidad, la convergencia "
         "puede lograrse si se cumplen ciertos requisitos. Las funciones de forma incompatibles no son "
         "únicas, es decir, asumen diversas formas en diferentes problemas.")

    text("La aplicación de modos incompatibles en discontinuidades de deformación y desplazamiento "
         "en problemas 1D se describe en la Sección 2. La aplicación a la mejora del comportamiento "
         "de elementos 2D y 3D en flexión se describe en la Sección 3.")
    columns_end()

    # =========================================================
    # 2. MÉTODO EN 1D
    # =========================================================
    title("2. Método de modos incompatibles en 1D")

    heading("2.1. Formulación variacional mixta", 3)

    columns_start(2)
    text("La solución estándar de elementos finitos generalmente se construye a partir de la formulación "
         "variacional de tipo desplazamiento. Si queremos enriquecer el campo de deformaciones, debemos "
         "construir la forma débil a partir de los tres conjuntos de ecuaciones: cinemáticas, "
         "constitutivas y de equilibrio. Esto se llama formulación variacional mixta o de Hu-Washizu.")
    text("El campo de deformaciones se enriquece de la siguiente manera:")
    columns_end()

    eq_block("ε(x) = ∂u/∂x + ε̃(x)   (1)")

    columns_start(2)
    text("donde el primer término es la deformación estándar y el segundo es la deformación "
         "enriquecida o incompatible.")
    text("La ecuación de equilibrio en forma débil se obtiene multiplicando por la función de "
         "ponderación *w* e integrando sobre el dominio:")
    columns_end()

    eq_block(
        "∫_{Ω} w · (∂σ)/(∂x) dΩ + ∫_{Ω} w · b dΩ = 0   (2)",
    )

    columns_start(2)
    text("Que, después de integrar por partes, asume la forma:")
    columns_end()

    eq_block(
        "∫_{Ω} (∂w)/(∂x) · σ dΩ = ∫_{Ω} w · b dΩ + [w · σ]_Γ   (3)",
    )

    columns_start(2)
    text("La forma débil de la ecuación constitutiva en el espacio del campo virtual de "
         "deformaciones γ(x), para el caso de elasticidad lineal:")
    columns_end()

    eq_block(
        "∫_{Ω} γ · (σ − E · ε) dΩ = 0   (4)",
    )

    columns_start(2)
    text("La forma débil de la ecuación cinemática en el espacio del campo virtual de tensiones τ(x):")
    columns_end()

    eq_block(
        "∫_{Ω} τ · (ε − (∂u)/(∂x)) dΩ = 0   (5)",
    )

    columns_start(2)
    text("Las Ecs. (3), (4) y (5) constituyen la forma débil mixta equivalente al funcional "
         "de Hu-Washizu para el caso 1D:")
    columns_end()

    eq_block(
        "Π(u, ε, σ) = ∫_{Ω} σ · (ε − (∂u)/(∂x)) dΩ − ∫_{Ω} (1)/(2) · E · ε^2 dΩ + ∫_{Ω} b · u dΩ   (6)",
    )

    columns_start(2)
    text("El campo virtual de deformaciones enriquecido se define de la misma forma que Ec. (1):")
    columns_end()

    eq_block(
        "γ(x) = (∂w)/(∂x) + γ̃(x)   (7)",
    )

    columns_start(2)
    text("Introduciendo las aproximaciones enriquecidas de Ecs. (1) y (7) en Ecs. (4) y (5):")
    columns_end()

    eq_block(
        "∫_{Ω} ((∂w)/(∂x) + γ̃) · (σ − E · ((∂u)/(∂x) + ε̃)) dΩ = 0   (8)",
        "∫_{Ω} τ · ((∂u)/(∂x) + ε̃ − (∂u)/(∂x)) dΩ = 0   (9)",
    )

    columns_start(2)
    text("La Ec. (8) puede dividirse en dos ecuaciones, y de la primera se obtiene:")
    columns_end()

    eq_block(
        "σ = E · ((∂u)/(∂x) + ε̃)   (12)",
    )

    columns_start(2)
    text("Las Ecs. (9), (11) y (13) constituyen la forma débil para el campo de deformaciones "
         "enriquecido usando la formulación mixta:")
    columns_end()

    _emit('<div style="background:#f4f4f4;padding:8px 16px;margin:8px 0;border-left:2px solid #666;">')
    eq_block(
        "∫_{Ω} (∂w)/(∂x) · E · ((∂u)/(∂x) + ε̃) dΩ = ∫_{Ω} w · b dΩ + [w · σ]_Γ   (13)",
        "∫_{Ω} γ̃ · E · ((∂u)/(∂x) + ε̃) dΩ = 0   (11)",
        "∫_{Ω} τ · ε̃ dΩ = 0   (9)",
    )
    _emit('</div>')

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 2.2 Implementación FE
    # =========================================================
    heading("2.2. Implementación en elementos finitos", 2)

    columns_start(2)
    text("Se considera un elemento barra (*truss*) de 2 nodos, longitud *L* y área transversal *A*, "
         "con un grado de libertad axial *u_i* en cada nodo.")
    columns_end()

    figure(_svg_truss_bar(), "Elemento barra de 2 nodos con funciones de forma", "1")

    columns_start(2)
    text("El campo de desplazamientos enriquecido se representa como la suma de la parte "
         "compatible y la incompatible:")
    columns_end()

    eq_block(
        "u(x) = N(x) · u + M(x) · α   (15)",
    )

    columns_start(2)
    text("donde *N* son funciones de forma lineales, *u* es el vector de desplazamientos "
         "nodales, *M(x)* es la función de modo incompatible, y *α* es el parámetro de modo "
         "incompatible. El campo de deformaciones enriquecido:")
    columns_end()

    eq_block(
        "ε(x) = B(x) · u + G(x) · α   (16)",
    )

    columns_start(2)
    text("donde la matriz *B* contiene las derivadas de las funciones de forma, y *G(x)* es "
         "la derivada de la función de modo incompatible.")
    text("Las condiciones adicionales que deben satisfacer los modos incompatibles son:")
    columns_end()

    eq_block(
        "span{B(x)} ∩ span{G(x)} = Ø   (17)",
        "∫_{Ω} G^T · σ dΩ = 0   (18)",
        "∫_{Ω} G^T dΩ = 0   (19)",
    )

    columns_start(2)
    text("Introduciendo las aproximaciones de elementos finitos en la formulación mixta, "
         "se obtiene el sistema de ecuaciones. Con las sustituciones:")
    columns_end()

    _emit('<div style="background:#f8f8f8;padding:8px 16px;margin:8px 0;">')
    eq_block(
        "K = ∫_{Ω} B^T · E · B dΩ  ;  F = ∫_{Ω} B^T · E · G dΩ   (21a)",
        "H = ∫_{Ω} G^T · E · G dΩ   (21b)",
    )
    _emit('</div>')

    columns_start(2)
    text("El sistema en forma matricial:")
    columns_end()

    matrix(
        [["K", "F"],
         ["F^T", "H"]],
        pre="",
        post="",
    )
    _emit('<div class="eq" style="justify-content:center;">·')
    matrix([["u"], ["α"]])
    _emit(' = ')
    matrix([["f_{ext}"], ["0"]])
    _emit('<span class="eq-num">(23)</span></div>')

    columns_start(2)
    text("La condensación estática elimina *α* del sistema. De la segunda ecuación:")
    columns_end()

    eq_block(
        "α = −H^{−1} · F^T · u   (24)",
    )

    columns_start(2)
    text("Sustituyendo, se obtiene la rigidez condensada:")
    columns_end()

    eq_block(
        "K̃ = K − F · H^{−1} · F^T   (26)",
    )
    eq_block(
        "K̃ · u = f_{ext}   (27)",
    )

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 2.3 Ejemplos
    # =========================================================
    heading("2.3. Ejemplos", 2)

    heading("2.3.1. Discontinuidad de deformación — Barra heterogénea", 3)

    columns_start(2)
    text("Considere una barra de longitud *L* compuesta de dos materiales diferentes, "
         "con la interfaz ubicada en el centro del elemento x̄ = L/2. Los módulos elásticos "
         "son *E*₁ y *E*₂, con *E*₁ > *E*₂.")
    text("Propiedades: *E*₁ = 1000, *E*₂ = 500, *L* = 1, *A* = 1, *F* = 300.")
    columns_end()

    figure(_svg_heterogeneous_bar(), "Barra heterogénea compuesta de dos materiales diferentes", "2")

    columns_start(2)
    text("Con dos elementos estándar, los desplazamientos nodales son:")
    columns_end()

    eq_block(
        "u_1 = (F · L)/(2 · E_1 · A) = 0.15  ;  u_2 = u_1 + (F · L)/(2 · E_2 · A) = 0.45   (28)",
    )

    columns_start(2)
    text("Para resolver con un solo elemento enriquecido, la función de modo incompatible *M(x)* "
         "y su derivada *G(x)* se eligen como:")
    columns_end()

    figure(_svg_incomp_mode_strain(),
           "Función de modo incompatible M(x) y su derivada G(x) para modelado de discontinuidad de deformación", "3")

    eq_block(
        "M(x) = x/x̄ , x ≤ x̄  ;  M(x) = (L−x)/(L−x̄) , x > x̄   (29)",
    )
    eq_block(
        "G(x) = 1/x̄ , x ≤ x̄  ;  G(x) = −1/(L−x̄) , x > x̄   (30)",
    )

    columns_start(2)
    text("Calculando las integrales del sistema (21) dividiendo el dominio en dos subdominios "
         "Ω₁ y Ω₂, y realizando la condensación estática, se obtiene:")
    columns_end()

    eq_block(
        "u_2 = 0.45   (32)",
    )
    eq_block(
        "α = 0.075   (33)",
    )

    columns_start(2)
    text("El desplazamiento en la discontinuidad se calcula desde la Ec. (15):")
    columns_end()

    eq_block(
        "u(x̄) = N(x̄) · u + M(x̄) · α = 0.15   (34)",
    )

    columns_start(2)
    text("Se demuestra que con un solo elemento enriquecido se obtienen los mismos resultados "
         "que con dos elementos estándar.")
    columns_end()

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 2.3.2 Discontinuidad de desplazamiento
    # =========================================================
    heading("2.3.2. Discontinuidad de desplazamiento — Modelo de falla localizada", 3)

    columns_start(2)
    text("El objetivo es presentar la aplicación del método de modos incompatibles a un modelo "
         "de daño con ablandamiento (*softening*) capaz de representar la falla del concreto en "
         "tracción. Cuando se alcanza la resistencia límite, una macro-fisura comienza a desarrollarse.")

    text("Se introduce un salto de desplazamiento *α* en el centro del elemento x̄ = L/2, "
         "que simula la apertura de fisura. La función de modo incompatible *M(x)* para este caso:")
    columns_end()

    figure(_svg_incomp_mode_disp(),
           "Función de modo incompatible M(x) y su derivada para modelado de discontinuidad de desplazamiento", "4")

    eq_block(
        "M(x) = H_{x̄}(x) − (x)/(L)   (35)",
    )

    columns_start(2)
    text("donde *H* es la función Heaviside. La derivada de *M(x)* es:")
    columns_end()

    eq_block(
        "G(x) = δ(x − x̄) − (1)/(L)   (36)",
    )
    eq_block(
        "δ(x − x̄) = 0, x ≠ x̄  ;  δ(x − x̄) → ∞, x = x̄   (37)",
    )

    columns_start(2)
    text("El campo de tensiones del material a granel se obtiene como:")
    columns_end()

    eq_block(
        "σ = E · (B · u + Ḡ · α) , Ḡ = −(1)/(L)   (38)",
    )

    columns_start(2)
    text("En la discontinuidad se introduce un modelo de daño 1D con ablandamiento. "
         "La disipación de energía se describe mediante la ley cohesiva tracción-separación:")
    columns_end()

    figure(_svg_cohesive_law(), "Ley cohesiva en la discontinuidad", "5")

    eq_block(
        "t_{Γ} = (1)/(C̄) · α = K̄ · α   (39)",
    )

    columns_start(2)
    text("donde C̄ es el módulo de *compliance* en la discontinuidad. "
         "La función de daño verifica si la tracción es admisible:")
    columns_end()

    eq_block(
        "Φ(t_{Γ}, q̄) = |t_{Γ}| − (σ_f − q̄) ≤ 0   (40)",
    )
    eq_block(
        "q̄ = K̄ · ξ̄   (41)",
    )

    columns_start(2)
    text("donde σ_f es el límite elástico, K̄ es el módulo de ablandamiento, "
         "y ξ̄ es la variable de ablandamiento tipo desplazamiento.")
    text("Las ecuaciones de evolución del daño:")
    columns_end()

    eq_block(
        "α̇ = γ̇ · sign(t_{Γ})  ;  ξ̄̇ = γ̇  ;  γ̇ ≥ 0   (42)",
    )
    eq_block(
        "γ̇ ≥ 0  ;  Φ ≤ 0  ;  γ̇ · Φ = 0   (43)",
    )

    columns_start(2)
    text("La energía de fractura *G_f* es igual al área bajo la curva de ablandamiento:")
    columns_end()

    eq_block(
        "G_f = (σ_f^2)/(2 · |K̄|)   (45)",
    )

    columns_start(2)
    text("**Ejemplo numérico:** Barra en voladizo discretizada con 2 elementos. "
         "Propiedades: *E* = 1000, σ_f = 150, K̄ = −200, *L* = 1, *A* = 1, ū = 0.75.")
    columns_end()

    figure(_svg_force_disp(),
           "Diagrama fuerza-desplazamiento con dos ciclos de carga-descarga", "8")

    columns_start(2)
    text("El comportamiento es elástico lineal hasta alcanzar el límite elástico. "
         "Luego comienza el ablandamiento y la fisura se abre en el centro del elemento 1. "
         "La descarga y recarga siguen el mismo camino, mostrando el comportamiento típico de "
         "modelos de daño, hasta la falla completa (σ = 0).")
    columns_end()

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 3. MÉTODO EN 2D Y 3D
    # =========================================================
    title("3. Método de modos incompatibles en 2D y 3D")

    heading("3.1. Elemento finito 2D", 2)

    heading("3.1.1. Formulación del elemento Q4", 3)

    columns_start(2)
    text("La geometría del elemento isoparamétrico cuadrilátero 2D se representa como:")
    columns_end()

    eq_block(
        "x = Σ_{i=1}^{4} N_i(ξ,η) · x_i  ;  y = Σ_{i=1}^{4} N_i(ξ,η) · y_i   (47)",
    )

    columns_start(2)
    text("donde ξ, η ∈ [−1, 1] son coordenadas isoparamétricas, N_i son funciones de forma "
         "y x_i, y_i son coordenadas nodales.")
    columns_end()

    figure(_svg_q4_element(), "Elemento isoparamétrico Q4", "9")

    columns_start(2)
    text("Los desplazamientos se interpolan de la misma forma:")
    columns_end()

    eq_block(
        "u = Σ_{i=1}^{4} N_i · u_i  ;  v = Σ_{i=1}^{4} N_i · v_i   (48)",
    )

    columns_start(2)
    text("Las funciones de forma isoparamétricas en notación matricial:")
    columns_end()

    eq_block(
        "N_i = (1)/(4) · (1 + ξ_i · ξ) · (1 + η_i · η)   (49)",
    )

    columns_start(2)
    text("La relación deformación-desplazamiento para tensión/deformación plana:")
    columns_end()

    matrix(
        [["ε_x"],
         ["ε_y"],
         ["γ_{xy}"]],
        name="ε",
        post=" = B · u",
    )
    _emit('<span class="eq-num">(50)</span>')

    columns_start(2)
    text("con la matriz *B* definida como:")
    columns_end()

    matrix(
        [["∂N_i/∂x", "0"],
         ["0", "∂N_i/∂y"],
         ["∂N_i/∂y", "∂N_i/∂x"]],
        name="B_i",
        post=", i = 1,...,4",
    )
    _emit('<span class="eq-num">(51)</span>')

    columns_start(2)
    text("Las derivadas se obtienen usando la regla de la cadena con la matriz Jacobiana *J*:")
    columns_end()

    eq_block(
        "J = Σ_{i=1}^{4} [(∂N_i)/(∂ξ) · x_i, (∂N_i)/(∂ξ) · y_i; (∂N_i)/(∂η) · x_i, (∂N_i)/(∂η) · y_i]   (53)",
    )

    columns_start(2)
    text("La matriz de rigidez del elemento y el vector de carga:")
    columns_end()

    eq_block(
        "K = ∫_{Ω} B^T · D · B · t dΩ  ;  f = ∫_{Ω} N^T · q · t dΩ   (55)",
    )

    columns_start(2)
    text("Transformando a coordenadas isoparamétricas:")
    columns_end()

    eq_block(
        "dΩ = |J| · dξ · dη   (56)",
    )
    eq_block(
        "K = ∫_{-1}^{1} ∫_{-1}^{1} B^T · D · B · t · |J| dξ dη   (57)",
    )

    columns_start(2)
    text("La integración numérica se realiza con cuadratura de Gauss:")
    columns_end()

    eq_block(
        "K ≈ Σ_{i=1}^{m} Σ_{j=1}^{n} B^T(ξ_i, η_j) · D · B(ξ_i, η_j) · t · |J(ξ_i, η_j)| · w_i · w_j   (58)",
    )

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 3.1.2 Adición de modos incompatibles
    # =========================================================
    heading("3.1.2. Adición de modos incompatibles al Q4 → Q6", 3)

    columns_start(2)
    text("El elemento Q4 no puede describir bien la flexión porque carece de las funciones de forma "
         "de tipo cuadrático necesarias. Esto se compensa añadiendo los desplazamientos faltantes:")
    columns_end()

    eq_block(
        "u_i = Σ_{j=1}^{4} N_j · u_{ij} + M_1(ξ) · α_1 + M_2(η) · α_2   (60)",
    )

    columns_start(2)
    text("donde α₁, α₂ son los parámetros de modos incompatibles, y las funciones de forma "
         "de modos incompatibles son:")
    columns_end()

    _emit('<div style="background:#f4f8fc;padding:10px 16px;margin:8px 0;border:1px solid #cde;">')
    eq_block(
        "M_1 = 1 − ξ^2   ;   M_2 = 1 − η^2",
    )
    _emit('</div>')

    columns_start(2)
    text("La matriz de deformación-desplazamiento *G* para el elemento Q4 que satisface la "
         "condición del *patch test* (Ec. 19):")
    columns_end()

    eq_block(
        "G_j = (∂M_j)/(∂x) − (1)/(|Ω_e|) · ∫_{Ω_e} (∂M_j)/(∂x) dΩ , j = 1, 2   (61)",
    )

    columns_start(2)
    text("La matriz *B* completa con los modos incompatibles es **B**_m = [**B** **G**], "
         "y la rigidez se formula como:")
    columns_end()

    eq_block(
        "K_m = ∫_{Ω} B_m^T · D · B_m · t dΩ   (62)",
    )

    columns_start(2)
    text("El vector de desplazamientos y fuerzas se aumenta con los parámetros incompatibles α. "
         "El sistema de ecuaciones tiene la misma forma que la Ec. (23). "
         "El elemento resultante se denomina **Q6**.")
    columns_end()

    heading("3.1.3. Ejemplo — Viga en voladizo 2D", 3)

    figure(_svg_cantilever_2d(), "Viga en voladizo con carga de fuerza y momento", "12")

    columns_start(2)
    text("Propiedades: *E* = 3×10⁷, ν = 0.33, tensión plana, *L* = 3.0, *h* = 0.2 "
         "(relación largo/alto = 15). Comparación de resultados sin y con modos incompatibles:")
    columns_end()

    table(
        caption="Tabla 1. Comparación de resultados de desplazamiento en 2D",
        headers=["Desplazamientos", "Exacto", "Sin MI", "Con MI"],
        rows=[
            ["Carga 1 (Fuerza)", "0.562", "0.169", "0.544"],
            ["Carga 2 (Momento)", "0.112", "0.033", "0.113"],
        ],
    )

    columns_start(2)
    text("Los resultados muestran que el elemento Q4 estándar (sin MI) es excesivamente rígido, "
         "capturando solo el 30% del desplazamiento real. Con modos incompatibles (Q6), "
         "los resultados se aproximan al 97% de la solución exacta.")
    columns_end()

    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    # =========================================================
    # 3.2 Elemento 3D
    # =========================================================
    heading("3.2. Elemento finito 3D — Q8", 2)

    heading("3.2.1. Formulación del elemento Q8", 3)

    columns_start(2)
    text("El elemento Q8 es la extensión 3D del Q4, con más grados de libertad. "
         "La aproximación isoparamétrica para geometría y desplazamientos:")
    columns_end()

    eq_block(
        "x = Σ_{i=1}^{8} N_i · x_i  ;  u = Σ_{i=1}^{8} N_i · u_i , i = 1,...,8   (65)",
    )
    eq_block(
        "N_i = (1)/(8) · (1 + ξ_i · ξ) · (1 + η_i · η) · (1 + ζ_i · ζ)   (66)",
    )

    heading("3.2.2. Adición de modos incompatibles", 3)

    columns_start(2)
    text("El elemento Q8 sufre la misma falta de formas de flexión que el Q4. "
         "El remedio es el mismo: la adición de las formas faltantes:")
    columns_end()

    eq_block(
        "u_i = Σ_{j=1}^{8} N_j · u_{ij} + M_1 · α_1 + M_2 · α_2 + M_3 · α_3 , j = 1,2,3   (67)",
    )

    _emit('<div style="background:#f4f8fc;padding:10px 16px;margin:8px 0;border:1px solid #cde;">')
    eq_block(
        "M_1 = 1 − ξ^2   ;   M_2 = 1 − η^2   ;   M_3 = 1 − ζ^2   (68)",
    )
    _emit('</div>')

    heading("3.2.3. Ejemplo — Viga en voladizo 3D", 3)

    columns_start(2)
    text("Propiedades: *E* = 3×10⁷, ν = 0.33, tensión plana, *L* = 3.0, *h* = 0.2, *t* = 0.1.")
    columns_end()

    table(
        caption="Tabla 2. Comparación de resultados de desplazamiento en 3D",
        headers=["Desplazamientos", "Exacto", "Sin MI", "Con MI"],
        rows=[
            ["Carga 1 (Fuerza)", "0.562", "0.167", "0.544"],
            ["Carga 2 (Momento)", "0.112", "0.033", "0.112"],
        ],
    )

    columns_start(2)
    text("La comparación con el elemento Q4 muestra que los desplazamientos debidos a la fuerza "
         "son ligeramente menos precisos, mientras que la precisión para el momento es similar. "
         "El elemento Q8 es numéricamente mucho más complejo: la rigidez es de 24×24, "
         "y con modos incompatibles (sin condensación) es de 32×32.")
    columns_end()

    # =========================================================
    # 4. CONCLUSIÓN
    # =========================================================
    page_break(left="Kožar, Rukavina, Ibrahimbegović", right="GRAĐEVINAR 70 (2018)")

    title("4. Conclusión")

    columns_start(2)
    text("Este trabajo presenta una breve revisión del método de modos incompatibles. Los autores "
         "lo han situado en dos áreas comunes de aplicación: análisis de falla y mejora del "
         "comportamiento del elemento.")

    text("En el contexto del análisis de falla, los modos incompatibles se han utilizado para "
         "introducir funciones de desplazamiento y deformación cuyos parámetros permiten describir "
         "la falla dentro del propio elemento. El procedimiento se ha demostrado en un elemento 1D "
         "tanto para discontinuidades de deformación como de desplazamiento.")

    text("En el contexto de la mejora de las capacidades de los elementos finitos, los modos "
         "incompatibles se han utilizado para enriquecer el campo de desplazamientos disponibles "
         "de las funciones de forma estándar. Las formas cuadráticas adicionales mejoran "
         "significativamente el comportamiento a flexión.")

    text("Se ha demostrado que el método tiene buenas propiedades en ambas áreas y que su uso "
         "debe fomentarse. Los grados de libertad adicionales pueden eliminarse mediante "
         "condensación estática o reducirse mediante el procedimiento de *operator split*. "
         "Los resultados obtenidos con modos incompatibles son superiores a los de elementos "
         "\"estándar\", es decir, resultados comparables solo podrían obtenerse usando un número "
         "mucho mayor de elementos sin enriquecer.")
    columns_end()

    hr_line()
    _emit('<p style="font-size:8.5pt;color:#888;text-align:center;margin-top:20px;">'
          'Generado con <strong>pyhekatan</strong> v0.7.0 — Réplica del paper JCE-70-2018-1-3-2078-EN</p>')

    # =========================================================
    # Generar HTML
    # =========================================================
    show(filename=os.path.join(os.path.dirname(__file__), "test_jce_output.html"))
