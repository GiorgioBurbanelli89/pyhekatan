"""
Test: Integracion — Area Bajo una Curva & La Integral Definida
Replicates textbook content using pyhekatan library v0.8.0
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from hekatan import (
    paper, header, footer, author, abstract_block,
    title, heading, text, markdown, eq_block, figure,
    columns, column, end_columns, table, show, clear,
    page_break, hr, note, image, html_raw,
    integral, summation, derivative, partial,
)

clear()

# ── Paper configuration ──────────────────────────────────────
paper(
    size="A4",
    margin="20mm 18mm 25mm 18mm",
    font='"Georgia", "Times New Roman", Times, serif',
    fontsize="10pt",
    color="#000000",
    accent="#1a5276",
    background="#FFFFFF",
    lineheight=1.45,
    columngap="8mm",
    startpage=1,
    pagenumber="right",
)

header(left="CALCULO INTEGRAL", right="Capitulo 5")

# ══════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════
heading("5. Integracion", 1)

# ── Two-column layout ────────────────────────────────────────
columns(2, css_columns=True)

# ══════════════════════════════════════════════════════════════
# 5.1 Area bajo una curva
# ══════════════════════════════════════════════════════════════
heading("5.1 Area bajo una curva", 2)

markdown("""
El problema de calcular el area bajo una curva es uno de los problemas fundamentales del calculo integral. Consideremos una funcion continua *f(x)* definida en un intervalo [*a*, *b*] donde *f(x)* >= 0.

Para aproximar el area bajo la curva, dividimos el intervalo [*a*, *b*] en *N* subintervalos de igual ancho. El ancho de cada subintervalo es:
""")

# Eq (5.1): Delta x
eq_block("Delta*x = (x_m)/(N) = (b - a)/(N)  (5.1)")

markdown("""
donde *x_m* = *b* - *a* es la longitud total del intervalo.

En cada subintervalo se construye un rectangulo cuya altura es el valor de la funcion evaluada en algun punto del subintervalo. La suma de las areas de todos los rectangulos proporciona una aproximacion al area bajo la curva:
""")

# Eq (5.2): Riemann sum
eq_block("A ≈ Σ_{i=1}^{N} f_i * Delta*x  (5.2)")

markdown("""
A medida que *N* aumenta (y por tanto *Delta x* disminuye), la aproximacion mejora. En el limite, cuando *N* tiende a infinito, la suma se convierte en el area exacta:
""")

# Eq (5.3): limit definition
eq_block("A = lim_{N→∞} Σ_{i=1}^{N} f(x_i) * Delta*x  (5.3)")

markdown("""
Este limite, cuando existe, define la **integral definida** de *f(x)* en el intervalo [*a*, *b*].
""")

# ── SVG diagram: rectangles under curve ──────────────────────
svg_area = """<svg viewBox="0 0 400 250" xmlns="http://www.w3.org/2000/svg" style="font-family:Georgia,serif;">
  <!-- Axes -->
  <line x1="50" y1="200" x2="370" y2="200" stroke="#000" stroke-width="1.5"/>
  <line x1="50" y1="200" x2="50" y2="20" stroke="#000" stroke-width="1.5"/>
  <!-- Arrow tips -->
  <polygon points="370,200 362,196 362,204" fill="#000"/>
  <polygon points="50,20 46,28 54,28" fill="#000"/>

  <!-- Rectangles (Riemann sum) -->
  <rect x="80" y="120" width="40" height="80" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>
  <rect x="120" y="90" width="40" height="110" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>
  <rect x="160" y="68" width="40" height="132" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>
  <rect x="200" y="55" width="40" height="145" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>
  <rect x="240" y="50" width="40" height="150" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>
  <rect x="280" y="58" width="40" height="142" fill="#aed6f1" stroke="#1a5276" stroke-width="0.8" opacity="0.7"/>

  <!-- Curve f(x) -->
  <path d="M 80,120 Q 140,60 200,50 Q 260,42 320,58" fill="none" stroke="#c0392b" stroke-width="2.5"/>

  <!-- Labels -->
  <text x="380" y="205" font-size="14" font-style="italic">x</text>
  <text x="38" y="15" font-size="14" font-style="italic">y</text>
  <text x="75" y="218" font-size="12" font-style="italic">a</text>
  <text x="315" y="218" font-size="12" font-style="italic">b</text>
  <text x="180" y="240" font-size="11">Δx</text>
  <text x="340" y="45" font-size="13" font-style="italic" fill="#c0392b">f(x)</text>
  <text x="195" y="105" font-size="11" fill="#1a5276">f_i</text>

  <!-- Delta x bracket -->
  <line x1="160" y1="225" x2="200" y2="225" stroke="#000" stroke-width="1"/>
  <line x1="160" y1="222" x2="160" y2="228" stroke="#000" stroke-width="1"/>
  <line x1="200" y1="222" x2="200" y2="228" stroke="#000" stroke-width="1"/>
</svg>"""

figure(svg_area,
       caption="Aproximacion del area bajo f(x) mediante rectangulos (suma de Riemann)",
       number="5.1", width="85%")

# ══════════════════════════════════════════════════════════════
# 5.2 La integral definida
# ══════════════════════════════════════════════════════════════

heading("5.2 La integral definida", 2)

markdown("""
La **integral definida** de una funcion *f(x)* desde *a* hasta *b* se define como:
""")

# Eq (5.4): definite integral definition
eq_block("∫_{a}^{b} f(x)*dx = lim_{N→∞} Σ_{i=1}^{N} f(x_i)*Delta*x  (5.4)")

markdown("""
siempre que el limite exista. Cuando esto ocurre, se dice que *f* es **integrable** en [*a*, *b*].

**Teorema fundamental del calculo:** Si *f* es continua en [*a*, *b*] y *F* es una antiderivada de *f*, entonces:
""")

# Eq (5.5): fundamental theorem
eq_block("∫_{a}^{b} f(x)*dx = F(b) - F(a)  (5.5)")

markdown("""
donde *F'(x)* = *f(x)*. Se utiliza la notacion:
""")

eq_block("∫_{a}^{b} f(x)*dx = [F(x)]_{a}^{b} = F(b) - F(a)  (5.6)")

# ── Page break ───────────────────────────────────────────────
page_break(left="CALCULO INTEGRAL", right="2",
           linecolor="#000", textcolor="#000")

heading("5.3 Propiedades de la integral definida", 2)

markdown("""
Las integrales definidas satisfacen las siguientes propiedades fundamentales:
""")

html_raw("""
<ol>
<li><b>Linealidad:</b></li>
</ol>
""")

eq_block(
    "∫_{a}^{b} [f(x) + g(x)]*dx = ∫_{a}^{b} f(x)*dx + ∫_{a}^{b} g(x)*dx  (5.7a)",
    "∫_{a}^{b} c*f(x)*dx = c * ∫_{a}^{b} f(x)*dx  (5.7b)",
)

html_raw("""
<ol start="2">
<li><b>Aditividad respecto al intervalo:</b></li>
</ol>
""")

eq_block("∫_{a}^{b} f(x)*dx = ∫_{a}^{c} f(x)*dx + ∫_{c}^{b} f(x)*dx  (5.8)")

markdown("""
para cualquier *c* en [*a*, *b*].
""")

html_raw("""
<ol start="3">
<li><b>Limites invertidos:</b></li>
</ol>
""")

eq_block("∫_{a}^{b} f(x)*dx = -∫_{b}^{a} f(x)*dx  (5.9)")

html_raw("""
<ol start="4">
<li><b>Integral nula:</b></li>
</ol>
""")

eq_block("∫_{a}^{a} f(x)*dx = 0  (5.10)")

heading("5.4 Integracion por partes", 2)

markdown("""
Si *u* y *v* son funciones diferenciables, la regla de integracion por partes establece:
""")

eq_block("∫_{a}^{b} u*dv = [u*v]_{a}^{b} - ∫_{a}^{b} v*du  (5.11)")

markdown("""
o equivalentemente, en forma indefinida:
""")

eq_block("∫ u*dv = u*v - ∫ v*du  (5.12)")

markdown("""
Esta formula es particularmente util cuando el integrando es un producto de dos funciones y una de ellas se simplifica al derivar.
""")

heading("5.5 Sustitucion trigonometrica", 2)

markdown("""
Para integrales que contienen expresiones de la forma *a^2 - x^2*, *a^2 + x^2* o *x^2 - a^2*, se utilizan las siguientes sustituciones:
""")

table([
    ["Expresion", "Sustitucion", "Identidad"],
    ["a^2 - x^2", "x = a*sin(theta)", "1 - sin^2 = cos^2"],
    ["a^2 + x^2", "x = a*tan(theta)", "1 + tan^2 = sec^2"],
    ["x^2 - a^2", "x = a*sec(theta)", "sec^2 - 1 = tan^2"],
])

# ── Page break ───────────────────────────────────────────────
page_break(left="CALCULO INTEGRAL", right="3",
           linecolor="#000", textcolor="#000")

heading("5.6 Ejercicios resueltos", 2)

markdown("""
**Ejercicio 1.** Evaluar la integral:
""")

eq_block("I_1 = ∫_{0}^{1} (1)/(e^{3x})*dx  (5.13)")

markdown("""
**Solucion:** Reescribimos el integrando como *e^{-3x}* y aplicamos la regla de integracion:
""")

eq_block(
    "I_1 = ∫_{0}^{1} e^{-3x}*dx = [(-1)/(3)*e^{-3x}]_{0}^{1}  (5.14)",
)

eq_block(
    "I_1 = (-1)/(3)*e^{-3} - (-1)/(3)*e^{0} = (-1)/(3)*(e^{-3} - 1)  (5.15)",
)

eq_block("I_1 = (1)/(3)*(1 - e^{-3}) ≈ 0.3167  (5.16)")

hr()

markdown("""
**Ejercicio 2.** Evaluar la integral:
""")

eq_block("I_2 = ∫_{0}^{π/2} cos(φ)*dφ  (5.17)")

markdown("""
**Solucion:** La antiderivada de *cos(phi)* es *sin(phi)*:
""")

eq_block(
    "I_2 = [sin(φ)]_{0}^{π/2} = sin(π/2) - sin(0) = 1 - 0 = 1  (5.18)",
)

hr()

markdown("""
**Ejercicio 3.** Evaluar la integral por partes:
""")

eq_block("I_3 = ∫_{0}^{1} x*e^{x}*dx  (5.19)")

markdown("""
**Solucion:** Sea *u = x* y *dv = e^x dx*. Entonces *du = dx* y *v = e^x*. Aplicando la formula (5.11):
""")

eq_block(
    "I_3 = [x*e^{x}]_{0}^{1} - ∫_{0}^{1} e^{x}*dx  (5.20)",
)

eq_block(
    "I_3 = (1*e^1 - 0*e^0) - [e^{x}]_{0}^{1} = e - (e - 1) = 1  (5.21)",
)

hr()

markdown("""
**Ejercicio 4.** Calcular el area bajo la curva *f(x) = x^2* en el intervalo [0, 3]:
""")

eq_block("A = ∫_{0}^{3} x^2*dx = [(x^3)/(3)]_{0}^{3} = (27)/(3) - 0 = 9  (5.22)")

hr()

markdown("""
**Ejercicio 5.** Evaluar:
""")

eq_block("I_5 = ∫_{1}^{4} (1)/(√x)*dx = ∫_{1}^{4} x^{-1/2}*dx  (5.23)")

eq_block("I_5 = [2*√x]_{1}^{4} = 2*√4 - 2*√1 = 4 - 2 = 2  (5.24)")

# ══════════════════════════════════════════════════════════════
# End
# ══════════════════════════════════════════════════════════════

end_columns()

show("test_integral_output.html")
