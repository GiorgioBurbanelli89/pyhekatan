"""
Test: Teoría de Placas — Chapter 1
Replicates the .hcalc document using pyhekatan library v0.8.0
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from hekatan import (
    paper, header, footer, author, abstract_block,
    title, heading, text, markdown, eq_block, figure,
    columns, column, end_columns, table, show, clear,
    page_break, hr, note, image, html_raw,
)

clear()

# ── Paper configuration ──────────────────────────────────────
paper(
    size="A4",
    margin="20mm 18mm 25mm 18mm",
    font='"Georgia", "Times New Roman", Times, serif',
    fontsize="10pt",
    color="#000000",
    accent="#000000",
    background="#FFFFFF",
    lineheight=1.45,
    columngap="8mm",
    startpage=15,
    pagenumber="right",
)

# ── Two-column layout ────────────────────────────────────────
columns(2, css_columns=True)

# ══════════════════════════════════════════════════════════════
# 1. Introduccion
# ══════════════════════════════════════════════════════════════

heading("1. Introduccion", 1)

heading("1.1 Definicion de placa", 2)

markdown("""
Una **placa** es un elemento estructural plano cuyo espesor *h* es significativamente menor que sus dimensiones en planta *a* y *b*. La superficie media de la placa, equidistante de ambas caras, define el plano de referencia. Las cargas actuan predominantemente perpendiculares a este plano medio.

La relacion entre el espesor y la menor dimension en planta permite clasificar las placas segun su comportamiento mecanico. Para placas delgadas, donde la relacion *h/a* es pequena (tipicamente menor a 1/10), las hipotesis de Kirchhoff-Love proporcionan resultados satisfactorios.

En ingenieria estructural, las placas aparecen como losas de entrepiso, muros de contencion, tapas de tanques, tableros de puentes y muchos otros elementos. Su analisis requiere la solucion de ecuaciones diferenciales parciales que relacionan las deflexiones con las cargas aplicadas.
""")

heading("1.2 Clasificacion de placas", 2)

markdown("""
Las placas se clasifican segun diferentes criterios:

**Segun el espesor relativo:**
- *Placas delgadas* (*h/a* < 1/10): teoria de Kirchhoff
- *Placas moderadamente gruesas* (1/10 < *h/a* < 1/5): teoria de Reissner-Mindlin
- *Placas gruesas* (*h/a* > 1/5): analisis tridimensional

**Segun la geometria:**
- Placas rectangulares
- Placas circulares
- Placas anulares
- Placas de forma irregular

**Segun las condiciones de apoyo:**
- Simplemente apoyada en todos los bordes
- Empotrada en todos los bordes
- Voladizo (un borde empotrado, tres libres)
- Combinaciones de apoyos

**Segun el material:**
- Placas isotropas (propiedades iguales en todas las direcciones)
- Placas ortotropas (propiedades diferentes en dos direcciones perpendiculares)
- Placas laminadas (compuestas por capas de diferentes materiales)
""")

heading("1.3 Hipotesis fundamentales", 2)

markdown("""
La teoria clasica de placas delgadas (Kirchhoff-Love) se basa en las siguientes hipotesis:
""")

html_raw("""
<ol>
<li>El espesor <i>h</i> de la placa es peque&ntilde;o comparado con las dimensiones en planta.</li>
<li>Las deflexiones <i>w</i> son peque&ntilde;as comparadas con el espesor.</li>
<li>Las secciones planas normales al plano medio permanecen planas y normales despues de la deformacion.</li>
<li>Las tensiones normales en la direccion del espesor son despreciables.</li>
</ol>
""")

# ── Page break with running header ───────────────────────────
page_break(left="TEORIA DE PLACAS", right="15",
           linecolor="#000", textcolor="#000")

heading("1.4 Ecuacion diferencial de la placa", 2)

markdown("""
La ecuacion diferencial que gobierna la deflexion *w* de una placa delgada isotropa bajo carga transversal *q* es la ecuacion biharmonica. Esta ecuacion fue derivada por primera vez por Lagrange en 1811 y posteriormente refinada por Sophie Germain y Navier.
""")

# Eq (1.1): partial derivatives
eq_block(
    "(∂^2M_x)/(∂x^2) + 2*(∂^2M_{xy})/(∂x*∂y) + (∂^2M_y)/(∂y^2) + q = 0    (1.1)"
)

# Eq (1.2): biharmonic
eq_block("D*∇^4*w = q    (1.2)")

markdown("""
donde *D* es la rigidez flexural de la placa:
""")

# Eq (1.3): flexural rigidity — NOTE: (num)/(den) format required
eq_block("D = (E*h^3)/(12*(1 - ν^2))    (1.3)")

markdown("""
con *E* el modulo de elasticidad, *ν* el coeficiente de Poisson y *h* el espesor.

La deflexion maxima de una placa rectangular simplemente apoyada bajo carga uniforme puede expresarse como:
""")

# Eq (1.4): max deflection coefficient
eq_block("θ = α*(q)/(E)*(a)/(h)^4    (1.4)")

markdown("""
donde *α* es un coeficiente que depende de la relacion de lados *a/b* y de las condiciones de apoyo.
""")

heading("1.5 Parametros de comparacion", 2)

markdown("""
La siguiente tabla muestra los coeficientes *α* para placas rectangulares con diferentes condiciones de apoyo y relaciones de lados:
""")

table([
    ["Condicion de apoyo", "a/b=1.0", "a/b=1.5", "a/b=2.0", "a/b=3.0", "a/b=∞"],
    ["4 lados simp. apoyados",  "0.0444", "0.0843", "0.1106", "0.1335", "0.1422"],
    ["4 lados empotrados",      "0.0138", "0.0277", "0.0404", "0.0519", "0.0571"],
    ["2 opuestos simp.+2 emp.", "0.0209", "0.0428", "0.0611", "0.0778", "0.0834"],
    ["Voladizo (1 emp.+3 lib.)", "0.1265", "0.1478", "0.1510", "0.1520", "0.1522"],
])

# ── Page break with running header ───────────────────────────
page_break(left="TEORIA DE PLACAS", right="16",
           linecolor="#000", textcolor="#000")

heading("1.6 Tipos de calculo", 2)

markdown("""
El analisis de placas comprende los siguientes tipos de calculo:

**Fuerzas internas:**

Las fuerzas internas por unidad de longitud se obtienen integrando las tensiones a traves del espesor. Estas resultantes incluyen fuerzas normales, cortantes, momentos flectores y momentos torsores:
""")

# Eq (1.5a, 1.5b, 1.5c): stress resultants with integrals
eq_block(
    "N_x = ∫_{-h/2}^{h/2} σ_x*dz    (1.5a)",
    "M_x = ∫_{-h/2}^{h/2} σ_x*z*dz    (1.5b)",
    "T_{xy} = ∫_{-h/2}^{h/2} τ_{xy}*dz    (1.5c)",
)

markdown("""
**Frecuencias propias:**

El analisis dinamico de placas requiere la solucion del problema de valores propios. Las frecuencias naturales de vibracion se determinan resolviendo:
""")

# Eq (1.6): vibration
eq_block("D*∇^4*w - ρ*h*ω^2*w = 0    (1.6)")

markdown("""
donde *ρ* es la densidad del material y *ω* las frecuencias circulares.

**Estabilidad:**

El pandeo de placas se analiza mediante la ecuacion de estabilidad que incluye las fuerzas en el plano:
""")

# Eq (1.7): stability equation
eq_block(
    "D*∇^4*w + N_x*(∂^2w)/(∂x^2) + 2*N_{xy}*(∂^2w)/(∂x*∂y) + N_y*(∂^2w)/(∂y^2) = 0    (1.7)"
)

markdown("""
donde *N_x*, *N_y* y *N_{xy}* son las fuerzas de compresion en el plano.

**Metodos de solucion:**
- *Metodos analiticos:* Series de Navier, series de Levy, funciones de Green
- *Metodos aproximados:* Rayleigh-Ritz, Galerkin, diferencias finitas
- *Metodos numericos:* Elementos finitos (FEM), elementos de contorno (BEM)

En la practica moderna, el metodo de elementos finitos es el mas utilizado debido a su versatilidad para geometrias complejas, condiciones de apoyo arbitrarias y cargas no uniformes. Sin embargo, las soluciones analiticas siguen siendo importantes como referencia para la validacion de modelos numericos y para la comprension del comportamiento fundamental de las placas.

Los capitulos siguientes desarrollan cada uno de estos temas en detalle, comenzando por el analisis de esfuerzos internos en el Capitulo 2.
""")

# ── Page break with running header ───────────────────────────
page_break(left="TEORIA DE PLACAS", right="17",
           linecolor="#000", textcolor="#000")

end_columns()

show("test_placas_output.html")
