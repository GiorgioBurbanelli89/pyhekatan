"""
Test: Paper JCE-70-2018 usando la librería pyhekatan v0.8.0 mejorada.
"Method of incompatible modes – overview and application"

Usa las funciones de la librería (no self-contained) para verificar que
paper(), eq_block(), figure(), markdown(), columns(), author(), abstract_block()
funcionan correctamente.
"""
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hekatan import (
    paper, header, footer, author, abstract_block,
    title, heading, text, markdown, eq_block, figure,
    matrix, table, columns, column, end_columns,
    eq, var, check, note, hr, page_break, html_raw,
    show, clear,
)

# ============================================================
# Paper configuration
# ============================================================
clear()
paper(
    size="A4",
    margin="20mm 18mm 25mm 18mm",
    font='"Georgia", "Times New Roman", Times, serif',
    fontsize="10pt",
    accent="#F27835",
    lineheight=1.45,
    columngap="8mm",
)

# ============================================================
# Header
# ============================================================
header(
    left="Građevinar · Revista de Ingeniería Civil",
    right="JCE 70 (2018) 1, 19-29",
    barside="left",
    color="#F27835",
)

# ============================================================
# Title
# ============================================================
title("Método de Modos Incompatibles — Resumen y Aplicación")

# Authors
columns(2, "50:50")
author("Ivo Kožar", "Universidad de Rijeka, Croacia")
column()
author("Tea Rukavina", "Universidad de Rijeka, Croacia")
end_columns()

# ============================================================
# Abstract
# ============================================================
abstract_block(
    "El método de modos incompatibles es una técnica de elementos finitos que mejora "
    "significativamente la precisión de elementos de bajo orden sin incrementar el número "
    "de grados de libertad globales. Este trabajo presenta una formulación unificada del método, "
    "mostrando su conexión con la discontinuidad fuerte (localización de deformaciones). "
    "Se presentan ejemplos numéricos en 1D, 2D y 3D que demuestran la efectividad del método.",
    keywords=["elementos finitos", "modos incompatibles", "localización", "discontinuidad fuerte"],
    lang="español",
)

hr()

# ============================================================
# 1. Introducción
# ============================================================
columns(2, css_columns=True)

heading("1. Introducción", 2)

markdown("""
La formulación clásica de elementos finitos se basa en funciones de forma compatibles
que garantizan la continuidad del campo de desplazamientos entre elementos. Sin embargo,
elementos de bajo orden como el Q4 bilineal presentan el problema de **bloqueo por corte**
(*shear locking*) en flexión.

Wilson y Taylor (1973) propusieron agregar **modos incompatibles** internos al elemento,
que mejoran la representación del campo de deformaciones sin afectar la compatibilidad global.
""")

heading("1.1 Formulación variacional", 3)

text("El principio de los trabajos virtuales se expresa como:")

eq_block(
    "∫_{Ω} sigma · delta epsilon  dΩ = ∫_{Ω} b · delta u  dΩ + ∫_{Γ} t · delta u  dΓ  (1)"
)

text("Donde el campo de desplazamientos se enriquece con modos incompatibles:")

eq_block(
    "u(x) = N(x) · d + M(x) · alpha  (2)",
    "epsilon(x) = B(x) · d + G(x) · alpha  (3)",
)

text("siendo N las funciones de forma estándar, M los modos incompatibles, "
     "B = (partial N)/(partial x), G = (partial M)/(partial x), "
     "d los desplazamientos nodales y alpha los parámetros incompatibles.")

heading("1.2 Sistema de ecuaciones", 3)

text("La discretización conduce al sistema aumentado:")

eq_block(
    "[K]{d} + [F]{alpha} = {f}  (4)",
    "[F]^T{d} + [H]{alpha} = {0}  (5)",
)

text("donde las matrices se definen como:")

eq_block(
    "K_{ij} = ∫_{Ω} B_i^T · C · B_j  dΩ  (6)",
    "F_{ij} = ∫_{Ω} B_i^T · C · G_j  dΩ  (7)",
    "H_{ij} = ∫_{Ω} G_i^T · C · G_j  dΩ  (8)",
)

text("De la ecuación (5) se obtiene alpha en función de d:")

eq_block(
    "alpha = -[H]^{-1} · [F]^T · {d}  (9)",
)

text("Sustituyendo en (4):")

eq_block(
    "K* · d = f  (10)",
    "K* = K - F · H^{-1} · F^T  (11)",
)

end_columns()

# ============================================================
# Page break + new header
# ============================================================
page_break()
header(
    left="Kožar, Rukavina, Ibrahimbegović",
    right="JCE 70 (2018) 1, 19-29",
    barside="left",
    color="#F27835",
)

# ============================================================
# 2. Ejemplo 1D: Barra heterogénea
# ============================================================
columns(2, css_columns=True)

heading("2. Ejemplos 1D", 2)

heading("2.1 Barra de dos nodos — modo estándar", 3)

text("Para un elemento de barra con 2 nodos, las funciones de forma estándar son:")

eq_block(
    "N_1(x) = 1 - (x)/(L),  N_2(x) = (x)/(L)  (12)",
)

# Figure: truss bar element
figure(
    '''<svg width="480" height="180" viewBox="0 0 480 180" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <line x1="60" y1="60" x2="420" y2="60" stroke="#333" stroke-width="2"/>
  <circle cx="60" cy="60" r="5" fill="#333"/>
  <circle cx="420" cy="60" r="5" fill="#333"/>
  <text x="50" y="48" text-anchor="middle" font-style="italic">u<tspan dy="3" font-size="8">i</tspan></text>
  <text x="430" y="48" text-anchor="middle" font-style="italic">u<tspan dy="3" font-size="8">j</tspan></text>
  <text x="240" y="45" text-anchor="middle" font-style="italic">L</text>
  <polyline points="60,120 60,160 420,120" fill="none" stroke="#0066cc" stroke-width="1.5"/>
  <text x="35" y="165" fill="#0066cc" font-style="italic">N<tspan dy="3" font-size="8">1</tspan></text>
  <text x="55" y="118" fill="#0066cc" font-size="9">1</text>
  <polyline points="60,120 420,160 420,120" fill="none" stroke="#cc3300" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="430" y="165" fill="#cc3300" font-style="italic">N<tspan dy="3" font-size="8">2</tspan></text>
  <text x="425" y="118" fill="#cc3300" font-size="9">1</text>
  <line x1="60" y1="120" x2="440" y2="120" stroke="#999" stroke-width="0.5"/>
  <text x="450" y="124" fill="#999" font-size="9" font-style="italic">x</text>
</svg>''',
    caption="Elemento barra de 2 nodos con funciones de forma lineales",
    number="1",
    width="90%",
)

text("La rigidez del elemento estándar es:")

eq_block(
    "K = (E · A)/(L) · [1, -1; -1, 1]  (13)",
)

heading("2.2 Modo incompatible para deformación", 3)

text("Se introduce el modo incompatible interno M(x) que es cero en los nodos:")

eq_block(
    "M(x) = 4 · (x)/(L) · (1 - (x)/(L))  (14)",
)

text("con su derivada (campo de deformación adicional):")

eq_block(
    "G(x) = (partial M)/(partial x) = (4)/(L) · (1 - (2x)/(L))  (15)",
)

# Figure: incompatible modes
figure(
    '''<svg width="480" height="170" viewBox="0 0 480 170" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <text x="20" y="15" font-style="italic" font-weight="600">M(x)</text>
  <line x1="50" y1="80" x2="430" y2="80" stroke="#999" stroke-width="0.5"/>
  <polyline points="50,80 240,30 430,80" fill="none" stroke="#0066cc" stroke-width="2"/>
  <text x="235" y="23" fill="#0066cc" font-size="9">max</text>
  <text x="20" y="105" font-style="italic" font-weight="600">G(x)</text>
  <line x1="50" y1="135" x2="430" y2="135" stroke="#999" stroke-width="0.5"/>
  <line x1="50" y1="115" x2="240" y2="115" stroke="#cc3300" stroke-width="2"/>
  <line x1="240" y1="155" x2="430" y2="155" stroke="#cc3300" stroke-width="2"/>
  <line x1="240" y1="115" x2="240" y2="155" stroke="#cc3300" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="130" y="110" fill="#cc3300" font-size="9">+</text>
  <text x="330" y="168" fill="#cc3300" font-size="9">−</text>
</svg>''',
    caption="Funciones de modo incompatible M(x) y su derivada G(x)",
    number="2",
    width="90%",
)

heading("2.3 Barra heterogénea con dos materiales", 3)

text("Consideremos una barra con módulo E_1 en la primera mitad y E_2 en la segunda:")

eq_block(
    "F = ∫_{0}^{L} B^T · E(x) · G  dx  (16)",
    "H = ∫_{0}^{L} G^T · E(x) · G  dx  (17)",
)

text("Calculando las integrales para una barra de longitud L dividida en x̄ = L/2:")

eq_block(
    "F = (A)/(3) · (E_1 - E_2) · [-1; 1]  (18)",
    "H = (4A)/(3L) · (E_1 + E_2)  (19)",
)

text("La matriz de rigidez efectiva resulta:")

eq_block(
    "K* = K - F · H^{-1} · F^T  (20)",
)

heading("2.4 Localización de deformaciones", 3)

text("Para problemas con ablandamiento del material (daño, fractura), "
     "el modo incompatible permite capturar la localización de deformaciones "
     "en una banda de ancho cero (discontinuidad fuerte).")

eq_block(
    "u(x) = N(x) · d + H_{S}(x - x̄) · alpha  (21)",
)

text("donde H_S es la función de Heaviside. La ley cohesiva en la discontinuidad:")

eq_block(
    "t(ū) = sigma_f - K̄ · ū  (22)",
)

# Figure: cohesive law
figure(
    '''<svg width="280" height="190" viewBox="0 0 280 190" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <line x1="40" y1="165" x2="260" y2="165" stroke="#333" stroke-width="1"/>
  <line x1="40" y1="165" x2="40" y2="20" stroke="#333" stroke-width="1"/>
  <text x="265" y="170" font-style="italic">ū</text>
  <text x="25" y="18" font-style="italic">t</text>
  <polyline points="40,40 110,40 240,165" fill="none" stroke="#0066cc" stroke-width="2"/>
  <line x1="35" y1="40" x2="40" y2="40" stroke="#333" stroke-width="1"/>
  <text x="8" y="44" font-style="italic" font-size="10">σ<tspan dy="3" font-size="7">f</tspan></text>
  <text x="150" y="90" fill="#c00" font-size="10" transform="rotate(-42,150,90)">−K̄</text>
</svg>''',
    caption="Ley cohesiva de tracción-separación en la discontinuidad",
    number="3",
    width="50%",
)

end_columns()

# ============================================================
# Page break
# ============================================================
page_break()
header(
    left="Kožar, Rukavina, Ibrahimbegović",
    right="JCE 70 (2018) 1, 19-29",
    barside="left",
    color="#F27835",
)

# ============================================================
# 3. Extensión 2D y 3D
# ============================================================
columns(2, css_columns=True)

heading("3. Extensión a 2D y 3D", 2)

heading("3.1 Elemento Q4 con modos incompatibles", 3)

text("El elemento cuadrilateral bilineal Q4 se enriquece con dos modos incompatibles "
     "en cada dirección (ξ, η) del espacio isoparamétrico:")

eq_block(
    "u = N · d + M_{xi} · alpha_{xi} + M_{eta} · alpha_{eta}  (23)",
)

eq_block(
    "M_{xi} = 1 - xi^2,  M_{eta} = 1 - eta^2  (24)",
)

# Figure: Q4 element
figure(
    '''<svg width="280" height="240" viewBox="0 0 280 240" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <polygon points="50,190 220,200 235,45 70,35" fill="#f0f4f8" stroke="#333" stroke-width="1.5"/>
  <circle cx="50" cy="190" r="4" fill="#333"/>
  <circle cx="220" cy="200" r="4" fill="#333"/>
  <circle cx="235" cy="45" r="4" fill="#333"/>
  <circle cx="70" cy="35" r="4" fill="#333"/>
  <text x="32" y="205" font-size="10">1</text>
  <text x="228" y="218" font-size="10">2</text>
  <text x="243" y="43" font-size="10">3</text>
  <text x="52" y="28" font-size="10">4</text>
  <line x1="140" y1="118" x2="200" y2="118" stroke="#c00" stroke-width="1"
        marker-end="url(#aq)"/>
  <line x1="140" y1="118" x2="140" y2="58" stroke="#c00" stroke-width="1"
        marker-end="url(#aq)"/>
  <text x="205" y="123" fill="#c00" font-style="italic">ξ</text>
  <text x="130" y="53" fill="#c00" font-style="italic">η</text>
  <defs><marker id="aq" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <path d="M0,0 L8,3 L0,6" fill="#c00"/></marker></defs>
</svg>''',
    caption="Elemento isoparamétrico Q4 con coordenadas naturales",
    number="4",
    width="50%",
)

text("Las matrices de rigidez se obtienen mediante integración de Gauss:")

eq_block(
    "K = ∫_{-1}^{1} ∫_{-1}^{1} B^T · C · B · |J|  d xi  d eta  (25)",
    "F = ∫_{-1}^{1} ∫_{-1}^{1} B^T · C · G · |J|  d xi  d eta  (26)",
    "H = ∫_{-1}^{1} ∫_{-1}^{1} G^T · C · G · |J|  d xi  d eta  (27)",
)

heading("3.2 Mejora del test de parche", 3)

text("El requisito fundamental del test de parche para modos incompatibles exige que "
     "cuando un campo de deformación constante se impone, el modo incompatible "
     "no contribuya al campo de tensiones:")

eq_block(
    "∫_{Ω_e} G^T  dΩ = 0  (28)",
)

text("Esta condición se satisface automáticamente cuando G se define como:")

eq_block(
    "G = (partial M)/(partial xi) · J^{-1} - (1)/(Ω_e) · ∫_{Ω_e} (partial M)/(partial xi) · J^{-1}  dΩ  (29)",
)

heading("3.3 Extensión a 3D", 3)

text("En el espacio tridimensional, el elemento hexaédrico H8 se enriquece con tres "
     "modos incompatibles en las coordenadas isoparamétricas (ξ, η, ζ):")

eq_block(
    "M_1 = 1 - xi^2,  M_2 = 1 - eta^2,  M_3 = 1 - zeta^2  (30)",
)

end_columns()

# ============================================================
# 4. Resultados numéricos
# ============================================================
page_break()
header(
    left="Kožar, Rukavina, Ibrahimbegović",
    right="JCE 70 (2018) 1, 19-29",
    barside="left",
    color="#F27835",
)

columns(2, css_columns=True)

heading("4. Resultados numéricos", 2)

heading("4.1 Viga en voladizo — flexión pura", 3)

text("Se analiza una viga en voladizo de longitud L = 10 m y sección b × h = 1 × 2 m, "
     "con E = 10000 MPa y ν = 0.3, sometida a un momento M en el extremo libre.")

# Figure: cantilever beam
figure(
    '''<svg width="420" height="120" viewBox="0 0 420 120" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <line x1="40" y1="35" x2="40" y2="85" stroke="#333" stroke-width="2"/>
  <line x1="30" y1="35" x2="40" y2="45" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="50" x2="40" y2="60" stroke="#333" stroke-width="1"/>
  <line x1="30" y1="65" x2="40" y2="75" stroke="#333" stroke-width="1"/>
  <rect x="40" y="40" width="320" height="40" fill="#e8f0f8" stroke="#333" stroke-width="1.5"/>
  <text x="200" y="65" text-anchor="middle" font-style="italic">L = 10 m</text>
  <path d="M 370,45 A 15,15 0 0,1 370,75" fill="none" stroke="#c00" stroke-width="1.5"
        marker-end="url(#am)"/>
  <text x="390" y="65" fill="#c00" font-weight="bold" font-style="italic">M</text>
  <defs><marker id="am" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <path d="M0,0 L8,3 L0,6" fill="#c00"/></marker></defs>
</svg>''',
    caption="Viga en voladizo con momento en el extremo libre",
    number="5",
    width="90%",
)

text("Desplazamiento vertical máximo teórico:")

eq_block(
    "w_{max} = (M · L^2)/(2 · E · I)  (31)",
)

text("Resultados de la deflexión máxima con diferentes mallas:")

table([
    ["Malla", "Q4 estándar", "Q4 + modos incomp.", "Solución exacta"],
    ["2×1",   "-3.000",      "-6.525",              "-6.635"],
    ["4×1",   "-4.125",      "-6.594",              "-6.635"],
    ["8×2",   "-5.625",      "-6.628",              "-6.635"],
    ["16×4",  "-6.328",      "-6.634",              "-6.635"],
])

note("El elemento con modos incompatibles converge mucho más rápido que el Q4 estándar", "success")

heading("4.2 Verificación con SAP2000", 3)

text("La comparación con el software comercial SAP2000 confirma los resultados:")

var("w_incomp", -6.628, "mm", "Deflexión con modos incompatibles (8×2)")
var("w_exact", -6.635, "mm", "Solución exacta")
var("error", 0.11, "%", "Error relativo")

check("error", 0.11, 1.0, "%", "<=", "Error < 1%")

heading("4.3 Ejemplo de localización 1D", 3)

text("Para una barra sometida a tracción con ablandamiento del material, "
     "el método de modos incompatibles captura la localización de la fractura:")

# Figure: force-displacement
figure(
    '''<svg width="320" height="200" viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg"
     style="font-family:Georgia,serif;font-size:11px;">
  <line x1="40" y1="180" x2="300" y2="180" stroke="#333" stroke-width="1"/>
  <line x1="40" y1="180" x2="40" y2="20" stroke="#333" stroke-width="1"/>
  <text x="305" y="185" font-style="italic">ū</text>
  <text x="25" y="18" font-style="italic">F</text>
  <polyline points="40,180 90,50 130,75" fill="none" stroke="#0066cc" stroke-width="2"/>
  <polyline points="130,75 75,180" fill="none" stroke="#0066cc" stroke-width="1.5" stroke-dasharray="4,3"/>
  <polyline points="75,180 150,75 190,110" fill="none" stroke="#0066cc" stroke-width="2"/>
  <polyline points="190,110 105,180" fill="none" stroke="#0066cc" stroke-width="1.5" stroke-dasharray="4,3"/>
  <polyline points="105,180 210,110 280,180" fill="none" stroke="#0066cc" stroke-width="2"/>
  <text x="75" y="42" fill="#0066cc" font-size="9">σ<tspan dy="3" font-size="7">f</tspan></text>
  <text x="230" y="135" fill="#666" font-size="9">softening</text>
</svg>''',
    caption="Respuesta fuerza-desplazamiento con ciclos de carga y descarga",
    number="6",
    width="55%",
)

end_columns()

# ============================================================
# 5. Conclusiones
# ============================================================
page_break()
header(
    left="Kožar, Rukavina, Ibrahimbegović",
    right="JCE 70 (2018) 1, 19-29",
    barside="left",
    color="#F27835",
)

heading("5. Conclusiones", 2)

markdown("""
El método de modos incompatibles ha demostrado ser una herramienta efectiva para:

- **Mejorar la precisión** de elementos finitos de bajo orden sin incrementar el costo computacional global
- **Eliminar el bloqueo por corte** en elementos Q4 y H8 para problemas de flexión
- **Capturar la localización** de deformaciones en problemas con ablandamiento del material
- **Modelar discontinuidades fuertes** (grietas, interfaces) sin necesidad de re-mallado

La formulación presenta una elegante conexión entre la mejora de elementos y la mecánica
de fractura, unificando ambos conceptos bajo el marco de los modos incompatibles.
""")

heading("Referencias", 2)

markdown("""
- Wilson, E.L., Taylor, R.L., Doherty, W.P., Ghaboussi, J. (1973). *Incompatible displacement models*. Numerical and Computer Methods in Structural Mechanics.
- Taylor, R.L., Beresford, P.J., Wilson, E.L. (1976). *A non-conforming element for stress analysis*. Int. J. Numerical Methods in Engineering.
- Ibrahimbegović, A., Wilson, E.L. (1991). *A modified method of incompatible modes*. Communications in Applied Numerical Methods.
- Simo, J.C., Rifai, M.S. (1990). *A class of mixed assumed strain methods and the method of incompatible modes*. Int. J. Numerical Methods in Engineering.
""")

hr()
footer(
    left="pyhekatan v0.8.0 · https://github.com/pyhekatan",
    right="DOI: 10.14256/JCE.70.2018.1.3.2078",
)

# ============================================================
# Generate HTML
# ============================================================
output_file = os.path.join(os.path.dirname(__file__), "test_jce_library_output.html")
show(output_file)
print(f"\nGenerado: {output_file}")
