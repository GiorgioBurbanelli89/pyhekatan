"""Quick test for _format_subscript fixes."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')
from hekatan.display import _format_subscript, _parse_calc_eq

print("=== _format_subscript tests ===")
print("T1 v^2):", _format_subscript("v^2)"))
print("T2 ∂^2M_x:", _format_subscript("∂^2M_x"))
print("T3 N_x + N_y:", _format_subscript("N_x + N_y"))
print("T4 ∇^4:", _format_subscript("∇^4"))
print("T5 σ_x*dz:", _format_subscript("σ_x*dz"))
print("T6 12*(1 - ν^2):", _format_subscript("12*(1 - ν^2)"))
print("T7 N_x*stuff + N_y*stuff:", _format_subscript("N_x*stuff + N_y*more"))
print("T8 τ_{xy}:", _format_subscript("τ_{xy}"))
print("T9 ω^2:", _format_subscript("ω^2"))

print("\n=== _parse_calc_eq tests ===")
# Eq 1.1 - partial derivatives
eq11 = "(∂^2M_x)/(∂x^2) + 2*(∂^2M_{xy})/(∂x*∂y) + (∂^2M_y)/(∂y^2) + q = 0    (1.1)"
print("Eq 1.1:", _parse_calc_eq(eq11))

# Eq 1.3 - flexural rigidity
eq13 = "D = (E*h^3)/(12*(1 - ν^2))    (1.3)"
print("\nEq 1.3:", _parse_calc_eq(eq13))

# Eq 1.7 - stability
eq17 = "D*∇^4*w + N_x*(∂^2w)/(∂x^2) + 2*N_{xy}*(∂^2w)/(∂x*∂y) + N_y*(∂^2w)/(∂y^2) = 0    (1.7)"
print("\nEq 1.7:", _parse_calc_eq(eq17))

# Eq 1.5a - integral
eq15a = "N_x = ∫_{-h/2}^{h/2} σ_x*dz    (1.5a)"
print("\nEq 1.5a:", _parse_calc_eq(eq15a))

print("\nDone!")
