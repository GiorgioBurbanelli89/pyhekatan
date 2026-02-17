"""
Rectangular Slab FEA - Bogner-Fox-Schmit (BFS) Element
=======================================================
Replicates the Hekatan Calc example using Python + numpy + hekatan display.

Simply supported rectangular slab under uniform load.
Uses 16-DOF BFS plate bending element with Kirchhoff theory.
Compares results with SAP2000 and analytical (Navier) solution.
"""

import numpy as np
import sys
import os

# Use local dev version if available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hekatan import (
    title, heading, text, var, eq, fraction, matrix, table,
    partial, double_integral, sqrt, summation, show, clear
)

clear()

# ============================================================
# Title
# ============================================================
title("Finite Element Analysis of Rectangular Slab", 1)
text("Bogner-Fox-Schmit (BFS) 16-DOF plate bending element.")
text("Simply supported rectangular slab under uniform load.")
text("Kirchhoff thin plate theory.")

# ============================================================
# Input Data
# ============================================================
heading("Input Data", 2)

a_val = 6.0   # m - dimension along x
b_val = 4.0   # m - dimension along y
t_val = 0.1   # m - thickness
E_val = 35000  # MPa = 35 GPa
nu_val = 0.15
q_val = -10.0  # kN/m² (downward)

var("a", a_val, "m", "slab dimension along x")
var("b", b_val, "m", "slab dimension along y")
var("t", t_val, "m", "slab thickness")
var("E", E_val, "MPa", "modulus of elasticity")
var("nu", nu_val, "", "Poisson's ratio")
var("q", q_val, "kN/m^2", "uniform load (downward)")

# Mesh parameters
heading("Finite Element Mesh", 2)

na = 6  # elements along a
nb = 4  # elements along b
ne = na * nb  # total elements
nj = (na + 1) * (nb + 1)  # total joints
dof_per_node = 4  # w, dw/dx, dw/dy, d²w/dxdy

var("n_a", na, "", "elements along a")
var("n_b", nb, "", "elements along b")
eq("n_e", ne, "")
text(f"Total number of elements: {ne}")
eq("n_j", nj, "")
text(f"Total number of joints: {nj}")

ae = a_val / na  # element dimension along x
be = b_val / nb  # element dimension along y

var("a_e", f"{ae:.4f}", "m", "element size along x")
var("b_e", f"{be:.4f}", "m", "element size along y")
text(f"DOFs per node: {dof_per_node} (w, dw/dx, dw/dy, d²w/dxdy)")
eq("n_dof", nj * dof_per_node)

# ============================================================
# Flexural Rigidity
# ============================================================
heading("Flexural Rigidity", 2)

fraction("E * t^3", "12 * (1 - nu^2)", "D")

D_val = E_val * 1000 * t_val**3 / (12 * (1 - nu_val**2))  # N·m (E in kPa)
# E_val is in MPa = 1000 kN/m² = 1e6 N/m²
D_val = E_val * 1e6 * t_val**3 / (12 * (1 - nu_val**2))  # N·m

var("D", f"{D_val:.2f}", "N*m", "flexural rigidity")

# ============================================================
# Constitutive Matrix
# ============================================================
heading("Constitutive Matrix", 2)

text("Stress-strain relationship for Kirchhoff plate:")

D_matrix = D_val * np.array([
    [1,      nu_val, 0],
    [nu_val, 1,      0],
    [0,      0,      (1 - nu_val) / 2]
])

# Display D matrix symbolically
text("D = D₀ · [C]")
C_display = [
    [1, "nu", 0],
    ["nu", 1, 0],
    [0, 0, "(1-nu)/2"]
]
matrix(C_display, "C")

# ============================================================
# BFS Shape Functions
# ============================================================
heading("BFS Shape Functions", 2)

text("The BFS element uses Hermite cubic polynomials in each direction.")
text("Each node has 4 DOFs: w, theta_x = dw/dx, theta_y = dw/dy, theta_xy = d²w/dxdy")

text("Hermite polynomials along xi (0 to 1):")
var("H_1", "1 - 3*xi^2 + 2*xi^3")
var("H_2", "xi*(1 - xi)^2 * a_e")
var("H_3", "3*xi^2 - 2*xi^3")
var("H_4", "xi^2*(xi - 1) * a_e")

text("Similarly along eta (0 to 1) with b_e.")

# ============================================================
# Element Stiffness Matrix (numerical integration)
# ============================================================
heading("Element Stiffness Matrix", 2)

text("The element stiffness matrix is computed by numerical integration:")
double_integral("B^T * C * B * |J|", "xi", "0", "1", "eta", "0", "1", "K_e")
text("Using 4x4 Gauss-Legendre quadrature.")


def hermite_functions(xi, L):
    """Hermite basis functions and derivatives for BFS element."""
    H1 = 1 - 3*xi**2 + 2*xi**3
    H2 = xi * (1 - xi)**2 * L
    H3 = 3*xi**2 - 2*xi**3
    H4 = xi**2 * (xi - 1) * L

    dH1 = (-6*xi + 6*xi**2) / L
    dH2 = (1 - 4*xi + 3*xi**2)
    dH3 = (6*xi - 6*xi**2) / L
    dH4 = (-2*xi + 3*xi**2)

    ddH1 = (-6 + 12*xi) / L**2
    ddH2 = (-4 + 6*xi) / L
    ddH3 = (6 - 12*xi) / L**2
    ddH4 = (-2 + 6*xi) / L

    return (H1, H2, H3, H4), (dH1, dH2, dH3, dH4), (ddH1, ddH2, ddH3, ddH4)


def bfs_stiffness(ae_l, be_l, D_mat):
    """Compute 16x16 BFS element stiffness matrix using Gauss quadrature."""
    ndof = 16
    K = np.zeros((ndof, ndof))

    # 4-point Gauss quadrature on [0,1]
    gp = np.array([
        0.5 - np.sqrt(3/7 + 2/7*np.sqrt(6/5)) / 2,
        0.5 - np.sqrt(3/7 - 2/7*np.sqrt(6/5)) / 2,
        0.5 + np.sqrt(3/7 - 2/7*np.sqrt(6/5)) / 2,
        0.5 + np.sqrt(3/7 + 2/7*np.sqrt(6/5)) / 2,
    ])
    gw = np.array([
        (18 - np.sqrt(30)) / 72,
        (18 + np.sqrt(30)) / 72,
        (18 + np.sqrt(30)) / 72,
        (18 - np.sqrt(30)) / 72,
    ])

    for i in range(4):
        for j in range(4):
            xi = gp[i]
            eta = gp[j]
            w = gw[i] * gw[j]

            Hx, dHx, ddHx = hermite_functions(xi, ae_l)
            Hy, dHy, ddHy = hermite_functions(eta, be_l)

            # Build B matrix (3 x 16)
            # Curvatures: kappa_x = -d²w/dx², kappa_y = -d²w/dy², kappa_xy = -2*d²w/dxdy
            # Node order: 1(i,j), 2(i+1,j), 3(i+1,j+1), 4(i,j+1)
            # DOFs per node: w, dw/dx, dw/dy, d²w/dxdy

            B = np.zeros((3, 16))

            # Shape functions: N_k = Hx_m(xi) * Hy_n(eta)
            # For node 1: (Hx1,Hy1), (Hx2,Hy1), (Hx1,Hy2), (Hx2,Hy2)
            nodes_hx = [0, 2, 2, 0]  # indices into Hx for each node
            nodes_hy = [0, 0, 2, 2]  # indices into Hy for each node

            for n in range(4):  # 4 nodes
                ix1 = nodes_hx[n]
                iy1 = nodes_hy[n]

                for d in range(4):  # 4 DOFs per node
                    col = n * 4 + d

                    if d == 0:    # w
                        hx_idx, hy_idx = ix1, iy1
                    elif d == 1:  # dw/dx
                        hx_idx, hy_idx = ix1 + 1, iy1
                    elif d == 2:  # dw/dy
                        hx_idx, hy_idx = ix1, iy1 + 1
                    else:         # d²w/dxdy
                        hx_idx, hy_idx = ix1 + 1, iy1 + 1

                    # kappa_x = -d²N/dx²
                    B[0, col] = -ddHx[hx_idx] * Hy[hy_idx]
                    # kappa_y = -d²N/dy²
                    B[1, col] = -Hx[hx_idx] * ddHy[hy_idx]
                    # kappa_xy = -2 * d²N/dxdy
                    B[2, col] = -2 * dHx[hx_idx] * dHy[hy_idx]

            # K += B^T * D * B * |J| * w
            # Jacobian for [0,1] mapping: |J| = ae * be
            K += w * ae_l * be_l * (B.T @ D_mat @ B)

    return K


def bfs_load_vector(ae_l, be_l, q):
    """Compute consistent load vector for uniform load."""
    ndof = 16
    f = np.zeros(ndof)

    gp = np.array([
        0.5 - np.sqrt(3/7 + 2/7*np.sqrt(6/5)) / 2,
        0.5 - np.sqrt(3/7 - 2/7*np.sqrt(6/5)) / 2,
        0.5 + np.sqrt(3/7 - 2/7*np.sqrt(6/5)) / 2,
        0.5 + np.sqrt(3/7 + 2/7*np.sqrt(6/5)) / 2,
    ])
    gw = np.array([
        (18 - np.sqrt(30)) / 72,
        (18 + np.sqrt(30)) / 72,
        (18 + np.sqrt(30)) / 72,
        (18 - np.sqrt(30)) / 72,
    ])

    for i in range(4):
        for j in range(4):
            xi = gp[i]
            eta = gp[j]
            w = gw[i] * gw[j]

            Hx, _, _ = hermite_functions(xi, ae_l)
            Hy, _, _ = hermite_functions(eta, be_l)

            nodes_hx = [0, 2, 2, 0]
            nodes_hy = [0, 0, 2, 2]

            N = np.zeros(16)
            for n in range(4):
                ix1 = nodes_hx[n]
                iy1 = nodes_hy[n]
                for d in range(4):
                    col = n * 4 + d
                    if d == 0:
                        hx_idx, hy_idx = ix1, iy1
                    elif d == 1:
                        hx_idx, hy_idx = ix1 + 1, iy1
                    elif d == 2:
                        hx_idx, hy_idx = ix1, iy1 + 1
                    else:
                        hx_idx, hy_idx = ix1 + 1, iy1 + 1
                    N[col] = Hx[hx_idx] * Hy[hy_idx]

            f += w * ae_l * be_l * q * N

    return f


# Compute element stiffness
Ke = bfs_stiffness(ae, be, D_matrix)
text(f"Element stiffness matrix: {Ke.shape[0]}x{Ke.shape[1]}")

# ============================================================
# Mesh Generation
# ============================================================
heading("Mesh Generation", 2)

text("Joint coordinates and element connectivity.")

# Joint coordinates
coords = np.zeros((nj, 2))
for j_idx in range(nb + 1):
    for i_idx in range(na + 1):
        node = j_idx * (na + 1) + i_idx
        coords[node, 0] = i_idx * ae  # x
        coords[node, 1] = j_idx * be  # y

# Element connectivity (4 nodes per element)
# Node numbering: bottom-left, bottom-right, top-right, top-left
conn = np.zeros((ne, 4), dtype=int)
for j_idx in range(nb):
    for i_idx in range(na):
        el = j_idx * na + i_idx
        n1 = j_idx * (na + 1) + i_idx         # bottom-left
        n2 = n1 + 1                             # bottom-right
        n3 = n2 + (na + 1)                      # top-right
        n4 = n1 + (na + 1)                      # top-left
        conn[el] = [n1, n2, n3, n4]

text(f"Joints: {nj}, Elements: {ne}")

# ============================================================
# Boundary Conditions
# ============================================================
heading("Boundary Conditions", 2)

text("Simply supported: w = 0 on all edges.")
text("Rotations theta_x and theta_y are free.")

# Identify boundary nodes
bc_nodes = set()
for i in range(na + 1):
    bc_nodes.add(i)                        # bottom edge
    bc_nodes.add(nb * (na + 1) + i)       # top edge
for j in range(nb + 1):
    bc_nodes.add(j * (na + 1))            # left edge
    bc_nodes.add(j * (na + 1) + na)       # right edge

text(f"Supported joints: {len(bc_nodes)}")

# DOF mapping
total_dof = nj * dof_per_node
eq("N_dof,total", total_dof)

# ============================================================
# Global Assembly
# ============================================================
heading("Global Assembly & Solution", 2)

text("Assembling global stiffness matrix and load vector...")

K_global = np.zeros((total_dof, total_dof))
F_global = np.zeros(total_dof)

# Element load vector
fe = bfs_load_vector(ae, be, q_val * 1000)  # convert kN/m² to N/m²

for el in range(ne):
    nodes = conn[el]
    # DOF indices for this element
    dofs = []
    for n in nodes:
        for d in range(dof_per_node):
            dofs.append(n * dof_per_node + d)

    # Assemble
    for i_loc in range(16):
        F_global[dofs[i_loc]] += fe[i_loc]
        for j_loc in range(16):
            K_global[dofs[i_loc], dofs[j_loc]] += Ke[i_loc, j_loc]

# Apply boundary conditions (w = 0 at supported nodes)
# Also fix corners: w, dw/dx, dw/dy, d²w/dxdy at corner nodes
corner_nodes = [
    0, na, nb * (na + 1), nb * (na + 1) + na
]

# Simply supported: only w = 0 at boundary nodes
# Rotations (dw/dx, dw/dy) remain FREE
constrained = set()
for n in bc_nodes:
    constrained.add(n * dof_per_node)  # w = 0 only

constrained = sorted(constrained)
free_dofs = [i for i in range(total_dof) if i not in constrained]

text(f"Total DOFs: {total_dof}")
text(f"Constrained DOFs: {len(constrained)}")
text(f"Free DOFs: {len(free_dofs)}")

# Solve reduced system
K_ff = K_global[np.ix_(free_dofs, free_dofs)]
F_ff = F_global[free_dofs]

text("Solving system of equations...")
U_free = np.linalg.solve(K_ff, F_ff)

# Full displacement vector
U = np.zeros(total_dof)
for i, dof in enumerate(free_dofs):
    U[dof] = U_free[i]

# ============================================================
# Results - Displacements
# ============================================================
heading("Results - Joint Displacements", 2)

text("Vertical displacements at interior joints (mm):")

# Find max displacement
w_all = np.array([U[n * dof_per_node] for n in range(nj)])
w_max_idx = np.argmin(w_all)  # most negative = max deflection
w_max = w_all[w_max_idx] * 1000  # convert to mm

# Center node
center_node = (nb // 2) * (na + 1) + (na // 2)
w_center = U[center_node * dof_per_node] * 1000  # mm

var("w_max", f"{w_max:.4f}", "mm", f"max deflection (node {w_max_idx + 1})")
var("w_center", f"{w_center:.4f}", "mm", f"deflection at center (node {center_node + 1})")

# Display displacement grid
heading("Displacement Map (mm)", 3)
disp_grid = []
for j_idx in range(nb, -1, -1):  # top to bottom for display
    row = []
    for i_idx in range(na + 1):
        node = j_idx * (na + 1) + i_idx
        w_mm = U[node * dof_per_node] * 1000
        row.append(f"{w_mm:.3f}")
    disp_grid.append(row)

matrix(disp_grid, "w")

# ============================================================
# Results - Bending Moments
# ============================================================
heading("Results - Bending Moments", 2)

text("Computing bending moments at element centers...")


def compute_moments_at_center(el_idx, U_vec, conn_arr, ae_l, be_l, D_mat):
    """Compute Mx, My, Mxy at element center (xi=0.5, eta=0.5)."""
    nodes = conn_arr[el_idx]
    xi, eta = 0.5, 0.5

    Hx, dHx, ddHx = hermite_functions(xi, ae_l)
    Hy, dHy, ddHy = hermite_functions(eta, be_l)

    nodes_hx = [0, 2, 2, 0]
    nodes_hy = [0, 0, 2, 2]

    B = np.zeros((3, 16))
    for n in range(4):
        ix1 = nodes_hx[n]
        iy1 = nodes_hy[n]
        for d in range(4):
            col = n * 4 + d
            if d == 0:
                hx_idx, hy_idx = ix1, iy1
            elif d == 1:
                hx_idx, hy_idx = ix1 + 1, iy1
            elif d == 2:
                hx_idx, hy_idx = ix1, iy1 + 1
            else:
                hx_idx, hy_idx = ix1 + 1, iy1 + 1
            B[0, col] = -ddHx[hx_idx] * Hy[hy_idx]
            B[1, col] = -Hx[hx_idx] * ddHy[hy_idx]
            B[2, col] = -2 * dHx[hx_idx] * dHy[hy_idx]

    # Get element DOFs
    dofs = []
    for n_node in nodes:
        for d in range(4):
            dofs.append(n_node * 4 + d)

    u_e = U_vec[dofs]
    M = D_mat @ B @ u_e  # [Mx, My, Mxy] in N·m/m

    return M / 1000  # convert to kN·m/m


# Compute moments for all elements
Mx_all = np.zeros(ne)
My_all = np.zeros(ne)
Mxy_all = np.zeros(ne)

for el in range(ne):
    M = compute_moments_at_center(el, U, conn, ae, be, D_matrix)
    Mx_all[el] = M[0]
    My_all[el] = M[1]
    Mxy_all[el] = M[2]

var("M_x,max", f"{np.max(np.abs(Mx_all)):.2f}", "kN*m/m", "max |Mx|")
var("M_y,max", f"{np.max(np.abs(My_all)):.2f}", "kN*m/m", "max |My|")
var("M_xy,max", f"{np.max(np.abs(Mxy_all)):.2f}", "kN*m/m", "max |Mxy|")

# ============================================================
# Analytical Solution (Navier)
# ============================================================
heading("Analytical Solution (Navier Series)", 2)

text("Double Fourier series solution for simply supported plate:")
fraction("q * a^4", "pi^4 * D", "w_0")
text("multiplied by the series summation.")

summation("a_mn * sin(m*pi*x/a) * sin(n*pi*y/b)", "m,n", "1", "infty", "w(x,y)")

text("where:")
fraction("1", "m * n * ((m/a)^2 + (n/b)^2)^2")

# Compute analytical deflection at center
w_analytical = 0
Mx_analytical = 0
My_analytical = 0

x_c = a_val / 2
y_c = b_val / 2

for m in range(1, 40, 2):  # odd terms only
    for n in range(1, 40, 2):
        amn = 16 * q_val * 1000 / (np.pi**6 * D_val * m * n)
        denom = (m / a_val)**2 + (n / b_val)**2
        w_mn = amn / denom**2
        sin_val = np.sin(m * np.pi * x_c / a_val) * np.sin(n * np.pi * y_c / b_val)
        w_analytical += w_mn * sin_val

        # Moments at center
        Mx_mn = -D_val * w_mn * (-(m * np.pi / a_val)**2 - nu_val * (n * np.pi / b_val)**2) * sin_val
        My_mn = -D_val * w_mn * (-nu_val * (m * np.pi / a_val)**2 - (n * np.pi / b_val)**2) * sin_val
        Mx_analytical += Mx_mn
        My_analytical += My_mn

w_analytical_mm = w_analytical * 1000  # to mm
Mx_analytical_kNm = Mx_analytical / 1000  # to kN·m/m
My_analytical_kNm = My_analytical / 1000

heading("Analytical Results", 3)
var("w_analytical", f"{w_analytical_mm:.4f}", "mm", "deflection at center")
var("M_x,analytical", f"{Mx_analytical_kNm:.2f}", "kN*m/m", "Mx at center")
var("M_y,analytical", f"{My_analytical_kNm:.2f}", "kN*m/m", "My at center")

# ============================================================
# Comparison
# ============================================================
heading("Comparison of Results", 2)

text("FEM vs Analytical solution at slab center:")

err_w = abs((w_center - w_analytical_mm) / w_analytical_mm) * 100

# Center element moments
center_el = (nb // 2 - 1) * na + (na // 2 - 1)  # element at center region
# Average of 4 center elements
center_els = []
for dj in range(2):
    for di in range(2):
        el_idx = (nb // 2 - 1 + dj) * na + (na // 2 - 1 + di)
        center_els.append(el_idx)

Mx_fem_center = np.mean([Mx_all[e] for e in center_els])
My_fem_center = np.mean([My_all[e] for e in center_els])

err_Mx = abs((Mx_fem_center - Mx_analytical_kNm) / Mx_analytical_kNm) * 100
err_My = abs((My_fem_center - My_analytical_kNm) / My_analytical_kNm) * 100

comparison = [
    ["", "FEM", "Analytical", "Error %"],
    ["w (mm)", f"{w_center:.4f}", f"{w_analytical_mm:.4f}", f"{err_w:.2f}"],
    ["Mx (kNm/m)", f"{Mx_fem_center:.2f}", f"{Mx_analytical_kNm:.2f}", f"{err_Mx:.2f}"],
    ["My (kNm/m)", f"{My_fem_center:.2f}", f"{My_analytical_kNm:.2f}", f"{err_My:.2f}"],
]
table(comparison)

# ============================================================
# SAP2000 Reference
# ============================================================
heading("SAP2000 Reference Data", 2)

text("SAP2000 results for the same model (from Hekatan Calc example):")
var("w_SAP2000", -6.529, "mm", "max deflection (node 18)")

err_sap = abs((w_center - (-6.529)) / (-6.529)) * 100
var("error_vs_SAP", f"{err_sap:.2f}", "%", "FEM vs SAP2000")

# ============================================================
# Show
# ============================================================
heading("Notes", 2)
text("This analysis uses the BFS (Bogner-Fox-Schmit) conforming plate element.")
text("The element has 16 DOFs (4 per node: w, dw/dx, dw/dy, d²w/dxdy).")
text("C1 continuity is maintained across element boundaries.")
text("Generated with <b>hekatan</b> Python library (pip install hekatan).")

show("rectangular_slab_fea.html")
