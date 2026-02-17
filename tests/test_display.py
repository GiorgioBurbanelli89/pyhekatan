"""Basic tests for hekatan display functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hekatan import (matrix, eq, var, fraction, integral, title, text, clear, set_mode,
                     derivative, partial, summation, product_op, sqrt, double_integral, limit_op)
import hekatan.display as disp


def test_matrix_console():
    set_mode("console")
    matrix([[1, 2], [3, 4]], "A")


def test_eq_console():
    set_mode("console")
    eq("F", 25.5, "kN")


def test_var_console():
    set_mode("console")
    var("b", 300, "mm", "ancho")


def test_fraction_console():
    set_mode("console")
    fraction("M_u", "phi*b*d^2", "R_n")


def test_standalone_buffer():
    set_mode("standalone")
    clear()
    matrix([[1, 2], [3, 4]], "A")
    eq("F", 25.5, "kN")
    var("b", 300, "mm")
    assert len(disp._BUFFER) == 3, f"Expected 3, got {len(disp._BUFFER)}"


# === New v0.3.0 symbolic functions ===

def test_derivative_console():
    set_mode("console")
    derivative("y", "x")
    derivative("f", "t", order=2)
    derivative("u", "x", name="v")


def test_partial_console():
    set_mode("console")
    partial("u", "x")
    partial("u", "t", order=2)
    partial("u", ["x", "y"])


def test_summation_console():
    set_mode("console")
    summation("a_i", "i", "1", "n")
    summation("k^2", "k", "0", "N", "S")


def test_product_op_console():
    set_mode("console")
    product_op("a_i", "i", "1", "n", "P")


def test_sqrt_console():
    set_mode("console")
    sqrt("a^2 + b^2", "c")
    sqrt("x", index=3)


def test_double_integral_console():
    set_mode("console")
    double_integral("f(x,y)", "x", "0", "a", "y", "0", "b", "I")


def test_limit_op_console():
    set_mode("console")
    limit_op("sin(x)/x", "x", "0")
    limit_op("1/x", "x", "0", direction="+")


def test_symbolic_standalone_buffer():
    set_mode("standalone")
    clear()
    derivative("y", "x")
    partial("u", "x")
    summation("a_i", "i", "1", "n")
    product_op("x_k", "k", "1", "N")
    sqrt("a^2+b^2", "c")
    double_integral("f", "x", "0", "1", "y", "0", "1")
    limit_op("sin(x)/x", "x", "0")
    assert len(disp._BUFFER) == 7, f"Expected 7, got {len(disp._BUFFER)}"
    # Check HTML contains expected elements
    html = "\n".join(disp._BUFFER)
    assert "dvl" in html, "derivative should use dvl class"
    assert "\u2202" in html, "partial should contain ∂"
    assert "&sum;" in html, "summation should contain &sum;"
    assert "&prod;" in html, "product should contain &prod;"
    assert "sqrt-sym" in html, "sqrt should use sqrt-sym class"
    assert "&#8747;" in html, "double_integral should contain integral symbol"
    assert "lim" in html, "limit should contain lim"


def test_hekatan_markers():
    set_mode("hekatan")
    # These just print markers; verify no exceptions
    derivative("y", "x")
    partial("u", ["x", "y"])
    summation("a_i", "i", "1", "n", "S")
    product_op("x_k", "k", "1", "N")
    sqrt("x", "r", index=3)
    double_integral("f", "x", "0", "1", "y", "0", "1", "I")
    limit_op("1/x", "x", "0", direction="+", name="L")


if __name__ == "__main__":
    test_matrix_console()
    test_eq_console()
    test_var_console()
    test_fraction_console()
    test_standalone_buffer()
    test_derivative_console()
    test_partial_console()
    test_summation_console()
    test_product_op_console()
    test_sqrt_console()
    test_double_integral_console()
    test_limit_op_console()
    test_symbolic_standalone_buffer()
    test_hekatan_markers()
    print("\nAll tests passed!")
