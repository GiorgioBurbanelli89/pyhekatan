"""Basic tests for hekatan display functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hekatan import matrix, eq, var, fraction, title, text, clear, set_mode
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


if __name__ == "__main__":
    test_matrix_console()
    test_eq_console()
    test_var_console()
    test_fraction_console()
    test_standalone_buffer()
    print("\nAll tests passed!")
