import pytest
from fractions import Fraction
import math
from little.procedural.math_cas import (
    SymbolicCAS,
    MathSolution,
    UnitConversionGraph,
)


def test_rational_fraction_arithmetic():
    cas = SymbolicCAS()

    # Exact rational arithmetic with simplification trace
    sol = cas.solve_rational_expression("2/3 + 3/4")
    assert sol.status == "SOLVED"
    assert sol.result == Fraction(17, 12)
    assert len(sol.steps) >= 2
    assert "17/12" in sol.steps[-1]

    sol_mul = cas.solve_rational_expression("5/6 * 3/10")
    assert sol_mul.status == "SOLVED"
    assert sol_mul.result == Fraction(1, 4)


def test_linear_system_2x2():
    cas = SymbolicCAS()

    # 2x + 3y = 12
    # x - y = 1
    # Expected: x = 3, y = 2
    sol = cas.solve_linear_system_2x2(
        a1=2, b1=3, c1=12,
        a2=1, b2=-1, c2=1,
    )
    assert sol.status == "SOLVED"
    assert sol.result == {"x": Fraction(3, 1), "y": Fraction(2, 1)}
    assert any("Cramer's rule" in step or "determinant" in step.lower() for step in sol.steps)

    # Singular / parallel system: 2x + 4y = 8 and x + 2y = 5 -> D = 0
    sol_sing = cas.solve_linear_system_2x2(
        a1=2, b1=4, c1=8,
        a2=1, b2=2, c2=5,
    )
    assert sol_sing.status == "NO_UNIQUE_SOLUTION"


def test_quadratic_solver():
    cas = SymbolicCAS()

    # x^2 - 5x + 6 = 0 -> roots 2 and 3
    sol = cas.solve_quadratic(a=1, b=-5, c=6)
    assert sol.status == "SOLVED"
    roots = sorted(sol.result["roots"])
    assert roots == [2.0, 3.0]
    assert any("discriminant" in step.lower() for step in sol.steps)

    # Negative discriminant
    sol_neg = cas.solve_quadratic(a=1, b=0, c=4)
    assert sol_neg.status == "COMPLEX_ROOTS"


def test_unit_conversions():
    ucg = UnitConversionGraph()

    # Speed: 72 km/h to m/s -> 20 m/s
    sol_speed = ucg.convert(72, "km/h", "m/s")
    assert sol_speed.status == "SOLVED"
    assert math.isclose(sol_speed.result, 20.0, rel_tol=1e-5)

    # Length: 5 km to meters -> 5000 m
    sol_len = ucg.convert(5, "km", "m")
    assert sol_len.status == "SOLVED"
    assert math.isclose(sol_len.result, 5000.0, rel_tol=1e-5)

    # Temperature: 100 Celsius to Fahrenheit -> 212 °F
    sol_temp = ucg.convert(100, "celsius", "fahrenheit")
    assert sol_temp.status == "SOLVED"
    assert math.isclose(sol_temp.result, 212.0, rel_tol=1e-5)

    # Temperature: 32 Fahrenheit to Celsius -> 0 °C
    sol_temp2 = ucg.convert(32, "fahrenheit", "celsius")
    assert sol_temp2.status == "SOLVED"
    assert math.isclose(sol_temp2.result, 0.0, abs_tol=1e-5)


def test_geometric_calculations():
    cas = SymbolicCAS()

    # Circle with radius 7
    sol_c = cas.geometry_circle(radius=7)
    assert sol_c.status == "SOLVED"
    assert math.isclose(sol_c.result["area"], math.pi * 49, rel_tol=1e-5)
    assert math.isclose(sol_c.result["circumference"], 2 * math.pi * 7, rel_tol=1e-5)

    # Rectangle: length 8, width 5
    sol_r = cas.geometry_rectangle(length=8, width=5)
    assert sol_r.status == "SOLVED"
    assert sol_r.result["area"] == 40
    assert sol_r.result["perimeter"] == 26

    # Right triangle hypotenuse: 3 and 4 -> 5
    sol_t = cas.geometry_right_triangle(a=3, b=4)
    assert sol_t.status == "SOLVED"
    assert math.isclose(sol_t.result["hypotenuse"], 5.0, rel_tol=1e-5)
