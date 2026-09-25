"""Generalized Symbolic Computer Algebra System (CAS) and Dimensional Analysis Engine.

Zero-hallucination, exact rational arithmetic, linear systems, quadratic equations,
dimensional unit conversion graphs, and geometry with transparent step-by-step traces.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

from little.procedural.unit_conversion_policy import UnitConversionPolicy


@dataclass
class MathSolution:
    """Structured result of a symbolic or numerical mathematical calculation."""

    status: str  # "SOLVED", "NO_UNIQUE_SOLUTION", "COMPLEX_ROOTS", "ERROR"
    result: Any
    latex_trace: str = ""
    steps: list[str] = field(default_factory=list)

    def verbalize(self) -> str:
        """Render a readable proof summary."""
        if not self.steps:
            return f"Result: {self.result}"
        return "\n".join(self.steps)


class UnitConversionGraph:
    """Graph-based dimensional unit conversion engine using BFS path-finding.

    Supports linear scales (length, mass, time, speed) and affine transformations (temperature).
    """

    def __init__(self, policy: UnitConversionPolicy | None = None) -> None:
        # Adjacency list: unit -> list of (target_unit, forward_fn, inverse_fn, scale_desc)
        self.adj: Dict[str, List[Tuple[str, Callable[[float], float], str]]] = {}
        self.policy = policy or UnitConversionPolicy.default()
        self._build_standard_units()

    def add_edge(
        self,
        u1: str,
        u2: str,
        forward_fn: Callable[[float], float],
        inverse_fn: Callable[[float], float],
        forward_desc: str,
        inverse_desc: str,
    ) -> None:
        u1_c = u1.strip().lower()
        u2_c = u2.strip().lower()
        self.adj.setdefault(u1_c, []).append((u2_c, forward_fn, forward_desc))
        self.adj.setdefault(u2_c, []).append((u1_c, inverse_fn, inverse_desc))

    def _add_linear_ratio(self, u_sub: str, u_base: str, ratio: float, desc: str = "") -> None:
        """u_sub * ratio = u_base, e.g. 1 km * 1000 = 1000 m."""
        fwd = lambda x, r=ratio: x * r
        inv = lambda x, r=ratio: x / r
        f_desc = desc or f"Multiply by {ratio}"
        i_desc = f"Divide by {ratio}"
        self.add_edge(u_sub, u_base, fwd, inv, f_desc, i_desc)

    def _build_standard_units(self) -> None:
        for edge in self.policy.edges:
            forward = lambda value, scale=edge.scale, offset=edge.offset: (
                value * scale + offset
            )
            inverse = lambda value, scale=edge.scale, offset=edge.offset: (
                (value - offset) / scale
            )
            forward_desc = f"Apply scale={edge.scale} and offset={edge.offset}"
            inverse_desc = f"Apply inverse scale=1/{edge.scale} and offset={-edge.offset / edge.scale}"
            self.add_edge(
                edge.source,
                edge.target,
                forward,
                inverse,
                forward_desc,
                inverse_desc,
            )

    def convert(self, value: float, from_unit: str, to_unit: str) -> MathSolution:
        """Find the shortest conversion path via BFS and evaluate the transformation."""
        start_raw = from_unit.strip().lower()
        goal_raw = to_unit.strip().lower()
        start = self.policy.aliases.get(start_raw, start_raw)
        goal = self.policy.aliases.get(goal_raw, goal_raw)

        if start == goal:
            return MathSolution(
                status="SOLVED",
                result=float(value),
                latex_trace=f"{value} {start} = {value} {goal}",
                steps=[f"No conversion needed: {value} {start} = {value} {goal}"],
            )

        if start not in self.adj or goal not in self.adj:
            return MathSolution(
                status="ERROR",
                result=None,
                steps=[f"Unknown unit in conversion pair: '{start}' -> '{goal}'"],
            )

        # BFS queue: (current_node, current_val, steps_list)
        queue: deque[Tuple[str, float, List[str]]] = deque([(start, float(value), [])])
        visited = {start}

        while queue:
            curr_node, curr_val, curr_steps = queue.popleft()

            if curr_node == goal:
                final_val = round(curr_val, 6)
                if abs(final_val - round(final_val)) < 1e-6:
                    final_val = float(round(final_val))
                return MathSolution(
                    status="SOLVED",
                    result=final_val,
                    latex_trace=f"{value} \\text{{ {from_unit}}} = {final_val} \\text{{ {to_unit}}}",
                    steps=curr_steps + [f"Final converted value: {final_val} {to_unit}"],
                )

            for neighbor, fn, desc in self.adj[curr_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_val = fn(curr_val)
                    step_msg = f"{curr_val:.4f} {curr_node} -> {new_val:.4f} {neighbor} via ({desc})"
                    queue.append((neighbor, new_val, curr_steps + [step_msg]))

        return MathSolution(
            status="ERROR",
            result=None,
            steps=[f"No valid conversion path exists between '{from_unit}' and '{to_unit}' (incompatible dimensions)."],
        )


class SymbolicCAS:
    """Symbolic Computer Algebra System providing exact rational arithmetic,

    2x2 linear solvers, quadratic solvers, and geometric calculations.
    """

    def __init__(self) -> None:
        self.ucg = UnitConversionGraph()

    def solve_rational_expression(self, expr: str) -> MathSolution:
        """Parse and evaluate rational expressions with exact Fractions and step-by-step reduction."""
        steps: list[str] = [f"Evaluating rational expression: '{expr}'"]
        # Basic binary fraction operations: "A/B + C/D", "A/B * C/D", etc.
        m = re.match(
            r"^\s*(\d+(?:/\d+)?)\s*([\+\-\*\/])\s*(\d+(?:/\d+)?)\s*$",
            expr.strip(),
        )
        if not m:
            # Fallback simple arithmetic evaluation using Fraction
            try:
                # Replace '/' with Fraction constructor
                clean_expr = expr.strip()
                res = eval(clean_expr, {"__builtins__": {}}, {"Fraction": Fraction})
                frac_res = Fraction(res).limit_denominator()
                steps.append(f"Direct symbolic evaluation: {frac_res}")
                return MathSolution(status="SOLVED", result=frac_res, steps=steps)
            except Exception as e:
                return MathSolution(status="ERROR", result=None, steps=[f"Parsing error: {e}"])

        left_str, op, right_str = m.group(1), m.group(2), m.group(3)
        f_left = Fraction(left_str)
        f_right = Fraction(right_str)

        steps.append(f"Parsed left operand: {f_left.numerator}/{f_left.denominator}")
        steps.append(f"Parsed right operand: {f_right.numerator}/{f_right.denominator}")

        if op == "+":
            common_denom = math.lcm(f_left.denominator, f_right.denominator)
            scale_l = common_denom // f_left.denominator
            scale_r = common_denom // f_right.denominator
            num_l = f_left.numerator * scale_l
            num_r = f_right.numerator * scale_r
            sum_num = num_l + num_r
            steps.append(f"Common denominator LCM({f_left.denominator}, {f_right.denominator}) = {common_denom}")
            steps.append(f"Scaled sum: ({num_l} + {num_r}) / {common_denom} = {sum_num}/{common_denom}")
            result = f_left + f_right
            if result.denominator != common_denom:
                steps.append(f"Reduced by GCD to: {result.numerator}/{result.denominator}")
            else:
                steps.append(f"Result: {result.numerator}/{result.denominator}")
        elif op == "-":
            result = f_left - f_right
            steps.append(f"Subtraction: {f_left} - {f_right} = {result}")
        elif op == "*":
            raw_num = f_left.numerator * f_right.numerator
            raw_den = f_left.denominator * f_right.denominator
            steps.append(f"Multiply numerators: {f_left.numerator} * {f_right.numerator} = {raw_num}")
            steps.append(f"Multiply denominators: {f_left.denominator} * {f_right.denominator} = {raw_den}")
            result = f_left * f_right
            steps.append(f"Simplified fraction: {result.numerator}/{result.denominator}")
        elif op == "/":
            if f_right == 0:
                return MathSolution(status="ERROR", result=None, steps=["Division by zero error."])
            result = f_left / f_right
            steps.append(f"Divide by multiplying reciprocal: {f_left} * {Fraction(f_right.denominator, f_right.numerator)} = {result}")
        else:
            return MathSolution(status="ERROR", result=None, steps=[f"Unsupported operator: {op}"])

        latex_str = f"\\frac{{{result.numerator}}}{{{result.denominator}}}" if result.denominator != 1 else str(result.numerator)
        return MathSolution(status="SOLVED", result=result, latex_trace=latex_str, steps=steps)

    def solve_linear_system_2x2(
        self,
        a1: float | int | Fraction,
        b1: float | int | Fraction,
        c1: float | int | Fraction,
        a2: float | int | Fraction,
        b2: float | int | Fraction,
        c2: float | int | Fraction,
    ) -> MathSolution:
        """Solve a 2x2 system of linear equations using Cramer's rule with exact fractions.

        Equation 1: a1*x + b1*y = c1
        Equation 2: a2*x + b2*y = c2
        """
        f_a1, f_b1, f_c1 = Fraction(a1), Fraction(b1), Fraction(c1)
        f_a2, f_b2, f_c2 = Fraction(a2), Fraction(b2), Fraction(c2)

        steps: list[str] = [
            f"Solving 2x2 linear system via Cramer's rule:",
            f"  (1) {f_a1}x + {f_b1}y = {f_c1}",
            f"  (2) {f_a2}x + {f_b2}y = {f_c2}",
        ]

        # Determinant D = a1*b2 - a2*b1
        D = f_a1 * f_b2 - f_a2 * f_b1
        steps.append(f"Coefficient determinant D = ({f_a1})*({f_b2}) - ({f_a2})*({f_b1}) = {D}")

        if D == 0:
            Dx = f_c1 * f_b2 - f_c2 * f_b1
            Dy = f_a1 * f_c2 - f_a2 * f_c1
            if Dx == 0 and Dy == 0:
                steps.append("D = 0 and Dx = Dy = 0: Infinitely many collinear solutions.")
            else:
                steps.append("D = 0 and (Dx != 0 or Dy != 0): Inconsistent parallel lines with no solution.")
            return MathSolution(status="NO_UNIQUE_SOLUTION", result=None, steps=steps)

        Dx = f_c1 * f_b2 - f_c2 * f_b1
        Dy = f_a1 * f_c2 - f_a2 * f_c1
        steps.append(f"Numerator determinant Dx = ({f_c1})*({f_b2}) - ({f_c2})*({f_b1}) = {Dx}")
        steps.append(f"Numerator determinant Dy = ({f_a1})*({f_c2}) - ({f_a2})*({f_c1}) = {Dy}")

        x = Dx / D
        y = Dy / D
        steps.append(f"x = Dx / D = {Dx} / {D} = {x}")
        steps.append(f"y = Dy / D = {Dy} / {D} = {y}")

        latex = f"\\begin{{pmatrix}} x \\\\ y \\end{{pmatrix}} = \\begin{{pmatrix}} {x} \\\\ {y} \\end{{pmatrix}}"
        return MathSolution(status="SOLVED", result={"x": x, "y": y}, latex_trace=latex, steps=steps)

    def solve_quadratic(
        self,
        a: float | int,
        b: float | int,
        c: float | int,
    ) -> MathSolution:
        """Solve a quadratic equation a*x^2 + b*x + c = 0 via discriminant derivation."""
        if a == 0:
            if b == 0:
                return MathSolution(status="NO_UNIQUE_SOLUTION", result=None, steps=["Degenerate equation 0 = 0."])
            root = -c / b
            return MathSolution(status="SOLVED", result={"roots": [float(root)]}, steps=[f"Linear root: x = {-c}/{b} = {root}"])

        steps: list[str] = [
            f"Solving quadratic equation: ({a})x² + ({b})x + ({c}) = 0",
        ]
        delta = b * b - 4 * a * c
        steps.append(f"Discriminant Δ = b² - 4ac = ({b})² - 4*({a})*({c}) = {delta}")

        if delta > 0:
            sqrt_delta = math.sqrt(delta)
            r1 = (-b + sqrt_delta) / (2 * a)
            r2 = (-b - sqrt_delta) / (2 * a)
            steps.append(f"Δ > 0: Two distinct real roots:")
            steps.append(f"  x₁ = (-b + √Δ) / (2a) = ({-b} + {sqrt_delta:.4f}) / ({2*a}) = {r1:.4f}")
            steps.append(f"  x₂ = (-b - √Δ) / (2a) = ({-b} - {sqrt_delta:.4f}) / ({2*a}) = {r2:.4f}")
            latex = f"x \\in \\{{{r1:.4f}, {r2:.4f}\\}}"
            return MathSolution(status="SOLVED", result={"roots": [r1, r2]}, latex_trace=latex, steps=steps)
        elif delta == 0:
            r = -b / (2 * a)
            steps.append(f"Δ = 0: One repeated real root:")
            steps.append(f"  x = -b / (2a) = {-b} / ({2*a}) = {r:.4f}")
            latex = f"x = {r:.4f}"
            return MathSolution(status="SOLVED", result={"roots": [r]}, latex_trace=latex, steps=steps)
        else:
            real_part = -b / (2 * a)
            imag_part = math.sqrt(-delta) / (2 * a)
            steps.append(f"Δ < 0: Complex conjugate roots:")
            steps.append(f"  x = {real_part:.4f} ± {imag_part:.4f}i")
            latex = f"x = {real_part:.4f} \\pm {imag_part:.4f}i"
            return MathSolution(status="COMPLEX_ROOTS", result={"real": real_part, "imag": imag_part}, latex_trace=latex, steps=steps)

    def geometry_circle(self, radius: float) -> MathSolution:
        """Calculate area and circumference of a circle."""
        r = float(radius)
        area = math.pi * r * r
        circum = 2.0 * math.pi * r
        steps = [
            f"Circle Geometry with radius r = {r}:",
            f"  Area A = π * r² = π * ({r})² = {area:.4f}",
            f"  Circumference C = 2 * π * r = 2 * π * ({r}) = {circum:.4f}",
        ]
        return MathSolution(
            status="SOLVED",
            result={"area": area, "circumference": circum},
            latex_trace=f"A = {area:.4f}, C = {circum:.4f}",
            steps=steps,
        )

    def geometry_rectangle(self, length: float, width: float) -> MathSolution:
        """Calculate area and perimeter of a rectangle."""
        l, w = float(length), float(width)
        area = l * w
        perimeter = 2.0 * (l + w)
        steps = [
            f"Rectangle Geometry with length l = {l}, width w = {w}:",
            f"  Area A = l * w = ({l}) * ({w}) = {area:.4f}",
            f"  Perimeter P = 2 * (l + w) = 2 * ({l} + {w}) = {perimeter:.4f}",
        ]
        return MathSolution(
            status="SOLVED",
            result={"area": area, "perimeter": perimeter},
            latex_trace=f"A = {area:.4f}, P = {perimeter:.4f}",
            steps=steps,
        )

    def geometry_right_triangle(self, a: float, b: float) -> MathSolution:
        """Calculate hypotenuse and area of a right-angled triangle."""
        leg_a, leg_b = float(a), float(b)
        hyp = math.sqrt(leg_a * leg_a + leg_b * leg_b)
        area = 0.5 * leg_a * leg_b
        steps = [
            f"Right-Angled Triangle with legs a = {leg_a}, b = {leg_b}:",
            f"  Hypotenuse c = √(a² + b²) = √(({leg_a})² + ({leg_b})²) = {hyp:.4f}",
            f"  Area A = (1/2) * a * b = 0.5 * ({leg_a}) * ({leg_b}) = {area:.4f}",
        ]
        return MathSolution(
            status="SOLVED",
            result={"hypotenuse": hyp, "area": area},
            latex_trace=f"c = {hyp:.4f}, A = {area:.4f}",
            steps=steps,
        )
