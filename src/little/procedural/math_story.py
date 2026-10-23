"""Generalized Math Word Problem Solver.

Decomposes natural language story problems into quantities, state transitions,
proportional rates, and fractional partitions with verifiable <math_trace> output.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from little.procedural.algebraic_word_problem import AlgebraicWordProblemSolver
from little.procedural.math_cas import MathSolution, SymbolicCAS, UnitConversionGraph
from little.procedural.unit_conversion_policy import UnitConversionPolicy


def _is_num(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return bool(s) and s.count(".") <= 1 and s.replace(".", "", 1).isdigit()


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    curr: list[str] = []
    for ch in text:
        curr.append(ch)
        if ch in ".!?;\n":
            seg = "".join(curr).strip()
            if seg:
                out.append(seg)
            curr = []
    tail = "".join(curr).strip()
    if tail:
        out.append(tail)
    return out


def _split_clauses(sent: str) -> list[str]:
    delims = (", and then ", ", then ", ", and ", " and then ", " then ", ",")
    parts = [sent]
    for d in delims:
        next_parts: list[str] = []
        for p in parts:
            low_p = p.lower()
            start = 0
            while True:
                idx = low_p.find(d, start)
                if idx == -1:
                    tail = p[start:].strip()
                    if tail:
                        next_parts.append(tail)
                    break
                piece = p[start:idx].strip()
                if piece:
                    next_parts.append(piece)
                start = idx + len(d)
        parts = next_parts
    return parts



@dataclass
class StoryAction:
    action_type: str  # "INITIAL", "SUBTRACT", "ADD", "SCALE", "PARTITION"
    quantity: float | Fraction
    unit: str
    description: str


class MathStorySolver:
    """Zero-hallucination semantic solver for multi-step math word problems."""

    def __init__(self, unit_policy: UnitConversionPolicy | None = None) -> None:
        self.cas = SymbolicCAS()
        self.unit_graph = UnitConversionGraph(unit_policy)
        self.algebraic_solver = AlgebraicWordProblemSolver()

    def is_math_story(self, text: str) -> bool:
        """Detect if input text constitutes a math story / word problem."""
        s = text.strip()
        if self.algebraic_solver.is_algebraic_problem(s):
            return True
        if not any(_is_num(w.strip(".,?!:;()$")) for w in s.split()):
            return False
        low = s.lower()
        return any(
            q in low
            for q in ("how many", "how much", "how far", "what is the total")
        )

    @staticmethod
    def _parse_direct_unit_question(clean: str) -> tuple[str, float, str] | None:
        stripped = clean.strip()
        if stripped.endswith("?"):
            stripped = stripped[:-1].strip()
        tokens = stripped.split()
        low = [t.lower() for t in tokens]
        if len(tokens) < 5 or low[0] != "how" or low[1] not in ("many", "much"):
            return None
        to_unit = tokens[2]
        idx = 3
        if idx < len(low) and low[idx] in ("are", "is"):
            idx += 1
            if idx < len(low) and low[idx] == "there":
                idx += 1
        if idx >= len(low) or low[idx] != "in":
            return None
        idx += 1
        if idx + 1 != len(tokens) - 1 or not _is_num(tokens[idx]):
            return None
        return to_unit, float(tokens[idx]), tokens[idx + 1]

    @staticmethod
    def _parse_rate_time(clean: str) -> tuple[float, str, float, str] | None:
        tokens = [t.rstrip(".,?!") for t in clean.split() if t.rstrip(".,?!")]
        low = [t.lower() for t in tokens]
        motion_verbs = {"travel", "travels", "drive", "drives", "run", "runs", "fly", "flies", "walk", "walks"}
        speed_units = {"km/h", "kph", "mph", "m/s"}
        time_units = {"hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins", "second", "seconds", "s"}
        for i in range(len(tokens)):
            if low[i] in motion_verbs and i + 6 < len(tokens) and low[i + 1] == "at":
                if (
                    _is_num(tokens[i + 2])
                    and low[i + 3] in speed_units
                    and low[i + 4] == "for"
                    and _is_num(tokens[i + 5])
                    and low[i + 6] in time_units
                ):
                    return (
                        float(tokens[i + 2]),
                        tokens[i + 3],
                        float(tokens[i + 5]),
                        tokens[i + 6],
                    )
        return None

    @staticmethod
    def _parse_multi_item_costs(clean: str) -> list[tuple[str, str, str]]:
        tokens = [t.strip(".,?!") for t in clean.split() if t.strip(".,?!")]
        low = [t.lower() for t in tokens]
        matches: list[tuple[str, str, str]] = []
        i = 0
        while i + 3 < len(tokens):
            if (
                tokens[i].isdigit()
                and tokens[i + 1].isalpha()
                and low[i + 2] in ("for", "at")
            ):
                price_tok = tokens[i + 3].lstrip("$")
                if _is_num(price_tok):
                    matches.append((tokens[i], tokens[i + 1], price_tok))
                    i += 4
                    continue
            i += 1
        return matches

    def solve_story(self, text: str) -> MathSolution | None:
        """Parse story problem, track quantity state transitions, and return exact MathSolution."""
        clean = text.strip()

        # 0. Algebraic linear systems (age problems, cost systems, etc.)
        if self.algebraic_solver.is_algebraic_problem(clean):
            alg_sol = self.algebraic_solver.solve(clean)
            if alg_sol:
                return alg_sol

        # 0b. Direct unit conversion question: e.g. "How many feet in 36 inches?"
        unit_q = self._parse_direct_unit_question(clean)
        if unit_q is not None:
            to_unit, val, from_unit = unit_q
            conv = self.unit_graph.convert(val, from_unit, to_unit)
            if conv.status == "SOLVED" and conv.result is not None:
                res_num = float(conv.result)
                final_res = int(res_num) if res_num.is_integer() else res_num
                return MathSolution(
                    status="SOLVED",
                    result=final_res,
                    latex_trace=conv.latex_trace,
                    steps=conv.steps,
                )

        # 1. Rate * Time problem: e.g. "travels at 60 km/h for 3 hours. How far does..."
        rate_q = self._parse_rate_time(clean)
        if rate_q is not None and "how far" in clean.lower():
            speed, speed_unit, time_val, time_unit = rate_q

            normalized_speed = self.unit_graph.convert(
                speed, speed_unit, "km/h"
            )
            normalized_time = self.unit_graph.convert(time_val, time_unit, "hr")
            if normalized_speed.status != "SOLVED" or normalized_time.status != "SOLVED":
                return None
            speed_kmh = float(normalized_speed.result)
            time_hrs = float(normalized_time.result)

            dist = speed_kmh * time_hrs
            dist_res = int(dist) if dist.is_integer() else round(dist, 2)
            steps = [
                "Rate & Distance Problem:",
                f"  Speed v = {speed} {speed_unit}",
                f"  Time t = {time_val} {time_unit} (= {time_hrs} hours)",
                "  Formula: Distance d = v * t",
                f"  Normalized speed: {speed_kmh} km/h",
                f"  Calculation: {speed_kmh} * {time_hrs} = {dist_res} km",
            ]
            latex = f"d = {speed_kmh} \\times {time_hrs} = {dist_res}"
            return MathSolution(
                status="SOLVED",
                result=dist_res,
                latex_trace=latex,
                steps=steps,
            )

        # 2. Multi-item cost / scaling sum:
        item_matches = self._parse_multi_item_costs(clean)
        if len(item_matches) >= 2 and (
            "how much" in clean.lower() or "total" in clean.lower()
        ):
            total_cost = 0.0
            steps = ["Multi-item Cost Calculation:"]
            sub_terms = []
            for qty_str, item_name, price_str in item_matches:
                q_val = float(qty_str)
                p_val = float(price_str)
                cost = q_val * p_val
                total_cost += cost
                q_int = int(q_val) if q_val.is_integer() else q_val
                p_int = int(p_val) if p_val.is_integer() else p_val
                c_int = int(cost) if cost.is_integer() else cost
                steps.append(f"  {q_int} {item_name} @ ${p_int} each = ${c_int}")
                sub_terms.append(f"({q_int} * {p_int})")

            total_res = int(total_cost) if total_cost.is_integer() else round(total_cost, 2)
            steps.append(f"Total spent: {' + '.join(sub_terms)} = ${total_res}")
            latex = f"\\text{{Total}} = {total_res}"
            return MathSolution(
                status="SOLVED",
                result=total_res,
                latex_trace=latex,
                steps=steps,
            )

        # 3. Sequential Changes and Partitions on an Entity's Inventory:
        sentences = _split_sentences(clean)

        current_val: float | None = None
        has_explicit_init = False
        has_transition = False
        item_name = "items"
        steps: list[str] = ["Sequential State Tracking:"]

        # Verbs classifying actions
        subtract_verbs = (
            "gives away", "gave away", "gives", "gave", "lost", "loses",
            "eats", "ate", "sells", "sold", "spends", "spent", "drops", "dropped",
        )
        add_verbs = (
            "buys", "bought", "finds", "found", "gets", "got", "receives", "received",
            "picks", "picked", "earns", "earned", "adds", "added",
        )

        # Step 3a: Look for Initial inventory in the first sentence or clause
        for sent in sentences:
            toks = [t.strip(".,?!") for t in sent.split() if t.strip(".,?!")]
            low_toks = [t.lower() for t in toks]
            for i in range(len(toks)):
                if low_toks[i] in ("has", "had", "owns") and i + 2 < len(toks):
                    if _is_num(toks[i + 1]) and toks[i + 2].isalpha():
                        current_val = float(toks[i + 1])
                        item_name = toks[i + 2]
                        has_explicit_init = True
                        break
                if (
                    low_toks[i] == "started"
                    and i + 3 < len(toks)
                    and low_toks[i + 1] == "with"
                    and _is_num(toks[i + 2])
                    and toks[i + 3].isalpha()
                ):
                    current_val = float(toks[i + 2])
                    item_name = toks[i + 3]
                    has_explicit_init = True
                    break
            if current_val is not None:
                c_disp = int(current_val) if current_val.is_integer() else current_val
                steps.append(f"  Initial: {c_disp} {item_name}")
                break

        if current_val is None:
            toks = [t.strip(".,?!") for t in clean.split() if t.strip(".,?!")]
            for i in range(len(toks) - 1):
                if _is_num(toks[i]) and toks[i + 1].isalpha():
                    current_val = float(toks[i])
                    item_name = toks[i + 1]
                    c_disp = int(current_val) if current_val.is_integer() else current_val
                    steps.append(f"  Initial: {c_disp} {item_name}")
                    break

        if current_val is None:
            return None

        # Step 3b: Process state transition sentences / clauses
        clauses = [c for sent in sentences for c in _split_clauses(sent)]

        for clause in clauses:
            low_clause = clause.lower()
            if "how many" in low_clause or "how much" in low_clause or clause.endswith("?"):
                continue

            c_toks = [t.strip(".,?!") for t in clause.split() if t.strip(".,?!")]
            c_low = [t.lower() for t in c_toks]

            # Fractional partition check: "gives half of them"
            if "half" in c_low and any(
                v in low_clause
                for v in ("give", "gives", "gave", "ate", "eat", "eats", "lost", "lose", "loses")
            ):
                part = current_val * 0.5
                current_val -= part
                has_transition = True
                c_disp = int(current_val) if current_val.is_integer() else current_val
                p_disp = int(part) if part.is_integer() else part
                steps.append(f"  Partition (- half): subtracted {p_disp}, remaining = {c_disp} {item_name}")
                continue

            # Subtract actions:
            sub_matched = False
            for sv in subtract_verbs:
                sv_parts = sv.split()
                k = len(sv_parts)
                for i in range(len(c_low) - k):
                    if c_low[i : i + k] == sv_parts and _is_num(c_toks[i + k]):
                        sub_qty = float(c_toks[i + k])
                        old_val = current_val
                        current_val -= sub_qty
                        has_transition = True
                        c_disp = int(current_val) if current_val.is_integer() else current_val
                        s_disp = int(sub_qty) if sub_qty.is_integer() else sub_qty
                        o_disp = int(old_val) if old_val.is_integer() else old_val
                        steps.append(f"  Subtract ({sv} {s_disp}): {o_disp} - {s_disp} = {c_disp} {item_name}")
                        sub_matched = True
                        break
                if sub_matched:
                    break

            if sub_matched:
                continue

            # Add actions:
            for av in add_verbs:
                av_parts = av.split()
                k = len(av_parts)
                added = False
                for i in range(len(c_low) - k):
                    if c_low[i : i + k] == av_parts and _is_num(c_toks[i + k]):
                        add_qty = float(c_toks[i + k])
                        old_val = current_val
                        current_val += add_qty
                        has_transition = True
                        c_disp = int(current_val) if current_val.is_integer() else current_val
                        a_disp = int(add_qty) if add_qty.is_integer() else add_qty
                        o_disp = int(old_val) if old_val.is_integer() else old_val
                        steps.append(f"  Add ({av} {a_disp}): {o_disp} + {a_disp} = {c_disp} {item_name}")
                        added = True
                        break
                if added:
                    break

        if not has_explicit_init and not has_transition:
            return None

        final_res = int(current_val) if current_val.is_integer() else round(current_val, 2)
        steps.append(f"Final Quantity: {final_res} {item_name}")
        latex = f"\\text{{Result}} = {final_res}"
        return MathSolution(
            status="SOLVED",
            result=final_res,
            latex_trace=latex,
            steps=steps,
        )
