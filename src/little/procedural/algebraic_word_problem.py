"""Algebraic word problem solver using exact 2x2 linear systems.

Translates multi-entity age, cost, and linear combination word problems
into symbolic systems solved via Cramer's rule with verifiable <math_trace>.
"""

from __future__ import annotations

import re
from fractions import Fraction

from little.procedural.algebraic_word_problem_policy import AlgebraicWordProblemPolicy
from little.procedural.math_cas import MathSolution, SymbolicCAS


class AlgebraicWordProblemSolver:
    """Zero-hallucination algebraic linear system solver for word problems."""

    def __init__(self, policy: AlgebraicWordProblemPolicy | None = None) -> None:
        self.policy = policy or AlgebraicWordProblemPolicy.default()
        self.cas = SymbolicCAS()

    def is_algebraic_problem(self, text: str) -> bool:
        """Heuristic check if input contains algebraic word problem indicators."""
        s = text.lower()
        has_mult = any(m in s for m in self.policy.multipliers)
        has_sum = any(k in s for k in self.policy.sum_keywords)
        has_offset = any(o in s for o in self.policy.positive_offsets + self.policy.negative_offsets)
        has_cost_system = bool(re.search(r"\b\d+\s+[a-zA-Z]+(?:\s+and\s+\d+\s+[a-zA-Z]+)?\s+cost\b", s))
        has_proportional = bool(re.search(r"\bas\s+many\s+[a-zA-Z]+\s+as\b", s))
        return (has_mult or has_offset or has_cost_system or has_proportional) and (
            has_sum or bool(re.search(r"\b(?:in|ago|cost|price|sum|total|together)\b", s))
        )

    def _extract_temporal_shift(self, text: str) -> int:
        """Extract signed temporal shift in years (e.g. 'in 5 years' -> +5, '4 years ago' -> -4)."""
        fut_pat = self.policy.temporal_shifts.get("future_regex", "")
        if fut_pat:
            m_fut = re.search(fut_pat, text, re.IGNORECASE)
            if m_fut:
                num_str = m_fut.group(1) or m_fut.group(2)
                if num_str:
                    return int(num_str)

        past_pat = self.policy.temporal_shifts.get("past_regex", "")
        if past_pat:
            m_past = re.search(past_pat, text, re.IGNORECASE)
            if m_past:
                num_str = m_past.group(1)
                if num_str:
                    return -int(num_str)

        return 0

    def solve(self, text: str) -> MathSolution | None:
        """Parse natural language algebraic story, solve the linear system, and answer."""
        clean = text.strip()

        # Step 1: Detect Age Problems
        res_age = self._solve_age_problem(clean)
        if res_age:
            return res_age

        # Step 2: Detect Cost / Quantity Linear Problems
        res_cost = self._solve_cost_problem(clean)
        if res_cost:
            return res_cost

        # Step 3: Detect Proportional Quantity Problems
        res_prop = self._solve_proportional_quantity_problem(clean)
        if res_prop:
            return res_prop

        return None

    def _solve_age_problem(self, text: str) -> MathSolution | None:
        """Solve age problems with multipliers/offsets and temporal sum/differences."""
        # Find 2 entities (proper nouns / names)
        # e.g. "Alice is twice as old as Bob. In 5 years, the sum of their ages will be 40. How old is Alice?"
        # Relation 1: E1 is [multiplier / offset] as old as E2 (or older/younger than E2)
        # Pattern: ([A-Z][a-zA-Z0-9_]*)\s+is\s+(.+?)\s+as\s+old\s+as\s+([A-Z][a-zA-Z0-9_]*)
        # or ([A-Z][a-zA-Z0-9_]*)\s+is\s+(\d+)\s+years\s+(older|younger)\s+than\s+([A-Z][a-zA-Z0-9_]*)

        e1: str | None = None
        e2: str | None = None
        a1: Fraction = Fraction(1)
        b1: Fraction = Fraction(0)
        c1: Fraction = Fraction(0)

        # Check multiplier pattern: "Alice is twice as old as Bob" or "Alice is 3 times as old as Bob"
        m_mult = re.search(
            r"([A-Z][a-zA-Z0-9_]*)\s+is\s+([a-zA-Z0-9_\s]+?)\s+(?:times\s+)?as\s+old\s+as\s+([A-Z][a-zA-Z0-9_]*)",
            text,
            re.IGNORECASE,
        )
        if m_mult:
            e1 = m_mult.group(1)
            mult_str = m_mult.group(2).strip().lower()
            e2 = m_mult.group(3)

            # Map multiplier
            mult_val = self.policy.multipliers.get(mult_str)
            if mult_val is None:
                # Try single number e.g. "3 times"
                m_dig = re.match(r"^(\d+(?:\.\d+)?)$", mult_str)
                if m_dig:
                    mult_val = float(m_dig.group(1))

            if mult_val is not None:
                # E1 = mult * E2  =>  1 * E1 - mult * E2 = 0
                b1 = -Fraction(mult_val).limit_denominator()
                c1 = Fraction(0)

        # Check offset pattern: "David is 4 years older than Sarah"
        if e1 is None:
            m_offset = re.search(
                r"([A-Z][a-zA-Z0-9_]*)\s+is\s+(\d+(?:\.\d+)?)\s+years?\s+([a-zA-Z\s]+?)\s+than\s+([A-Z][a-zA-Z0-9_]*)",
                text,
                re.IGNORECASE,
            )
            if m_offset:
                e1 = m_offset.group(1)
                offset_val = float(m_offset.group(2))
                rel = m_offset.group(3).strip().lower()
                e2 = m_offset.group(4)

                rel_full = f"{rel} than"
                is_positive = any(p in rel_full or p.startswith(rel) for p in self.policy.positive_offsets)
                is_negative = any(n in rel_full or n.startswith(rel) for n in self.policy.negative_offsets)

                if is_positive:
                    # E1 = E2 + offset => E1 - E2 = offset
                    b1 = Fraction(-1)
                    c1 = Fraction(offset_val).limit_denominator()
                elif is_negative:
                    # E1 = E2 - offset => E1 - E2 = -offset
                    b1 = Fraction(-1)
                    c1 = Fraction(-offset_val).limit_denominator()

        if e1 is None or e2 is None:
            return None

        # Equation 2: Sum of their ages with optional temporal shift
        # "In 5 years, the sum of their ages will be 40"
        # "4 years ago, the sum of their ages was 24"
        # "The sum of their ages is 28"
        shift_years = self._extract_temporal_shift(text)
        m_sum = re.search(
            r"(?:sum\s+of\s+(?:their|the)\s+ages|combined\s+age|sum\s+is|sum\s+will\s+be|sum\s+was|total\s+age\s+is|together\s+they\s+are)\s+(?:will\s+be\s+|is\s+|was\s+)?(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if not m_sum:
            # Try alternate phrasing: "sum of their ages will be 40"
            m_sum = re.search(r"sum\s+of\s+their\s+ages\s+(?:will\s+be|is|was)\s+(\d+(?:\.\d+)?)", text, re.IGNORECASE)

        if not m_sum:
            return None

        total_future_or_past = Fraction(m_sum.group(1)).limit_denominator()

        # (E1 + shift) + (E2 + shift) = total => E1 + E2 = total - 2 * shift
        a2 = Fraction(1)
        b2 = Fraction(1)
        c2 = total_future_or_past - 2 * Fraction(shift_years)

        # Solve via SymbolicCAS 2x2
        sol = self.cas.solve_linear_system_2x2(a1, b1, c1, a2, b2, c2)
        if sol.status != "SOLVED":
            return sol

        vals = sol.result
        x_val = vals["x"]
        y_val = vals["y"]

        # Determine target question: e.g. "How old is Alice?" or "How old is Bob?"
        q_target = None
        m_q = re.search(r"how\s+old\s+is\s+([A-Z][a-zA-Z0-9_]*)", text, re.IGNORECASE)
        if m_q:
            q_target = m_q.group(1).strip().capitalize()

        e1_cap = e1.capitalize()
        e2_cap = e2.capitalize()

        target_name = q_target or e1_cap
        if target_name.lower() == e2_cap.lower():
            target_val = y_val
            other_name = e1_cap
            other_val = x_val
        else:
            target_name = e1_cap
            target_val = x_val
            other_name = e2_cap
            other_val = y_val

        # Format steps and answer
        val_display = int(target_val) if target_val.denominator == 1 else f"{target_val.numerator}/{target_val.denominator}"
        other_display = int(other_val) if other_val.denominator == 1 else f"{other_val.numerator}/{other_val.denominator}"

        steps = [
            self.policy.templates.get("trace_header", "Algebraic Word Problem (<math_trace>):"),
            f"  Variables: x = age of {e1_cap}, y = age of {e2_cap}",
            f"  Constraint 1: {a1}x + ({b1})y = {c1}",
            f"  Constraint 2 (temporal shift={shift_years}): {a2}x + {b2}y = {c2} (adjusted from total {total_future_or_past})",
            f"  Exact Solution via Cramer's Rule: x = {int(x_val) if x_val.denominator == 1 else x_val}, y = {int(y_val) if y_val.denominator == 1 else y_val}",
            f"  Answering target query for {target_name}:",
            f"  {target_name} is {val_display} years old ({other_name} is {other_display} years old).",
        ]

        ans_tpl = self.policy.templates.get("age_answer", "{entity} is {val} years old.")
        final_answer = ans_tpl.format(entity=target_name, val=val_display)

        return MathSolution(
            status="SOLVED",
            result=final_answer,
            latex_trace=sol.latex_trace,
            steps=steps,
        )

    def _solve_cost_problem(self, text: str) -> MathSolution | None:
        """Solve cost/pricing problems: 'A pen costs $3 more than a notebook. 2 pens and 3 notebooks cost $21.'"""
        # Equation 1: Item A costs $D more/less than Item B
        # Pattern: (?:a|an)?\s*([a-zA-Z]+)\s+costs?\s+(\d+(?:\.\d+)?)\s*(?:dollars?|\$)?\s*([a-zA-Z\s]+?)\s+than\s+(?:a|an)?\s*([a-zA-Z]+)
        m_diff = re.search(
            r"(?:a|an)?\s*([a-zA-Z]+)\s+costs?\s+(\d+(?:\.\d+)?)\s*(?:dollars?|\$)?\s*(more|less)\s+than\s+(?:a|an)?\s*([a-zA-Z]+)",
            text,
            re.IGNORECASE,
        )
        if not m_diff:
            return None

        item1 = m_diff.group(1).lower()
        diff_val = float(m_diff.group(2))
        direction = m_diff.group(3).lower()
        item2 = m_diff.group(4).lower()

        # Equation 1: x - y = diff (or -diff if less)
        a1 = Fraction(1)
        b1 = Fraction(-1)
        c1 = Fraction(diff_val if direction == "more" else -diff_val).limit_denominator()

        # Equation 2: N1 items1 and N2 items2 cost $TOTAL
        # e.g. "2 pens and 3 notebooks cost 21 dollars"
        m_combo = re.search(
            r"(\d+)\s+([a-zA-Z]+)\s+and\s+(\d+)\s+([a-zA-Z]+)\s+cost\s+(\d+(?:\.\d+)?)\s*(?:dollars?|\$)?",
            text,
            re.IGNORECASE,
        )
        if not m_combo:
            return None

        q_a = int(m_combo.group(1))
        name_a = m_combo.group(2).lower().rstrip("s")
        q_b = int(m_combo.group(3))
        total_cost = Fraction(m_combo.group(5)).limit_denominator()

        # Match name_a and name_b to item1 and item2
        if name_a.startswith(item1[:3]):
            a2 = Fraction(q_a)
            b2 = Fraction(q_b)
        else:
            a2 = Fraction(q_b)
            b2 = Fraction(q_a)
        c2 = total_cost

        sol = self.cas.solve_linear_system_2x2(a1, b1, c1, a2, b2, c2)
        if sol.status != "SOLVED":
            return sol

        vals = sol.result
        x_val = vals["x"]
        y_val = vals["y"]

        # Target question
        # "What is the cost of a pen?"
        q_item = item1
        m_q = re.search(r"cost\s+of\s+(?:a|an|the)?\s*([a-zA-Z]+)", text, re.IGNORECASE)
        if m_q:
            raw_q = m_q.group(1).lower().rstrip("s")
            if raw_q.startswith(item2[:3]):
                q_item = item2

        target_val = x_val if q_item == item1 else y_val
        val_display = int(target_val) if target_val.denominator == 1 else f"{target_val.numerator}/{target_val.denominator}"

        ans_tpl = self.policy.templates.get("cost_answer", "The cost of a {entity} is ${val}.")
        final_answer = ans_tpl.format(entity=q_item, val=val_display)

        steps = [
            self.policy.templates.get("trace_header", "Algebraic Word Problem (<math_trace>):"),
            f"  Variables: x = cost of {item1}, y = cost of {item2}",
            f"  Constraint 1: {a1}x + ({b1})y = {c1}",
            f"  Constraint 2: {a2}x + {b2}y = {c2}",
            f"  Exact Solution via Cramer's Rule: x = ${x_val}, y = ${y_val}",
            f"  {final_answer}",
        ]

        return MathSolution(
            status="SOLVED",
            result=final_answer,
            latex_trace=sol.latex_trace,
            steps=steps,
        )

    def _solve_proportional_quantity_problem(self, text: str) -> MathSolution | None:
        """Solve proportional quantity problems: 'Lucas has 3 times as many books as Maya. Together they have 48 books. How many books does Lucas have?'"""
        # Relation 1: E1 has [multiplier] as many [noun] as E2
        m_prop = re.search(
            r"([A-Z][a-zA-Z0-9_]*)\s+has\s+([a-zA-Z0-9_\s]+?)\s+(?:times\s+)?as\s+many\s+([a-zA-Z0-9_]+)\s+as\s+([A-Z][a-zA-Z0-9_]*)",
            text,
            re.IGNORECASE,
        )
        if not m_prop:
            return None

        e1 = m_prop.group(1).strip()
        mult_str = m_prop.group(2).strip().lower()
        noun = m_prop.group(3).strip()
        e2 = m_prop.group(4).strip()

        mult_val = self.policy.multipliers.get(mult_str)
        if mult_val is None:
            mult_val = self.policy.multipliers.get(f"{mult_str} times")
        if mult_val is None:
            m_dig = re.match(r"^(\d+(?:\.\d+)?)$", mult_str)
            if m_dig:
                mult_val = float(m_dig.group(1))

        if mult_val is None:
            return None

        # Equation 1: 1 * E1 - mult * E2 = 0
        a1 = Fraction(1)
        b1 = -Fraction(mult_val).limit_denominator()
        c1 = Fraction(0)

        # Equation 2: E1 + E2 = total
        m_sum = re.search(
            r"(?:together\s+(?:they\s+have)?|the\s+sum\s+of\s+their\s+[a-zA-Z0-9_]+\s+is|in\s+total\s+(?:they\s+have)?|combined\s+they\s+have|they\s+have)\s*(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if not m_sum:
            return None

        total_val = Fraction(m_sum.group(1)).limit_denominator()
        a2 = Fraction(1)
        b2 = Fraction(1)
        c2 = total_val

        sol = self.cas.solve_linear_system_2x2(a1, b1, c1, a2, b2, c2)
        if sol.status != "SOLVED":
            return sol

        vals = sol.result
        x_val = vals["x"]
        y_val = vals["y"]

        # Target entity: "How many books does Lucas have?"
        e1_cap = e1.capitalize()
        e2_cap = e2.capitalize()
        target_name = e1_cap

        m_q = re.search(r"how\s+many\s+(?:[a-zA-Z0-9_]+\s+)?does\s+([A-Z][a-zA-Z0-9_]*)\s+have", text, re.IGNORECASE)
        if m_q:
            cand = m_q.group(1).strip().capitalize()
            if cand.lower() == e2_cap.lower():
                target_name = e2_cap
            elif cand.lower() == e1_cap.lower():
                target_name = e1_cap

        if target_name.lower() == e2_cap.lower():
            target_val = y_val
            other_name = e1_cap
            other_val = x_val
        else:
            target_name = e1_cap
            target_val = x_val
            other_name = e2_cap
            other_val = y_val

        val_display = int(target_val) if target_val.denominator == 1 else f"{target_val.numerator}/{target_val.denominator}"
        other_display = int(other_val) if other_val.denominator == 1 else f"{other_val.numerator}/{other_val.denominator}"

        ans_tpl = self.policy.templates.get("quantity_entity_answer", "{entity} has {val} {noun}.")
        final_answer = ans_tpl.format(entity=target_name, val=val_display, noun=noun)

        steps = [
            self.policy.templates.get("trace_header", "Algebraic Word Problem (<math_trace>):"),
            f"  Variables: x = {noun} of {e1_cap}, y = {noun} of {e2_cap}",
            f"  Constraint 1: {a1}x + ({b1})y = {c1}",
            f"  Constraint 2: {a2}x + {b2}y = {c2}",
            f"  Exact Solution via Cramer's Rule: x = {int(x_val) if x_val.denominator == 1 else x_val}, y = {int(y_val) if y_val.denominator == 1 else y_val}",
            f"  Answering target query for {target_name}:",
            f"  {final_answer} ({other_name} has {other_display} {noun}).",
        ]

        return MathSolution(
            status="SOLVED",
            result=final_answer,
            latex_trace=sol.latex_trace,
            steps=steps,
        )

