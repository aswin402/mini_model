"""Algebraic word problem solver using exact 2x2 linear systems.

Translates multi-entity age, cost, and linear combination word problems
into symbolic systems solved via Cramer's rule with verifiable <math_trace>.
"""

from __future__ import annotations

from fractions import Fraction

from little.procedural.algebraic_word_problem_policy import AlgebraicWordProblemPolicy
from little.procedural.math_cas import MathSolution, SymbolicCAS


def _is_num(tok: str) -> bool:
    s = tok.strip().lstrip("$")
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return bool(s) and s.count(".") <= 1 and s.replace(".", "", 1).isdigit()



class AlgebraicWordProblemSolver:
    """Zero-hallucination algebraic linear system solver for word problems."""

    def __init__(self, policy: AlgebraicWordProblemPolicy | None = None) -> None:
        self.policy = policy or AlgebraicWordProblemPolicy.default()
        self.cas = SymbolicCAS()

    def is_algebraic_problem(self, text: str) -> bool:
        """Heuristic check if input contains algebraic word problem indicators."""
        s = text.lower()
        toks = [t.strip(".,?!:;()") for t in s.split() if t.strip(".,?!:;()")]
        has_mult = any(m in s for m in self.policy.multipliers)
        has_sum = any(k in s for k in self.policy.sum_keywords)
        has_offset = any(o in s for o in self.policy.positive_offsets + self.policy.negative_offsets)
        has_cost_system = "cost" in toks and any(
            toks[i].isdigit() and toks[i + 1].isalpha()
            for i in range(len(toks) - 1)
        )
        has_proportional = any(
            toks[i] == "as" and toks[i + 1] == "many" and toks[i + 2].isalpha() and toks[i + 3] == "as"
            for i in range(len(toks) - 3)
        )
        return (has_mult or has_offset or has_cost_system or has_proportional) and (
            has_sum or any(w in toks for w in ("in", "ago", "cost", "price", "sum", "total", "together"))
        )

    def _extract_temporal_shift(self, text: str) -> int:
        """Extract signed temporal shift in years (e.g. 'in 5 years' -> +5, '4 years ago' -> -4)."""
        toks = [t.strip(".,?!:;()") for t in text.lower().split() if t.strip(".,?!:;()")]
        for i, tok in enumerate(toks):
            if tok == "in" and i + 2 < len(toks) and toks[i + 1].isdigit() and toks[i + 2] in ("year", "years"):
                return int(toks[i + 1])
            if tok.isdigit() and i + 2 < len(toks) and toks[i + 1] in ("year", "years"):
                if toks[i + 2] in ("later", "from"):
                    return int(tok)
                if toks[i + 2] in ("ago", "earlier", "before"):
                    return -int(tok)
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
        toks = [t.strip(".,?!:;()") for t in text.split() if t.strip(".,?!:;()")]
        low = [t.lower() for t in toks]

        e1: str | None = None
        e2: str | None = None
        a1: Fraction = Fraction(1)
        b1: Fraction = Fraction(0)
        c1: Fraction = Fraction(0)

        # Check multiplier pattern: "Alice is twice as old as Bob" or "Alice is 3 times as old as Bob"
        for i in range(1, len(toks)):
            if low[i] == "is":
                for j in range(i + 2, min(len(toks) - 3, i + 6)):
                    if low[j : j + 3] == ["as", "old", "as"] and j + 3 < len(toks):
                        cand_e1 = toks[i - 1]
                        cand_e2 = toks[j + 3]
                        mult_words = low[i + 1 : j]
                        if mult_words and mult_words[-1] == "times" and len(mult_words) > 1:
                            mult_clean = " ".join(mult_words[:-1])
                        else:
                            mult_clean = " ".join(mult_words)
                        mult_val = self.policy.multipliers.get(" ".join(mult_words))
                        if mult_val is None:
                            mult_val = self.policy.multipliers.get(mult_clean)
                        if mult_val is None and _is_num(mult_clean):
                            mult_val = float(mult_clean)
                        if mult_val is not None:
                            e1 = cand_e1
                            e2 = cand_e2
                            b1 = -Fraction(mult_val).limit_denominator()
                            c1 = Fraction(0)
                            break
                if e1 is not None:
                    break

        # Check offset pattern: "David is 4 years older than Sarah"
        if e1 is None:
            for i in range(1, len(toks) - 5):
                if (
                    low[i] == "is"
                    and _is_num(toks[i + 1])
                    and low[i + 2] in ("year", "years")
                    and low[i + 4] == "than"
                ):
                    cand_e1 = toks[i - 1]
                    offset_val = float(toks[i + 1])
                    rel = low[i + 3]
                    cand_e2 = toks[i + 5]
                    rel_full = f"{rel} than"
                    is_positive = any(p in rel_full or p.startswith(rel) for p in self.policy.positive_offsets)
                    is_negative = any(n in rel_full or n.startswith(rel) for n in self.policy.negative_offsets)
                    if is_positive:
                        e1, e2 = cand_e1, cand_e2
                        b1 = Fraction(-1)
                        c1 = Fraction(offset_val).limit_denominator()
                        break
                    if is_negative:
                        e1, e2 = cand_e1, cand_e2
                        b1 = Fraction(-1)
                        c1 = Fraction(-offset_val).limit_denominator()
                        break

        if e1 is None or e2 is None:
            return None

        # Equation 2: Sum of their ages with optional temporal shift
        shift_years = self._extract_temporal_shift(text)
        total_str: str | None = None
        for i in range(len(toks)):
            if low[i] in ("ages", "age", "sum", "together") or (
                i > 0 and low[i - 1] in ("ages", "age", "sum", "together")
            ):
                # Look ahead up to 4 tokens for a number after is/was/will be
                for k in range(i + 1, min(len(toks), i + 5)):
                    if _is_num(toks[k]) and (
                        k == 0 or low[k - 1] not in ("in", "years", "year")
                    ):
                        if any(v in low[i:k] for v in ("is", "be", "was", "are")):
                            total_str = toks[k]
                            break
            if total_str is not None:
                break

        if total_str is None:
            return None

        total_future_or_past = Fraction(total_str).limit_denominator()

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
        for i in range(len(toks) - 3):
            if low[i : i + 3] == ["how", "old", "is"]:
                q_target = toks[i + 3].strip().capitalize()
                break

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
        toks = [t.strip(".,?!:;()") for t in text.split() if t.strip(".,?!:;()")]
        low = [t.lower() for t in toks]

        item1: str | None = None
        diff_val: float = 0.0
        direction: str = "more"
        item2: str | None = None

        for i in range(1, len(toks) - 3):
            if low[i] in ("cost", "costs") and _is_num(toks[i + 1]):
                cand_item1 = low[i - 1]
                cand_diff = float(toks[i + 1].lstrip("$"))
                k = i + 2
                if k < len(low) and low[k] in ("dollar", "dollars", "$"):
                    k += 1
                if k + 1 < len(low) and low[k] in ("more", "less") and low[k + 1] == "than":
                    dir_tok = low[k]
                    m_idx = k + 2
                    if m_idx < len(low) and low[m_idx] in ("a", "an", "the"):
                        m_idx += 1
                    if m_idx < len(low) and low[m_idx].isalpha():
                        item1 = cand_item1
                        diff_val = cand_diff
                        direction = dir_tok
                        item2 = low[m_idx]
                        break

        if item1 is None or item2 is None:
            return None

        a1 = Fraction(1)
        b1 = Fraction(-1)
        c1 = Fraction(diff_val if direction == "more" else -diff_val).limit_denominator()

        # Equation 2: "<q_a> <name_a> and <q_b> <name_b> cost <total_cost>"
        q_a: int | None = None
        name_a: str = ""
        q_b: int | None = None
        total_cost: Fraction | None = None

        for i in range(len(toks) - 5):
            if (
                toks[i].isdigit()
                and low[i + 1].isalpha()
                and low[i + 2] == "and"
                and toks[i + 3].isdigit()
                and low[i + 4].isalpha()
                and low[i + 5] in ("cost", "costs")
                and i + 6 < len(toks)
                and _is_num(toks[i + 6])
            ):
                q_a = int(toks[i])
                name_a = low[i + 1].rstrip("s")
                q_b = int(toks[i + 3])
                total_cost = Fraction(toks[i + 6].lstrip("$")).limit_denominator()
                break

        if q_a is None or q_b is None or total_cost is None:
            return None

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

        q_item = item1
        for i in range(len(low) - 2):
            if low[i : i + 2] == ["cost", "of"]:
                k = i + 2
                if k < len(low) and low[k] in ("a", "an", "the"):
                    k += 1
                if k < len(low):
                    raw_q = low[k].rstrip("s")
                    if raw_q.startswith(item2[:3]):
                        q_item = item2
                break

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
        toks = [t.strip(".,?!:;()") for t in text.split() if t.strip(".,?!:;()")]
        low = [t.lower() for t in toks]

        e1: str | None = None
        mult_val: float | None = None
        noun: str = ""
        e2: str | None = None

        for i in range(1, len(toks)):
            if low[i] == "has":
                for j in range(i + 2, min(len(toks) - 4, i + 6)):
                    if low[j : j + 2] == ["as", "many"] and low[j + 3] == "as" and j + 4 < len(toks):
                        cand_e1 = toks[i - 1]
                        cand_noun = toks[j + 2]
                        cand_e2 = toks[j + 4]
                        mult_words = low[i + 1 : j]
                        mult_full = " ".join(mult_words)
                        mult_clean = (
                            " ".join(mult_words[:-1])
                            if mult_words and mult_words[-1] == "times" and len(mult_words) > 1
                            else mult_full
                        )
                        mv = self.policy.multipliers.get(mult_full)
                        if mv is None:
                            mv = self.policy.multipliers.get(mult_clean)
                        if mv is None:
                            mv = self.policy.multipliers.get(f"{mult_clean} times")
                        if mv is None and _is_num(mult_clean):
                            mv = float(mult_clean)
                        if mv is not None:
                            e1, mult_val, noun, e2 = cand_e1, mv, cand_noun, cand_e2
                            break
                if e1 is not None:
                    break

        if e1 is None or mult_val is None or e2 is None:
            return None

        a1 = Fraction(1)
        b1 = -Fraction(mult_val).limit_denominator()
        c1 = Fraction(0)

        # Equation 2: E1 + E2 = total
        total_str: str | None = None
        for i in range(len(toks) - 1):
            if low[i] in ("have", "is") and _is_num(toks[i + 1]):
                if any(k in low[max(0, i - 4) : i + 1] for k in ("together", "sum", "total", "combined", "they")):
                    total_str = toks[i + 1]
                    break

        if total_str is None:
            return None

        total_val = Fraction(total_str).limit_denominator()
        a2 = Fraction(1)
        b2 = Fraction(1)
        c2 = total_val

        sol = self.cas.solve_linear_system_2x2(a1, b1, c1, a2, b2, c2)
        if sol.status != "SOLVED":
            return sol

        vals = sol.result
        x_val = vals["x"]
        y_val = vals["y"]

        e1_cap = e1.capitalize()
        e2_cap = e2.capitalize()
        target_name = e1_cap

        for i in range(len(low) - 2):
            if low[i] == "does" and low[i + 2] == "have":
                cand = toks[i + 1].capitalize()
                if cand.lower() == e2_cap.lower():
                    target_name = e2_cap
                elif cand.lower() == e1_cap.lower():
                    target_name = e1_cap
                break

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

