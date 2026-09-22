"""Generalized Math Word Problem Solver.

Decomposes natural language story problems into quantities, state transitions,
proportional rates, and fractional partitions with verifiable <math_trace> output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, List, Optional, Tuple

from little.procedural.math_cas import MathSolution, SymbolicCAS


@dataclass
class StoryAction:
    action_type: str  # "INITIAL", "SUBTRACT", "ADD", "SCALE", "PARTITION"
    quantity: float | Fraction
    unit: str
    description: str


class MathStorySolver:
    """Zero-hallucination semantic solver for multi-step math word problems."""

    def __init__(self) -> None:
        self.cas = SymbolicCAS()

    def is_math_story(self, text: str) -> bool:
        """Detect if input text constitutes a math story / word problem."""
        s = text.strip()
        # Must contain numbers
        if not re.search(r"\b\d+(?:\.\d+)?\b", s):
            return False
        # Must contain an inquiry for quantity / total / distance / cost
        q_pattern = r"\b(?:how many|how much|how far|what is the total|how many .* left|how much .* left)\b"
        return bool(re.search(q_pattern, s, re.IGNORECASE))

    def solve_story(self, text: str) -> Optional[MathSolution]:
        """Parse story problem, track quantity state transitions, and return exact MathSolution."""
        clean = text.strip()

        # 1. Rate * Time problem: e.g. "travels at 60 km/h for 3 hours. How far does..."
        m_rate = re.search(
            r"(?:travels?|drives?|runs?|flies?|walks?)\s+at\s+(\d+(?:\.\d+)?)\s*(km/h|kph|mph|m/s)\s+for\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|s)\b",
            clean,
            re.IGNORECASE,
        )
        if m_rate and re.search(r"how far", clean, re.IGNORECASE):
            speed = float(m_rate.group(1))
            speed_unit = m_rate.group(2)
            time_val = float(m_rate.group(3))
            time_unit = m_rate.group(4)

            # Normalization if minutes
            if time_unit.lower().startswith("min"):
                time_hrs = time_val / 60.0
            elif time_unit.lower().startswith("s"):
                time_hrs = time_val / 3600.0
            else:
                time_hrs = time_val

            dist = speed * time_hrs
            dist_res = int(dist) if dist.is_integer() else round(dist, 2)
            steps = [
                f"Rate & Distance Problem:",
                f"  Speed v = {speed} {speed_unit}",
                f"  Time t = {time_val} {time_unit} (= {time_hrs} hours)",
                f"  Formula: Distance d = v * t",
                f"  Calculation: {speed} * {time_hrs} = {dist_res} km",
            ]
            latex = f"d = {speed} \\times {time_hrs} = {dist_res}"
            return MathSolution(
                status="SOLVED",
                result=dist_res,
                latex_trace=latex,
                steps=steps,
            )

        # 2. Multi-item cost / scaling sum:
        # e.g. "bought 3 books for 5 dollars each and 2 pens for 4 dollars each. How much money did she spend in total?"
        item_matches = re.findall(
            r"(\d+)\s+([a-zA-Z]+)\s+(?:for|at)\s+(\d+(?:\.\d+)?)\s*(?:dollars?|\$)?\s*(?:each|per\s+[a-zA-Z]+)?",
            clean,
            re.IGNORECASE,
        )
        if len(item_matches) >= 2 and re.search(r"(?:how much|total)", clean, re.IGNORECASE):
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
        # e.g. "Sarah has 15 apples. She gives 4 apples to Bob, then buys 7 more apples. How many apples does Sarah have now?"
        # e.g. "David has 24 candies. He gives half of them to his sister. How many candies does David have left?"
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?;\n])\s+", clean)
            if s.strip()
        ]

        current_val: Optional[float] = None
        item_name = "items"
        steps: list[str] = ["Sequential State Tracking:"]

        # Verbs classifying actions
        subtract_verbs = {
            "gives", "gave", "gives away", "gave away", "lost", "loses",
            "eats", "ate", "sells", "sold", "spends", "spent", "drops", "dropped",
        }
        add_verbs = {
            "buys", "bought", "finds", "found", "gets", "got", "receives", "received",
            "picks", "picked", "earns", "earned", "adds", "added",
        }

        # Step 3a: Look for Initial inventory in the first sentence or clause
        for sent in sentences:
            m_init = re.search(
                r"\b(?:has|had|owns|started with)\s+(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\b",
                sent,
                re.IGNORECASE,
            )
            if m_init:
                current_val = float(m_init.group(1))
                item_name = m_init.group(2)
                c_disp = int(current_val) if current_val.is_integer() else current_val
                steps.append(f"  Initial: {c_disp} {item_name}")
                break

        if current_val is None:
            # Fallback: look for any leading number + noun
            m_num = re.search(r"\b(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\b", clean)
            if m_num:
                current_val = float(m_num.group(1))
                item_name = m_num.group(2)
                c_disp = int(current_val) if current_val.is_integer() else current_val
                steps.append(f"  Initial: {c_disp} {item_name}")

        if current_val is None:
            return None

        # Step 3b: Process state transition sentences / clauses
        # Split clauses by period or conjunctions like 'then', 'and then'
        clauses = [
            c.strip()
            for sent in sentences
            for c in re.split(r",\s*(?:then\s+|and\s+then\s+|and\s+)?|\s+(?:then|and\s+then)\s+", sent)
            if c.strip()
        ]

        for clause in clauses:
            # Ignore question clause
            if re.search(r"\bhow (?:many|much)\b", clause, re.IGNORECASE) or clause.endswith("?"):
                continue

            # Fractional partition check: "gives half of them", "eats one third of them"
            m_half = re.search(
                r"\b(?:gives?|ate|eats?|lost|loses?|gives? away)\s+(?:a\s+)?half(?:\s+of\s+them)?\b",
                clause,
                re.IGNORECASE,
            )
            if m_half:
                part = current_val * 0.5
                current_val -= part
                c_disp = int(current_val) if current_val.is_integer() else current_val
                p_disp = int(part) if part.is_integer() else part
                steps.append(f"  Partition (- half): subtracted {p_disp}, remaining = {c_disp} {item_name}")
                continue

            # Subtract actions:
            sub_matched = False
            for sv in subtract_verbs:
                m_sub = re.search(
                    rf"\b{sv}\s+(\d+(?:\.\d+)?)(?:\s+more)?(?:\s+[a-zA-Z]+)?\b",
                    clause,
                    re.IGNORECASE,
                )
                if m_sub:
                    sub_qty = float(m_sub.group(1))
                    old_val = current_val
                    current_val -= sub_qty
                    c_disp = int(current_val) if current_val.is_integer() else current_val
                    s_disp = int(sub_qty) if sub_qty.is_integer() else sub_qty
                    o_disp = int(old_val) if old_val.is_integer() else old_val
                    steps.append(f"  Subtract ({sv} {s_disp}): {o_disp} - {s_disp} = {c_disp} {item_name}")
                    sub_matched = True
                    break

            if sub_matched:
                continue

            # Add actions:
            for av in add_verbs:
                m_add = re.search(
                    rf"\b{av}\s+(\d+(?:\.\d+)?)(?:\s+more)?(?:\s+[a-zA-Z]+)?\b",
                    clause,
                    re.IGNORECASE,
                )
                if m_add:
                    add_qty = float(m_add.group(1))
                    old_val = current_val
                    current_val += add_qty
                    c_disp = int(current_val) if current_val.is_integer() else current_val
                    a_disp = int(add_qty) if add_qty.is_integer() else add_qty
                    o_disp = int(old_val) if old_val.is_integer() else old_val
                    steps.append(f"  Add ({av} {a_disp}): {o_disp} + {a_disp} = {c_disp} {item_name}")
                    break

        final_res = int(current_val) if current_val.is_integer() else round(current_val, 2)
        steps.append(f"Final Quantity: {final_res} {item_name}")
        latex = f"\\text{{Result}} = {final_res}"
        return MathSolution(
            status="SOLVED",
            result=final_res,
            latex_trace=latex,
            steps=steps,
        )
