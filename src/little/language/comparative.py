"""Comparative order and strict partial order (poset) reasoning engine.

Zero-hallucination transitive chain proofs, cycle/contradiction detection,
extremum (superlative) queries, and dual-polarity refutations.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from little.language.comparative_policy import (
    ComparativeOrderPolicy,
)

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


class ComparativeOrderReasoner:
    """Deterministic reasoner over strict partial orders (A > B)."""

    def __init__(
        self,
        memory: MemoryStore | None = None,
        policy: ComparativeOrderPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or ComparativeOrderPolicy.default()
        # In-memory graph fallback: dimension -> {u: {v1, v2}} where u > v
        self._local_graphs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def _get_adj(self, dim: str) -> dict[str, set[str]]:
        """Retrieve adjacency list (u -> {v} where u > v) for dimension."""
        adj: dict[str, set[str]] = defaultdict(set)
        # Pull from MemoryStore if available
        if self.memory is not None:
            pred = f"{dim}_greater"
            relations = self.memory.get_relations(predicate=pred)
            for r in relations:
                if r.weight_positive > r.weight_negative:
                    adj[r.subject_id.lower()].add(r.object_id.lower())
        # Union with local graph
        for u, neighbors in self._local_graphs[dim].items():
            adj[u].update(neighbors)
        return adj

    def _find_path(self, adj: dict[str, set[str]], start: str, goal: str) -> list[str] | None:
        """BFS path-finding to return shortest directed path from start to goal."""
        if start == goal:
            return [start]
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
        visited: set[str] = {start}
        while queue:
            curr, path = queue.popleft()
            for neighbor in adj.get(curr, set()):
                if neighbor == goal:
                    return path + [goal]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def assert_comparison(
        self,
        entity_a: str,
        entity_b: str,
        comp_word: str,
    ) -> tuple[bool, str]:
        """Assert a comparative statement: 'entity_a is <comp_word> than entity_b'."""
        lookup = self.policy.find_dimension_by_comparative(comp_word)
        if not lookup:
            return False, f"Unknown comparative relation: '{comp_word}'"

        dim_spec, polarity = lookup
        dim = dim_spec.dimension

        a = entity_a.strip().lower()
        b = entity_b.strip().lower()

        # Polarity: +1 means a > b; -1 means a < b (so b > a)
        u, v = (a, b) if polarity > 0 else (b, a)
        pos_comp = dim_spec.positive_comparatives[0]
        neg_comp = dim_spec.negative_comparatives[0]

        adj = self._get_adj(dim)

        # Check if v can reach u (which would mean v > u, creating a cycle/contradiction)
        cycle_path = self._find_path(adj, v, u)
        if cycle_path:
            template = self.policy.templates.get(
                "cycle_contradiction",
                "Contradiction detected: {a} cannot be simultaneously {comp_pos} and {comp_neg} than {b}."
            )
            msg = template.format(a=u, b=v, comp_pos=pos_comp, comp_neg=neg_comp)
            return False, msg

        # Add edge u > v
        self._local_graphs[dim][u].add(v)
        if self.memory is not None:
            pred = f"{dim}_greater"
            # Avoid duplicate inserts
            existing = self.memory.get_relations(subject_id=u, predicate=pred, object_id=v)
            if not existing:
                self.memory.add_relation(subject_id=u, predicate=pred, object_id=v)

        return True, f"Learned that {u} is {pos_comp} than {v}."

    def query_comparison(
        self,
        entity_a: str,
        entity_b: str,
        comp_word: str,
    ) -> str:
        """Evaluate 'Is entity_a <comp_word> than entity_b?'."""
        lookup = self.policy.find_dimension_by_comparative(comp_word)
        if not lookup:
            return f"I do not recognize the comparative term '{comp_word}'."

        dim_spec, polarity = lookup
        dim = dim_spec.dimension
        a = entity_a.strip().lower()
        b = entity_b.strip().lower()

        target_u, target_v = (a, b) if polarity > 0 else (b, a)
        pos_comp = dim_spec.positive_comparatives[0]
        neg_comp = dim_spec.negative_comparatives[0]
        used_comp = pos_comp if polarity > 0 else neg_comp

        if a == b:
            return f"No. {a.capitalize()} cannot be {used_comp} than itself."

        adj = self._get_adj(dim)

        # 1. Check if target_u -> target_v exists (Supports query)
        fwd_path = self._find_path(adj, target_u, target_v)
        if fwd_path:
            if len(fwd_path) == 2:
                trace = f"{a.capitalize()} is {used_comp} than {b.capitalize()}."
            else:
                steps = []
                for i in range(len(fwd_path) - 1):
                    n1 = fwd_path[i]
                    n2 = fwd_path[i + 1]
                    if polarity > 0:
                        steps.append(f"{n1.capitalize()} is {pos_comp} than {n2.capitalize()}")
                    else:
                        steps.append(f"{n2.capitalize()} is {neg_comp} than {n1.capitalize()}")
                trace = ", and ".join(steps) + f". Therefore, {a.capitalize()} is {used_comp} than {b.capitalize()}."

            template = self.policy.templates.get("transitive_affirmation", "Yes. {trace}")
            return template.format(trace=trace)

        # 2. Check if target_v -> target_u exists (Refutes query)
        rev_path = self._find_path(adj, target_v, target_u)
        if rev_path:
            opposite_comp = neg_comp if polarity > 0 else pos_comp
            template = self.policy.templates.get("refutation", "No. In fact, {b} is {comp} than {a}.")
            return template.format(a=a.capitalize(), b=b.capitalize(), comp=opposite_comp)

        # 3. Unknown
        template = self.policy.templates.get(
            "unknown",
            "It is unknown whether {a} is {comp} than {b} from the given premises."
        )
        return template.format(a=a.capitalize(), b=b.capitalize(), comp=used_comp)

    def query_extremum(self, superlative_word: str) -> str:
        """Evaluate 'Who/Which is the <superlative>?'."""
        lookup = self.policy.find_dimension_by_superlative(superlative_word)
        if not lookup:
            return f"I do not recognize the superlative term '{superlative_word}'."

        dim_spec, polarity = lookup
        dim = dim_spec.dimension
        adj = self._get_adj(dim)

        all_nodes: set[str] = set(adj.keys())
        for targets in adj.values():
            all_nodes.update(targets)

        if not all_nodes:
            return f"No comparison data available for dimension '{dim_spec.noun}'."

        # If polarity > 0 (e.g. tallest, fastest): looking for maximal element (no incoming edges)
        # If polarity < 0 (e.g. shortest, slowest): looking for minimal element (no outgoing edges)
        in_degrees: dict[str, int] = defaultdict(int)
        for u in all_nodes:
            for v in adj.get(u, set()):
                in_degrees[v] += 1

        if polarity > 0:
            candidates = [n for n in all_nodes if in_degrees[n] == 0]
        else:
            candidates = [n for n in all_nodes if len(adj.get(n, set())) == 0]

        if len(candidates) == 1:
            best = candidates[0]
            template = self.policy.templates.get(
                "extremum_result",
                "{entity} is the {superlative}."
            )
            return template.format(
                entity=best.capitalize(),
                superlative=superlative_word,
                chain=f"{best} > others"
            )
        elif len(candidates) > 1:
            names = ", ".join(c.capitalize() for c in candidates)
            return f"There is a tie or incomplete information among: {names}."
        else:
            return f"Unable to determine the {superlative_word} due to incomplete relations."

    def evaluate_text(self, text: str) -> str | None:
        """Detect comparative assertions or questions in text and evaluate."""
        clean = text.strip()

        # Check for compound premises + query:
        # e.g. "John is taller than Mary, and Mary is taller than Peter. Who is the tallest?"
        parts = [p.strip() for p in re.split(r"[.;\n]|,\s*(?:and\s+)?", clean) if p.strip()]

        # Learn all premise parts
        for part in parts:
            for pat in self.policy.statement_patterns:
                m = re.search(pat, part, re.IGNORECASE)
                if m:
                    entity_a, comp, entity_b = m.group(1), m.group(2), m.group(3)
                    if self.policy.find_dimension_by_comparative(comp):
                        success, msg = self.assert_comparison(entity_a, entity_b, comp)
                        if not success:
                            return msg

        # Evaluate question in parts or clean
        ext_pat = self.policy.question_patterns.get("extremum", "")
        if ext_pat:
            clean_ext_pat = ext_pat.lstrip("^").rstrip("$")
            for segment in parts + [clean]:
                m_ext = re.search(clean_ext_pat, segment, re.IGNORECASE)
                if m_ext:
                    sup = m_ext.group(1).strip()
                    if self.policy.find_dimension_by_superlative(sup):
                        return self.query_extremum(sup)

        trans_pat = self.policy.question_patterns.get("transitive", "")
        if trans_pat:
            clean_trans_pat = trans_pat.lstrip("^").rstrip("$")
            for segment in parts + [clean]:
                m_trans = re.search(clean_trans_pat, segment, re.IGNORECASE)
                if m_trans:
                    e_a, comp, e_b = m_trans.group(1).strip(), m_trans.group(2).strip(), m_trans.group(3).strip()
                    if self.policy.find_dimension_by_comparative(comp):
                        res = self.query_comparison(e_a, e_b, comp)
                        if not res.startswith("I do not recognize"):
                            return res

        return None
