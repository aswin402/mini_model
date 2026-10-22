"""Comparative order and strict partial order (poset) reasoning engine.

Zero-hallucination transitive chain proofs, cycle/contradiction detection,
extremum (superlative) queries, and dual-polarity refutations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from little.language.comparative_policy import (
    ComparativeOrderPolicy,
)

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


def _strip_leading_article(text: str) -> str:
    s = text.strip()
    lower = s.lower()
    for art in ("a ", "an ", "the "):
        if lower.startswith(art):
            return s[len(art) :].strip()
    return s


def _split_comparative_clauses(text: str) -> list[str]:
    """Split comparative text on [.;\\n] or ', (and )?' without regex."""
    out: list[str] = []
    n = len(text)
    last = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".;\n":
            seg = text[last:i].strip()
            if seg:
                out.append(seg)
            last = i + 1
            i += 1
            continue
        if ch == ",":
            seg = text[last:i].strip()
            if seg:
                out.append(seg)
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j + 3 <= n and text[j : j + 3].lower() == "and" and (
                j + 3 == n or text[j + 3].isspace()
            ):
                j += 3
                while j < n and text[j].isspace():
                    j += 1
            last = j
            i = j
            continue
        i += 1
    tail = text[last:].strip()
    if tail:
        out.append(tail)
    return out


def _parse_comparative_statement(part: str) -> tuple[str, str, str] | None:
    """Parse '<A> is <comp> than <B>' without regex."""
    s = part.strip()
    if not s or "?" in s:
        return None
    lower = s.lower()
    for prefix in ("consider that ", "given that ", "suppose that "):
        if lower.startswith(prefix):
            s = s[len(prefix) :].strip()
            lower = s.lower()
            break
    s = _strip_leading_article(s)
    lower = s.lower()
    than_idx = lower.find(" than ")
    if than_idx == -1:
        return None
    lhs = s[:than_idx].strip()
    rhs = _strip_leading_article(s[than_idx + 6 :].strip())
    lhs_lower = lhs.lower()
    is_idx = lhs_lower.rfind(" is ")
    if is_idx == -1:
        return None
    entity_a = _strip_leading_article(lhs[:is_idx].strip())
    comp = lhs[is_idx + 4 :].strip()
    comp_words = comp.split()
    if not (1 <= len(comp_words) <= 2 and all(w.isalpha() for w in comp_words)):
        return None
    if not entity_a or not rhs:
        return None
    return entity_a, comp, rhs


def _parse_extremum_query(segment: str) -> str | None:
    """Parse 'Who/Which is (the) <superlative> ...?' without regex."""
    s = segment.strip().rstrip("?").strip()
    words = s.split()
    if len(words) < 3:
        return None
    if words[0].lower() not in ("who", "which") or words[1].lower() != "is":
        return None
    idx = 2
    if words[idx].lower() == "the":
        idx += 1
    if idx >= len(words):
        return None
    # Check for 2-word superlative before 'between/among/of' or end
    rem = words[idx:]
    stop_idx = len(rem)
    for k, w in enumerate(rem):
        if w.lower() in ("between", "among", "of"):
            stop_idx = k
            break
    sup_tokens = rem[:stop_idx]
    if 1 <= len(sup_tokens) <= 2 and all(w.isalpha() for w in sup_tokens):
        return " ".join(sup_tokens)
    return None


def _parse_transitive_query(segment: str) -> tuple[str, str, str] | None:
    """Parse 'Is <A> <comp> than <B>?' without regex."""
    s = segment.strip().rstrip("?").strip()
    lower = s.lower()
    if not lower.startswith("is "):
        return None
    body = _strip_leading_article(s[3:].strip())
    body_lower = body.lower()
    than_idx = body_lower.find(" than ")
    if than_idx == -1:
        return None
    lhs = body[:than_idx].strip()
    entity_b = _strip_leading_article(body[than_idx + 6 :].strip())
    lhs_words = lhs.split()
    if len(lhs_words) < 2 or not entity_b:
        return None
    # Check 2-word comparative (e.g. "more expensive", "less expensive") first
    if len(lhs_words) >= 3 and lhs_words[-2].lower() in ("more", "less") and lhs_words[-1].isalpha():
        entity_a = " ".join(lhs_words[:-2]).strip()
        comp = " ".join(lhs_words[-2:]).strip()
        return entity_a, comp, entity_b
    if lhs_words[-1].isalpha():
        entity_a = " ".join(lhs_words[:-1]).strip()
        comp = lhs_words[-1].strip()
        return entity_a, comp, entity_b
    return None


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
        parts = _split_comparative_clauses(clean)

        # Learn all premise parts
        if self.policy.statement_patterns:
            for part in parts:
                parsed_stmt = _parse_comparative_statement(part)
                if parsed_stmt is not None:
                    entity_a, comp, entity_b = parsed_stmt
                    if self.policy.find_dimension_by_comparative(comp):
                        success, msg = self.assert_comparison(entity_a, entity_b, comp)
                        if not success:
                            return msg

        # Evaluate question in parts or clean
        ext_pat = self.policy.question_patterns.get("extremum", "")
        if ext_pat:
            for segment in parts + [clean]:
                sup = _parse_extremum_query(segment)
                if sup and self.policy.find_dimension_by_superlative(sup):
                    return self.query_extremum(sup)

        trans_pat = self.policy.question_patterns.get("transitive", "")
        if trans_pat:
            for segment in parts + [clean]:
                parsed_trans = _parse_transitive_query(segment)
                if parsed_trans is not None:
                    e_a, comp, e_b = parsed_trans
                    if self.policy.find_dimension_by_comparative(comp):
                        res = self.query_comparison(e_a, e_b, comp)
                        if not res.startswith("I do not recognize"):
                            return res

        return None
