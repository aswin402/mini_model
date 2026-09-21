"""Deterministic Knowledge Graph Inference Engine for LITTLE.

Implements multi-hop transitivity (e.g., is_a, part_of chains),
property inheritance, contradiction checking, and explicit UNKNOWN
detection under the Open-World Assumption.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

from little.core.models import BeliefStatus, InferenceResult
from little.memory.store import MemoryStore


class InferenceEngine:
    """Performs graph-based multi-hop reasoning with evidence tracking and uncertainty."""

    TRANSITIVE_PREDICATES: ClassVar[set[str]] = {
        "is_a",
        "subclass_of",
        "part_of",
        "instance_of",
    }
    DISJOINT_PREDICATES: ClassVar[set[str]] = {
        "disjoint_with",
        "cannot_be",
        "different_from",
    }

    def __init__(self, memory: MemoryStore, max_depth: int = 8) -> None:
        self.memory = memory
        self.max_depth = max_depth

    def infer(
        self,
        subject: str,
        predicate: str,
        target: str,
    ) -> InferenceResult:
        """Evaluate whether (subject, predicate, target) holds based on stored knowledge."""
        clean_subj = subject.strip().lower()
        clean_pred = predicate.strip().lower()
        clean_target = target.strip().lower()

        query_str = f"({clean_subj}, {clean_pred}, {clean_target})"
        trace: list[str] = [f"Inference initiated for: {query_str}"]

        # 1. Resolve concepts
        subj_concept = self.memory.get_concept(clean_subj)
        target_concept = self.memory.get_concept(clean_target)

        if not subj_concept:
            trace.append(
                f"Subject concept '{clean_subj}' is completely unknown to the system."
            )
            return InferenceResult(
                query=query_str,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=trace,
            )

        if not target_concept:
            trace.append(
                f"Target concept '{clean_target}' is completely unknown to the system."
            )
            return InferenceResult(
                query=query_str,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=trace,
            )

        # 2. Check Direct Relationship
        direct_relations = self.memory.get_relations(
            subject_id=subj_concept.id,
            predicate=clean_pred,
            object_id=target_concept.id,
        )
        if direct_relations:
            rel = direct_relations[0]
            trace.append(
                f"Direct relation found: {rel.id} with weight_pos={rel.weight_positive}, weight_neg={rel.weight_negative}"
            )
            if rel.weight_positive > rel.weight_negative:
                return InferenceResult(
                    query=query_str,
                    status=BeliefStatus.SUPPORTED,
                    answer=True,
                    confidence=rel.confidence,
                    evidence=[
                        f"{subj_concept.name} {clean_pred} {target_concept.name}"
                    ],
                    trace=trace,
                )
            elif rel.weight_negative > rel.weight_positive:
                return InferenceResult(
                    query=query_str,
                    status=BeliefStatus.REFUTED,
                    answer=False,
                    confidence=rel.confidence,
                    evidence=[
                        f"NOT ({subj_concept.name} {clean_pred} {target_concept.name})"
                    ],
                    trace=trace,
                )

        # 3. Transitive Graph Reasoning (if predicate supports transitivity)
        if clean_pred in self.TRANSITIVE_PREDICATES:
            trace.append(f"Evaluating transitive path via '{clean_pred}'...")
            path_result = self._find_transitive_path(
                start_id=subj_concept.id,
                target_id=target_concept.id,
                predicate=clean_pred,
            )
            if path_result:
                path_names, path_conf = path_result
                chain_str = " -> ".join(path_names)
                trace.append(
                    f"Transitive path discovered: {chain_str} (Confidence: {path_conf})"
                )
                return InferenceResult(
                    query=query_str,
                    status=BeliefStatus.SUPPORTED,
                    answer=True,
                    confidence=path_conf,
                    evidence=[f"Chain: {chain_str}"],
                    trace=trace,
                )

        # 4. Check Disjoint / Mutual Exclusivity Constraints
        trace.append("Checking for mutual exclusivity / disjoint relations...")
        disjoint_conflict = self._check_disjoint_conflict(
            subj_concept.id, target_concept.id
        )
        if disjoint_conflict:
            conflict_msg, conf = disjoint_conflict
            trace.append(f"Disjoint constraint triggered: {conflict_msg}")
            return InferenceResult(
                query=query_str,
                status=BeliefStatus.REFUTED,
                answer=False,
                confidence=conf,
                evidence=[conflict_msg],
                trace=trace,
            )

        # 5. Open-World Assumption: Insufficient Evidence -> UNKNOWN
        trace.append(
            "No supporting path or refuting constraints found. Epistemic state is UNKNOWN."
        )
        return InferenceResult(
            query=query_str,
            status=BeliefStatus.UNKNOWN,
            answer=None,
            confidence=0.1,  # baseline low epistemic confidence
            evidence=[],
            trace=trace,
        )

    def _find_transitive_path(
        self,
        start_id: str,
        target_id: str,
        predicate: str,
    ) -> tuple[list[str], float] | None:
        """Breadth-first search for transitive chains (e.g. dog -> animal -> living_thing)."""
        queue: deque[tuple[str, list[str], float]] = deque()
        queue.append((start_id, [start_id], 1.0))
        visited: set[str] = {start_id}

        while queue:
            current_id, path_ids, current_conf = queue.popleft()

            if len(path_ids) > self.max_depth:
                continue

            relations = self.memory.get_relations(
                subject_id=current_id, predicate=predicate
            )
            for rel in relations:
                if rel.weight_positive <= rel.weight_negative:
                    continue

                next_id = rel.object_id
                next_conf = round(current_conf * rel.confidence, 4)

                if next_id == target_id:
                    # Target reached! Map IDs to canonical names
                    full_id_path = path_ids + [next_id]
                    name_path: list[str] = []
                    for node_id in full_id_path:
                        c = self.memory.get_concept(node_id)
                        name_path.append(c.name if c else node_id)
                    return name_path, next_conf

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path_ids + [next_id], next_conf))

        return None

    def _check_disjoint_conflict(
        self,
        subj_id: str,
        target_id: str,
    ) -> tuple[str, float] | None:
        """Check if subject and target possess conflicting categories or disjoint ancestors."""
        # Get ancestors of subject and target
        subj_ancestors = set(self.get_ancestor_ids(subj_id))
        subj_ancestors.add(subj_id)

        target_ancestors = set(self.get_ancestor_ids(target_id))
        target_ancestors.add(target_id)

        # Check disjoint relations between any pair of ancestors (symmetric check)
        for disj_pred in self.DISJOINT_PREDICATES:
            # 1. Outgoing from subj_ancestors
            for s_node in subj_ancestors:
                for rel in self.memory.get_relations(
                    subject_id=s_node, predicate=disj_pred
                ):
                    if (
                        rel.object_id in target_ancestors
                        and rel.weight_positive > rel.weight_negative
                    ):
                        c1 = self.memory.get_concept(s_node)
                        c2 = self.memory.get_concept(rel.object_id)
                        name1 = c1.name if c1 else s_node
                        name2 = c2.name if c2 else rel.object_id
                        return (f"{name1} is disjoint with {name2}", rel.confidence)

            # 2. Outgoing from target_ancestors (symmetric)
            for t_node in target_ancestors:
                for rel in self.memory.get_relations(
                    subject_id=t_node, predicate=disj_pred
                ):
                    if (
                        rel.object_id in subj_ancestors
                        and rel.weight_positive > rel.weight_negative
                    ):
                        c1 = self.memory.get_concept(t_node)
                        c2 = self.memory.get_concept(rel.object_id)
                        name1 = c1.name if c1 else t_node
                        name2 = c2.name if c2 else rel.object_id
                        return (f"{name1} is disjoint with {name2}", rel.confidence)

        return None

    def get_ancestor_ids(self, concept_id: str, predicate: str = "is_a") -> list[str]:
        """Collect all ancestor IDs up the transitive chain."""
        ancestors: list[str] = []
        visited: set[str] = {concept_id}
        queue: deque[str] = deque([concept_id])

        while queue:
            curr = queue.popleft()
            rels = self.memory.get_relations(subject_id=curr, predicate=predicate)
            for r in rels:
                if r.weight_positive > r.weight_negative and r.object_id not in visited:
                    visited.add(r.object_id)
                    ancestors.append(r.object_id)
                    queue.append(r.object_id)

        return ancestors

    def query_ancestors(self, concept_name: str, predicate: str = "is_a") -> list[str]:
        """Return human-readable names of all ancestors of a concept."""
        concept = self.memory.get_concept(concept_name)
        if not concept:
            return []
        ancestor_ids = self.get_ancestor_ids(concept.id, predicate=predicate)
        names = []
        for aid in ancestor_ids:
            ac = self.memory.get_concept(aid)
            if ac:
                names.append(ac.name)
        return names

    def get_inherited_attributes(self, concept_name: str) -> dict[str, Any]:
        """Compute all inherited attributes from parent concepts."""
        concept = self.memory.get_concept(concept_name)
        if not concept:
            return {}

        attributes: dict[str, Any] = {}
        # Start from top ancestors down to specific concept
        ancestor_ids = list(reversed(self.get_ancestor_ids(concept.id, "is_a")))
        for aid in ancestor_ids:
            ac = self.memory.get_concept(aid)
            if ac:
                attributes.update(ac.attributes)

        # Child attributes override or specialize ancestor attributes
        attributes.update(concept.attributes)
        return attributes
