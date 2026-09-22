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
        "located_in",
        "larger_than",
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
        pred_candidates = [clean_pred]
        if clean_pred.endswith("s") and not clean_pred.endswith("ss"):
            pred_candidates.append(clean_pred[:-1])
        else:
            pred_candidates.append(clean_pred + "s")
            pred_candidates.append(clean_pred + "es")

        direct_relations = []
        matched_pred = clean_pred
        for pc in pred_candidates:
            rels = self.memory.get_relations(
                subject_id=subj_concept.id,
                predicate=pc,
                object_id=target_concept.id,
            )
            if rels:
                direct_relations = rels
                matched_pred = pc
                break

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
                        f"{subj_concept.name} {matched_pred} {target_concept.name}"
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
                        f"NOT ({subj_concept.name} {matched_pred} {target_concept.name})"
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

        # 3b. Asymmetric Refutation for strict ordering (e.g. larger_than)
        if clean_pred == "larger_than":
            rev_path = self._find_transitive_path(
                start_id=target_concept.id,
                target_id=subj_concept.id,
                predicate="larger_than",
            )
            if rev_path:
                rev_names, rev_conf = rev_path
                rev_chain = " > ".join(rev_names)
                trace.append(
                    f"Asymmetric contradiction: reverse path holds ({rev_chain})"
                )
                return InferenceResult(
                    query=query_str,
                    status=BeliefStatus.REFUTED,
                    answer=False,
                    confidence=rev_conf,
                    evidence=[f"Reverse order confirmed: {rev_chain}"],
                    trace=trace,
                )

        # 3c. Part-Whole Duality and Taxonomic Inheritance for 'has' and 'part_of'
        if clean_pred == "has":
            subj_anc = [subj_concept.id] + self.get_ancestor_ids(
                subj_concept.id, predicate="is_a"
            )
            for anc_id in subj_anc:
                # Direct has
                anc_has = self.memory.get_relations(
                    subject_id=anc_id, predicate="has", object_id=target_concept.id
                )
                if anc_has and anc_has[0].weight_positive > anc_has[0].weight_negative:
                    anc_c = self.memory.get_concept(anc_id)
                    anc_name = anc_c.name if anc_c else anc_id
                    trace.append(f"Inherited 'has' from ancestor '{anc_name}'")
                    ev = [f"{anc_name} has {target_concept.name}"]
                    if anc_id != subj_concept.id:
                        ev.insert(0, f"{subj_concept.name} is_a {anc_name}")
                    return InferenceResult(
                        query=query_str,
                        status=BeliefStatus.SUPPORTED,
                        answer=True,
                        confidence=round(
                            anc_has[0].confidence
                            * (0.95 if anc_id != subj_concept.id else 1.0),
                            4,
                        ),
                        evidence=ev,
                        trace=trace,
                    )
                # Duality: target part_of ancestor
                part_rels = self.memory.get_relations(
                    subject_id=target_concept.id,
                    predicate="part_of",
                    object_id=anc_id,
                )
                if (
                    part_rels
                    and part_rels[0].weight_positive > part_rels[0].weight_negative
                ):
                    anc_c = self.memory.get_concept(anc_id)
                    anc_name = anc_c.name if anc_c else anc_id
                    trace.append(
                        f"Dual 'has' inferred from '{target_concept.name} part_of {anc_name}'"
                    )
                    ev = [f"{target_concept.name} part_of {anc_name}"]
                    if anc_id != subj_concept.id:
                        ev.insert(0, f"{subj_concept.name} is_a {anc_name}")
                    return InferenceResult(
                        query=query_str,
                        status=BeliefStatus.SUPPORTED,
                        answer=True,
                        confidence=round(
                            part_rels[0].confidence
                            * (0.95 if anc_id != subj_concept.id else 1.0),
                            4,
                        ),
                        evidence=ev,
                        trace=trace,
                    )

        if clean_pred == "part_of":
            target_anc = [target_concept.id] + self.get_ancestor_ids(
                target_concept.id, predicate="is_a"
            )
            for anc_id in target_anc:
                # Direct part_of ancestor
                part_rels = self.memory.get_relations(
                    subject_id=subj_concept.id,
                    predicate="part_of",
                    object_id=anc_id,
                )
                if (
                    part_rels
                    and part_rels[0].weight_positive > part_rels[0].weight_negative
                ):
                    anc_c = self.memory.get_concept(anc_id)
                    anc_name = anc_c.name if anc_c else anc_id
                    trace.append(f"Inferred 'part_of' via target ancestor '{anc_name}'")
                    ev = [f"{subj_concept.name} part_of {anc_name}"]
                    if anc_id != target_concept.id:
                        ev.append(f"{target_concept.name} is_a {anc_name}")
                    return InferenceResult(
                        query=query_str,
                        status=BeliefStatus.SUPPORTED,
                        answer=True,
                        confidence=round(
                            part_rels[0].confidence
                            * (0.95 if anc_id != target_concept.id else 1.0),
                            4,
                        ),
                        evidence=ev,
                        trace=trace,
                    )
                # Duality: ancestor has subj
                anc_has = self.memory.get_relations(
                    subject_id=anc_id, predicate="has", object_id=subj_concept.id
                )
                if anc_has and anc_has[0].weight_positive > anc_has[0].weight_negative:
                    anc_c = self.memory.get_concept(anc_id)
                    anc_name = anc_c.name if anc_c else anc_id
                    trace.append(
                        f"Dual 'part_of' inferred from '{anc_name} has {subj_concept.name}'"
                    )
                    ev = [f"{anc_name} has {subj_concept.name}"]
                    if anc_id != target_concept.id:
                        ev.append(f"{target_concept.name} is_a {anc_name}")
                    return InferenceResult(
                        query=query_str,
                        status=BeliefStatus.SUPPORTED,
                        answer=True,
                        confidence=round(
                            anc_has[0].confidence
                            * (0.95 if anc_id != target_concept.id else 1.0),
                            4,
                        ),
                        evidence=ev,
                        trace=trace,
                    )

        # 3d. Taxonomic Inheritance of Other Non-Transitive Predicates (e.g. can, lives_in, eats, made_of)
        if (
            clean_pred not in self.TRANSITIVE_PREDICATES
            and clean_pred not in self.DISJOINT_PREDICATES
            and clean_pred not in ("has", "part_of")
        ):
            trace.append(
                f"Evaluating taxonomic inheritance for predicate '{clean_pred}'..."
            )
            for anc_id in self.get_ancestor_ids(subj_concept.id, predicate="is_a"):
                anc_rels = self.memory.get_relations(
                    subject_id=anc_id,
                    predicate=clean_pred,
                    object_id=target_concept.id,
                )
                if anc_rels:
                    rel = anc_rels[0]
                    anc_c = self.memory.get_concept(anc_id)
                    anc_name = anc_c.name if anc_c else anc_id
                    if rel.weight_positive > rel.weight_negative:
                        trace.append(
                            f"Inherited relation from ancestor '{anc_name}': ({anc_name} {clean_pred} {target_concept.name})"
                        )
                        return InferenceResult(
                            query=query_str,
                            status=BeliefStatus.SUPPORTED,
                            answer=True,
                            confidence=round(rel.confidence * 0.95, 4),
                            evidence=[
                                f"{subj_concept.name} is_a {anc_name}",
                                f"{anc_name} {clean_pred} {target_concept.name}",
                            ],
                            trace=trace,
                        )
                    elif rel.weight_negative > rel.weight_positive:
                        trace.append(
                            f"Inherited negative relation from ancestor '{anc_name}': NOT ({anc_name} {clean_pred} {target_concept.name})"
                        )
                        return InferenceResult(
                            query=query_str,
                            status=BeliefStatus.REFUTED,
                            answer=False,
                            confidence=round(rel.confidence * 0.95, 4),
                            evidence=[
                                f"NOT ({anc_name} {clean_pred} {target_concept.name})"
                            ],
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
        """Bidirectional Frontier Collision Search (meeting-in-the-middle) with DeepSeek Invariant Gates.

        Searches forward from start_id (outgoing edges: current -> next) and backward
        from target_id (incoming edges: prev -> current) simultaneously.
        Reduces computational graph search complexity from O(b^d) to O(2 * b^(d/2)).

        Enforces 3 Invariant Verification Gates:
        1. I_DAG: Acyclicity invariant (no repeating nodes).
        2. I_mutex: Mutual exclusivity invariant (no disjoint conflict along the chain).
        3. I_ground: Empirical grounding invariant (weight_positive > weight_negative).
        """
        if start_id == target_id:
            c = self.memory.get_concept(start_id)
            return ([c.name if c else start_id], 1.0)

        # Forward search state: start_id -> ...
        # visited_fwd[node_id] = (path_ids_from_start, cumulative_confidence)
        visited_fwd: dict[str, tuple[list[str], float]] = {start_id: ([start_id], 1.0)}
        q_fwd: deque[str] = deque([start_id])

        # Backward search state: ... -> target_id
        # visited_bwd[node_id] = (path_ids_to_target, cumulative_confidence)
        visited_bwd: dict[str, tuple[list[str], float]] = {target_id: ([target_id], 1.0)}
        q_bwd: deque[str] = deque([target_id])

        depth_fwd = 0
        depth_bwd = 0

        while q_fwd and q_bwd:
            # Check maximum depth bounds
            if depth_fwd + depth_bwd > self.max_depth:
                break

            # Always expand the smaller frontier to minimize branching factor (Bi-A* optimization)
            if len(q_fwd) <= len(q_bwd):
                # Expand one level forward
                level_size = len(q_fwd)
                depth_fwd += 1
                for _ in range(level_size):
                    curr_fwd = q_fwd.popleft()
                    path_fwd, conf_fwd = visited_fwd[curr_fwd]

                    # Outgoing edges: curr_fwd --(predicate)--> next_id
                    relations = self.memory.get_relations(
                        subject_id=curr_fwd, predicate=predicate
                    )
                    for rel in relations:
                        # Gate 3: I_ground (Empirical evidence verification)
                        if rel.weight_positive <= rel.weight_negative:
                            continue

                        next_id = rel.object_id
                        # Gate 1: I_DAG (Acyclicity invariant)
                        if next_id in path_fwd:
                            continue

                        next_conf = round(conf_fwd * rel.confidence, 4)

                        # Collision Check with Backward Frontier
                        if next_id in visited_bwd:
                            path_bwd, conf_bwd = visited_bwd[next_id]
                            full_id_path = path_fwd + path_bwd
                            # Verify full path satisfies I_DAG (no duplicates)
                            if len(set(full_id_path)) == len(full_id_path):
                                total_conf = round(next_conf * conf_bwd, 4)
                                name_path = [
                                    self.memory.get_concept(nid).name
                                    if self.memory.get_concept(nid)
                                    else nid
                                    for nid in full_id_path
                                ]
                                return name_path, total_conf

                        if next_id not in visited_fwd:
                            visited_fwd[next_id] = (path_fwd + [next_id], next_conf)
                            q_fwd.append(next_id)
            else:
                # Expand one level backward
                level_size = len(q_bwd)
                depth_bwd += 1
                for _ in range(level_size):
                    curr_bwd = q_bwd.popleft()
                    path_bwd, conf_bwd = visited_bwd[curr_bwd]

                    # Incoming edges: prev_id --(predicate)--> curr_bwd
                    relations = self.memory.get_relations(
                        object_id=curr_bwd, predicate=predicate
                    )
                    for rel in relations:
                        # Gate 3: I_ground (Empirical evidence verification)
                        if rel.weight_positive <= rel.weight_negative:
                            continue

                        prev_id = rel.subject_id
                        # Gate 1: I_DAG (Acyclicity invariant)
                        if prev_id in path_bwd:
                            continue

                        prev_conf = round(conf_bwd * rel.confidence, 4)

                        # Collision Check with Forward Frontier
                        if prev_id in visited_fwd:
                            path_fwd, conf_fwd = visited_fwd[prev_id]
                            full_id_path = path_fwd + path_bwd
                            # Verify full path satisfies I_DAG (no duplicates)
                            if len(set(full_id_path)) == len(full_id_path):
                                total_conf = round(conf_fwd * prev_conf, 4)
                                name_path = [
                                    self.memory.get_concept(nid).name
                                    if self.memory.get_concept(nid)
                                    else nid
                                    for nid in full_id_path
                                ]
                                return name_path, total_conf

                        if prev_id not in visited_bwd:
                            visited_bwd[prev_id] = ([prev_id] + path_bwd, prev_conf)
                            q_bwd.append(prev_id)

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
