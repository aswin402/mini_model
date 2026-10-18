"""DeepSeek-R1 Inspired Deterministic Invariant Verification Gates.

Enforces 4 strict rule gates over every proposed relation or proof chain:
1. I_DAG: Directed Acyclic Graph invariant (acyclicity over hierarchical predicates).
2. I_MUTEX: Mutual exclusivity refutation (disjoint concepts cannot overlap).
3. I_SORT: Sort & signature soundness (domain/range type validation).
4. I_GROUND: Supported persisted premises for proofs; assertions are structural only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from little.core.contracts import VerificationResult, VerificationStatus
from little.knowledge.registry import SchemaRegistry
from little.memory.store import MemoryStore


@dataclass
class InvariantGateResult:
    passed: bool
    violated_gate: str | None = None
    error_message: str = ""
    proof_trace: list[str] = field(default_factory=list)

    def to_verification_result(self) -> VerificationResult:
        checks = {
            gate.split(":", 1)[0]: True
            for gate in self.proof_trace
            if ":" in gate and gate.split(":", 1)[0].startswith("I_")
        }
        if self.violated_gate:
            checks[self.violated_gate] = False
        return VerificationResult(
            status=(
                VerificationStatus.PASSED if self.passed else VerificationStatus.FAILED
            ),
            checks=checks,
            reasons=[self.error_message]
            if self.error_message
            else list(self.proof_trace),
        )


class DeepSeekInvariantVerifier:
    """Deterministic rule-based verifier for schema and graph invariants."""

    def __init__(
        self, memory: MemoryStore, registry: SchemaRegistry | None = None
    ) -> None:
        self.memory = memory
        self.registry = registry or SchemaRegistry.default()
        self.acyclic_predicates = self.registry.predicates(acyclic=True)
        self.disjoint_predicates = self.registry.predicates(disjoint=True)
        self.taxonomy_predicate = self.registry.primary_predicate("taxonomy")

    def _clean_id(self, name: str) -> str:
        return name.strip().upper().replace(" ", "_")

    def _resolve_target_ids(self, name: str) -> set[str]:
        clean = self._clean_id(name)
        ids = {clean, name.strip().lower()}
        concept = self.memory.get_concept(
            name.strip().lower()
        ) or self.memory.get_concept(name.strip().lower().replace("_", " "))
        if concept:
            ids.add(concept.id)
            ids.add(self._clean_id(concept.name))
        return ids

    def _get_outgoing_objects(self, node: str, predicate: str) -> set[str]:
        node_ids = self._resolve_target_ids(node)
        outgoing: set[str] = set()
        for nid in node_ids:
            for rel in self.memory.get_relations(subject_id=nid, predicate=predicate):
                if rel.weight_positive <= rel.weight_negative:
                    continue
                outgoing.add(self._clean_id(rel.object_id))
                c_obj = self.memory.get_concept(rel.object_id)
                if c_obj:
                    outgoing.add(self._clean_id(c_obj.name))
        return outgoing

    def _check_cycle(self, start_node: str, target_node: str, predicate: str) -> bool:
        """DFS check if start_node can reach target_node via hierarchical predicates."""
        target_ids = self._resolve_target_ids(target_node)
        start_clean = self._clean_id(start_node)
        visited: set[str] = set()
        stack = [start_clean]
        while stack:
            curr = stack.pop()
            if curr in target_ids:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for obj_clean in self._get_outgoing_objects(curr, predicate):
                if obj_clean in target_ids:
                    return True
                if obj_clean not in visited:
                    stack.append(obj_clean)
        return False

    def _get_ancestors(self, node: str) -> set[str]:
        """Collects all taxonomic ancestors of a node."""
        clean = self._clean_id(node)
        ancestors: set[str] = {clean}
        stack = [clean]
        while stack:
            curr = stack.pop()
            for pred in [self.taxonomy_predicate] if self.taxonomy_predicate else []:
                for obj_clean in self._get_outgoing_objects(curr, pred):
                    if obj_clean not in ancestors:
                        ancestors.add(obj_clean)
                        stack.append(obj_clean)
        return ancestors

    def _get_disjoints(self, node: str) -> set[str]:
        """Collects all concepts mutually exclusive with node."""
        node_ids = self._resolve_target_ids(node)
        disjoints: set[str] = set()
        for nid in node_ids:
            for predicate in self.disjoint_predicates:
                for rel in self.memory.get_relations(
                    subject_id=nid, predicate=predicate
                ):
                    # Legacy negative disjoint_with also represents exclusion.
                    if rel.weight_positive == rel.weight_negative:
                        continue
                    disjoints.add(self._clean_id(rel.object_id))
                    c_obj = self.memory.get_concept(rel.object_id)
                    if c_obj:
                        disjoints.add(self._clean_id(c_obj.name))
                for rel in self.memory.get_relations(
                    object_id=nid, predicate=predicate
                ):
                    if rel.weight_positive == rel.weight_negative:
                        continue
                    disjoints.add(self._clean_id(rel.subject_id))
                    c_subj = self.memory.get_concept(rel.subject_id)
                    if c_subj:
                        disjoints.add(self._clean_id(c_subj.name))
        return disjoints

    def verify_relation(
        self, subject: str, predicate: str, obj: str
    ) -> InvariantGateResult:
        """Verifies a single relation against the 4 Invariant Gates."""
        trace = []
        s_clean = self._clean_id(subject)
        o_clean = self._clean_id(obj)
        p_clean = predicate.strip().lower()

        if self.registry.relation(p_clean) is None:
            return InvariantGateResult(
                passed=False,
                violated_gate="I_SORT",
                error_message=f"Unregistered predicate: {predicate}",
            )

        # Gate 1: I_DAG (Acyclicity)
        if p_clean in self.acyclic_predicates:
            if s_clean == o_clean:
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_DAG",
                    error_message=f"Self-referential cycle: {subject} {predicate} {obj}",
                )
            if self._check_cycle(o_clean, s_clean, p_clean):
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_DAG",
                    error_message=f"Cycle detected in DAG: {obj} already leads to {subject}",
                )
        trace.append("I_DAG: Acyclicity check passed")

        # Gate 2: I_MUTEX (Mutual Exclusivity)
        if p_clean == self.taxonomy_predicate:
            subj_ancestors = self._get_ancestors(s_clean)
            obj_ancestors = self._get_ancestors(o_clean)
            for sa in subj_ancestors:
                disjoints = self._get_disjoints(sa)
                for oa in obj_ancestors:
                    if oa in disjoints:
                        return InvariantGateResult(
                            passed=False,
                            violated_gate="I_MUTEX",
                            error_message=(
                                f"Mutual exclusivity violated: {subject} ({sa}) is disjoint with {obj} ({oa})"
                            ),
                        )
        elif p_clean in self.disjoint_predicates:
            if s_clean == o_clean:
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_MUTEX",
                    error_message=f"Self-disjointness error: {subject} cannot be disjoint with itself",
                )
            subj_ancestors = self._get_ancestors(s_clean)
            obj_ancestors = self._get_ancestors(o_clean)
            if o_clean in subj_ancestors or s_clean in obj_ancestors:
                return InvariantGateResult(
                    passed=False,
                    violated_gate="I_MUTEX",
                    error_message=(
                        f"Mutual exclusivity violated: {subject} and {obj} share a taxonomic lineage"
                    ),
                )
        trace.append("I_MUTEX: Mutual exclusivity check passed")

        # Gate 3: I_SORT (Domain / Range sound structure)
        if not subject or not predicate or not obj:
            return InvariantGateResult(
                passed=False,
                violated_gate="I_SORT",
                error_message="Sort error: Subject, predicate, or object is empty",
            )

        def concept_type(name: str) -> str:
            concept = self.memory.get_concept(
                name.strip().lower()
            ) or self.memory.get_concept(name.strip().lower().replace("_", " "))
            declared = concept.category if concept else None
            # Ordinary categories (bird, food, etc.) are taxonomy, not sorts.
            return (
                declared
                if declared in {"concept", "entity", "value", "event", "quantity"}
                else "concept"
            )

        schema = self.registry.relation(p_clean)
        object_type = "value" if schema and schema.attribute_key is not None else concept_type(obj)
        signature = self.registry.validate_relation(p_clean, concept_type(subject), object_type)
        if not signature.valid:
            return InvariantGateResult(False, "I_SORT", signature.reason)
        trace.append("I_SORT: Type & sort signature sound")

        # Gate 4: no evidence input exists at this assertion boundary.
        trace.append(
            "I_GROUND: Structural assertion only; factual evidence not assessed"
        )

        return InvariantGateResult(passed=True, proof_trace=trace)

    def verify_proof_chain(
        self, chain: list[tuple[str, str, str]]
    ) -> InvariantGateResult:
        """Verifies an entire multi-hop proof chain."""
        trace = []
        for i, (s, p, o) in enumerate(chain):
            premises = []
            for sid in self._resolve_target_ids(s):
                for oid in self._resolve_target_ids(o):
                    premises.extend(
                        self.memory.get_relations(
                            subject_id=sid, predicate=p, object_id=oid
                        )
                    )
            if not any(r.weight_positive > r.weight_negative for r in premises):
                return InvariantGateResult(
                    False,
                    "I_GROUND",
                    f"Step {i + 1} has no supported persisted premise",
                )
            step_res = self.verify_relation(s, p, o)
            if not step_res.passed:
                return InvariantGateResult(
                    passed=False,
                    violated_gate=step_res.violated_gate,
                    error_message=f"Step {i + 1} ({s} {p} {o}) failed: {step_res.error_message}",
                )
            trace.append(f"Step {i + 1}: ({s}, {p}, {o}) verified")
            trace.extend(
                line
                for line in step_res.proof_trace
                if not line.startswith("I_GROUND:")
            )
            trace.append(
                "I_GROUND: Persisted premise has positive net evidence; external truth not established"
            )
        return InvariantGateResult(passed=True, proof_trace=trace)
