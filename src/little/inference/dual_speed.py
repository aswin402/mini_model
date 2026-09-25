"""GLM-5.3-Flash Inspired Dual-Speed Bidirectional Infilling Engine.

Provides Fast Mode (<0.2ms direct reflex lookup) and Thinking Mode (<2.0ms
Bidirectional Frontier Collision Search cutting complexity from O(b^d) to O(2 * b^(d/2))).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.reasoning_policy import ReasoningPolicy
from little.memory.store import MemoryStore


@dataclass
class InfillingResult:
    mode: str
    path: List[str]
    confidence: float
    inspectable_trace: str
    latency_ms: float


class DualSpeedInfillingEngine:
    """Dual-speed cognitive reasoning engine with bidirectional frontier collision search."""

    def __init__(
        self,
        memory: MemoryStore,
        verifier: DeepSeekInvariantVerifier,
        policy: ReasoningPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.verifier = verifier
        self.policy = policy or ReasoningPolicy.default()

    def _resolve_variants(self, node: str) -> Set[str]:
        variants = {
            node,
            node.lower(),
            node.upper(),
            node.capitalize(),
            node.replace("_", " "),
        }
        for lookup in (node, node.replace("_", " ")):
            c = self.memory.get_concept(lookup)
            if c:
                variants.add(c.id)
                variants.add(c.name)
                variants.add(c.name.upper())
        return variants

    def _to_name(self, id_or_name: str) -> str:
        if id_or_name.startswith("concept_"):
            c = self.memory.get_concept(id_or_name)
            if c:
                return c.name.strip().upper().replace(" ", "_")
        return id_or_name.strip().upper().replace(" ", "_")

    def _get_outgoing(self, node: str, predicate: str) -> List[str]:
        objs: Set[str] = set()
        for variant in self._resolve_variants(node):
            for r in self.memory.get_relations(subject_id=variant, predicate=predicate):
                if r.weight_positive > r.weight_negative:
                    objs.add(self._to_name(r.object_id))
        return list(objs)

    def _get_incoming(self, node: str, predicate: str) -> List[str]:
        subjs: Set[str] = set()
        for variant in self._resolve_variants(node):
            for r in self.memory.get_relations(object_id=variant, predicate=predicate):
                if r.weight_positive > r.weight_negative:
                    subjs.add(self._to_name(r.subject_id))
        return list(subjs)

    def query(
        self, subject: str, target: str, predicate: str | None = None
    ) -> InfillingResult:
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        target_clean = target.strip().upper().replace(" ", "_")
        configured_predicate = predicate or self.verifier.registry.primary_predicate(
            "taxonomy"
        )
        pred_clean = (configured_predicate or "").strip().lower()

        # 1. FAST MODE: Single-hop direct index / cache hit (<0.2ms)
        direct_objs = self._get_outgoing(subj_clean, pred_clean)
        if target_clean in direct_objs:
            lat = (time.perf_counter() - t0) * 1000.0
            return InfillingResult(
                mode="FAST",
                path=[subj_clean, target_clean],
                confidence=self.policy.fast_confidence,
                inspectable_trace=f"Direct reflex hit: ({subj_clean}, {pred_clean}, {target_clean})",
                latency_ms=lat,
            )

        # 2. THINKING MODE: Bidirectional Frontier Collision Search (Meeting-in-the-Middle)
        # All discovered paths from subject and target
        fwd_paths: Dict[str, List[str]] = {subj_clean: [subj_clean]}
        bwd_paths: Dict[str, List[str]] = {target_clean: [target_clean]}

        fwd_frontier: Set[str] = {subj_clean}
        bwd_frontier: Set[str] = {target_clean}

        collision_node: Optional[str] = None
        max_depth = self.policy.max_depth
        depth = 0

        trace_log = [
            "<think>",
            f"Deliberative Thinking Mode initiated for: ({subj_clean}, {pred_clean}, [?], ..., {target_clean})",
            f"Initial frontiers: Fwd={{ {subj_clean} }}, Bwd={{ {target_clean} }}",
        ]

        while depth < max_depth and not collision_node:
            depth += 1

            # Expand Forward Frontier
            next_fwd: Set[str] = set()
            for node in fwd_frontier:
                curr_path = fwd_paths[node]
                for obj_c in self._get_outgoing(node, pred_clean):
                    if obj_c not in fwd_paths:
                        fwd_paths[obj_c] = curr_path + [obj_c]
                        next_fwd.add(obj_c)
                        if obj_c in bwd_paths:
                            collision_node = obj_c
                            break
                if collision_node:
                    break
            if collision_node:
                break
            fwd_frontier = next_fwd

            # Expand Backward Frontier
            next_bwd: Set[str] = set()
            for node in bwd_frontier:
                curr_path = bwd_paths[node]
                for sub_c in self._get_incoming(node, pred_clean):
                    if sub_c not in bwd_paths:
                        bwd_paths[sub_c] = [sub_c] + curr_path
                        next_bwd.add(sub_c)
                        if sub_c in fwd_paths:
                            collision_node = sub_c
                            break
                if collision_node:
                    break
            if collision_node:
                break
            bwd_frontier = next_bwd

            if not fwd_frontier and not bwd_frontier:
                break

        lat = (time.perf_counter() - t0) * 1000.0

        if collision_node:
            fwd_part = fwd_paths[collision_node]
            bwd_part = bwd_paths[collision_node]
            full_path = fwd_part[:-1] + bwd_part

            # Verify through DeepSeek-R1 Invariant Gates
            proof_steps = [
                (full_path[i], pred_clean, full_path[i + 1])
                for i in range(len(full_path) - 1)
            ]
            gate_res = self.verifier.verify_proof_chain(proof_steps)

            trace_log.append(
                f"Frontier collision verified at junction node: {collision_node}"
            )
            trace_log.append(f"Full proof path discovered: {' -> '.join(full_path)}")
            trace_log.extend([f"Gate check: {step}" for step in gate_res.proof_trace])
            trace_log.append("</think>")

            return InfillingResult(
                mode="THINKING",
                path=full_path,
                confidence=(
                    self.policy.verified_confidence
                    if gate_res.passed
                    else self.policy.unknown_confidence
                ),
                inspectable_trace="\n".join(trace_log),
                latency_ms=lat,
            )

        trace_log.append(
            f"No collision detected within depth {max_depth}. Status: UNKNOWN."
        )
        trace_log.append("</think>")
        return InfillingResult(
            mode="THINKING",
            path=[],
            confidence=self.policy.unknown_confidence,
            inspectable_trace="\n".join(trace_log),
            latency_ms=lat,
        )
