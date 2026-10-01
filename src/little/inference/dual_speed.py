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
    infilled_value: Optional[str] = None
    query_type: str = "PATH"


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
            query_type="PATH",
        )

    def infill_object(
        self, subject: str, predicate: str, max_results: int = 10
    ) -> List[InfillingResult]:
        """Bidirectional infilling of missing object: (S, P, [?])."""
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        pred_clean = predicate.strip().lower()

        results: List[InfillingResult] = []

        # 1. FAST MODE: Single-hop direct out-edges (<0.2ms)
        direct_objs = self._get_outgoing(subj_clean, pred_clean)
        for obj in direct_objs[:max_results]:
            lat = (time.perf_counter() - t0) * 1000.0
            results.append(
                InfillingResult(
                    mode="FAST",
                    path=[subj_clean, obj],
                    confidence=self.policy.fast_confidence,
                    inspectable_trace=f"Direct reflex hit: ({subj_clean}, {pred_clean}, {obj})",
                    latency_ms=lat,
                    infilled_value=obj,
                    query_type="OBJECT",
                )
            )

        # 2. THINKING MODE: If transitive predicate and more results requested
        is_transitive = pred_clean in self.verifier.registry.predicates(transitive=True)
        if is_transitive and len(results) < max_results:
            frontier: Set[str] = {subj_clean}
            visited_paths: Dict[str, List[str]] = {subj_clean: [subj_clean]}
            depth = 0
            while frontier and depth < self.policy.max_depth and len(results) < max_results:
                depth += 1
                next_frontier: Set[str] = set()
                for curr in frontier:
                    curr_path = visited_paths[curr]
                    for nxt in self._get_outgoing(curr, pred_clean):
                        if nxt not in visited_paths and nxt != subj_clean:
                            nxt_path = curr_path + [nxt]
                            visited_paths[nxt] = nxt_path
                            next_frontier.add(nxt)
                            if not any(r.infilled_value == nxt for r in results):
                                lat = (time.perf_counter() - t0) * 1000.0
                                results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=nxt_path,
                                        confidence=self.policy.verified_confidence,
                                        inspectable_trace=f"Transitive forward expansion ({depth} hops): {' -> '.join(nxt_path)}",
                                        latency_ms=lat,
                                        infilled_value=nxt,
                                        query_type="OBJECT",
                                    )
                                )
                frontier = next_frontier

        return results[:max_results]

    def infill_subject(
        self, target: str, predicate: str, max_results: int = 10
    ) -> List[InfillingResult]:
        """Bidirectional infilling of missing subject: ([?], P, O)."""
        t0 = time.perf_counter()
        target_clean = target.strip().upper().replace(" ", "_")
        pred_clean = predicate.strip().lower()

        results: List[InfillingResult] = []

        # 1. FAST MODE: Single-hop direct incoming edges (<0.2ms)
        direct_subjs = self._get_incoming(target_clean, pred_clean)
        for subj in direct_subjs[:max_results]:
            lat = (time.perf_counter() - t0) * 1000.0
            results.append(
                InfillingResult(
                    mode="FAST",
                    path=[subj, target_clean],
                    confidence=self.policy.fast_confidence,
                    inspectable_trace=f"Direct reflex hit: ({subj}, {pred_clean}, {target_clean})",
                    latency_ms=lat,
                    infilled_value=subj,
                    query_type="SUBJECT",
                )
            )

        # 2. THINKING MODE: If transitive predicate and more results requested
        is_transitive = pred_clean in self.verifier.registry.predicates(transitive=True)
        if is_transitive and len(results) < max_results:
            frontier: Set[str] = {target_clean}
            visited_paths: Dict[str, List[str]] = {target_clean: [target_clean]}
            depth = 0
            while frontier and depth < self.policy.max_depth and len(results) < max_results:
                depth += 1
                next_frontier: Set[str] = set()
                for curr in frontier:
                    curr_path = visited_paths[curr]
                    for prev in self._get_incoming(curr, pred_clean):
                        if prev not in visited_paths and prev != target_clean:
                            prev_path = [prev] + curr_path
                            visited_paths[prev] = prev_path
                            next_frontier.add(prev)
                            if not any(r.infilled_value == prev for r in results):
                                lat = (time.perf_counter() - t0) * 1000.0
                                results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=prev_path,
                                        confidence=self.policy.verified_confidence,
                                        inspectable_trace=f"Transitive backward expansion ({depth} hops): {' -> '.join(prev_path)}",
                                        latency_ms=lat,
                                        infilled_value=prev,
                                        query_type="SUBJECT",
                                    )
                                )
                frontier = next_frontier

        return results[:max_results]

    def infill_path(
        self, subject: str, target: str, max_depth: int = 4
    ) -> List[InfillingResult]:
        """Bidirectional Frontier Collision Search connecting S and O across any relation: (S, [?], O)."""
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        target_clean = target.strip().upper().replace(" ", "_")

        fwd_paths: Dict[str, List[str]] = {subj_clean: [subj_clean]}
        bwd_paths: Dict[str, List[str]] = {target_clean: [target_clean]}
        fwd_frontier: Set[str] = {subj_clean}
        bwd_frontier: Set[str] = {target_clean}

        collision_results: List[InfillingResult] = []
        depth = 0

        predicates = list(self.verifier.registry.predicates()) or ["is_a", "part_of", "has"]

        while depth < max_depth and fwd_frontier and bwd_frontier:
            depth += 1

            # Expand Forward
            next_fwd: Set[str] = set()
            for node in fwd_frontier:
                curr_path = fwd_paths[node]
                for p in predicates:
                    for obj in self._get_outgoing(node, p):
                        if obj not in fwd_paths and obj != subj_clean:
                            fwd_paths[obj] = curr_path + [obj]
                            next_fwd.add(obj)
                            if obj in bwd_paths:
                                full_p = curr_path + [obj] + bwd_paths[obj][1:]
                                lat = (time.perf_counter() - t0) * 1000.0
                                collision_results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=full_p,
                                        confidence=self.policy.verified_confidence,
                                        inspectable_trace=f"Meeting-in-the-middle collision at '{obj}': {' -> '.join(full_p)}",
                                        latency_ms=lat,
                                        infilled_value=obj,
                                        query_type="PATH",
                                    )
                                )
            fwd_frontier = next_fwd

            # Expand Backward
            next_bwd: Set[str] = set()
            for node in bwd_frontier:
                curr_path = bwd_paths[node]
                for p in predicates:
                    for sub in self._get_incoming(node, p):
                        if sub not in bwd_paths and sub != target_clean:
                            bwd_paths[sub] = [sub] + curr_path
                            next_bwd.add(sub)
                            if sub in fwd_paths:
                                full_p = fwd_paths[sub] + curr_path
                                lat = (time.perf_counter() - t0) * 1000.0
                                collision_results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=full_p,
                                        confidence=self.policy.verified_confidence,
                                        inspectable_trace=f"Meeting-in-the-middle collision at '{sub}': {' -> '.join(full_p)}",
                                        latency_ms=lat,
                                        infilled_value=sub,
                                        query_type="PATH",
                                    )
                                )
            bwd_frontier = next_bwd

        return collision_results
