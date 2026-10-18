"""GLM-5.3-Flash Inspired Dual-Speed Bidirectional Infilling Engine with Adaptive Depth Pondering.

Provides Fast Mode (<0.2ms direct reflex lookup) and Thinking Mode (<2.0ms
Bidirectional Frontier Collision Search cutting complexity from O(b^d) to O(2 * b^(d/2))).

Enhanced with Adaptive Depth Pondering (inspired by mini-AGI / PonderNet):
- Evaluates step-by-step halting condition H(d) at each recurrent depth.
- Halts early with calibrated confidence upon proof closure or invariant refutation.
- Halts early upon frontier exhaustion rather than looping wastefully.
- Emits fully inspectable pondering traces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise

from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.reasoning_policy import ReasoningPolicy
from little.memory.store import MemoryStore


@dataclass
class InfillingResult:
    mode: str
    path: list[str]
    confidence: float
    inspectable_trace: str
    latency_ms: float
    infilled_value: str | None = None
    query_type: str = "PATH"
    ponder_depth: int = 1
    halt_reason: str = "direct_reflex"
    halt_probability: float = 1.0
    ponder_cost: float = 0.0


class DualSpeedInfillingEngine:
    """Dual-speed cognitive reasoning engine with adaptive depth pondering and bidirectional frontier collision search."""

    def __init__(
        self,
        memory: MemoryStore,
        verifier: DeepSeekInvariantVerifier,
        policy: ReasoningPolicy | None = None,
        halt_threshold: float = 0.9,
        ponder_beta: float = 0.01,
        halt_freeze: bool = True,
    ) -> None:
        self.memory = memory
        self.verifier = verifier
        self.policy = policy or ReasoningPolicy.default()
        self.halt_threshold = float(halt_threshold)
        self.ponder_beta = float(ponder_beta)
        self.halt_freeze = bool(halt_freeze)

    def _compute_ponder_metrics(
        self, depth: int, solved: bool, frontier_size: int
    ) -> tuple[float, float]:
        """Compute PonderNet cumulative halting probability P(halt <= d) and KL-style ponder cost."""
        if solved or frontier_size == 0:
            halt_prob = 1.0
        else:
            # Geometric hazard increases as frontier thins or depth approaches max_depth
            max_d = max(1, self.policy.max_depth)
            step_hazard = min(0.95, (depth / max_d) * 0.45 + (1.0 / (1.0 + frontier_size)) * 0.35)
            survival = (1.0 - step_hazard) ** depth
            halt_prob = round(1.0 - survival, 4)
        ponder_cost = round(self.ponder_beta * depth, 4)
        return halt_prob, ponder_cost

    def _resolve_variants(self, node: str) -> set[str]:
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

    def _get_outgoing(self, node: str, predicate: str | None = None) -> list[str]:
        objs: set[str] = set()
        for variant in self._resolve_variants(node):
            for r in self.memory.get_relations(subject_id=variant, predicate=predicate):
                if r.weight_positive > r.weight_negative:
                    objs.add(self._to_name(r.object_id))
        return list(objs)

    def _get_incoming(self, node: str, predicate: str | None = None) -> list[str]:
        subjs: set[str] = set()
        for variant in self._resolve_variants(node):
            for r in self.memory.get_relations(object_id=variant, predicate=predicate):
                if r.weight_positive > r.weight_negative:
                    subjs.add(self._to_name(r.subject_id))
        return list(subjs)

    def _path_confidence(self, path: list[str], predicate: str | None) -> float:
        confidence = 1.0
        for subject, target in pairwise(path):
            premises = [
                r
                for sid in self._resolve_variants(subject)
                for r in self.memory.get_relations(subject_id=sid, predicate=predicate)
                if self._to_name(r.object_id) == target
                and r.weight_positive > r.weight_negative
            ]
            if not premises:
                return 0.0
            confidence *= max(r.confidence for r in premises)
        return round(confidence, 4)

    def _opposed(self, subject: str, predicate: str, target: str) -> bool:
        return any(
            r.weight_negative >= r.weight_positive
            for sid in self._resolve_variants(subject)
            for r in self.memory.get_relations(subject_id=sid, predicate=predicate)
            if self._to_name(r.object_id) == target
        )

    def _valid_path(self, path: list[str], predicate: str) -> bool:
        if self._opposed(path[0], predicate, path[-1]):
            return False
        return self.verifier.verify_proof_chain(
            [(a, predicate, b) for a, b in pairwise(path)]
        ).passed

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
        if target_clean in direct_objs and self.policy.max_depth >= 1:
            gate = self.verifier.verify_proof_chain(
                [(subj_clean, pred_clean, target_clean)]
            )
            if not gate.passed or self._opposed(subj_clean, pred_clean, target_clean):
                return InfillingResult(
                    "FAST",
                    [],
                    self.policy.unknown_confidence,
                    f"{gate.violated_gate or 'CONTRADICTED'}: {gate.error_message}",
                    (time.perf_counter() - t0) * 1000,
                    halt_reason="invariant_violation_refuted",
                )
            lat = (time.perf_counter() - t0) * 1000.0
            return InfillingResult(
                mode="FAST",
                path=[subj_clean, target_clean],
                confidence=self._path_confidence(
                    [subj_clean, target_clean], pred_clean
                ),
                inspectable_trace=f"Direct reflex hit: ({subj_clean}, {pred_clean}, {target_clean})",
                latency_ms=lat,
                ponder_depth=1,
                halt_reason="direct_reflex_hit",
            )

        if pred_clean not in self.verifier.registry.predicates(transitive=True):
            return InfillingResult(
                "THINKING",
                [],
                self.policy.unknown_confidence,
                "UNKNOWN: predicate does not permit transitive composition",
                (time.perf_counter() - t0) * 1000,
                halt_reason="nontransitive_predicate",
            )

        # 2. THINKING MODE: Adaptive Depth Pondering with Bidirectional Collision Search
        fwd_paths: dict[str, list[str]] = {subj_clean: [subj_clean]}
        bwd_paths: dict[str, list[str]] = {target_clean: [target_clean]}

        fwd_frontier: set[str] = {subj_clean}
        bwd_frontier: set[str] = {target_clean}

        collision_node: str | None = None
        max_depth = self.policy.max_depth
        depth = 0
        halt_reason = "unresolved"

        trace_log = [
            "<think>",
            f"Deliberative Thinking Mode initiated for: ({subj_clean}, {pred_clean}, [?], ..., {target_clean})",
            f"Initial frontiers: Fwd={{ {subj_clean} }}, Bwd={{ {target_clean} }}",
        ]

        while depth < max_depth and not collision_node:
            depth += 1
            fwd_count_before = len(fwd_frontier)
            bwd_count_before = len(bwd_frontier)

            # Expand Forward Frontier
            next_fwd: set[str] = set()
            for node in fwd_frontier:
                curr_path = fwd_paths[node]
                for obj_c in self._get_outgoing(node, pred_clean):
                    if obj_c not in fwd_paths:
                        fwd_paths[obj_c] = curr_path + [obj_c]
                        next_fwd.add(obj_c)
                        if (
                            obj_c in bwd_paths
                            and len(fwd_paths[obj_c]) + len(bwd_paths[obj_c]) - 2
                            <= max_depth
                        ):
                            collision_node = obj_c
                            halt_reason = "collision_verified"
                            break
                if collision_node:
                    break

            if not collision_node:
                fwd_frontier = next_fwd

                # Expand Backward Frontier
                next_bwd: set[str] = set()
                for node in bwd_frontier:
                    curr_path = bwd_paths[node]
                    for sub_c in self._get_incoming(node, pred_clean):
                        if sub_c not in bwd_paths:
                            bwd_paths[sub_c] = [sub_c] + curr_path
                            next_bwd.add(sub_c)
                            if (
                                sub_c in fwd_paths
                                and len(fwd_paths[sub_c]) + len(bwd_paths[sub_c]) - 2
                                <= max_depth
                            ):
                                collision_node = sub_c
                                halt_reason = "collision_verified"
                                break
                    if collision_node:
                        break
                bwd_frontier = next_bwd

            # Adaptive Halting Check (PonderNet cumulative halting probability & ponder cost)
            frontier_sz = len(fwd_frontier) + len(bwd_frontier)
            halt_prob, ponder_cost = self._compute_ponder_metrics(
                depth=depth, solved=collision_node is not None, frontier_size=frontier_sz
            )
            if collision_node:
                trace_log.append(
                    f'<ponder depth="{depth}" fwd_nodes="{fwd_count_before}" bwd_nodes="{bwd_count_before}" collision="{collision_node}" halt="true" halt_prob="{halt_prob}" ponder_cost="{ponder_cost}" reason="collision_detected" />'
                )
                break

            if not fwd_frontier and not bwd_frontier:
                halt_reason = "frontier_exhausted"
                trace_log.append(
                    f'<ponder depth="{depth}" fwd_nodes="0" bwd_nodes="0" halt="true" halt_prob="{halt_prob}" ponder_cost="{ponder_cost}" reason="frontier_exhausted" />'
                )
                break

            if self.halt_freeze and halt_prob >= self.halt_threshold:
                halt_reason = "ponder_threshold_reached"
                trace_log.append(
                    f'<ponder depth="{depth}" fwd_nodes="{len(fwd_frontier)}" bwd_nodes="{len(bwd_frontier)}" halt="true" halt_prob="{halt_prob}" ponder_cost="{ponder_cost}" reason="ponder_threshold_reached" />'
                )
                break

            trace_log.append(
                f'<ponder depth="{depth}" fwd_nodes="{len(fwd_frontier)}" bwd_nodes="{len(bwd_frontier)}" halt="false" halt_prob="{halt_prob}" ponder_cost="{ponder_cost}" status="expanding" />'
            )

        if not collision_node and depth >= max_depth and halt_reason == "unresolved":
            halt_reason = "max_depth_exceeded"

        final_halt_prob, final_ponder_cost = self._compute_ponder_metrics(
            depth=max(1, depth),
            solved=collision_node is not None,
            frontier_size=len(fwd_frontier) + len(bwd_frontier),
        )
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

            if gate_res.passed and not self._opposed(
                subj_clean, pred_clean, target_clean
            ):
                trace_log.append(
                    f"Frontier collision verified at junction node: {collision_node}"
                )
                trace_log.append(
                    f"Full proof path discovered: {' -> '.join(full_path)}"
                )
                trace_log.extend(
                    [f"Gate check: {step}" for step in gate_res.proof_trace]
                )
                trace_log.append("</think>")

                return InfillingResult(
                    mode="THINKING",
                    path=full_path,
                    confidence=self._path_confidence(full_path, pred_clean),
                    inspectable_trace="\n".join(trace_log),
                    latency_ms=lat,
                    ponder_depth=depth,
                    halt_reason="proof_closed_verified",
                    halt_probability=final_halt_prob,
                    ponder_cost=final_ponder_cost,
                )
            else:
                if self._opposed(subj_clean, pred_clean, target_clean):
                    trace_log.append(
                        "CONTRADICTED: supported path has direct opposition"
                    )
                trace_log.append(
                    f"Frontier collision refuted by invariant gates at junction: {collision_node}"
                )
                trace_log.extend(
                    [f"Gate violation: {step}" for step in gate_res.proof_trace]
                )
                trace_log.append("</think>")
                return InfillingResult(
                    mode="THINKING",
                    path=[],
                    confidence=self.policy.unknown_confidence,
                    inspectable_trace="\n".join(trace_log),
                    latency_ms=lat,
                    query_type="PATH",
                    ponder_depth=depth,
                    halt_reason="invariant_violation_refuted",
                    halt_probability=final_halt_prob,
                    ponder_cost=final_ponder_cost,
                )

        trace_log.append(
            f"No collision detected within depth {max_depth}. Status: UNKNOWN. Reason: {halt_reason}."
        )
        trace_log.append("</think>")
        return InfillingResult(
            mode="THINKING",
            path=[],
            confidence=self.policy.unknown_confidence,
            inspectable_trace="\n".join(trace_log),
            latency_ms=lat,
            query_type="PATH",
            ponder_depth=depth,
            halt_reason=halt_reason,
            halt_probability=final_halt_prob,
            ponder_cost=final_ponder_cost,
        )

    def infill_object(
        self, subject: str, predicate: str, max_results: int = 10
    ) -> list[InfillingResult]:
        """Bidirectional infilling of missing object: (S, P, [?]) with adaptive depth expansion."""
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        pred_clean = predicate.strip().lower()

        results: list[InfillingResult] = []

        # 1. FAST MODE: Single-hop direct out-edges (<0.2ms)
        direct_objs = self._get_outgoing(subj_clean, pred_clean)
        for obj in direct_objs[:max_results]:
            lat = (time.perf_counter() - t0) * 1000.0
            results.append(
                InfillingResult(
                    mode="FAST",
                    path=[subj_clean, obj],
                    confidence=self._path_confidence([subj_clean, obj], pred_clean),
                    inspectable_trace=f"Direct reflex hit: ({subj_clean}, {pred_clean}, {obj})",
                    latency_ms=lat,
                    infilled_value=obj,
                    query_type="OBJECT",
                    ponder_depth=1,
                    halt_reason="direct_reflex_hit",
                )
            )

        # 2. THINKING MODE: If transitive predicate and more results requested
        is_transitive = pred_clean in self.verifier.registry.predicates(transitive=True)
        if is_transitive and len(results) < max_results:
            frontier: set[str] = {subj_clean}
            visited_paths: dict[str, list[str]] = {subj_clean: [subj_clean]}
            depth = 0
            while (
                frontier
                and depth < self.policy.max_depth
                and len(results) < max_results
            ):
                depth += 1
                next_frontier: set[str] = set()
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
                                        confidence=self._path_confidence(
                                            nxt_path, pred_clean
                                        ),
                                        inspectable_trace=f"Transitive forward expansion ({depth} hops): {' -> '.join(nxt_path)}",
                                        latency_ms=lat,
                                        infilled_value=nxt,
                                        query_type="OBJECT",
                                        ponder_depth=depth,
                                        halt_reason="transitive_expansion",
                                    )
                                )
                frontier = next_frontier

        return [
            r
            for r in results
            if len(r.path) - 1 <= self.policy.max_depth
            and self._valid_path(r.path, pred_clean)
        ][:max_results]

    def infill_subject(
        self, target: str, predicate: str, max_results: int = 10
    ) -> list[InfillingResult]:
        """Bidirectional infilling of missing subject: ([?], P, O) with adaptive depth expansion."""
        t0 = time.perf_counter()
        target_clean = target.strip().upper().replace(" ", "_")
        pred_clean = predicate.strip().lower()

        results: list[InfillingResult] = []

        # 1. FAST MODE: Single-hop direct incoming edges (<0.2ms)
        direct_subjs = self._get_incoming(target_clean, pred_clean)
        for subj in direct_subjs[:max_results]:
            lat = (time.perf_counter() - t0) * 1000.0
            results.append(
                InfillingResult(
                    mode="FAST",
                    path=[subj, target_clean],
                    confidence=self._path_confidence([subj, target_clean], pred_clean),
                    inspectable_trace=f"Direct reflex hit: ({subj}, {pred_clean}, {target_clean})",
                    latency_ms=lat,
                    infilled_value=subj,
                    query_type="SUBJECT",
                    ponder_depth=1,
                    halt_reason="direct_reflex_hit",
                )
            )

        # 2. THINKING MODE: If transitive predicate and more results requested
        is_transitive = pred_clean in self.verifier.registry.predicates(transitive=True)
        if is_transitive and len(results) < max_results:
            frontier: set[str] = {target_clean}
            visited_paths: dict[str, list[str]] = {target_clean: [target_clean]}
            depth = 0
            while (
                frontier
                and depth < self.policy.max_depth
                and len(results) < max_results
            ):
                depth += 1
                next_frontier: set[str] = set()
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
                                        confidence=self._path_confidence(
                                            prev_path, pred_clean
                                        ),
                                        inspectable_trace=f"Transitive backward expansion ({depth} hops): {' -> '.join(prev_path)}",
                                        latency_ms=lat,
                                        infilled_value=prev,
                                        query_type="SUBJECT",
                                        ponder_depth=depth,
                                        halt_reason="transitive_expansion",
                                    )
                                )
                frontier = next_frontier

        return [
            r
            for r in results
            if len(r.path) - 1 <= self.policy.max_depth
            and self._valid_path(r.path, pred_clean)
        ][:max_results]

    def infill_path(
        self, subject: str, target: str, max_depth: int = 4
    ) -> list[InfillingResult]:
        """Bidirectional Frontier Collision Search connecting S and O across any relation: (S, [?], O)."""
        t0 = time.perf_counter()
        subj_clean = subject.strip().upper().replace(" ", "_")
        target_clean = target.strip().upper().replace(" ", "_")

        fwd_paths: dict[str, list[str]] = {subj_clean: [subj_clean]}
        bwd_paths: dict[str, list[str]] = {target_clean: [target_clean]}
        fwd_frontier: set[str] = {subj_clean}
        bwd_frontier: set[str] = {target_clean}

        collision_results: list[InfillingResult] = []
        seen_paths: set[tuple[str, ...]] = set()
        depth = 0

        while (
            depth < max_depth
            and fwd_frontier
            and bwd_frontier
            and len(collision_results) < 10
        ):
            depth += 1

            # Expand Forward
            next_fwd: set[str] = set()
            for node in fwd_frontier:
                curr_path = fwd_paths[node]
                for obj in self._get_outgoing(node, None):
                    if obj not in fwd_paths and obj != subj_clean:
                        fwd_paths[obj] = curr_path + [obj]
                        next_fwd.add(obj)
                        if obj in bwd_paths:
                            full_p = curr_path + [obj] + bwd_paths[obj][1:]
                            p_tup = tuple(full_p)
                            if len(full_p) - 1 <= max_depth and p_tup not in seen_paths:
                                seen_paths.add(p_tup)
                                lat = (time.perf_counter() - t0) * 1000.0
                                collision_results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=full_p,
                                        confidence=self._path_confidence(full_p, None),
                                        inspectable_trace=f"Meeting-in-the-middle collision at '{obj}': {' -> '.join(full_p)}",
                                        latency_ms=lat,
                                        infilled_value=obj,
                                        query_type="PATH",
                                        ponder_depth=depth,
                                        halt_reason="collision_verified",
                                    )
                                )
            fwd_frontier = next_fwd

            # Expand Backward
            next_bwd: set[str] = set()
            for node in bwd_frontier:
                curr_path = bwd_paths[node]
                for sub in self._get_incoming(node, None):
                    if sub not in bwd_paths and sub != target_clean:
                        bwd_paths[sub] = [sub] + curr_path
                        next_bwd.add(sub)
                        if sub in fwd_paths:
                            full_p = fwd_paths[sub] + curr_path
                            p_tup = tuple(full_p)
                            if len(full_p) - 1 <= max_depth and p_tup not in seen_paths:
                                seen_paths.add(p_tup)
                                lat = (time.perf_counter() - t0) * 1000.0
                                collision_results.append(
                                    InfillingResult(
                                        mode="THINKING",
                                        path=full_p,
                                        confidence=self._path_confidence(full_p, None),
                                        inspectable_trace=f"Meeting-in-the-middle collision at '{sub}': {' -> '.join(full_p)}",
                                        latency_ms=lat,
                                        infilled_value=sub,
                                        query_type="PATH",
                                        ponder_depth=depth,
                                        halt_reason="collision_verified",
                                    )
                                )
            bwd_frontier = next_bwd

        verified = []
        for result in collision_results:
            chain = []
            for step_subj, step_obj in pairwise(result.path):
                candidates = [
                    r
                    for sid in self._resolve_variants(step_subj)
                    for r in self.memory.get_relations(subject_id=sid)
                    if self._to_name(r.object_id) == step_obj
                    and r.weight_positive > r.weight_negative
                    and self.verifier.verify_proof_chain(
                        [(step_subj, r.predicate, step_obj)]
                    ).passed
                ]
                if not candidates:
                    break
                premise = max(candidates, key=lambda r: r.confidence)
                chain.append(premise)
            else:
                confidence = 1.0
                for premise in chain:
                    confidence *= premise.confidence
                result.confidence = round(confidence, 4)
                verified.append(result)
        return verified
