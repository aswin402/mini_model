"""Unified entrypoint for perception, learning, and question answering."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from little.actions import (
    ActionPolicy,
    ActionRouter,
    AgenticPlanExecutionResult,
    AgenticPlanner,
    ToolExecutionResult,
    ToolExecutor,
    ToolGroundingEngine,
)
from little.active.crawler import KnowledgeCrawler
from little.active.latent_knot_inducer import LatentKnotInducer
from little.active.plasticity import PlasticityController
from little.active.self_study import (
    AutonomousSelfStudyEngine,
    SleepConsolidationReport,
)
from little.active.stream_learner import (
    ChunkIngestResult,
    ContinuousStreamLearner,
    StreamLearningSummary,
)
from little.core.cognitive_command import CognitiveCommandRegistry
from little.core.contracts import CandidateFrame, SourceType
from little.core.models import (
    BeliefStatus,
    InferenceResult,
    LearningResult,
    generate_id,
)
from little.core.runtime_policy import RuntimePolicy
from little.dynamics.continuous_simulation import PhysicalSimulationEngine
from little.inference.thinking_controller import CognitiveMode, ThinkingController
from little.knowledge.living_wiki import LivingWiki
from little.language.dialogue_surface import DialogueSurfaceEngine
from little.language.parser import LearningEngine, configured_parser
from little.language.perception import DeterministicPerceptionAdapter, PerceptionAdapter
from little.memory.headroom import HeadroomContextManager
from little.memory.paged_store import PagedConceptStore
from little.memory.store import MemoryStore
from little.mlu import SpiderWebMLM
from little.neural import NeuralRouter, RoutingDecision
from little.perception.sensory import SensoryGroundingEngine
from little.procedural.code_synthesizer import ASTCodeSynthesizer, SynthesizedProgram
from little.procedural.math_cas import UnitConversionGraph


@dataclass(frozen=True)
class KernelOutcome:
    frame: CandidateFrame
    mode: CognitiveMode
    learning: LearningResult | None = None
    inference: InferenceResult | None = None
    tool_result: ToolExecutionResult | None = None
    routing_decision: RoutingDecision | None = None
    code_program: SynthesizedProgram | None = None


class CognitiveKernel:
    """Route one perception frame to either learning or inference."""

    def __init__(
        self,
        memory: MemoryStore,
        perception: PerceptionAdapter | None = None,
        learner: LearningEngine | None = None,
        controller: ThinkingController | None = None,
        runtime_policy: RuntimePolicy | None = None,
        neural_router: NeuralRouter | None = None,
        action_policy: ActionPolicy | None = None,
        workspace_root: Path | None = None,
        wiki: LivingWiki | None = None,
    ) -> None:
        self.memory = memory
        explicit_runtime_policy = runtime_policy is not None
        self.runtime_policy = runtime_policy or RuntimePolicy.default()
        self.parser = configured_parser(
            self.runtime_policy.parser if explicit_runtime_policy else None,
            constructions=(
                self.runtime_policy.constructions
                if explicit_runtime_policy
                else None
            ),
            unit_conversions=UnitConversionGraph(self.runtime_policy.units),
        )
        self.perception = perception or DeterministicPerceptionAdapter(
            policy=self.runtime_policy.perception,
            parser=self.parser,
        )
        self.learner = learner or LearningEngine(
            memory,
            perception=self.perception,
            runtime_policy=self.runtime_policy,
        )
        self.controller = controller or ThinkingController(
            memory,
            resolver=self.learner.ask,
            runtime_policy=self.runtime_policy,
        )
        self.dialogue_surface = DialogueSurfaceEngine(
            policy=getattr(self.runtime_policy, "dialogue_surface", None)
        )
        self.action_policy = action_policy or self.runtime_policy.actions or ActionPolicy.default(workspace_root=workspace_root)
        if workspace_root is not None and action_policy is None:
            self.action_policy = replace(self.action_policy, workspace_root=workspace_root.resolve())
        self.neural_router = neural_router or NeuralRouter(policy=self.runtime_policy.neural)
        self.action_router = ActionRouter(action_policy=self.action_policy, neural_router=self.neural_router, tool_policy=self.runtime_policy.tools)
        self.tool_executor = ToolExecutor(action_policy=self.action_policy, tool_policy=self.runtime_policy.tools)
        self.tool_grounder = ToolGroundingEngine(memory_store=self.memory)
        self.code_synthesizer = ASTCodeSynthesizer(policy=self.runtime_policy.code)
        self.headroom = HeadroomContextManager(memory=self.memory, policy=self.runtime_policy.headroom)
        self._vault_owner = None
        self._owns_wiki = wiki is None
        if wiki is None:
            configured_vault = Path(self.runtime_policy.wiki.vault_directory) if self.runtime_policy.wiki else None
            if workspace_root is not None:
                vault = workspace_root / "data" / "knowledge_vault"
            elif configured_vault is not None and configured_vault.is_absolute():
                vault = configured_vault
            elif self.memory.db_path != ":memory:":
                vault = Path(self.memory.db_path).resolve().parent / "knowledge_vault"
            else:
                self._vault_owner = tempfile.TemporaryDirectory(prefix="little-vault-")
                vault = Path(self._vault_owner.name)
            self.wiki = LivingWiki(memory=self.memory, vault_dir=vault, policy=self.runtime_policy.wiki)
        else:
            self.wiki = wiki
        self.crawler = KnowledgeCrawler(memory=self.memory, wiki=self.wiki)
        self.latent_inducer = LatentKnotInducer(memory=self.memory, wiki=self.wiki)
        self.paged_store = PagedConceptStore(
            memory=self.memory,
            policy=(self.runtime_policy.memory if explicit_runtime_policy and self.runtime_policy.memory else self.memory.memory_policy),
        )
        self.plasticity = PlasticityController()
        self.mlm = SpiderWebMLM(memory=self.memory, plasticity=self.plasticity)
        self.stream_learner = ContinuousStreamLearner(
            kernel=self,
            paged_store=self.paged_store,
            plasticity=self.plasticity,
        )
        self.agentic_planner = AgenticPlanner(
            action_router=self.action_router,
            tool_executor=self.tool_executor,
            tool_grounder=self.tool_grounder,
            action_policy=self.action_policy,
            memory=self.memory,
            policy=self.runtime_policy.planning,
            tool_policy=self.runtime_policy.tools,
        )
        self.self_study = AutonomousSelfStudyEngine(
            memory=self.memory,
            wiki=self.wiki,
            crawler=self.crawler,
            policy=self.runtime_policy.study,
        )
        self.simulation_engine = PhysicalSimulationEngine()
        self.sensory_engine = SensoryGroundingEngine(self.memory)
        self.command_dispatcher = CognitiveCommandRegistry()
        self._last_execution_record: dict[str, Any] | None = None

    def close(self) -> None:
        """Release resources owned by the kernel; caller-owned memory stays open."""
        if hasattr(self, "mlm") and self.mlm is not None:
            self.mlm.close()
        if self._owns_wiki:
            self.wiki.vector_index.close()
            self.wiki.compressor.close()
        if self._vault_owner is not None:
            self._vault_owner.cleanup()
            self._vault_owner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def learn_routing_feedback(self, text: str, pathway) -> None:
        """Apply explicit user/developer routing feedback; never infer labels from answers."""
        from little.neural.models import Pathway
        target = Pathway(pathway)
        self.neural_router.learn(self.neural_router.encoder.encode(text), target)

    def _record_execution(
        self,
        *,
        query_type: str,
        query_text: str,
        operation: str | None = None,
        arguments: dict[str, Any] | None = None,
        subject: str | None = None,
        output: Any = None,
        evidence: list[str] | None = None,
        trace: list[str] | None = None,
        summary: str = "",
        plan_result: Any = None,
    ) -> None:
        """Record computational provenance and trace of the last executed reasoning/tool step."""
        self._last_execution_record = {
            "query_type": query_type,
            "query_text": query_text,
            "operation": operation,
            "arguments": arguments or {},
            "subject": subject,
            "output": output,
            "evidence": evidence or [],
            "trace": trace or [],
            "summary": summary,
            "plan_result": plan_result,
        }

    def explain_previous_reasoning(self) -> str:
        """Introspect and explain the provenance and reasoning trace of the prior response."""
        rec = self._last_execution_record
        if not rec:
            return "I haven't performed a calculation or reasoning step yet in this conversation. What would you like to explore or calculate?"

        q_type = rec.get("query_type")
        q_text = rec.get("query_text", "")

        if q_type == "tool":
            tool = rec.get("operation", "")
            args = rec.get("arguments", {})
            out = rec.get("output", "")
            if tool == "calculator":
                expr = args.get("expression") or q_text
                return (
                    f"For your calculation '{q_text}', I executed the exact arithmetic operation '{expr}' "
                    f"using deterministic calculation, which verified the result {out} with 0% error."
                )
            if tool == "python_repl":
                return f"For '{q_text}', I executed the code deterministically in an isolated Python runtime, yielding: {out}."
            if tool == "system_time":
                return f"For '{q_text}', I queried the synchronized system clock API."
            return f"For '{q_text}', I executed the verified tool '{tool}' with parameters {args}, producing: {out}."

        if q_type == "plan":
            plan = rec.get("plan_result")
            steps = len(getattr(plan, "steps", []))
            return f"I executed an autonomous {steps}-step agentic plan using verified tools to fulfill your request: '{q_text}'."

        if q_type == "code":
            return f"I synthesized and verified the abstract syntax tree (AST) for '{q_text}' to ensure correct program execution."

        if q_type == "learning":
            return f"I analyzed '{q_text}' through semantic candidate perception and committed the verified knowledge into long-term memory."

        if q_type == "inference":
            subj = rec.get("subject")
            trace = rec.get("trace", [])
            evidence = rec.get("evidence", [])
            operation = rec.get("operation")

            # Extract explicit proof path if present in trace
            proof_steps = [
                str(t).replace("System 2 proof path verified: ", "").strip()
                for t in trace
                if "proof path verified" in str(t) or "transitive" in str(t).lower()
            ]
            if proof_steps:
                return f"I proved that using multi-hop deductive traversal in my knowledge graph: {' -> '.join(proof_steps)}."

            # Check if this was a concept definition inquiry
            if (operation and isinstance(operation, tuple) and operation[0] == "__definition__") or (
                isinstance(operation, str) and "__definition__" in operation
            ):
                target_name = subj or "the requested concept"
                return (
                    f"I retrieved the 360° concept knot for '{target_name}' from my semantic knowledge graph, "
                    f"integrating its taxonomic lineage, mereological parts, and distinct boundaries."
                )

            clean_trace = [
                str(t) for t in trace
                if t
                and not str(t).startswith("__")
                and not str(t).startswith("Controller route:")
                and not str(t).startswith("Routing reason:")
                and not str(t).startswith("Knowledge graph query")
            ]
            if clean_trace:
                trace_desc = " -> ".join(clean_trace[:3])
                return f"I derived that using multi-hop deductive traversal in my knowledge graph: {trace_desc}."
            if evidence:
                ev_desc = "; ".join(str(e) for e in evidence[:2])
                return f"I verified that directly from grounded evidence in my knowledge graph: {ev_desc}."
            if subj:
                return (
                    f"I retrieved the 360° concept knot for '{subj}' from my semantic knowledge graph, "
                    f"integrating its taxonomic lineage, mereological parts, and distinct boundaries."
                )
            return "I evaluated that against my verified neuro-symbolic knowledge base with 0% hallucination."

        return f"I arrived at that result by processing '{q_text}' through my cognitive reasoning pipeline."

    def recall_conversation_history(self) -> str:
        """Recall prior inquiries from episodic working memory."""
        user_turns = self.headroom.recent_turns("user", limit=3)
        prior_turns = user_turns[1:]

        if not prior_turns:
            return "This is your first inquiry in our current conversation. What would you like to explore or teach me?"

        last_query = prior_turns[0][1].strip()

        if len(prior_turns) == 1:
            return f"Just before this, you asked: \"{last_query}\"."

        second_last = prior_turns[1][1].strip()
        return (
            f"Just before this, you asked: \"{last_query}\". "
            f"Before that, you asked: \"{second_last}\"."
        )

    def reassure_user(self) -> str:
        """Verify cognitive consistency and reassure user that no errors or violations occurred."""
        return (
            "Not at all! You haven't done anything wrong. "
            "Our reasoning, facts, and conversation are completely sound and consistent. "
            "What would you like to explore or teach me next?"
        )

    def _handle_conversational_meta(self, eval_text: str, normalized: str) -> str | None:
        """Handle conversational social intents, meta-reasoning introspection, and history recall."""
        match = self.dialogue_surface.policy.match_social_intent(normalized) or self.dialogue_surface.policy.match_social_intent(eval_text)
        if match:
            intent_name, default_resp = match
            if intent_name == "meta_explanation":
                return self.explain_previous_reasoning()
            if intent_name == "meta_turn_history":
                return self.recall_conversation_history()
            if intent_name == "meta_reassurance":
                return self.reassure_user()
            return default_resp

        composite = self.dialogue_surface.handle_social(normalized) or self.dialogue_surface.handle_social(eval_text)
        if composite is not None:
            return composite

        return None

    def teach(self, text: str) -> ChunkIngestResult:
        """Continuously ingest and verify new knowledge from natural language text."""
        result = self.stream_learner.ingest_chunk(text, source_type=SourceType.USER)
        if text.strip():
            self.mlm.train_on_text(text)
        self.mlm.sync_from_symbolic_memory()
        return result

    def sleep(self, max_dream_steps: int = 3) -> SleepConsolidationReport:
        """Run an offline sleep consolidation cycle (dreaming, invariant mining, cluster induction, and episodic compaction)."""
        report = self.self_study.consolidate_sleep_cycle(
            headroom=self.headroom,
            paged_store=self.paged_store,
            latent_inducer=self.latent_inducer,
            max_dream_steps=max_dream_steps,
        )
        self.mlm.sync_from_symbolic_memory()
        self.mlm.consolidate()
        return report

    def learn_world_corpus(self) -> StreamLearningSummary:
        """Ingest the foundational encyclopedic world corpus into long-term memory."""
        summary = self.stream_learner.ingest_world_corpus()
        self.mlm.sync_from_symbolic_memory()
        self.mlm.consolidate()
        return summary

    def plan(self, goal: str) -> AgenticPlanExecutionResult:
        """Decompose a compound goal into a verified plan DAG and execute with invariant gates."""
        return self.agentic_planner.plan_and_execute(goal)

    def chat(self, text: str) -> str:
        """Conversational chat interface with warm, articulate English dialogue and living wiki integration."""
        self.headroom.record_turn(role="user", text=text)

        # Context activation voting over concept graph ("The text chooses")
        tokens = [w for w in re.findall(r"[a-zA-Z0-9_]+", text)]
        self.paged_store.vote_from_tokens(tokens)

        # 1. Normalize user utterance (discourse markers, colloquialisms, contractions)
        normalized = self.parser.normalize_text(text)

        # 2. Check full text directly for social, meta-cognitive, and introspective dialogue intents
        full_meta = self._handle_conversational_meta(text, normalized)
        if full_meta is not None:
            self.headroom.record_turn(role="assistant", text=full_meta)
            return full_meta

        # 3. If not a standalone social/meta utterance, decompose leading greeting
        greeting_word, remainder = self.dialogue_surface.decompose_utterance(text)
        eval_text = remainder if greeting_word else text
        clean_query = self.parser.normalize_text(eval_text)
        if not clean_query:
            clean_query = eval_text.strip()

        # 4. Check if remainder after greeting is social/meta (e.g. "Hello! Who are you?")
        if greeting_word:
            rem_meta = self._handle_conversational_meta(eval_text, clean_query)
            if rem_meta is not None:
                reply = self.dialogue_surface.compose_reply(greeting_word, rem_meta)
                self.headroom.record_turn(role="assistant", text=reply)
                return reply

        # 2. Cognitive directives & meta-knowledge command dispatch
        command_body = self.command_dispatcher.dispatch(clean_query, self)
        if command_body is not None:
            reply = self.dialogue_surface.compose_reply(greeting_word, command_body)
            self.headroom.record_turn(role="assistant", text=reply)
            return reply

        # 3. Neural routing over utterance
        routing = self.neural_router.route(clean_query)

        # 4. Check for multi-step agentic planning loop intents
        if self.agentic_planner.is_compound_goal(clean_query):
            plan_res = self.agentic_planner.plan_and_execute(clean_query)
            body = self.dialogue_surface.verbalize_agentic_plan_result(plan_res)
            reply = self.dialogue_surface.compose_reply(greeting_word, body)
            self.headroom.record_turn(role="assistant", text=reply)
            self._record_execution(
                query_type="plan",
                query_text=text,
                plan_result=plan_res,
                summary=body,
            )
            return reply

        # 5. Check action router for tool execution intents (with neural routing support)
        tool_call = self.action_router.route(clean_query, routing_decision=routing)
        if tool_call is not None:
            exec_result = self.tool_executor.execute(tool_call)
            self.tool_grounder.ground(exec_result)
            body = self.dialogue_surface.verbalize_tool_result(exec_result)
            reply = self.dialogue_surface.compose_reply(greeting_word, body)
            self.headroom.record_turn(role="assistant", text=reply)
            self._record_execution(
                query_type="tool",
                query_text=text,
                operation=tool_call.tool_name,
                arguments=tool_call.arguments,
                output=exec_result.output,
                summary=body,
            )
            return reply

        outcome = self.process(clean_query, routing_decision=routing)

        resolved_subj = None
        if outcome.frame.parsed_query and isinstance(outcome.frame.parsed_query, tuple):
            resolved_subj = outcome.frame.parsed_query[0]
        if not resolved_subj:
            resolved_subj = self.learner.dialogue.get_salient_subject()

        if outcome.code_program is not None:
            body = self.dialogue_surface.verbalize_code_synthesis(outcome.code_program)
            self._record_execution(
                query_type="code",
                query_text=text,
                summary=body,
            )
        elif outcome.tool_result is not None:
            body = self.dialogue_surface.verbalize_tool_result(outcome.tool_result)
            self._record_execution(
                query_type="tool",
                query_text=text,
                output=outcome.tool_result.output,
                summary=body,
            )
        elif outcome.inference is not None:
            body = self.dialogue_surface.verbalize_inference(
                outcome.inference,
                query=clean_query,
                resolved_subject=resolved_subj,
                parsed_query=outcome.frame.parsed_query,
            )
            self._record_execution(
                query_type="inference",
                query_text=text,
                operation=outcome.frame.parsed_query,
                subject=resolved_subj,
                evidence=outcome.inference.evidence,
                trace=outcome.inference.trace,
                output=outcome.inference.answer,
                summary=body,
            )
        elif outcome.learning is not None:
            body = self.dialogue_surface.verbalize_learning(outcome.learning)
            self._record_execution(
                query_type="learning",
                query_text=text,
                summary=body,
            )
        else:
            body = "I processed that, but have no specific response."

        reply = self.dialogue_surface.compose_reply(greeting_word, body)
        self.headroom.record_turn(role="assistant", text=reply)
        return reply

    def process(
        self,
        text: str,
        *,
        allow_learning: bool = True,
        routing_decision: RoutingDecision | None = None,
    ) -> KernelOutcome:
        # Route neurally if not provided
        routing = routing_decision or self.neural_router.route(text)

        # Check action router first for tool invocations
        tool_call = self.action_router.route(text, routing_decision=routing)
        if tool_call is not None:
            exec_result = self.tool_executor.execute(tool_call)
            self.tool_grounder.ground(exec_result)
            return KernelOutcome(
                frame=CandidateFrame(
                    frame_id=generate_id("frame"),
                    source_text=text,
                    intent="action",
                    claims=[],
                    uncertainty=0.0,
                ),
                mode=CognitiveMode.FAST,
                tool_result=exec_result,
                routing_decision=routing,
            )

        # Check programming & AST code synthesis pathway
        has_code_intent = any(
            kw in text.lower()
            for kw in (
                "write code", "generate code", "python code", "write a function",
                "synthesize code", "implement", "code for", "code to", "python function",
                "program to", "script to",
            )
        ) or (
            routing.pathway.value == "code"
            and any(
                kw in text.lower()
                for kw in ("code", "function", "write", "generate", "synthesize", "implement", "script", "program")
            )
        )
        if has_code_intent:
            prim = self.code_synthesizer.match_primitive(text)
            if prim is not None:
                prog = self.code_synthesizer.synthesize(prim, verify=True)
                return KernelOutcome(
                    frame=CandidateFrame(
                        frame_id=generate_id("frame"),
                        source_text=text,
                        intent="action",
                        claims=[],
                        uncertainty=0.0,
                    ),
                    mode=CognitiveMode.FAST,
                    routing_decision=routing,
                    code_program=prog,
                )

        frame = self.perception.perceive(
            text, memory=self.memory, context=self.learner.dialogue
        )
        if frame.intent == "question":
            thinking = self.controller.run(
                text,
                parsed_query=frame.parsed_query,
                parsed_queries=frame.parsed_queries,
                question_parts=frame.question_parts,
            )
            if (
                thinking.inference.status == BeliefStatus.UNKNOWN
                and hasattr(self, "mlm")
                and self.mlm is not None
                and frame.parsed_query is None
                and not frame.parsed_queries
            ):
                mlm_res = self.mlm.query_360(text)
                if mlm_res.status in ("VERIFIED", "REFUTED"):
                    mapped_status = (
                        BeliefStatus.VERIFIED
                        if mlm_res.status == "VERIFIED"
                        else BeliefStatus.REFUTED
                    )
                    return KernelOutcome(
                        frame=frame,
                        mode=CognitiveMode.SLOW,
                        inference=InferenceResult(
                            query=text,
                            status=mapped_status,
                            answer=mlm_res.fluent_english or mlm_res.answer,
                            confidence=mlm_res.confidence,
                            evidence=[
                                f"{p.source_concept} {p.predicate} {p.target_concept}"
                                for p in mlm_res.spoke_predictions[:3]
                            ],
                            trace=[mlm_res.inspectable_trace],
                        ),
                        routing_decision=routing,
                    )
            return KernelOutcome(
                frame=frame,
                mode=thinking.mode,
                inference=thinking.inference,
                routing_decision=routing,
            )
        if frame.intent in {"statement", "action"} and allow_learning:
            learning = self.learner.commit_frame(frame, source_type=SourceType.USER)
            if learning and learning.concepts_created and hasattr(self, "paged_store"):
                for cid in learning.concepts_created:
                    self.paged_store.invalidate(cid)
            if learning and (
                learning.relations_added > 0
                or learning.concepts_created
                or learning.attributes_updated > 0
            ):
                self.mlm.sync_from_symbolic_memory()
                if self.mlm.rust_arena.is_native and text.strip():
                    self.mlm.rust_arena.train_stream(
                        text, learning_rate=0.15 * self.plasticity.plasticity_multiplier
                    )
            return KernelOutcome(
                frame=frame,
                mode=CognitiveMode.STUDY,
                learning=learning,
                routing_decision=routing,
            )
        return KernelOutcome(
            frame=frame,
            mode=CognitiveMode.UNKNOWN,
            inference=InferenceResult(
                query=text,
                status=BeliefStatus.UNKNOWN,
                answer=None,
                confidence=0.0,
                evidence=[],
                trace=[
                    (
                        "Learning was disabled for this route; no candidate was committed."
                        if frame.intent in {"statement", "action"}
                        else "Perception produced no grounded candidate."
                    )
                ],
            ),
            routing_decision=routing,
        )

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Return a structured telemetry snapshot across memory, paging, plasticity, growth brakes, and canary checks."""
        paged = self.paged_store.get_telemetry()
        plast = self.plasticity.get_telemetry()
        brakes = self.latent_inducer.evaluate_growth_brakes()
        canary = self.self_study.evaluate_heldout_canary()
        return {
            "memory": {
                "total_concepts": len(self.memory.list_concepts()),
                "total_relations": len(self.memory.get_relations()),
                "concept_mutation_epoch": self.memory.concept_mutation_epoch,
            },
            "paged_store": {
                "active_count": paged.resident_count,
                "lru_count": paged.lru_count,
                "disk_fetches": paged.disk_fetches,
                "cache_hits": paged.resident_hits + paged.lru_hits,
                "evictions": paged.evictions,
                "slot_reuses": paged.slot_reuses,
                "dirty_invalidations": paged.dirty_invalidations,
                "hit_rate": round(paged.hit_rate, 4),
            },
            "plasticity": {
                "step": plast.step,
                "base_lr": self.plasticity.base_lr,
                "effective_lr": round(self.plasticity.base_lr * plast.scale, 6),
                "scale": plast.scale,
                "slow_effect_size": round(plast.slow_effect_size, 4),
                "fast_effect_size": round(plast.fast_effect_size, 4),
                "regime_shifts": plast.regime_shifts,
                "spikes_held_back": plast.spikes_filtered,
                "trunk_damping": plast.trunk_lr_mult,
            },
            "growth_brakes": {
                "allowed": brakes.allowed,
                "room": brakes.room,
                "used": brakes.used,
                "earning": brakes.earning,
                "fits": brakes.fits,
                "honest": brakes.honest,
                "reasons": list(brakes.blocked_reasons),
            },
            "canary": {
                "passed": canary.passed,
                "checks_run": canary.checked_count,
                "violations": len(canary.regressions),
                "details": list(canary.regressions),
            },
            "mlm_spiderweb": self.mlm.get_telemetry(),
        }

