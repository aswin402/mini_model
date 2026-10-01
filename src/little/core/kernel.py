"""Unified entrypoint for perception, learning, and question answering."""

from __future__ import annotations

from dataclasses import dataclass
import re

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
from little.active.self_study import AutonomousSelfStudyEngine, DreamReport
from little.core.cognitive_command import CognitiveCommandRegistry
from little.core.contracts import CandidateFrame, SourceType
from little.core.models import BeliefStatus, InferenceResult, LearningResult, generate_id
from little.core.runtime_policy import RuntimePolicy
from little.dynamics.continuous_simulation import PhysicalSimulationEngine
from little.perception.sensory import SensoryGroundingEngine
from little.inference.thinking_controller import CognitiveMode, ThinkingController
from little.knowledge.living_wiki import LivingWiki
from little.language.dialogue_surface import DialogueSurfaceEngine
from little.language.parser import LearningEngine, configured_parser
from little.language.perception import DeterministicPerceptionAdapter, PerceptionAdapter
from little.memory.headroom import HeadroomContextManager
from little.memory.store import MemoryStore
from little.neural import NeuralRouter, RoutingDecision
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
        self.action_policy = action_policy or ActionPolicy.default(workspace_root=workspace_root)
        self.neural_router = neural_router or NeuralRouter()
        self.action_router = ActionRouter(action_policy=self.action_policy, neural_router=self.neural_router)
        self.tool_executor = ToolExecutor(action_policy=self.action_policy)
        self.tool_grounder = ToolGroundingEngine(memory_store=self.memory)
        self.code_synthesizer = ASTCodeSynthesizer()
        self.headroom = HeadroomContextManager(memory=self.memory)
        self.wiki = LivingWiki(memory=self.memory)
        self.crawler = KnowledgeCrawler(memory=self.memory, wiki=self.wiki)
        self.latent_inducer = LatentKnotInducer(memory=self.memory, wiki=self.wiki)
        self.agentic_planner = AgenticPlanner(
            action_router=self.action_router,
            tool_executor=self.tool_executor,
            tool_grounder=self.tool_grounder,
            action_policy=self.action_policy,
            memory=self.memory,
        )
        self.self_study = AutonomousSelfStudyEngine(
            memory=self.memory,
            wiki=self.wiki,
            crawler=self.crawler,
        )
        self.simulation_engine = PhysicalSimulationEngine()
        self.sensory_engine = SensoryGroundingEngine(self.memory)
        self.command_dispatcher = CognitiveCommandRegistry()

    def chat(self, text: str) -> str:
        """Conversational chat interface with warm, articulate English dialogue and living wiki integration."""
        self.headroom.record_turn(role="user", text=text)

        social_reply = self.dialogue_surface.handle_social(text)
        if social_reply is not None:
            self.headroom.record_turn(role="assistant", text=social_reply)
            return social_reply

        greeting_word, remainder = self.dialogue_surface.decompose_utterance(text)
        if greeting_word:
            rem_social = self.dialogue_surface.handle_social(remainder)
            if rem_social is not None:
                reply = f"{greeting_word.capitalize()}! {rem_social}"
                self.headroom.record_turn(role="assistant", text=reply)
                return reply

        clean_rem = remainder.strip()

        # Cognitive directives & meta-knowledge command dispatch
        command_body = self.command_dispatcher.dispatch(clean_rem, self)
        if command_body is not None:
            reply = self.dialogue_surface.compose_reply(greeting_word, command_body)
            self.headroom.record_turn(role="assistant", text=reply)
            return reply

        # Neural routing over utterance
        routing = self.neural_router.route(remainder)

        # Check for multi-step agentic planning loop intents
        if self.agentic_planner.is_compound_goal(remainder):
            plan_res = self.agentic_planner.plan_and_execute(remainder)
            body = self.dialogue_surface.verbalize_agentic_plan_result(plan_res)
            reply = self.dialogue_surface.compose_reply(greeting_word, body)
            self.headroom.record_turn(role="assistant", text=reply)
            return reply

        # Check action router for tool execution intents (with neural routing support)
        tool_call = self.action_router.route(remainder, routing_decision=routing)
        if tool_call is not None:
            exec_result = self.tool_executor.execute(tool_call)
            self.tool_grounder.ground(exec_result)
            body = self.dialogue_surface.verbalize_tool_result(exec_result)
            reply = self.dialogue_surface.compose_reply(greeting_word, body)
            self.headroom.record_turn(role="assistant", text=reply)
            return reply

        outcome = self.process(remainder, routing_decision=routing)

        resolved_subj = None
        if outcome.frame.parsed_query and isinstance(outcome.frame.parsed_query, tuple):
            resolved_subj = outcome.frame.parsed_query[0]
        if not resolved_subj:
            resolved_subj = self.learner.dialogue.get_salient_subject()

        if outcome.code_program is not None:
            body = self.dialogue_surface.verbalize_code_synthesis(outcome.code_program)
        elif outcome.tool_result is not None:
            body = self.dialogue_surface.verbalize_tool_result(outcome.tool_result)
        elif outcome.inference is not None:
            body = self.dialogue_surface.verbalize_inference(
                outcome.inference,
                query=remainder,
                resolved_subject=resolved_subj,
                parsed_query=outcome.frame.parsed_query,
            )
        elif outcome.learning is not None:
            body = self.dialogue_surface.verbalize_learning(outcome.learning)
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
            return KernelOutcome(
                frame=frame,
                mode=thinking.mode,
                inference=thinking.inference,
                routing_decision=routing,
            )
        if frame.intent in {"statement", "action"} and allow_learning:
            learning = self.learner.commit_frame(frame, source_type=SourceType.USER)
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
