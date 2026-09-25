"""Unified entrypoint for perception, learning, and question answering."""

from __future__ import annotations

from dataclasses import dataclass

from little.core.contracts import CandidateFrame, SourceType
from little.core.models import BeliefStatus, InferenceResult, LearningResult
from little.core.runtime_policy import RuntimePolicy
from little.inference.thinking_controller import CognitiveMode, ThinkingController
from little.language.parser import LearningEngine, configured_parser
from little.language.perception import DeterministicPerceptionAdapter, PerceptionAdapter
from little.memory.store import MemoryStore
from little.procedural.math_cas import UnitConversionGraph


@dataclass(frozen=True)
class KernelOutcome:
    frame: CandidateFrame
    mode: CognitiveMode
    learning: LearningResult | None = None
    inference: InferenceResult | None = None


class CognitiveKernel:
    """Route one perception frame to either learning or inference."""

    def __init__(
        self,
        memory: MemoryStore,
        perception: PerceptionAdapter | None = None,
        learner: LearningEngine | None = None,
        controller: ThinkingController | None = None,
        runtime_policy: RuntimePolicy | None = None,
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

    def process(self, text: str, *, allow_learning: bool = True) -> KernelOutcome:
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
                frame=frame, mode=thinking.mode, inference=thinking.inference
            )
        if frame.intent in {"statement", "action"} and allow_learning:
            learning = self.learner.commit_frame(frame, source_type=SourceType.USER)
            return KernelOutcome(
                frame=frame, mode=CognitiveMode.STUDY, learning=learning
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
        )
