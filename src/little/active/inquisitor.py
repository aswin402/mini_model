"""Active Learning Inquisitor for autonomous epistemic uncertainty resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from little.core.models import BeliefStatus, InferenceResult, LearningResult

if TYPE_CHECKING:
    from little.language.parser import LearningEngine
    from little.memory.store import MemoryStore


@dataclass
class ClarificationPrompt:
    """A generated clarification question targeting an epistemic gap."""

    original_query: str
    missing_concept: str | None
    question_for_user: str
    pending_subject: str
    pending_predicate: str
    pending_object: str
    expected_type: str  # "boolean", "text"


class ActiveInquisitor:
    """Monitors inference uncertainty and generates targeted questions to reduce epistemic entropy."""

    def __init__(self, memory: MemoryStore, learning_engine: LearningEngine) -> None:
        self.memory = memory
        self.learning_engine = learning_engine

    def inspect_uncertainty(
        self,
        inference_result: InferenceResult,
        subject: str,
        predicate: str,
        target: str | None,
    ) -> ClarificationPrompt | None:
        """If the inference result is UNKNOWN or ambiguous, generate a clarification prompt."""
        if inference_result.status != BeliefStatus.UNKNOWN:
            return None

        clean_subj = subject.strip().lower()
        clean_pred = predicate.strip().lower() if predicate else "is_a"
        clean_target = str(target).strip().lower() if target is not None else ""

        subj_concept = self.memory.get_concept(clean_subj)

        # Case 0: Concept definition query on unknown subject (e.g. "What is a mango?")
        if not clean_target:
            if not subj_concept:
                return ClarificationPrompt(
                    original_query=inference_result.query,
                    missing_concept=clean_subj,
                    question_for_user=f"I have no memory of '{clean_subj}'. What category or type of thing is a {clean_subj}?",
                    pending_subject=clean_subj,
                    pending_predicate="is_a",
                    pending_object="",
                    expected_type="text",
                )
            return None

        target_concept = self.memory.get_concept(clean_target)

        # Case 1: Subject concept completely missing
        if not subj_concept:
            return ClarificationPrompt(
                original_query=inference_result.query,
                missing_concept=clean_subj,
                question_for_user=f"I do not know what '{clean_subj}' is. Is '{clean_subj}' a '{clean_target}'?",
                pending_subject=clean_subj,
                pending_predicate=clean_pred,
                pending_object=clean_target,
                expected_type="boolean",
            )

        # Case 2: Target concept completely missing
        if not target_concept:
            return ClarificationPrompt(
                original_query=inference_result.query,
                missing_concept=clean_target,
                question_for_user=f"I am unfamiliar with '{clean_target}'. Can '{clean_subj}' be considered a '{clean_target}'?",
                pending_subject=clean_subj,
                pending_predicate=clean_pred,
                pending_object=clean_target,
                expected_type="boolean",
            )

        # Case 3: Both concepts known, but relationship is unobserved/unknown
        return ClarificationPrompt(
            original_query=inference_result.query,
            missing_concept=None,
            question_for_user=f"I know '{clean_subj}' and '{clean_target}', but I don't know if {clean_subj} {clean_pred} {clean_target}. Is that true?",
            pending_subject=clean_subj,
            pending_predicate=clean_pred,
            pending_object=clean_target,
            expected_type="boolean",
        )

    def resolve_response(
        self, prompt: ClarificationPrompt, user_response: str
    ) -> LearningResult:
        """Parse user response, incorporate into persistent memory, and return LearningResult."""
        clean_resp = user_response.strip().lower()

        # If pending_object is empty (we asked for the category of pending_subject)
        if not prompt.pending_object:
            from little.language.parser import SimpleParser

            cat = SimpleParser.clean_noun(clean_resp)
            if cat:
                statement = f"A {prompt.pending_subject} is a {cat}."
                return self.learning_engine.learn(statement)

        # Check positive confirmation
        positive_affirmations = {"yes", "true", "correct", "indeed", "yep", "sure", "y"}
        negative_denials = {"no", "false", "incorrect", "nope", "never", "not", "n"}

        if any(clean_resp.startswith(p) for p in positive_affirmations):
            # Assert positive statement
            statement = f"{prompt.pending_subject} {prompt.pending_predicate} {prompt.pending_object}."
            return self.learning_engine.learn(statement)
        elif any(clean_resp.startswith(n) for n in negative_denials):
            # Assert negative / disjoint statement
            statement = f"{prompt.pending_subject} is not a {prompt.pending_object}."
            return self.learning_engine.learn(statement)
        else:
            # Assume user gave descriptive text
            return self.learning_engine.learn(user_response)

    @classmethod
    def calculate_information_gain(
        cls, prior_confidence: float, posterior_confidence: float
    ) -> float:
        """Calculate Shannon information gain / entropy reduction in bits.

        Under complete ignorance/UNKNOWN, prior is equiprobable (p = 0.5, H = 1.0 bit).
        Confidence c shifts posterior probability towards certainty: p = 0.5 + 0.5 * c.
        """
        def binary_entropy(p: float) -> float:
            p = max(1e-6, min(1.0 - 1e-6, p))
            return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

        p_prior = 0.5 if prior_confidence <= 0.1 else (0.5 + 0.5 * prior_confidence)
        p_post = 0.5 + 0.5 * posterior_confidence

        h_prior = binary_entropy(p_prior)
        h_post = binary_entropy(p_post)
        return max(0.0, round(h_prior - h_post, 4))

