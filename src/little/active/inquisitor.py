"""Active Learning Inquisitor for autonomous epistemic uncertainty resolution."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from little.core.models import BeliefStatus, InferenceResult, LearningResult, UpdateType
from little.active.active_policy import ActiveLearningPolicy
from little.knowledge.policy import LanguagePolicy

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

    def __init__(
        self,
        memory: MemoryStore,
        learning_engine: LearningEngine,
        policy: LanguagePolicy | None = None,
        *,
        active_policy: ActiveLearningPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.learning_engine = learning_engine
        self.policy = policy or LanguagePolicy.default()
        self.active_policy = active_policy or ActiveLearningPolicy.default()

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
        clean_pred = (
            predicate.strip().lower()
            if predicate
            else self.active_policy.default_predicate
        )
        clean_target = str(target).strip().lower() if target is not None else ""

        from little.language.parser import SimpleParser

        # Guard: never ask curiosity questions about pronouns, conversational phrases, wh-words, or meta-entities
        meta_words = self.active_policy.meta_words
        if clean_subj in meta_words or any(w in meta_words for w in clean_subj.split()):
            return None
        if clean_target and (
            clean_target in meta_words
            or any(w in meta_words for w in clean_target.split())
        ):
            return None
        if not SimpleParser.is_valid_concept(clean_subj):
            return None
        if clean_target and not SimpleParser.is_valid_concept(clean_target):
            return None

        subj_concept = self.memory.get_concept(clean_subj)

        # Case 0: Concept definition query on unknown subject (e.g. "What is a mango?")
        if not clean_target:
            if not subj_concept:
                return ClarificationPrompt(
                    original_query=inference_result.query,
                    missing_concept=clean_subj,
                    question_for_user=self.active_policy.prompt_templates[
                        "definition_unknown"
                    ].format(subject=clean_subj, target=clean_target, predicate=clean_pred),
                    pending_subject=clean_subj,
                    pending_predicate=clean_pred,
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
                question_for_user=self.active_policy.prompt_templates[
                    "subject_unknown"
                ].format(subject=clean_subj, target=clean_target, predicate=clean_pred),
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
                question_for_user=self.active_policy.prompt_templates[
                    "target_unknown"
                ].format(subject=clean_subj, target=clean_target, predicate=clean_pred),
                pending_subject=clean_subj,
                pending_predicate=clean_pred,
                pending_object=clean_target,
                expected_type="boolean",
            )

        # Case 3: Both concepts known, but relationship is unobserved/unknown
        return ClarificationPrompt(
            original_query=inference_result.query,
            missing_concept=None,
            question_for_user=self.active_policy.prompt_templates[
                "known_relation"
            ].format(subject=clean_subj, target=clean_target, predicate=clean_pred),
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

        # Check if user cancelled or wants to skip
        cancel_words = self.policy.cancel_words
        if clean_resp in cancel_words or any(
            clean_resp.startswith(c) for c in cancel_words
        ):
            exp = self.memory.add_experience(
                input_text=user_response, extracted_triples=[]
            )
            return LearningResult(
                input_text=user_response,
                update_type=UpdateType.NO_OP,
                experience_id=exp.id,
                message="Curiosity query skipped by user.",
            )

        # Check if user asked a question instead of answering
        if "?" in user_response or any(
            clean_resp.startswith(q)
            for q in self.policy.question_prefixes
        ):
            exp = self.memory.add_experience(
                input_text=user_response, extracted_triples=[]
            )
            return LearningResult(
                input_text=user_response,
                update_type=UpdateType.NO_OP,
                experience_id=exp.id,
                message="User asked a new question.",
            )

        statement = self.candidate_statement(prompt, user_response)
        if statement is None:
            exp = self.memory.add_experience(
                input_text=user_response, extracted_triples=[]
            )
            return LearningResult(
                input_text=user_response,
                update_type=UpdateType.NO_OP,
                experience_id=exp.id,
                message="Provided response was not a valid concept name.",
            )
        return self.learning_engine.learn(statement)

    def candidate_statement(
        self, prompt: ClarificationPrompt, user_response: str
    ) -> str | None:
        """Return kernel-ready clarification text without changing memory or dialogue."""
        clean_response = user_response.strip().lower()
        if clean_response in self.policy.cancel_words:
            return None
        if "?" in user_response or any(
            clean_response.startswith(prefix)
            for prefix in self.policy.question_prefixes
        ):
            return None

        if prompt.pending_object:
            if any(
                clean_response.startswith(word)
                for word in self.policy.positive_affirmations
            ):
                return (
                    f"{prompt.pending_subject} {prompt.pending_predicate} "
                    f"{prompt.pending_object}."
                )
            if any(
                clean_response.startswith(word)
                for word in self.policy.negative_denials
            ):
                return f"{prompt.pending_subject} is not a {prompt.pending_object}."
            return user_response

        candidate = clean_response
        match = re.search(
            r"(?:is\s+(?:a|an)\s+|kind\s+of\s+|type\s+of\s+)(.+)",
            candidate,
        )
        if match:
            candidate = match.group(1).strip()
        from little.language.parser import SimpleParser

        candidate = SimpleParser.clean_noun(candidate)
        if not candidate or not SimpleParser.is_valid_concept(candidate):
            return None
        return f"A {prompt.pending_subject} is a {candidate}."

    def calculate_information_gain(
        self, prior_confidence: float, posterior_confidence: float
    ) -> float:
        """Calculate Shannon information gain / entropy reduction in bits.

        Under complete ignorance/UNKNOWN, prior is equiprobable (p = 0.5, H = 1.0 bit).
        Confidence c shifts posterior probability towards certainty: p = 0.5 + 0.5 * c.
        """

        def binary_entropy(p: float) -> float:
            floor = self.active_policy.entropy_probability_floor
            p = max(floor, min(1.0 - floor, p))
            return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

        prior = self.active_policy.binary_prior
        scale = self.active_policy.confidence_scale
        p_prior = (
            prior
            if prior_confidence <= self.active_policy.unknown_confidence_threshold
            else (prior + scale * prior_confidence)
        )
        p_post = prior + scale * posterior_confidence

        h_prior = binary_entropy(p_prior)
        h_post = binary_entropy(p_post)
        return max(0.0, round(h_prior - h_post, self.active_policy.round_digits))
