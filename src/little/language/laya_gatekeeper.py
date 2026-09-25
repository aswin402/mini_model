"""Non-Autoregressive System 1 Perception and Decision Gatekeeper.

Implements single-forward-pass typed evaluation (choice, noul, score)
with calibrated probabilities and policy-driven normalized entropy gating.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from little.knowledge.policy import LanguagePolicy
from little.language.laya_policy import LayaDecisionPolicy


class QueryIntent(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    ACTION = "action"
    MATH = "math"
    CURIOSITY = "curiosity"


class BeliefStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CalibratedDecision:
    winner: str
    probabilities: dict[str, float]
    raw_entropy: float
    normalized_entropy: float
    confidence: float
    status: BeliefStatus


class LayaSystem1Gatekeeper:
    """Non-autoregressive System 1 perceptual gatekeeper and calibrated decision engine.

    Provides single-forward pass intent routing, symbol grounding, and epistemic gating.
    """

    def __init__(
        self,
        entropy_unknown_threshold: float | None = None,
        policy: LanguagePolicy | None = None,
        decision_policy: LayaDecisionPolicy | None = None,
    ) -> None:
        self.decision_policy = decision_policy or LayaDecisionPolicy.default()
        self.entropy_unknown_threshold = (
            self.decision_policy.entropy_unknown_threshold
            if entropy_unknown_threshold is None
            else entropy_unknown_threshold
        )
        self.policy = policy or LanguagePolicy.default()
        self.temperature_buckets = dict(self.decision_policy.choice.temperature_buckets)

    def _compute_entropy(self, probs: list[float]) -> tuple[float, float]:
        """Calculates raw Shannon entropy and normalized entropy H / ln(K) in [0, 1]."""
        k = len(probs)
        if k <= 1:
            return 0.0, 0.0
        probability_floor = self.decision_policy.entropy_probability_floor
        h_raw = -sum(p * math.log(max(p, probability_floor)) for p in probs)
        h_norm = h_raw / math.log(k)
        return h_raw, min(1.0, max(0.0, h_norm))

    def evaluate_choice(
        self, state: str, question: str, options: list[str]
    ) -> CalibratedDecision:
        """Evaluates choice over K discrete categorical options in single-pass calibrated scoring."""
        k = len(options)
        if k == 0:
            raise ValueError("Options list cannot be empty.")
        if k == 1:
            return CalibratedDecision(
                winner=options[0],
                probabilities={options[0]: 1.0},
                raw_entropy=0.0,
                normalized_entropy=0.0,
                confidence=1.0,
                status=BeliefStatus.SUPPORTED,
            )

        lower_state = state.lower()
        lower_q = question.lower()
        state_words = set(re.findall(r"\w+", lower_state))

        scores = []
        for opt in options:
            opt_lower = opt.lower()
            minimum_token_length = self.decision_policy.choice.minimum_token_length
            opt_words = [
                w
                for w in re.findall(r"\w+", opt_lower)
                if len(w) > minimum_token_length
            ]
            score = self.decision_policy.choice.baseline_prior
            for w in opt_words:
                if w in lower_state:
                    score += self.decision_policy.choice.state_token_match_weight
                if w in lower_q:
                    score += self.decision_policy.choice.question_token_match_weight
            # Check concept associations
            for c_name, c_words in self.policy.concept_associations.items():
                if c_name in opt_lower:
                    overlap = len(state_words & c_words)
                    score += (
                        overlap
                        * self.decision_policy.choice.association_overlap_weight
                    )
            scores.append(score)

        # Softmax with temperature scaling
        max_s = max(scores)
        temp = self.temperature_buckets[
            self.decision_policy.choice.default_temperature_bucket
        ]
        exp_scores = [math.exp((s - max_s) / temp) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]

        prob_dict = {opt: probs[i] for i, opt in enumerate(options)}
        h_raw, h_norm = self._compute_entropy(probs)
        confidence = 1.0 - h_norm

        best_idx = int(max(range(k), key=lambda i: probs[i]))
        winner = options[best_idx]

        # Epistemic Gate: If Normalized Entropy >= threshold, mark UNKNOWN
        if h_norm >= self.entropy_unknown_threshold:
            status = BeliefStatus.UNKNOWN
        elif (
            probs[best_idx]
            >= self.decision_policy.choice.supported_probability_threshold
        ):
            status = BeliefStatus.SUPPORTED
        else:
            status = BeliefStatus.UNCERTAIN

        return CalibratedDecision(
            winner=winner,
            probabilities=prob_dict,
            raw_entropy=h_raw,
            normalized_entropy=h_norm,
            confidence=confidence,
            status=status,
        )

    def classify_intent(self, user_input: str) -> QueryIntent:
        """Fast intent classification without token generation."""
        text = user_input.strip()
        lower = text.lower()

        checks = {
            "math": lambda: any(m in lower for m in self.policy.math_markers)
            and any(c.isdigit() for c in text),
            "curiosity": lambda: any(
                marker in lower for marker in self.policy.curiosity_markers
            ),
            "action": lambda: any(
                lower.startswith(prefix) for prefix in self.policy.action_prefixes
            ),
            "question": lambda: text.endswith("?")
            or any(
                lower.startswith(prefix) for prefix in self.policy.question_prefixes
            ),
            "statement": lambda: True,
        }
        for intent in self.policy.intent_priority:
            if checks[intent]():
                return QueryIntent(intent)

        return QueryIntent.STATEMENT

    def verify_proposition(
        self, proposition: str, evidence_context: str = ""
    ) -> CalibratedDecision:
        """Evaluates proposition truth probability via binary 'noul' primitive."""
        if not evidence_context.strip():
            # Zero context evidence -> Epistemic uncertainty
            neutral_probability = self.decision_policy.proposition.neutral_probability
            probs = [1.0 - neutral_probability, neutral_probability]
            h_raw, h_norm = self._compute_entropy(probs)
            return CalibratedDecision(
                winner="unknown",
                probabilities={
                    "false": 1.0 - neutral_probability,
                    "true": neutral_probability,
                },
                raw_entropy=h_raw,
                normalized_entropy=h_norm,
                confidence=0.0,
                status=BeliefStatus.UNKNOWN,
            )

        lower_prop = proposition.lower()
        lower_ctx = evidence_context.lower()

        minimum_token_length = self.decision_policy.proposition.minimum_token_length
        prop_words = [
            w
            for w in re.findall(r"\w+", lower_prop)
            if len(w) > minimum_token_length
        ]
        matches = sum(1 for w in prop_words if w in lower_ctx)
        ratio = matches / max(1, len(prop_words))

        proposition_policy = self.decision_policy.proposition
        p_true = min(
            proposition_policy.maximum_probability,
            max(
                proposition_policy.minimum_probability,
                proposition_policy.prior_true
                + proposition_policy.match_weight * ratio,
            ),
        )
        p_false = 1.0 - p_true
        probs = [p_false, p_true]
        h_raw, h_norm = self._compute_entropy(probs)
        confidence = proposition_policy.confidence_scale * abs(
            p_true - proposition_policy.neutral_probability
        )

        if h_norm >= self.entropy_unknown_threshold:
            status = BeliefStatus.UNKNOWN
        elif p_true >= proposition_policy.supported_probability_threshold:
            status = BeliefStatus.SUPPORTED
        elif p_true <= proposition_policy.contradicted_probability_threshold:
            status = BeliefStatus.CONTRADICTED
        else:
            status = BeliefStatus.UNCERTAIN

        return CalibratedDecision(
            winner=(
                "true"
                if p_true >= proposition_policy.neutral_probability
                else "false"
            ),
            probabilities={"false": p_false, "true": p_true},
            raw_entropy=h_raw,
            normalized_entropy=h_norm,
            confidence=confidence,
            status=status,
        )
