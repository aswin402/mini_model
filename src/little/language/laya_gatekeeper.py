"""Non-Autoregressive System 1 Perception and Decision Gatekeeper.

Implements single-forward-pass typed evaluation (choice, noul, score)
with calibrated probabilities and normalized Shannon entropy gating (H >= 0.35 -> UNKNOWN).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


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
    probabilities: Dict[str, float]
    raw_entropy: float
    normalized_entropy: float
    confidence: float
    status: BeliefStatus


class LayaSystem1Gatekeeper:
    """Non-autoregressive System 1 perceptual gatekeeper and calibrated decision engine.

    Provides single-forward pass intent routing, symbol grounding, and epistemic gating.
    """

    def __init__(self, entropy_unknown_threshold: float = 0.35) -> None:
        self.entropy_unknown_threshold = entropy_unknown_threshold
        self.temperature_buckets = {
            "choice:2": 1.12,
            "choice:3-5": 1.28,
            "choice:6-10": 1.45,
            "noul": 1.08,
            "score": 1.20,
        }

    def _compute_entropy(self, probs: List[float]) -> Tuple[float, float]:
        """Calculates raw Shannon entropy and normalized entropy H / ln(K) in [0, 1]."""
        k = len(probs)
        if k <= 1:
            return 0.0, 0.0
        h_raw = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        h_norm = h_raw / math.log(k)
        return h_raw, min(1.0, max(0.0, h_norm))

    def evaluate_choice(
        self, state: str, question: str, options: List[str]
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

        # Semantic concept associations for zero-shot grounding
        concept_priors: Dict[str, Set[str]] = {
            "apple": {"fruit", "red", "gala", "tree", "sweet", "crisp", "pome"},
            "dog": {"animal", "bark", "pet", "puppy", "canine"},
            "bicycle": {"wheels", "ride", "pedal", "cycle", "bike"},
            "galaxy": {"stars", "space", "milky", "universe", "nebula", "cosmos"},
        }

        scores = []
        for opt in options:
            opt_lower = opt.lower()
            opt_words = [w for w in re.findall(r"\w+", opt_lower) if len(w) > 2]
            score = 0.1  # baseline prior
            for w in opt_words:
                if w in lower_state:
                    score += 4.0
                if w in lower_q:
                    score += 0.5
            # Check concept associations
            for c_name, c_words in concept_priors.items():
                if c_name in opt_lower:
                    overlap = len(state_words & c_words)
                    score += overlap * 2.5
            scores.append(score)

        # Softmax with temperature scaling
        max_s = max(scores)
        temp = self.temperature_buckets.get("choice:3-5", 1.28)
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
        elif probs[best_idx] >= 0.70:
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

        # Deterministic Math intent check
        math_markers = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "square of",
            "cube of",
            "calculate",
            "factorial",
            "solve",
        ]
        if any(m in lower for m in math_markers) and any(
            c.isdigit() for c in text
        ):
            return QueryIntent.MATH

        # Curiosity / Meta inquiry check
        curiosity_markers = [
            "uncertain",
            "curiosity",
            "what don't you know",
            "what do you wonder",
            "self-reflect",
        ]
        if any(m in lower for m in curiosity_markers):
            return QueryIntent.CURIOSITY

        # Action command check (imperative verbs)
        action_prefixes = [
            "slice",
            "peel",
            "juice",
            "dehydrate",
            "cut",
            "transform",
            "heat",
            "cool",
            "execute",
        ]
        if any(lower.startswith(p) for p in action_prefixes):
            return QueryIntent.ACTION

        # Question check
        question_starters = [
            "is ",
            "are ",
            "what ",
            "where ",
            "who ",
            "which ",
            "why ",
            "how ",
            "can ",
            "does ",
        ]
        if text.endswith("?") or any(
            lower.startswith(q) for q in question_starters
        ):
            return QueryIntent.QUESTION

        return QueryIntent.STATEMENT

    def verify_proposition(
        self, proposition: str, evidence_context: str = ""
    ) -> CalibratedDecision:
        """Evaluates proposition truth probability via binary 'noul' primitive."""
        if not evidence_context.strip():
            # Zero context evidence -> Epistemic uncertainty
            probs = [0.5, 0.5]
            h_raw, h_norm = self._compute_entropy(probs)
            return CalibratedDecision(
                winner="unknown",
                probabilities={"false": 0.5, "true": 0.5},
                raw_entropy=h_raw,
                normalized_entropy=h_norm,
                confidence=0.0,
                status=BeliefStatus.UNKNOWN,
            )

        lower_prop = proposition.lower()
        lower_ctx = evidence_context.lower()

        prop_words = [w for w in re.findall(r"\w+", lower_prop) if len(w) > 2]
        matches = sum(1 for w in prop_words if w in lower_ctx)
        ratio = matches / max(1, len(prop_words))

        p_true = min(0.99, max(0.01, 0.1 + 0.88 * ratio))
        p_false = 1.0 - p_true
        probs = [p_false, p_true]
        h_raw, h_norm = self._compute_entropy(probs)
        confidence = 2.0 * abs(p_true - 0.5)

        if h_norm >= self.entropy_unknown_threshold:
            status = BeliefStatus.UNKNOWN
        elif p_true >= 0.80:
            status = BeliefStatus.SUPPORTED
        elif p_true <= 0.20:
            status = BeliefStatus.CONTRADICTED
        else:
            status = BeliefStatus.UNCERTAIN

        return CalibratedDecision(
            winner="true" if p_true >= 0.5 else "false",
            probabilities={"false": p_false, "true": p_true},
            raw_entropy=h_raw,
            normalized_entropy=h_norm,
            confidence=confidence,
            status=status,
        )
