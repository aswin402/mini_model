"""Non-Autoregressive System 1 Decision Head & Calibrated Proper Scoring (Laya Architecture).

Implements single-forward-pass classification, typed structured decisions
(`choice`, `noul`, `score`), and epistemic uncertainty gating based on
Shannon entropy and strictly proper scoring rules (Brier score & RPS).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from little.memory.store import MemoryStore
from little.neural.encoder import NeuralStateEncoder
from little.neural.models import NeuralState


class EpistemicBeliefState(str, Enum):
    """Calibrated truth state for non-autoregressive truth-value evaluations."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ChoiceDecision:
    """A categorical decision over K candidate choices with calibrated probabilities."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    entropy: float
    normalized_entropy: float
    margin: float
    brier_score: float | None = None
    is_fast_path: bool = False
    latent_state: NeuralState | None = None


@dataclass(frozen=True)
class NoulDecision:
    """Non-autoregressive truth-value unit evaluating proposition validity without token generation."""

    truth_probability: float
    confidence: float
    status: EpistemicBeliefState
    entropy: float
    is_fast_path: bool = False
    latent_state: NeuralState | None = None


@dataclass(frozen=True)
class ScoreDecision:
    """Ordinal decision predicting a calibrated discrete rating or intensity scale."""

    expected_score: float
    discrete_choice: int
    probabilities: list[float]
    confidence: float
    ranked_probability_score: float | None = None
    latent_state: NeuralState | None = None


def _validate_distribution(probabilities: Sequence[float]) -> None:
    if not probabilities or any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 <= p <= 1
        for p in probabilities
    ):
        raise ValueError(
            "Probabilities must be nonempty finite values between zero and one"
        )
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-3):
        raise ValueError("Probabilities must sum to one")


def compute_brier_score(
    probabilities: dict[str, float] | Sequence[float], target: str | int
) -> float:
    """Compute the strictly proper Brier score for a categorical probability distribution.

    BS = (1/K) * sum_{k=1}^K (p_k - y_k)^2, where y_k is 1 for target, else 0.
    Lower is better (0.0 is perfect, 1.0 is maximum error).
    """
    if isinstance(probabilities, dict):
        keys = list(probabilities.keys())
        p_vals = [probabilities[k] for k in keys]
        if target not in keys:
            raise ValueError("Unknown target label")
        target_idx = keys.index(target)
    else:
        p_vals = list(probabilities)
        if type(target) is not int or not 0 <= target < len(p_vals):
            raise ValueError("Target index is outside the distribution")
        target_idx = target

    _validate_distribution(p_vals)
    k = len(p_vals)
    sum_sq = 0.0
    for idx, p in enumerate(p_vals):
        y = 1.0 if idx == target_idx else 0.0
        sum_sq += (p - y) ** 2

    return round(sum_sq / k, 6)


def compute_shannon_entropy(probabilities: Sequence[float]) -> float:
    """Compute the Shannon entropy: H(p) = -sum(p_k * ln(p_k))."""
    h = 0.0
    for p in probabilities:
        if p > 1e-12:
            h -= p * math.log(p)
    return round(h, 6)


def compute_normalized_entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy normalized by ln(K) into [0.0, 1.0]."""
    _validate_distribution(probabilities)
    k = len(probabilities)
    if k <= 1:
        return 0.0
    h = compute_shannon_entropy(probabilities)
    max_h = math.log(k)
    return round(min(1.0, max(0.0, h / max_h)), 6)


def compute_ranked_probability_score(
    probabilities: Sequence[float], target_idx: int
) -> float:
    """Compute the Ranked Probability Score (RPS) for ordinal scales."""
    if type(target_idx) is not int or not 0 <= target_idx < len(probabilities):
        raise ValueError("Target index is outside the distribution")
    _validate_distribution(probabilities)
    k = len(probabilities)
    if k <= 1:
        return 0.0

    cum_p = 0.0
    cum_y = 0.0
    rps_sum = 0.0

    for i in range(k - 1):
        cum_p += probabilities[i]
        cum_y += 1.0 if i >= target_idx else 0.0
        rps_sum += (cum_p - cum_y) ** 2

    return round(rps_sum / (k - 1), 6)


class LayaDecisionHead:
    """High-speed Non-Autoregressive System 1 Perception and Decision Evaluator.

    Inspired by Laya (ModernBERT-based parallel decision architecture) and Jev.
    Classifies intent with a linear projection over hashed encoder features; grounds truth symbolically
    without autoregressive token generation.
    """

    DEFAULT_INTENTS: Sequence[str] = (
        "statement",
        "question",
        "action",
        "math",
        "curiosity",
        "social",
        "physics",
        "code",
    )

    def __init__(
        self,
        encoder: NeuralStateEncoder | None = None,
        memory: MemoryStore | None = None,
        temperature: float = 1.0,
        entropy_threshold: float = 0.35,
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.20,
    ) -> None:
        self.encoder = encoder or NeuralStateEncoder()
        self.memory = memory
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Temperature must be finite and positive")
        self.temperature = temperature
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        # Linear intent weights; truth evaluation uses the injected symbolic memory.
        self._intent_labels = list(self.DEFAULT_INTENTS)
        self._w, self._b = self._init_intent_weights()

    def classify_intent(self, text_or_state: str | NeuralState) -> ChoiceDecision:
        """Classify input intent across primary cognitive pathways in a single forward pass."""
        state = self._resolve_state(text_or_state)
        S = state.embedding
        in_dim = len(S)

        # z_k = sum_i (W_{k,i} * S_i) + b_k
        logits = [0.0] * len(self._intent_labels)
        for k in range(len(self._intent_labels)):
            dot = sum(S[i] * self._w[k][i] for i in range(in_dim))
            logits[k] = (dot + self._b[k]) / self.temperature

        # Numerically stable softmax
        max_z = max(logits)
        exps = [math.exp(z - max_z) for z in logits]
        sum_exp = sum(exps)
        probs = [e / sum_exp for e in exps]

        prob_dict = {
            self._intent_labels[idx]: round(probs[idx], 4)
            for idx in range(len(self._intent_labels))
        }

        entropy = compute_shannon_entropy(probs)
        norm_entropy = compute_normalized_entropy(probs)

        ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_conf = ranked[0]
        second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = round(top_conf - second_conf, 4)

        # Fast path is triggered when decision has low normalized entropy and distinct margin
        is_fast_path = (norm_entropy < self.entropy_threshold) and (
            margin >= self.margin_threshold
        )

        return ChoiceDecision(
            choice=top_intent,
            probabilities=prob_dict,
            confidence=top_conf,
            entropy=entropy,
            normalized_entropy=norm_entropy,
            margin=margin,
            is_fast_path=is_fast_path,
            latent_state=state,
        )

    def evaluate_choice(
        self,
        text_or_state: str | NeuralState,
        options: Sequence[str],
        *,
        temperature: float | None = None,
    ) -> ChoiceDecision:
        """Evaluate an arbitrary set of candidate options non-autoregressively."""
        if (
            not options
            or len(set(options)) != len(options)
            or any(
                not isinstance(option, str) or not option.strip() for option in options
            )
        ):
            raise ValueError("Candidate options must be nonempty, unique strings.")

        state = self._resolve_state(text_or_state)
        temp = self.temperature if temperature is None else temperature
        if (
            isinstance(temp, bool)
            or not isinstance(temp, (int, float))
            or not math.isfinite(temp)
            or temp <= 0
        ):
            raise ValueError("Temperature must be finite and positive")
        in_dim = len(state.embedding)

        # Project candidate options into normalized semantic embedding vectors
        opt_prototypes: list[tuple[float, ...]] = [
            self.encoder.encode(opt).embedding for opt in options
        ]

        # Dot product logits: S . P_k * (5.0 / temp)
        scale = 5.0 / temp
        logits = [
            sum(state.embedding[i] * opt_prototypes[k][i] for i in range(in_dim))
            * scale
            for k in range(len(options))
        ]

        # Check token / semantic grounding with memory
        prompt_text = (
            text_or_state if isinstance(text_or_state, str) else text_or_state.text
        )
        prompt_tokens = set(re.findall(r"[a-z0-9_]+", prompt_text.lower()))

        grounded_targets: set[str] = set()
        if self.memory:
            for tok in prompt_tokens:
                c = self.memory.get_concept(tok)
                if c:
                    grounded_targets.add(c.name.lower())
                    rels = self.memory.get_relations(subject_id=c.id)
                    for r in rels:
                        obj = self.memory.get_concept(r.object_id)
                        if obj:
                            grounded_targets.add(obj.name.lower())

        for k, opt in enumerate(options):
            opt_lower = opt.lower()
            opt_words = set(re.findall(r"[a-z0-9_]+", opt_lower))
            direct_overlap = len(opt_words & prompt_tokens)
            if direct_overlap:
                logits[k] += direct_overlap * 2.0
            mem_overlap = len(opt_words & grounded_targets)
            if mem_overlap:
                logits[k] += mem_overlap * 3.5

        max_z = max(logits)
        exps = [math.exp(z - max_z) for z in logits]
        sum_exp = sum(exps)
        probs = [e / sum_exp for e in exps]

        prob_dict = {options[idx]: round(probs[idx], 4) for idx in range(len(options))}
        if prob_dict:
            top_opt = max(prob_dict, key=lambda k: prob_dict[k])
            residual = round(1.0 - sum(prob_dict.values()), 4)
            if residual != 0.0:
                prob_dict[top_opt] = round(max(0.0, min(1.0, prob_dict[top_opt] + residual)), 4)

        entropy = compute_shannon_entropy(probs)
        norm_entropy = compute_normalized_entropy(probs)

        ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_choice, top_conf = ranked[0]
        second_conf = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = round(top_conf - second_conf, 4)

        is_fast = (norm_entropy < self.entropy_threshold) and (
            margin >= self.margin_threshold
        )

        return ChoiceDecision(
            choice=top_choice,
            probabilities=prob_dict,
            confidence=top_conf,
            entropy=entropy,
            normalized_entropy=norm_entropy,
            margin=margin,
            is_fast_path=is_fast,
            latent_state=state,
        )

    def evaluate_noul(
        self,
        text_or_state: str | NeuralState,
        statement: str | None = None,
    ) -> NoulDecision:
        """Evaluate a binary truth proposition (truth-value unit) with calibrated bounds.

        Maps subjective probability of truth into SUPPORTED, CONTRADICTED, UNCERTAIN, or UNKNOWN.
        """
        prompt = statement or (
            text_or_state if isinstance(text_or_state, str) else text_or_state.text
        )
        state = self._resolve_state(prompt)

        from little.core.models import BeliefStatus
        from little.inference.engine import InferenceEngine
        from little.language.parser import SimpleParser

        p_true, c_star = 0.5, 0.0
        status = EpistemicBeliefState.UNKNOWN
        is_fast = False
        triples = SimpleParser.parse_statement(prompt)
        if self.memory is not None and len(triples) == 1:
            triple = triples[0]
            result = InferenceEngine(self.memory).infer(
                triple.subject, triple.predicate, triple.object_
            )
            if result.status in (BeliefStatus.SUPPORTED, BeliefStatus.REFUTED):
                supported = result.status == BeliefStatus.SUPPORTED
                if triple.is_negative:
                    supported = not supported
                c_star = result.confidence
                p_true = 0.5 + (0.5 * c_star if supported else -0.5 * c_star)
                status = (
                    EpistemicBeliefState.SUPPORTED
                    if supported
                    else EpistemicBeliefState.CONTRADICTED
                )
                is_fast = c_star >= self.confidence_threshold
            elif result.status in (BeliefStatus.AMBIGUOUS, BeliefStatus.CONTRADICTED):
                status = EpistemicBeliefState.UNCERTAIN

        probs = [round(1.0 - p_true, 4), p_true]
        entropy = compute_shannon_entropy(probs)

        return NoulDecision(
            truth_probability=round(p_true, 4),
            confidence=c_star,
            status=status,
            entropy=entropy,
            is_fast_path=is_fast,
            latent_state=state,
        )

    def evaluate_score(
        self,
        text_or_state: str | NeuralState,
        levels: Sequence[str] | int = 5,
    ) -> ScoreDecision:
        """Evaluate an ordinal scale (e.g. priority 0 to 4, or rating levels)."""
        state = self._resolve_state(text_or_state)

        if isinstance(levels, int):
            level_labels = [f"Level {i}" for i in range(levels)]
        else:
            level_labels = list(levels)

        choice = self.evaluate_choice(state, level_labels)
        probs = [choice.probabilities[lbl] for lbl in level_labels]

        # Expected score: sum_i (i * p_i)
        expected = sum(i * p for i, p in enumerate(probs))
        discrete_choice = max(range(len(probs)), key=lambda i: probs[i])

        return ScoreDecision(
            expected_score=round(expected, 4),
            discrete_choice=discrete_choice,
            probabilities=probs,
            confidence=choice.confidence,
            latent_state=state,
        )

    def learn(
        self,
        state: NeuralState,
        target_intent: str,
        learning_rate: float = 0.05,
    ) -> float:
        """Update linear weights with a softmax cross-entropy gradient; return Brier diagnostic."""
        if target_intent not in self._intent_labels:
            raise ValueError("Unknown target intent")

        target_idx = self._intent_labels.index(target_intent)
        decision = self.classify_intent(state)
        current_probs = [
            decision.probabilities.get(lbl, 0.0) for lbl in self._intent_labels
        ]

        # Proper scoring loss (Brier score)
        bs = compute_brier_score(current_probs, target_idx)

        # Gradient step on softmax cross-entropy (Brier above is diagnostic)
        for k in range(len(self._intent_labels)):
            y_k = 1.0 if k == target_idx else 0.0
            error = y_k - current_probs[k]
            grad_scale = learning_rate * error / self.temperature
            for i in range(len(state.embedding)):
                self._w[k][i] += grad_scale * state.embedding[i]
            self._b[k] += grad_scale

        return bs

    def is_fast_path(self, decision: ChoiceDecision) -> bool:
        """Determine if decision satisfies the fast-path threshold."""
        return decision.is_fast_path

    def should_escalate_to_system2(self, decision: ChoiceDecision) -> bool:
        """Determine if decision has high epistemic uncertainty and should escalate to System 2."""
        return (
            (decision.normalized_entropy >= self.entropy_threshold)
            or (decision.margin < self.margin_threshold)
            or (decision.confidence < self.confidence_threshold)
        )

    def _resolve_state(self, text_or_state: str | NeuralState) -> NeuralState:
        if isinstance(text_or_state, NeuralState):
            return text_or_state
        return self.encoder.encode(text_or_state)

    def _init_intent_weights(self) -> tuple[list[list[float]], list[float]]:
        in_dim = self.encoder.dim
        num_classes = len(self._intent_labels)

        # Dynamic mapping of cognitive intents to policy pathways
        from little.neural.models import Pathway

        pathway_map = {
            "math": Pathway.MATH,
            "action": Pathway.TOOL,
            "code": Pathway.CODE,
            "physics": Pathway.PHYSICS,
            "social": Pathway.CHAT,
        }

        # Structural linguistic and curiosity markers
        structural_markers: dict[str, list[str]] = {
            "statement": [
                "is",
                "are",
                "has",
                "have",
                "fact",
                "record",
                "named",
                "known",
                "called",
            ],
            "question": ["what", "who", "where", "when", "why", "how", "which", "?"],
            "curiosity": [
                "gap",
                "gaps",
                "dream",
                "study",
                "explore",
                "learn",
                "hypothesize",
                "curious",
                "knowledge",
                "epistemic",
            ],
            "social": [
                "hello",
                "hi",
                "hey",
                "thanks",
                "thank",
                "welcome",
                "bye",
                "goodbye",
            ],
        }

        w: list[list[float]] = []
        b: list[float] = [0.0] * num_classes

        for lbl in self._intent_labels:
            vec = [0.0] * in_dim
            # 1. Load dynamic keywords from policy semantic clusters if available
            pw = pathway_map.get(lbl)
            if pw and pw in self.encoder.policy.semantic_clusters:
                for kw in self.encoder.policy.semantic_clusters[pw]:
                    self.encoder._project_string(kw, 3.0, vec)
                self.encoder._project_string(f"pathway:{pw.value}", 8.0, vec)

            # 2. Add structural linguistic markers
            for kw in structural_markers.get(lbl, []):
                self.encoder._project_string(kw, 3.0, vec)

            self.encoder._project_string(f"intent:{lbl}", 5.0, vec)
            if lbl == "math":
                self.encoder._project_string("type:numeric", 5.0, vec)
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-12:
                vec = [x / norm for x in vec]
            w.append([5.0 * v for v in vec])

        return w, b
