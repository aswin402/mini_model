"""Unit tests for Laya Non-Autoregressive Decision Head & Calibrated Proper Scoring."""

from __future__ import annotations

import pytest

from little.neural.decision_head import (
    ChoiceDecision,
    EpistemicBeliefState,
    LayaDecisionHead,
    NoulDecision,
    ScoreDecision,
    compute_brier_score,
    compute_normalized_entropy,
    compute_ranked_probability_score,
    compute_shannon_entropy,
)
from little.neural.encoder import NeuralStateEncoder


def test_brier_score_computation():
    """Verify Brier score properties: 0.0 for perfect certainty, positive for errors."""
    # 3 classes: target is class 0
    perfect = [1.0, 0.0, 0.0]
    assert compute_brier_score(perfect, 0) == 0.0

    # Total error: predicted class 1 when target is class 0
    worst = [0.0, 1.0, 0.0]
    # BS = (1/3) * ((0-1)^2 + (1-0)^2 + (0-0)^2) = 2/3 = 0.666667
    assert abs(compute_brier_score(worst, 0) - 0.666667) < 1e-4

    # Dict format
    prob_dict = {"A": 0.8, "B": 0.1, "C": 0.1}
    bs = compute_brier_score(prob_dict, "A")
    assert bs < 0.05


def test_shannon_and_normalized_entropy():
    """Verify Shannon entropy and normalized entropy bounded in [0.0, 1.0]."""
    # Deterministic distribution has 0 entropy
    certain = [1.0, 0.0, 0.0, 0.0]
    assert compute_shannon_entropy(certain) == 0.0
    assert compute_normalized_entropy(certain) == 0.0

    # Uniform distribution has maximum entropy (1.0 normalized)
    uniform = [0.25, 0.25, 0.25, 0.25]
    assert abs(compute_normalized_entropy(uniform) - 1.0) < 1e-4


def test_ranked_probability_score():
    """Verify ordinal Ranked Probability Score penalizes distant errors more than near errors."""
    # Target is level 0
    target_idx = 0
    # Near prediction (level 1)
    near_pred = [0.0, 1.0, 0.0, 0.0]
    rps_near = compute_ranked_probability_score(near_pred, target_idx)

    # Distant prediction (level 3)
    distant_pred = [0.0, 0.0, 0.0, 1.0]
    rps_distant = compute_ranked_probability_score(distant_pred, target_idx)

    assert rps_near < rps_distant


def test_laya_intent_classification():
    """Verify non-autoregressive single-pass intent classification across cognitive domains."""
    head = LayaDecisionHead()

    # Math intent
    math_dec = head.classify_intent("calculate 250 * 4 + 180")
    assert isinstance(math_dec, ChoiceDecision)
    assert math_dec.choice == "math"
    assert math_dec.confidence > 0.30

    # Code intent
    code_dec = head.classify_intent("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
    assert code_dec.choice == "code"

    # Social intent
    social_dec = head.classify_intent("Hello there, how are you today?")
    assert social_dec.choice == "social"

    # Physics intent
    physics_dec = head.classify_intent("simulate apple decay over 10 days at 22 Celsius")
    assert physics_dec.choice == "physics"


def test_laya_evaluate_choice_arbitrary_options():
    """Verify evaluate_choice non-autoregressively evaluates candidate options."""
    head = LayaDecisionHead()

    options = ["software engineer", "celestial star", "african mammal", "kitchen utensil"]
    dec = head.evaluate_choice("Betelgeuse is a giant astronomical sun in Orion", options)

    assert dec.choice == "celestial star"
    assert dec.confidence > 0.40
    assert dec.margin > 0.10


def test_laya_evaluate_noul_truth_unit():
    """Verify noul truth proposition evaluation and calibration under OWA."""
    head = LayaDecisionHead()

    # Truth statement
    true_noul = head.evaluate_noul("Linux is a free open-source computer operating system")
    assert isinstance(true_noul, NoulDecision)
    assert true_noul.truth_probability > 0.50

    # Random nonsense under Open-World Assumption
    nonsense_noul = head.evaluate_noul("A zanthocar flurblegorps on Quazilopia")
    # Low confidence or near 0.5
    assert nonsense_noul.status in (EpistemicBeliefState.UNKNOWN, EpistemicBeliefState.UNCERTAIN)


def test_laya_evaluate_score():
    """Verify ordinal score head."""
    head = LayaDecisionHead()

    # Priority levels
    levels = ["low priority", "medium priority", "high priority", "critical priority"]
    score_res = head.evaluate_score("URGENT: system security vulnerability found in authentication module", levels)

    assert isinstance(score_res, ScoreDecision)
    assert score_res.discrete_choice in (2, 3)  # high or critical
    assert score_res.expected_score >= 1.5


def test_laya_fast_path_and_system2_escalation():
    """Verify fast-path triggering and System 2 escalation thresholds."""
    head = LayaDecisionHead(entropy_threshold=0.35, confidence_threshold=0.75, margin_threshold=0.20)

    # High certainty choice
    clear_dec = head.classify_intent("Hello!")
    if clear_dec.normalized_entropy < 0.35 and clear_dec.margin >= 0.20:
        assert head.is_fast_path(clear_dec) is True
        assert head.should_escalate_to_system2(clear_dec) is False

    # Ambiguous input
    vague_dec = head.classify_intent("x")
    assert head.should_escalate_to_system2(vague_dec) is True


def test_laya_online_learning():
    """Verify online learning gradient step reduces Brier score on target."""
    head = LayaDecisionHead()
    state = head.encoder.encode("quantum entanglement teleportation experiment")

    # Initial loss for 'physics'
    dec_before = head.classify_intent(state)
    p_before = dec_before.probabilities["physics"]

    # Perform online learning step on 'physics'
    head.learn(state, "physics", learning_rate=0.2)

    dec_after = head.classify_intent(state)
    p_after = dec_after.probabilities["physics"]

    assert p_after > p_before
