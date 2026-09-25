"""Standalone heuristic fallback tests; this is not the kernel adapter."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from little.knowledge.policy import LanguagePolicy
from little.language.laya_gatekeeper import (
    BeliefStatus,
    CalibratedDecision,
    LayaSystem1Gatekeeper,
    QueryIntent,
)
from little.language.laya_policy import LayaDecisionPolicy


def test_gatekeeper_intent_classification():
    gatekeeper = LayaSystem1Gatekeeper()

    # 1. Statements
    assert gatekeeper.classify_intent("Apples are fruits.") == QueryIntent.STATEMENT
    assert gatekeeper.classify_intent("A dog is an animal.") == QueryIntent.STATEMENT

    # 2. Questions
    assert gatekeeper.classify_intent("Is an apple a fruit?") == QueryIntent.QUESTION
    assert gatekeeper.classify_intent("What color is a banana?") == QueryIntent.QUESTION

    # 3. Actions
    assert gatekeeper.classify_intent("Slice the apple into 4 pieces.") == QueryIntent.ACTION
    assert gatekeeper.classify_intent("Peel the orange.") == QueryIntent.ACTION

    # 4. Mathematics
    assert gatekeeper.classify_intent("What is 12345 + 67890?") == QueryIntent.MATH
    assert gatekeeper.classify_intent("Calculate the square of 9.") == QueryIntent.MATH

    # 5. Curiosity / Meta-inquiry
    assert gatekeeper.classify_intent("What are you uncertain about?") == QueryIntent.CURIOSITY


def test_gatekeeper_uses_configured_intent_priority(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/language_policy.json").read_text(encoding="utf-8")
    )
    payload["language_policy"]["intent_priority"] = [
        "question",
        "math",
        "curiosity",
        "action",
        "statement",
    ]
    (tmp_path / "language_policy.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    policy = LanguagePolicy.load(tmp_path)
    gatekeeper = LayaSystem1Gatekeeper(policy=policy)

    assert gatekeeper.classify_intent("Calculate 2 + 2?") == QueryIntent.QUESTION


def test_gatekeeper_calibrated_choice_and_entropy_gating():
    gatekeeper = LayaSystem1Gatekeeper()

    # High confidence decision
    res = gatekeeper.evaluate_choice(
        state="The object is a red gala fruit picked from a tree.",
        question="Which concept does this match?",
        options=["Apple", "Bicycle", "Galaxy"],
    )
    assert isinstance(res, CalibratedDecision)
    assert res.winner == "Apple"
    assert res.normalized_entropy < 0.35
    assert res.status == BeliefStatus.SUPPORTED

    # High entropy decision (insufficient evidence / ambiguous)
    res_ambiguous = gatekeeper.evaluate_choice(
        state="It is a mysterious entity from an unknown dimension.",
        question="Which concept does this match?",
        options=["Apple", "Bicycle", "Galaxy"],
    )
    assert res_ambiguous.normalized_entropy >= 0.35
    assert res_ambiguous.status == BeliefStatus.UNKNOWN


def test_gatekeeper_verify_proposition_noul():
    gatekeeper = LayaSystem1Gatekeeper()

    # Confirmed proposition
    res = gatekeeper.verify_proposition(
        "Apples grow on trees.", evidence_context="Apples grow on deciduous trees."
    )
    assert res.winner == "true"
    assert res.status == BeliefStatus.SUPPORTED

    # Ambiguous proposition with no context -> UNKNOWN (H >= 0.35)
    res_unknown = gatekeeper.verify_proposition("Martians eat apples.", evidence_context="")
    assert res_unknown.status == BeliefStatus.UNKNOWN


def test_gatekeeper_loads_decision_policy_separately_from_language_policy():
    decision_policy = LayaDecisionPolicy.default()
    language_policy = LanguagePolicy(
        math_markers=(),
        curiosity_markers=(),
        action_prefixes=(),
        question_prefixes=("is ",),
        concept_associations={"quartz": frozenset({"silica"})},
    )

    gatekeeper = LayaSystem1Gatekeeper(
        policy=language_policy,
        decision_policy=decision_policy,
    )

    assert gatekeeper.policy is language_policy
    assert gatekeeper.decision_policy == decision_policy
    assert gatekeeper.entropy_unknown_threshold == (
        decision_policy.entropy_unknown_threshold
    )


def test_gatekeeper_accepts_decision_policy_injection_and_entropy_override():
    default_policy = LayaDecisionPolicy.default()
    decision_policy = replace(
        default_policy,
        choice=replace(
            default_policy.choice,
            temperature_buckets={
                bucket: 2.0 for bucket in default_policy.choice.temperature_buckets
            },
        ),
        entropy_unknown_threshold=0.01,
    )
    gatekeeper = LayaSystem1Gatekeeper(
        entropy_unknown_threshold=0.95,
        decision_policy=decision_policy,
    )

    result = gatekeeper.evaluate_choice(
        state="The object is red.",
        question="Which concept does this match?",
        options=["Apple", "Bicycle", "Galaxy"],
    )

    assert gatekeeper.entropy_unknown_threshold == 0.95
    assert result.status != BeliefStatus.UNKNOWN


def test_decision_policy_rejects_missing_or_malformed_data(tmp_path: Path):
    policy_path = tmp_path / "laya_policy.json"

    policy_path.write_text(
        json.dumps({"laya_policy": {"entropy_unknown_threshold": 0.35}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing field"):
        LayaDecisionPolicy.load(tmp_path)

    policy_path.write_text(
        json.dumps({"laya_policy": {"entropy_unknown_threshold": "high"}}),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="must be a finite number"):
        LayaDecisionPolicy.load(tmp_path)


def test_decision_policy_rejects_out_of_range_calibration(tmp_path: Path):
    source = Path("data/schemas/laya_policy.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["laya_policy"]["choice"]["supported_probability_threshold"] = 1.5
    (tmp_path / "laya_policy.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        LayaDecisionPolicy.load(tmp_path)


def test_decision_policy_rejects_invalid_injected_values():
    policy = LayaDecisionPolicy.default()

    with pytest.raises(ValueError, match="entropy_probability_floor"):
        replace(policy, entropy_probability_floor=2.0)

    with pytest.raises(ValueError, match="supported_probability_threshold"):
        replace(
            policy,
            choice=replace(
                policy.choice,
                supported_probability_threshold=float("nan"),
            ),
        )

    with pytest.raises(ValueError, match="match_weight"):
        replace(
            policy,
            proposition=replace(policy.proposition, match_weight=-1.0),
        )

    with pytest.raises(ValueError, match="minimum token length"):
        replace(
            policy,
            choice=replace(policy.choice, minimum_token_length=True),
        )

    with pytest.raises(ValueError, match="bucket names"):
        replace(
            policy,
            choice=replace(policy.choice, temperature_buckets={1: 1.0}),
        )
