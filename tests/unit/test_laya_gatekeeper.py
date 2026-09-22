import pytest
from little.language.laya_gatekeeper import (
    LayaSystem1Gatekeeper,
    QueryIntent,
    BeliefStatus,
    CalibratedDecision,
)


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
