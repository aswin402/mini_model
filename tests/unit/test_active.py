"""Unit tests for Active Learning and Curiosity Engine."""

from little.active.inquisitor import ActiveInquisitor
from little.core.models import BeliefStatus
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_active_inquisitor_unknown_resolution():
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    inquisitor = ActiveInquisitor(store, engine)

    # 1. Ask about an unknown concept "pear" vs "fruit"
    res = engine.ask("Is a pear a fruit?")
    assert res.status == BeliefStatus.UNKNOWN

    # 2. Inquisitor detects epistemic gap and generates question
    prompt = inquisitor.inspect_uncertainty(res, "pear", "is_a", "fruit")
    assert prompt is not None
    assert "I do not know what 'pear' is" in prompt.question_for_user
    assert prompt.pending_subject == "pear"
    assert prompt.pending_object == "fruit"

    # 3. User responds affirmatively: "Yes"
    learn_res = inquisitor.resolve_response(prompt, "Yes")
    assert learn_res.update_type.value in {"NEW_RELATION", "NEW_CONCEPT"}

    # 4. Re-query: Now it should be SUPPORTED with high confidence
    res_after = engine.ask("Is a pear a fruit?")
    assert res_after.status == BeliefStatus.SUPPORTED
    assert res_after.answer is True
    assert res_after.confidence >= 0.5

    # 5. Verify information gain / entropy reduction
    info_gain = inquisitor.calculate_information_gain(
        prior_confidence=res.confidence, posterior_confidence=res_after.confidence
    )
    assert info_gain > 0.0


def test_active_inquisitor_negative_denial():
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    inquisitor = ActiveInquisitor(store, engine)

    # Teach fish
    store.create_concept("fish")
    store.create_concept("mammal")

    res = engine.ask("Is a fish a mammal?")
    assert res.status == BeliefStatus.UNKNOWN

    prompt = inquisitor.inspect_uncertainty(res, "fish", "is_a", "mammal")
    assert prompt is not None

    # User replies: "No"
    inquisitor.resolve_response(prompt, "No")

    # Now asking if fish is a mammal should be REFUTED
    res_refuted = engine.ask("Is a fish a mammal?")
    assert res_refuted.status == BeliefStatus.REFUTED
    assert res_refuted.answer is False
