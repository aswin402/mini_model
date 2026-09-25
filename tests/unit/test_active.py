"""Unit tests for Active Learning and Curiosity Engine."""

from little.active.inquisitor import ActiveInquisitor
from little.core.models import BeliefStatus
from little.language.parser import LearningEngine
from pathlib import Path

from little.memory.store import MemoryStore
from little.active.active_policy import ActiveLearningPolicy


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


def test_active_inquisitor_uses_policy_for_default_predicate_and_prompt(monkeypatch):
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    policy = ActiveLearningPolicy.load(Path("data/schemas"))
    policy = policy.__class__(
        version=policy.version,
        meta_words=(*policy.meta_words, "synthetic-meta"),
        default_predicate="subclass_of",
        entropy_probability_floor=policy.entropy_probability_floor,
        unknown_confidence_threshold=policy.unknown_confidence_threshold,
        binary_prior=policy.binary_prior,
        confidence_scale=policy.confidence_scale,
        round_digits=policy.round_digits,
        prompt_templates=policy.prompt_templates,
    )
    inquisitor = ActiveInquisitor(store, engine, active_policy=policy)

    result = engine.ask("Is synthetic-meta a fruit?")
    assert inquisitor.inspect_uncertainty(result, "synthetic-meta", "", "fruit") is None

    result = engine.ask("What is pear?")
    prompt = inquisitor.inspect_uncertainty(result, "pear", "", None)
    assert prompt is not None
    assert prompt.pending_predicate == "subclass_of"
