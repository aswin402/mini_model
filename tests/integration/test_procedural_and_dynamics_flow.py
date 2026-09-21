"""Integration tests for procedural execution, slicing actions, and continuous-time dynamics."""

from little.core.models import BeliefStatus
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_procedural_mathematics_generalization():
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # Arithmetic via symbols
    res_add = engine.ask("What is 9876 + 5432?")
    assert res_add.status == BeliefStatus.SUPPORTED
    assert res_add.answer == 15308
    assert res_add.confidence == 1.0

    # Arithmetic via words
    res_mul = engine.ask("Calculate 12 times 15")
    assert res_mul.status == BeliefStatus.SUPPORTED
    assert res_mul.answer == 180

    # Factorial
    res_fact = engine.ask("What is the factorial of 5?")
    assert res_fact.status == BeliefStatus.SUPPORTED
    assert res_fact.answer == 120


def test_apple_slicing_and_liquid_dynamics_flow():
    """Canonical test representing user vision:

    Apple -> sliced into 4 pieces -> creates part_of relations ->
    evaluates continuous enzymatic oxidation ODE over delta_t.
    """
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    # 1. Teach foundational concept
    engine.learn("An apple is a fruit.")

    # 2. Perform action: Slice apple into 4 pieces
    slice_res = engine.learn("Slice an apple into 4 pieces.")
    assert "Successfully sliced apple into 4 pieces" in slice_res.message
    assert "apple slice" in slice_res.concepts_created

    # 3. Transitive and part-whole reasoning:
    # Is an apple slice part of an apple?
    q_part = engine.ask("Is an apple slice part of an apple?")
    assert q_part.status == BeliefStatus.SUPPORTED
    assert q_part.answer is True

    # Transitive deduction: Is an apple slice a fruit?
    q_fruit = engine.ask("Is an apple slice a fruit?")
    assert q_fruit.status == BeliefStatus.SUPPORTED
    assert q_fruit.answer is True

    # 4. Continuous-Time Dynamics (Liquid AI ODE):
    # At t = 0 seconds: exposed apple slice flesh is white/fresh
    q_color_0 = engine.ask("What color is the apple slice after 0 seconds?")
    assert q_color_0.status == BeliefStatus.SUPPORTED
    assert q_color_0.answer == "white"

    q_fresh_0 = engine.ask("Is the apple slice fresh after 0 seconds?")
    assert q_fresh_0.status == BeliefStatus.SUPPORTED
    assert q_fresh_0.answer is True

    # After 2 hours (7200 seconds): continuous oxidation causes browning
    q_color_2hr = engine.ask("What color is the apple slice after 2 hours?")
    assert q_color_2hr.status == BeliefStatus.SUPPORTED
    assert q_color_2hr.answer == "brown"

    q_fresh_2hr = engine.ask("Is the apple slice fresh after 2 hours?")
    assert q_fresh_2hr.status == BeliefStatus.REFUTED
    assert q_fresh_2hr.answer is False
