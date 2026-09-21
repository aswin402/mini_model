"""Unit tests for Continuous-Time Dynamics and Transformation Engine."""

from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.transformations import TransformationEngine
from little.inference.engine import InferenceEngine
from little.memory.store import MemoryStore


def test_continuous_dynamics_evolution():
    # Test uncut whole apple: 14 day tau
    whole_state = ContinuousDynamicsEngine.create_initial_state(exposed_to_air=False)
    assert whole_state.freshness == 1.0
    assert whole_state.oxidation == 0.0

    # After 1 hour (3600s), uncut apple is still basically 100% fresh and unoxidized
    after_1hr = ContinuousDynamicsEngine.evolve_state(whole_state, delta_seconds=3600)
    assert after_1hr.freshness > 0.99
    assert after_1hr.oxidation < 0.01
    assert ContinuousDynamicsEngine.get_perceived_condition(after_1hr) == "fresh"
    assert ContinuousDynamicsEngine.get_perceived_color(after_1hr, "white") == "white"

    # Test sliced exposed apple: 30 min tau (1800s)
    slice_state = ContinuousDynamicsEngine.create_initial_state(exposed_to_air=True)
    assert slice_state.exposed_to_air is True

    # After 15 minutes (900s): starts browning
    after_15min = ContinuousDynamicsEngine.evolve_state(slice_state, delta_seconds=900)
    assert 0.35 <= after_15min.oxidation <= 0.45
    assert (
        ContinuousDynamicsEngine.get_perceived_color(after_15min, "white")
        == "light brown"
    )

    # After 2 hours (7200s): heavily oxidized/brown
    after_2hr = ContinuousDynamicsEngine.evolve_state(slice_state, delta_seconds=7200)
    assert after_2hr.oxidation > 0.95
    assert ContinuousDynamicsEngine.get_perceived_color(after_2hr, "white") == "brown"


def test_transformation_slicing_and_relational_reasoning():
    store = MemoryStore(":memory:")
    # Teach that apple is a fruit
    apple = store.create_concept(
        "apple", category="fruit", attributes={"interior_color": "white"}
    )
    fruit = store.create_concept("fruit")
    store.add_relation(subject_id=apple.id, predicate="is_a", object_id=fruit.id)

    # Perform action: Slice apple into 4 pieces
    result = TransformationEngine.slice_object(store, "apple", num_pieces=4)
    assert result.num_pieces == 4
    assert result.slice_concept == "apple slice"
    assert len(result.entities_created) == 4

    # Verify semantic reasoning:
    # 1. Is an apple slice part of an apple?
    engine = InferenceEngine(store)
    res_part = engine.infer("apple slice", "part_of", "apple")
    assert res_part.is_supported is True
    assert res_part.confidence >= 0.5

    # 2. Is an apple slice a fruit? (Transitive deduction: apple slice -> apple -> fruit)
    res_trans = engine.infer("apple slice", "is_a", "fruit")
    assert res_trans.is_supported is True
    assert "apple slice -> apple -> fruit" in res_trans.trace[-1] or any(
        "apple slice -> apple -> fruit" in t for t in res_trans.trace
    )
