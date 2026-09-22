import pytest
from little.memory.store import MemoryStore
from little.inference.invariant_gates import (
    DeepSeekInvariantVerifier,
    InvariantGateResult,
)


def test_invariant_gate_dag_acyclicity():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Valid chain: Dog -> Animal -> LivingThing
    memory.add_relation("DOG", "is_a", "ANIMAL")
    memory.add_relation("ANIMAL", "is_a", "LIVING_THING")

    res = verifier.verify_relation("LIVING_THING", "is_a", "DOG")
    # Circular causality: LivingThing is_a Dog creates a cycle in DAG!
    assert not res.passed
    assert res.violated_gate == "I_DAG"
    assert "Cycle detected" in res.error_message


def test_invariant_gate_mutex_disjoint():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Mutex: Plant disjoint_with Animal
    memory.add_relation("PLANT", "disjoint_with", "ANIMAL")
    memory.add_relation("APPLE", "is_a", "PLANT")

    # Apple is_a Animal violates I_mutex
    res = verifier.verify_relation("APPLE", "is_a", "ANIMAL")
    assert not res.passed
    assert res.violated_gate == "I_MUTEX"
    assert "Mutual exclusivity violated" in res.error_message


def test_invariant_gate_sort_and_ground():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)

    # Valid relation passes all 4 gates
    res = verifier.verify_relation("APPLE", "color", "red")
    assert res.passed
    assert res.violated_gate is None
    assert len(res.proof_trace) == 4
