"""Integration and verification tests for the LITTLE cognitive architecture.

Verifies:
- Milestone 1: Persistent memory across full database close & reopen.
- Milestone 2 & 3: Transitive graph inference (A -> B -> C).
- Milestone 4: Explicit UNKNOWN and REFUTED detection under Open-World Assumption.
- Milestone 5: Multi-valued property learning (apple is red + green).
- Zero Catastrophic Forgetting benchmark.
"""

from pathlib import Path

from little.core.models import BeliefStatus, UpdateType
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_core_persistence_and_unknown_flow(tmp_path: Path) -> None:
    """The canonical milestone test from todo.md Section 22."""
    db_file = tmp_path / "little_milestone.db"

    # 1. Teach: "A dog is an animal."
    with MemoryStore(db_file) as store1:
        engine1 = LearningEngine(store1)
        learn_res = engine1.learn("A dog is an animal.")
        assert "dog" in learn_res.concepts_created
        assert "animal" in learn_res.concepts_created

        # Query while in session
        res1 = engine1.ask("Is a dog an animal?")
        assert res1.status == BeliefStatus.SUPPORTED
        assert res1.answer is True
        assert res1.confidence > 0.0

    # 2. Simulate complete program termination & restart
    with MemoryStore(db_file) as store2:
        engine2 = LearningEngine(store2)

        # Confirm knowledge persisted across restarts
        res_persisted = engine2.ask("Is a dog an animal?")
        assert res_persisted.status == BeliefStatus.SUPPORTED
        assert res_persisted.answer is True

        # Test unobserved relationship -> must return UNKNOWN, NOT hallucinate
        res_unknown = engine2.ask("Is a dog a vehicle?")
        assert res_unknown.status == BeliefStatus.UNKNOWN
        assert res_unknown.answer is None


def test_transitive_multi_hop_inference(tmp_path: Path) -> None:
    """Verify deductive chaining: Dog -> Animal -> Living Thing => Dog is a Living Thing."""
    db_file = tmp_path / "little_transitive.db"

    with MemoryStore(db_file) as store:
        engine = LearningEngine(store)

        engine.learn("A dog is an animal.")
        engine.learn("An animal is a living thing.")

        # Multi-hop deduction
        res = engine.ask("Is a dog a living thing?")
        assert res.status == BeliefStatus.SUPPORTED
        assert res.answer is True
        assert any("Chain:" in ev for ev in res.evidence)
        assert any("dog -> animal -> living thing" in step for step in res.trace)


def test_disjoint_refutation(tmp_path: Path) -> None:
    """Verify negative evidence & mutual exclusivity."""
    db_file = tmp_path / "little_disjoint.db"

    with MemoryStore(db_file) as store:
        engine = LearningEngine(store)

        engine.learn("A dog is an animal.")
        engine.learn("A car is a vehicle.")
        engine.learn("An animal is not a vehicle.")

        # Dog -> Animal --[disjoint]--> Vehicle => Dog cannot be a vehicle
        res = engine.ask("Is a dog a vehicle?")
        assert res.status == BeliefStatus.REFUTED
        assert res.answer is False
        assert any("disjoint" in ev.lower() for ev in res.evidence)


def test_multi_valued_attributes_apple_example(tmp_path: Path) -> None:
    """Verify the core apple concept requirement: color is variable, not identity-defining."""
    db_file = tmp_path / "little_apple.db"

    with MemoryStore(db_file) as store:
        engine = LearningEngine(store)

        res1 = engine.learn("The apple is red.")
        assert res1.update_type == UpdateType.PROPERTY_UPDATE

        # Add second color: green
        res2 = engine.learn("The apple is green.")
        assert res2.update_type == UpdateType.PROPERTY_UPDATE

        # Query colors
        q_res = engine.ask("What color is the apple?")
        assert q_res.status == BeliefStatus.SUPPORTED
        assert "red" in str(q_res.answer)
        assert "green" in str(q_res.answer)

        # Concept is not overwritten
        apple_concept = store.get_concept("apple")
        assert apple_concept is not None
        assert isinstance(apple_concept.attributes["color"], list)
        assert set(apple_concept.attributes["color"]) == {"red", "green"}


def test_zero_catastrophic_forgetting(tmp_path: Path) -> None:
    """Continually learn 10 sequential facts and verify earlier knowledge remains intact."""
    db_file = tmp_path / "little_continual.db"

    facts = [
        ("A cat is an animal.", "Is a cat an animal?"),
        ("A dog is an animal.", "Is a dog an animal?"),
        ("A sparrow is a bird.", "Is a sparrow a bird?"),
        ("A bird is an animal.", "Is a sparrow an animal?"),  # transitive
        ("A car is a vehicle.", "Is a car a vehicle?"),
        ("A bicycle is a vehicle.", "Is a bicycle a vehicle?"),
        ("A rose is a flower.", "Is a rose a flower?"),
        ("An apple is a fruit.", "Is an apple a fruit?"),
        ("A fruit is a food.", "Is an apple a food?"),  # transitive
        ("Alice has a dog.", "Does Alice have a dog?"),
    ]

    with MemoryStore(db_file) as store:
        engine = LearningEngine(store)

        # Learn all sequentially
        for statement, _ in facts:
            engine.learn(statement)

        # Verify all earlier facts are still 100% known
        for _, query in facts:
            res = engine.ask(query)
            assert res.status == BeliefStatus.SUPPORTED, f"Failed on query: {query}"
            assert res.answer is True or res.answer == "dog"
