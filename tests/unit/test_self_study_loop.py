import pytest
from pathlib import Path
from little.memory.store import MemoryStore
from little.language.parser import LearningEngine
from little.core.models import BeliefStatus
from little.active.self_study import AutonomousSelfStudyEngine, CuriosityStepResult


def test_radial_spoke_entropy_calculation():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = AutonomousSelfStudyEngine(store)

    # Add a sparse concept with only is_a
    c = store.create_concept(name="titanium")
    metal = store.get_concept("metal")
    if not metal:
        metal = store.create_concept(name="metal")
    store.add_relation(c.id, "is_a", metal.id)

    entropy_dict = engine.compute_all_knot_entropies()
    assert "titanium" in entropy_dict
    assert entropy_dict["titanium"] > 0.5  # High entropy due to missing axes


def test_analogical_spoke_proposal_and_gate_verification():
    store = MemoryStore(":memory:", seed_ontology=True)
    learning_engine = LearningEngine(store)
    study_engine = AutonomousSelfStudyEngine(store)

    # 1. Establish parent and sibling with capability
    learning_engine.learn("A falcon is a bird.")
    learning_engine.learn("A falcon can fly.")

    # 2. Introduce new bird lacking procedural capability
    learning_engine.learn("A hawk is a bird.")

    # Prior to self-study, hawk capability is unknown
    res_before = learning_engine.ask("Can a hawk fly?")
    # May be supported via bird if bird has fly, but here bird was not directly given fly, only falcon was
    # Now trigger autonomous self-study
    step_res = study_engine.study_step(target_concept="hawk")
    assert step_res is not None
    assert step_res.status == "PERSISTED"
    assert step_res.hypothesis["subject"] == "hawk"
    assert step_res.hypothesis["predicate"] in ("can", "eats", "lives_in", "part_of")
    assert step_res.hypothesis["object"] == "fly"

    # Now verify that SQLite store was updated live
    res_after = learning_engine.ask("Can a hawk fly?")
    assert res_after.status == BeliefStatus.SUPPORTED


def test_mutex_rejection_by_invariant_gate():
    store = MemoryStore(":memory:", seed_ontology=True)
    learning_engine = LearningEngine(store)
    study_engine = AutonomousSelfStudyEngine(store)

    # Bird can fly
    learning_engine.learn("An eagle is a bird.")
    learning_engine.learn("An eagle can fly.")

    # Penguin is a bird, but explicitly cannot fly
    learning_engine.learn("A penguin is a bird.")
    learning_engine.learn("A penguin cannot fly.")

    # Attempting to hypothesize (penguin, can, fly) must be rejected by invariant gate
    hyp = {"subject": "penguin", "predicate": "can", "object": "fly"}
    passed, reason = study_engine.verify_invariant_gates(hyp)
    assert not passed
    assert "disjoint" in reason.lower() or "negative" in reason.lower() or "mutex" in reason.lower()


def test_autonomous_multi_step_study_cycle():
    store = MemoryStore(":memory:", seed_ontology=True)
    learning_engine = LearningEngine(store)
    study_engine = AutonomousSelfStudyEngine(store)

    learning_engine.learn("A trout is a fish.")
    learning_engine.learn("A salmon is a fish.")
    learning_engine.learn("A salmon lives in water.")

    # Run 2 autonomous study cycles
    results = study_engine.study_cycle(max_steps=2)
    assert len(results) >= 1
    assert any(r.status == "PERSISTED" for r in results)


def test_self_study_episodic_axis_is_covered_by_matching_experience():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = AutonomousSelfStudyEngine(store)
    store.create_concept("apple")
    store.add_experience(
        "I observed an apple.",
        [{"subject": "apple", "predicate": "observed", "object": "fruit"}],
        source="sensor",
    )

    assert "episodic" not in engine.identify_missing_axes("apple")


def test_self_study_uses_policy_metadata_for_hypothesis_frame(monkeypatch):
    store = MemoryStore(":memory:", seed_ontology=True)
    learning_engine = LearningEngine(store)
    engine = AutonomousSelfStudyEngine(store)
    captured = []
    original = store.ledger.propose_relation

    def capture(frame, claim, *args, **kwargs):
        if frame.model_id == engine.policy.model_id:
            captured.append(frame)
        return original(frame, claim, *args, **kwargs)

    monkeypatch.setattr(store.ledger, "propose_relation", capture)
    learning_engine.learn("A falcon is a bird.")
    learning_engine.learn("A falcon can fly.")
    learning_engine.learn("A hawk is a bird.")

    result = engine.study_step("hawk")

    assert result is not None
    assert result.status == "PERSISTED"
    assert captured
    frame = captured[0]
    assert frame.model_id == engine.policy.model_id
    assert frame.model_version == engine.policy.model_version
    assert frame.claims[0].probability == pytest.approx(
        engine.policy.hypothesis_probability
    )
    assert frame.uncertainty == pytest.approx(engine.policy.hypothesis_uncertainty)
