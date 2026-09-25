from pathlib import Path

from little.core.concept_knot import (
    ConceptKnot,
)
from little.dynamics.cfc_ode import (
    CfCContinuousODE,
    apply_action_jump,
)


def test_concept_knot_initialization():
    apple = ConceptKnot.create_apple_exemplar()

    # 90° Taxonomic Axis
    assert "Fruit" in apple.taxonomy_hypernyms

    # 135° Mereological Axis
    assert "skin" in apple.mereology_parts
    assert "pulp" in apple.mereology_parts

    # 45° Continuous Dynamics
    assert apple.dynamics_state.freshness == 1.0
    assert apple.dynamics_state.oxidation == 0.0

    # 180° Invariant Axioms
    assert "Animal" in apple.invariant_disjoints
    assert apple.invariant_mass == 180.0

    # 225° Procedural Skills
    assert "slice" in apple.procedural_skills


def test_concept_knot_loads_data_fixture_without_object_specific_constructor():
    apple = ConceptKnot.load_fixture(
        "apple", Path("data/fixtures/concept_knots")
    )

    assert apple.concept_id == "APPLE"
    assert apple.dynamics_state.moisture == 0.86
    assert apple.invariant_mass == 180.0


def test_cfc_continuous_ode_evolution():
    apple = ConceptKnot.create_apple_exemplar()
    ode = CfCContinuousODE()

    # Evolve 24 hours at 22°C (intact skin)
    state_24h = ode.evolve(
        apple.dynamics_state, delta_t_hours=24.0, skin_intact=True
    )
    assert state_24h.freshness < 1.0
    assert state_24h.oxidation > 0.0
    assert state_24h.freshness > 0.90  # intact skin slows decay


def test_hybrid_automaton_action_jump_slice():
    apple = ConceptKnot.create_apple_exemplar()
    ode = CfCContinuousODE()

    # Action jump: slice apple into 4 pieces
    pieces = apply_action_jump(apple, "slice", num_pieces=4)
    assert len(pieces) == 4
    for p in pieces:
        assert p.invariant_mass == 45.0  # Mass conservation QPT: 180 / 4 = 45g
        assert p.mereology_parts["skin"] == "partial_boundary"
        assert p.mereology_parts["pulp"] == "exposed"

    # Evolved sliced piece for 2 hours (severed skin accelerates oxidation)
    p0_evolved = ode.evolve(
        pieces[0].dynamics_state, delta_t_hours=2.0, skin_intact=False
    )
    assert p0_evolved.oxidation > 0.20  # Rapid enzymatic browning!


def test_action_jump_uses_knot_parts_and_skills_for_arbitrary_material():
    metal = ConceptKnot(
        concept_id="METAL_BLOCK",
        taxonomy_hypernyms=["Material"],
        mereology_parts={"shell": "external_boundary", "core": "interior"},
        invariant_mass=10.0,
        procedural_skills=["polish"],
    )

    pieces = apply_action_jump(metal, "slice", num_pieces=2)

    assert len(pieces) == 2
    assert pieces[0].invariant_mass == 5.0
    assert pieces[0].mereology_parts == {
        "shell": "partial_boundary",
        "core": "exposed",
    }
    assert pieces[0].procedural_skills == ["polish"]
