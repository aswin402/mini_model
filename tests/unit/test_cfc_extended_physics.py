"""Unit tests for extended physical action jumps and QPT mass conservation."""

from __future__ import annotations

import pytest
from little.core.concept_knot import ConceptKnot, ContinuousPhysicalState
from little.dynamics.cfc_ode import CfCContinuousODE, apply_action_jump
from little.dynamics.transformations import TransformationEngine
from little.memory.store import MemoryStore


def test_hybrid_automaton_action_jump_peel():
    apple = ConceptKnot.create_apple_exemplar()
    initial_mass = apple.invariant_mass  # 180.0g

    pieces = apply_action_jump(apple, "peel")
    assert len(pieces) == 2

    skin_piece = next((p for p in pieces if "PEEL" in p.concept_id or "SKIN" in p.concept_id), None)
    flesh_piece = next((p for p in pieces if "FLESH" in p.concept_id or "PULP" in p.concept_id), None)

    assert skin_piece is not None
    assert flesh_piece is not None

    # QPT Invariant: Mass conservation
    total_mass = skin_piece.invariant_mass + flesh_piece.invariant_mass
    assert total_mass == pytest.approx(initial_mass, abs=0.01)

    # Boundary topology transition
    assert flesh_piece.mereology_parts.get("pulp") == "exposed"
    assert flesh_piece.mereology_parts.get("skin") == "removed"

    # Accelerated oxidation on exposed flesh
    ode = CfCContinuousODE()
    flesh_evolved = ode.evolve(flesh_piece.dynamics_state, delta_t_hours=2.0, skin_intact=False)
    assert flesh_evolved.oxidation > 0.20


def test_hybrid_automaton_action_jump_heat_and_chill():
    apple = ConceptKnot.create_apple_exemplar()
    t0 = apple.dynamics_state.temperature  # 22.0 C
    m0 = apple.invariant_mass

    # 1. Heat action jump (+25 C)
    heated = apply_action_jump(apple, "heat", delta_celsius=25.0)
    assert len(heated) == 1
    assert heated[0].dynamics_state.temperature == pytest.approx(t0 + 25.0, abs=0.1)
    assert heated[0].invariant_mass == pytest.approx(m0, abs=0.01)

    # 2. Chill action jump (-18 C)
    chilled = apply_action_jump(apple, "chill", delta_celsius=18.0)
    assert len(chilled) == 1
    assert chilled[0].dynamics_state.temperature == pytest.approx(t0 - 18.0, abs=0.1)
    assert chilled[0].invariant_mass == pytest.approx(m0, abs=0.01)


def test_hybrid_automaton_action_jump_dehydrate():
    apple = ConceptKnot.create_apple_exemplar()
    m0 = apple.invariant_mass
    moisture0 = apple.dynamics_state.moisture  # 0.86

    dried = apply_action_jump(apple, "dehydrate")
    assert len(dried) == 1
    d = dried[0]
    assert d.dynamics_state.moisture < 0.10
    # Mass decreased due to water loss, but dry mass conserved
    assert d.invariant_mass < m0
    dry_mass_orig = m0 * (1.0 - moisture0)
    dry_mass_new = d.invariant_mass * (1.0 - d.dynamics_state.moisture)
    assert dry_mass_new == pytest.approx(dry_mass_orig, rel=0.05)


def test_transformation_engine_peel_object():
    store = MemoryStore(":memory:", seed_ontology=True)
    res = TransformationEngine.peel_object(store, "apple")

    assert res is not None
    assert res.parent_concept == "apple"
    assert len(res.entities_created) == 2
    assert "peel" in res.message.lower() or "flesh" in res.message.lower()

    # Verify both entities are persisted in SQLite memory
    assert store.get_entity(res.entities_created[0].id) is not None
    assert store.get_entity(res.entities_created[1].id) is not None


def test_transformation_engine_heat_and_chill_object():
    store = MemoryStore(":memory:", seed_ontology=True)
    res_heat = TransformationEngine.heat_object(store, "apple", delta_celsius=30.0)
    assert res_heat is not None
    assert "heated" in res_heat.message.lower()
    assert res_heat.resulting_temperature == pytest.approx(52.0, abs=1.0)

    res_chill = TransformationEngine.chill_object(store, "apple", delta_celsius=18.0)
    assert res_chill is not None
    assert "chilled" in res_chill.message.lower()
    assert res_chill.resulting_temperature == pytest.approx(34.0, abs=1.0)
