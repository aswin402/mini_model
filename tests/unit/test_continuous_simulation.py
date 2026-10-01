"""Unit tests for the Continuous-Time Physical Simulation Engine."""

from __future__ import annotations

import pytest
from little.core.concept_knot import ConceptKnot, ContinuousPhysicalState
from little.dynamics.continuous_simulation import PhysicalSimulationEngine, SimulationResult


def test_simulation_engine_predicts_decay_over_10_days():
    apple = ConceptKnot.create_apple_exemplar()
    engine = PhysicalSimulationEngine()

    # Simulate whole apple left on counter for 10 days (240 hours) at 22°C
    result = engine.simulate(apple, delta_t_hours=240.0, ambient_temperature_c=22.0, skin_intact=True)

    assert isinstance(result, SimulationResult)
    assert result.concept_name == "APPLE"
    assert result.delta_t_hours == 240.0
    assert result.final_state.freshness < result.initial_state.freshness
    assert result.final_state.oxidation > result.initial_state.oxidation
    assert result.final_state.moisture < result.initial_state.moisture
    assert len(result.milestones) > 0
    assert "freshness" in result.summary.lower()


def test_simulation_engine_shows_accelerated_decay_at_elevated_temperature():
    state_room = ContinuousPhysicalState(freshness=1.0, oxidation=0.0, moisture=0.86, temperature=20.0)
    state_hot = ContinuousPhysicalState(freshness=1.0, oxidation=0.0, moisture=0.86, temperature=35.0)

    engine = PhysicalSimulationEngine()
    res_room = engine.simulate(state_room, delta_t_hours=48.0, ambient_temperature_c=20.0, skin_intact=True)
    res_hot = engine.simulate(state_hot, delta_t_hours=48.0, ambient_temperature_c=35.0, skin_intact=True)

    # Elevated temperature accelerates Arrhenius decay factor
    assert res_hot.final_state.freshness < res_room.final_state.freshness
    assert res_hot.final_state.oxidation > res_room.final_state.oxidation


def test_simulation_engine_exposed_vs_intact_boundary():
    apple = ConceptKnot.create_apple_exemplar()
    engine = PhysicalSimulationEngine()

    # Sliced/peeled (exposed) vs whole (intact skin) over 6 hours
    res_intact = engine.simulate(apple, delta_t_hours=6.0, skin_intact=True)
    res_exposed = engine.simulate(apple, delta_t_hours=6.0, skin_intact=False)

    # Exposed flesh rapidly oxidizes (enzymatic browning)
    assert res_exposed.final_state.oxidation > res_intact.final_state.oxidation
    assert res_exposed.final_state.oxidation > 0.30
