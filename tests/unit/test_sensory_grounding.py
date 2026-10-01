"""Unit tests for Multimodal Sensory Grounding Engine (0° Episodic Axis)."""

from __future__ import annotations

import pytest
from little.memory.store import MemoryStore
from little.perception.sensory import (
    SensoryGroundingEngine,
    SensoryObservation,
    GroundingResult,
)


def test_sensory_directive_parsing():
    engine = SensoryGroundingEngine(MemoryStore(":memory:"))

    text = "observe sensor: [color: red, mass: 180g, firmness: 0.88, temp: 22C]"
    obs = engine.parse_sensory_directive(text)

    assert obs is not None
    assert obs.features["color"] == "red"
    assert obs.features["mass_grams"] == 180.0
    assert obs.features["firmness"] == 0.88
    assert obs.features["temperature_c"] == 22.0


def test_sensory_grounding_anchors_apple_at_0_degree_episodic_axis():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = SensoryGroundingEngine(store)

    obs = SensoryObservation(
        modality="multimodal",
        features={"color": "red", "mass_grams": 175.0, "texture": "smooth", "category": "fruit"},
    )

    result = engine.ground_observation(obs)

    assert isinstance(result, GroundingResult)
    assert result.status == "GROUNDED"
    assert result.matched_concept.lower() == "apple"
    assert result.confidence > 0.70
    assert result.entity_id is not None

    # Verify episodic instance is recorded in memory
    entity = store.get_entity(result.entity_id)
    assert entity is not None
    assert entity.properties["sensory_features"]["color"] == "red"

    # Verify episodic experience
    experiences = store.list_experiences(limit=5)
    assert any("sensor" in exp.source.lower() for exp in experiences)


def test_sensory_grounding_rejects_invariant_violation():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = SensoryGroundingEngine(store)

    # A 5,000 kg entity claims to be an apple (violates physical mass bounds)
    obs = SensoryObservation(
        modality="multimodal",
        features={"color": "red", "mass_grams": 5_000_000.0, "candidate": "apple"},
    )

    result = engine.ground_observation(obs)
    assert result.status in ("MUTEX_REJECTED", "UNKNOWN")
    assert "mass" in result.reason.lower() or "bound" in result.reason.lower() or "violation" in result.reason.lower()
