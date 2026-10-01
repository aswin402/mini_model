"""Integration tests for Multimodal Sensory Grounding and Continuous Dynamic Physics in CognitiveKernel chat."""

from __future__ import annotations

import pytest
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_conversational_continuous_physics_simulation():
    """Verify predictive continuous-time physical simulation in chat dialogue."""
    mem = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=mem)

    # 1. Ask about leaving an apple for 10 days
    reply = kernel.chat("What happens to apple over 10 days?")
    assert "simulation" in reply.lower() or "apple" in reply.lower()
    assert "freshness" in reply.lower()
    assert "oxidation" in reply.lower()
    assert "moisture" in reply.lower()


def test_conversational_physical_action_jump_peel_and_slice():
    """Verify procedural action jumps and QPT mass conservation via chat dialogue."""
    mem = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=mem)

    # 1. Action jump: peel an apple
    reply_peel = kernel.chat("What happens if I peel an apple?")
    assert "peel" in reply_peel.lower() or "flesh" in reply_peel.lower()
    assert "conserved" in reply_peel.lower() or "mass" in reply_peel.lower()

    # 2. Action jump: slice an apple into 4 pieces
    reply_slice = kernel.chat("If I slice apple into 4 pieces?")
    assert "slice" in reply_slice.lower()
    assert "4" in reply_slice or "pieces" in reply_slice
    assert "conserved" in reply_slice.lower() or "mass" in reply_slice.lower()


def test_conversational_sensory_observation_grounding():
    """Verify multimodal sensory grounding to 0° episodic axis in chat dialogue."""
    mem = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=mem)

    # Send structured sensory stream
    reply = kernel.chat("observe sensor: [color: red, mass: 180g, category: fruit, firmness: 0.9]")
    assert "grounded" in reply.lower()
    assert "apple" in reply.lower()
    assert "episodic" in reply.lower() or "0°" in reply.lower() or "instance" in reply.lower()
