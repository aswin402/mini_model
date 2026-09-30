"""Unit tests for Headroom reversible context compression and memory budgeting."""

import time
import pytest

from little.memory.headroom import HeadroomContextManager, WorkingTurn
from little.memory.headroom_policy import HeadroomPolicy
from little.memory.store import MemoryStore


def test_headroom_salience_calculation():
    """Verify multi-factor salience scoring."""
    store = MemoryStore()
    manager = HeadroomContextManager(store)

    turn = manager.record_turn(
        role="user",
        text="Apples are delicious fruits that grow on trees.",
        concept_ids=["apple", "fruit"],
    )

    salience = manager.compute_turn_salience(turn)
    assert 0.0 < salience.total_salience <= 1.0
    assert salience.recency_score > 0.9  # Just recorded
    assert salience.frequency_score > 0.0
    assert salience.activation_score > 0.0


def test_headroom_reversible_context_compression():
    """Verify CCR compresses older turns when token bounds are reached and allows decompression."""
    store = MemoryStore()
    # Configure tiny token limit to force compression
    policy = HeadroomPolicy(
        max_active_tokens=35,
        max_active_concepts=10,
        max_working_episodes=5,
        recency_decay_lambda=0.01,
        salience_weights={"recency": 0.4, "frequency": 0.3, "activation": 0.3},
        eviction_salience_threshold=0.25,
        memory_headroom_budget_mb=10.0,
    )
    manager = HeadroomContextManager(store, policy=policy)

    # 1. Record an initial experience in store
    exp_id = store.add_experience(
        input_text="The quick brown fox jumps over the lazy dog repeatedly in the forest.",
        source="user",
    )

    turn1 = manager.record_turn(
        role="user",
        text="The quick brown fox jumps over the lazy dog repeatedly in the forest.",
        concept_ids=["fox", "dog"],
        experience_id=exp_id,
    )
    turn2 = manager.record_turn(
        role="assistant",
        text="Indeed, foxes are known for their agile leaping capabilities in woodlands.",
        concept_ids=["fox"],
    )
    turn3 = manager.record_turn(
        role="user",
        text="Tell me more about other animals that inhabit the dense temperate forest ecosystems.",
        concept_ids=["animal", "forest"],
    )

    stats = manager.stats()
    # Turn 1 should be compressed to preserve headroom
    assert stats.compressed_turns > 0
    assert turn1.is_compressed is True
    assert "[USER discussed" in turn1.compressed_summary

    # 2. Test reversible decompression
    decompressed = manager.decompress_turn(turn1.turn_id)
    assert decompressed == "The quick brown fox jumps over the lazy dog repeatedly in the forest."


def test_headroom_concept_pruning():
    """Verify active concept pruning when concept bound is exceeded."""
    store = MemoryStore()
    policy = HeadroomPolicy(
        max_active_tokens=1000,
        max_active_concepts=3,  # Strict cap of 3
        max_working_episodes=10,
        recency_decay_lambda=0.05,
        salience_weights={"recency": 0.5, "frequency": 0.25, "activation": 0.25},
        eviction_salience_threshold=0.25,
        memory_headroom_budget_mb=10.0,
    )
    manager = HeadroomContextManager(store, policy=policy)

    manager.record_turn(role="user", text="alpha", concept_ids=["concept_a"])
    manager.record_turn(role="user", text="beta", concept_ids=["concept_b"])
    manager.record_turn(role="user", text="gamma", concept_ids=["concept_c"])
    assert len(manager.get_active_concepts()) == 3

    # Add a 4th concept: should prune the oldest/least frequent
    manager.record_turn(role="user", text="delta", concept_ids=["concept_d"])
    active = manager.get_active_concepts()
    assert len(active) == 3
    assert "concept_d" in active
