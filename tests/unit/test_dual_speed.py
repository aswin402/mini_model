from dataclasses import replace

import pytest

from little.inference.dual_speed import (
    DualSpeedInfillingEngine,
)
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.reasoning_policy import ReasoningPolicy
from little.knowledge.registry import SchemaRegistry
from little.memory.store import MemoryStore


def test_dual_speed_fast_mode_hit():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.add_relation("APPLE", "is_a", "FRUIT")

    res = engine.query("APPLE", "FRUIT")
    assert res.mode == "FAST"
    assert res.path == ["APPLE", "FRUIT"]
    assert (
        res.confidence
        == memory.get_relations(
            subject_id="APPLE", predicate="is_a", object_id="FRUIT"
        )[0].confidence
    )
    assert "<think>" not in res.inspectable_trace


def test_dual_speed_direct_lookup_rejects_negative_evidence():
    memory = MemoryStore(":memory:", seed_ontology=False)
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.add_relation("APPLE", "is_a", "FRUIT", positive=False)

    res = engine.query("APPLE", "FRUIT")

    assert res.path == []
    assert res.confidence == 0.0


def test_dual_speed_thinking_mode_multi_hop_collision():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    # 4-hop chain: GalaApple -> Apple -> PomeFruit -> Fruit -> Plant
    memory.add_relation("GALA_APPLE", "is_a", "APPLE")
    memory.add_relation("APPLE", "is_a", "POME_FRUIT")
    memory.add_relation("POME_FRUIT", "is_a", "FRUIT")
    memory.add_relation("FRUIT", "is_a", "PLANT")

    res = engine.query("GALA_APPLE", "PLANT")
    assert res.mode == "THINKING"
    assert res.path == ["GALA_APPLE", "APPLE", "POME_FRUIT", "FRUIT", "PLANT"]
    assert "<think>" in res.inspectable_trace
    assert "Frontier collision verified" in res.inspectable_trace


def test_dual_speed_uses_injected_reasoning_policy_for_confidence():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    policy = ReasoningPolicy(
        version="test",
        max_depth=8,
        fast_confidence=0.71,
        verified_confidence=0.82,
        unknown_confidence=0.13,
        inheritance_confidence_factor=0.95,
        inference_unknown_confidence=0.1,
    )
    engine = DualSpeedInfillingEngine(memory, verifier, policy=policy)

    memory.add_relation("APPLE", "is_a", "FRUIT")
    direct = engine.query("APPLE", "FRUIT")
    unknown = engine.query("APPLE", "SPACESHIP")

    assert direct.confidence == pytest.approx(
        memory.get_relations(subject_id="APPLE", predicate="is_a", object_id="FRUIT")[
            0
        ].confidence
    )
    assert unknown.confidence == pytest.approx(0.13)


def test_dual_speed_default_taxonomy_comes_from_registry():
    memory = MemoryStore(":memory:", seed_ontology=False)
    default = SchemaRegistry.default()
    custom = SchemaRegistry(
        relations={
            name: replace(schema, primary=name == "subclass_of")
            for name, schema in default._relations.items()
        },
        actions=default._actions,
        state_variables=default._state_variables,
    )
    verifier = DeepSeekInvariantVerifier(memory, registry=custom)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.add_relation("APPLE", "subclass_of", "FRUIT")

    result = engine.query("APPLE", "FRUIT")

    assert result.mode == "FAST"
    assert result.path == ["APPLE", "FRUIT"]
