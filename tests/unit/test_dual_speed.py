import pytest
from little.memory.store import MemoryStore
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.dual_speed import (
    DualSpeedInfillingEngine,
    InfillingResult,
)


def test_dual_speed_fast_mode_hit():
    memory = MemoryStore(":memory:")
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.add_relation("APPLE", "is_a", "FRUIT")

    res = engine.query("APPLE", "FRUIT")
    assert res.mode == "FAST"
    assert res.path == ["APPLE", "FRUIT"]
    assert res.confidence >= 0.8
    assert "<think>" not in res.inspectable_trace


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
