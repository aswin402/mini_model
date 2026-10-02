"""Unit tests for Adaptive Depth Pondering (PonderNet Halting) in DualSpeedInfillingEngine."""

from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.memory.store import MemoryStore


def test_adaptive_pondering_early_halt_on_collision():
    memory = MemoryStore(":memory:", seed_ontology=False)
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    # 2-hop connection: CHEETAH -> FELINE -> ANIMAL
    memory.get_or_create_concept("CHEETAH", category="animal")
    memory.get_or_create_concept("FELINE", category="animal")
    memory.get_or_create_concept("ANIMAL", category="living_thing")

    memory.add_relation("CHEETAH", "is_a", "FELINE")
    memory.add_relation("FELINE", "is_a", "ANIMAL")

    res = engine.query("CHEETAH", "ANIMAL")

    assert res.mode == "THINKING"
    assert res.path == ["CHEETAH", "FELINE", "ANIMAL"]
    # Verify it halted early at depth 1 or 2 instead of max_depth 8
    assert res.ponder_depth <= 2
    assert res.halt_reason == "proof_closed_verified"
    assert "<ponder" in res.inspectable_trace
    assert "halt=\"true\"" in res.inspectable_trace


def test_adaptive_pondering_halt_on_frontier_exhausted():
    memory = MemoryStore(":memory:", seed_ontology=False)
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.get_or_create_concept("TIGER", category="animal")
    memory.get_or_create_concept("SUBMARINE", category="vehicle")

    res = engine.query("TIGER", "SUBMARINE")

    assert res.mode == "THINKING"
    assert res.path == []
    assert res.confidence == 0.0
    # Frontiers empty immediately at depth 1
    assert res.ponder_depth == 1
    assert res.halt_reason == "frontier_exhausted"
    assert "<ponder depth=\"1\"" in res.inspectable_trace
    assert "reason=\"frontier_exhausted\"" in res.inspectable_trace


def test_adaptive_pondering_fast_reflex_mode():
    memory = MemoryStore(":memory:", seed_ontology=False)
    verifier = DeepSeekInvariantVerifier(memory)
    engine = DualSpeedInfillingEngine(memory, verifier)

    memory.get_or_create_concept("DOG", category="animal")
    memory.get_or_create_concept("MAMMAL", category="animal")
    memory.add_relation("DOG", "is_a", "MAMMAL")

    res = engine.query("DOG", "MAMMAL")

    assert res.mode == "FAST"
    assert res.ponder_depth == 1
    assert res.halt_reason == "direct_reflex_hit"
    assert res.path == ["DOG", "MAMMAL"]
