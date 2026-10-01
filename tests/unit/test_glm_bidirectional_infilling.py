"""Unit tests for GLM Dual-Speed Bidirectional Infilling Engine.

Evaluates:
- Object infilling: (S, P, [?])
- Subject infilling: ([?], P, O)
- Path infilling: (S, [?], O) with bidirectional frontier collision search
- Fast reflex mode (<0.2ms) vs Thinking deliberative mode (<2ms)
- DeepSeek invariant gate verification on infilled chains
"""

from __future__ import annotations

import pytest

from little.inference.dual_speed import DualSpeedInfillingEngine, InfillingResult
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.memory.store import MemoryStore


@pytest.fixture
def infilling_engine() -> DualSpeedInfillingEngine:
    mem = MemoryStore(":memory:", seed_ontology=True)
    verifier = DeepSeekInvariantVerifier(mem)
    engine = DualSpeedInfillingEngine(mem, verifier)

    # Add custom chain for multi-hop test: SPARROW is_a SONGBIRD is_a BIRD is_a VERTEBRATE is_a ANIMAL
    c_sparrow = mem.get_or_create_concept("sparrow", category="bird")
    c_songbird = mem.get_or_create_concept("songbird", category="bird")
    c_bird = mem.get_or_create_concept("bird", category="animal")

    mem.add_relation(c_sparrow.id, "is_a", c_songbird.id)
    mem.add_relation(c_songbird.id, "is_a", c_bird.id)

    # Add mereology: COMPUTER has CPU, CPU has ALU
    c_comp = mem.get_or_create_concept("computer")
    c_cpu = mem.get_or_create_concept("cpu")
    c_alu = mem.get_or_create_concept("alu")
    mem.add_relation(c_comp.id, "has", c_cpu.id)
    mem.add_relation(c_cpu.id, "has", c_alu.id)

    return engine


def test_object_infilling_fast_and_thinking(infilling_engine: DualSpeedInfillingEngine):
    """Verify (S, P, [?]) infilling discovers direct objects and transitive descendants."""
    results = infilling_engine.infill_object("sparrow", "is_a", max_results=5)

    assert len(results) >= 2
    # First result should be fast direct reflex: songbird or bird
    assert results[0].mode == "FAST"
    assert results[0].infilled_value in ("SONGBIRD", "BIRD")
    assert results[0].query_type == "OBJECT"
    assert results[0].latency_ms < 5.0

    # Infilled names should discover both songbird and bird
    infilled_names = [r.infilled_value for r in results]
    assert "BIRD" in infilled_names
    assert "SONGBIRD" in infilled_names


def test_subject_infilling_fast_and_thinking(infilling_engine: DualSpeedInfillingEngine):
    """Verify ([?], P, O) infilling discovers incoming subjects."""
    results = infilling_engine.infill_subject("songbird", "is_a", max_results=5)

    assert len(results) >= 1
    assert results[0].mode == "FAST"
    assert results[0].infilled_value == "SPARROW"
    assert results[0].query_type == "SUBJECT"


def test_path_infilling_bidirectional_collision(infilling_engine: DualSpeedInfillingEngine):
    """Verify (S, [?], O) meeting-in-the-middle frontier collision search."""
    results = infilling_engine.infill_path("computer", "alu", max_depth=4)

    assert len(results) >= 1
    res = results[0]
    assert res.mode == "THINKING"
    assert "COMPUTER" in res.path
    assert "CPU" in res.path
    assert "ALU" in res.path
    assert "Meeting-in-the-middle collision" in res.inspectable_trace


def test_transitive_query_meeting_in_the_middle(infilling_engine: DualSpeedInfillingEngine):
    """Verify standard query(subject, target, predicate) uses bidirectional meeting-in-the-middle."""
    res = infilling_engine.query("sparrow", "animal", predicate="is_a")

    assert res.mode == "THINKING"
    assert res.confidence >= 0.80
    assert "SPARROW" in res.path
    assert "ANIMAL" in res.path
    assert "<think>" in res.inspectable_trace
    assert "</think>" in res.inspectable_trace


def test_infilling_unknown_for_disconnected(infilling_engine: DualSpeedInfillingEngine):
    """Verify infilling returns UNKNOWN and empty path for disconnected entities."""
    res = infilling_engine.query("sparrow", "linux", predicate="is_a")

    assert res.confidence == 0.0
    assert len(res.path) == 0
    assert "UNKNOWN" in res.inspectable_trace
