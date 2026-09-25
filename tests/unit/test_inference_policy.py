import pytest

from little.inference.engine import InferenceEngine
from little.inference.reasoning_policy import ReasoningPolicy
from little.memory.store import MemoryStore


def test_inference_engine_uses_injected_reasoning_policy():
    memory = MemoryStore(":memory:", seed_ontology=False)
    child = memory.create_concept("child")
    parent = memory.create_concept("parent")
    capability = memory.create_concept("capability")
    unrelated = memory.create_concept("unrelated")
    memory.add_relation(child.id, "is_a", parent.id)
    memory.add_relation(parent.id, "can", capability.id)

    policy = ReasoningPolicy(
        version="test",
        max_depth=3,
        fast_confidence=0.71,
        verified_confidence=0.82,
        unknown_confidence=0.13,
        inheritance_confidence_factor=0.5,
        inference_unknown_confidence=0.23,
    )
    engine = InferenceEngine(memory, policy=policy)

    inherited = engine.infer("child", "can", "capability")
    unknown = engine.infer("child", "can", unrelated.name)

    assert inherited.is_supported is True
    assert inherited.confidence == pytest.approx(0.25)
    assert unknown.confidence == pytest.approx(0.23)
