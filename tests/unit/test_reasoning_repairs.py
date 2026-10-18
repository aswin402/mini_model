from dataclasses import replace

import pytest

from little.core.models import BeliefStatus
from little.inference.dual_speed import DualSpeedInfillingEngine
from little.inference.engine import InferenceEngine
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.inference.reasoning_policy import ReasoningPolicy
from little.inference.thinking_controller import ThinkingController


def graph(memory):
    for name in ("penguin", "bird", "animal"):
        memory.create_concept(name)
    for s, o in [("penguin", "bird"), ("bird", "animal")]:
        memory.add_relation(memory.get_concept(s).id, "is_a", memory.get_concept(o).id)


@pytest.fixture
def memory():
    from little.memory.store import MemoryStore

    store = MemoryStore(":memory:", seed_ontology=False)
    yield store
    store.close()


def test_direct_opposition_and_transitive_proof_are_contradicted(memory):
    graph(memory)
    memory.add_relation(
        memory.get_concept("penguin").id,
        "is_a",
        memory.get_concept("animal").id,
        positive=False,
    )
    result = InferenceEngine(memory).infer("penguin", "is_a", "animal")
    assert result.status is BeliefStatus.CONTRADICTED
    assert result.answer is None
    assert any("Chain:" in e for e in result.evidence)
    assert any("NOT" in e for e in result.evidence)
    controller = ThinkingController(memory).run(
        "Is penguin an animal?", parsed_query=("penguin", "is_a", "animal")
    )
    assert controller.inference.status is BeliefStatus.CONTRADICTED


def test_negative_taxonomy_does_not_create_cycle(memory):
    memory.add_relation("B", "is_a", "A", positive=False)
    assert DeepSeekInvariantVerifier(memory).verify_relation("A", "is_a", "B").passed


def test_schema_checks_actual_concept_type(memory):
    memory.create_concept("event", category="event")
    memory.create_concept("thing")
    result = DeepSeekInvariantVerifier(memory).verify_relation("event", "is_a", "thing")
    assert not result.passed
    assert result.violated_gate == "I_SORT"


def test_structural_assertion_is_not_claimed_as_factual_grounding(memory):
    verifier = DeepSeekInvariantVerifier(memory)
    result = verifier.verify_relation("apple", "is_a", "fruit")
    assert result.passed
    assert any("not assessed" in line for line in result.proof_trace)
    assert not verifier.verify_proof_chain([("apple", "is_a", "fruit")]).passed


@pytest.mark.parametrize("depth", [1])
def test_max_depth_counts_graph_hops(memory, depth):
    graph(memory)
    assert (
        InferenceEngine(memory, max_depth=depth)
        .infer("penguin", "is_a", "animal")
        .status
        is BeliefStatus.UNKNOWN
    )
    dual = DualSpeedInfillingEngine(
        memory,
        DeepSeekInvariantVerifier(memory),
        replace(ReasoningPolicy.default(), max_depth=depth),
    )
    assert not dual.query("penguin", "animal").path
    assert not dual.infill_path("penguin", "animal", max_depth=depth)


def test_confidence_is_product_of_premise_evidence(memory):
    graph(memory)
    dual = DualSpeedInfillingEngine(memory, DeepSeekInvariantVerifier(memory))
    premises = memory.get_relations(predicate="is_a")
    assert dual.query("penguin", "animal").confidence == pytest.approx(
        premises[0].confidence * premises[1].confidence
    )


def test_fast_reflex_does_not_bypass_invariants(memory):
    memory.add_relation("APPLE", "is_a", "ANIMAL")
    memory.add_relation("APPLE", "disjoint_with", "ANIMAL", positive=False)
    result = DualSpeedInfillingEngine(memory, DeepSeekInvariantVerifier(memory)).query(
        "APPLE", "ANIMAL"
    )
    assert not result.path
    assert "I_MUTEX" in result.inspectable_trace


def test_nontransitive_predicate_does_not_close_multihop_proof(memory):
    memory.add_relation("A", "eats", "B")
    memory.add_relation("B", "eats", "C")
    dual = DualSpeedInfillingEngine(memory, DeepSeekInvariantVerifier(memory))
    assert not dual.query("A", "C", predicate="eats").path


def test_path_enumeration_checks_each_premise_invariant(memory):
    memory.add_relation("APPLE", "is_a", "ANIMAL")
    memory.add_relation("APPLE", "disjoint_with", "ANIMAL", positive=False)
    dual = DualSpeedInfillingEngine(memory, DeepSeekInvariantVerifier(memory))
    assert not dual.infill_path("APPLE", "ANIMAL", max_depth=1)
