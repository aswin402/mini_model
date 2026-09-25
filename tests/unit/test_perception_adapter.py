import json

import pytest

from little.core.contracts import CandidateClaim, EvidenceRecord, SourceType
from little.language.perception import DeterministicPerceptionAdapter, PerceptionPolicy
from little.memory.store import MemoryStore


def test_perception_returns_statement_candidates_without_mutating_memory():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = DeterministicPerceptionAdapter()
    before = (
        len(memory.list_concepts()),
        memory.count_relations(),
        [(c.id, c.confidence, c.evidence_positive, c.evidence_negative)
         for c in memory.list_constructions()],
    )

    frame = adapter.perceive("A falcon is a bird.", memory=memory)

    after = (
        len(memory.list_concepts()),
        memory.count_relations(),
        [(c.id, c.confidence, c.evidence_positive, c.evidence_negative)
         for c in memory.list_constructions()],
    )
    assert frame.intent == "statement"
    assert [(c.subject, c.predicate, c.object) for c in frame.claims] == [
        ("falcon", "is_a", "bird")
    ]
    assert before == after
    assert frame.model_id == "deterministic-parser"


def test_perception_marks_questions_without_creating_claims():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "Is a falcon a bird?", memory=memory
    )

    assert frame.intent == "question"
    assert frame.claims == []
    assert frame.parsed_query == ("falcon", "is_a", "bird")
    assert memory.count_relations() == 0


def test_perception_carries_structured_compound_question_parts():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "Can an eagle fly and does it have wings?", memory=memory
    )

    assert frame.intent == "question"
    assert frame.question_parts == ["Can an eagle fly", "does it have wings?"]
    assert frame.parsed_queries == [
        ("eagle", "can", "fly"),
        ("eagle", "has", "wing"),
    ]
    assert frame.parsed_query == frame.parsed_queries[0]


def test_perception_carries_questions_across_sentence_boundaries():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "Can an eagle fly? Does it have wings?", memory=memory
    )

    assert frame.intent == "question"
    assert frame.question_parts == ["Can an eagle fly", "Does it have wings"]
    assert frame.parsed_queries == [
        ("eagle", "can", "fly"),
        ("eagle", "has", "wing"),
    ]


def test_unparsed_question_preserves_explicit_missing_query():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive("Why?", memory=memory)

    assert frame.intent == "question"
    assert frame.parsed_query is None


def test_perception_preserves_negative_and_property_metadata():
    memory = MemoryStore(":memory:", seed_ontology=False)

    negative = DeterministicPerceptionAdapter().perceive(
        "A bird is not a vehicle.", memory=memory
    )
    property_frame = DeterministicPerceptionAdapter().perceive(
        "The apple is red.", memory=memory
    )

    assert negative.claims[0].positive is False
    assert property_frame.claims[0].is_property is True


def test_perception_assigns_negation_per_claim_in_mixed_sentence():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "A bird is not a vehicle, but a falcon is a bird.", memory=memory
    )

    assert [(claim.subject, claim.positive) for claim in frame.claims] == [
        ("bird", False),
        ("falcon", True),
    ]


def test_unrecognized_input_is_unknown_and_pure():
    memory = MemoryStore(":memory:", seed_ontology=False)
    frame = DeterministicPerceptionAdapter().perceive("qwerty zorp", memory=memory)

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert memory.count_relations() == 0


def test_policy_load_rejects_missing_and_out_of_range_values(tmp_path):
    policy_path = tmp_path / "perception_policy.json"
    policy_path.write_text(json.dumps({"perception_policy": {"version": "1"}}))
    with pytest.raises(ValueError, match="deterministic_claim_probability"):
        PerceptionPolicy.load(tmp_path)

    policy_path.write_text(json.dumps({"perception_policy": {
        "version": "1",
        "deterministic_claim_probability": 1.1,
        "deterministic_frame_uncertainty": 0.01,
        "question_frame_uncertainty": 0.25,
        "unknown_frame_uncertainty": 1.0,
        "action_probability": 0.95,
    }}))
    with pytest.raises(ValueError, match="deterministic_claim_probability"):
        PerceptionPolicy.load(tmp_path)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_policy_load_rejects_non_finite_probability(tmp_path, non_finite):
    policy_path = tmp_path / "perception_policy.json"
    policy_path.write_text(json.dumps({"perception_policy": {
        "version": "1",
        "deterministic_claim_probability": non_finite,
        "deterministic_frame_uncertainty": 0.01,
        "question_frame_uncertainty": 0.25,
        "unknown_frame_uncertainty": 1.0,
        "action_probability": 0.95,
    }}))

    with pytest.raises(ValueError, match="finite"):
        PerceptionPolicy.load(tmp_path)


def test_context_is_used_to_resolve_anaphora_without_changing_source_text():
    from little.language.dialogue import DialogueContext

    memory = MemoryStore(":memory:", seed_ontology=False)
    context = DialogueContext()
    context.register_entity("falcon")

    frame = DeterministicPerceptionAdapter().perceive(
        "It is a bird.", memory=memory, context=context
    )

    assert [(c.subject, c.predicate, c.object) for c in frame.claims] == [
        ("falcon", "is_a", "bird")
    ]
    assert frame.source_text == "It is a bird."


def test_paragraph_perception_resolves_pronouns_within_one_frame():
    memory = MemoryStore(":memory:", seed_ontology=False)
    frame = DeterministicPerceptionAdapter().perceive(
        "A golden retriever is a dog. It has thick fur. It can swim.",
        memory=memory,
    )

    assert [(c.subject, c.predicate, c.object) for c in frame.claims] == [
        ("golden retriever", "is_a", "dog"),
        ("golden retriever", "has", "thick fur"),
        ("golden retriever", "can", "swim"),
    ]
    assert memory.count_relations() == 0


def test_paragraph_pronouns_prefer_the_paragraph_subject_over_older_dialogue():
    from little.language.dialogue import DialogueContext

    memory = MemoryStore(":memory:", seed_ontology=False)
    context = DialogueContext()
    context.register_entity("dog", role="subject")

    frame = DeterministicPerceptionAdapter().perceive(
        "A cat is a mammal. It is black.", memory=memory, context=context
    )

    assert [(c.subject, c.predicate, c.object) for c in frame.claims] == [
        ("cat", "is_a", "mammal"),
        ("cat", "color", "black"),
    ]


def test_negative_claim_metadata_flows_to_evidence_unless_overridden():
    claim = CandidateClaim("bird", "disjoint_with", "vehicle", 0.99, {}, False)
    evidence = EvidenceRecord.from_claim(
        claim, SourceType.USER, "conversation:test-negative"
    )
    overridden = EvidenceRecord.from_claim(
        claim, SourceType.USER, "conversation:test-positive", positive=True
    )

    assert evidence.positive is False
    assert overridden.positive is True
