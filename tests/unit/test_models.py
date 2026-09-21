"""Unit tests for LITTLE core domain models."""

from little.core.models import (
    BeliefStatus,
    Concept,
    ConceptStatus,
    Entity,
    Experience,
    InferenceResult,
    Relation,
)


def test_concept_creation() -> None:
    c = Concept.create("Apple", category="Fruit", aliases=["Red Delicious"])
    assert c.name == "apple"
    assert c.category == "fruit"
    assert c.aliases == ["red delicious"]
    assert c.status == ConceptStatus.ACTIVE
    assert c.id.startswith("concept_")


def test_relation_evidence_and_confidence() -> None:
    rel = Relation.create("concept_dog", "is_a", "concept_animal", positive=True)
    assert rel.weight_positive == 1
    assert rel.weight_negative == 0
    # Confidence = w+ / (w+ + w- + 1) = 1 / 2 = 0.5
    assert rel.confidence == 0.5

    # Reinforce with positive evidence
    rel.add_evidence(positive=True)
    # w+ = 2, total = 2, conf = 2 / 3 = 0.6667
    assert rel.confidence == 0.6667

    # Add negative evidence
    rel.add_evidence(positive=False)
    # w+ = 2, w- = 1, total = 3, conf = 2 / 4 = 0.5
    assert rel.confidence == 0.5


def test_entity_creation() -> None:
    ent = Entity.create(
        "my_dog_01", concept_id="concept_dog", properties={"breed": "Golden Retriever"}
    )
    assert ent.name == "my_dog_01"
    assert ent.properties["breed"] == "Golden Retriever"


def test_experience_creation() -> None:
    exp = Experience.create(
        "A dog is an animal.",
        extracted_triples=[{"subject": "dog", "predicate": "is_a", "object": "animal"}],
    )
    assert exp.input_text == "A dog is an animal."
    assert len(exp.extracted_triples) == 1
    assert exp.id.startswith("exp_")


def test_inference_result_properties() -> None:
    supported = InferenceResult(
        query="test", status=BeliefStatus.SUPPORTED, answer=True, confidence=0.9
    )
    assert supported.is_supported is True
    assert supported.is_unknown is False

    unknown = InferenceResult(
        query="test", status=BeliefStatus.UNKNOWN, answer=None, confidence=0.1
    )
    assert unknown.is_supported is False
    assert unknown.is_unknown is True
