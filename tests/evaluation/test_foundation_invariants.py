from pathlib import Path

from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.knowledge.registry import SchemaRegistry
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_split_schema_validates_multiple_material_records_without_special_cases():
    registry = SchemaRegistry.load(Path("data/schemas"))

    results = [
        registry.validate_action("split", {"object": "entity", "count": "quantity"})
        for _material in ("organic", "grain", "metal")
    ]

    assert all(result.valid for result in results)
    assert registry.action("split") is not None
    assert "mass_conservation" in registry.action("split").invariants


def test_unknown_relation_never_becomes_an_accepted_edge():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "A falcon has an invented relation to a bird.",
        claims=[
            CandidateClaim(
                "falcon", "invented_predicate", "bird", probability=0.99
            )
        ],
    )
    evidence = store.ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="evaluation:unknown-relation",
    )

    decision = store.ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.UNKNOWN,
            checks={"schema": False},
            reasons=["unregistered predicate"],
        ),
    )

    assert decision.status is CommitStatus.UNKNOWN
    assert store.count_relations() == 0


def test_accepted_relation_has_provenance_in_temporary_memory():
    store = MemoryStore(":memory:", seed_ontology=False)
    LearningEngine(store).learn("A falcon is a bird.")

    accepted = [record for record in store.evidence.list_by_source("user") if record.status.value == "accepted"]

    assert len(accepted) == 1
    assert accepted[0].source_text == "A falcon is a bird."
    assert accepted[0].source_reference.startswith("frame:")
