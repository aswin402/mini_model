from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    EvidenceRecord,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.memory.store import MemoryStore


def test_memory_store_creates_evidence_tables_without_deleting_graph_data():
    store = MemoryStore(":memory:", seed_ontology=False)
    subject = store.create_concept("falcon")
    object_ = store.create_concept("bird")
    store.add_relation(subject.id, "is_a", object_.id)

    assert store.count_relations() == 1
    assert store.has_evidence_tables()


def test_evidence_round_trip_preserves_provenance():
    store = MemoryStore(":memory:", seed_ontology=False)
    evidence = EvidenceRecord.from_claim(
        CandidateClaim("falcon", "is_a", "bird", probability=0.98),
        source_type=SourceType.USER,
        source_reference="conversation:test-2",
    )

    saved = store.evidence.append(evidence)
    loaded = store.evidence.get(saved.evidence_id)

    assert loaded is not None
    assert loaded.source_reference == "conversation:test-2"
    assert loaded.status.value == "candidate"


def test_unverified_candidate_does_not_create_relation():
    store = MemoryStore(":memory:", seed_ontology=False)
    ledger = store.ledger
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )

    evidence = ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="conversation:test-3",
    )
    decision = ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.UNKNOWN,
            checks={"grounded": False},
            reasons=["object is unresolved"],
        ),
    )

    assert decision.status is CommitStatus.UNKNOWN
    assert store.find_relation_by_names("falcon", "is_a", "bird") is None


def test_verified_candidate_creates_relation_and_decision():
    store = MemoryStore(":memory:", seed_ontology=False)
    ledger = store.ledger
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )
    evidence = ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="conversation:test-4",
    )
    decision = ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.PASSED,
            checks={"grounded": True, "schema": True},
            reasons=["terms resolved"],
        ),
    )

    assert decision.status is CommitStatus.ACCEPTED
    assert store.find_relation_by_names("falcon", "is_a", "bird") is not None


def test_verified_negative_candidate_adds_opposition_evidence():
    store = MemoryStore(":memory:", seed_ontology=False)
    ledger = store.ledger
    frame = CandidateFrame.from_text(
        "A penguin cannot fly.",
        claims=[CandidateClaim("penguin", "can", "fly", probability=0.98)],
    )
    evidence = ledger.propose_relation(
        frame,
        frame.claims[0],
        source_type=SourceType.USER,
        source_reference="conversation:test-5",
        positive=False,
    )
    decision = ledger.commit_relation(
        evidence,
        VerificationResult(
            status=VerificationStatus.PASSED,
            checks={"grounded": True, "schema": True},
            reasons=["terms resolved"],
        ),
        positive=False,
    )

    assert decision.status is CommitStatus.ACCEPTED
    relation = store.find_relation_by_names("penguin", "can", "fly")
    assert relation is not None
    assert relation.weight_negative == 1
