import pytest

from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    CommitStatus,
    EvidenceRecord,
    EvidenceStatus,
    SourceType,
    VerificationResult,
    VerificationStatus,
)


def test_candidate_frame_is_not_an_accepted_evidence_record():
    frame = CandidateFrame.from_text(
        "A falcon is a bird.",
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.98)],
    )

    assert frame.claims[0].predicate == "is_a"
    assert frame.claims[0].probability == 0.98
    assert frame.accepted is False


def test_evidence_record_requires_explicit_source_and_status():
    evidence = EvidenceRecord.from_claim(
        CandidateClaim("falcon", "is_a", "bird", probability=0.98),
        source_type=SourceType.USER,
        source_reference="conversation:test-1",
    )

    assert evidence.status is EvidenceStatus.CANDIDATE
    assert evidence.source_type is SourceType.USER
    assert evidence.source_reference == "conversation:test-1"


def test_commit_decision_serializes_verification_result():
    verification = VerificationResult(
        status=VerificationStatus.PASSED,
        checks={"grounded": True, "schema": True},
        reasons=["All terms resolved"],
    )

    decision = verification.to_commit_decision("evidence_1")

    assert decision.status is CommitStatus.ACCEPTED
    assert decision.evidence_id == "evidence_1"
    assert decision.checks == {"grounded": True, "schema": True}


def test_contract_probability_and_uncertainty_are_bounded():
    with pytest.raises(ValueError):
        CandidateClaim("falcon", "is_a", "bird", probability=1.1)

    with pytest.raises(ValueError):
        CandidateFrame.from_text("fact", [], uncertainty=-0.1)
