"""Stable boundaries between perception, evidence, verification, and memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from little.core.models import current_iso_timestamp, generate_id


def _validate_probability(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")


class SourceType(str, Enum):
    USER = "user"
    DOCUMENT = "document"
    SENSOR = "sensor"
    TOOL = "tool"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONSISTENT = "inconsistent"


class CommitStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CandidateEntity:
    name: str
    type_hint: str | None = None
    span: tuple[int, int] | None = None
    probability: float = 0.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "probability")


@dataclass(frozen=True)
class CandidateClaim:
    subject: str
    predicate: str
    object: str
    probability: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    positive: bool = True
    is_property: bool = False

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "probability")
        if not self.subject.strip() or not self.predicate.strip() or not self.object.strip():
            raise ValueError("claim subject, predicate, and object must be non-empty")


@dataclass(frozen=True)
class CandidateAction:
    name: str
    arguments: dict[str, str]
    probability: float = 0.0

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "probability")
        if not self.name.strip():
            raise ValueError("action name must be non-empty")


ParsedQuery = tuple[str, str, Any]


@dataclass
class CandidateFrame:
    frame_id: str
    source_text: str
    intent: str
    claims: list[CandidateClaim]
    entities: list[CandidateEntity] = field(default_factory=list)
    actions: list[CandidateAction] = field(default_factory=list)
    uncertainty: float = 1.0
    model_id: str = "deterministic-parser"
    model_version: str = "builtin"
    accepted: bool = False
    parsed_query: ParsedQuery | None = None
    parsed_queries: list[ParsedQuery | None] = field(default_factory=list)
    question_parts: list[str] = field(default_factory=list)
    intent_probabilities: dict[str, float] = field(default_factory=dict)
    intent_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_probability(self.uncertainty, "uncertainty")
        if not self.source_text.strip():
            raise ValueError("source_text must be non-empty")
        if not isinstance(self.intent_probabilities, Mapping):
            raise TypeError("intent_probabilities must be a mapping")
        for label, probability in self.intent_probabilities.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("intent probability labels must be non-empty strings")
            _validate_probability(probability, f"intent probability for {label!r}")
        if self.intent_probabilities and not math.isclose(
            sum(self.intent_probabilities.values()),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("intent probabilities must sum to 1.0")
        if not isinstance(self.intent_alternatives, tuple):
            raise TypeError("intent_alternatives must be a tuple of strings")
        if any(
            not isinstance(label, str) or not label.strip()
            for label in self.intent_alternatives
        ):
            raise ValueError("intent alternatives must be non-empty strings")

    @classmethod
    def from_text(
        cls,
        source_text: str,
        claims: list[CandidateClaim],
        **kwargs: Any,
    ) -> CandidateFrame:
        return cls(
            frame_id=generate_id("frame"),
            source_text=source_text,
            intent=kwargs.pop("intent", "statement"),
            claims=claims,
            **kwargs,
        )


@dataclass
class EvidenceRecord:
    evidence_id: str
    subject_id: str
    predicate: str
    object_id: str | None
    source_type: SourceType
    source_reference: str
    source_text: str
    extraction_confidence: float
    positive: bool = True
    status: EvidenceStatus = EvidenceStatus.CANDIDATE
    derivation_proof_id: str | None = None
    observed_at: str = field(default_factory=current_iso_timestamp)
    created_at: str = field(default_factory=current_iso_timestamp)

    def __post_init__(self) -> None:
        _validate_probability(self.extraction_confidence, "extraction_confidence")
        if not self.subject_id.strip() or not self.predicate.strip():
            raise ValueError("evidence subject and predicate must be non-empty")
        if self.object_id is not None and not self.object_id.strip():
            raise ValueError("evidence object must be non-empty when provided")
        if not self.source_reference.strip():
            raise ValueError("source_reference must be non-empty")

    @classmethod
    def from_claim(
        cls,
        claim: CandidateClaim,
        source_type: SourceType,
        source_reference: str,
        source_text: str = "",
        positive: bool | None = None,
    ) -> EvidenceRecord:
        return cls(
            evidence_id=generate_id("evidence"),
            subject_id=claim.subject.strip().lower(),
            predicate=claim.predicate.strip().lower(),
            object_id=claim.object.strip().lower(),
            source_type=source_type,
            source_reference=source_reference,
            source_text=source_text,
            extraction_confidence=claim.probability,
            positive=claim.positive if positive is None else positive,
        )


@dataclass(frozen=True)
class ProofTrace:
    proof_id: str
    conclusion: dict[str, Any]
    premises: list[str] = field(default_factory=list)
    operations: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.UNKNOWN


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_commit_decision(self, evidence_id: str) -> CommitDecision:
        if self.status is VerificationStatus.PASSED:
            status = CommitStatus.ACCEPTED
        elif self.status is VerificationStatus.UNKNOWN:
            status = CommitStatus.UNKNOWN
        else:
            status = CommitStatus.REJECTED
        return CommitDecision(
            decision_id=generate_id("decision"),
            evidence_id=evidence_id,
            status=status,
            checks=dict(self.checks),
            reasons=list(self.reasons),
        )


@dataclass(frozen=True)
class CommitDecision:
    decision_id: str
    evidence_id: str
    status: CommitStatus
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def accepted(cls, evidence_id: str, verification: VerificationResult) -> CommitDecision:
        return verification.to_commit_decision(evidence_id)

    @classmethod
    def rejected(cls, evidence_id: str, verification: VerificationResult) -> CommitDecision:
        return verification.to_commit_decision(evidence_id)

    @classmethod
    def unknown(cls, evidence_id: str, verification: VerificationResult) -> CommitDecision:
        return verification.to_commit_decision(evidence_id)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result
