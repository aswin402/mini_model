"""Core domain models and data structures for the LITTLE cognitive architecture.

Defines the fundamental building blocks: Concepts, Entities, Relations,
Experiences, Beliefs, and Inference/Learning results.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ConceptStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANDIDATE = "CANDIDATE"
    HYPOTHESIS = "HYPOTHESIS"
    ARCHIVED = "ARCHIVED"


class BeliefStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    CONTRADICTED = "CONTRADICTED"


class UpdateType(str, Enum):
    NEW_CONCEPT = "NEW_CONCEPT"
    NEW_ENTITY = "NEW_ENTITY"
    NEW_RELATION = "NEW_RELATION"
    PROPERTY_UPDATE = "PROPERTY_UPDATE"
    EVIDENCE_ADDITION = "EVIDENCE_ADDITION"
    DUPLICATE = "DUPLICATE"
    CONTRADICTION = "CONTRADICTION"
    REVISION = "REVISION"
    NO_OP = "NO_OP"


def current_iso_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Concept:
    """A semantic abstraction representing a class of objects, properties, or ideas."""

    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    category: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    status: ConceptStatus = ConceptStatus.ACTIVE
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        category: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> Concept:
        clean_name = name.strip().lower()
        return cls(
            id=generate_id("concept"),
            name=clean_name,
            aliases=[a.strip().lower() for a in (aliases or [])],
            category=category.strip().lower() if category else None,
            attributes=attributes or {},
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class Entity:
    """A specific, grounded instance of a concept (e.g. 'my_apple_01', 'Alice')."""

    id: str
    name: str
    concept_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls, name: str, concept_id: str, properties: dict[str, Any] | None = None
    ) -> Entity:
        return cls(
            id=generate_id("entity"),
            name=name.strip(),
            concept_id=concept_id,
            properties=properties or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """A typed edge connecting two concepts or entities in the semantic world model.

    Supports positive and negative evidence accumulation based on NARS
    (Non-Axiomatic Reasoning) principles.
    """

    id: str
    subject_id: str
    predicate: str  # e.g., 'is_a', 'has_part', 'color', 'can_be', 'disjoint_with'
    object_id: str
    weight_positive: int = 1
    weight_negative: int = 0
    confidence: float = 0.5
    source_experience_id: str | None = None
    created_at: str = field(default_factory=current_iso_timestamp)

    def __post_init__(self) -> None:
        self.recompute_confidence()

    def recompute_confidence(self) -> float:
        total = self.weight_positive + self.weight_negative
        if total == 0:
            self.confidence = 0.0
        else:
            # Evidence-grounded confidence: c = w / (w + 1)
            self.confidence = round(self.weight_positive / (total + 1.0), 4)
        return self.confidence

    def add_evidence(self, positive: bool = True) -> None:
        if positive:
            self.weight_positive += 1
        else:
            self.weight_negative += 1
        self.recompute_confidence()

    @classmethod
    def create(
        cls,
        subject_id: str,
        predicate: str,
        object_id: str,
        source_experience_id: str | None = None,
        positive: bool = True,
    ) -> Relation:
        rel = cls(
            id=generate_id("rel"),
            subject_id=subject_id,
            predicate=predicate.strip().lower(),
            object_id=object_id,
            weight_positive=1 if positive else 0,
            weight_negative=0 if positive else 1,
            source_experience_id=source_experience_id,
        )
        rel.recompute_confidence()
        return rel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Experience:
    """Episodic record of a learning event or interaction."""

    id: str
    input_text: str
    extracted_triples: list[dict[str, Any]] = field(default_factory=list)
    source: str = "user"
    timestamp: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        input_text: str,
        extracted_triples: list[dict[str, Any]] | None = None,
        source: str = "user",
    ) -> Experience:
        return cls(
            id=generate_id("exp"),
            input_text=input_text.strip(),
            extracted_triples=extracted_triples or [],
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Belief:
    """An asserted or derived proposition accompanied by confidence and evidence trace."""

    proposition: str
    status: BeliefStatus
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class InferenceResult:
    """Result returned by the reasoning engine in response to a query."""

    query: str
    status: BeliefStatus
    answer: bool | str | None
    confidence: float
    evidence: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return self.status == BeliefStatus.SUPPORTED

    @property
    def is_unknown(self) -> bool:
        return self.status == BeliefStatus.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class LearningResult:
    """Result of processing a learning interaction."""

    input_text: str
    update_type: UpdateType
    experience_id: str
    concepts_created: list[str] = field(default_factory=list)
    relations_created: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["update_type"] = self.update_type.value
        return d


@dataclass
class Skill:
    """A deterministic procedure or computational capability stored in procedural memory."""

    id: str
    name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    code_body: str = ""
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        parameters: list[str],
        code_body: str,
        description: str = "",
    ) -> Skill:
        clean_name = name.strip().upper()
        return cls(
            id=f"skill_{clean_name.lower()}",
            name=clean_name,
            parameters=[p.strip() for p in parameters],
            code_body=code_body.strip(),
            description=description.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicState:
    """Continuous physical state of an entity governed by continuous-time dynamical ODEs."""

    entity_id: str
    freshness: float = 1.0  # 1.0 = completely fresh, 0.0 = completely decayed
    oxidation: float = 0.0  # 0.0 = unexposed, 1.0 = fully oxidized/brown
    temperature: float = 20.0  # Celsius
    time_constant: float = 3600.0  # tau in seconds for decay/oxidation
    last_updated: str = field(default_factory=current_iso_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Construction:
    """A Construction Grammar pairing of form (token pattern) and meaning (semantic frame).

    Grammar rules are stored in memory, not hardcoded into engine source code.
    """

    id: str
    name: str
    pattern_tokens: list[str]  # e.g. ["{X}", "is", "a", "{Y}"]
    slot_roles: dict[str, str]  # e.g. {"X": "subject", "Y": "object"}
    predicate_template: str  # e.g. "is_a", "disjoint_with", "part_of"
    construction_type: str = "statement"  # "statement", "question", "action"
    is_negative: bool = False
    is_property: bool = False
    confidence: float = 1.0
    evidence_positive: int = 1
    evidence_negative: int = 0
    created_at: str = field(default_factory=current_iso_timestamp)

    @classmethod
    def create(
        cls,
        name: str,
        pattern_tokens: list[str],
        slot_roles: dict[str, str],
        predicate_template: str,
        construction_type: str = "statement",
        is_negative: bool = False,
        is_property: bool = False,
        confidence: float = 1.0,
    ) -> Construction:
        return cls(
            id=generate_id("cxn"),
            name=name.strip().lower(),
            pattern_tokens=[t.strip().lower() for t in pattern_tokens],
            slot_roles={k.strip(): v.strip().lower() for k, v in slot_roles.items()},
            predicate_template=predicate_template.strip().lower(),
            construction_type=construction_type.strip().lower(),
            is_negative=is_negative,
            is_property=is_property,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
