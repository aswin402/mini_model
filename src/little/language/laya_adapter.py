"""Optional typed Laya perception boundary with no runtime dependency."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

from little.core.contracts import (
    CandidateAction,
    CandidateClaim,
    CandidateEntity,
    CandidateFrame,
    ParsedQuery,
)

if TYPE_CHECKING:
    from little.language.dialogue import DialogueContext
    from little.memory.store import MemoryStore


def _probability(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return float(value)


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


def _freeze(value: Any) -> Any:
    """Recursively freeze backend-owned containers at the model boundary."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


_PAYLOAD_FIELDS = (
    "claims",
    "entities",
    "actions",
    "parsed_query",
    "parsed_queries",
    "question_parts",
)
_PAYLOAD_FIELD_SET = frozenset(_PAYLOAD_FIELDS)


@dataclass(frozen=True)
class LayaAdapterPolicy:
    """Data-controlled translation policy for external model decisions."""

    version: str
    fallback_intent: str
    rejected_model_ids: tuple[str, ...]
    intent_mapping: Mapping[str, str]
    candidate_requirements: Mapping[str, tuple[str, ...]]
    allowed_payloads: Mapping[str, tuple[str, ...]]
    structured_query_predicate_prefixes: tuple[str, ...]

    @classmethod
    def load(cls, directory: Path) -> LayaAdapterPolicy:
        path = Path(directory) / "laya_adapter_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = payload["laya_adapter_policy"]
            if not isinstance(data, dict):
                raise TypeError("laya_adapter_policy must be an object")
            version = _non_empty_string(data["version"], "version")
            fallback_intent = _non_empty_string(
                data["fallback_intent"], "fallback_intent"
            )

            raw_rejected = data["rejected_model_ids"]
            rejected_model_ids = tuple(
                _non_empty_string(value, "rejected_model_ids entry")
                for value in _sequence(raw_rejected, "rejected_model_ids")
            )

            raw_mapping = data["intent_mapping"]
            if not isinstance(raw_mapping, dict) or not raw_mapping:
                raise ValueError("intent_mapping must be a non-empty object")
            intent_mapping = {
                _non_empty_string(key, "intent mapping key"): _non_empty_string(
                    value, f"intent mapping for {key!r}"
                )
                for key, value in raw_mapping.items()
            }

            raw_requirements = data["candidate_requirements"]
            if not isinstance(raw_requirements, dict):
                raise TypeError("candidate_requirements must be an object")
            candidate_requirements: dict[str, tuple[str, ...]] = {}
            for intent, raw_fields in raw_requirements.items():
                canonical_intent = _non_empty_string(
                    intent, "candidate requirement intent"
                )
                requirement_fields = tuple(
                    _non_empty_string(value, "candidate requirement field")
                    for value in _sequence(
                        raw_fields, f"candidate_requirements[{canonical_intent!r}]"
                    )
                )
                unknown_fields = set(requirement_fields) - _PAYLOAD_FIELD_SET
                if unknown_fields:
                    raise ValueError(
                        "candidate requirement fields are not supported payload fields: "
                        + ", ".join(sorted(unknown_fields))
                    )
                candidate_requirements[canonical_intent] = requirement_fields

            raw_allowed = data["allowed_payloads"]
            if not isinstance(raw_allowed, dict):
                raise TypeError("allowed_payloads must be an object")
            allowed_payloads: dict[str, tuple[str, ...]] = {}
            for intent, raw_fields in raw_allowed.items():
                canonical_intent = _non_empty_string(
                    intent, "allowed payload intent"
                )
                payload_fields = tuple(
                    _non_empty_string(value, "allowed payload field")
                    for value in _sequence(
                        raw_fields, f"allowed_payloads[{canonical_intent!r}]"
                    )
                )
                unknown_fields = set(payload_fields) - _PAYLOAD_FIELD_SET
                if unknown_fields:
                    raise ValueError(
                        "allowed payload fields are not supported payload fields: "
                        + ", ".join(sorted(unknown_fields))
                    )
                allowed_payloads[canonical_intent] = payload_fields

            if fallback_intent not in candidate_requirements:
                raise ValueError(
                    "candidate_requirements must define fallback_intent"
                )
            if fallback_intent not in allowed_payloads:
                raise ValueError("allowed_payloads must define fallback_intent")
            missing_requirements = set(intent_mapping.values()) - set(
                candidate_requirements
            )
            if missing_requirements:
                raise ValueError(
                    "candidate_requirements missing mapped intents: "
                    + ", ".join(sorted(missing_requirements))
                )
            missing_payloads = set(intent_mapping.values()) - set(allowed_payloads)
            if missing_payloads:
                raise ValueError(
                    "allowed_payloads missing mapped intents: "
                    + ", ".join(sorted(missing_payloads))
                )
            raw_prefixes = data["structured_query_predicate_prefixes"]
            structured_prefixes = tuple(
                _non_empty_string(value, "structured query predicate prefix")
                for value in _sequence(
                    raw_prefixes, "structured_query_predicate_prefixes"
                )
            )
            for intent, requirements in candidate_requirements.items():
                disallowed_requirements = set(requirements) - set(
                    allowed_payloads.get(intent, ())
                )
                if disallowed_requirements:
                    raise ValueError(
                        "candidate requirements are not allowed payloads for "
                        f"{intent!r}: "
                        + ", ".join(sorted(disallowed_requirements))
                    )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Laya adapter policy at {path}: {exc}") from exc

        return cls(
            version=version,
            fallback_intent=fallback_intent,
            rejected_model_ids=rejected_model_ids,
            intent_mapping=MappingProxyType(intent_mapping),
            candidate_requirements=MappingProxyType(candidate_requirements),
            allowed_payloads=MappingProxyType(allowed_payloads),
            structured_query_predicate_prefixes=structured_prefixes,
        )

    @classmethod
    def default(cls) -> LayaAdapterPolicy:
        return cls.load(RuntimePaths.default().schema_directory)


@dataclass(frozen=True)
class LayaDecision:
    """Immutable, dependency-free output contract for an injected Laya runtime."""

    intent: str
    uncertainty: float
    model_id: str
    model_version: str
    intent_probabilities: Mapping[str, float]
    intent_alternatives: tuple[str, ...] = ()
    claims: tuple[CandidateClaim, ...] = ()
    entities: tuple[CandidateEntity, ...] = ()
    actions: tuple[CandidateAction, ...] = ()
    parsed_query: ParsedQuery | None = None
    parsed_queries: tuple[ParsedQuery | None, ...] = ()
    question_parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        intent = _non_empty_string(self.intent, "intent")
        model_id = _non_empty_string(self.model_id, "model_id")
        model_version = _non_empty_string(self.model_version, "model_version")
        uncertainty = _probability(self.uncertainty, "uncertainty")
        if not isinstance(self.intent_probabilities, Mapping):
            raise TypeError("intent_probabilities must be a mapping")

        probabilities: dict[str, float] = {}
        for label, value in self.intent_probabilities.items():
            clean_label = _non_empty_string(label, "intent probability label")
            probabilities[clean_label] = _probability(
                value, f"intent probability for {clean_label!r}"
            )
        if not probabilities:
            raise ValueError("intent_probabilities must be non-empty")
        if intent not in probabilities:
            raise ValueError("intent_probabilities must include the selected intent")
        if not math.isclose(
            sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("intent probabilities must sum to 1.0")

        alternatives = tuple(
            _non_empty_string(value, "intent alternative")
            for value in _sequence(self.intent_alternatives, "intent_alternatives")
        )
        claims = tuple(
            CandidateClaim(
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.object,
                probability=claim.probability,
                attributes=_freeze(claim.attributes),
                positive=claim.positive,
                is_property=claim.is_property,
            )
            for claim in self._typed_items(self.claims, CandidateClaim, "claims")
        )
        entities = tuple(
            CandidateEntity(
                name=entity.name,
                type_hint=entity.type_hint,
                span=self._span(entity.span),
                probability=entity.probability,
            )
            for entity in self._typed_items(self.entities, CandidateEntity, "entities")
        )
        actions = tuple(
            CandidateAction(
                name=action.name,
                arguments=_freeze(action.arguments),
                probability=action.probability,
            )
            for action in self._typed_items(self.actions, CandidateAction, "actions")
        )
        question_parts = tuple(
            _non_empty_string(value, "question part")
            for value in _sequence(self.question_parts, "question_parts")
        )
        parsed_queries = tuple(
            _freeze(self._query(value, "parsed query"))
            for value in _sequence(self.parsed_queries, "parsed_queries")
        )

        object.__setattr__(self, "intent", intent)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "intent_probabilities", MappingProxyType(probabilities))
        object.__setattr__(self, "intent_alternatives", alternatives)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "parsed_query",
            _freeze(self._query(self.parsed_query, "parsed_query")),
        )
        object.__setattr__(self, "parsed_queries", parsed_queries)
        object.__setattr__(self, "question_parts", question_parts)

    @staticmethod
    def _typed_items(
        values: object, expected_type: type[Any], field_name: str
    ) -> tuple[Any, ...]:
        items = _sequence(values, field_name)
        if not all(isinstance(value, expected_type) for value in items):
            raise TypeError(f"{field_name} must contain {expected_type.__name__} values")
        return cast(tuple[Any, ...], items)

    @staticmethod
    def _query(value: object, field_name: str) -> ParsedQuery | None:
        if value is None:
            return None
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError(f"{field_name} must be a three-part tuple or None")
        subject, predicate, _target = value
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(f"{field_name} subject must be a non-empty string")
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError(f"{field_name} predicate must be a non-empty string")
        return cast(ParsedQuery, value)

    @staticmethod
    def _span(value: object) -> tuple[int, int] | None:
        if value is None:
            return None
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise TypeError("entity span must be a two-part sequence or None")
        start, end = value
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError("entity span must contain ordered non-negative integers")
        return start, end


class LayaDecisionBackend(Protocol):
    """Minimal backend contract; the runtime receives text and nothing else."""

    def decide(self, text: str) -> LayaDecision: ...


class LayaPerceptionAdapter:
    """Translate injected Laya decisions into non-durable candidate frames."""

    def __init__(
        self,
        backend: LayaDecisionBackend,
        policy: LayaAdapterPolicy | None = None,
    ) -> None:
        if backend is None or not callable(getattr(backend, "decide", None)):
            raise TypeError("LayaPerceptionAdapter requires an injected backend")
        self.backend = backend
        self.policy = policy or LayaAdapterPolicy.default()

    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        del memory, context
        if not isinstance(text, str) or not text.strip():
            raise ValueError("perception input must be non-empty")

        decision = self.backend.decide(text)
        if not isinstance(decision, LayaDecision):
            raise TypeError("Laya backend must return LayaDecision")
        if decision.model_id.casefold() in {
            model_id.casefold() for model_id in self.policy.rejected_model_ids
        }:
            raise ValueError(
                f"Laya backend model_id {decision.model_id!r} is reserved for "
                "the deterministic fallback"
            )

        self._validate_query_targets(decision)

        canonical_intent = self.policy.intent_mapping.get(decision.intent)
        requirements = (
            self.policy.candidate_requirements.get(canonical_intent, ())
            if canonical_intent is not None
            else ()
        )
        allowed_payloads = (
            self.policy.allowed_payloads.get(canonical_intent, ())
            if canonical_intent is not None
            else ()
        )
        payload_present = any(
            self._has_payload(getattr(decision, field_name))
            for field_name in requirements
        )
        disallowed_payloads = tuple(
            field_name
            for field_name in self._payload_fields()
            if field_name not in allowed_payloads
            and self._has_payload(getattr(decision, field_name))
        )
        if (
            canonical_intent is None
            or canonical_intent == self.policy.fallback_intent
            or (requirements and not payload_present)
            or disallowed_payloads
        ):
            return self._unknown_frame(text, decision)

        return CandidateFrame.from_text(
            text,
            claims=list(decision.claims),
            entities=list(decision.entities),
            actions=list(decision.actions),
            intent=canonical_intent,
            uncertainty=decision.uncertainty,
            model_id=decision.model_id,
            model_version=decision.model_version,
            parsed_query=decision.parsed_query,
            parsed_queries=list(decision.parsed_queries),
            question_parts=list(decision.question_parts),
            intent_probabilities=dict(decision.intent_probabilities),
            intent_alternatives=decision.intent_alternatives,
        )

    def _unknown_frame(self, text: str, decision: LayaDecision) -> CandidateFrame:
        return CandidateFrame.from_text(
            text,
            claims=[],
            entities=[],
            actions=[],
            intent=self.policy.fallback_intent,
            uncertainty=decision.uncertainty,
            model_id=decision.model_id,
            model_version=decision.model_version,
            intent_probabilities=dict(decision.intent_probabilities),
            intent_alternatives=decision.intent_alternatives,
        )

    @staticmethod
    def _has_payload(value: object) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes, Sequence, Mapping)):
            return bool(value)
        return True

    @staticmethod
    def _payload_fields() -> tuple[str, ...]:
        return _PAYLOAD_FIELDS

    def _validate_query_targets(self, decision: LayaDecision) -> None:
        queries = (decision.parsed_query, *decision.parsed_queries)
        for query in queries:
            if query is None:
                continue
            _subject, predicate, target = query
            if isinstance(target, str) and target.strip():
                continue
            if any(
                predicate.startswith(prefix)
                for prefix in self.policy.structured_query_predicate_prefixes
            ):
                continue
            raise ValueError(
                "ordinary query targets must be non-empty strings or a wildcard"
            )
