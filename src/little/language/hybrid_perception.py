"""Versioned policy and explicit reconciliation for hybrid perception."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from string import Formatter
from typing import TYPE_CHECKING, Any

from little.core.contracts import (
    CandidateAction,
    CandidateClaim,
    CandidateEntity,
    CandidateFrame,
)
from little.language.laya_adapter import (
    LayaAdapterPolicy,
    LayaDecision,
    LayaDecisionBackend,
)
from little.language.perception import PerceptionAdapter

if TYPE_CHECKING:
    from little.language.dialogue import DialogueContext
    from little.memory.store import MemoryStore


_PAYLOAD_FIELDS = (
    "claims",
    "entities",
    "actions",
    "parsed_query",
    "parsed_queries",
    "question_parts",
)


@dataclass(frozen=True)
class HybridPerceptionPolicy:
    version: str
    unknown_intent: str
    require_route_match: bool
    uncertainty_strategy: str
    model_id_format: str
    model_version_format: str

    @classmethod
    def load(cls, directory: Path) -> HybridPerceptionPolicy:
        path = Path(directory) / "hybrid_perception_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Hybrid perception policy not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} contains invalid JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        data = payload.get("hybrid_perception_policy")
        if not isinstance(data, dict):
            raise ValueError(
                f"{path} field 'hybrid_perception_policy' must be an object"
            )

        fields = (
            "version",
            "unknown_intent",
            "uncertainty_strategy",
            "model_id_format",
            "model_version_format",
        )
        values: dict[str, Any] = {}
        for field in fields:
            if field not in data:
                raise ValueError(f"{path} missing field {field!r}")
            value = data[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path} field {field!r} must be a non-empty string")
            values[field] = value

        if "require_route_match" not in data:
            raise ValueError(f"{path} missing field 'require_route_match'")
        require_route_match = data["require_route_match"]
        if require_route_match is not True:
            raise ValueError(f"{path} field 'require_route_match' must be true")

        if values["uncertainty_strategy"] != "maximum":
            raise ValueError(
                f"{path} field 'uncertainty_strategy' must be 'maximum'"
            )

        placeholders = {
            "model_id_format": ("system1_model_id", "grounding_model_id"),
            "model_version_format": (
                "system1_model_version",
                "grounding_model_version",
            ),
        }
        allowed_fields = {
            "system1_model_id",
            "grounding_model_id",
            "system1_model_version",
            "grounding_model_version",
        }
        for field, required in placeholders.items():
            try:
                parsed = tuple(Formatter().parse(values[field]))
            except ValueError as exc:
                raise ValueError(
                    f"{path} field {field!r} has invalid format syntax: {exc}"
                ) from exc
            actual_fields = set()
            for _, field_name, format_spec, conversion in parsed:
                if field_name is None:
                    continue
                actual_fields.add(field_name)
                if field_name not in allowed_fields:
                    raise ValueError(
                        f"{path} field {field!r} contains unsupported field "
                        f"{field_name!r}"
                    )
                if conversion is not None:
                    raise ValueError(
                        f"{path} field {field!r} contains unsupported conversion "
                        f"{conversion!r}"
                    )
                if format_spec:
                    raise ValueError(
                        f"{path} field {field!r} contains unsupported format specifier "
                        f"{format_spec!r}"
                    )
            if not set(required).issubset(actual_fields):
                raise ValueError(
                    f"{path} field {field!r} must include placeholders {required!r}"
                )

        return cls(
            version=values["version"],
            unknown_intent=values["unknown_intent"],
            require_route_match=require_route_match,
            uncertainty_strategy=values["uncertainty_strategy"],
            model_id_format=values["model_id_format"],
            model_version_format=values["model_version_format"],
        )

    @classmethod
    def default(cls) -> HybridPerceptionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)


class HybridPerceptionAdapter:
    """Reconcile an injected System 1 route with an injected grounded frame."""

    def __init__(
        self,
        system1: LayaDecisionBackend,
        grounding: PerceptionAdapter,
        *,
        policy: HybridPerceptionPolicy | None = None,
        laya_policy: LayaAdapterPolicy | None = None,
    ) -> None:
        if system1 is None or not callable(getattr(system1, "decide", None)):
            raise TypeError("HybridPerceptionAdapter requires an injected System 1 backend")
        if grounding is None or not callable(getattr(grounding, "perceive", None)):
            raise TypeError("HybridPerceptionAdapter requires an injected grounding provider")
        self.system1 = system1
        self.grounding = grounding
        self.policy = policy if policy is not None else HybridPerceptionPolicy.default()
        self.laya_policy = (
            laya_policy if laya_policy is not None else LayaAdapterPolicy.default()
        )

    def perceive(
        self,
        text: str,
        *,
        memory: MemoryStore,
        context: DialogueContext | None = None,
    ) -> CandidateFrame:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("perception input must be non-empty")

        decision = self.system1.decide(text)
        if not isinstance(decision, LayaDecision):
            return self._unknown_frame(text)
        if decision.model_id.casefold() in {
            model_id.casefold() for model_id in self.laya_policy.rejected_model_ids
        }:
            return self._unknown_frame(text)

        route = self.laya_policy.intent_mapping.get(decision.intent)
        grounding = self.grounding.perceive(text, memory=memory, context=context)
        if not isinstance(grounding, CandidateFrame):
            return self._unknown_frame(text, decision=decision)

        if (
            route is None
            or route == self.policy.unknown_intent
            or route == self.laya_policy.fallback_intent
            or route != grounding.intent
        ):
            return self._unknown_frame(text, decision=decision, grounding=grounding)

        requirements = self.laya_policy.candidate_requirements.get(route)
        allowed = self.laya_policy.allowed_payloads.get(route)
        if requirements is None or allowed is None:
            return self._unknown_frame(text, decision=decision, grounding=grounding)

        candidate_fields = (
            (grounding.claims, self._valid_claim),
            (grounding.entities, self._valid_entity),
            (grounding.actions, self._valid_action),
        )
        if any(
            not isinstance(items, (list, tuple))
            or not all(valid(item) for item in items)
            for items, valid in candidate_fields
        ):
            return self._unknown_frame(text, decision=decision, grounding=grounding)

        query_fields = ("parsed_query", "parsed_queries", "question_parts")
        if any(
            field in allowed
            and self._is_supplied_payload(field, getattr(grounding, field))
            and not self._has_payload(field, getattr(grounding, field))
            for field in query_fields
        ):
            return self._unknown_frame(text, decision=decision, grounding=grounding)

        if (
            requirements
            and not any(
                self._has_payload(field, getattr(grounding, field))
                for field in requirements
            )
        ) or any(
            self._is_supplied_payload(field, getattr(grounding, field))
            for field in _PAYLOAD_FIELDS
            if field not in allowed
        ):
            return self._unknown_frame(text, decision=decision, grounding=grounding)

        payload = {
            field: self._copy_payload(field, getattr(grounding, field))
            for field in allowed
        }
        return CandidateFrame.from_text(
            text,
            claims=payload.pop("claims", []),
            intent=route,
            uncertainty=self._uncertainty(decision, grounding),
            model_id=self._model_id(decision, grounding),
            model_version=self._model_version(decision, grounding),
            intent_probabilities=dict(decision.intent_probabilities),
            intent_alternatives=decision.intent_alternatives,
            **payload,
        )

    def _unknown_frame(
        self,
        text: str,
        *,
        decision: LayaDecision | None = None,
        grounding: CandidateFrame | None = None,
    ) -> CandidateFrame:
        return CandidateFrame.from_text(
            text,
            claims=[],
            intent=self.policy.unknown_intent,
            uncertainty=(
                self._uncertainty(decision, grounding)
                if decision is not None and grounding is not None
                else decision.uncertainty if decision is not None else 1.0
            ),
            model_id=self._model_id(decision, grounding),
            model_version=self._model_version(decision, grounding),
            intent_probabilities=(
                dict(decision.intent_probabilities) if decision is not None else {}
            ),
            intent_alternatives=(
                decision.intent_alternatives if decision is not None else ()
            ),
        )

    def _uncertainty(self, decision: LayaDecision, grounding: CandidateFrame) -> float:
        if self.policy.uncertainty_strategy == "maximum":
            return max(decision.uncertainty, grounding.uncertainty)
        raise ValueError("unsupported hybrid uncertainty strategy")

    def _model_id(
        self,
        decision: LayaDecision | None,
        grounding: CandidateFrame | None,
    ) -> str:
        return self.policy.model_id_format.format(
            system1_model_id=(
                decision.model_id if decision is not None else self.policy.unknown_intent
            ),
            grounding_model_id=(
                grounding.model_id if grounding is not None else self.policy.unknown_intent
            ),
        )

    def _model_version(
        self,
        decision: LayaDecision | None,
        grounding: CandidateFrame | None,
    ) -> str:
        return self.policy.model_version_format.format(
            system1_model_version=(
                decision.model_version if decision is not None else self.policy.version
            ),
            grounding_model_version=(
                grounding.model_version if grounding is not None else self.policy.version
            ),
        )

    def _has_payload(self, field: str, value: object) -> bool:
        if field == "parsed_query":
            return self._is_usable_query(value)
        if field == "parsed_queries":
            if not isinstance(value, (list, tuple)):
                return False
            queries = [query for query in value if query is not None]
            return bool(queries) and all(
                self._is_usable_query(query) for query in queries
            )
        if field == "question_parts":
            return isinstance(value, (list, tuple)) and bool(value) and all(
                isinstance(part, str) and bool(part.strip()) for part in value
            )
        return value is not None and bool(value)

    @staticmethod
    def _is_supplied_payload(field: str, value: object) -> bool:
        if field == "parsed_query":
            return value is not None
        if field == "parsed_queries" and isinstance(value, (list, tuple)):
            return any(query is not None for query in value)
        return not (isinstance(value, list) and not value)

    @staticmethod
    def _valid_probability(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )

    @staticmethod
    def _nonblank_string(value: object) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @classmethod
    def _valid_claim(cls, item: object) -> bool:
        return (
            isinstance(item, CandidateClaim)
            and all(
                cls._nonblank_string(value)
                for value in (item.subject, item.predicate, item.object)
            )
            and cls._valid_probability(item.probability)
            and isinstance(item.attributes, Mapping)
            and all(isinstance(key, str) for key in item.attributes)
            and isinstance(item.positive, bool)
            and isinstance(item.is_property, bool)
        )

    @classmethod
    def _valid_entity(cls, item: object) -> bool:
        if not isinstance(item, CandidateEntity):
            return False
        span = item.span
        return (
            cls._nonblank_string(item.name)
            and (item.type_hint is None or isinstance(item.type_hint, str))
            and (
                span is None
                or (
                    not isinstance(span, (str, bytes))
                    and isinstance(span, Sequence)
                    and len(span) == 2
                    and all(
                        isinstance(point, int) and not isinstance(point, bool)
                        for point in span
                    )
                    and span[0] >= 0
                    and span[1] >= span[0]
                )
            )
            and cls._valid_probability(item.probability)
        )

    @classmethod
    def _valid_action(cls, item: object) -> bool:
        return (
            isinstance(item, CandidateAction)
            and cls._nonblank_string(item.name)
            and isinstance(item.arguments, Mapping)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in item.arguments.items()
            )
            and cls._valid_probability(item.probability)
        )

    def _is_usable_query(self, value: object) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[0], str)
            and bool(value[0].strip())
            and isinstance(value[1], str)
            and bool(value[1].strip())
            and (
                (isinstance(value[2], str) and bool(value[2].strip()))
                or any(
                    value[1].startswith(prefix)
                    for prefix in self.laya_policy.structured_query_predicate_prefixes
                )
            )
        )

    @classmethod
    def _copy_payload(cls, field: str, value: Any) -> Any:
        if field == "parsed_queries" and not cls._is_supplied_payload(field, value):
            return []
        return list(value) if isinstance(value, list) else value
