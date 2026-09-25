"""Data-driven schemas for relations, actions, and continuous state variables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class RelationSchema:
    name: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    axis: str | None = None
    inverse: str | None = None
    symmetric: bool = False
    transitive: bool = False
    acyclic: bool = False
    disjoint: bool = False
    reverse_refutation: bool = False
    evidence_policy: str = "observed_or_derived"
    attribute_key: str | None = None
    attribute_update_policy: str = "append"
    attribute_value_format: str = "preserve"
    primary: bool = False


@dataclass(frozen=True)
class ActionSchema:
    name: str
    argument_types: dict[str, str]
    defaults: dict[str, Any] = field(default_factory=dict)
    preconditions: tuple[dict[str, Any], ...] = ()
    effects: tuple[dict[str, Any], ...] = ()
    invariants: tuple[str, ...] = ()
    executor: str | None = None


@dataclass(frozen=True)
class StateVariableSchema:
    name: str
    value_type: str
    unit: str | None = None
    valid_range: tuple[float, float] | None = None
    transition_model: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""


class SchemaRegistry:
    def __init__(
        self,
        relations: dict[str, RelationSchema],
        actions: dict[str, ActionSchema],
        state_variables: dict[str, StateVariableSchema],
    ) -> None:
        self._relations = relations
        self._actions = actions
        self._state_variables = state_variables

    @classmethod
    def load(cls, directory: Path) -> SchemaRegistry:
        directory = Path(directory)
        relation_data = cls._load_json(directory / "relation_types.json", "relation_types")
        action_data = cls._load_json(directory / "action_schemas.json", "action_schemas")
        state_data = cls._load_json(directory / "state_variables.json", "state_variables")

        relations = {
            item["name"]: RelationSchema(
                name=item["name"],
                domain=tuple(item["domain"]),
                range=tuple(item["range"]),
                axis=item.get("axis"),
                inverse=item.get("inverse"),
                symmetric=bool(item.get("symmetric", False)),
                transitive=bool(item.get("transitive", False)),
                acyclic=bool(item.get("acyclic", False)),
                disjoint=bool(item.get("disjoint", False)),
                reverse_refutation=bool(item.get("reverse_refutation", False)),
                evidence_policy=item.get("evidence_policy", "observed_or_derived"),
                attribute_key=item.get("attribute_key"),
                attribute_update_policy=item.get("attribute_update_policy", "append"),
                attribute_value_format=item.get("attribute_value_format", "preserve"),
                primary=bool(item.get("primary", False)),
            )
            for item in relation_data
        }
        if len(relations) != len(relation_data):
            raise ValueError("duplicate relation schema name")
        for schema in relations.values():
            if schema.inverse and schema.inverse not in relations:
                raise ValueError(f"unknown inverse relation: {schema.inverse}")

        actions = {
            item["name"]: ActionSchema(
                name=item["name"],
                argument_types=dict(item.get("argument_types", {})),
                defaults=dict(item.get("defaults", {})),
                preconditions=tuple(item.get("preconditions", [])),
                effects=tuple(item.get("effects", [])),
                invariants=tuple(item.get("invariants", [])),
                executor=item.get("executor"),
            )
            for item in action_data
        }
        if len(actions) != len(action_data):
            raise ValueError("duplicate action schema name")

        state_variables = {
            item["name"]: StateVariableSchema(
                name=item["name"],
                value_type=item["value_type"],
                unit=item.get("unit"),
                valid_range=(
                    tuple(item["valid_range"]) if item.get("valid_range") else None
                ),
                transition_model=item.get("transition_model"),
                parameters=dict(item.get("parameters", {})),
            )
            for item in state_data
        }
        if len(state_variables) != len(state_data):
            raise ValueError("duplicate state variable schema name")

        return cls(relations, actions, state_variables)

    @classmethod
    def default(cls) -> SchemaRegistry:
        return cls.load(RuntimePaths.default().schema_directory)

    @staticmethod
    def _load_json(path: Path, key: str) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get(key)
        if not isinstance(records, list):
            raise TypeError(f"{path} must contain a list under {key!r}")
        return records

    def relation(self, name: str) -> RelationSchema | None:
        return self._relations.get(name.strip().lower())

    def action(self, name: str) -> ActionSchema | None:
        return self._actions.get(name.strip().lower())

    def state_variable(self, name: str) -> StateVariableSchema | None:
        return self._state_variables.get(name.strip().lower())

    def predicates(
        self,
        *,
        transitive: bool = False,
        acyclic: bool = False,
        disjoint: bool = False,
        reverse_refutation: bool = False,
        axis: str | None = None,
    ) -> set[str]:
        """Return predicates matching declarative reasoning capabilities."""
        return {
            name
            for name, schema in self._relations.items()
            if (not transitive or schema.transitive)
            and (not acyclic or schema.acyclic)
            and (not disjoint or schema.disjoint)
            and (not reverse_refutation or schema.reverse_refutation)
            and (axis is None or schema.axis == axis.strip().lower())
        }

    def primary_predicate(self, axis: str) -> str | None:
        """Return the data-declared representative predicate for an axis."""
        clean_axis = axis.strip().lower()
        primary = sorted(
            schema.name
            for schema in self._relations.values()
            if schema.axis == clean_axis and schema.primary
        )
        if primary:
            return primary[0]
        candidates = sorted(
            schema.name
            for schema in self._relations.values()
            if schema.axis == clean_axis
        )
        return candidates[0] if candidates else None

    def register_learned_relation(self, name: str) -> RelationSchema:
        """Register a predicate backed by a persisted learned construction."""
        clean_name = name.strip().lower()
        existing = self._relations.get(clean_name)
        if existing:
            return existing
        schema = RelationSchema(
            name=clean_name,
            domain=("concept", "entity"),
            range=("concept", "entity", "value", "event"),
            evidence_policy="learned_construction",
        )
        self._relations[clean_name] = schema
        return schema

    def validate_relation(
        self, predicate: str, subject_type: str, object_type: str
    ) -> ValidationResult:
        schema = self.relation(predicate)
        if schema is None:
            return ValidationResult(False, f"unregistered predicate: {predicate}")
        if subject_type not in schema.domain:
            return ValidationResult(
                False,
                f"domain type {subject_type!r} is invalid for {schema.name}",
            )
        if object_type not in schema.range:
            return ValidationResult(
                False,
                f"range type {object_type!r} is invalid for {schema.name}",
            )
        return ValidationResult(True, "relation schema accepted")

    def validate_action(
        self, name: str, arguments: dict[str, str]
    ) -> ValidationResult:
        schema = self.action(name)
        if schema is None:
            return ValidationResult(False, f"unregistered action: {name}")
        for argument, expected_type in schema.argument_types.items():
            actual_type = arguments.get(argument)
            if actual_type is None:
                return ValidationResult(
                    False, f"missing action argument: {argument}"
                )
            if actual_type != expected_type:
                return ValidationResult(
                    False,
                    f"argument {argument!r} has type {actual_type!r}; expected {expected_type!r}",
                )
        return ValidationResult(True, "action schema accepted")
