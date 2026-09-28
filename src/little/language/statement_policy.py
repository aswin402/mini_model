"""Versioned catalog for statement-level fallback strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class StatementPattern:
    """One declarative statement fallback strategy."""

    name: str
    strategy: str
    predicate_role: str | None = None
    verb_source: str | None = None
    predicate_source: str | None = None


@dataclass(frozen=True)
class StatementPolicy:
    """Immutable statement fallback catalog."""

    patterns: tuple[StatementPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> StatementPolicy:
        directory = Path(directory)
        path = directory / "parser_statement_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_statement_policy.v1":
            raise ValueError(f"{path} has an unsupported statement-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[StatementPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")

            values: dict[str, str] = {}
            for field in ("name", "strategy"):
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise TypeError(
                        f"{path} patterns[{index}] field {field!r} must be a non-empty string"
                    )
                values[field] = value.strip()

            predicate_role = raw.get("predicate_role")
            verb_source = raw.get("verb_source")
            predicate_source = raw.get("predicate_source")
            optional_values = {
                "predicate_role": predicate_role,
                "verb_source": verb_source,
                "predicate_source": predicate_source,
            }
            for field, value in optional_values.items():
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise TypeError(
                        f"{path} patterns[{index}] field {field!r} must be a non-empty string"
                    )
                if isinstance(value, str):
                    values[field] = value.strip()

            if values["strategy"] not in {"action_verb_tail", "novel_transitive"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported statement strategy"
                )
            if values["strategy"] == "action_verb_tail" and (
                values.get("predicate_role") is None
                or values.get("verb_source") != "action_verbs"
                or values.get("predicate_source") is not None
            ):
                raise ValueError(
                    f"{path} patterns[{index}] action_verb_tail requires predicate_role and action_verbs"
                )
            if values["strategy"] == "novel_transitive" and (
                values.get("predicate_source") != "middle_token"
                or values.get("predicate_role") is not None
                or values.get("verb_source") is not None
            ):
                raise ValueError(
                    f"{path} patterns[{index}] novel_transitive requires middle_token predicate_source"
                )

            patterns.append(StatementPattern(**values))

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> StatementPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
