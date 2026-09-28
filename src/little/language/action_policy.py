"""Versioned catalog for parser-level procedural action patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _required_text(
    raw: dict[str, Any], field: str, path: Path, index: int
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"{path} patterns[{index}] field {field!r} must be a non-empty string"
        )
    return value.strip()


@dataclass(frozen=True)
class ActionPattern:
    """One declarative procedural action parsing strategy."""

    name: str
    strategy: str
    skill: str
    verb_source: str
    connector: str
    count_units: tuple[str, ...]
    object_argument: str
    count_argument: str


@dataclass(frozen=True)
class ActionPolicy:
    """Immutable procedural action pattern catalog."""

    patterns: tuple[ActionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> ActionPolicy:
        directory = Path(directory)
        path = directory / "parser_action_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_action_policy.v1":
            raise ValueError(f"{path} has an unsupported action-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[ActionPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")

            count_units = raw.get("count_units")
            if (
                not isinstance(count_units, list)
                or not count_units
                or not all(isinstance(value, str) and value.strip() for value in count_units)
            ):
                raise TypeError(
                    f"{path} patterns[{index}] count_units must be a non-empty list of strings"
                )

            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"verb_object_count"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported action strategy"
                )

            patterns.append(
                ActionPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    skill=_required_text(raw, "skill", path, index).upper(),
                    verb_source=_required_text(raw, "verb_source", path, index),
                    connector=_required_text(raw, "connector", path, index).lower(),
                    count_units=tuple(
                        value.strip().lower() for value in count_units if value.strip()
                    ),
                    object_argument=_required_text(
                        raw, "object_argument", path, index
                    ),
                    count_argument=_required_text(
                        raw, "count_argument", path, index
                    ),
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> ActionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
