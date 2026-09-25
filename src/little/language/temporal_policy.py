"""Versioned catalog for duration-based temporal question routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class TemporalQuestionPattern:
    """One duration query with capture groups for subject and elapsed time."""

    name: str
    pattern: str
    predicate: str
    subject_group: int
    value_group: int
    unit_group: int


@dataclass(frozen=True)
class TemporalQuestionPolicy:
    """Immutable temporal question catalog."""

    patterns: tuple[TemporalQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> TemporalQuestionPolicy:
        directory = Path(directory)
        path = directory / "parser_temporal_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_temporal_policy.v1":
            raise ValueError(f"{path} has an unsupported temporal-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[TemporalQuestionPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")

            def text(name: str) -> str:
                value = raw.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise TypeError(
                        f"{path} patterns[{index}] field {name!r} must be a string"
                    )
                return value.strip()

            def group(name: str) -> int:
                value = raw.get(name)
                if not isinstance(value, int) or value <= 0:
                    raise TypeError(
                        f"{path} patterns[{index}] field {name!r} must be a positive integer"
                    )
                return value

            patterns.append(
                TemporalQuestionPattern(
                    name=text("name"),
                    pattern=text("pattern"),
                    predicate=text("predicate"),
                    subject_group=group("subject_group"),
                    value_group=group("value_group"),
                    unit_group=group("unit_group"),
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> TemporalQuestionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
