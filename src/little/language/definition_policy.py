"""Versioned catalog for concept-definition question routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class DefinitionQuestionPattern:
    """One definition query and its captured concept subject."""

    name: str
    pattern: str
    predicate: str
    subject_group: int
    exclude_non_concept: bool


@dataclass(frozen=True)
class DefinitionQuestionPolicy:
    """Immutable concept-definition question catalog."""

    patterns: tuple[DefinitionQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> DefinitionQuestionPolicy:
        directory = Path(directory)
        path = directory / "parser_definition_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_definition_policy.v1":
            raise ValueError(f"{path} has an unsupported definition-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[DefinitionQuestionPattern] = []
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

            subject_group = raw.get("subject_group")
            if not isinstance(subject_group, int) or subject_group <= 0:
                raise TypeError(
                    f"{path} patterns[{index}] subject_group must be a positive integer"
                )
            exclude_non_concept = raw.get("exclude_non_concept", True)
            if not isinstance(exclude_non_concept, bool):
                raise TypeError(
                    f"{path} patterns[{index}] exclude_non_concept must be boolean"
                )
            patterns.append(
                DefinitionQuestionPattern(
                    name=text("name"),
                    pattern=text("pattern"),
                    predicate=text("predicate"),
                    subject_group=subject_group,
                    exclude_non_concept=exclude_non_concept,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> DefinitionQuestionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
