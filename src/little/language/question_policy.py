"""Versioned catalog of parser-level conversational question patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SpecialQuestionPattern:
    """One regex-backed special question route."""

    name: str
    pattern: str
    subject: str
    predicate: str
    target: str | None = None


@dataclass(frozen=True)
class QuestionPatternPolicy:
    """Immutable conversational question pattern catalog."""

    patterns: tuple[SpecialQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> QuestionPatternPolicy:
        directory = Path(directory)
        path = directory / "parser_question_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_question_policy.v1":
            raise ValueError(f"{path} has an unsupported question-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[SpecialQuestionPattern] = []
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

            target = raw.get("target")
            if target is not None and (
                not isinstance(target, str) or not target.strip()
            ):
                raise TypeError(
                    f"{path} patterns[{index}] target must be a string or null"
                )
            patterns.append(
                SpecialQuestionPattern(
                    name=text("name"),
                    pattern=text("pattern"),
                    subject=text("subject"),
                    predicate=text("predicate"),
                    target=target.strip() if isinstance(target, str) else None,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> QuestionPatternPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
