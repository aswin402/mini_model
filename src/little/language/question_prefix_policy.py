"""Versioned catalog for question-prefix delegation routes."""

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
class QuestionPrefixPattern:
    """One declarative prefix vocabulary and delegation strategy."""

    name: str
    strategy: str
    prefix_source: str
    separator_pattern: str
    body_group: int


@dataclass(frozen=True)
class QuestionPrefixPolicy:
    """Immutable question-prefix catalog."""

    patterns: tuple[QuestionPrefixPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> QuestionPrefixPolicy:
        directory = Path(directory)
        path = directory / "parser_question_prefix_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_question_prefix_policy.v1":
            raise ValueError(
                f"{path} has an unsupported question-prefix-policy format"
            )

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[QuestionPrefixPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"prefix_delegate"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported question-prefix strategy"
                )
            body_group = raw.get("body_group")
            if not isinstance(body_group, int) or body_group <= 0:
                raise TypeError(
                    f"{path} patterns[{index}] body_group must be a positive integer"
                )
            patterns.append(
                QuestionPrefixPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    prefix_source=_required_text(
                        raw, "prefix_source", path, index
                    ),
                    separator_pattern=_required_text(
                        raw, "separator_pattern", path, index
                    ),
                    body_group=body_group,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> QuestionPrefixPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
