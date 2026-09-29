"""Versioned catalog for question-clause boundary patterns."""

from __future__ import annotations

import json
import re
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
class QuestionBoundaryPattern:
    """One declarative question-clause boundary."""

    name: str
    strategy: str
    pattern: str
    case_insensitive: bool


@dataclass(frozen=True)
class QuestionBoundaryPolicy:
    """Immutable question-clause boundary catalog."""

    patterns: tuple[QuestionBoundaryPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> QuestionBoundaryPolicy:
        directory = Path(directory)
        path = directory / "parser_question_boundary_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_question_boundary_policy.v1":
            raise ValueError(
                f"{path} has an unsupported question-boundary-policy format"
            )

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[QuestionBoundaryPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"question_mark", "coordinated"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported question-boundary strategy"
                )
            case_insensitive = raw.get("case_insensitive", False)
            if not isinstance(case_insensitive, bool):
                raise TypeError(
                    f"{path} patterns[{index}] case_insensitive must be boolean"
                )
            patterns.append(
                QuestionBoundaryPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    pattern=_required_text(raw, "pattern", path, index),
                    case_insensitive=case_insensitive,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> QuestionBoundaryPolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def split(self, text: str) -> list[str]:
        clean = text.strip()
        for pattern in self.patterns:
            flags = re.IGNORECASE if pattern.case_insensitive else 0
            if re.search(pattern.pattern, clean, flags=flags) is None:
                continue
            parts = [
                part.strip()
                for part in re.split(pattern.pattern, clean, flags=flags)
                if part.strip()
            ]
            if len(parts) > 1:
                return parts
        return [clean] if clean else []
