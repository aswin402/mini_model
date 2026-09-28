"""Versioned catalog for compound-question boundary patterns."""

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
class CompoundQuestionPattern:
    """One declarative compound-question delimiter."""

    name: str
    strategy: str
    pattern: str
    case_insensitive: bool
    terminal_suffix: str | None = None


@dataclass(frozen=True)
class CompoundQuestionPolicy:
    """Immutable compound-question boundary catalog."""

    patterns: tuple[CompoundQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> CompoundQuestionPolicy:
        directory = Path(directory)
        path = directory / "parser_compound_question_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_compound_question_policy.v1":
            raise ValueError(
                f"{path} has an unsupported compound-question-policy format"
            )

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[CompoundQuestionPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"delimiter"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported compound-question strategy"
                )
            case_insensitive = raw.get("case_insensitive", False)
            if not isinstance(case_insensitive, bool):
                raise TypeError(
                    f"{path} patterns[{index}] case_insensitive must be boolean"
                )
            terminal_suffix = raw.get("terminal_suffix")
            if terminal_suffix is not None and (
                not isinstance(terminal_suffix, str) or not terminal_suffix
            ):
                raise TypeError(
                    f"{path} patterns[{index}] terminal_suffix must be a non-empty string or null"
                )
            patterns.append(
                CompoundQuestionPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    pattern=_required_text(raw, "pattern", path, index),
                    case_insensitive=case_insensitive,
                    terminal_suffix=terminal_suffix,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> CompoundQuestionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
